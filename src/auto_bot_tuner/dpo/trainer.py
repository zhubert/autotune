"""
Trainer for Direct Preference Optimization (DPO).

DPO trains a language model using preference data without requiring
a separate reward model. It maintains a frozen reference model and
optimizes the policy model to prefer chosen responses over rejected ones.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Callable
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
import json
import copy

from .loss import compute_dpo_loss_with_metrics, get_batch_logps
from .dataset import PreferenceDataset


@dataclass
class DPOConfig:
    """Configuration for DPO training."""

    # DPO-specific parameters
    beta: float = 0.1  # Temperature for DPO loss (controls KL penalty)
    label_smoothing: float = 0.0  # Label smoothing (0 = no smoothing)

    # Training hyperparameters
    learning_rate: float = 5e-7  # Lower than SFT since we're fine-tuning aligned model
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    num_epochs: int = 1  # Usually 1 epoch is enough for DPO
    max_steps: Optional[int] = None
    warmup_steps: int = 50
    max_grad_norm: float = 1.0

    # Optimization
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8

    # Logging and checkpointing
    logging_steps: int = 10
    eval_steps: Optional[int] = 500
    save_steps: int = 500
    save_total_limit: int = 3

    # Paths
    output_dir: str = "checkpoints/dpo"
    resume_from_checkpoint: Optional[str] = None

    # Mixed precision
    fp16: bool = False
    bf16: bool = False

    # Misc
    seed: int = 42
    dataloader_num_workers: int = 0


class DPOTrainer:
    """
    Trainer for Direct Preference Optimization.

    DPO training requires:
    1. A policy model (the model being trained)
    2. A reference model (frozen copy of initial policy)
    3. Preference dataset (prompt, chosen, rejected)

    The trainer optimizes the policy to increase the probability of chosen
    responses relative to rejected ones, while staying close to the reference
    model (controlled by beta parameter).
    """

    def __init__(
        self,
        policy_model: torch.nn.Module,
        reference_model: torch.nn.Module,
        tokenizer,
        train_dataset: PreferenceDataset,
        eval_dataset: Optional[PreferenceDataset] = None,
        config: Optional[DPOConfig] = None,
        callbacks: Optional[list[Callable]] = None
    ):
        """
        Args:
            policy_model: The model being trained
            reference_model: Frozen reference model (typically a copy of initial policy)
            tokenizer: Tokenizer for the models
            train_dataset: Training dataset with preferences
            eval_dataset: Optional evaluation dataset
            config: Training configuration
            callbacks: Optional callbacks called each step
        """
        self.policy_model = policy_model
        self.reference_model = reference_model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.config = config or DPOConfig()
        self.callbacks = callbacks or []

        # Freeze reference model
        self.reference_model.eval()
        for param in self.reference_model.parameters():
            param.requires_grad = False

        # Setup device
        self.device = next(policy_model.parameters()).device

        # Move reference model to same device
        self.reference_model = self.reference_model.to(self.device)

        # Setup optimizer
        self.optimizer = self._create_optimizer()

        # Setup learning rate scheduler
        self.scheduler = self._create_scheduler()

        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_eval_accuracy = 0.0

        # Create output directory
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)

        # Save config
        self._save_config()

    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create AdamW optimizer with weight decay."""
        decay_params = []
        no_decay_params = []

        for name, param in self.policy_model.named_parameters():
            if not param.requires_grad:
                continue

            if "bias" in name or "layer_norm" in name or "layernorm" in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        optimizer_grouped_parameters = [
            {"params": decay_params, "weight_decay": self.config.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]

        return AdamW(
            optimizer_grouped_parameters,
            lr=self.config.learning_rate,
            betas=(self.config.adam_beta1, self.config.adam_beta2),
            eps=self.config.adam_epsilon,
        )

    def _create_scheduler(self) -> LambdaLR:
        """Create learning rate scheduler with warmup and decay."""
        def lr_lambda(current_step: int) -> float:
            if current_step < self.config.warmup_steps:
                return float(current_step) / float(max(1, self.config.warmup_steps))
            else:
                total_steps = self.config.max_steps or (
                    len(self.train_dataset) // (self.config.batch_size * self.config.gradient_accumulation_steps)
                    * self.config.num_epochs
                )
                progress = float(current_step - self.config.warmup_steps) / float(
                    max(1, total_steps - self.config.warmup_steps)
                )
                return max(0.0, 1.0 - progress)

        return LambdaLR(self.optimizer, lr_lambda)

    def train(self) -> dict:
        """
        Run the DPO training loop.

        Returns:
            Dictionary with training statistics
        """
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.dataloader_num_workers,
            pin_memory=True if self.device.type == "cuda" else False
        )

        steps_per_epoch = len(train_loader) // self.config.gradient_accumulation_steps
        total_steps = self.config.max_steps or (steps_per_epoch * self.config.num_epochs)

        self.policy_model.train()
        total_loss = 0.0
        total_accuracy = 0.0
        total_reward_margin = 0.0

        progress_bar = tqdm(total=total_steps, desc="DPO Training")

        for epoch in range(self.config.num_epochs):
            self.epoch = epoch

            for batch_idx, batch in enumerate(train_loader):
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}

                # Forward pass through policy model for chosen responses
                policy_chosen_outputs = self.policy_model(
                    input_ids=batch["chosen_input_ids"],
                    attention_mask=batch["chosen_attention_mask"]
                )

                # Forward pass through policy model for rejected responses
                policy_rejected_outputs = self.policy_model(
                    input_ids=batch["rejected_input_ids"],
                    attention_mask=batch["rejected_attention_mask"]
                )

                # Forward pass through reference model (no gradients)
                with torch.no_grad():
                    reference_chosen_outputs = self.reference_model(
                        input_ids=batch["chosen_input_ids"],
                        attention_mask=batch["chosen_attention_mask"]
                    )
                    reference_rejected_outputs = self.reference_model(
                        input_ids=batch["rejected_input_ids"],
                        attention_mask=batch["rejected_attention_mask"]
                    )

                # Compute log probabilities
                policy_chosen_logps = get_batch_logps(
                    policy_chosen_outputs.logits,
                    batch["chosen_input_ids"],
                    batch["chosen_attention_mask"]
                )
                policy_rejected_logps = get_batch_logps(
                    policy_rejected_outputs.logits,
                    batch["rejected_input_ids"],
                    batch["rejected_attention_mask"]
                )
                reference_chosen_logps = get_batch_logps(
                    reference_chosen_outputs.logits,
                    batch["chosen_input_ids"],
                    batch["chosen_attention_mask"]
                )
                reference_rejected_logps = get_batch_logps(
                    reference_rejected_outputs.logits,
                    batch["rejected_input_ids"],
                    batch["rejected_attention_mask"]
                )

                # Compute DPO loss with metrics
                metrics = compute_dpo_loss_with_metrics(
                    policy_chosen_logps,
                    policy_rejected_logps,
                    reference_chosen_logps,
                    reference_rejected_logps,
                    beta=self.config.beta,
                    label_smoothing=self.config.label_smoothing
                )

                loss = metrics["loss"]

                # Scale loss for gradient accumulation
                loss = loss / self.config.gradient_accumulation_steps

                # Backward pass
                loss.backward()

                # Accumulate metrics
                total_loss += loss.item() * self.config.gradient_accumulation_steps
                total_accuracy += metrics["accuracy"].item()
                total_reward_margin += metrics["reward_margin"].item()

                # Update weights every gradient_accumulation_steps
                if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                    # Clip gradients
                    torch.nn.utils.clip_grad_norm_(
                        self.policy_model.parameters(),
                        self.config.max_grad_norm
                    )

                    # Update weights
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()

                    self.global_step += 1
                    progress_bar.update(1)

                    # Logging
                    if self.global_step % self.config.logging_steps == 0:
                        avg_loss = total_loss / self.config.logging_steps
                        avg_accuracy = total_accuracy / self.config.logging_steps
                        avg_reward_margin = total_reward_margin / self.config.logging_steps
                        lr = self.scheduler.get_last_lr()[0]

                        progress_bar.set_postfix({
                            "loss": f"{avg_loss:.4f}",
                            "acc": f"{avg_accuracy:.3f}",
                            "margin": f"{avg_reward_margin:.3f}",
                            "lr": f"{lr:.2e}"
                        })

                        total_loss = 0.0
                        total_accuracy = 0.0
                        total_reward_margin = 0.0

                    # Evaluation
                    if self.config.eval_steps and self.global_step % self.config.eval_steps == 0:
                        if self.eval_dataset is not None:
                            eval_metrics = self.evaluate()
                            self.policy_model.train()

                            if eval_metrics["accuracy"] > self.best_eval_accuracy:
                                self.best_eval_accuracy = eval_metrics["accuracy"]
                                self.save_checkpoint(is_best=True)

                    # Checkpointing
                    if self.global_step % self.config.save_steps == 0:
                        self.save_checkpoint()

                    # Run callbacks
                    for callback in self.callbacks:
                        callback(self, metrics)

                    # Check if we've reached max steps
                    if self.config.max_steps and self.global_step >= self.config.max_steps:
                        progress_bar.close()
                        return self._get_training_summary()

        progress_bar.close()

        # Save final checkpoint
        self.save_checkpoint(is_final=True)

        return self._get_training_summary()

    def evaluate(self) -> dict:
        """
        Evaluate the policy model on the eval dataset.

        Returns:
            Dictionary with evaluation metrics
        """
        if self.eval_dataset is None:
            return {}

        self.policy_model.eval()

        eval_loader = DataLoader(
            self.eval_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.dataloader_num_workers
        )

        total_loss = 0.0
        total_accuracy = 0.0
        total_reward_margin = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in tqdm(eval_loader, desc="Evaluating"):
                batch = {k: v.to(self.device) for k, v in batch.items()}

                # Policy forward passes
                policy_chosen_outputs = self.policy_model(
                    input_ids=batch["chosen_input_ids"],
                    attention_mask=batch["chosen_attention_mask"]
                )
                policy_rejected_outputs = self.policy_model(
                    input_ids=batch["rejected_input_ids"],
                    attention_mask=batch["rejected_attention_mask"]
                )

                # Reference forward passes
                reference_chosen_outputs = self.reference_model(
                    input_ids=batch["chosen_input_ids"],
                    attention_mask=batch["chosen_attention_mask"]
                )
                reference_rejected_outputs = self.reference_model(
                    input_ids=batch["rejected_input_ids"],
                    attention_mask=batch["rejected_attention_mask"]
                )

                # Compute log probabilities
                policy_chosen_logps = get_batch_logps(
                    policy_chosen_outputs.logits,
                    batch["chosen_input_ids"],
                    batch["chosen_attention_mask"]
                )
                policy_rejected_logps = get_batch_logps(
                    policy_rejected_outputs.logits,
                    batch["rejected_input_ids"],
                    batch["rejected_attention_mask"]
                )
                reference_chosen_logps = get_batch_logps(
                    reference_chosen_outputs.logits,
                    batch["chosen_input_ids"],
                    batch["chosen_attention_mask"]
                )
                reference_rejected_logps = get_batch_logps(
                    reference_rejected_outputs.logits,
                    batch["rejected_input_ids"],
                    batch["rejected_attention_mask"]
                )

                # Compute metrics
                metrics = compute_dpo_loss_with_metrics(
                    policy_chosen_logps,
                    policy_rejected_logps,
                    reference_chosen_logps,
                    reference_rejected_logps,
                    beta=self.config.beta,
                    label_smoothing=self.config.label_smoothing
                )

                total_loss += metrics["loss"].item()
                total_accuracy += metrics["accuracy"].item()
                total_reward_margin += metrics["reward_margin"].item()
                num_batches += 1

        avg_loss = total_loss / num_batches
        avg_accuracy = total_accuracy / num_batches
        avg_reward_margin = total_reward_margin / num_batches

        eval_metrics = {
            "loss": avg_loss,
            "accuracy": avg_accuracy,
            "reward_margin": avg_reward_margin,
        }

        print(f"\nEvaluation - Loss: {avg_loss:.4f}, Accuracy: {avg_accuracy:.4f}, Margin: {avg_reward_margin:.4f}")

        return eval_metrics

    def save_checkpoint(self, is_best: bool = False, is_final: bool = False) -> None:
        """Save model checkpoint."""
        if is_final:
            checkpoint_dir = Path(self.config.output_dir) / "final"
        elif is_best:
            checkpoint_dir = Path(self.config.output_dir) / "best"
        else:
            checkpoint_dir = Path(self.config.output_dir) / f"checkpoint-{self.global_step}"

        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Save policy model and tokenizer
        self.policy_model.save_pretrained(checkpoint_dir)
        self.tokenizer.save_pretrained(checkpoint_dir)

        # Save training state
        state = {
            "global_step": self.global_step,
            "epoch": self.epoch,
            "best_eval_accuracy": self.best_eval_accuracy,
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
        }

        torch.save(state, checkpoint_dir / "training_state.pt")

        print(f"\nCheckpoint saved to {checkpoint_dir}")

    def _save_config(self) -> None:
        """Save training configuration."""
        config_path = Path(self.config.output_dir) / "training_config.json"
        with open(config_path, 'w') as f:
            json.dump(vars(self.config), f, indent=2)

    def _get_training_summary(self) -> dict:
        """Get summary of training run."""
        return {
            "total_steps": self.global_step,
            "epochs_completed": self.epoch + 1,
            "best_eval_accuracy": self.best_eval_accuracy if self.best_eval_accuracy > 0 else None,
        }


def create_reference_model(policy_model: torch.nn.Module, device: torch.device) -> torch.nn.Module:
    """
    Create a frozen reference model from the policy model.

    The reference model is a copy of the initial policy model that remains
    frozen during training. It's used to compute KL divergence penalties.

    Args:
        policy_model: The policy model to copy
        device: Device to place the reference model on

    Returns:
        Frozen copy of the policy model
    """
    # Create a deep copy of the model
    reference_model = copy.deepcopy(policy_model)

    # Move to device
    reference_model = reference_model.to(device)

    # Freeze all parameters
    reference_model.eval()
    for param in reference_model.parameters():
        param.requires_grad = False

    return reference_model
