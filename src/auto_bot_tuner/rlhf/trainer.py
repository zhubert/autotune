"""
Trainer for Reward Models.

Trains a reward model to predict human preferences from comparison data.
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

from .loss import compute_ranking_loss_with_metrics, compute_reward_model_loss
from .dataset import RewardModelDataset
from .reward_model import RewardModel


@dataclass
class RewardModelConfig:
    """Configuration for reward model training."""

    # Training hyperparameters
    learning_rate: float = 1e-5  # Lower LR for stability
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    num_epochs: int = 1  # Usually 1 epoch is enough
    max_steps: Optional[int] = None
    warmup_steps: int = 100
    max_grad_norm: float = 1.0

    # Loss parameters
    margin: float = 0.0  # Margin for ranking loss

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
    output_dir: str = "checkpoints/reward_model"
    resume_from_checkpoint: Optional[str] = None

    # Mixed precision
    fp16: bool = False
    bf16: bool = False

    # Misc
    seed: int = 42
    dataloader_num_workers: int = 0


class RewardModelTrainer:
    """
    Trainer for reward models.

    Trains a reward model to assign higher scores to preferred responses
    using ranking losses on human preference data.
    """

    def __init__(
        self,
        model: RewardModel,
        tokenizer,
        train_dataset: RewardModelDataset,
        eval_dataset: Optional[RewardModelDataset] = None,
        config: Optional[RewardModelConfig] = None,
        callbacks: Optional[list[Callable]] = None
    ):
        """
        Args:
            model: RewardModel to train
            tokenizer: Tokenizer for the model
            train_dataset: Training dataset with preference pairs
            eval_dataset: Optional evaluation dataset
            config: Training configuration
            callbacks: Optional callbacks called each step
        """
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.config = config or RewardModelConfig()
        self.callbacks = callbacks or []

        # Setup device
        self.device = next(model.parameters()).device

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

        for name, param in self.model.named_parameters():
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
        Run the reward model training loop.

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

        self.model.train()
        total_loss = 0.0
        total_accuracy = 0.0
        total_margin = 0.0

        progress_bar = tqdm(total=total_steps, desc="Reward Model Training")

        for epoch in range(self.config.num_epochs):
            self.epoch = epoch

            for batch_idx, batch in enumerate(train_loader):
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}

                # Forward pass for chosen responses
                chosen_rewards = self.model.get_rewards(
                    batch["chosen_input_ids"],
                    batch["chosen_attention_mask"]
                )

                # Forward pass for rejected responses
                rejected_rewards = self.model.get_rewards(
                    batch["rejected_input_ids"],
                    batch["rejected_attention_mask"]
                )

                # Compute ranking loss with metrics
                metrics = compute_ranking_loss_with_metrics(
                    chosen_rewards,
                    rejected_rewards,
                    margin=self.config.margin
                )

                loss = metrics["loss"]

                # Scale loss for gradient accumulation
                loss = loss / self.config.gradient_accumulation_steps

                # Backward pass
                loss.backward()

                # Accumulate metrics
                total_loss += loss.item() * self.config.gradient_accumulation_steps
                total_accuracy += metrics["accuracy"].item()
                total_margin += metrics["mean_margin"].item()

                # Update weights every gradient_accumulation_steps
                if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                    # Clip gradients
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
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
                        avg_margin = total_margin / self.config.logging_steps
                        lr = self.scheduler.get_last_lr()[0]

                        progress_bar.set_postfix({
                            "loss": f"{avg_loss:.4f}",
                            "acc": f"{avg_accuracy:.3f}",
                            "margin": f"{avg_margin:.3f}",
                            "lr": f"{lr:.2e}"
                        })

                        total_loss = 0.0
                        total_accuracy = 0.0
                        total_margin = 0.0

                    # Evaluation
                    if self.config.eval_steps and self.global_step % self.config.eval_steps == 0:
                        if self.eval_dataset is not None:
                            eval_metrics = self.evaluate()
                            self.model.train()

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
        Evaluate the reward model on the eval dataset.

        Returns:
            Dictionary with evaluation metrics
        """
        if self.eval_dataset is None:
            return {}

        self.model.eval()

        eval_loader = DataLoader(
            self.eval_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.dataloader_num_workers
        )

        total_loss = 0.0
        total_accuracy = 0.0
        total_margin = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in tqdm(eval_loader, desc="Evaluating"):
                batch = {k: v.to(self.device) for k, v in batch.items()}

                # Get rewards
                chosen_rewards = self.model.get_rewards(
                    batch["chosen_input_ids"],
                    batch["chosen_attention_mask"]
                )
                rejected_rewards = self.model.get_rewards(
                    batch["rejected_input_ids"],
                    batch["rejected_attention_mask"]
                )

                # Compute metrics
                metrics = compute_ranking_loss_with_metrics(
                    chosen_rewards,
                    rejected_rewards,
                    margin=self.config.margin
                )

                total_loss += metrics["loss"].item()
                total_accuracy += metrics["accuracy"].item()
                total_margin += metrics["mean_margin"].item()
                num_batches += 1

        avg_loss = total_loss / num_batches
        avg_accuracy = total_accuracy / num_batches
        avg_margin = total_margin / num_batches

        eval_metrics = {
            "loss": avg_loss,
            "accuracy": avg_accuracy,
            "mean_margin": avg_margin,
        }

        print(f"\nEvaluation - Loss: {avg_loss:.4f}, Accuracy: {avg_accuracy:.4f}, Margin: {avg_margin:.4f}")

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

        # Save reward model
        self.model.save_pretrained(str(checkpoint_dir))

        # Save tokenizer
        self.tokenizer.save_pretrained(str(checkpoint_dir))

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
