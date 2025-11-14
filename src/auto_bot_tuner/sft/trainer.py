"""
Trainer for Supervised Fine-Tuning (SFT).

This module implements the training loop with gradient accumulation,
learning rate scheduling, and progress tracking.

Integration with progress tracking system:
- Registers training sessions when training starts
- Updates progress periodically during training
- Registers models in the registry upon completion
- Handles failures gracefully
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
import time

from .loss import compute_sft_loss_with_metrics
from .dataset import InstructionDataset


@dataclass
class SFTConfig:
    """Configuration for SFT training."""

    # Training hyperparameters
    learning_rate: float = 2e-5
    batch_size: int = 4
    gradient_accumulation_steps: int = 4  # Effective batch size = batch_size * grad_accum
    num_epochs: int = 3
    max_steps: Optional[int] = None  # If set, overrides num_epochs
    warmup_steps: int = 100
    max_grad_norm: float = 1.0  # Gradient clipping

    # Optimization
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8

    # Logging and checkpointing
    logging_steps: int = 10
    eval_steps: Optional[int] = 500
    save_steps: int = 1000
    save_total_limit: int = 3  # Keep only N most recent checkpoints

    # Paths
    output_dir: str = "checkpoints/sft"
    resume_from_checkpoint: Optional[str] = None

    # Mixed precision
    fp16: bool = False  # Use FP16 training
    bf16: bool = False  # Use BF16 training (better for modern GPUs)

    # Misc
    seed: int = 42
    dataloader_num_workers: int = 0


class SFTTrainer:
    """
    Trainer for supervised fine-tuning of language models.

    Implements:
    - Training loop with gradient accumulation
    - Learning rate warmup and decay
    - Gradient clipping
    - Checkpoint saving and resuming
    - Progress tracking and logging
    - Integration with progress tracking system for multi-model management
    """

    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer,
        train_dataset: InstructionDataset,
        eval_dataset: Optional[InstructionDataset] = None,
        config: Optional[SFTConfig] = None,
        callbacks: Optional[list[Callable]] = None,
        model_id: Optional[str] = None,
        base_model_name: Optional[str] = None,
        dataset_name: Optional[str] = None,
        enable_progress_tracking: bool = True
    ):
        """
        Args:
            model: The language model to fine-tune
            tokenizer: Tokenizer for the model
            train_dataset: Training dataset
            eval_dataset: Optional evaluation dataset
            config: Training configuration
            callbacks: Optional list of callback functions called each step
            model_id: Unique identifier for this model (for progress tracking)
            base_model_name: Name of the base model (e.g., "gpt2", "meta-llama/Llama-3.2-1B")
            dataset_name: Name of the dataset (e.g., "yahma/alpaca-cleaned")
            enable_progress_tracking: Whether to enable progress tracking system
        """
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.config = config or SFTConfig()
        self.callbacks = callbacks or []

        # Progress tracking configuration
        self.model_id = model_id
        self.base_model_name = base_model_name
        self.dataset_name = dataset_name
        self.enable_progress_tracking = enable_progress_tracking
        self.progress_tracker = None
        self.training_session_id = None

        # Setup device
        self.device = next(model.parameters()).device

        # Setup optimizer
        self.optimizer = self._create_optimizer()

        # Setup learning rate scheduler
        self.scheduler = self._create_scheduler()

        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_eval_loss = float('inf')

        # Create output directory
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)

        # Save config
        self._save_config()

    def _create_optimizer(self) -> torch.optim.Optimizer:
        """
        Create AdamW optimizer with weight decay.

        Educational note:
        AdamW is a variant of Adam that properly implements weight decay
        (L2 regularization). It prevents overfitting by penalizing large weights.
        """
        # Separate parameters that should and shouldn't have weight decay
        decay_params = []
        no_decay_params = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue

            # Don't apply weight decay to biases and layer norms
            if "bias" in name or "layer_norm" in name or "layernorm" in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        optimizer_grouped_parameters = [
            {
                "params": decay_params,
                "weight_decay": self.config.weight_decay,
            },
            {
                "params": no_decay_params,
                "weight_decay": 0.0,
            },
        ]

        return AdamW(
            optimizer_grouped_parameters,
            lr=self.config.learning_rate,
            betas=(self.config.adam_beta1, self.config.adam_beta2),
            eps=self.config.adam_epsilon,
        )

    def _create_scheduler(self) -> LambdaLR:
        """
        Create learning rate scheduler with linear warmup and decay.

        Educational note:
        Learning rate warmup gradually increases LR from 0 to target LR
        over the first N steps. This helps stabilize training in early stages.
        After warmup, we linearly decay back to 0 by the end of training.
        """
        def lr_lambda(current_step: int) -> float:
            if current_step < self.config.warmup_steps:
                # Linear warmup
                return float(current_step) / float(max(1, self.config.warmup_steps))
            else:
                # Linear decay
                total_steps = self.config.max_steps or (
                    len(self.train_dataset) // (self.config.batch_size * self.config.gradient_accumulation_steps)
                    * self.config.num_epochs
                )
                progress = float(current_step - self.config.warmup_steps) / float(
                    max(1, total_steps - self.config.warmup_steps)
                )
                return max(0.0, 1.0 - progress)

        return LambdaLR(self.optimizer, lr_lambda)

    def _initialize_progress_tracking(self, total_steps: int):
        """
        Initialize progress tracking system if enabled.

        This integrates with the global progress tracking system to:
        - Register this training session
        - Enable live progress updates
        - Support model registry and lineage tracking
        """
        if not self.enable_progress_tracking:
            return

        try:
            # Import here to avoid circular dependencies and make it optional
            import sys
            from pathlib import Path
            sys.path.insert(0, str(Path(__file__).parent.parent.parent))
            from wizard import ProgressTracker

            self.progress_tracker = ProgressTracker()

            # Start training session
            self.training_session_id = self.progress_tracker.start_training_session(
                method="sft",
                config={
                    "model_id": self.model_id,
                    "base_model": self.base_model_name,
                    "dataset": self.dataset_name,
                    "batch_size": self.config.batch_size,
                    "learning_rate": self.config.learning_rate,
                    "epochs": self.config.num_epochs,
                },
                model_id=self.model_id,
                total_steps=total_steps
            )

            # Mark model as training if model_id provided
            if self.model_id and self.base_model_name:
                checkpoint_path = str(Path(self.config.output_dir) / "final")
                if self.model_id not in self.progress_tracker.registry.state["models"]:
                    # Create new model entry
                    self.progress_tracker.registry.state["models"][self.model_id] = {
                        "id": self.model_id,
                        "base_model": self.base_model_name,
                        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                        "lineage": [],
                        "tags": [],
                        "notes": "",
                        "status": "training",
                        "capabilities": [],
                        "model_type": "generative"
                    }
                    self.progress_tracker.registry.save()

                self.progress_tracker.registry.mark_model_training(
                    model_id=self.model_id,
                    training_id=self.training_session_id,
                    method="sft",
                    checkpoint_path=checkpoint_path,
                    dataset=self.dataset_name,
                    total_steps=total_steps
                )
        except Exception as e:
            # Progress tracking is optional - don't fail if it doesn't work
            print(f"Warning: Could not initialize progress tracking: {e}")
            self.enable_progress_tracking = False

    def _update_progress_tracking(self, current_step: int, metrics: dict):
        """
        Update progress tracking system with current training metrics.

        Called periodically during training to keep the progress system
        in sync with actual training progress.
        """
        if not self.enable_progress_tracking or not self.progress_tracker:
            return

        try:
            self.progress_tracker.update_training_progress(
                session_id=self.training_session_id,
                current_step=current_step,
                metrics=metrics
            )
        except Exception:
            # Silently fail - progress tracking shouldn't interrupt training
            pass

    def _complete_progress_tracking(self, checkpoint_path: str, final_metrics: dict):
        """
        Finalize progress tracking after successful training completion.

        This:
        - Registers the trained model in the model registry
        - Updates the training session as complete
        - Makes the model available for next training stages (DPO, RLHF)
        """
        if not self.enable_progress_tracking or not self.progress_tracker:
            return

        try:
            # Register model in registry
            if self.model_id and self.base_model_name:
                self.progress_tracker.registry.register_model(
                    model_id=self.model_id,
                    base_model=self.base_model_name,
                    training_id=self.training_session_id,
                    method="sft",
                    checkpoint_path=checkpoint_path,
                    dataset=self.dataset_name,
                    tags=["instruction-following"],
                    notes=f"SFT training on {self.dataset_name}",
                    metrics=final_metrics
                )

            # Complete training session
            self.progress_tracker.complete_training_session(
                session_id=self.training_session_id,
                checkpoint_path=checkpoint_path,
                final_metrics=final_metrics
            )
        except Exception as e:
            print(f"Warning: Could not complete progress tracking: {e}")

    def _fail_progress_tracking(self, error: str):
        """
        Mark training as failed in progress tracking system.

        Called when training encounters an error to properly clean up
        the progress tracking state.
        """
        if not self.enable_progress_tracking or not self.progress_tracker:
            return

        try:
            self.progress_tracker.fail_training_session(
                session_id=self.training_session_id,
                error=error
            )
        except Exception:
            pass

    def train(self) -> dict:
        """
        Run the full training loop.

        Returns:
            Dictionary with training statistics
        """
        # Create dataloader
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.dataloader_num_workers,
            pin_memory=True if self.device.type == "cuda" else False
        )

        # Calculate total steps
        steps_per_epoch = len(train_loader) // self.config.gradient_accumulation_steps
        total_steps = self.config.max_steps or (steps_per_epoch * self.config.num_epochs)

        # Initialize progress tracking
        self._initialize_progress_tracking(total_steps)

        # Training loop
        self.model.train()
        total_loss = 0.0
        total_tokens = 0

        progress_bar = tqdm(total=total_steps, desc="Training")

        try:
            for epoch in range(self.config.num_epochs):
                self.epoch = epoch

                for batch_idx, batch in enumerate(train_loader):
                    # Move batch to device
                    batch = {k: v.to(self.device) for k, v in batch.items()}

                    # Forward pass
                    outputs = self.model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"]
                    )

                    # Compute loss
                    metrics = compute_sft_loss_with_metrics(outputs.logits, batch["labels"])
                    loss = metrics["loss"]

                    # Scale loss for gradient accumulation
                    loss = loss / self.config.gradient_accumulation_steps

                    # Backward pass
                    loss.backward()

                    # Accumulate metrics
                    total_loss += loss.item() * self.config.gradient_accumulation_steps
                    total_tokens += metrics["num_tokens"].item()

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
                            lr = self.scheduler.get_last_lr()[0]

                            progress_bar.set_postfix({
                                "loss": f"{avg_loss:.4f}",
                                "lr": f"{lr:.2e}",
                                "tokens": total_tokens
                            })

                            # Update progress tracking
                            self._update_progress_tracking(
                                current_step=self.global_step,
                                metrics={"loss": avg_loss, "learning_rate": lr}
                            )

                            total_loss = 0.0

                    # Evaluation
                    if self.config.eval_steps and self.global_step % self.config.eval_steps == 0:
                        if self.eval_dataset is not None:
                            eval_metrics = self.evaluate()
                            self.model.train()  # Back to training mode

                            if eval_metrics["loss"] < self.best_eval_loss:
                                self.best_eval_loss = eval_metrics["loss"]
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
                        summary = self._get_training_summary()

                        # Complete progress tracking
                        final_checkpoint = str(Path(self.config.output_dir) / "final")
                        self._complete_progress_tracking(
                            checkpoint_path=final_checkpoint,
                            final_metrics=summary
                        )

                        return summary

            progress_bar.close()

            # Save final checkpoint
            final_checkpoint_path = self.save_checkpoint(is_final=True)

            # Get training summary
            summary = self._get_training_summary()

            # Complete progress tracking
            self._complete_progress_tracking(
                checkpoint_path=final_checkpoint_path or str(Path(self.config.output_dir) / "final"),
                final_metrics=summary
            )

            return summary

        except Exception as e:
            # Handle training failure
            progress_bar.close()
            self._fail_progress_tracking(str(e))
            raise

    def evaluate(self) -> dict:
        """
        Evaluate the model on the eval dataset.

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
        total_tokens = 0
        num_batches = 0

        with torch.no_grad():
            for batch in tqdm(eval_loader, desc="Evaluating"):
                batch = {k: v.to(self.device) for k, v in batch.items()}

                outputs = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"]
                )

                metrics = compute_sft_loss_with_metrics(outputs.logits, batch["labels"])

                total_loss += metrics["loss"].item()
                total_accuracy += metrics["accuracy"].item()
                total_tokens += metrics["num_tokens"].item()
                num_batches += 1

        avg_loss = total_loss / num_batches
        avg_accuracy = total_accuracy / num_batches
        perplexity = torch.exp(torch.tensor(avg_loss)).item()

        eval_metrics = {
            "loss": avg_loss,
            "perplexity": perplexity,
            "accuracy": avg_accuracy,
            "num_tokens": total_tokens
        }

        print(f"\nEvaluation - Loss: {avg_loss:.4f}, Perplexity: {perplexity:.2f}, Accuracy: {avg_accuracy:.4f}")

        return eval_metrics

    def save_checkpoint(self, is_best: bool = False, is_final: bool = False) -> None:
        """
        Save model checkpoint.

        Args:
            is_best: Whether this is the best model so far
            is_final: Whether this is the final checkpoint
        """
        if is_final:
            checkpoint_dir = Path(self.config.output_dir) / "final"
        elif is_best:
            checkpoint_dir = Path(self.config.output_dir) / "best"
        else:
            checkpoint_dir = Path(self.config.output_dir) / f"checkpoint-{self.global_step}"

        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Save model and tokenizer
        self.model.save_pretrained(checkpoint_dir)
        self.tokenizer.save_pretrained(checkpoint_dir)

        # Save training state
        state = {
            "global_step": self.global_step,
            "epoch": self.epoch,
            "best_eval_loss": self.best_eval_loss,
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
        }

        torch.save(state, checkpoint_dir / "training_state.pt")

        print(f"\nCheckpoint saved to {checkpoint_dir}")

    def _save_config(self) -> None:
        """Save training configuration to output directory."""
        config_path = Path(self.config.output_dir) / "training_config.json"
        with open(config_path, 'w') as f:
            json.dump(vars(self.config), f, indent=2)

    def _get_training_summary(self) -> dict:
        """Get summary of training run."""
        return {
            "total_steps": self.global_step,
            "epochs_completed": self.epoch + 1,
            "best_eval_loss": self.best_eval_loss if self.best_eval_loss != float('inf') else None,
        }
