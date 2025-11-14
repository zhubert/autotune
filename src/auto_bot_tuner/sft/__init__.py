"""
Supervised Fine-Tuning (SFT) module.

This module provides components for instruction-based fine-tuning of language models.
"""

from .trainer import SFTTrainer, SFTConfig
from .dataset import InstructionDataset, load_instruction_dataset, preview_formatted_sample
from .loss import compute_sft_loss, compute_sft_loss_with_metrics, SFTLoss

__all__ = [
    "SFTTrainer",
    "SFTConfig",
    "InstructionDataset",
    "load_instruction_dataset",
    "preview_formatted_sample",
    "compute_sft_loss",
    "compute_sft_loss_with_metrics",
    "SFTLoss",
]
