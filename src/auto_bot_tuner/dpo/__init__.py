"""
Direct Preference Optimization (DPO) module.

DPO is a simpler alternative to RLHF that directly optimizes language models
using preference data without requiring a reward model or RL training.
"""

from .trainer import DPOTrainer, DPOConfig, create_reference_model
from .dataset import (
    PreferenceDataset,
    ConversationalPreferenceDataset,
    load_preference_dataset,
    convert_reward_dataset_to_preference,
    preview_preference_sample,
    validate_preference_dataset
)
from .loss import (
    compute_dpo_loss,
    compute_dpo_loss_with_metrics,
    get_batch_logps,
    compute_dpo_loss_from_logits,
    DPOLoss
)

__all__ = [
    "DPOTrainer",
    "DPOConfig",
    "create_reference_model",
    "PreferenceDataset",
    "ConversationalPreferenceDataset",
    "load_preference_dataset",
    "convert_reward_dataset_to_preference",
    "preview_preference_sample",
    "validate_preference_dataset",
    "compute_dpo_loss",
    "compute_dpo_loss_with_metrics",
    "get_batch_logps",
    "compute_dpo_loss_from_logits",
    "DPOLoss",
]
