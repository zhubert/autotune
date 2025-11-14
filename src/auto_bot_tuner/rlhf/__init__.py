"""
RLHF (Reinforcement Learning from Human Feedback) module.

This module provides components for training reward models from human preference
data and using them to train language models with PPO (Proximal Policy Optimization).

Implements:
- Reward Model architecture and training
- PPO trainer for policy optimization
- Value network for advantage estimation
- Rollout buffer and GAE
- Ranking losses for preference learning
- Dataset utilities for preference data
"""

from .reward_model import (
    RewardModel,
    EnsembleRewardModel,
    create_reward_model_from_pretrained,
    compare_responses
)
from .trainer import RewardModelTrainer, RewardModelConfig
from .dataset import (
    RewardModelDataset,
    RankedResponsesDataset,
    ScoredResponsesDataset,
    load_reward_dataset,
    convert_preference_to_reward_format,
    validate_reward_dataset,
    preview_reward_sample,
    create_synthetic_preference_data
)
from .loss import (
    compute_ranking_loss,
    compute_ranking_loss_with_metrics,
    compute_pairwise_loss,
    compute_listwise_loss,
    compute_contrastive_loss,
    RankingLoss,
    compute_reward_model_loss,
    validate_reward_predictions
)
from .ppo_trainer import PPOTrainer, PPOConfig, PromptDataset, create_reference_model
from .value_network import (
    ValueNetwork,
    create_value_network_from_pretrained,
    create_value_network_from_policy
)
from .rollout_buffer import (
    RolloutBuffer,
    RolloutBatch,
    compute_gae,
    whiten_advantages
)
from .ppo_loss import (
    compute_ppo_loss,
    compute_value_loss,
    compute_entropy_bonus,
    compute_kl_penalty,
    compute_ppo_total_loss
)

__all__ = [
    # Reward Model
    "RewardModel",
    "EnsembleRewardModel",
    "create_reward_model_from_pretrained",
    "compare_responses",
    # Reward Model Trainer
    "RewardModelTrainer",
    "RewardModelConfig",
    # PPO Trainer
    "PPOTrainer",
    "PPOConfig",
    "PromptDataset",
    "create_reference_model",
    # Value Network
    "ValueNetwork",
    "create_value_network_from_pretrained",
    "create_value_network_from_policy",
    # Rollout Buffer
    "RolloutBuffer",
    "RolloutBatch",
    "compute_gae",
    "whiten_advantages",
    # PPO Loss
    "compute_ppo_loss",
    "compute_value_loss",
    "compute_entropy_bonus",
    "compute_kl_penalty",
    "compute_ppo_total_loss",
    # Dataset
    "RewardModelDataset",
    "RankedResponsesDataset",
    "ScoredResponsesDataset",
    "load_reward_dataset",
    "convert_preference_to_reward_format",
    "validate_reward_dataset",
    "preview_reward_sample",
    "create_synthetic_preference_data",
    # Reward Model Loss
    "compute_ranking_loss",
    "compute_ranking_loss_with_metrics",
    "compute_pairwise_loss",
    "compute_listwise_loss",
    "compute_contrastive_loss",
    "RankingLoss",
    "compute_reward_model_loss",
    "validate_reward_predictions",
]
