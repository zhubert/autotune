"""
Tests for RLHF trainer (basic structure tests).
"""

import pytest
import torch
from pathlib import Path
import tempfile
import shutil
from transformers import GPT2LMHeadModel, GPT2Config, GPT2Tokenizer
from datasets import Dataset

from src.auto_bot_tuner.rlhf.trainer import RewardModelConfig as RLHFConfig
from src.auto_bot_tuner.rlhf.dataset import RewardModelDataset
from src.auto_bot_tuner.rlhf.reward_model import RewardModel


@pytest.fixture(scope="function")
def small_gpt2_model():
    """Create a small GPT-2 model for testing."""
    config = GPT2Config(
        vocab_size=100,
        n_positions=128,
        n_embd=64,
        n_layer=2,
        n_head=2
    )
    model = GPT2LMHeadModel(config)
    return model


@pytest.fixture(scope="function")
def tokenizer(small_vocab_tokenizer):
    """Create a tokenizer for testing."""
    tokenizer = small_vocab_tokenizer
    
    return small_vocab_tokenizer


@pytest.fixture(scope="function")
def reward_dataset(tokenizer):
    """Create a small reward dataset."""
    data = {
        "prompt": ["What is Python?", "Explain ML"],
        "chosen": ["Python is a programming language.", "ML learns from data."],
        "rejected": ["Python is a snake.", "ML is hard."]
    }
    raw_dataset = Dataset.from_dict(data)
    return RewardModelDataset(raw_dataset, tokenizer, max_length=128)


@pytest.fixture(scope="function")
def temp_dir():
    """Create a temporary directory for outputs."""
    temp_path = tempfile.mkdtemp()
    yield temp_path
    shutil.rmtree(temp_path, ignore_errors=True)


class TestRLHFConfig:
    """Test RLHFConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = RLHFConfig()

        assert config.learning_rate == 1e-5
        assert config.batch_size == 4
        assert config.num_epochs == 1

    def test_custom_config(self):
        """Test custom configuration."""
        config = RLHFConfig(
            learning_rate=5e-6,
            batch_size=8,
            num_epochs=2
        )

        assert config.learning_rate == 5e-6
        assert config.batch_size == 8
        assert config.num_epochs == 2


class TestRewardModelTraining:
    """Test reward model training setup."""

    def test_reward_model_creation(self, small_gpt2_model):
        """Test creating a reward model."""
        reward_model = RewardModel(small_gpt2_model)

        assert reward_model is not None
        assert hasattr(reward_model, "value_head")

    def test_reward_model_forward(self, small_gpt2_model):
        """Test reward model forward pass."""
        reward_model = RewardModel(small_gpt2_model)

        batch_size = 2
        seq_len = 10
        input_ids = torch.randint(0, 100, (batch_size, seq_len))

        outputs = reward_model(input_ids, return_dict=True)

        assert "rewards" in outputs
        assert outputs["rewards"].shape == (batch_size,)

    def test_reward_model_frozen_base(self, small_gpt2_model):
        """Test reward model with frozen base."""
        reward_model = RewardModel(small_gpt2_model, freeze_base=True)

        # Base should be frozen
        for param in reward_model.base_model.parameters():
            assert not param.requires_grad

        # Value head should be trainable
        for param in reward_model.value_head.parameters():
            assert param.requires_grad


class TestRLHFComponents:
    """Test RLHF training components."""

    def test_dataset_batch_loading(self, reward_dataset):
        """Test loading batches from reward dataset."""
        from torch.utils.data import DataLoader

        dataloader = DataLoader(reward_dataset, batch_size=2)
        batch = next(iter(dataloader))

        assert "chosen_input_ids" in batch
        assert "rejected_input_ids" in batch
        assert batch["chosen_input_ids"].shape[0] == 2

    def test_reward_model_output_shape(self, small_gpt2_model, reward_dataset):
        """Test that reward model outputs correct shapes."""
        from torch.utils.data import DataLoader

        reward_model = RewardModel(small_gpt2_model)
        dataloader = DataLoader(reward_dataset, batch_size=2)
        batch = next(iter(dataloader))

        chosen_rewards = reward_model.get_rewards(
            batch["chosen_input_ids"],
            batch["chosen_attention_mask"]
        )
        rejected_rewards = reward_model.get_rewards(
            batch["rejected_input_ids"],
            batch["rejected_attention_mask"]
        )

        assert chosen_rewards.shape == (2,)
        assert rejected_rewards.shape == (2,)

    def test_reward_comparison(self, small_gpt2_model, reward_dataset):
        """Test comparing rewards."""
        from torch.utils.data import DataLoader

        reward_model = RewardModel(small_gpt2_model)
        dataloader = DataLoader(reward_dataset, batch_size=2)
        batch = next(iter(dataloader))

        chosen_rewards = reward_model.get_rewards(
            batch["chosen_input_ids"],
            batch["chosen_attention_mask"]
        )
        rejected_rewards = reward_model.get_rewards(
            batch["rejected_input_ids"],
            batch["rejected_attention_mask"]
        )

        # Should produce finite rewards
        assert torch.all(torch.isfinite(chosen_rewards))
        assert torch.all(torch.isfinite(rejected_rewards))


class TestRLHFIntegration:
    """Integration tests for RLHF components."""

    def test_reward_model_training_step(self, small_gpt2_model, reward_dataset):
        """Test a single reward model training step."""
        from torch.utils.data import DataLoader
        from src.auto_bot_tuner.rlhf.loss import compute_ranking_loss_with_metrics

        reward_model = RewardModel(small_gpt2_model)
        optimizer = torch.optim.Adam(reward_model.parameters(), lr=1e-3)

        dataloader = DataLoader(reward_dataset, batch_size=2)
        batch = next(iter(dataloader))

        # Forward pass
        chosen_rewards = reward_model.get_rewards(
            batch["chosen_input_ids"],
            batch["chosen_attention_mask"]
        )
        rejected_rewards = reward_model.get_rewards(
            batch["rejected_input_ids"],
            batch["rejected_attention_mask"]
        )

        # Compute loss
        metrics = compute_ranking_loss_with_metrics(chosen_rewards, rejected_rewards)
        loss = metrics["loss"]

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Check that training worked
        assert torch.isfinite(loss).all()

    def test_output_directory_structure(self, temp_dir):
        """Test that output directory can be created."""
        config = RLHFConfig(output_dir=temp_dir)

        output_path = Path(temp_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        assert output_path.exists()
