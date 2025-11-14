"""
Tests for DPO trainer (basic structure tests).
"""

import pytest
import torch
from pathlib import Path
import tempfile
import shutil
from transformers import GPT2LMHeadModel, GPT2Config, GPT2Tokenizer
from datasets import Dataset

from src.auto_bot_tuner.dpo.trainer import DPOTrainer, DPOConfig
from src.auto_bot_tuner.dpo.dataset import PreferenceDataset


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
def preference_dataset(tokenizer):
    """Create a small preference dataset."""
    data = {
        "prompt": ["What is Python?", "Explain ML"],
        "chosen": ["Python is a programming language.", "ML learns from data."],
        "rejected": ["Python is a snake.", "ML is hard."]
    }
    raw_dataset = Dataset.from_dict(data)
    return PreferenceDataset(raw_dataset, tokenizer, max_length=128)


@pytest.fixture(scope="function")
def temp_dir():
    """Create a temporary directory for outputs."""
    temp_path = tempfile.mkdtemp()
    yield temp_path
    shutil.rmtree(temp_path, ignore_errors=True)


class TestDPOConfig:
    """Test DPOConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = DPOConfig()

        assert config.beta == 0.1
        assert config.label_smoothing == 0.0
        assert config.learning_rate == 5e-7
        assert config.batch_size == 4

    def test_custom_config(self):
        """Test custom configuration."""
        config = DPOConfig(
            beta=0.2,
            learning_rate=1e-6,
            batch_size=8
        )

        assert config.beta == 0.2
        assert config.learning_rate == 1e-6
        assert config.batch_size == 8


class TestDPOTrainer:
    """Test DPOTrainer class."""

    def test_trainer_creation(self, small_gpt2_model, tokenizer, preference_dataset, temp_dir):
        """Test creating a DPO trainer."""
        import copy
        policy_model = small_gpt2_model
        reference_model = copy.deepcopy(small_gpt2_model)

        # Freeze reference model
        for param in reference_model.parameters():
            param.requires_grad = False

        config = DPOConfig(output_dir=temp_dir)

        trainer = DPOTrainer(
            policy_model=policy_model,
            reference_model=reference_model,
            tokenizer=tokenizer,
            train_dataset=preference_dataset,
            config=config
        )

        assert trainer.policy_model is not None
        assert trainer.reference_model is not None
        assert trainer.config == config

    def test_reference_model_frozen(self, small_gpt2_model, tokenizer, preference_dataset, temp_dir):
        """Test that reference model is frozen."""
        import copy
        policy_model = small_gpt2_model
        reference_model = copy.deepcopy(small_gpt2_model)

        # Freeze reference model
        for param in reference_model.parameters():
            param.requires_grad = False

        config = DPOConfig(output_dir=temp_dir)

        trainer = DPOTrainer(
            policy_model=policy_model,
            reference_model=reference_model,
            tokenizer=tokenizer,
            train_dataset=preference_dataset,
            config=config
        )

        # All reference model params should be frozen
        for param in trainer.reference_model.parameters():
            assert not param.requires_grad

    def test_policy_model_trainable(self, small_gpt2_model, tokenizer, preference_dataset, temp_dir):
        """Test that policy model is trainable."""
        import copy
        policy_model = small_gpt2_model
        reference_model = copy.deepcopy(small_gpt2_model)

        for param in reference_model.parameters():
            param.requires_grad = False

        config = DPOConfig(output_dir=temp_dir)

        trainer = DPOTrainer(
            policy_model=policy_model,
            reference_model=reference_model,
            tokenizer=tokenizer,
            train_dataset=preference_dataset,
            config=config
        )

        # Policy model should have trainable params
        trainable = sum(p.requires_grad for p in trainer.policy_model.parameters())
        assert trainable > 0

    def test_optimizer_creation(self, small_gpt2_model, tokenizer, preference_dataset, temp_dir):
        """Test that optimizer is created."""
        import copy
        policy_model = small_gpt2_model
        reference_model = copy.deepcopy(small_gpt2_model)

        for param in reference_model.parameters():
            param.requires_grad = False

        config = DPOConfig(output_dir=temp_dir)

        trainer = DPOTrainer(
            policy_model=policy_model,
            reference_model=reference_model,
            tokenizer=tokenizer,
            train_dataset=preference_dataset,
            config=config
        )

        assert trainer.optimizer is not None
        assert isinstance(trainer.optimizer, torch.optim.Optimizer)

    def test_output_dir_created(self, small_gpt2_model, tokenizer, preference_dataset, temp_dir):
        """Test that output directory is created."""
        import copy
        policy_model = small_gpt2_model
        reference_model = copy.deepcopy(small_gpt2_model)

        for param in reference_model.parameters():
            param.requires_grad = False

        output_path = Path(temp_dir) / "dpo_output"
        config = DPOConfig(output_dir=str(output_path))

        trainer = DPOTrainer(
            policy_model=policy_model,
            reference_model=reference_model,
            tokenizer=tokenizer,
            train_dataset=preference_dataset,
            config=config
        )

        assert output_path.exists()


class TestDPOBetaParameter:
    """Test DPO beta parameter."""

    def test_beta_stored(self, small_gpt2_model, tokenizer, preference_dataset, temp_dir):
        """Test that beta parameter is stored."""
        import copy
        policy_model = small_gpt2_model
        reference_model = copy.deepcopy(small_gpt2_model)

        for param in reference_model.parameters():
            param.requires_grad = False

        config = DPOConfig(output_dir=temp_dir, beta=0.5)

        trainer = DPOTrainer(
            policy_model=policy_model,
            reference_model=reference_model,
            tokenizer=tokenizer,
            train_dataset=preference_dataset,
            config=config
        )

        assert trainer.config.beta == 0.5

    def test_different_beta_values(self, small_gpt2_model, tokenizer, preference_dataset, temp_dir):
        """Test creating trainers with different beta values."""
        import copy

        for beta in [0.1, 0.2, 0.5]:
            policy_model = copy.deepcopy(small_gpt2_model)
            reference_model = copy.deepcopy(small_gpt2_model)

            for param in reference_model.parameters():
                param.requires_grad = False

            config = DPOConfig(output_dir=temp_dir, beta=beta)

            trainer = DPOTrainer(
                policy_model=policy_model,
                reference_model=reference_model,
                tokenizer=tokenizer,
                train_dataset=preference_dataset,
                config=config
            )

            assert trainer.config.beta == beta
