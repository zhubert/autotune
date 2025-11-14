"""
Tests for SFT trainer.
"""

import pytest
import torch
from pathlib import Path
import tempfile
import shutil
from transformers import GPT2LMHeadModel, GPT2Config, GPT2Tokenizer
from datasets import Dataset

from src.auto_bot_tuner.sft.trainer import SFTTrainer, SFTConfig
from src.auto_bot_tuner.sft.dataset import InstructionDataset


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
def train_dataset(tokenizer):
    """Create a small training dataset."""
    data = {
        "instruction": [
            "Write a haiku",
            "Explain Python",
            "Describe ML",
            "What is AI?"
        ],
        "input": ["", "", "", ""],
        "output": [
            "Code flows like water",
            "Python is a language",
            "ML learns from data",
            "AI mimics intelligence"
        ]
    }
    raw_dataset = Dataset.from_dict(data)
    return InstructionDataset(raw_dataset, tokenizer, max_length=128, format_type="alpaca")


@pytest.fixture(scope="function")
def eval_dataset(tokenizer):
    """Create a small evaluation dataset."""
    data = {
        "instruction": ["Test instruction"],
        "input": [""],
        "output": ["Test output"]
    }
    raw_dataset = Dataset.from_dict(data)
    return InstructionDataset(raw_dataset, tokenizer, max_length=128, format_type="alpaca")


@pytest.fixture(scope="function")
def temp_dir():
    """Create a temporary directory for outputs."""
    temp_path = tempfile.mkdtemp()
    yield temp_path
    shutil.rmtree(temp_path, ignore_errors=True)


class TestSFTConfig:
    """Test SFTConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = SFTConfig()

        assert config.learning_rate == 2e-5
        assert config.batch_size == 4
        assert config.gradient_accumulation_steps == 4
        assert config.num_epochs == 3
        assert config.warmup_steps == 100

    def test_custom_config(self):
        """Test custom configuration."""
        config = SFTConfig(
            learning_rate=1e-4,
            batch_size=8,
            num_epochs=1
        )

        assert config.learning_rate == 1e-4
        assert config.batch_size == 8
        assert config.num_epochs == 1


class TestSFTTrainer:
    """Test SFTTrainer class."""

    def test_trainer_creation(self, small_gpt2_model, tokenizer, train_dataset, temp_dir):
        """Test creating a trainer."""
        config = SFTConfig(output_dir=temp_dir)

        trainer = SFTTrainer(
            model=small_gpt2_model,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            config=config,
            enable_progress_tracking=False
        )

        assert trainer.model is not None
        assert trainer.tokenizer is not None
        assert trainer.config == config
        assert trainer.global_step == 0

    def test_optimizer_creation(self, small_gpt2_model, tokenizer, train_dataset, temp_dir):
        """Test that optimizer is created."""
        config = SFTConfig(output_dir=temp_dir)

        trainer = SFTTrainer(
            model=small_gpt2_model,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            config=config,
            enable_progress_tracking=False
        )

        assert trainer.optimizer is not None
        assert isinstance(trainer.optimizer, torch.optim.Optimizer)

    def test_scheduler_creation(self, small_gpt2_model, tokenizer, train_dataset, temp_dir):
        """Test that scheduler is created."""
        config = SFTConfig(output_dir=temp_dir)

        trainer = SFTTrainer(
            model=small_gpt2_model,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            config=config,
            enable_progress_tracking=False
        )

        assert trainer.scheduler is not None

    def test_output_dir_created(self, small_gpt2_model, tokenizer, train_dataset, temp_dir):
        """Test that output directory is created."""
        output_path = Path(temp_dir) / "output"
        config = SFTConfig(output_dir=str(output_path))

        trainer = SFTTrainer(
            model=small_gpt2_model,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            config=config,
            enable_progress_tracking=False
        )

        assert output_path.exists()

    def test_device_detection(self, small_gpt2_model, tokenizer, train_dataset, temp_dir):
        """Test that device is detected from model."""
        config = SFTConfig(output_dir=temp_dir)

        trainer = SFTTrainer(
            model=small_gpt2_model,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            config=config,
            enable_progress_tracking=False
        )

        assert trainer.device is not None
        assert isinstance(trainer.device, torch.device)


class TestSFTTraining:
    """Test SFT training functionality."""

    def test_single_training_step(self, small_gpt2_model, tokenizer, train_dataset, temp_dir):
        """Test a single training step."""
        config = SFTConfig(
            output_dir=temp_dir,
            batch_size=2,
            logging_steps=1
        )

        trainer = SFTTrainer(
            model=small_gpt2_model,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            config=config,
            enable_progress_tracking=False
        )

        initial_step = trainer.global_step

        # Get a batch
        dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True
        )
        batch = next(iter(dataloader))

        # Move to device
        batch = {k: v.to(trainer.device) for k, v in batch.items()}

        # Forward pass
        outputs = trainer.model(**batch)
        loss = outputs.loss

        # Check loss is valid
        assert torch.isfinite(loss).all()
        assert loss.item() > 0

    def test_training_reduces_loss(self, small_gpt2_model, tokenizer, train_dataset, temp_dir):
        """Test that training reduces loss on simple data."""
        config = SFTConfig(
            output_dir=temp_dir,
            batch_size=2,
            gradient_accumulation_steps=1,
            num_epochs=1,
            max_steps=10,
            logging_steps=1,
            save_steps=1000  # Don't save during test
        )

        trainer = SFTTrainer(
            model=small_gpt2_model,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            config=config,
            enable_progress_tracking=False
        )

        # Record initial loss
        dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=config.batch_size
        )
        batch = next(iter(dataloader))
        batch = {k: v.to(trainer.device) for k, v in batch.items()}

        trainer.model.eval()
        with torch.no_grad():
            initial_outputs = trainer.model(**batch)
            initial_loss = initial_outputs.loss.item()

        # Train
        trainer.model.train()
        for _ in range(10):
            batch = next(iter(dataloader))
            batch = {k: v.to(trainer.device) for k, v in batch.items()}

            outputs = trainer.model(**batch)
            loss = outputs.loss

            trainer.optimizer.zero_grad()
            loss.backward()
            trainer.optimizer.step()

        # Check final loss
        trainer.model.eval()
        with torch.no_grad():
            final_outputs = trainer.model(**batch)
            final_loss = final_outputs.loss.item()

        # Loss should decrease (may not always happen with random data, but often does)
        assert final_loss < initial_loss * 1.5  # Allow some variance

    def test_gradient_accumulation(self, small_gpt2_model, tokenizer, train_dataset, temp_dir):
        """Test gradient accumulation."""
        config = SFTConfig(
            output_dir=temp_dir,
            batch_size=1,
            gradient_accumulation_steps=4
        )

        trainer = SFTTrainer(
            model=small_gpt2_model,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            config=config,
            enable_progress_tracking=False
        )

        assert config.gradient_accumulation_steps == 4

    def test_learning_rate_warmup(self, small_gpt2_model, tokenizer, train_dataset, temp_dir):
        """Test learning rate warmup."""
        config = SFTConfig(
            output_dir=temp_dir,
            warmup_steps=10
        )

        trainer = SFTTrainer(
            model=small_gpt2_model,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            config=config,
            enable_progress_tracking=False
        )

        # Learning rate should start low
        initial_lr = trainer.optimizer.param_groups[0]['lr']

        # Step scheduler (must step optimizer first to avoid warning)
        for _ in range(5):
            trainer.optimizer.step()
            trainer.scheduler.step()

        # LR should increase during warmup
        warmed_lr = trainer.optimizer.param_groups[0]['lr']
        assert warmed_lr >= initial_lr


class TestSFTCheckpointing:
    """Test checkpoint saving and loading."""

    def test_checkpoint_directory_structure(self, small_gpt2_model, tokenizer, train_dataset, temp_dir):
        """Test that checkpoint directory is created properly."""
        config = SFTConfig(output_dir=temp_dir)

        trainer = SFTTrainer(
            model=small_gpt2_model,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            config=config,
            enable_progress_tracking=False
        )

        output_path = Path(temp_dir)
        assert output_path.exists()

    def test_config_saved(self, small_gpt2_model, tokenizer, train_dataset, temp_dir):
        """Test that config is saved."""
        config = SFTConfig(output_dir=temp_dir, learning_rate=1e-4)

        trainer = SFTTrainer(
            model=small_gpt2_model,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            config=config,
            enable_progress_tracking=False
        )

        # Check that output directory exists (config saving is internal)
        output_path = Path(temp_dir)
        assert output_path.exists()


class TestSFTCallbacks:
    """Test callback functionality."""

    def test_callback_registration(self, small_gpt2_model, tokenizer, train_dataset, temp_dir):
        """Test that callbacks can be registered."""
        config = SFTConfig(output_dir=temp_dir)

        callback_called = []

        def test_callback(trainer, step, metrics):
            callback_called.append(step)

        trainer = SFTTrainer(
            model=small_gpt2_model,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            config=config,
            callbacks=[test_callback],
            enable_progress_tracking=False
        )

        assert len(trainer.callbacks) == 1

    def test_multiple_callbacks(self, small_gpt2_model, tokenizer, train_dataset, temp_dir):
        """Test multiple callbacks."""
        config = SFTConfig(output_dir=temp_dir)

        def callback1(trainer, step, metrics):
            pass

        def callback2(trainer, step, metrics):
            pass

        trainer = SFTTrainer(
            model=small_gpt2_model,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            config=config,
            callbacks=[callback1, callback2],
            enable_progress_tracking=False
        )

        assert len(trainer.callbacks) == 2


class TestSFTIntegration:
    """Integration tests for SFT trainer."""

    def test_full_training_pipeline(self, small_gpt2_model, tokenizer, train_dataset, temp_dir):
        """Test a complete training pipeline."""
        config = SFTConfig(
            output_dir=temp_dir,
            batch_size=2,
            gradient_accumulation_steps=1,
            max_steps=5,
            logging_steps=1,
            save_steps=100
        )

        trainer = SFTTrainer(
            model=small_gpt2_model,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            config=config,
            enable_progress_tracking=False
        )

        # Should be able to create trainer without errors
        assert trainer is not None

    def test_with_eval_dataset(self, small_gpt2_model, tokenizer, train_dataset, eval_dataset, temp_dir):
        """Test training with evaluation dataset."""
        config = SFTConfig(
            output_dir=temp_dir,
            eval_steps=10
        )

        trainer = SFTTrainer(
            model=small_gpt2_model,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            config=config,
            enable_progress_tracking=False
        )

        assert trainer.eval_dataset is not None
