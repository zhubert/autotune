"""
Tests for RLHF reward model.
"""

import pytest
import torch
from transformers import GPT2LMHeadModel, GPT2Config, GPT2Tokenizer

from src.auto_bot_tuner.rlhf.reward_model import (
    RewardModel,
    create_reward_model_from_pretrained
)
from src.auto_bot_tuner.rlhf.loss import compute_ranking_loss_with_metrics


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
def device():
    """Get device for testing."""
    return torch.device("cpu")


class TestRewardModel:
    """Test RewardModel class."""

    def test_model_creation(self, small_gpt2_model):
        """Test creating a reward model."""
        reward_model = RewardModel(small_gpt2_model)

        assert reward_model is not None
        assert hasattr(reward_model, "base_model")
        assert hasattr(reward_model, "value_head")

    def test_forward_pass(self, small_gpt2_model, device):
        """Test forward pass through reward model."""
        reward_model = RewardModel(small_gpt2_model).to(device)

        batch_size = 2
        seq_len = 10
        input_ids = torch.randint(0, 100, (batch_size, seq_len)).to(device)
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long).to(device)

        outputs = reward_model(input_ids, attention_mask, return_dict=True)

        assert "rewards" in outputs
        assert outputs["rewards"].shape == (batch_size,)

    def test_reward_shape(self, small_gpt2_model, device):
        """Test that rewards have correct shape."""
        reward_model = RewardModel(small_gpt2_model).to(device)

        batch_size = 4
        seq_len = 15
        input_ids = torch.randint(0, 100, (batch_size, seq_len)).to(device)

        rewards = reward_model(input_ids, return_dict=False)

        assert rewards.shape == (batch_size,)
        assert rewards.dim() == 1

    def test_with_attention_mask(self, small_gpt2_model, device):
        """Test reward model with attention mask."""
        reward_model = RewardModel(small_gpt2_model).to(device)

        batch_size = 2
        seq_len = 10
        input_ids = torch.randint(0, 100, (batch_size, seq_len)).to(device)
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long).to(device)
        # Mask last 2 tokens
        attention_mask[:, -2:] = 0

        rewards = reward_model(input_ids, attention_mask, return_dict=False)

        assert rewards.shape == (batch_size,)
        assert torch.all(torch.isfinite(rewards))

    def test_frozen_base_model(self, small_gpt2_model):
        """Test freezing base model."""
        reward_model = RewardModel(small_gpt2_model, freeze_base=True)

        # Base model should be frozen
        for param in reward_model.base_model.parameters():
            assert not param.requires_grad

        # Value head should be trainable
        for param in reward_model.value_head.parameters():
            assert param.requires_grad

    def test_trainable_base_model(self, small_gpt2_model):
        """Test with trainable base model."""
        reward_model = RewardModel(small_gpt2_model, freeze_base=False)

        # All parameters should be trainable
        trainable_params = sum(p.requires_grad for p in reward_model.parameters())
        assert trainable_params > 0

    def test_value_head_initialization(self, small_gpt2_model):
        """Test that value head is initialized with small weights."""
        reward_model = RewardModel(small_gpt2_model)

        # Get linear layer weights
        linear_layer = reward_model.value_head[1]

        # Weights should be small
        assert linear_layer.weight.abs().mean() < 0.1
        # Bias should be zero
        assert linear_layer.bias.abs().max() < 1e-6

    def test_get_rewards_method(self, small_gpt2_model, device):
        """Test get_rewards convenience method."""
        reward_model = RewardModel(small_gpt2_model).to(device)

        batch_size = 2
        seq_len = 10
        input_ids = torch.randint(0, 100, (batch_size, seq_len)).to(device)

        rewards = reward_model.get_rewards(input_ids)

        assert rewards.shape == (batch_size,)


class TestCreateRewardModelFromPretrained:
    """Test create_reward_model_from_pretrained function."""

    def test_creation_from_pretrained(self, tokenizer):
        """Test creating reward model from pretrained model name."""
        # Use gpt2 as it's small and widely available
        reward_model = create_reward_model_from_pretrained("gpt2", tokenizer)

        assert reward_model is not None
        assert isinstance(reward_model, RewardModel)

    def test_forward_after_creation(self, tokenizer, device):
        """Test that created model works."""
        reward_model = create_reward_model_from_pretrained("gpt2", tokenizer)
        reward_model = reward_model.to(device)

        input_ids = torch.tensor([[1, 2, 3, 4, 5]]).to(device)

        rewards = reward_model.get_rewards(input_ids)

        assert rewards.shape == (1,)
        assert torch.isfinite(rewards).all()

    def test_tokenizer_compatibility(self, tokenizer, device):
        """Test that tokenizer works with created model."""
        reward_model = create_reward_model_from_pretrained("gpt2", tokenizer)
        reward_model = reward_model.to(device)

        text = "This is a test"
        tokens = tokenizer(text, return_tensors="pt").to(device)

        rewards = reward_model.get_rewards(
            tokens["input_ids"],
            tokens["attention_mask"]
        )

        assert rewards.shape == (1,)


class TestComputeRewardModelLoss:
    """Test compute_ranking_loss_with_metrics function."""

    def test_basic_loss_computation(self, device):
        """Test basic loss computation."""
        batch_size = 4

        chosen_rewards = torch.randn(batch_size).to(device)
        rejected_rewards = torch.randn(batch_size).to(device)

        metrics = compute_ranking_loss_with_metrics(chosen_rewards, rejected_rewards)
        loss = metrics["loss"]

        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0  # Scalar
        assert loss.item() >= 0

        assert "loss" in metrics
        assert "accuracy" in metrics
        assert "mean_margin" in metrics

    def test_loss_when_chosen_better(self, device):
        """Test that loss is low when chosen > rejected."""
        batch_size = 4

        # Chosen always better
        chosen_rewards = torch.ones(batch_size).to(device) * 5.0
        rejected_rewards = torch.ones(batch_size).to(device) * 1.0

        metrics = compute_ranking_loss_with_metrics(chosen_rewards, rejected_rewards)
        loss = metrics["loss"]

        # Loss should be relatively low
        assert loss.item() < 1.0
        # Accuracy should be 100%
        assert metrics["accuracy"].item() == 1.0

    def test_loss_when_rejected_better(self, device):
        """Test that loss is high when rejected > chosen."""
        batch_size = 4

        # Rejected better (wrong!)
        chosen_rewards = torch.ones(batch_size).to(device) * 1.0
        rejected_rewards = torch.ones(batch_size).to(device) * 5.0

        metrics = compute_ranking_loss_with_metrics(chosen_rewards, rejected_rewards)
        loss = metrics["loss"]

        # Loss should be high
        assert loss.item() > 0.5
        # Accuracy should be 0%
        assert metrics["accuracy"].item() == 0.0

    def test_margin_metric(self, device):
        """Test that margin is computed correctly."""
        batch_size = 4

        chosen_rewards = torch.tensor([5.0, 4.0, 3.0, 2.0]).to(device)
        rejected_rewards = torch.tensor([1.0, 1.0, 1.0, 1.0]).to(device)

        metrics = compute_ranking_loss_with_metrics(chosen_rewards, rejected_rewards)

        # Margin should be positive (chosen - rejected)
        expected_margin = (chosen_rewards - rejected_rewards).mean()
        assert torch.isclose(metrics["mean_margin"], expected_margin)

    def test_accuracy_computation(self, device):
        """Test accuracy computation."""
        batch_size = 4

        # Half correct, half incorrect
        chosen_rewards = torch.tensor([5.0, 5.0, 1.0, 1.0]).to(device)
        rejected_rewards = torch.tensor([1.0, 1.0, 5.0, 5.0]).to(device)

        metrics = compute_ranking_loss_with_metrics(chosen_rewards, rejected_rewards)

        # Accuracy should be 50%
        assert metrics["accuracy"].item() == 0.5


class TestRewardModelTraining:
    """Test reward model training."""

    def test_training_step(self, small_gpt2_model, device):
        """Test a single training step."""
        reward_model = RewardModel(small_gpt2_model).to(device)
        optimizer = torch.optim.Adam(reward_model.parameters(), lr=1e-3)

        batch_size = 2
        seq_len = 10

        chosen_input_ids = torch.randint(0, 100, (batch_size, seq_len)).to(device)
        rejected_input_ids = torch.randint(0, 100, (batch_size, seq_len)).to(device)

        # Forward pass
        chosen_rewards = reward_model.get_rewards(chosen_input_ids)
        rejected_rewards = reward_model.get_rewards(rejected_input_ids)

        # Compute loss
        metrics = compute_ranking_loss_with_metrics(chosen_rewards, rejected_rewards)
        loss = metrics["loss"]

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Check that gradients exist
        has_gradients = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in reward_model.parameters() if p.requires_grad
        )
        assert has_gradients

    def test_frozen_base_training(self, small_gpt2_model, device):
        """Test training with frozen base model."""
        reward_model = RewardModel(small_gpt2_model, freeze_base=True).to(device)

        optimizer = torch.optim.Adam(
            [p for p in reward_model.parameters() if p.requires_grad],
            lr=1e-3
        )

        batch_size = 2
        seq_len = 10

        chosen_input_ids = torch.randint(0, 100, (batch_size, seq_len)).to(device)
        rejected_input_ids = torch.randint(0, 100, (batch_size, seq_len)).to(device)

        # Forward
        chosen_rewards = reward_model.get_rewards(chosen_input_ids)
        rejected_rewards = reward_model.get_rewards(rejected_input_ids)

        # Loss and backward
        metrics = compute_ranking_loss_with_metrics(chosen_rewards, rejected_rewards)
        loss = metrics["loss"]
        optimizer.zero_grad()
        loss.backward()

        # Base model should have no gradients
        for param in reward_model.base_model.parameters():
            assert param.grad is None

        # Value head should have gradients
        for param in reward_model.value_head.parameters():
            assert param.grad is not None

    def test_convergence_on_simple_task(self, small_gpt2_model, device):
        """Test that reward model can learn simple preferences."""
        reward_model = RewardModel(small_gpt2_model).to(device)
        optimizer = torch.optim.Adam(reward_model.parameters(), lr=0.01)

        # Create simple data: sequence of 1s is better than 0s
        batch_size = 4
        seq_len = 10

        chosen_input_ids = torch.ones(batch_size, seq_len, dtype=torch.long).to(device)
        rejected_input_ids = torch.zeros(batch_size, seq_len, dtype=torch.long).to(device)

        initial_loss = None

        # Train for several steps
        for step in range(20):
            chosen_rewards = reward_model.get_rewards(chosen_input_ids)
            rejected_rewards = reward_model.get_rewards(rejected_input_ids)

            metrics = compute_ranking_loss_with_metrics(chosen_rewards, rejected_rewards)
            loss = metrics["loss"]

            if step == 0:
                initial_loss = loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Loss should decrease
        final_loss = loss.item()
        assert final_loss < initial_loss

        # Accuracy should improve
        assert metrics["accuracy"].item() > 0.5


class TestRewardModelIntegration:
    """Integration tests for reward model."""

    def test_full_pipeline(self, tokenizer, device):
        """Test complete reward model pipeline."""
        # Create model
        reward_model = create_reward_model_from_pretrained("gpt2", tokenizer)
        reward_model = reward_model.to(device)

        # Tokenize texts
        chosen_text = "This is a helpful response."
        rejected_text = "Bad response."

        chosen_tokens = tokenizer(chosen_text, return_tensors="pt").to(device)
        rejected_tokens = tokenizer(rejected_text, return_tensors="pt").to(device)

        # Get rewards
        chosen_reward = reward_model.get_rewards(
            chosen_tokens["input_ids"],
            chosen_tokens["attention_mask"]
        )
        rejected_reward = reward_model.get_rewards(
            rejected_tokens["input_ids"],
            rejected_tokens["attention_mask"]
        )

        assert chosen_reward.shape == (1,)
        assert rejected_reward.shape == (1,)

    def test_batch_processing(self, tokenizer, device):
        """Test processing batches of text."""
        reward_model = create_reward_model_from_pretrained("gpt2", tokenizer)
        reward_model = reward_model.to(device)

        texts = [
            "First response",
            "Second response",
            "Third response"
        ]

        tokens = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=32
        ).to(device)

        rewards = reward_model.get_rewards(
            tokens["input_ids"],
            tokens["attention_mask"]
        )

        assert rewards.shape == (3,)
        assert torch.all(torch.isfinite(rewards))
