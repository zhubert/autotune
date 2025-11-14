"""
Tests for PPO (RLHF) components.

Tests cover:
- Rollout buffer operations
- GAE computation
- Value network
- PPO loss functions
"""

import pytest
import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Config, GPT2Tokenizer

from src.auto_bot_tuner.rlhf.rollout_buffer import (
    RolloutBuffer,
    RolloutBatch,
    compute_gae,
    whiten_advantages
)
from src.auto_bot_tuner.rlhf.value_network import (
    ValueNetwork,
    create_value_network_from_policy
)
from src.auto_bot_tuner.rlhf.ppo_loss import (
    compute_ppo_loss,
    compute_value_loss,
    compute_entropy_bonus,
    compute_kl_penalty,
    compute_ppo_total_loss
)


@pytest.fixture(scope="function")
def device():
    """Get device for testing."""
    return torch.device("cpu")


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


# ============================================================================
# Rollout Buffer Tests
# ============================================================================

def test_rollout_buffer_creation():
    """Test creating a rollout buffer."""
    buffer = RolloutBuffer(buffer_size=10)
    assert len(buffer) == 0
    assert not buffer.is_full()


def test_rollout_buffer_add():
    """Test adding rollouts to buffer."""
    buffer = RolloutBuffer(buffer_size=5)

    # Add a rollout
    query = torch.tensor([1, 2, 3])
    response = torch.tensor([4, 5, 6])
    logprobs = torch.tensor([0.1, 0.2, 0.3])
    values = torch.tensor([1.0, 1.1, 1.2])
    reward = torch.tensor(5.0)
    advantage = torch.tensor([0.5, 0.6, 0.7])
    returns = torch.tensor([1.5, 1.6, 1.7])

    buffer.add(query, response, logprobs, values, reward, advantage, returns)

    assert len(buffer) == 1
    assert not buffer.is_full()


def test_rollout_buffer_full():
    """Test buffer becoming full."""
    buffer = RolloutBuffer(buffer_size=2)

    for i in range(2):
        buffer.add(
            torch.tensor([i]),
            torch.tensor([i]),
            torch.tensor([0.1]),
            torch.tensor([1.0]),
            torch.tensor(1.0),
            torch.tensor([0.5]),
            torch.tensor([1.5])
        )

    assert buffer.is_full()

    # Should raise error when adding to full buffer
    with pytest.raises(ValueError):
        buffer.add(
            torch.tensor([3]),
            torch.tensor([3]),
            torch.tensor([0.1]),
            torch.tensor([1.0]),
            torch.tensor(1.0),
            torch.tensor([0.5]),
            torch.tensor([1.5])
        )


def test_rollout_buffer_get_batch():
    """Test retrieving a batch from buffer."""
    buffer = RolloutBuffer(buffer_size=3)

    # Add rollouts with different lengths
    for i in range(3):
        length = i + 2  # lengths: 2, 3, 4
        buffer.add(
            torch.ones(length),
            torch.ones(length) * i,
            torch.ones(length) * 0.1,
            torch.ones(length) * 1.0,
            torch.tensor(float(i)),
            torch.ones(length) * 0.5,
            torch.ones(length) * 1.5
        )

    # Get batch
    batch = buffer.get_batch([0, 1, 2])

    assert isinstance(batch, RolloutBatch)
    assert batch.query_tensors.shape[0] == 3
    assert batch.response_tensors.shape[0] == 3
    assert batch.rewards.shape[0] == 3


def test_rollout_buffer_clear():
    """Test clearing the buffer."""
    buffer = RolloutBuffer(buffer_size=5)

    # Add some rollouts
    for i in range(3):
        buffer.add(
            torch.tensor([i]),
            torch.tensor([i]),
            torch.tensor([0.1]),
            torch.tensor([1.0]),
            torch.tensor(1.0),
            torch.tensor([0.5]),
            torch.tensor([1.5])
        )

    assert len(buffer) == 3

    buffer.clear()

    assert len(buffer) == 0
    assert not buffer.is_full()


# ============================================================================
# GAE Tests
# ============================================================================

def test_compute_gae_basic():
    """Test basic GAE computation."""
    rewards = torch.tensor([0.0, 0.0, 1.0])
    values = torch.tensor([0.5, 0.6, 0.7])

    advantages, returns = compute_gae(rewards, values, gamma=0.99, lam=0.95)

    assert advantages.shape == rewards.shape
    assert returns.shape == rewards.shape

    # Advantages should sum to approximately the total reward
    # (not exactly due to discounting and baseline)
    assert advantages.sum() > 0  # Should be positive for positive reward


def test_compute_gae_with_mask():
    """Test GAE computation with masking."""
    rewards = torch.tensor([0.0, 0.0, 1.0, 0.0])  # Last is padding
    values = torch.tensor([0.5, 0.6, 0.7, 0.0])
    mask = torch.tensor([1.0, 1.0, 1.0, 0.0])  # Last position masked

    advantages, returns = compute_gae(rewards, values, gamma=0.99, lam=0.95, mask=mask)

    # Masked position should have zero advantage
    assert advantages[3] == 0.0
    assert returns[3] == 0.0


def test_compute_gae_batch():
    """Test GAE computation with batch dimension."""
    batch_size = 2
    seq_len = 3

    rewards = torch.zeros(batch_size, seq_len)
    rewards[:, -1] = 1.0  # Reward at end

    values = torch.rand(batch_size, seq_len)

    advantages, returns = compute_gae(rewards, values)

    assert advantages.shape == (batch_size, seq_len)
    assert returns.shape == (batch_size, seq_len)


def test_whiten_advantages():
    """Test advantage normalization."""
    advantages = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])

    normalized = whiten_advantages(advantages)

    # Should have approximately mean 0 and std 1
    assert abs(normalized.mean().item()) < 1e-5
    assert abs(normalized.std().item() - 1.0) < 1e-5


def test_whiten_advantages_with_mask():
    """Test advantage normalization with masking."""
    advantages = torch.tensor([1.0, 2.0, 3.0, 0.0])
    mask = torch.tensor([1.0, 1.0, 1.0, 0.0])

    normalized = whiten_advantages(advantages, mask)

    # Masked position should remain 0
    assert normalized[3] == 0.0


# ============================================================================
# Value Network Tests
# ============================================================================

def test_value_network_creation(small_gpt2_model):
    """Test creating a value network."""
    value_net = ValueNetwork(small_gpt2_model)

    assert value_net is not None
    assert hasattr(value_net, "value_head")


def test_value_network_forward(small_gpt2_model):
    """Test forward pass through value network."""
    value_net = ValueNetwork(small_gpt2_model)

    batch_size = 2
    seq_len = 10
    input_ids = torch.randint(0, 100, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)

    outputs = value_net(input_ids, attention_mask, return_dict=True)

    assert "values" in outputs
    assert outputs["values"].shape == (batch_size, seq_len)


def test_value_network_from_policy(small_gpt2_model):
    """Test creating value network from policy."""
    value_net = create_value_network_from_policy(small_gpt2_model)

    assert value_net is not None
    assert isinstance(value_net, ValueNetwork)


def test_value_network_freeze_base(small_gpt2_model):
    """Test freezing base model."""
    value_net = ValueNetwork(small_gpt2_model, freeze_base=True)

    # Base model should be frozen
    for param in value_net.base_model.parameters():
        assert not param.requires_grad

    # Value head should be trainable
    for param in value_net.value_head.parameters():
        assert param.requires_grad


# ============================================================================
# PPO Loss Tests
# ============================================================================

def test_compute_ppo_loss_basic():
    """Test basic PPO loss computation."""
    batch_size = 2
    seq_len = 5

    logprobs = torch.randn(batch_size, seq_len)
    old_logprobs = logprobs.clone()
    advantages = torch.randn(batch_size, seq_len)

    loss, metrics = compute_ppo_loss(logprobs, old_logprobs, advantages)

    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0  # Scalar
    assert "ppo_loss" in metrics
    assert "clip_fraction" in metrics


def test_compute_ppo_loss_clipping():
    """Test PPO clipping behavior."""
    batch_size = 2
    seq_len = 5

    # Create log probs that differ significantly
    old_logprobs = torch.zeros(batch_size, seq_len)
    logprobs = torch.ones(batch_size, seq_len)  # ratio = exp(1) ≈ 2.7

    advantages = torch.ones(batch_size, seq_len)

    loss, metrics = compute_ppo_loss(
        logprobs, old_logprobs, advantages, clip_ratio=0.2
    )

    # Clip fraction should be > 0 since ratio > 1.2
    assert metrics["clip_fraction"] > 0


def test_compute_value_loss():
    """Test value function loss."""
    batch_size = 2
    seq_len = 5

    values = torch.randn(batch_size, seq_len)
    returns = torch.randn(batch_size, seq_len)

    loss, metrics = compute_value_loss(values, returns)

    assert isinstance(loss, torch.Tensor)
    assert "value_loss" in metrics
    assert "explained_variance" in metrics


def test_compute_entropy_bonus():
    """Test entropy computation."""
    batch_size = 2
    seq_len = 5
    vocab_size = 100

    logits = torch.randn(batch_size, seq_len, vocab_size)

    entropy, metrics = compute_entropy_bonus(logits)

    assert isinstance(entropy, torch.Tensor)
    assert entropy.dim() == 0  # Scalar
    assert "entropy" in metrics
    assert entropy.item() > 0  # Entropy should be positive


def test_compute_kl_penalty():
    """Test KL penalty computation."""
    batch_size = 2
    seq_len = 5

    logprobs = torch.randn(batch_size, seq_len)
    ref_logprobs = torch.randn(batch_size, seq_len)

    penalty, metrics = compute_kl_penalty(logprobs, ref_logprobs)

    assert isinstance(penalty, torch.Tensor)
    assert penalty.dim() == 0  # Scalar
    assert "kl_kl" in metrics


def test_compute_ppo_total_loss():
    """Test total PPO loss computation."""
    batch_size = 2
    seq_len = 5
    vocab_size = 100

    policy_logprobs = torch.randn(batch_size, seq_len)
    old_logprobs = policy_logprobs.clone()
    ref_logprobs = torch.randn(batch_size, seq_len)
    values = torch.randn(batch_size, seq_len)
    old_values = values.clone()
    advantages = torch.randn(batch_size, seq_len)
    returns = torch.randn(batch_size, seq_len)
    logits = torch.randn(batch_size, seq_len, vocab_size)

    total_loss, metrics = compute_ppo_total_loss(
        policy_logprobs=policy_logprobs,
        old_logprobs=old_logprobs,
        ref_logprobs=ref_logprobs,
        values=values,
        old_values=old_values,
        advantages=advantages,
        returns=returns,
        logits=logits
    )

    assert isinstance(total_loss, torch.Tensor)
    assert total_loss.dim() == 0  # Scalar

    # Check all expected metrics are present
    assert "total_loss" in metrics
    assert "ppo_loss" in metrics
    assert "value_loss" in metrics
    assert "entropy" in metrics
    assert "kl_kl" in metrics


def test_ppo_loss_with_mask():
    """Test PPO loss with masking."""
    batch_size = 2
    seq_len = 5

    logprobs = torch.randn(batch_size, seq_len)
    old_logprobs = logprobs.clone()
    advantages = torch.randn(batch_size, seq_len)

    # Create mask (last position is padding)
    mask = torch.ones(batch_size, seq_len)
    mask[:, -1] = 0

    loss_with_mask, _ = compute_ppo_loss(
        logprobs, old_logprobs, advantages, mask=mask
    )

    loss_without_mask, _ = compute_ppo_loss(
        logprobs[:, :-1], old_logprobs[:, :-1], advantages[:, :-1]
    )

    # Losses should be similar (masking vs truncating)
    assert abs(loss_with_mask.item() - loss_without_mask.item()) < 0.1


# ============================================================================
# Integration Tests
# ============================================================================

def test_ppo_loss_gradients():
    """Test that PPO loss produces valid gradients."""
    batch_size = 2
    seq_len = 5

    # Create tensors that require gradients
    logprobs = torch.randn(batch_size, seq_len, requires_grad=True)
    old_logprobs = torch.randn(batch_size, seq_len)
    advantages = torch.randn(batch_size, seq_len)

    loss, _ = compute_ppo_loss(logprobs, old_logprobs, advantages)

    # Backward pass
    loss.backward()

    # Check gradients exist
    assert logprobs.grad is not None
    assert not torch.isnan(logprobs.grad).any()


def test_value_network_training_step(small_gpt2_model):
    """Test a single training step with value network."""
    value_net = ValueNetwork(small_gpt2_model)

    optimizer = torch.optim.Adam(value_net.parameters(), lr=1e-3)

    # Create dummy data
    batch_size = 2
    seq_len = 10
    input_ids = torch.randint(0, 100, (batch_size, seq_len))
    returns = torch.randn(batch_size, seq_len)

    # Forward pass
    outputs = value_net(input_ids, return_dict=True)
    values = outputs["values"]

    # Compute loss
    loss, _ = compute_value_loss(values, returns)

    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Check loss is finite
    assert torch.isfinite(loss).all()
