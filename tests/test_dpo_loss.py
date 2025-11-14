"""
Tests for DPO loss functions.
"""

import pytest
import torch

from src.auto_bot_tuner.dpo.loss import (
    compute_dpo_loss,
    compute_dpo_loss_with_metrics,
    get_batch_logps,
    DPOLoss
)


@pytest.fixture
def sample_logps():
    """Create sample log probabilities."""
    batch_size = 4
    # Simulate log probs for chosen and rejected from policy and reference
    return {
        "policy_chosen": torch.randn(batch_size) - 2.0,  # Around -2
        "policy_rejected": torch.randn(batch_size) - 3.0,  # Around -3 (worse)
        "reference_chosen": torch.randn(batch_size) - 2.5,
        "reference_rejected": torch.randn(batch_size) - 2.5,
    }


class TestComputeDPOLoss:
    """Test compute_dpo_loss function."""

    def test_basic_loss_computation(self, sample_logps):
        """Test basic DPO loss computation."""
        loss = compute_dpo_loss(
            sample_logps["policy_chosen"],
            sample_logps["policy_rejected"],
            sample_logps["reference_chosen"],
            sample_logps["reference_rejected"],
            beta=0.1
        )

        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0  # Scalar
        assert loss.item() >= 0  # Loss should be non-negative

    def test_perfect_preferences(self):
        """Test loss when policy perfectly prefers chosen over rejected."""
        batch_size = 4

        # Policy strongly prefers chosen
        policy_chosen = torch.zeros(batch_size)  # log prob = 0
        policy_rejected = torch.full((batch_size,), -10.0)  # log prob = -10

        # Reference is neutral
        reference_chosen = torch.full((batch_size,), -5.0)
        reference_rejected = torch.full((batch_size,), -5.0)

        loss = compute_dpo_loss(
            policy_chosen,
            policy_rejected,
            reference_chosen,
            reference_rejected,
            beta=0.1
        )

        # Loss should be relatively low when policy is correct
        # (Not zero because of reference model divergence)
        assert loss.item() < 0.5

    def test_wrong_preferences(self):
        """Test loss when policy prefers rejected over chosen."""
        batch_size = 4

        # Policy prefers rejected (wrong!)
        policy_chosen = torch.full((batch_size,), -10.0)
        policy_rejected = torch.zeros(batch_size)

        # Reference is neutral
        reference_chosen = torch.full((batch_size,), -5.0)
        reference_rejected = torch.full((batch_size,), -5.0)

        loss = compute_dpo_loss(
            policy_chosen,
            policy_rejected,
            reference_chosen,
            reference_rejected,
            beta=0.1
        )

        # Loss should be high when policy gets it wrong
        assert loss.item() > 1.0

    def test_beta_effect(self, sample_logps):
        """Test that beta parameter affects loss magnitude."""
        loss_low_beta = compute_dpo_loss(
            sample_logps["policy_chosen"],
            sample_logps["policy_rejected"],
            sample_logps["reference_chosen"],
            sample_logps["reference_rejected"],
            beta=0.1
        )

        loss_high_beta = compute_dpo_loss(
            sample_logps["policy_chosen"],
            sample_logps["policy_rejected"],
            sample_logps["reference_chosen"],
            sample_logps["reference_rejected"],
            beta=0.5
        )

        # Different beta should give different losses
        assert not torch.isclose(loss_low_beta, loss_high_beta)

    def test_label_smoothing(self, sample_logps):
        """Test label smoothing effect."""
        loss_no_smoothing = compute_dpo_loss(
            sample_logps["policy_chosen"],
            sample_logps["policy_rejected"],
            sample_logps["reference_chosen"],
            sample_logps["reference_rejected"],
            beta=0.1,
            label_smoothing=0.0
        )

        loss_with_smoothing = compute_dpo_loss(
            sample_logps["policy_chosen"],
            sample_logps["policy_rejected"],
            sample_logps["reference_chosen"],
            sample_logps["reference_rejected"],
            beta=0.1,
            label_smoothing=0.1
        )

        # Label smoothing should change the loss
        assert not torch.isclose(loss_no_smoothing, loss_with_smoothing)

    def test_reduction_modes(self, sample_logps):
        """Test different reduction modes."""
        loss_mean = compute_dpo_loss(
            sample_logps["policy_chosen"],
            sample_logps["policy_rejected"],
            sample_logps["reference_chosen"],
            sample_logps["reference_rejected"],
            reduction="mean"
        )

        loss_sum = compute_dpo_loss(
            sample_logps["policy_chosen"],
            sample_logps["policy_rejected"],
            sample_logps["reference_chosen"],
            sample_logps["reference_rejected"],
            reduction="sum"
        )

        loss_none = compute_dpo_loss(
            sample_logps["policy_chosen"],
            sample_logps["policy_rejected"],
            sample_logps["reference_chosen"],
            sample_logps["reference_rejected"],
            reduction="none"
        )

        assert loss_mean.dim() == 0
        assert loss_sum.dim() == 0
        assert loss_none.dim() == 1
        assert loss_sum.item() > loss_mean.item()


class TestComputeDPOLossWithMetrics:
    """Test compute_dpo_loss_with_metrics function."""

    def test_returns_all_metrics(self, sample_logps):
        """Test that all metrics are returned."""
        metrics = compute_dpo_loss_with_metrics(
            sample_logps["policy_chosen"],
            sample_logps["policy_rejected"],
            sample_logps["reference_chosen"],
            sample_logps["reference_rejected"]
        )

        assert "loss" in metrics
        assert "accuracy" in metrics
        assert "chosen_rewards" in metrics
        assert "rejected_rewards" in metrics
        assert "reward_margin" in metrics
        assert "kl_divergence" in metrics

    def test_accuracy_range(self, sample_logps):
        """Test that accuracy is between 0 and 1."""
        metrics = compute_dpo_loss_with_metrics(
            sample_logps["policy_chosen"],
            sample_logps["policy_rejected"],
            sample_logps["reference_chosen"],
            sample_logps["reference_rejected"]
        )

        accuracy = metrics["accuracy"].item()
        assert 0.0 <= accuracy <= 1.0

    def test_perfect_accuracy(self):
        """Test that perfect preferences give 100% accuracy."""
        batch_size = 4

        # Policy always prefers chosen
        policy_chosen = torch.zeros(batch_size)
        policy_rejected = torch.full((batch_size,), -10.0)

        reference_chosen = torch.full((batch_size,), -5.0)
        reference_rejected = torch.full((batch_size,), -5.0)

        metrics = compute_dpo_loss_with_metrics(
            policy_chosen,
            policy_rejected,
            reference_chosen,
            reference_rejected
        )

        assert metrics["accuracy"].item() == 1.0

    def test_reward_margin_positive(self):
        """Test that reward margin is positive when policy is correct."""
        batch_size = 4

        # Policy prefers chosen
        policy_chosen = torch.zeros(batch_size)
        policy_rejected = torch.full((batch_size,), -5.0)

        reference_chosen = torch.full((batch_size,), -2.5)
        reference_rejected = torch.full((batch_size,), -2.5)

        metrics = compute_dpo_loss_with_metrics(
            policy_chosen,
            policy_rejected,
            reference_chosen,
            reference_rejected,
            beta=0.1
        )

        # Reward margin should be positive (chosen better than rejected)
        assert metrics["reward_margin"].item() > 0


class TestGetBatchLogps:
    """Test get_batch_logps function."""

    def test_basic_logp_computation(self):
        """Test basic log probability computation."""
        batch_size = 2
        seq_len = 5
        vocab_size = 10

        logits = torch.randn(batch_size, seq_len, vocab_size)
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))

        logps = get_batch_logps(logits, labels)

        assert logps.shape == (batch_size,)
        assert torch.all(torch.isfinite(logps))  # No NaN or Inf

    def test_with_attention_mask(self):
        """Test log probability computation with attention mask."""
        batch_size = 2
        seq_len = 5
        vocab_size = 10

        logits = torch.randn(batch_size, seq_len, vocab_size)
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)
        # Mask out last 2 tokens for second sample
        attention_mask[1, -2:] = 0

        logps = get_batch_logps(logits, labels, attention_mask)

        assert logps.shape == (batch_size,)
        # Second sample should have different (likely lower) log prob due to masking
        assert logps[0].item() != logps[1].item()

    def test_per_token_mode(self):
        """Test per-token log probability mode."""
        batch_size = 2
        seq_len = 5
        vocab_size = 10

        logits = torch.randn(batch_size, seq_len, vocab_size)
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))

        logps_total = get_batch_logps(logits, labels, return_per_token=False)
        logps_per_token = get_batch_logps(logits, labels, return_per_token=True)

        assert logps_total.shape == (batch_size,)
        assert logps_per_token.shape == (batch_size,)

        # Per-token should be averaged, so magnitude should be smaller
        assert torch.all(torch.abs(logps_per_token) < torch.abs(logps_total))


class TestDPOLossModule:
    """Test DPOLoss module."""

    def test_module_forward(self, sample_logps):
        """Test forward pass of module."""
        loss_module = DPOLoss(beta=0.1)
        loss = loss_module(
            sample_logps["policy_chosen"],
            sample_logps["policy_rejected"],
            sample_logps["reference_chosen"],
            sample_logps["reference_rejected"]
        )

        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0

    def test_module_beta_parameter(self, sample_logps):
        """Test that module respects beta parameter."""
        loss_module_low = DPOLoss(beta=0.1)
        loss_module_high = DPOLoss(beta=0.5)

        loss_low = loss_module_low(
            sample_logps["policy_chosen"],
            sample_logps["policy_rejected"],
            sample_logps["reference_chosen"],
            sample_logps["reference_rejected"]
        )

        loss_high = loss_module_high(
            sample_logps["policy_chosen"],
            sample_logps["policy_rejected"],
            sample_logps["reference_chosen"],
            sample_logps["reference_rejected"]
        )

        assert not torch.isclose(loss_low, loss_high)


class TestDPOLossProperties:
    """Test mathematical properties of DPO loss."""

    def test_loss_decreases_with_better_policy(self):
        """Test that loss decreases as policy improves."""
        batch_size = 4

        # Reference is neutral
        reference_chosen = torch.full((batch_size,), -5.0)
        reference_rejected = torch.full((batch_size,), -5.0)

        # Bad policy: barely distinguishes
        bad_policy_chosen = torch.full((batch_size,), -4.9)
        bad_policy_rejected = torch.full((batch_size,), -5.1)

        # Good policy: strongly distinguishes
        good_policy_chosen = torch.full((batch_size,), -2.0)
        good_policy_rejected = torch.full((batch_size,), -8.0)

        loss_bad = compute_dpo_loss(
            bad_policy_chosen,
            bad_policy_rejected,
            reference_chosen,
            reference_rejected,
            beta=0.1
        )

        loss_good = compute_dpo_loss(
            good_policy_chosen,
            good_policy_rejected,
            reference_chosen,
            reference_rejected,
            beta=0.1
        )

        # Better policy should have lower loss
        assert loss_good.item() < loss_bad.item()

    def test_symmetric_reference_neutral(self):
        """Test that equal reference logps doesn't bias the loss."""
        batch_size = 4

        policy_chosen = torch.randn(batch_size)
        policy_rejected = policy_chosen - 1.0  # Policy prefers chosen

        # Reference is exactly neutral
        reference_logps = torch.full((batch_size,), -5.0)

        loss = compute_dpo_loss(
            policy_chosen,
            policy_rejected,
            reference_logps,
            reference_logps,
            beta=0.1
        )

        # Should have reasonable loss (not extreme)
        assert 0.0 < loss.item() < 2.0
