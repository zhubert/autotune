"""
Tests for SFT loss functions.
"""

import pytest
import torch

from src.auto_bot_tuner.sft.loss import (
    compute_sft_loss,
    compute_sft_loss_with_metrics,
    compute_token_level_loss,
    SFTLoss,
    validate_loss_inputs
)


@pytest.fixture
def sample_logits():
    """Create sample logits tensor."""
    # Shape: (batch_size=2, seq_len=10, vocab_size=100)
    return torch.randn(2, 10, 100)


@pytest.fixture
def sample_labels():
    """Create sample labels tensor with masking."""
    # Shape: (batch_size=2, seq_len=10)
    labels = torch.randint(0, 100, (2, 10))
    # Mask first 3 tokens (prompt)
    labels[:, :3] = -100
    return labels


class TestComputeSFTLoss:
    """Test compute_sft_loss function."""

    def test_basic_loss_computation(self, sample_logits, sample_labels):
        """Test basic loss computation."""
        loss = compute_sft_loss(sample_logits, sample_labels)

        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0  # Scalar
        assert loss.item() >= 0  # Loss should be non-negative

    def test_reduction_mean(self, sample_logits, sample_labels):
        """Test reduction='mean'."""
        loss = compute_sft_loss(sample_logits, sample_labels, reduction="mean")
        assert loss.dim() == 0

    def test_reduction_sum(self, sample_logits, sample_labels):
        """Test reduction='sum'."""
        loss = compute_sft_loss(sample_logits, sample_labels, reduction="sum")
        assert loss.dim() == 0

        # Sum should be larger than mean
        loss_mean = compute_sft_loss(sample_logits, sample_labels, reduction="mean")
        assert loss.item() > loss_mean.item()

    def test_reduction_none(self, sample_logits, sample_labels):
        """Test reduction='none'."""
        loss = compute_sft_loss(sample_logits, sample_labels, reduction="none")
        # Should return per-sample losses
        assert loss.dim() > 0

    def test_masked_tokens_ignored(self):
        """Test that masked tokens (-100) don't contribute to loss."""
        vocab_size = 10
        logits = torch.zeros(1, 5, vocab_size)
        # Set high probability for token 0
        logits[0, :, 0] = 10.0

        # Labels with some masked and some unmasked
        labels = torch.zeros(1, 5, dtype=torch.long)
        # Mask first 3 tokens
        labels[0, :3] = -100

        # Compute loss - should only use non-masked tokens
        loss = compute_sft_loss(logits, labels)

        # Loss should be very low since model predicts correctly on non-masked tokens
        assert loss.item() < 0.1


class TestComputeSFTLossWithMetrics:
    """Test compute_sft_loss_with_metrics function."""

    def test_returns_all_metrics(self, sample_logits, sample_labels):
        """Test that all metrics are returned."""
        metrics = compute_sft_loss_with_metrics(sample_logits, sample_labels)

        assert "loss" in metrics
        assert "perplexity" in metrics
        assert "accuracy" in metrics
        assert "num_tokens" in metrics

    def test_perplexity_calculation(self, sample_logits, sample_labels):
        """Test perplexity calculation."""
        metrics = compute_sft_loss_with_metrics(sample_logits, sample_labels)

        # Perplexity should be exp(loss)
        expected_perplexity = torch.exp(metrics["loss"])
        assert torch.isclose(metrics["perplexity"], expected_perplexity)

    def test_accuracy_range(self, sample_logits, sample_labels):
        """Test that accuracy is between 0 and 1."""
        metrics = compute_sft_loss_with_metrics(sample_logits, sample_labels)

        accuracy = metrics["accuracy"].item()
        assert 0.0 <= accuracy <= 1.0

    def test_num_tokens_count(self, sample_logits, sample_labels):
        """Test that num_tokens counts non-masked tokens."""
        metrics = compute_sft_loss_with_metrics(sample_logits, sample_labels)

        # Count non-masked tokens (excluding -100 and shifted by 1)
        expected_count = ((sample_labels[:, 1:] != -100).sum()).item()
        assert metrics["num_tokens"].item() == expected_count


class TestComputeTokenLevelLoss:
    """Test compute_token_level_loss function."""

    def test_token_level_shape(self, sample_logits, sample_labels):
        """Test that output shape is correct."""
        token_loss = compute_token_level_loss(sample_logits, sample_labels)

        # Shape should be (batch_size, seq_len - 1) due to shifting
        assert token_loss.shape[0] == sample_logits.shape[0]
        assert token_loss.shape[1] == sample_logits.shape[1] - 1

    def test_masked_tokens_zero_loss(self):
        """Test that masked tokens have zero loss.

        Note: Token-level loss shifts by 1 position, so masking position 1
        in labels affects position 0 in the output loss.
        """
        vocab_size = 10
        logits = torch.randn(1, 5, vocab_size)
        labels = torch.randint(0, vocab_size, (1, 5))

        # Mask second token (index 1) - this will affect first position in shifted loss
        labels[0, 1] = -100

        token_loss = compute_token_level_loss(logits, labels)

        # First token in output (corresponding to labels[1]) should have zero loss
        assert token_loss[0, 0].item() == 0.0


class TestSFTLossModule:
    """Test SFTLoss module."""

    def test_module_forward(self, sample_logits, sample_labels):
        """Test forward pass of module."""
        loss_module = SFTLoss()
        loss = loss_module(sample_logits, sample_labels)

        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0

    def test_module_reduction(self, sample_logits, sample_labels):
        """Test different reduction modes."""
        loss_module_mean = SFTLoss(reduction="mean")
        loss_module_sum = SFTLoss(reduction="sum")

        loss_mean = loss_module_mean(sample_logits, sample_labels)
        loss_sum = loss_module_sum(sample_logits, sample_labels)

        # Sum should be larger
        assert loss_sum.item() > loss_mean.item()


class TestValidateLossInputs:
    """Test validate_loss_inputs function."""

    def test_valid_inputs(self, sample_logits, sample_labels):
        """Test that valid inputs pass validation."""
        # Should not raise
        validate_loss_inputs(sample_logits, sample_labels)

    def test_wrong_logits_dims(self, sample_labels):
        """Test that wrong logits dimensions are caught."""
        wrong_logits = torch.randn(2, 10)  # Missing vocab dimension

        with pytest.raises(ValueError, match="3 dimensions"):
            validate_loss_inputs(wrong_logits, sample_labels)

    def test_wrong_labels_dims(self, sample_logits):
        """Test that wrong labels dimensions are caught."""
        wrong_labels = torch.randint(0, 100, (2,))  # Missing seq dimension

        with pytest.raises(ValueError, match="2 dimensions"):
            validate_loss_inputs(sample_logits, wrong_labels)

    def test_batch_size_mismatch(self):
        """Test that batch size mismatch is caught."""
        logits = torch.randn(2, 10, 100)
        labels = torch.randint(0, 100, (3, 10))

        with pytest.raises(ValueError, match="Batch size mismatch"):
            validate_loss_inputs(logits, labels)

    def test_seq_length_mismatch(self):
        """Test that sequence length mismatch is caught."""
        logits = torch.randn(2, 10, 100)
        labels = torch.randint(0, 100, (2, 8))

        with pytest.raises(ValueError, match="Sequence length mismatch"):
            validate_loss_inputs(logits, labels)

    def test_all_masked_labels(self):
        """Test that all-masked labels are caught."""
        logits = torch.randn(2, 10, 100)
        labels = torch.full((2, 10), -100)

        with pytest.raises(ValueError, match="All tokens are masked"):
            validate_loss_inputs(logits, labels)


class TestLossProperties:
    """Test mathematical properties of the loss."""

    def test_perfect_prediction_low_loss(self):
        """Test that perfect predictions have low loss."""
        vocab_size = 10
        seq_len = 5
        batch_size = 1

        # Create logits with very high probability for correct token
        logits = torch.zeros(batch_size, seq_len, vocab_size)
        labels = torch.zeros(batch_size, seq_len, dtype=torch.long)

        # Set very high logit for token 0
        logits[:, :, 0] = 100.0

        loss = compute_sft_loss(logits, labels)

        # Loss should be very close to 0
        assert loss.item() < 0.01

    def test_random_prediction_high_loss(self):
        """Test that random predictions have higher loss."""
        vocab_size = 100
        seq_len = 5
        batch_size = 1

        # Random logits
        logits = torch.randn(batch_size, seq_len, vocab_size)
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))

        loss = compute_sft_loss(logits, labels)

        # Loss should be reasonably high (around log(vocab_size))
        expected_loss = torch.log(torch.tensor(vocab_size, dtype=torch.float))
        # Allow some variance but should be in reasonable range
        assert loss.item() > 1.0  # At least above 1
        assert loss.item() < expected_loss.item() * 2  # Not too high
