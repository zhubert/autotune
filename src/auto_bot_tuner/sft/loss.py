"""
Loss functions for Supervised Fine-Tuning (SFT).

This module implements the cross-entropy loss with proper masking
for instruction fine-tuning.
"""

import torch
import torch.nn.functional as F
from typing import Optional


def compute_sft_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    reduction: str = "mean"
) -> torch.Tensor:
    """
    Compute masked cross-entropy loss for SFT.

    In instruction tuning, we only want to compute loss on the model's
    generated responses, not on the input prompts. Labels with value -100
    are ignored (these correspond to prompt tokens).

    Args:
        logits: Model output logits of shape (batch_size, seq_len, vocab_size)
        labels: Target token IDs of shape (batch_size, seq_len)
                Tokens to ignore have value -100
        reduction: How to reduce the loss ('mean', 'sum', or 'none')

    Returns:
        Scalar loss tensor (or per-sample losses if reduction='none')

    Educational Note:
        Cross-entropy loss measures how well the model's predicted probability
        distribution matches the target distribution. For language modeling,
        we're teaching the model to assign high probability to the correct
        next token and low probability to incorrect tokens.

        Formula: L = -log(P(correct_token))

        The loss is:
        - 0 when model assigns probability 1.0 to correct token (perfect)
        - ∞ when model assigns probability 0 to correct token (terrible)
    """
    # Shift logits and labels for next-token prediction
    # The model predicts token i+1 from tokens 0...i
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    # Flatten for cross-entropy computation
    # Shape: (batch_size * seq_len, vocab_size) and (batch_size * seq_len)
    shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    shift_labels = shift_labels.view(-1)

    # Compute cross-entropy loss
    # ignore_index=-100 means tokens with label -100 don't contribute to loss
    loss = F.cross_entropy(
        shift_logits,
        shift_labels,
        ignore_index=-100,
        reduction=reduction
    )

    return loss


def compute_sft_loss_with_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor
) -> dict[str, torch.Tensor]:
    """
    Compute SFT loss along with additional metrics for monitoring.

    Args:
        logits: Model output logits of shape (batch_size, seq_len, vocab_size)
        labels: Target token IDs of shape (batch_size, seq_len)

    Returns:
        Dictionary containing:
        - loss: The main training loss
        - perplexity: Perplexity score (exp(loss))
        - accuracy: Token-level accuracy on non-masked tokens
        - num_tokens: Number of tokens contributing to loss
    """
    # Compute main loss
    loss = compute_sft_loss(logits, labels, reduction="mean")

    # Compute perplexity
    # Perplexity is exp(loss) - measures how "surprised" the model is
    # Lower perplexity = better model
    perplexity = torch.exp(loss)

    # Compute accuracy
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    # Get predictions
    predictions = shift_logits.argmax(dim=-1)

    # Mask out ignored tokens (label = -100)
    mask = (shift_labels != -100)
    correct = (predictions == shift_labels) & mask

    # Compute accuracy only on non-masked tokens
    accuracy = correct.sum().float() / mask.sum().float() if mask.sum() > 0 else torch.tensor(0.0)
    num_tokens = mask.sum()

    return {
        "loss": loss,
        "perplexity": perplexity,
        "accuracy": accuracy,
        "num_tokens": num_tokens
    }


def compute_token_level_loss(
    logits: torch.Tensor,
    labels: torch.Tensor
) -> torch.Tensor:
    """
    Compute per-token loss values (unreduced).

    Useful for identifying which tokens are hard for the model to predict.

    Args:
        logits: Model output logits of shape (batch_size, seq_len, vocab_size)
        labels: Target token IDs of shape (batch_size, seq_len)

    Returns:
        Per-token loss tensor of shape (batch_size, seq_len)
        Masked tokens have loss value of 0
    """
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    batch_size, seq_len, vocab_size = shift_logits.shape

    # Compute unreduced loss
    flat_logits = shift_logits.view(-1, vocab_size)
    flat_labels = shift_labels.view(-1)

    flat_loss = F.cross_entropy(
        flat_logits,
        flat_labels,
        ignore_index=-100,
        reduction='none'
    )

    # Reshape back to (batch_size, seq_len)
    token_loss = flat_loss.view(batch_size, seq_len)

    return token_loss


class SFTLoss(torch.nn.Module):
    """
    Wrapper class for SFT loss as a PyTorch module.

    This can be useful for integrating with training frameworks that
    expect loss functions to be nn.Modules.
    """

    def __init__(self, reduction: str = "mean"):
        """
        Args:
            reduction: How to reduce the loss ('mean', 'sum', or 'none')
        """
        super().__init__()
        self.reduction = reduction

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the loss.

        Args:
            logits: Model output logits
            labels: Target labels with -100 for ignored tokens

        Returns:
            Loss tensor
        """
        return compute_sft_loss(logits, labels, reduction=self.reduction)


def validate_loss_inputs(logits: torch.Tensor, labels: torch.Tensor) -> None:
    """
    Validate that loss inputs have correct shapes and values.

    Args:
        logits: Model output logits
        labels: Target labels

    Raises:
        ValueError: If inputs are invalid
    """
    if logits.dim() != 3:
        raise ValueError(f"Expected logits to have 3 dimensions (batch, seq, vocab), got {logits.dim()}")

    if labels.dim() != 2:
        raise ValueError(f"Expected labels to have 2 dimensions (batch, seq), got {labels.dim()}")

    if logits.shape[0] != labels.shape[0]:
        raise ValueError(f"Batch size mismatch: logits {logits.shape[0]}, labels {labels.shape[0]}")

    if logits.shape[1] != labels.shape[1]:
        raise ValueError(f"Sequence length mismatch: logits {logits.shape[1]}, labels {labels.shape[1]}")

    # Check for non-masked tokens
    num_valid_tokens = (labels != -100).sum().item()
    if num_valid_tokens == 0:
        raise ValueError("All tokens are masked (label = -100), cannot compute loss")
