"""
Loss functions for Direct Preference Optimization (DPO).

DPO directly optimizes language models using preference data without
requiring a reward model or reinforcement learning. It's simpler and
more stable than RLHF with PPO.

Paper: "Direct Preference Optimization: Your Language Model is Secretly a Reward Model"
https://arxiv.org/abs/2305.18290
"""

import torch
import torch.nn.functional as F
from typing import Optional, Tuple


def compute_dpo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    reference_chosen_logps: torch.Tensor,
    reference_rejected_logps: torch.Tensor,
    beta: float = 0.1,
    label_smoothing: float = 0.0,
    reduction: str = "mean"
) -> torch.Tensor:
    """
    Compute the DPO loss.

    DPO loss teaches the model to:
    1. Increase probability of chosen responses
    2. Decrease probability of rejected responses
    3. Stay close to the reference model (controlled by beta)

    Educational Note:
        The DPO loss is derived from the Bradley-Terry model of preferences.
        It implicitly defines a reward function as:
            r(x, y) = beta * log(π_θ(y|x) / π_ref(y|x))

        The loss maximizes the likelihood that chosen responses are preferred
        over rejected ones according to this implicit reward.

    Mathematical Formula:
        L_DPO = -log(σ(beta * (log_ratio_chosen - log_ratio_rejected)))

        where:
        - σ is the sigmoid function
        - log_ratio_chosen = log(π_θ(y_chosen|x)) - log(π_ref(y_chosen|x))
        - log_ratio_rejected = log(π_θ(y_rejected|x)) - log(π_ref(y_rejected|x))
        - beta controls how much we penalize deviation from reference model

    Args:
        policy_chosen_logps: Log probabilities of chosen responses from policy model
        policy_rejected_logps: Log probabilities of rejected responses from policy model
        reference_chosen_logps: Log probabilities of chosen responses from reference model
        reference_rejected_logps: Log probabilities of rejected responses from reference model
        beta: Temperature parameter controlling KL penalty (typically 0.1-0.5)
        label_smoothing: Label smoothing factor (0 = no smoothing, 1 = uniform)
        reduction: How to reduce the loss ('mean', 'sum', or 'none')

    Returns:
        DPO loss tensor
    """
    # Compute log ratios: log(π_θ/π_ref) for chosen and rejected
    chosen_log_ratios = policy_chosen_logps - reference_chosen_logps
    rejected_log_ratios = policy_rejected_logps - reference_rejected_logps

    # DPO loss: -log(sigmoid(beta * (chosen_ratio - rejected_ratio)))
    # This can be rewritten using log-sigmoid for numerical stability
    logits = beta * (chosen_log_ratios - rejected_log_ratios)

    if label_smoothing > 0.0:
        # Label smoothing: interpolate between correct label and uniform distribution
        # This makes the model less confident and can improve generalization
        loss = -F.logsigmoid(logits) * (1 - label_smoothing) - F.logsigmoid(-logits) * label_smoothing
    else:
        # Standard DPO loss
        loss = -F.logsigmoid(logits)

    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    else:
        return loss


def compute_dpo_loss_with_metrics(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    reference_chosen_logps: torch.Tensor,
    reference_rejected_logps: torch.Tensor,
    beta: float = 0.1,
    label_smoothing: float = 0.0
) -> dict:
    """
    Compute DPO loss along with additional metrics for monitoring.

    Args:
        policy_chosen_logps: Log probabilities from policy model for chosen
        policy_rejected_logps: Log probabilities from policy model for rejected
        reference_chosen_logps: Log probabilities from reference model for chosen
        reference_rejected_logps: Log probabilities from reference model for rejected
        beta: Temperature parameter
        label_smoothing: Label smoothing factor

    Returns:
        Dictionary containing:
        - loss: The DPO loss
        - accuracy: Fraction of examples where policy prefers chosen over rejected
        - chosen_rewards: Implicit rewards for chosen responses
        - rejected_rewards: Implicit rewards for rejected responses
        - reward_margin: Average difference between chosen and rejected rewards
        - kl_divergence: Approximate KL divergence from reference model
    """
    # Compute loss
    loss = compute_dpo_loss(
        policy_chosen_logps,
        policy_rejected_logps,
        reference_chosen_logps,
        reference_rejected_logps,
        beta,
        label_smoothing,
        reduction="mean"
    )

    # Compute implicit rewards: r(x,y) = beta * log(π_θ(y|x) / π_ref(y|x))
    chosen_rewards = beta * (policy_chosen_logps - reference_chosen_logps)
    rejected_rewards = beta * (policy_rejected_logps - reference_rejected_logps)

    # Reward margin: how much better chosen is than rejected
    reward_margin = (chosen_rewards - rejected_rewards).mean()

    # Accuracy: fraction where policy correctly prefers chosen over rejected
    accuracy = ((chosen_rewards > rejected_rewards).float().mean())

    # Approximate KL divergence from reference model
    # KL(π_θ || π_ref) ≈ log(π_θ) - log(π_ref)
    kl_divergence = ((policy_chosen_logps - reference_chosen_logps).mean() +
                     (policy_rejected_logps - reference_rejected_logps).mean()) / 2

    return {
        "loss": loss,
        "accuracy": accuracy,
        "chosen_rewards": chosen_rewards.mean(),
        "rejected_rewards": rejected_rewards.mean(),
        "reward_margin": reward_margin,
        "kl_divergence": kl_divergence,
    }


def get_batch_logps(
    logits: torch.Tensor,
    labels: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    return_per_token: bool = False
) -> torch.Tensor:
    """
    Compute log probabilities for a batch of sequences.

    This is a helper function for computing the log probabilities
    of target sequences under the model's distribution.

    Args:
        logits: Model output logits of shape (batch_size, seq_len, vocab_size)
        labels: Target token IDs of shape (batch_size, seq_len)
        attention_mask: Optional attention mask (1 for real tokens, 0 for padding)
        return_per_token: If True, return per-token log probs instead of sum

    Returns:
        Log probabilities of shape (batch_size,) or (batch_size, seq_len)

    Educational Note:
        We compute log P(y|x) = sum_{t} log P(y_t | y_{<t}, x)
        by gathering the log probabilities assigned to the actual tokens
        in the sequence and summing them (or averaging for per-token).
    """
    # Shift logits and labels for next-token prediction
    # Model predicts token i+1 from tokens 0...i
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()

    # Compute log probabilities
    log_probs = F.log_softmax(shift_logits, dim=-1)

    # Gather log probs of the actual tokens
    # Shape: (batch_size, seq_len-1)
    token_log_probs = torch.gather(
        log_probs,
        dim=2,
        index=shift_labels.unsqueeze(2)
    ).squeeze(2)

    # Apply attention mask if provided
    if attention_mask is not None:
        shift_mask = attention_mask[:, 1:].contiguous()
        token_log_probs = token_log_probs * shift_mask

        if return_per_token:
            # Return average log prob per real token
            return token_log_probs.sum(dim=1) / shift_mask.sum(dim=1)
        else:
            # Return total log prob (sum over real tokens)
            return token_log_probs.sum(dim=1)
    else:
        if return_per_token:
            return token_log_probs.mean(dim=1)
        else:
            return token_log_probs.sum(dim=1)


def compute_dpo_loss_from_logits(
    policy_chosen_logits: torch.Tensor,
    policy_rejected_logits: torch.Tensor,
    reference_chosen_logits: torch.Tensor,
    reference_rejected_logits: torch.Tensor,
    chosen_labels: torch.Tensor,
    rejected_labels: torch.Tensor,
    chosen_attention_mask: Optional[torch.Tensor] = None,
    rejected_attention_mask: Optional[torch.Tensor] = None,
    beta: float = 0.1,
    label_smoothing: float = 0.0
) -> Tuple[torch.Tensor, dict]:
    """
    Compute DPO loss directly from model logits.

    This is a convenience function that combines logit-to-logprob conversion
    with DPO loss computation.

    Args:
        policy_chosen_logits: Policy model logits for chosen responses
        policy_rejected_logits: Policy model logits for rejected responses
        reference_chosen_logits: Reference model logits for chosen responses
        reference_rejected_logits: Reference model logits for rejected responses
        chosen_labels: Token IDs for chosen responses
        rejected_labels: Token IDs for rejected responses
        chosen_attention_mask: Attention mask for chosen responses
        rejected_attention_mask: Attention mask for rejected responses
        beta: Temperature parameter
        label_smoothing: Label smoothing factor

    Returns:
        Tuple of (loss, metrics_dict)
    """
    # Compute log probabilities from logits
    policy_chosen_logps = get_batch_logps(
        policy_chosen_logits,
        chosen_labels,
        chosen_attention_mask
    )
    policy_rejected_logps = get_batch_logps(
        policy_rejected_logits,
        rejected_labels,
        rejected_attention_mask
    )
    reference_chosen_logps = get_batch_logps(
        reference_chosen_logits,
        chosen_labels,
        chosen_attention_mask
    )
    reference_rejected_logps = get_batch_logps(
        reference_rejected_logits,
        rejected_labels,
        rejected_attention_mask
    )

    # Compute loss with metrics
    metrics = compute_dpo_loss_with_metrics(
        policy_chosen_logps,
        policy_rejected_logps,
        reference_chosen_logps,
        reference_rejected_logps,
        beta,
        label_smoothing
    )

    return metrics["loss"], metrics


class DPOLoss(torch.nn.Module):
    """
    DPO Loss as a PyTorch module.

    This wraps the DPO loss computation in a standard PyTorch nn.Module
    for easier integration with training frameworks.
    """

    def __init__(self, beta: float = 0.1, label_smoothing: float = 0.0):
        """
        Args:
            beta: Temperature parameter (default: 0.1)
            label_smoothing: Label smoothing factor (default: 0.0)
        """
        super().__init__()
        self.beta = beta
        self.label_smoothing = label_smoothing

    def forward(
        self,
        policy_chosen_logps: torch.Tensor,
        policy_rejected_logps: torch.Tensor,
        reference_chosen_logps: torch.Tensor,
        reference_rejected_logps: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute DPO loss.

        Args:
            policy_chosen_logps: Log probs from policy for chosen
            policy_rejected_logps: Log probs from policy for rejected
            reference_chosen_logps: Log probs from reference for chosen
            reference_rejected_logps: Log probs from reference for rejected

        Returns:
            Loss tensor
        """
        return compute_dpo_loss(
            policy_chosen_logps,
            policy_rejected_logps,
            reference_chosen_logps,
            reference_rejected_logps,
            self.beta,
            self.label_smoothing
        )
