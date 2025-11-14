"""
Loss functions for Proximal Policy Optimization (PPO).

PPO uses a clipped surrogate objective to prevent large policy updates
while still allowing meaningful learning.
"""

import torch
import torch.nn.functional as F
from typing import Optional, Dict, Any


def compute_ppo_loss(
    logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    clip_ratio: float = 0.2
) -> tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute PPO clipped surrogate loss.

    PPO's key insight is to clip the policy ratio to prevent overly large
    updates. This maintains training stability while still allowing learning.

    Args:
        logprobs: Log probabilities under current policy (batch_size, seq_len)
        old_logprobs: Log probabilities under old policy (batch_size, seq_len)
        advantages: Advantage estimates (batch_size, seq_len)
        mask: Optional mask for valid positions (batch_size, seq_len)
        clip_ratio: Clipping parameter epsilon (typically 0.2)

    Returns:
        loss: PPO loss (scalar)
        metrics: Dictionary of metrics for logging

    Educational Note:
        PPO loss is:
        L^{CLIP} = -E[min(r_t * A_t, clip(r_t, 1-ε, 1+ε) * A_t)]

        where r_t = π_θ(a_t|s_t) / π_θ_old(a_t|s_t) is the probability ratio

        The clipping prevents the new policy from deviating too much from
        the old policy, which stabilizes training.
    """
    # Compute probability ratio: r = π_new / π_old
    # In log space: log(r) = log π_new - log π_old
    log_ratio = logprobs - old_logprobs
    ratio = torch.exp(log_ratio)

    # PPO clipped objective
    # Unclipped objective: ratio * advantage
    unclipped_objective = ratio * advantages

    # Clipped objective: clip(ratio, 1-ε, 1+ε) * advantage
    clipped_ratio = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio)
    clipped_objective = clipped_ratio * advantages

    # Take minimum (conservative update)
    # Shape: (batch_size, seq_len)
    per_token_loss = -torch.min(unclipped_objective, clipped_objective)

    # Apply mask if provided
    if mask is not None:
        per_token_loss = per_token_loss * mask
        num_tokens = mask.sum()
    else:
        num_tokens = per_token_loss.numel()

    # Average over all tokens
    loss = per_token_loss.sum() / (num_tokens + 1e-8)

    # Compute metrics
    with torch.no_grad():
        # Fraction of times clipping was active
        clipped = ((ratio < 1 - clip_ratio) | (ratio > 1 + clip_ratio)).float()
        if mask is not None:
            clipped = clipped * mask
        clip_fraction = clipped.sum() / (num_tokens + 1e-8)

        # Average ratio
        if mask is not None:
            avg_ratio = (ratio * mask).sum() / (num_tokens + 1e-8)
        else:
            avg_ratio = ratio.mean()

        # KL divergence approximation
        # KL(old || new) ≈ (log_ratio)²/2 for small divergences
        # or exact: KL = old_probs * (old_log_probs - new_log_probs)
        approx_kl = ((ratio - 1) - log_ratio).mean()

        metrics = {
            "ppo_loss": loss.item(),
            "clip_fraction": clip_fraction.item(),
            "avg_ratio": avg_ratio.item(),
            "approx_kl": approx_kl.item(),
        }

    return loss, metrics


def compute_value_loss(
    values: torch.Tensor,
    returns: torch.Tensor,
    old_values: Optional[torch.Tensor] = None,
    mask: Optional[torch.Tensor] = None,
    clip_value: bool = True,
    clip_ratio: float = 0.2
) -> tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute value function loss.

    Trains the value network to predict returns. Optionally uses clipping
    similar to the policy loss.

    Args:
        values: Value predictions from current network (batch_size, seq_len)
        returns: Target returns (batch_size, seq_len)
        old_values: Value predictions from old network (for clipping)
        mask: Optional mask for valid positions
        clip_value: If True, use value clipping like PPO-Clip
        clip_ratio: Clipping parameter

    Returns:
        loss: Value loss (scalar)
        metrics: Dictionary of metrics

    Educational Note:
        Standard value loss is MSE: (V(s) - R)²

        With clipping (as in PPO-Clip), we clip the value updates:
        L^V = max((V - R)², (V_old + clip(V - V_old, -ε, ε) - R)²)

        This prevents the value function from changing too quickly.
    """
    if clip_value and old_values is not None:
        # Compute clipped value loss (like PPO paper)
        value_pred_clipped = old_values + torch.clamp(
            values - old_values,
            -clip_ratio,
            clip_ratio
        )

        # Unclipped loss
        value_loss_unclipped = (values - returns) ** 2

        # Clipped loss
        value_loss_clipped = (value_pred_clipped - returns) ** 2

        # Take maximum (conservative update)
        per_token_loss = torch.max(value_loss_unclipped, value_loss_clipped)
    else:
        # Standard MSE loss
        per_token_loss = (values - returns) ** 2

    # Apply mask if provided
    if mask is not None:
        per_token_loss = per_token_loss * mask
        num_tokens = mask.sum()
    else:
        num_tokens = per_token_loss.numel()

    # Average over all tokens
    loss = per_token_loss.sum() / (num_tokens + 1e-8)

    # Compute metrics
    with torch.no_grad():
        if mask is not None:
            explained_variance = 1 - (
                ((returns - values) ** 2 * mask).sum() /
                (returns.var() * num_tokens + 1e-8)
            )
        else:
            explained_variance = 1 - (returns - values).var() / (returns.var() + 1e-8)

        metrics = {
            "value_loss": loss.item(),
            "explained_variance": explained_variance.item(),
        }

    return loss, metrics


def compute_entropy_bonus(
    logits: torch.Tensor,
    mask: Optional[torch.Tensor] = None
) -> tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute entropy bonus for exploration.

    Adding entropy to the loss encourages the policy to explore more,
    preventing premature convergence to a deterministic policy.

    Args:
        logits: Logits from the policy (batch_size, seq_len, vocab_size)
        mask: Optional mask for valid positions (batch_size, seq_len)

    Returns:
        entropy: Mean entropy (scalar, to be added to loss with negative coefficient)
        metrics: Dictionary of metrics

    Educational Note:
        Entropy H = -Σ p(a) log p(a)

        Higher entropy = more randomness = more exploration
        Lower entropy = more deterministic = more exploitation

        We typically add -c * H to the loss (c > 0), which means
        maximizing entropy, encouraging exploration.
    """
    # Compute probabilities
    probs = F.softmax(logits, dim=-1)

    # Compute log probabilities
    log_probs = F.log_softmax(logits, dim=-1)

    # Entropy: -Σ p(a) log p(a)
    # Shape: (batch_size, seq_len)
    entropy = -(probs * log_probs).sum(dim=-1)

    # Apply mask if provided
    if mask is not None:
        entropy = entropy * mask
        num_tokens = mask.sum()
    else:
        num_tokens = entropy.numel()

    # Average over all tokens
    mean_entropy = entropy.sum() / (num_tokens + 1e-8)

    metrics = {
        "entropy": mean_entropy.item(),
    }

    return mean_entropy, metrics


def compute_kl_penalty(
    logprobs: torch.Tensor,
    ref_logprobs: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    kl_penalty_type: str = "kl"  # "kl" or "abs"
) -> tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute KL divergence penalty to keep policy close to reference.

    This prevents the policy from drifting too far from the initial model,
    which can help maintain desirable properties like fluency.

    Args:
        logprobs: Log probabilities under current policy (batch_size, seq_len)
        ref_logprobs: Log probabilities under reference policy (batch_size, seq_len)
        mask: Optional mask for valid positions
        kl_penalty_type: Type of penalty ("kl" or "abs")

    Returns:
        penalty: KL penalty (scalar)
        metrics: Dictionary of metrics

    Educational Note:
        KL divergence: D_KL(π || π_ref) = E[log π - log π_ref]

        In expectation over actions from π, this is:
        E_a~π[log π(a|s) - log π_ref(a|s)]

        We already have log probabilities for the sampled actions,
        so we can approximate this as the mean over samples.
    """
    if kl_penalty_type == "kl":
        # KL divergence: log(π) - log(π_ref)
        kl = logprobs - ref_logprobs
    elif kl_penalty_type == "abs":
        # Absolute difference (simpler alternative)
        kl = torch.abs(logprobs - ref_logprobs)
    else:
        raise ValueError(f"Unknown KL penalty type: {kl_penalty_type}")

    # Apply mask if provided
    if mask is not None:
        kl = kl * mask
        num_tokens = mask.sum()
    else:
        num_tokens = kl.numel()

    # Average over all tokens
    penalty = kl.sum() / (num_tokens + 1e-8)

    metrics = {
        f"kl_{kl_penalty_type}": penalty.item(),
    }

    return penalty, metrics


def compute_ppo_total_loss(
    policy_logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    ref_logprobs: torch.Tensor,
    values: torch.Tensor,
    old_values: torch.Tensor,
    advantages: torch.Tensor,
    returns: torch.Tensor,
    logits: Optional[torch.Tensor] = None,
    mask: Optional[torch.Tensor] = None,
    clip_ratio: float = 0.2,
    vf_coef: float = 0.5,
    entropy_coef: float = 0.01,
    kl_coef: float = 0.1,
) -> tuple[torch.Tensor, Dict[str, Any]]:
    """
    Compute total PPO loss combining policy, value, entropy, and KL.

    This is the main loss function for PPO training, combining all components.

    Args:
        policy_logprobs: Log probs under current policy
        old_logprobs: Log probs under old policy
        ref_logprobs: Log probs under reference policy
        values: Value predictions from current network
        old_values: Value predictions from old network
        advantages: Advantage estimates
        returns: Target returns
        logits: Logits for entropy computation (optional)
        mask: Mask for valid positions
        clip_ratio: PPO clipping parameter
        vf_coef: Value function loss coefficient
        entropy_coef: Entropy bonus coefficient
        kl_coef: KL penalty coefficient

    Returns:
        total_loss: Combined loss
        metrics: Dictionary of all metrics

    Educational Note:
        Total PPO loss:
        L = L^{CLIP} + c_1 * L^{V} - c_2 * H + c_3 * D_KL

        where:
        - L^{CLIP}: Clipped policy loss (main objective)
        - L^{V}: Value function loss (helps with advantage estimation)
        - H: Entropy (encourages exploration)
        - D_KL: KL divergence from reference (prevents drift)

        The coefficients (c_1, c_2, c_3) balance these objectives.
    """
    metrics = {}

    # Policy loss (PPO clipped objective)
    policy_loss, policy_metrics = compute_ppo_loss(
        logprobs=policy_logprobs,
        old_logprobs=old_logprobs,
        advantages=advantages,
        mask=mask,
        clip_ratio=clip_ratio
    )
    metrics.update(policy_metrics)

    # Value loss
    value_loss, value_metrics = compute_value_loss(
        values=values,
        returns=returns,
        old_values=old_values,
        mask=mask,
        clip_value=True,
        clip_ratio=clip_ratio
    )
    metrics.update(value_metrics)

    # Entropy bonus (if logits provided)
    if logits is not None and entropy_coef > 0:
        entropy, entropy_metrics = compute_entropy_bonus(
            logits=logits,
            mask=mask
        )
        metrics.update(entropy_metrics)
    else:
        entropy = torch.tensor(0.0, device=policy_logprobs.device)

    # KL penalty (to stay close to reference model)
    if kl_coef > 0:
        kl_penalty, kl_metrics = compute_kl_penalty(
            logprobs=policy_logprobs,
            ref_logprobs=ref_logprobs,
            mask=mask,
            kl_penalty_type="kl"
        )
        metrics.update(kl_metrics)
    else:
        kl_penalty = torch.tensor(0.0, device=policy_logprobs.device)

    # Combine losses
    total_loss = (
        policy_loss +
        vf_coef * value_loss -
        entropy_coef * entropy +
        kl_coef * kl_penalty
    )

    metrics["total_loss"] = total_loss.item()

    return total_loss, metrics
