"""
Loss functions for Reward Model training.

Reward models are trained using ranking losses that encourage the model
to assign higher scores to preferred responses than to rejected ones.
"""

import torch
import torch.nn.functional as F
from typing import Optional


def compute_ranking_loss(
    chosen_rewards: torch.Tensor,
    rejected_rewards: torch.Tensor,
    margin: float = 0.0,
    reduction: str = "mean"
) -> torch.Tensor:
    """
    Compute ranking loss for reward model training.

    The ranking loss encourages the model to assign higher rewards to
    chosen responses than rejected responses, with an optional margin.

    Loss = -log(sigmoid(chosen_reward - rejected_reward - margin))

    This is equivalent to binary cross-entropy on whether chosen is better.

    Educational Note:
        The sigmoid function converts the reward difference into a probability
        that the chosen response is better than the rejected one. We want this
        probability to be high (close to 1), so we minimize the negative log
        probability.

    Args:
        chosen_rewards: Rewards for chosen responses, shape (batch_size,)
        rejected_rewards: Rewards for rejected responses, shape (batch_size,)
        margin: Margin to enforce between chosen and rejected (default: 0.0)
        reduction: How to reduce the loss ('mean', 'sum', or 'none')

    Returns:
        Ranking loss tensor

    Example:
        >>> chosen_rewards = torch.tensor([1.0, 2.0, 3.0])
        >>> rejected_rewards = torch.tensor([0.5, 1.5, 2.5])
        >>> loss = compute_ranking_loss(chosen_rewards, rejected_rewards)
    """
    # Compute difference: chosen should be higher than rejected
    logits = chosen_rewards - rejected_rewards - margin

    # Apply log-sigmoid for numerical stability
    # -log(sigmoid(x)) = log(1 + exp(-x))
    loss = F.softplus(-logits)  # softplus(x) = log(1 + exp(x))

    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    else:
        return loss


def compute_ranking_loss_with_metrics(
    chosen_rewards: torch.Tensor,
    rejected_rewards: torch.Tensor,
    margin: float = 0.0
) -> dict:
    """
    Compute ranking loss along with additional metrics.

    Args:
        chosen_rewards: Rewards for chosen responses
        rejected_rewards: Rewards for rejected responses
        margin: Margin for ranking loss

    Returns:
        Dictionary with:
        - loss: The ranking loss
        - accuracy: Fraction where chosen_reward > rejected_reward
        - mean_chosen_reward: Average chosen reward
        - mean_rejected_reward: Average rejected reward
        - mean_margin: Average margin between chosen and rejected
    """
    loss = compute_ranking_loss(chosen_rewards, rejected_rewards, margin)

    # Accuracy: how often does the model rank chosen higher?
    accuracy = (chosen_rewards > rejected_rewards).float().mean()

    # Mean rewards
    mean_chosen = chosen_rewards.mean()
    mean_rejected = rejected_rewards.mean()

    # Mean margin
    mean_margin = (chosen_rewards - rejected_rewards).mean()

    return {
        "loss": loss,
        "accuracy": accuracy,
        "mean_chosen_reward": mean_chosen,
        "mean_rejected_reward": mean_rejected,
        "mean_margin": mean_margin,
    }


def compute_pairwise_loss(
    chosen_rewards: torch.Tensor,
    rejected_rewards: torch.Tensor,
    margin: float = 1.0,
    reduction: str = "mean"
) -> torch.Tensor:
    """
    Compute pairwise hinge loss.

    Hinge loss: max(0, margin - (chosen_reward - rejected_reward))

    This is similar to SVM-style ranking loss. It penalizes when the
    margin between chosen and rejected is less than the desired margin.

    Args:
        chosen_rewards: Rewards for chosen responses
        rejected_rewards: Rewards for rejected responses
        margin: Desired margin between chosen and rejected (default: 1.0)
        reduction: How to reduce the loss

    Returns:
        Pairwise hinge loss
    """
    # Difference should be at least 'margin'
    diff = chosen_rewards - rejected_rewards
    loss = F.relu(margin - diff)

    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    else:
        return loss


def compute_listwise_loss(
    rewards: torch.Tensor,
    rankings: torch.Tensor,
    temperature: float = 1.0
) -> torch.Tensor:
    """
    Compute listwise ranking loss for multiple responses.

    This is useful when you have rankings of multiple responses
    rather than just pairwise comparisons.

    Uses ListNet loss based on top-one probability distribution.

    Args:
        rewards: Predicted rewards of shape (batch_size, num_responses)
        rankings: Ground truth rankings of shape (batch_size, num_responses)
                  Lower rank = better (0 is best)
        temperature: Temperature for softmax (default: 1.0)

    Returns:
        Listwise ranking loss

    Educational Note:
        ListNet treats ranking as predicting a probability distribution
        over which response is best. It minimizes KL divergence between
        predicted and ground truth distributions.
    """
    # Convert rankings to probability distribution (lower rank = higher prob)
    # Use negative exp of rankings to get probabilities
    target_probs = F.softmax(-rankings.float() / temperature, dim=-1)

    # Convert rewards to probability distribution
    pred_probs = F.softmax(rewards / temperature, dim=-1)

    # KL divergence loss
    loss = F.kl_div(
        pred_probs.log(),
        target_probs,
        reduction='batchmean'
    )

    return loss


def compute_contrastive_loss(
    chosen_rewards: torch.Tensor,
    rejected_rewards: torch.Tensor,
    temperature: float = 0.1
) -> torch.Tensor:
    """
    Compute contrastive loss for reward model training.

    This treats reward modeling as a contrastive learning problem,
    similar to SimCLR or CLIP.

    Args:
        chosen_rewards: Rewards for chosen responses
        rejected_rewards: Rewards for rejected responses
        temperature: Temperature parameter for scaling

    Returns:
        Contrastive loss
    """
    # Stack rewards: (batch_size, 2)
    rewards = torch.stack([chosen_rewards, rejected_rewards], dim=1)

    # Scale by temperature
    rewards = rewards / temperature

    # Labels: 0 = chosen (should be higher)
    labels = torch.zeros(chosen_rewards.shape[0], dtype=torch.long, device=rewards.device)

    # Cross-entropy loss
    loss = F.cross_entropy(rewards, labels)

    return loss


class RankingLoss(torch.nn.Module):
    """
    Ranking loss as a PyTorch module.

    Useful for integration with training frameworks that expect nn.Module losses.
    """

    def __init__(self, margin: float = 0.0, loss_type: str = "ranking"):
        """
        Args:
            margin: Margin for ranking loss
            loss_type: Type of loss ('ranking', 'hinge', 'contrastive')
        """
        super().__init__()
        self.margin = margin
        self.loss_type = loss_type

    def forward(
        self,
        chosen_rewards: torch.Tensor,
        rejected_rewards: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the ranking loss.

        Args:
            chosen_rewards: Rewards for chosen responses
            rejected_rewards: Rewards for rejected responses

        Returns:
            Loss tensor
        """
        if self.loss_type == "ranking":
            return compute_ranking_loss(chosen_rewards, rejected_rewards, self.margin)
        elif self.loss_type == "hinge":
            return compute_pairwise_loss(chosen_rewards, rejected_rewards, self.margin)
        elif self.loss_type == "contrastive":
            return compute_contrastive_loss(chosen_rewards, rejected_rewards, temperature=self.margin)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")


def compute_reward_model_loss(
    model,
    chosen_input_ids: torch.Tensor,
    chosen_attention_mask: torch.Tensor,
    rejected_input_ids: torch.Tensor,
    rejected_attention_mask: torch.Tensor,
    margin: float = 0.0,
    return_metrics: bool = False
):
    """
    Convenience function to compute reward model loss from inputs.

    Args:
        model: RewardModel instance
        chosen_input_ids: Token IDs for chosen responses
        chosen_attention_mask: Attention mask for chosen
        rejected_input_ids: Token IDs for rejected responses
        rejected_attention_mask: Attention mask for rejected
        margin: Margin for ranking loss
        return_metrics: If True, return metrics dict instead of just loss

    Returns:
        Loss tensor or metrics dictionary
    """
    # Get rewards from model
    chosen_rewards = model.get_rewards(chosen_input_ids, chosen_attention_mask)
    rejected_rewards = model.get_rewards(rejected_input_ids, rejected_attention_mask)

    if return_metrics:
        return compute_ranking_loss_with_metrics(chosen_rewards, rejected_rewards, margin)
    else:
        return compute_ranking_loss(chosen_rewards, rejected_rewards, margin)


def validate_reward_predictions(
    chosen_rewards: torch.Tensor,
    rejected_rewards: torch.Tensor
) -> dict:
    """
    Validate and analyze reward predictions.

    Args:
        chosen_rewards: Predicted rewards for chosen responses
        rejected_rewards: Predicted rewards for rejected responses

    Returns:
        Dictionary with validation statistics
    """
    stats = {
        "accuracy": (chosen_rewards > rejected_rewards).float().mean().item(),
        "mean_chosen": chosen_rewards.mean().item(),
        "mean_rejected": rejected_rewards.mean().item(),
        "std_chosen": chosen_rewards.std().item(),
        "std_rejected": rejected_rewards.std().item(),
        "mean_diff": (chosen_rewards - rejected_rewards).mean().item(),
        "median_diff": (chosen_rewards - rejected_rewards).median().item(),
        "num_correct": (chosen_rewards > rejected_rewards).sum().item(),
        "num_total": chosen_rewards.shape[0],
    }

    # Check for potential issues
    if stats["accuracy"] < 0.6:
        stats["warning"] = "Low accuracy - model may not be learning well"
    elif stats["mean_diff"] < 0.1:
        stats["warning"] = "Small reward difference - may need stronger signal"

    return stats
