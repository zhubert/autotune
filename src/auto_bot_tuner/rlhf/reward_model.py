"""
Reward Model architecture for RLHF.

A reward model is a language model with a scalar output head that predicts
how good/helpful/safe a response is. It's trained on human preference data
to align with human judgments.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple
from transformers import PreTrainedModel


class RewardModel(nn.Module):
    """
    Reward model that wraps a language model with a scalar value head.

    The model processes a (prompt, response) pair and outputs a single scalar
    reward score. During training, we compare scores for chosen vs rejected
    responses and optimize using ranking losses.

    Architecture:
        Language Model (frozen or trainable)
        ↓
        [CLS] or last token representation
        ↓
        Value Head (linear layer)
        ↓
        Scalar reward

    Educational Note:
        The reward model learns to predict which response a human would prefer.
        It doesn't generate text - it just scores existing text. The scores
        are used to guide RL training of the policy model in RLHF.
    """

    def __init__(
        self,
        base_model: PreTrainedModel,
        freeze_base: bool = False,
        hidden_size: Optional[int] = None,
        dropout: float = 0.1
    ):
        """
        Args:
            base_model: Pre-trained language model
            freeze_base: If True, freeze the base model parameters
            hidden_size: Hidden size for value head (defaults to model hidden size)
            dropout: Dropout probability for value head
        """
        super().__init__()

        self.base_model = base_model

        # Freeze base model if requested
        if freeze_base:
            for param in self.base_model.parameters():
                param.requires_grad = False

        # Get hidden size from base model
        if hidden_size is None:
            if hasattr(base_model.config, "hidden_size"):
                hidden_size = base_model.config.hidden_size
            elif hasattr(base_model.config, "n_embd"):
                hidden_size = base_model.config.n_embd
            else:
                raise ValueError("Could not determine hidden size from base model")

        # Value head: projects hidden states to scalar reward
        self.value_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1)
        )

        # Initialize value head with small weights
        self.value_head[1].weight.data.normal_(mean=0.0, std=0.01)
        self.value_head[1].bias.data.zero_()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_dict: bool = True
    ) -> torch.Tensor:
        """
        Forward pass through reward model.

        Args:
            input_ids: Input token IDs of shape (batch_size, seq_len)
            attention_mask: Attention mask of shape (batch_size, seq_len)
            return_dict: If True, return dict with additional info

        Returns:
            Scalar rewards of shape (batch_size,)

        Educational Note:
            We extract the representation of the last non-padding token
            (typically where the response ends) and pass it through the
            value head to get a scalar reward.
        """
        # Forward pass through base model
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )

        # Get last layer hidden states
        # Shape: (batch_size, seq_len, hidden_size)
        hidden_states = outputs.hidden_states[-1]

        # Extract the last non-padding token's representation
        if attention_mask is not None:
            # Find the last non-padding token for each sequence
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = hidden_states.shape[0]

            # Gather the last token's hidden state for each sequence
            last_hidden = hidden_states[
                torch.arange(batch_size, device=hidden_states.device),
                sequence_lengths
            ]
        else:
            # If no mask, use the last token
            last_hidden = hidden_states[:, -1, :]

        # Pass through value head to get scalar reward
        # Shape: (batch_size, 1)
        rewards = self.value_head(last_hidden)

        # Squeeze to get (batch_size,)
        rewards = rewards.squeeze(-1)

        if return_dict:
            return {
                "rewards": rewards,
                "hidden_states": hidden_states,
                "base_model_outputs": outputs
            }
        else:
            return rewards

    def get_rewards(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Convenience method to get just the reward scores.

        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask

        Returns:
            Scalar rewards
        """
        return self.forward(input_ids, attention_mask, return_dict=False)

    def save_pretrained(self, save_directory: str):
        """
        Save the reward model.

        Args:
            save_directory: Directory to save to
        """
        import os
        os.makedirs(save_directory, exist_ok=True)

        # Save base model
        self.base_model.save_pretrained(save_directory)

        # Save value head separately
        value_head_path = os.path.join(save_directory, "value_head.pt")
        torch.save(self.value_head.state_dict(), value_head_path)

        # Save config
        config = {
            "freeze_base": not next(self.base_model.parameters()).requires_grad,
            "hidden_size": self.value_head[1].in_features,
            "dropout": self.value_head[0].p if isinstance(self.value_head[0], nn.Dropout) else 0.1
        }
        config_path = os.path.join(save_directory, "reward_model_config.json")
        import json
        with open(config_path, 'w') as f:
            json.dump(config, f)

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        base_model_class,
        **kwargs
    ):
        """
        Load a pre-trained reward model.

        Args:
            model_name_or_path: Path to saved model
            base_model_class: Class to use for loading base model
            **kwargs: Additional arguments

        Returns:
            RewardModel instance
        """
        import os
        import json

        # Load base model
        base_model = base_model_class.from_pretrained(model_name_or_path)

        # Load config if available
        config_path = os.path.join(model_name_or_path, "reward_model_config.json")
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
        else:
            config = {}

        # Create reward model
        reward_model = cls(
            base_model=base_model,
            freeze_base=config.get("freeze_base", False),
            hidden_size=config.get("hidden_size"),
            dropout=config.get("dropout", 0.1)
        )

        # Load value head weights if available
        value_head_path = os.path.join(model_name_or_path, "value_head.pt")
        if os.path.exists(value_head_path):
            reward_model.value_head.load_state_dict(torch.load(value_head_path))

        return reward_model


class EnsembleRewardModel(nn.Module):
    """
    Ensemble of multiple reward models for more robust scoring.

    Combining multiple reward models can reduce variance and make
    the reward signal more stable during RL training.
    """

    def __init__(self, reward_models: list[RewardModel]):
        """
        Args:
            reward_models: List of RewardModel instances
        """
        super().__init__()
        self.reward_models = nn.ModuleList(reward_models)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        aggregation: str = "mean"
    ) -> torch.Tensor:
        """
        Get ensemble reward predictions.

        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            aggregation: How to combine rewards ('mean', 'median', 'min', 'max')

        Returns:
            Aggregated reward scores
        """
        all_rewards = []

        for model in self.reward_models:
            rewards = model.get_rewards(input_ids, attention_mask)
            all_rewards.append(rewards)

        # Stack: (num_models, batch_size)
        all_rewards = torch.stack(all_rewards)

        if aggregation == "mean":
            return all_rewards.mean(dim=0)
        elif aggregation == "median":
            return all_rewards.median(dim=0).values
        elif aggregation == "min":
            return all_rewards.min(dim=0).values
        elif aggregation == "max":
            return all_rewards.max(dim=0).values
        else:
            raise ValueError(f"Unknown aggregation: {aggregation}")


def create_reward_model_from_pretrained(
    model_name: str,
    tokenizer,
    freeze_base: bool = False,
    device: Optional[torch.device] = None
) -> RewardModel:
    """
    Create a reward model from a pre-trained language model.

    Args:
        model_name: HuggingFace model name or path
        tokenizer: Tokenizer for the model
        freeze_base: If True, freeze the base model
        device: Device to place model on

    Returns:
        RewardModel instance

    Example:
        >>> from transformers import AutoTokenizer, AutoModelForCausalLM
        >>> tokenizer = AutoTokenizer.from_pretrained("gpt2")
        >>> reward_model = create_reward_model_from_pretrained(
        ...     "gpt2", tokenizer, freeze_base=False
        ... )
    """
    from transformers import AutoModelForCausalLM

    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(model_name)

    # Create reward model
    reward_model = RewardModel(
        base_model=base_model,
        freeze_base=freeze_base
    )

    # Move to device if specified
    if device is not None:
        reward_model = reward_model.to(device)

    return reward_model


def compare_responses(
    reward_model: RewardModel,
    prompt: str,
    response_a: str,
    response_b: str,
    tokenizer,
    device: Optional[torch.device] = None
) -> Tuple[float, float, str]:
    """
    Compare two responses using a reward model.

    Args:
        reward_model: Trained reward model
        prompt: Input prompt
        response_a: First response
        response_b: Second response
        tokenizer: Tokenizer
        device: Device to run on

    Returns:
        Tuple of (score_a, score_b, preferred)
        where preferred is "A", "B", or "tie"

    Example usage for human evaluation:
        >>> score_a, score_b, preferred = compare_responses(
        ...     reward_model, "What is Python?",
        ...     "Python is a programming language",
        ...     "Python is a snake",
        ...     tokenizer
        ... )
        >>> print(f"Preferred: {preferred} (scores: A={score_a:.2f}, B={score_b:.2f})")
    """
    device = device or next(reward_model.parameters()).device

    # Tokenize both full sequences
    seq_a = tokenizer(prompt + response_a, return_tensors="pt", padding=True, truncation=True)
    seq_b = tokenizer(prompt + response_b, return_tensors="pt", padding=True, truncation=True)

    seq_a = {k: v.to(device) for k, v in seq_a.items()}
    seq_b = {k: v.to(device) for k, v in seq_b.items()}

    # Get rewards
    reward_model.eval()
    with torch.no_grad():
        score_a = reward_model.get_rewards(**seq_a).item()
        score_b = reward_model.get_rewards(**seq_b).item()

    # Determine preferred
    if abs(score_a - score_b) < 0.1:
        preferred = "tie"
    elif score_a > score_b:
        preferred = "A"
    else:
        preferred = "B"

    return score_a, score_b, preferred
