"""
Value network for PPO training.

The value network estimates the expected return from each state.
It's used to compute advantages and is trained alongside the policy.
"""

import torch
import torch.nn as nn
from typing import Optional
from transformers import PreTrainedModel


class ValueNetwork(nn.Module):
    """
    Value network that estimates state values for PPO.

    Similar to the reward model, but outputs per-token value estimates
    instead of a single scalar. Used during PPO training to compute
    advantages via GAE.

    Architecture:
        Language Model (shared with policy or separate)
        ↓
        Hidden states for each token
        ↓
        Value Head (linear layer)
        ↓
        Value estimates per token

    Educational Note:
        In PPO, we need value estimates V(s_t) for each timestep to compute
        advantages. The value network learns to predict future rewards,
        helping the policy understand which actions lead to better outcomes.
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

        # Value head: projects hidden states to scalar values
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
        Forward pass through value network.

        Args:
            input_ids: Input token IDs of shape (batch_size, seq_len)
            attention_mask: Attention mask of shape (batch_size, seq_len)
            return_dict: If True, return dict with additional info

        Returns:
            Value estimates of shape (batch_size, seq_len)

        Educational Note:
            Unlike the reward model which returns a single scalar,
            we return per-token value estimates for computing GAE.
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

        # Pass through value head to get per-token values
        # Shape: (batch_size, seq_len, 1)
        values = self.value_head(hidden_states)

        # Squeeze to get (batch_size, seq_len)
        values = values.squeeze(-1)

        if return_dict:
            return {
                "values": values,
                "hidden_states": hidden_states,
                "base_model_outputs": outputs
            }
        else:
            return values

    def get_values(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Convenience method to get values.

        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask

        Returns:
            Value estimates of shape (batch_size, seq_len)
        """
        outputs = self.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        return outputs["values"]


def create_value_network_from_pretrained(
    model_name: str,
    tokenizer,
    freeze_base: bool = False,
    device: Optional[torch.device] = None
) -> ValueNetwork:
    """
    Create a value network from a pretrained model.

    Args:
        model_name: Name or path of pretrained model
        tokenizer: Tokenizer for the model
        freeze_base: If True, freeze base model parameters
        device: Device to load model on

    Returns:
        ValueNetwork instance

    Example:
        >>> value_net = create_value_network_from_pretrained("gpt2", tokenizer)
    """
    from transformers import AutoModelForCausalLM

    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(model_name)

    # Create value network
    value_network = ValueNetwork(
        base_model=base_model,
        freeze_base=freeze_base
    )

    # Move to device if specified
    if device is not None:
        value_network = value_network.to(device)

    return value_network


def create_value_network_from_policy(
    policy_model: torch.nn.Module,
    freeze_base: bool = False
) -> ValueNetwork:
    """
    Create a value network using the same architecture as a policy model.

    This is useful for sharing the base model between policy and value networks,
    though they will have separate parameters.

    Args:
        policy_model: Policy model to copy architecture from
        freeze_base: If True, freeze base model parameters

    Returns:
        ValueNetwork instance with same architecture as policy

    Example:
        >>> # Create value network matching policy
        >>> value_net = create_value_network_from_policy(policy_model)
    """
    import copy

    # Deep copy the base model to avoid sharing parameters
    base_model = copy.deepcopy(policy_model)

    # Remove any task-specific heads if present
    if hasattr(base_model, "lm_head"):
        base_model.lm_head = None
    if hasattr(base_model, "score"):
        base_model.score = None

    # Create value network
    value_network = ValueNetwork(
        base_model=base_model,
        freeze_base=freeze_base
    )

    return value_network
