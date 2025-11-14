"""
Trainer for RLHF with PPO (Proximal Policy Optimization).

PPO is a policy gradient method that uses clipping to prevent overly large
policy updates. It's the most common algorithm for RLHF.
"""

import os
import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Callable, List
import torch
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
import json

from .ppo_loss import compute_ppo_total_loss
from .rollout_buffer import RolloutBuffer, compute_gae, whiten_advantages
from .value_network import ValueNetwork, create_value_network_from_policy
from .reward_model import RewardModel


@dataclass
class PPOConfig:
    """Configuration for PPO training."""

    # PPO-specific parameters
    clip_ratio: float = 0.2  # PPO clipping parameter (epsilon)
    vf_coef: float = 0.5  # Value function loss coefficient
    entropy_coef: float = 0.01  # Entropy bonus coefficient
    kl_coef: float = 0.1  # KL penalty coefficient (stay close to reference)

    # Advantage estimation
    gamma: float = 0.99  # Discount factor
    gae_lambda: float = 0.95  # GAE lambda parameter
    whiten_advantages: bool = True  # Normalize advantages

    # Training hyperparameters
    learning_rate: float = 1e-6  # Very low LR for stability
    batch_size: int = 4  # Batch size for rollouts
    mini_batch_size: int = 2  # Mini-batch size for PPO updates
    gradient_accumulation_steps: int = 1
    ppo_epochs: int = 4  # Number of PPO update epochs per rollout
    num_rollouts: int = 100  # Total number of rollouts to collect
    rollout_batch_size: int = 64  # Number of rollouts per update
    max_steps: Optional[int] = None
    warmup_steps: int = 10
    max_grad_norm: float = 0.5  # Lower than SFT for stability

    # Generation parameters
    max_new_tokens: int = 128  # Max tokens to generate per response
    temperature: float = 1.0
    top_p: float = 0.9
    top_k: int = 50

    # Optimization
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8

    # Logging and checkpointing
    logging_steps: int = 1
    save_steps: int = 10
    save_total_limit: int = 3

    # Paths
    output_dir: str = "checkpoints/ppo"
    resume_from_checkpoint: Optional[str] = None

    # Mixed precision
    fp16: bool = False
    bf16: bool = False

    # Misc
    seed: int = 42


class PromptDataset(Dataset):
    """Simple dataset of prompts for PPO rollouts."""

    def __init__(self, prompts: List[str], tokenizer, max_length: int = 512):
        self.prompts = prompts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        prompt = self.prompts[idx]
        encoding = self.tokenizer(
            prompt,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "prompt": prompt
        }


class PPOTrainer:
    """
    Trainer for RLHF using Proximal Policy Optimization.

    PPO training involves:
    1. Generate responses from the policy for a batch of prompts
    2. Score responses with the reward model
    3. Compute advantages using GAE
    4. Update policy and value function using PPO loss

    This process repeats, gradually improving the policy to maximize rewards.

    Educational Note:
        RLHF (Reinforcement Learning from Human Feedback) uses RL to optimize
        a language model according to human preferences captured by a reward model.

        PPO is the most common algorithm because it:
        - Prevents overly large policy updates (via clipping)
        - Is stable and sample-efficient
        - Works well with high-dimensional action spaces (vocabulary)
    """

    def __init__(
        self,
        policy_model: torch.nn.Module,
        value_network: ValueNetwork,
        reward_model: RewardModel,
        reference_model: torch.nn.Module,
        tokenizer,
        prompt_dataset: Dataset,
        config: Optional[PPOConfig] = None,
        callbacks: Optional[List[Callable]] = None
    ):
        """
        Args:
            policy_model: The model being trained
            value_network: Value network for advantage estimation
            reward_model: Trained reward model for scoring responses
            reference_model: Frozen reference model (initial policy)
            tokenizer: Tokenizer for the models
            prompt_dataset: Dataset of prompts to generate responses for
            config: Training configuration
            callbacks: Optional callbacks called each step
        """
        self.policy_model = policy_model
        self.value_network = value_network
        self.reward_model = reward_model
        self.reference_model = reference_model
        self.tokenizer = tokenizer
        self.prompt_dataset = prompt_dataset
        self.config = config or PPOConfig()
        self.callbacks = callbacks or []

        # Setup device
        self.device = next(policy_model.parameters()).device

        # Freeze reference and reward models
        self.reference_model.eval()
        self.reward_model.eval()
        for param in self.reference_model.parameters():
            param.requires_grad = False
        for param in self.reward_model.parameters():
            param.requires_grad = False

        # Create data loader
        self.dataloader = DataLoader(
            prompt_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=0
        )

        # Setup optimizers (separate for policy and value)
        self.policy_optimizer = AdamW(
            self.policy_model.parameters(),
            lr=config.learning_rate,
            betas=(config.adam_beta1, config.adam_beta2),
            eps=config.adam_epsilon,
            weight_decay=config.weight_decay
        )

        self.value_optimizer = AdamW(
            self.value_network.parameters(),
            lr=config.learning_rate,
            betas=(config.adam_beta1, config.adam_beta2),
            eps=config.adam_epsilon,
            weight_decay=config.weight_decay
        )

        # Learning rate schedulers
        num_training_steps = config.num_rollouts
        self.policy_scheduler = self._get_linear_schedule_with_warmup(
            self.policy_optimizer,
            num_warmup_steps=config.warmup_steps,
            num_training_steps=num_training_steps
        )
        self.value_scheduler = self._get_linear_schedule_with_warmup(
            self.value_optimizer,
            num_warmup_steps=config.warmup_steps,
            num_training_steps=num_training_steps
        )

        # Training state
        self.global_step = 0
        self.epoch = 0

        # Create output directory
        os.makedirs(config.output_dir, exist_ok=True)

    def _get_linear_schedule_with_warmup(self, optimizer, num_warmup_steps, num_training_steps):
        """Create learning rate scheduler with linear warmup and decay."""
        def lr_lambda(current_step):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            return max(
                0.0,
                float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
            )
        return LambdaLR(optimizer, lr_lambda)

    @torch.no_grad()
    def generate_rollout(self, prompt_batch: dict) -> dict:
        """
        Generate a rollout for a batch of prompts.

        This includes:
        1. Generate responses from the policy
        2. Get log probabilities from policy, reference, and reward
        3. Get value estimates
        4. Compute rewards

        Args:
            prompt_batch: Batch of prompts with input_ids and attention_mask

        Returns:
            Dictionary containing rollout data
        """
        self.policy_model.eval()
        self.value_network.eval()

        query_input_ids = prompt_batch["input_ids"].to(self.device)
        query_attention_mask = prompt_batch["attention_mask"].to(self.device)

        # Generate responses
        # We'll use sampling for exploration
        outputs = self.policy_model.generate(
            input_ids=query_input_ids,
            attention_mask=query_attention_mask,
            max_new_tokens=self.config.max_new_tokens,
            do_sample=True,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            top_k=self.config.top_k,
            pad_token_id=self.tokenizer.pad_token_id,
            return_dict_in_generate=True,
            output_scores=False
        )

        # Extract generated sequences
        # Shape: (batch_size, prompt_len + response_len)
        generated_ids = outputs.sequences

        # Split into query and response
        prompt_len = query_input_ids.shape[1]
        response_ids = generated_ids[:, prompt_len:]

        # Now we need to get log probabilities for the generated tokens
        # We'll do a forward pass with the full sequence
        policy_outputs = self.policy_model(
            input_ids=generated_ids,
            attention_mask=(generated_ids != self.tokenizer.pad_token_id).long(),
            return_dict=True
        )

        # Get logits for response tokens
        # policy_outputs.logits shape: (batch_size, seq_len, vocab_size)
        response_logits = policy_outputs.logits[:, prompt_len - 1:-1, :]  # Shift for next-token prediction

        # Get log probabilities for the generated tokens
        response_logprobs = torch.nn.functional.log_softmax(response_logits, dim=-1)

        # Gather log probs for actual generated tokens
        # Shape: (batch_size, response_len)
        policy_logprobs = torch.gather(
            response_logprobs,
            dim=-1,
            index=response_ids.unsqueeze(-1)
        ).squeeze(-1)

        # Get reference model log probs
        ref_outputs = self.reference_model(
            input_ids=generated_ids,
            attention_mask=(generated_ids != self.tokenizer.pad_token_id).long(),
            return_dict=True
        )
        ref_logits = ref_outputs.logits[:, prompt_len - 1:-1, :]
        ref_logprobs_all = torch.nn.functional.log_softmax(ref_logits, dim=-1)
        ref_logprobs = torch.gather(
            ref_logprobs_all,
            dim=-1,
            index=response_ids.unsqueeze(-1)
        ).squeeze(-1)

        # Get value estimates
        value_outputs = self.value_network(
            input_ids=generated_ids,
            attention_mask=(generated_ids != self.tokenizer.pad_token_id).long(),
            return_dict=True
        )
        # Extract values for response tokens
        values = value_outputs["values"][:, prompt_len:]

        # Get rewards from reward model
        # The reward model outputs a single score per sequence
        reward_outputs = self.reward_model(
            input_ids=generated_ids,
            attention_mask=(generated_ids != self.tokenizer.pad_token_id).long(),
            return_dict=True
        )
        rewards = reward_outputs["rewards"]  # Shape: (batch_size,)

        # Create attention mask for responses
        response_mask = (response_ids != self.tokenizer.pad_token_id).float()

        return {
            "queries": query_input_ids,
            "responses": response_ids,
            "policy_logprobs": policy_logprobs,
            "ref_logprobs": ref_logprobs,
            "values": values,
            "rewards": rewards,
            "masks": response_mask,
            "logits": response_logits  # For entropy computation
        }

    def compute_advantages(self, rollout_data: dict) -> dict:
        """
        Compute advantages and returns using GAE.

        Args:
            rollout_data: Rollout data from generate_rollout

        Returns:
            Updated rollout_data with advantages and returns
        """
        batch_size = rollout_data["responses"].shape[0]
        response_len = rollout_data["responses"].shape[1]

        # In language modeling, we typically get a single reward at the end
        # Create reward sequence: 0, 0, ..., 0, R
        reward_sequence = torch.zeros_like(rollout_data["values"])
        reward_sequence[:, -1] = rollout_data["rewards"]

        # Compute GAE for each sequence in batch
        advantages_list = []
        returns_list = []

        for i in range(batch_size):
            adv, ret = compute_gae(
                rewards=reward_sequence[i],
                values=rollout_data["values"][i],
                gamma=self.config.gamma,
                lam=self.config.gae_lambda,
                mask=rollout_data["masks"][i]
            )
            advantages_list.append(adv)
            returns_list.append(ret)

        advantages = torch.stack(advantages_list)
        returns = torch.stack(returns_list)

        # Whiten advantages
        if self.config.whiten_advantages:
            advantages = whiten_advantages(advantages, rollout_data["masks"])

        rollout_data["advantages"] = advantages
        rollout_data["returns"] = returns

        return rollout_data

    def train_step(self, rollout_data: dict) -> dict:
        """
        Perform PPO update on a batch of rollouts.

        Args:
            rollout_data: Rollout data with advantages and returns

        Returns:
            Dictionary of metrics
        """
        self.policy_model.train()
        self.value_network.train()

        all_metrics = {}

        # Store old values for clipping
        with torch.no_grad():
            old_logprobs = rollout_data["policy_logprobs"].clone()
            old_values = rollout_data["values"].clone()

        # Perform multiple PPO epochs on this data
        for epoch in range(self.config.ppo_epochs):
            # Get current log probs and values
            # We need to re-compute these as the model updates
            full_input_ids = torch.cat([
                rollout_data["queries"],
                rollout_data["responses"]
            ], dim=1)
            full_attention_mask = (full_input_ids != self.tokenizer.pad_token_id).long()

            # Forward pass through policy
            policy_outputs = self.policy_model(
                input_ids=full_input_ids,
                attention_mask=full_attention_mask,
                return_dict=True
            )

            # Get log probs for response tokens
            prompt_len = rollout_data["queries"].shape[1]
            response_logits = policy_outputs.logits[:, prompt_len - 1:-1, :]
            response_logprobs = torch.nn.functional.log_softmax(response_logits, dim=-1)
            policy_logprobs = torch.gather(
                response_logprobs,
                dim=-1,
                index=rollout_data["responses"].unsqueeze(-1)
            ).squeeze(-1)

            # Get values
            value_outputs = self.value_network(
                input_ids=full_input_ids,
                attention_mask=full_attention_mask,
                return_dict=True
            )
            values = value_outputs["values"][:, prompt_len:]

            # Compute total PPO loss
            loss, metrics = compute_ppo_total_loss(
                policy_logprobs=policy_logprobs,
                old_logprobs=old_logprobs,
                ref_logprobs=rollout_data["ref_logprobs"],
                values=values,
                old_values=old_values,
                advantages=rollout_data["advantages"],
                returns=rollout_data["returns"],
                logits=response_logits,
                mask=rollout_data["masks"],
                clip_ratio=self.config.clip_ratio,
                vf_coef=self.config.vf_coef,
                entropy_coef=self.config.entropy_coef,
                kl_coef=self.config.kl_coef
            )

            # Backward pass
            self.policy_optimizer.zero_grad()
            self.value_optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.policy_model.parameters(),
                self.config.max_grad_norm
            )
            torch.nn.utils.clip_grad_norm_(
                self.value_network.parameters(),
                self.config.max_grad_norm
            )

            # Optimizer step
            self.policy_optimizer.step()
            self.value_optimizer.step()

            # Store metrics from last epoch
            all_metrics = metrics

        # Step schedulers
        self.policy_scheduler.step()
        self.value_scheduler.step()

        # Add reward metrics
        all_metrics["mean_reward"] = rollout_data["rewards"].mean().item()
        all_metrics["std_reward"] = rollout_data["rewards"].std().item()

        return all_metrics

    def train(self):
        """
        Main training loop for PPO.

        Alternates between:
        1. Collecting rollouts
        2. Computing advantages
        3. Updating policy and value function
        """
        print(f"\n{'='*80}")
        print("Starting PPO Training")
        print(f"{'='*80}\n")

        print(f"Configuration:")
        print(f"  Output directory: {self.config.output_dir}")
        print(f"  Number of rollouts: {self.config.num_rollouts}")
        print(f"  Batch size: {self.config.batch_size}")
        print(f"  PPO epochs: {self.config.ppo_epochs}")
        print(f"  Clip ratio: {self.config.clip_ratio}")
        print()

        # Training loop
        progress_bar = tqdm(total=self.config.num_rollouts, desc="PPO Training")

        rollout_count = 0
        data_iter = iter(self.dataloader)

        while rollout_count < self.config.num_rollouts:
            # Get next batch of prompts
            try:
                prompt_batch = next(data_iter)
            except StopIteration:
                # Restart dataloader
                data_iter = iter(self.dataloader)
                prompt_batch = next(data_iter)

            # Generate rollout
            rollout_data = self.generate_rollout(prompt_batch)

            # Compute advantages
            rollout_data = self.compute_advantages(rollout_data)

            # Training step
            metrics = self.train_step(rollout_data)

            # Update progress
            rollout_count += 1
            self.global_step += 1
            progress_bar.update(1)

            # Logging
            if rollout_count % self.config.logging_steps == 0:
                log_str = f"Step {rollout_count}: "
                log_str += f"Loss={metrics['total_loss']:.4f}, "
                log_str += f"Reward={metrics['mean_reward']:.4f}, "
                log_str += f"KL={metrics.get('kl_kl', 0):.4f}"
                progress_bar.write(log_str)

            # Checkpointing
            if rollout_count % self.config.save_steps == 0:
                self.save_checkpoint(f"step_{rollout_count}")

            # Callbacks
            for callback in self.callbacks:
                callback(self, rollout_data, metrics)

        progress_bar.close()

        # Save final checkpoint
        self.save_checkpoint("final")

        print(f"\n{'='*80}")
        print("Training complete!")
        print(f"{'='*80}\n")

    def save_checkpoint(self, name: str):
        """Save model checkpoint."""
        checkpoint_dir = Path(self.config.output_dir) / name
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Save policy model
        self.policy_model.save_pretrained(checkpoint_dir / "policy")
        self.tokenizer.save_pretrained(checkpoint_dir / "policy")

        # Save value network
        torch.save(
            self.value_network.state_dict(),
            checkpoint_dir / "value_network.pt"
        )

        # Save training state
        state = {
            "global_step": self.global_step,
            "epoch": self.epoch,
            "config": self.config,
        }
        with open(checkpoint_dir / "training_state.json", "w") as f:
            json.dump(state, f, indent=2, default=str)

        print(f"Checkpoint saved to {checkpoint_dir}")


def create_reference_model(policy_model: torch.nn.Module) -> torch.nn.Module:
    """
    Create a frozen copy of the policy model to use as reference.

    Args:
        policy_model: The policy model to copy

    Returns:
        Frozen reference model
    """
    reference_model = copy.deepcopy(policy_model)
    reference_model.eval()
    for param in reference_model.parameters():
        param.requires_grad = False
    return reference_model
