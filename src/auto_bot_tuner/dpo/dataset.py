"""
Dataset utilities for Direct Preference Optimization (DPO).

DPO learns from preference data where each example contains:
- A prompt
- A chosen (preferred) response
- A rejected (non-preferred) response

The model learns to increase the likelihood of chosen responses relative
to rejected responses, without requiring a separate reward model.
"""

from typing import Dict, List, Optional
from datasets import load_dataset, Dataset
from torch.utils.data import Dataset as TorchDataset
import torch


class PreferenceDataset(TorchDataset):
    """
    Dataset for preference-based training (DPO).

    Each sample contains:
    - prompt: The input context
    - chosen: The preferred response
    - rejected: The non-preferred response

    The dataset tokenizes both responses and returns them separately
    for DPO loss computation.
    """

    def __init__(
        self,
        dataset: Dataset,
        tokenizer,
        max_length: int = 512,
        max_prompt_length: Optional[int] = None
    ):
        """
        Args:
            dataset: HuggingFace dataset with 'prompt', 'chosen', 'rejected' columns
            tokenizer: Tokenizer from transformers
            max_length: Maximum sequence length for full prompt+response
            max_prompt_length: Maximum prompt length (defaults to max_length // 2)
        """
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_prompt_length = max_prompt_length or (max_length // 2)

        # Ensure tokenizer has pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns a tokenized preference sample.

        Returns:
            Dictionary with:
            - prompt_input_ids: Tokenized prompt
            - prompt_attention_mask: Attention mask for prompt
            - chosen_input_ids: Tokenized chosen response
            - chosen_attention_mask: Attention mask for chosen
            - rejected_input_ids: Tokenized rejected response
            - rejected_attention_mask: Attention mask for rejected
        """
        item = self.dataset[idx]

        prompt = item["prompt"]
        chosen = item["chosen"]
        rejected = item["rejected"]

        # Tokenize prompt
        prompt_tokens = self.tokenizer(
            prompt,
            max_length=self.max_prompt_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        # Tokenize chosen response (full sequence: prompt + chosen)
        chosen_full = prompt + chosen
        chosen_tokens = self.tokenizer(
            chosen_full,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        # Tokenize rejected response (full sequence: prompt + rejected)
        rejected_full = prompt + rejected
        rejected_tokens = self.tokenizer(
            rejected_full,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        return {
            "prompt_input_ids": prompt_tokens["input_ids"].squeeze(0),
            "prompt_attention_mask": prompt_tokens["attention_mask"].squeeze(0),
            "chosen_input_ids": chosen_tokens["input_ids"].squeeze(0),
            "chosen_attention_mask": chosen_tokens["attention_mask"].squeeze(0),
            "rejected_input_ids": rejected_tokens["input_ids"].squeeze(0),
            "rejected_attention_mask": rejected_tokens["attention_mask"].squeeze(0),
        }


class ConversationalPreferenceDataset(TorchDataset):
    """
    Dataset for conversational preference data.

    Handles datasets where preferences are expressed over full conversations
    rather than single-turn responses.
    """

    def __init__(
        self,
        dataset: Dataset,
        tokenizer,
        max_length: int = 512
    ):
        """
        Args:
            dataset: Dataset with 'prompt', 'chosen', 'rejected'
            tokenizer: Tokenizer from transformers
            max_length: Maximum sequence length
        """
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def __len__(self) -> int:
        return len(self.dataset)

    def _format_conversation(self, messages: List[Dict]) -> str:
        """Format a list of messages into a single string."""
        formatted = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            formatted.append(f"{role.capitalize()}: {content}")
        return "\n\n".join(formatted)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns tokenized conversational preference sample.

        Supports two formats:
        1. Simple: {'prompt': str, 'chosen': str, 'rejected': str}
        2. Conversational: {'prompt': List[dict], 'chosen': List[dict], 'rejected': List[dict]}
        """
        item = self.dataset[idx]

        # Check if conversational format
        if isinstance(item["prompt"], list):
            prompt = self._format_conversation(item["prompt"])
            chosen = self._format_conversation(item["chosen"])
            rejected = self._format_conversation(item["rejected"])
        else:
            prompt = item["prompt"]
            chosen = item["chosen"]
            rejected = item["rejected"]

        # Tokenize full sequences
        chosen_full = prompt + "\n\n" + chosen
        rejected_full = prompt + "\n\n" + rejected

        chosen_tokens = self.tokenizer(
            chosen_full,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        rejected_tokens = self.tokenizer(
            rejected_full,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        # Also tokenize prompt separately for masking
        prompt_length = len(self.tokenizer.encode(prompt, add_special_tokens=True))

        return {
            "chosen_input_ids": chosen_tokens["input_ids"].squeeze(0),
            "chosen_attention_mask": chosen_tokens["attention_mask"].squeeze(0),
            "rejected_input_ids": rejected_tokens["input_ids"].squeeze(0),
            "rejected_attention_mask": rejected_tokens["attention_mask"].squeeze(0),
            "prompt_length": torch.tensor(prompt_length, dtype=torch.long),
        }


def load_preference_dataset(
    dataset_name: str,
    split: str = "train",
    max_samples: Optional[int] = None,
    **kwargs
) -> Dataset:
    """
    Load a preference dataset from HuggingFace.

    Args:
        dataset_name: Dataset identifier
        split: Dataset split ('train', 'test', 'validation')
        max_samples: Optional limit on number of samples
        **kwargs: Additional arguments for load_dataset

    Returns:
        HuggingFace Dataset object

    Common preference datasets:
    - 'Anthropic/hh-rlhf': Human preference data for helpfulness/harmlessness
    - 'lvwerra/stack-exchange-paired': StackOverflow paired responses
    - 'Dahoas/rm-static': Reward modeling static dataset
    """
    dataset = load_dataset(dataset_name, split=split, **kwargs)

    if max_samples is not None:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    return dataset


def convert_reward_dataset_to_preference(dataset: Dataset) -> Dataset:
    """
    Convert a reward modeling dataset to DPO preference format.

    Reward datasets often have format:
    - prompt
    - response
    - score

    This function converts them to:
    - prompt
    - chosen (higher scored response)
    - rejected (lower scored response)

    Args:
        dataset: Reward modeling dataset

    Returns:
        Dataset in DPO preference format
    """
    # Group by prompt and find pairs
    from collections import defaultdict

    prompt_groups = defaultdict(list)

    for item in dataset:
        prompt = item["prompt"]
        response = item["response"] if "response" in item else item["chosen"]
        score = item.get("score", 0)

        prompt_groups[prompt].append({
            "response": response,
            "score": score
        })

    # Create preference pairs
    preference_pairs = []

    for prompt, responses in prompt_groups.items():
        if len(responses) < 2:
            continue

        # Sort by score
        responses = sorted(responses, key=lambda x: x["score"], reverse=True)

        # Create pairs: best vs worst, second-best vs second-worst, etc.
        for i in range(len(responses) // 2):
            chosen = responses[i]["response"]
            rejected = responses[-(i+1)]["response"]

            preference_pairs.append({
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected
            })

    return Dataset.from_list(preference_pairs)


def preview_preference_sample(
    dataset: PreferenceDataset,
    idx: int = 0
) -> Dict:
    """
    Preview a preference sample for debugging.

    Args:
        dataset: PreferenceDataset instance
        idx: Sample index

    Returns:
        Dictionary with decoded text for inspection
    """
    sample = dataset[idx]
    tokenizer = dataset.tokenizer

    prompt_text = tokenizer.decode(
        sample["prompt_input_ids"],
        skip_special_tokens=True
    )
    chosen_text = tokenizer.decode(
        sample["chosen_input_ids"],
        skip_special_tokens=True
    )
    rejected_text = tokenizer.decode(
        sample["rejected_input_ids"],
        skip_special_tokens=True
    )

    return {
        "prompt": prompt_text,
        "chosen_full": chosen_text,
        "rejected_full": rejected_text,
        "chosen_length": (sample["chosen_attention_mask"] == 1).sum().item(),
        "rejected_length": (sample["rejected_attention_mask"] == 1).sum().item(),
    }


def validate_preference_dataset(dataset: Dataset) -> bool:
    """
    Validate that a dataset has the required columns for DPO.

    Args:
        dataset: HuggingFace dataset to validate

    Returns:
        True if valid, False otherwise

    Raises:
        ValueError: If required columns are missing
    """
    required_columns = ["prompt", "chosen", "rejected"]

    missing = [col for col in required_columns if col not in dataset.column_names]

    if missing:
        raise ValueError(
            f"Dataset missing required columns: {missing}\n"
            f"Found columns: {dataset.column_names}\n"
            f"DPO datasets must have: {required_columns}"
        )

    return True
