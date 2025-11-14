"""
Dataset utilities for Reward Model training.

A reward model is trained to predict human preferences between pairs of responses.
It learns to assign scores that align with human judgments of quality, helpfulness,
safety, etc.
"""

from typing import Dict, List, Optional, Union
from datasets import load_dataset, Dataset
from torch.utils.data import Dataset as TorchDataset
import torch


class RewardModelDataset(TorchDataset):
    """
    Dataset for training reward models.

    Each sample contains a comparison between two responses to the same prompt:
    - prompt: The input context
    - chosen: The preferred response (higher quality)
    - rejected: The non-preferred response (lower quality)

    The reward model learns to assign higher scores to chosen responses.
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
            max_length: Maximum sequence length
            max_prompt_length: Maximum prompt length (defaults to max_length // 2)
        """
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_prompt_length = max_prompt_length or (max_length // 2)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns a tokenized comparison pair.

        Returns:
            Dictionary with:
            - chosen_input_ids: Tokenized chosen response
            - chosen_attention_mask: Attention mask for chosen
            - rejected_input_ids: Tokenized rejected response
            - rejected_attention_mask: Attention mask for rejected
        """
        item = self.dataset[idx]

        prompt = item["prompt"]
        chosen = item["chosen"]
        rejected = item["rejected"]

        # Tokenize full sequences (prompt + response)
        chosen_full = prompt + chosen
        rejected_full = prompt + rejected

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

        return {
            "chosen_input_ids": chosen_tokens["input_ids"].squeeze(0),
            "chosen_attention_mask": chosen_tokens["attention_mask"].squeeze(0),
            "rejected_input_ids": rejected_tokens["input_ids"].squeeze(0),
            "rejected_attention_mask": rejected_tokens["attention_mask"].squeeze(0),
        }


class RankedResponsesDataset(TorchDataset):
    """
    Dataset for reward model training with multiple ranked responses.

    Some datasets provide rankings of multiple responses rather than
    just pairwise comparisons. This dataset handles such cases.
    """

    def __init__(
        self,
        dataset: Dataset,
        tokenizer,
        max_length: int = 512,
        num_responses: int = 2
    ):
        """
        Args:
            dataset: Dataset with 'prompt' and 'responses' (list of responses)
            tokenizer: Tokenizer from transformers
            max_length: Maximum sequence length
            num_responses: Number of responses to sample per prompt
        """
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.num_responses = num_responses

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns tokenized ranked responses.

        Returns:
            Dictionary with tokenized responses and their rankings
        """
        item = self.dataset[idx]

        prompt = item["prompt"]
        responses = item["responses"][:self.num_responses]
        rankings = item.get("rankings", list(range(len(responses))))

        # Tokenize all responses
        tokenized = []
        for response in responses:
            full_text = prompt + response
            tokens = self.tokenizer(
                full_text,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            tokenized.append({
                "input_ids": tokens["input_ids"].squeeze(0),
                "attention_mask": tokens["attention_mask"].squeeze(0)
            })

        # Stack into tensors
        result = {
            "input_ids": torch.stack([t["input_ids"] for t in tokenized]),
            "attention_mask": torch.stack([t["attention_mask"] for t in tokenized]),
            "rankings": torch.tensor(rankings[:len(responses)], dtype=torch.long)
        }

        return result


class ScoredResponsesDataset(TorchDataset):
    """
    Dataset for reward models with absolute score labels.

    Some datasets provide direct quality scores rather than relative rankings.
    """

    def __init__(
        self,
        dataset: Dataset,
        tokenizer,
        max_length: int = 512
    ):
        """
        Args:
            dataset: Dataset with 'prompt', 'response', and 'score' columns
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

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns tokenized response with its score.

        Returns:
            Dictionary with input_ids, attention_mask, and score
        """
        item = self.dataset[idx]

        prompt = item["prompt"]
        response = item["response"]
        score = item["score"]

        full_text = prompt + response
        tokens = self.tokenizer(
            full_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        return {
            "input_ids": tokens["input_ids"].squeeze(0),
            "attention_mask": tokens["attention_mask"].squeeze(0),
            "score": torch.tensor(score, dtype=torch.float)
        }


def load_reward_dataset(
    dataset_name: str,
    split: str = "train",
    max_samples: Optional[int] = None,
    **kwargs
) -> Dataset:
    """
    Load a reward modeling dataset from HuggingFace.

    Args:
        dataset_name: Dataset identifier
        split: Dataset split
        max_samples: Optional limit on samples
        **kwargs: Additional arguments for load_dataset

    Returns:
        HuggingFace Dataset

    Common reward modeling datasets:
    - 'Anthropic/hh-rlhf': Human feedback on helpfulness/harmlessness
    - 'Dahoas/rm-static': Static reward modeling dataset
    - 'lvwerra/stack-exchange-paired': StackOverflow preferences
    - 'OpenAssistant/oasst1': Conversational rankings
    """
    dataset = load_dataset(dataset_name, split=split, **kwargs)

    if max_samples is not None:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    return dataset


def convert_preference_to_reward_format(dataset: Dataset) -> Dataset:
    """
    Ensure dataset has the required columns for reward modeling.

    Reward model datasets need: prompt, chosen, rejected

    Args:
        dataset: Input dataset

    Returns:
        Dataset in reward model format
    """
    # Check if dataset already has correct format
    required_cols = ["prompt", "chosen", "rejected"]
    if all(col in dataset.column_names for col in required_cols):
        return dataset

    # Handle different formats
    if "messages" in dataset.column_names:
        # Conversational format - need to convert
        def format_conversation(item):
            messages = item["messages"]
            # Extract last user message as prompt
            # and assistant responses as chosen/rejected
            # (This is a simplified conversion)
            prompt = ""
            chosen = ""
            rejected = ""

            for msg in messages:
                if msg["role"] == "user":
                    prompt = msg["content"]
                elif msg["role"] == "assistant":
                    chosen = msg["content"]

            return {
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected if rejected else chosen
            }

        dataset = dataset.map(format_conversation)

    return dataset


def validate_reward_dataset(dataset: Dataset) -> bool:
    """
    Validate that a dataset has the required format for reward modeling.

    Args:
        dataset: Dataset to validate

    Returns:
        True if valid

    Raises:
        ValueError: If dataset is invalid
    """
    required_columns = ["prompt", "chosen", "rejected"]
    missing = [col for col in required_columns if col not in dataset.column_names]

    if missing:
        raise ValueError(
            f"Dataset missing required columns: {missing}\n"
            f"Found columns: {dataset.column_names}\n"
            f"Reward model datasets must have: {required_columns}"
        )

    return True


def preview_reward_sample(
    dataset: RewardModelDataset,
    idx: int = 0
) -> Dict:
    """
    Preview a reward model sample for debugging.

    Args:
        dataset: RewardModelDataset instance
        idx: Sample index

    Returns:
        Dictionary with decoded text
    """
    sample = dataset[idx]
    tokenizer = dataset.tokenizer

    chosen_text = tokenizer.decode(
        sample["chosen_input_ids"],
        skip_special_tokens=True
    )
    rejected_text = tokenizer.decode(
        sample["rejected_input_ids"],
        skip_special_tokens=True
    )

    return {
        "chosen": chosen_text,
        "rejected": rejected_text,
        "chosen_length": (sample["chosen_attention_mask"] == 1).sum().item(),
        "rejected_length": (sample["rejected_attention_mask"] == 1).sum().item(),
    }


def create_synthetic_preference_data(
    instruction_dataset: Dataset,
    num_samples: int = 1000,
    seed: int = 42
) -> Dataset:
    """
    Create synthetic preference data for testing/prototyping.

    This takes an instruction dataset and creates artificial preference pairs
    by generating variations with different qualities.

    Args:
        instruction_dataset: Base instruction dataset
        num_samples: Number of samples to generate
        seed: Random seed

    Returns:
        Synthetic preference dataset

    Note: This is for testing only - real preference data requires human labels!
    """
    import random
    random.seed(seed)

    synthetic_data = []

    for i in range(min(num_samples, len(instruction_dataset))):
        item = instruction_dataset[i]

        prompt = item.get("instruction", item.get("prompt", ""))
        response = item.get("output", item.get("response", ""))

        # Create "better" and "worse" versions (very simplistic)
        # In reality, you'd use human feedback or model-based scoring
        chosen = response
        rejected = response[:len(response)//2] + "..."  # Truncated = worse

        synthetic_data.append({
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected
        })

    return Dataset.from_list(synthetic_data)
