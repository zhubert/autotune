"""
Dataset utilities for Supervised Fine-Tuning (SFT).

This module handles loading and formatting instruction datasets for SFT training.
We support common instruction formats like Alpaca and general conversational formats.
"""

from typing import Dict, List, Optional, Union
from datasets import load_dataset, Dataset
from torch.utils.data import Dataset as TorchDataset
import torch


class InstructionDataset(TorchDataset):
    """
    Dataset for instruction-following fine-tuning.

    Formats conversations into tokenized sequences with proper masking
    so that loss is only computed on assistant responses, not on prompts.

    Common formats:
    - Alpaca: instruction, input (optional), output
    - Chat: messages list with role/content
    """

    def __init__(
        self,
        dataset: Dataset,
        tokenizer,
        max_length: int = 512,
        format_type: str = "alpaca"
    ):
        """
        Args:
            dataset: HuggingFace dataset object
            tokenizer: Tokenizer from transformers
            max_length: Maximum sequence length
            format_type: Dataset format ('alpaca', 'chat', or 'auto')
        """
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.format_type = format_type

        # Ensure tokenizer has pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns a tokenized sample with labels masked for prompt tokens.

        Returns:
            Dictionary with:
            - input_ids: Token IDs
            - attention_mask: Attention mask
            - labels: Token IDs with prompt tokens set to -100 (ignored in loss)
        """
        item = self.dataset[idx]

        # Format the conversation based on dataset type
        if self.format_type == "alpaca":
            prompt, response = self._format_alpaca(item)
        elif self.format_type == "chat":
            prompt, response = self._format_chat(item)
        else:
            # Auto-detect format
            if "messages" in item:
                prompt, response = self._format_chat(item)
            elif "instruction" in item:
                prompt, response = self._format_alpaca(item)
            else:
                raise ValueError(f"Unknown dataset format: {item.keys()}")

        # Tokenize prompt and response separately to create proper labels
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=True)
        response_ids = self.tokenizer.encode(response, add_special_tokens=False)

        # Combine and truncate if necessary
        total_ids = prompt_ids + response_ids + [self.tokenizer.eos_token_id]
        if len(total_ids) > self.max_length:
            # Truncate from the response, keep full prompt if possible
            overflow = len(total_ids) - self.max_length
            response_ids = response_ids[:-overflow] if len(prompt_ids) < self.max_length else response_ids
            total_ids = (prompt_ids + response_ids + [self.tokenizer.eos_token_id])[:self.max_length]

        # Create labels: -100 for prompt tokens (ignored), actual tokens for response
        # This ensures we only compute loss on the model's generated responses
        labels = [-100] * len(prompt_ids) + response_ids + [self.tokenizer.eos_token_id]
        labels = labels[:self.max_length]  # Match length after truncation

        # Pad to max_length
        input_ids = total_ids + [self.tokenizer.pad_token_id] * (self.max_length - len(total_ids))
        attention_mask = [1] * len(total_ids) + [0] * (self.max_length - len(total_ids))
        labels = labels + [-100] * (self.max_length - len(labels))

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long)
        }

    def _format_alpaca(self, item: Dict) -> tuple[str, str]:
        """
        Format Alpaca-style instruction data.

        Alpaca format:
        - instruction: The task description
        - input: Optional context/input for the task
        - output: Expected response

        Returns:
            (prompt, response) tuple
        """
        instruction = item["instruction"]
        input_text = item.get("input", "")
        output = item["output"]

        if input_text:
            prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n"
        else:
            prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"

        return prompt, output

    def _format_chat(self, item: Dict) -> tuple[str, str]:
        """
        Format chat-style messages.

        Chat format:
        - messages: List of {role: "user"/"assistant", content: "..."}

        Returns:
            (prompt, response) tuple where prompt includes all messages
            except the last assistant message
        """
        messages = item["messages"]

        # Build prompt from all messages except last assistant response
        prompt_parts = []
        response = ""

        for msg in messages:
            role = msg["role"]
            content = msg["content"]

            if role == "user":
                prompt_parts.append(f"User: {content}")
            elif role == "assistant":
                if msg == messages[-1]:
                    # This is the target response
                    response = content
                else:
                    # Previous assistant messages are part of context
                    prompt_parts.append(f"Assistant: {content}")
            elif role == "system":
                prompt_parts.append(f"System: {content}")

        prompt = "\n\n".join(prompt_parts) + "\n\nAssistant: "

        return prompt, response


def load_instruction_dataset(
    dataset_name: str,
    split: str = "train",
    max_samples: Optional[int] = None,
    **kwargs
) -> Dataset:
    """
    Load a common instruction dataset from HuggingFace.

    Args:
        dataset_name: Dataset identifier (e.g., 'yahma/alpaca-cleaned')
        split: Dataset split ('train', 'test', 'validation')
        max_samples: Optional limit on number of samples
        **kwargs: Additional arguments for load_dataset

    Returns:
        HuggingFace Dataset object

    Common datasets:
    - 'yahma/alpaca-cleaned': Cleaned Alpaca instructions
    - 'OpenAssistant/oasst1': Conversational assistance data
    - 'Anthropic/hh-rlhf': Helpful and harmless conversations
    """
    dataset = load_dataset(dataset_name, split=split, **kwargs)

    if max_samples is not None:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    return dataset


def preview_formatted_sample(
    dataset: InstructionDataset,
    idx: int = 0,
    decode: bool = True
) -> Dict:
    """
    Preview a formatted sample from the dataset for debugging.

    Args:
        dataset: InstructionDataset instance
        idx: Sample index
        decode: If True, decode tokens back to text

    Returns:
        Dictionary with formatted sample information
    """
    sample = dataset[idx]

    if decode:
        tokenizer = dataset.tokenizer

        # Decode input
        input_text = tokenizer.decode(sample["input_ids"], skip_special_tokens=False)

        # Decode only the non-masked label tokens
        label_ids = sample["labels"]
        response_tokens = [token_id for token_id in label_ids if token_id != -100]
        response_text = tokenizer.decode(response_tokens, skip_special_tokens=True)

        return {
            "input_ids_shape": sample["input_ids"].shape,
            "full_text": input_text,
            "response_only": response_text,
            "num_prompt_tokens": (label_ids == -100).sum().item(),
            "num_response_tokens": len(response_tokens)
        }

    return {
        "input_ids": sample["input_ids"],
        "attention_mask": sample["attention_mask"],
        "labels": sample["labels"]
    }
