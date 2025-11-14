"""
Tests for SFT dataset utilities.
"""

import pytest
import torch
from datasets import Dataset
from transformers import AutoTokenizer

from src.auto_bot_tuner.sft.dataset import InstructionDataset, preview_formatted_sample


@pytest.fixture(scope="function")
def tokenizer(small_vocab_tokenizer):
    """Load a small tokenizer for testing."""
    return AutoTokenizer.from_pretrained("gpt2")


@pytest.fixture(scope="function")
def alpaca_dataset():
    """Create a small Alpaca-format dataset."""
    data = {
        "instruction": [
            "Write a haiku about coding",
            "Explain what Python is"
        ],
        "input": [
            "",
            "in simple terms"
        ],
        "output": [
            "Code flows like water\nBugs hide in the shadows deep\nDebug brings the light",
            "Python is a programming language that is easy to learn and widely used."
        ]
    }
    return Dataset.from_dict(data)


@pytest.fixture(scope="function")
def chat_dataset():
    """Create a small chat-format dataset."""
    data = {
        "messages": [
            [
                {"role": "user", "content": "Hello!"},
                {"role": "assistant", "content": "Hi! How can I help you?"}
            ],
            [
                {"role": "user", "content": "What is 2+2?"},
                {"role": "assistant", "content": "2+2 equals 4."}
            ]
        ]
    }
    return Dataset.from_dict(data)


class TestInstructionDataset:
    """Test InstructionDataset class."""

    def test_alpaca_format(self, alpaca_dataset, tokenizer):
        """Test Alpaca format dataset."""
        dataset = InstructionDataset(
            dataset=alpaca_dataset,
            tokenizer=tokenizer,
            max_length=128,
            format_type="alpaca"
        )

        assert len(dataset) == 2

        # Get a sample
        sample = dataset[0]

        # Check keys
        assert "input_ids" in sample
        assert "attention_mask" in sample
        assert "labels" in sample

        # Check shapes
        assert sample["input_ids"].shape == (128,)
        assert sample["attention_mask"].shape == (128,)
        assert sample["labels"].shape == (128,)

        # Check types
        assert sample["input_ids"].dtype == torch.long
        assert sample["attention_mask"].dtype == torch.long
        assert sample["labels"].dtype == torch.long

        # Check that some labels are masked (-100)
        assert (sample["labels"] == -100).any()

        # Check that some labels are not masked
        assert (sample["labels"] != -100).any()

    def test_chat_format(self, chat_dataset, tokenizer):
        """Test chat format dataset."""
        dataset = InstructionDataset(
            dataset=chat_dataset,
            tokenizer=tokenizer,
            max_length=128,
            format_type="chat"
        )

        assert len(dataset) == 2

        sample = dataset[0]

        assert "input_ids" in sample
        assert "attention_mask" in sample
        assert "labels" in sample

    def test_auto_format_detection(self, alpaca_dataset, tokenizer):
        """Test automatic format detection."""
        dataset = InstructionDataset(
            dataset=alpaca_dataset,
            tokenizer=tokenizer,
            max_length=128,
            format_type="auto"
        )

        # Should detect Alpaca format
        sample = dataset[0]
        assert sample is not None

    def test_padding(self, alpaca_dataset, tokenizer):
        """Test that sequences are padded to max_length."""
        dataset = InstructionDataset(
            dataset=alpaca_dataset,
            tokenizer=tokenizer,
            max_length=128,
            format_type="alpaca"
        )

        sample = dataset[0]

        # Check that attention mask has 0s (padding)
        assert (sample["attention_mask"] == 0).any()

        # Check that padded positions have pad_token_id
        pad_positions = sample["attention_mask"] == 0
        assert (sample["input_ids"][pad_positions] == tokenizer.pad_token_id).all()

        # Check that padded positions have -100 in labels
        assert (sample["labels"][pad_positions] == -100).all()

    def test_label_masking(self, alpaca_dataset, tokenizer):
        """Test that prompt tokens are masked in labels."""
        dataset = InstructionDataset(
            dataset=alpaca_dataset,
            tokenizer=tokenizer,
            max_length=512,
            format_type="alpaca"
        )

        sample = dataset[0]

        # Get the first non-padding token positions
        non_pad = sample["attention_mask"] == 1
        labels = sample["labels"][non_pad]

        # First part should be masked (prompt)
        assert labels[0] == -100

        # Last part should not be all masked (response)
        assert not (labels[-10:] == -100).all()


class TestPreviewFormattedSample:
    """Test preview_formatted_sample function."""

    def test_preview_with_decode(self, alpaca_dataset, tokenizer):
        """Test preview with text decoding."""
        dataset = InstructionDataset(
            dataset=alpaca_dataset,
            tokenizer=tokenizer,
            max_length=128,
            format_type="alpaca"
        )

        preview = preview_formatted_sample(dataset, idx=0, decode=True)

        assert "input_ids_shape" in preview
        assert "full_text" in preview
        assert "response_only" in preview
        assert "num_prompt_tokens" in preview
        assert "num_response_tokens" in preview

        # Check that we have some prompt and response tokens
        assert preview["num_prompt_tokens"] > 0
        assert preview["num_response_tokens"] > 0

        # Check that full_text contains the instruction marker
        assert "Instruction" in preview["full_text"] or "instruction" in preview["full_text"].lower()

    def test_preview_without_decode(self, alpaca_dataset, tokenizer):
        """Test preview without text decoding."""
        dataset = InstructionDataset(
            dataset=alpaca_dataset,
            tokenizer=tokenizer,
            max_length=128,
            format_type="alpaca"
        )

        preview = preview_formatted_sample(dataset, idx=0, decode=False)

        assert "input_ids" in preview
        assert "attention_mask" in preview
        assert "labels" in preview

        assert isinstance(preview["input_ids"], torch.Tensor)
        assert isinstance(preview["attention_mask"], torch.Tensor)
        assert isinstance(preview["labels"], torch.Tensor)
