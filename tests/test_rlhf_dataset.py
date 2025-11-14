"""
Tests for RLHF dataset utilities.
"""

import pytest
import torch
from datasets import Dataset
from transformers import AutoTokenizer

from src.auto_bot_tuner.rlhf.dataset import RewardModelDataset


@pytest.fixture(scope="function")
def tokenizer(small_vocab_tokenizer):
    """Load a small tokenizer for testing."""
    return AutoTokenizer.from_pretrained("gpt2")


@pytest.fixture(scope="function")
def reward_dataset():
    """Create a small reward modeling dataset."""
    data = {
        "prompt": [
            "What is Python?",
            "Explain machine learning"
        ],
        "chosen": [
            "Python is a high-level programming language.",
            "Machine learning is a subset of AI that learns from data."
        ],
        "rejected": [
            "Python is a snake.",
            "ML is hard."
        ]
    }
    return Dataset.from_dict(data)


class TestRewardModelDataset:
    """Test RewardModelDataset class."""

    def test_dataset_creation(self, reward_dataset, tokenizer):
        """Test creating a reward model dataset."""
        dataset = RewardModelDataset(
            dataset=reward_dataset,
            tokenizer=tokenizer,
            max_length=128
        )

        assert len(dataset) == 2

    def test_getitem_returns_correct_keys(self, reward_dataset, tokenizer):
        """Test that __getitem__ returns all required keys."""
        dataset = RewardModelDataset(
            dataset=reward_dataset,
            tokenizer=tokenizer,
            max_length=128
        )

        sample = dataset[0]

        assert "chosen_input_ids" in sample
        assert "chosen_attention_mask" in sample
        assert "rejected_input_ids" in sample
        assert "rejected_attention_mask" in sample

    def test_tensor_shapes(self, reward_dataset, tokenizer):
        """Test that tensors have correct shapes."""
        max_length = 128

        dataset = RewardModelDataset(
            dataset=reward_dataset,
            tokenizer=tokenizer,
            max_length=max_length
        )

        sample = dataset[0]

        assert sample["chosen_input_ids"].shape == (max_length,)
        assert sample["chosen_attention_mask"].shape == (max_length,)
        assert sample["rejected_input_ids"].shape == (max_length,)
        assert sample["rejected_attention_mask"].shape == (max_length,)

    def test_tensor_dtypes(self, reward_dataset, tokenizer):
        """Test that tensors have correct dtypes."""
        dataset = RewardModelDataset(
            dataset=reward_dataset,
            tokenizer=tokenizer,
            max_length=128
        )

        sample = dataset[0]

        assert sample["chosen_input_ids"].dtype == torch.long
        assert sample["chosen_attention_mask"].dtype == torch.long
        assert sample["rejected_input_ids"].dtype == torch.long
        assert sample["rejected_attention_mask"].dtype == torch.long

    def test_padding(self, reward_dataset, tokenizer):
        """Test that sequences are padded correctly."""
        dataset = RewardModelDataset(
            dataset=reward_dataset,
            tokenizer=tokenizer,
            max_length=128
        )

        sample = dataset[0]

        # Should have padding
        assert (sample["chosen_attention_mask"] == 0).any()

        # Padded positions should have pad_token_id
        chosen_pad_positions = sample["chosen_attention_mask"] == 0
        assert (sample["chosen_input_ids"][chosen_pad_positions] == tokenizer.pad_token_id).all()

    def test_chosen_rejected_different(self, reward_dataset, tokenizer):
        """Test that chosen and rejected are different."""
        dataset = RewardModelDataset(
            dataset=reward_dataset,
            tokenizer=tokenizer,
            max_length=128
        )

        sample = dataset[0]

        # Should be different
        assert not torch.equal(sample["chosen_input_ids"], sample["rejected_input_ids"])

    def test_prompt_in_sequences(self, reward_dataset, tokenizer):
        """Test that prompt is included in sequences."""
        dataset = RewardModelDataset(
            dataset=reward_dataset,
            tokenizer=tokenizer,
            max_length=256
        )

        sample = dataset[0]

        # Decode to check
        chosen_text = tokenizer.decode(sample["chosen_input_ids"], skip_special_tokens=True)
        rejected_text = tokenizer.decode(sample["rejected_input_ids"], skip_special_tokens=True)

        # Both should contain or start with prompt content
        assert "Python" in chosen_text
        assert "Python" in rejected_text

    def test_tokenizer_pad_token_set(self, reward_dataset):
        """Test that pad token is set if missing."""
        tokenizer = AutoTokenizer.from_pretrained("gpt2")

        dataset = RewardModelDataset(
            dataset=reward_dataset,
            tokenizer=tokenizer,
            max_length=128
        )

        # Pad token should be set
        assert dataset.tokenizer.pad_token is not None

    def test_batch_loading(self, reward_dataset, tokenizer):
        """Test loading batches."""
        from torch.utils.data import DataLoader

        dataset = RewardModelDataset(
            dataset=reward_dataset,
            tokenizer=tokenizer,
            max_length=128
        )

        dataloader = DataLoader(dataset, batch_size=2)
        batch = next(iter(dataloader))

        assert batch["chosen_input_ids"].shape == (2, 128)
        assert batch["rejected_input_ids"].shape == (2, 128)

    def test_max_prompt_length(self, reward_dataset, tokenizer):
        """Test max_prompt_length parameter."""
        dataset = RewardModelDataset(
            dataset=reward_dataset,
            tokenizer=tokenizer,
            max_length=256,
            max_prompt_length=64
        )

        # Should create dataset successfully
        assert len(dataset) == 2
        sample = dataset[0]
        assert sample["chosen_input_ids"].shape == (256,)

    def test_truncation(self, tokenizer):
        """Test that long sequences are truncated."""
        # Create very long text
        long_text = "word " * 200

        data = {
            "prompt": ["Short prompt"],
            "chosen": [long_text],
            "rejected": [long_text]
        }
        dataset_raw = Dataset.from_dict(data)

        dataset = RewardModelDataset(
            dataset=dataset_raw,
            tokenizer=tokenizer,
            max_length=128
        )

        sample = dataset[0]

        # Should be truncated to max_length
        assert sample["chosen_input_ids"].shape == (128,)
        # Should have actual content (not all padding)
        assert (sample["chosen_attention_mask"] == 1).sum() > 0
