"""
Tests for DPO dataset utilities.
"""

import pytest
import torch
from datasets import Dataset
from transformers import AutoTokenizer

from src.auto_bot_tuner.dpo.dataset import (
    PreferenceDataset,
    ConversationalPreferenceDataset,
    convert_reward_dataset_to_preference,
    preview_preference_sample,
    validate_preference_dataset
)


@pytest.fixture(scope="function")
def tokenizer(small_vocab_tokenizer):
    """Load a small tokenizer for testing."""
    return AutoTokenizer.from_pretrained("gpt2")


@pytest.fixture(scope="function")
def preference_dataset():
    """Create a small preference dataset."""
    data = {
        "prompt": [
            "What is Python?",
            "Explain machine learning"
        ],
        "chosen": [
            "Python is a high-level programming language known for its simplicity.",
            "Machine learning is a subset of AI that enables systems to learn from data."
        ],
        "rejected": [
            "Python is a snake.",
            "Machine learning is hard."
        ]
    }
    return Dataset.from_dict(data)


@pytest.fixture(scope="function")
def conversational_dataset():
    """Create a conversational preference dataset."""
    data = {
        "prompt": [
            [
                {"role": "user", "content": "Hello!"},
            ],
            [
                {"role": "user", "content": "What is AI?"},
            ]
        ],
        "chosen": [
            [
                {"role": "assistant", "content": "Hi! How can I help you today?"}
            ],
            [
                {"role": "assistant", "content": "AI stands for Artificial Intelligence."}
            ]
        ],
        "rejected": [
            [
                {"role": "assistant", "content": "Hi."}
            ],
            [
                {"role": "assistant", "content": "AI is computers."}
            ]
        ]
    }
    return Dataset.from_dict(data)


@pytest.fixture(scope="function")
def reward_dataset():
    """Create a reward modeling dataset."""
    data = {
        "prompt": [
            "What is Python?",
            "What is Python?",
            "Explain AI",
            "Explain AI"
        ],
        "response": [
            "Python is a programming language.",
            "Python is a snake.",
            "AI is artificial intelligence.",
            "AI is bad."
        ],
        "score": [
            5.0,
            1.0,
            4.5,
            1.5
        ]
    }
    return Dataset.from_dict(data)


class TestPreferenceDataset:
    """Test PreferenceDataset class."""

    def test_dataset_creation(self, preference_dataset, tokenizer):
        """Test creating a preference dataset."""
        dataset = PreferenceDataset(
            dataset=preference_dataset,
            tokenizer=tokenizer,
            max_length=128
        )

        assert len(dataset) == 2

    def test_getitem_returns_correct_keys(self, preference_dataset, tokenizer):
        """Test that __getitem__ returns all required keys."""
        dataset = PreferenceDataset(
            dataset=preference_dataset,
            tokenizer=tokenizer,
            max_length=128
        )

        sample = dataset[0]

        assert "prompt_input_ids" in sample
        assert "prompt_attention_mask" in sample
        assert "chosen_input_ids" in sample
        assert "chosen_attention_mask" in sample
        assert "rejected_input_ids" in sample
        assert "rejected_attention_mask" in sample

    def test_tensor_shapes(self, preference_dataset, tokenizer):
        """Test that tensors have correct shapes."""
        max_length = 128
        max_prompt_length = 64

        dataset = PreferenceDataset(
            dataset=preference_dataset,
            tokenizer=tokenizer,
            max_length=max_length,
            max_prompt_length=max_prompt_length
        )

        sample = dataset[0]

        # Prompt should be max_prompt_length
        assert sample["prompt_input_ids"].shape == (max_prompt_length,)
        assert sample["prompt_attention_mask"].shape == (max_prompt_length,)

        # Chosen/rejected should be max_length
        assert sample["chosen_input_ids"].shape == (max_length,)
        assert sample["chosen_attention_mask"].shape == (max_length,)
        assert sample["rejected_input_ids"].shape == (max_length,)
        assert sample["rejected_attention_mask"].shape == (max_length,)

    def test_tensor_dtypes(self, preference_dataset, tokenizer):
        """Test that tensors have correct dtypes."""
        dataset = PreferenceDataset(
            dataset=preference_dataset,
            tokenizer=tokenizer,
            max_length=128
        )

        sample = dataset[0]

        # All tensors should be long (int64)
        assert sample["prompt_input_ids"].dtype == torch.long
        assert sample["prompt_attention_mask"].dtype == torch.long
        assert sample["chosen_input_ids"].dtype == torch.long
        assert sample["chosen_attention_mask"].dtype == torch.long
        assert sample["rejected_input_ids"].dtype == torch.long
        assert sample["rejected_attention_mask"].dtype == torch.long

    def test_padding(self, preference_dataset, tokenizer):
        """Test that sequences are padded correctly."""
        dataset = PreferenceDataset(
            dataset=preference_dataset,
            tokenizer=tokenizer,
            max_length=128
        )

        sample = dataset[0]

        # Should have padding tokens
        assert (sample["chosen_attention_mask"] == 0).any()
        assert (sample["rejected_attention_mask"] == 0).any()

        # Padded positions should have pad_token_id
        chosen_pad_positions = sample["chosen_attention_mask"] == 0
        assert (sample["chosen_input_ids"][chosen_pad_positions] == tokenizer.pad_token_id).all()

    def test_chosen_rejected_different(self, preference_dataset, tokenizer):
        """Test that chosen and rejected are different."""
        dataset = PreferenceDataset(
            dataset=preference_dataset,
            tokenizer=tokenizer,
            max_length=128
        )

        sample = dataset[0]

        # Chosen and rejected should be different
        assert not torch.equal(sample["chosen_input_ids"], sample["rejected_input_ids"])

    def test_prompt_in_both_sequences(self, preference_dataset, tokenizer):
        """Test that prompt appears in both chosen and rejected sequences."""
        dataset = PreferenceDataset(
            dataset=preference_dataset,
            tokenizer=tokenizer,
            max_length=512
        )

        sample = dataset[0]

        # Decode to verify prompt is included
        prompt_text = tokenizer.decode(sample["prompt_input_ids"], skip_special_tokens=True)
        chosen_text = tokenizer.decode(sample["chosen_input_ids"], skip_special_tokens=True)
        rejected_text = tokenizer.decode(sample["rejected_input_ids"], skip_special_tokens=True)

        # Both chosen and rejected should start with or contain the prompt
        assert prompt_text.strip() in chosen_text or chosen_text.startswith(prompt_text.strip())
        assert prompt_text.strip() in rejected_text or rejected_text.startswith(prompt_text.strip())


class TestConversationalPreferenceDataset:
    """Test ConversationalPreferenceDataset class."""

    def test_dataset_creation(self, conversational_dataset, tokenizer):
        """Test creating conversational preference dataset."""
        dataset = ConversationalPreferenceDataset(
            dataset=conversational_dataset,
            tokenizer=tokenizer,
            max_length=128
        )

        assert len(dataset) == 2

    def test_conversational_format(self, conversational_dataset, tokenizer):
        """Test that conversational format is handled correctly."""
        dataset = ConversationalPreferenceDataset(
            dataset=conversational_dataset,
            tokenizer=tokenizer,
            max_length=256
        )

        sample = dataset[0]

        # Should have all required keys
        assert "chosen_input_ids" in sample
        assert "chosen_attention_mask" in sample
        assert "rejected_input_ids" in sample
        assert "rejected_attention_mask" in sample
        assert "prompt_length" in sample

        # Prompt length should be positive
        assert sample["prompt_length"].item() > 0

    def test_message_formatting(self, conversational_dataset, tokenizer):
        """Test that messages are formatted correctly."""
        dataset = ConversationalPreferenceDataset(
            dataset=conversational_dataset,
            tokenizer=tokenizer,
            max_length=256
        )

        sample = dataset[0]

        chosen_text = tokenizer.decode(sample["chosen_input_ids"], skip_special_tokens=True)

        # Should contain role labels
        assert "User:" in chosen_text or "user" in chosen_text.lower()

    def test_simple_string_format(self, preference_dataset, tokenizer):
        """Test that simple string format also works."""
        dataset = ConversationalPreferenceDataset(
            dataset=preference_dataset,
            tokenizer=tokenizer,
            max_length=256
        )

        sample = dataset[0]

        # Should work with simple string format too
        assert sample["chosen_input_ids"].shape == (256,)


class TestConvertRewardDatasetToPreference:
    """Test convert_reward_dataset_to_preference function."""

    def test_conversion(self, reward_dataset):
        """Test basic conversion from reward to preference format."""
        preference_dataset = convert_reward_dataset_to_preference(reward_dataset)

        # Should create preference pairs
        assert len(preference_dataset) > 0

        # Check required columns exist
        assert "prompt" in preference_dataset.column_names
        assert "chosen" in preference_dataset.column_names
        assert "rejected" in preference_dataset.column_names

    def test_correct_pairing(self, reward_dataset):
        """Test that higher scored responses become chosen."""
        preference_dataset = convert_reward_dataset_to_preference(reward_dataset)

        # For "What is Python?" prompt:
        # Score 5.0: "Python is a programming language."
        # Score 1.0: "Python is a snake."
        # So chosen should be the programming language one

        for item in preference_dataset:
            if "Python" in item["prompt"]:
                # Chosen should be longer/better response
                assert len(item["chosen"]) > len(item["rejected"])

    def test_handles_single_response_prompts(self):
        """Test that prompts with only one response are skipped."""
        data = {
            "prompt": ["Question 1", "Question 2"],
            "response": ["Answer 1", "Answer 2"],
            "score": [5.0, 3.0]
        }
        reward_dataset = Dataset.from_dict(data)

        preference_dataset = convert_reward_dataset_to_preference(reward_dataset)

        # Should have no pairs since each prompt appears only once
        assert len(preference_dataset) == 0


class TestPreviewPreferenceSample:
    """Test preview_preference_sample function."""

    def test_preview(self, preference_dataset, tokenizer):
        """Test previewing a preference sample."""
        dataset = PreferenceDataset(
            dataset=preference_dataset,
            tokenizer=tokenizer,
            max_length=128
        )

        preview = preview_preference_sample(dataset, idx=0)

        assert "prompt" in preview
        assert "chosen_full" in preview
        assert "rejected_full" in preview
        assert "chosen_length" in preview
        assert "rejected_length" in preview

        # Lengths should be positive
        assert preview["chosen_length"] > 0
        assert preview["rejected_length"] > 0

        # Texts should be strings
        assert isinstance(preview["prompt"], str)
        assert isinstance(preview["chosen_full"], str)
        assert isinstance(preview["rejected_full"], str)

    def test_preview_shows_different_responses(self, preference_dataset, tokenizer):
        """Test that preview shows different chosen/rejected."""
        dataset = PreferenceDataset(
            dataset=preference_dataset,
            tokenizer=tokenizer,
            max_length=256
        )

        preview = preview_preference_sample(dataset, idx=0)

        # Chosen and rejected should be different
        assert preview["chosen_full"] != preview["rejected_full"]


class TestValidatePreferenceDataset:
    """Test validate_preference_dataset function."""

    def test_valid_dataset(self, preference_dataset):
        """Test validation passes for valid dataset."""
        # Should not raise
        assert validate_preference_dataset(preference_dataset) is True

    def test_missing_prompt_column(self):
        """Test validation fails when prompt column is missing."""
        data = {
            "chosen": ["Response 1"],
            "rejected": ["Response 2"]
        }
        dataset = Dataset.from_dict(data)

        with pytest.raises(ValueError, match="missing required columns"):
            validate_preference_dataset(dataset)

    def test_missing_chosen_column(self):
        """Test validation fails when chosen column is missing."""
        data = {
            "prompt": ["Question 1"],
            "rejected": ["Response 2"]
        }
        dataset = Dataset.from_dict(data)

        with pytest.raises(ValueError, match="missing required columns"):
            validate_preference_dataset(dataset)

    def test_missing_rejected_column(self):
        """Test validation fails when rejected column is missing."""
        data = {
            "prompt": ["Question 1"],
            "chosen": ["Response 1"]
        }
        dataset = Dataset.from_dict(data)

        with pytest.raises(ValueError, match="missing required columns"):
            validate_preference_dataset(dataset)

    def test_error_message_shows_missing_columns(self):
        """Test that error message lists missing columns."""
        data = {"other": ["data"]}
        dataset = Dataset.from_dict(data)

        with pytest.raises(ValueError) as exc_info:
            validate_preference_dataset(dataset)

        error_msg = str(exc_info.value)
        assert "prompt" in error_msg
        assert "chosen" in error_msg
        assert "rejected" in error_msg


class TestDatasetIntegration:
    """Integration tests for dataset operations."""

    def test_batch_loading(self, preference_dataset, tokenizer):
        """Test loading batches from dataset."""
        from torch.utils.data import DataLoader

        dataset = PreferenceDataset(
            dataset=preference_dataset,
            tokenizer=tokenizer,
            max_length=128
        )

        dataloader = DataLoader(dataset, batch_size=2, shuffle=False)

        batch = next(iter(dataloader))

        # Check batch dimensions
        assert batch["chosen_input_ids"].shape == (2, 128)
        assert batch["rejected_input_ids"].shape == (2, 128)

    def test_tokenizer_pad_token_set(self, preference_dataset):
        """Test that pad token is set automatically if missing."""
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        # GPT-2 doesn't have pad_token by default

        dataset = PreferenceDataset(
            dataset=preference_dataset,
            tokenizer=tokenizer,
            max_length=128
        )

        # Pad token should be set to eos_token
        assert dataset.tokenizer.pad_token is not None
        assert dataset.tokenizer.pad_token == dataset.tokenizer.eos_token
