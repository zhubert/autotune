"""
Tests for evaluation metrics.
"""

import pytest
import torch
from transformers import GPT2LMHeadModel, GPT2Config, GPT2Tokenizer
from datasets import Dataset

from src.auto_bot_tuner.evaluation.metrics import (
    compute_perplexity,
    compute_perplexity_on_dataset,
    compute_diversity_metrics,
    compute_model_size_info
)


@pytest.fixture(scope="function")
def small_gpt2_model():
    """Create a small GPT-2 model for testing."""
    config = GPT2Config(
        vocab_size=100,
        n_positions=128,
        n_embd=64,
        n_layer=2,
        n_head=2
    )
    model = GPT2LMHeadModel(config)
    return model


@pytest.fixture(scope="function")
def tokenizer(small_vocab_tokenizer):
    """Create a tokenizer for testing."""
    tokenizer = small_vocab_tokenizer
    
    return small_vocab_tokenizer


@pytest.fixture(scope="function")
def device():
    """Get device for testing."""
    return torch.device("cpu")


class TestComputePerplexity:
    """Test compute_perplexity function."""

    def test_basic_perplexity(self, small_gpt2_model, device):
        """Test basic perplexity computation."""
        model = small_gpt2_model.to(device)
        batch_size = 2
        seq_len = 10

        input_ids = torch.randint(0, 100, (batch_size, seq_len)).to(device)

        ppl = compute_perplexity(model, input_ids)

        assert isinstance(ppl, float)
        assert ppl > 0
        assert not torch.isnan(torch.tensor(ppl))

    def test_perplexity_with_attention_mask(self, small_gpt2_model, device):
        """Test perplexity with attention mask."""
        model = small_gpt2_model.to(device)
        batch_size = 2
        seq_len = 10

        input_ids = torch.randint(0, 100, (batch_size, seq_len)).to(device)
        attention_mask = torch.ones(batch_size, seq_len).to(device)
        # Mask last 2 tokens
        attention_mask[:, -2:] = 0

        ppl = compute_perplexity(model, input_ids, attention_mask=attention_mask)

        assert isinstance(ppl, float)
        assert ppl > 0

    def test_perplexity_with_labels(self, small_gpt2_model, device):
        """Test perplexity with custom labels."""
        model = small_gpt2_model.to(device)
        batch_size = 2
        seq_len = 10

        input_ids = torch.randint(0, 100, (batch_size, seq_len)).to(device)
        labels = torch.randint(0, 100, (batch_size, seq_len)).to(device)

        ppl = compute_perplexity(model, input_ids, labels=labels)

        assert isinstance(ppl, float)
        assert ppl > 0

    def test_perfect_model_low_perplexity(self, small_gpt2_model, device):
        """Test that a perfectly fit model has low perplexity."""
        model = small_gpt2_model.to(device)

        # Create a simple repeating sequence
        input_ids = torch.tensor([[1, 2, 3, 1, 2, 3, 1, 2, 3, 1]]).to(device)

        # Get initial perplexity
        ppl_before = compute_perplexity(model, input_ids)

        # Train for a few steps to fit this sequence
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        for _ in range(50):
            outputs = model(input_ids, labels=input_ids)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # Perplexity should decrease after training
        ppl_after = compute_perplexity(model, input_ids)
        assert ppl_after < ppl_before


class TestComputePerplexityOnDataset:
    """Test compute_perplexity_on_dataset function."""

    def test_dataset_perplexity(self, tokenizer, device):
        """Test perplexity computation on dataset."""
        # Use real GPT-2 for this test to match tokenizer
        model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)

        # Create a small text dataset
        data = {
            "text": [
                "Hello world",
                "This is a test",
                "Machine learning is fun"
            ]
        }
        dataset = Dataset.from_dict(data)

        metrics = compute_perplexity_on_dataset(
            model,
            dataset,
            tokenizer,
            batch_size=2,
            max_length=32,
            device=device
        )

        assert "perplexity" in metrics
        assert "loss" in metrics
        assert "total_tokens" in metrics
        assert "num_samples" in metrics

        assert metrics["perplexity"] > 0
        assert metrics["total_tokens"] > 0
        assert metrics["num_samples"] == 3

    def test_dataset_perplexity_matches_batch(self, tokenizer, device):
        """Test that different batch sizes give similar results."""
        # Use real GPT-2 for this test to match tokenizer
        model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)

        data = {
            "text": ["Test sentence " + str(i) for i in range(10)]
        }
        dataset = Dataset.from_dict(data)

        metrics_bs2 = compute_perplexity_on_dataset(
            model, dataset, tokenizer, batch_size=2, max_length=32, device=device
        )

        metrics_bs5 = compute_perplexity_on_dataset(
            model, dataset, tokenizer, batch_size=5, max_length=32, device=device
        )

        # Should be very close (may have small numerical differences)
        assert abs(metrics_bs2["perplexity"] - metrics_bs5["perplexity"]) < 0.5


class TestComputeDiversityMetrics:
    """Test compute_diversity_metrics function."""

    def test_basic_diversity(self):
        """Test basic diversity computation."""
        texts = [
            "hello world this is a test",
            "another test sentence here",
            "more diverse text content"
        ]

        metrics = compute_diversity_metrics(texts)

        assert "distinct_1" in metrics
        assert "distinct_2" in metrics
        assert "avg_length" in metrics
        assert "unique_unigrams" in metrics
        assert "unique_bigrams" in metrics

        # All metrics should be positive
        assert metrics["distinct_1"] > 0
        assert metrics["distinct_2"] > 0
        assert metrics["avg_length"] > 0
        assert metrics["unique_unigrams"] > 0

    def test_repetitive_text_low_diversity(self):
        """Test that repetitive text has low diversity."""
        repetitive = [
            "test test test test",
            "test test test test",
            "test test test test"
        ]

        diverse = [
            "completely unique sentence one",
            "another different phrase here",
            "totally distinct content now"
        ]

        rep_metrics = compute_diversity_metrics(repetitive)
        div_metrics = compute_diversity_metrics(diverse)

        # Diverse text should have higher distinct-1
        assert div_metrics["distinct_1"] > rep_metrics["distinct_1"]

    def test_distinct_1_range(self):
        """Test that distinct-1 is between 0 and 1."""
        texts = ["hello world", "test sentence", "more text"]

        metrics = compute_diversity_metrics(texts)

        assert 0.0 <= metrics["distinct_1"] <= 1.0
        assert 0.0 <= metrics["distinct_2"] <= 1.0

    def test_empty_list(self):
        """Test with empty list."""
        metrics = compute_diversity_metrics([])

        assert metrics["distinct_1"] == 0
        assert metrics["distinct_2"] == 0
        assert metrics["avg_length"] == 0

    def test_single_word_texts(self):
        """Test with single-word texts (no bigrams)."""
        texts = ["hello", "world", "test"]

        metrics = compute_diversity_metrics(texts)

        # Should still compute distinct-1
        assert metrics["distinct_1"] == 1.0  # All unique
        # Distinct-2 should be 0 (no bigrams)
        assert metrics["distinct_2"] == 0

    def test_average_length(self):
        """Test average length calculation."""
        texts = [
            "one two three",  # 3 words
            "four five",       # 2 words
            "six"              # 1 word
        ]

        metrics = compute_diversity_metrics(texts)

        # Average should be (3 + 2 + 1) / 3 = 2.0
        assert metrics["avg_length"] == 2.0


class TestComputeModelSizeInfo:
    """Test compute_model_size_info function."""

    def test_basic_model_info(self, small_gpt2_model):
        """Test basic model size computation."""
        info = compute_model_size_info(small_gpt2_model)

        assert "total_parameters" in info
        assert "trainable_parameters" in info
        assert "frozen_parameters" in info
        assert "trainable_percentage" in info

        # All parameters should be trainable initially
        assert info["total_parameters"] > 0
        assert info["trainable_parameters"] == info["total_parameters"]
        assert info["frozen_parameters"] == 0
        assert info["trainable_percentage"] == 100.0

    def test_frozen_model(self, small_gpt2_model):
        """Test model info with frozen parameters."""
        # Freeze all parameters
        for param in small_gpt2_model.parameters():
            param.requires_grad = False

        info = compute_model_size_info(small_gpt2_model)

        assert info["trainable_parameters"] == 0
        assert info["frozen_parameters"] == info["total_parameters"]
        assert info["trainable_percentage"] == 0.0

    def test_partially_frozen_model(self, small_gpt2_model):
        """Test model with some frozen parameters."""
        # Freeze half the parameters
        params = list(small_gpt2_model.parameters())
        for i, param in enumerate(params):
            if i < len(params) // 2:
                param.requires_grad = False

        info = compute_model_size_info(small_gpt2_model)

        # Should have both trainable and frozen
        assert info["trainable_parameters"] > 0
        assert info["frozen_parameters"] > 0
        assert info["trainable_parameters"] + info["frozen_parameters"] == info["total_parameters"]
        assert 0 < info["trainable_percentage"] < 100

    def test_parameter_count_correctness(self):
        """Test that parameter count is correct."""
        # Simple model with known parameter count
        model = torch.nn.Linear(10, 5)
        # Parameters: weight (10x5=50) + bias (5) = 55

        info = compute_model_size_info(model)

        assert info["total_parameters"] == 55
        assert info["trainable_parameters"] == 55


class TestEvaluationIntegration:
    """Integration tests for evaluation metrics."""

    def test_full_evaluation_pipeline(self, tokenizer, device):
        """Test a complete evaluation pipeline."""
        # Use real GPT-2 for this test to match tokenizer
        model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)

        # Create dataset
        data = {"text": ["Test sentence " + str(i) for i in range(5)]}
        dataset = Dataset.from_dict(data)

        # Compute perplexity
        ppl_metrics = compute_perplexity_on_dataset(
            model, dataset, tokenizer, batch_size=2, max_length=32, device=device
        )

        # Get model size
        size_info = compute_model_size_info(model)

        # Verify results
        assert ppl_metrics["perplexity"] > 0
        assert size_info["total_parameters"] > 0

    def test_diversity_on_model_generations(self):
        """Test diversity metrics on a set of generations."""
        # Simulate model generations
        generations = [
            "The model generated this text",
            "Another unique generated response",
            "Different output from model",
            "Various text samples here"
        ]

        metrics = compute_diversity_metrics(generations)

        # Should have reasonable diversity
        assert metrics["distinct_1"] > 0.5
        assert metrics["unique_unigrams"] > 10
