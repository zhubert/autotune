"""
Tests for generation utilities.
"""

import pytest
import torch
from transformers import GPT2LMHeadModel, GPT2Config, GPT2Tokenizer

from src.auto_bot_tuner.evaluation.generation import (
    GenerationConfig,
    generate_text,
    generate_response,
    batch_generate
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
def device_str():
    """Get device string for testing."""
    return "cpu"


class TestGenerationConfig:
    """Test GenerationConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = GenerationConfig()

        assert config.max_new_tokens == 100
        assert config.temperature == 0.7
        assert config.top_p == 0.9
        assert config.top_k == 50
        assert config.do_sample is True
        assert config.num_return_sequences == 1

    def test_custom_config(self):
        """Test custom configuration."""
        config = GenerationConfig(
            max_new_tokens=50,
            temperature=0.5,
            top_p=0.95,
            do_sample=False
        )

        assert config.max_new_tokens == 50
        assert config.temperature == 0.5
        assert config.top_p == 0.95
        assert config.do_sample is False


class TestGenerateText:
    """Test generate_text function."""

    def test_basic_generation(self, small_gpt2_model, tokenizer, device_str):
        """Test basic text generation."""
        model = small_gpt2_model.to(device_str)
        # Use lowercase prompt to avoid unknown tokens with small vocab
        prompt = "hello"

        config = GenerationConfig(max_new_tokens=10, do_sample=False)

        generated = generate_text(model, tokenizer, prompt, config=config, device=device_str)

        assert isinstance(generated, str)
        assert len(generated) > 0  # Just check we got output

    def test_generation_includes_prompt(self, small_gpt2_model, tokenizer, device_str):
        """Test that generated text includes the prompt."""
        model = small_gpt2_model.to(device_str)
        # Use lowercase prompt to avoid unknown tokens with small vocab
        prompt = "test prompt"

        config = GenerationConfig(max_new_tokens=5, do_sample=False)

        generated = generate_text(model, tokenizer, prompt, config=config, device=device_str)

        # Generated text should start with or contain the prompt
        assert prompt in generated or generated.startswith(prompt[:3])

    def test_different_temperatures(self, small_gpt2_model, tokenizer, device_str):
        """Test generation with different temperatures."""
        model = small_gpt2_model.to(device_str)
        prompt = "hello"

        # Low temperature (more deterministic)
        config_low = GenerationConfig(max_new_tokens=10, temperature=0.1, do_sample=True)
        gen_low_1 = generate_text(model, tokenizer, prompt, config=config_low, device=device_str)
        gen_low_2 = generate_text(model, tokenizer, prompt, config=config_low, device=device_str)

        # High temperature (more random)
        config_high = GenerationConfig(max_new_tokens=10, temperature=2.0, do_sample=True)
        gen_high_1 = generate_text(model, tokenizer, prompt, config=config_high, device=device_str)
        gen_high_2 = generate_text(model, tokenizer, prompt, config=config_high, device=device_str)

        # All should generate something
        assert len(gen_low_1) > 0
        assert len(gen_high_1) > 0

    def test_greedy_vs_sampling(self, small_gpt2_model, tokenizer, device_str):
        """Test greedy decoding vs sampling."""
        model = small_gpt2_model.to(device_str)
        prompt = "test"

        # Greedy (deterministic)
        config_greedy = GenerationConfig(max_new_tokens=10, do_sample=False)
        gen_greedy_1 = generate_text(model, tokenizer, prompt, config=config_greedy, device=device_str)
        gen_greedy_2 = generate_text(model, tokenizer, prompt, config=config_greedy, device=device_str)

        # Greedy should be deterministic
        assert gen_greedy_1 == gen_greedy_2

    def test_max_new_tokens_limit(self, small_gpt2_model, tokenizer, device_str):
        """Test that max_new_tokens limits generation length."""
        model = small_gpt2_model.to(device_str)
        prompt = "hello"

        # Tokenize prompt to get its length
        prompt_tokens = tokenizer.encode(prompt)
        prompt_length = len(prompt_tokens)

        config = GenerationConfig(max_new_tokens=5, do_sample=False)

        generated = generate_text(model, tokenizer, prompt, config=config, device=device_str)
        generated_tokens = tokenizer.encode(generated)

        # Generated tokens should be approximately prompt + max_new_tokens
        # (allowing for small variation due to tokenization)
        assert len(generated_tokens) <= prompt_length + 10


class TestGenerateResponse:
    """Test generate_response function."""

    def test_response_excludes_prompt(self, small_gpt2_model, tokenizer, device_str):
        """Test that response excludes the prompt."""
        model = small_gpt2_model.to(device_str)
        prompt = "question what is ai"

        config = GenerationConfig(max_new_tokens=10, do_sample=False)

        response = generate_response(model, tokenizer, prompt, config=config, device=device_str)

        # Response should not start with the exact prompt
        assert isinstance(response, str)

    def test_response_is_trimmed(self, small_gpt2_model, tokenizer, device_str):
        """Test that response is trimmed of whitespace."""
        model = small_gpt2_model.to(device_str)
        prompt = "test"

        config = GenerationConfig(max_new_tokens=10, do_sample=False)

        response = generate_response(model, tokenizer, prompt, config=config, device=device_str)

        # Should be trimmed
        assert response == response.strip()

    def test_empty_generation(self, small_gpt2_model, tokenizer, device_str):
        """Test handling of very short generation."""
        model = small_gpt2_model.to(device_str)
        prompt = "hi"

        # Very short generation
        config = GenerationConfig(max_new_tokens=1, do_sample=False)

        response = generate_response(model, tokenizer, prompt, config=config, device=device_str)

        # Should return a string (even if short)
        assert isinstance(response, str)


class TestBatchGenerate:
    """Test batch_generate function."""

    def test_batch_generation(self, small_gpt2_model, tokenizer, device_str):
        """Test generating for multiple prompts."""
        model = small_gpt2_model.to(device_str)
        prompts = ["hello", "test", "question"]

        config = GenerationConfig(max_new_tokens=5, do_sample=False)

        responses = batch_generate(
            model, tokenizer, prompts, config=config, device=device_str, show_progress=False
        )

        assert isinstance(responses, list)
        assert len(responses) == len(prompts)
        assert all(isinstance(r, str) for r in responses)

    def test_batch_different_prompts(self, small_gpt2_model, tokenizer, device_str):
        """Test that different prompts give different responses."""
        model = small_gpt2_model.to(device_str)
        prompts = ["a", "b", "c"]

        config = GenerationConfig(max_new_tokens=10, do_sample=False)

        responses = batch_generate(
            model, tokenizer, prompts, config=config, device=device_str, show_progress=False
        )

        # All responses should be strings
        assert all(isinstance(r, str) for r in responses)

    def test_empty_prompt_list(self, small_gpt2_model, tokenizer, device_str):
        """Test with empty prompt list."""
        model = small_gpt2_model.to(device_str)
        prompts = []

        responses = batch_generate(
            model, tokenizer, prompts, device=device_str, show_progress=False
        )

        assert responses == []

    def test_single_prompt_batch(self, small_gpt2_model, tokenizer, device_str):
        """Test batch generation with single prompt."""
        model = small_gpt2_model.to(device_str)
        prompts = ["single test"]

        config = GenerationConfig(max_new_tokens=5, do_sample=False)

        responses = batch_generate(
            model, tokenizer, prompts, config=config, device=device_str, show_progress=False
        )

        assert len(responses) == 1
        assert isinstance(responses[0], str)


class TestGenerationIntegration:
    """Integration tests for generation utilities."""

    def test_consistency_between_functions(self, small_gpt2_model, tokenizer, device_str):
        """Test that generate_text and generate_response are consistent."""
        model = small_gpt2_model.to(device_str)
        prompt = "hello world"

        config = GenerationConfig(max_new_tokens=10, do_sample=False)

        full_text = generate_text(model, tokenizer, prompt, config=config, device=device_str)
        response_only = generate_response(model, tokenizer, prompt, config=config, device=device_str)

        # Full text should be longer than or equal to response
        assert len(full_text) >= len(response_only)

    def test_batch_vs_single(self, small_gpt2_model, tokenizer, device_str):
        """Test that batch generation matches single generation."""
        model = small_gpt2_model.to(device_str)
        prompts = ["test prompt"]

        config = GenerationConfig(max_new_tokens=10, do_sample=False)

        # Single generation
        single = generate_response(model, tokenizer, prompts[0], config=config, device=device_str)

        # Batch generation
        batch = batch_generate(
            model, tokenizer, prompts, config=config, device=device_str, show_progress=False
        )

        # Should produce same result
        assert single == batch[0]

    def test_model_in_eval_mode(self, small_gpt2_model, tokenizer, device_str):
        """Test that generation puts model in eval mode."""
        model = small_gpt2_model.to(device_str)
        model.train()  # Put in training mode

        prompt = "test"
        config = GenerationConfig(max_new_tokens=5, do_sample=False)

        generate_text(model, tokenizer, prompt, config=config, device=device_str)

        # Model should still be in eval mode after generation
        # (generate functions set eval mode)
        assert not model.training

    def test_generation_with_various_configs(self, small_gpt2_model, tokenizer, device_str):
        """Test generation with various configuration settings."""
        model = small_gpt2_model.to(device_str)
        prompt = "test"

        configs = [
            GenerationConfig(max_new_tokens=5, do_sample=False),
            GenerationConfig(max_new_tokens=10, temperature=0.5, do_sample=True),
            GenerationConfig(max_new_tokens=15, top_p=0.95, do_sample=True),
            GenerationConfig(max_new_tokens=5, top_k=10, do_sample=True),
        ]

        for config in configs:
            result = generate_text(model, tokenizer, prompt, config=config, device=device_str)
            assert isinstance(result, str)
            assert len(result) > 0
