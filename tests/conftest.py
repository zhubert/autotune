"""
Shared pytest fixtures for all tests.
"""

import pytest
import torch
from transformers import GPT2LMHeadModel, GPT2Config, PreTrainedTokenizerFast
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders


@pytest.fixture(scope="session")
def small_vocab_tokenizer():
    """
    Create a small BPE tokenizer with vocab_size=100 to match test models.
    This prevents index out of range errors and keeps tests fast.
    """
    # Create a simple BPE tokenizer with unknown token handling
    tokenizer = Tokenizer(models.BPE(unk_token='<unk>'))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

    # Train on a small corpus to get ~100 vocab
    trainer = trainers.BpeTrainer(
        vocab_size=100,
        special_tokens=["<pad>", "<eos>", "<bos>", "<unk>"]
    )

    # Simple training corpus
    corpus = [
        "hello world this is a test",
        "machine learning is fun",
        "python programming language",
        "artificial intelligence and deep learning",
        "natural language processing",
    ]

    tokenizer.train_from_iterator(corpus, trainer=trainer)

    # Add decoder for proper string conversion
    tokenizer.decoder = decoders.ByteLevel()

    # Wrap in HuggingFace tokenizer
    wrapped_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        pad_token="<pad>",
        eos_token="<eos>",
        bos_token="<bos>",
        unk_token="<unk>",
        model_max_length=128
    )

    return wrapped_tokenizer


@pytest.fixture
def small_gpt2_model():
    """Create a small GPT-2 model for fast testing (vocab=100)."""
    config = GPT2Config(
        vocab_size=100,
        n_positions=128,
        n_embd=64,
        n_layer=2,
        n_head=2
    )
    model = GPT2LMHeadModel(config)
    return model


@pytest.fixture
def device():
    """CPU device for testing."""
    return torch.device("cpu")
