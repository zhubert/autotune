"""
Evaluation utilities for trained models.

This module provides tools for evaluating language models after training,
including perplexity computation, generation quality metrics, and interactive
testing utilities.
"""

from .metrics import (
    compute_perplexity,
    compute_perplexity_on_dataset,
    evaluate_generation_quality,
    compare_model_outputs
)
from .generation import (
    generate_text,
    generate_response,
    batch_generate,
    GenerationConfig
)

__all__ = [
    "compute_perplexity",
    "compute_perplexity_on_dataset",
    "evaluate_generation_quality",
    "compare_model_outputs",
    "generate_text",
    "generate_response",
    "batch_generate",
    "GenerationConfig",
]
