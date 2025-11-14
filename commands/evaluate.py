"""
Model evaluation command implementation.

This module provides functions for evaluating models using various metrics.
"""

import sys
from pathlib import Path
from rich.console import Console

# Add src directory to Python path for imports
src_path = Path(__file__).parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

console = Console()


def evaluate_perplexity(
    model_path: str,
    dataset_name: str,
    dataset_config: str = None,
    max_samples: int = 100
):
    """
    Evaluate model perplexity on a dataset.

    Args:
        model_path: Model path or checkpoint
        dataset_name: HuggingFace dataset identifier
        dataset_config: Optional dataset configuration
        max_samples: Maximum samples to evaluate (0 for all)

    Returns:
        dict: Evaluation results with perplexity, loss, tokens, and samples
    """
    from src.auto_bot_tuner.utils.model_loading import load_model_and_tokenizer
    from src.auto_bot_tuner.evaluation import compute_perplexity_on_dataset
    from datasets import load_dataset

    console.print(f"\n[cyan]Loading model from {model_path}...[/cyan]")
    model, tokenizer, device = load_model_and_tokenizer(model_path, use_lora=False)
    console.print("[green]✓ Model loaded![/green]")

    console.print(f"\n[cyan]Loading dataset {dataset_name}...[/cyan]")
    if dataset_config:
        dataset = load_dataset(dataset_name, dataset_config, split="test")
    else:
        dataset = load_dataset(dataset_name, split="test")

    if max_samples > 0:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    console.print(f"[green]✓ Loaded {len(dataset)} samples[/green]")

    console.print("\n[cyan]Computing perplexity...[/cyan]")
    results = compute_perplexity_on_dataset(
        model=model,
        dataset=dataset,
        tokenizer=tokenizer,
        batch_size=8,
        max_length=512,
        text_column="text",
        device=device
    )

    console.print("\n[bold green]Results:[/bold green]")
    console.print(f"  Perplexity: {results['perplexity']:.2f}")
    console.print(f"  Loss: {results['loss']:.4f}")
    console.print(f"  Total tokens: {results['total_tokens']:,}")
    console.print(f"  Samples evaluated: {results['num_samples']}")

    return results


def test_generation_quality(
    model_path: str,
    prompts: list = None,
    max_new_tokens: int = 100,
    temperature: float = 0.7
):
    """
    Test generation quality on sample prompts.

    Args:
        model_path: Model path or checkpoint
        prompts: List of prompts to test (uses defaults if None)
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature

    Returns:
        list: Generated responses for each prompt
    """
    from src.auto_bot_tuner.utils.model_loading import load_model_and_tokenizer
    from src.auto_bot_tuner.evaluation import evaluate_generation_quality

    # Default prompts if none provided
    if prompts is None:
        prompts = [
            "Explain what machine learning is in simple terms:",
            "Write a short story about a robot:",
            "What are the benefits of exercise?",
            "How do you make a peanut butter sandwich?"
        ]

    console.print(f"\n[cyan]Loading model from {model_path}...[/cyan]")
    model, tokenizer, device = load_model_and_tokenizer(model_path, use_lora=False)
    console.print("[green]✓ Model loaded![/green]")

    console.print("\n[cyan]Generating responses...[/cyan]\n")
    results = evaluate_generation_quality(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        device=device,
        max_new_tokens=max_new_tokens,
        temperature=temperature
    )

    console.print("\n[bold green]Generated Responses:[/bold green]\n")
    for i, result in enumerate(results, 1):
        console.print(f"[bold]Prompt {i}:[/bold] {result['prompt']}")
        console.print(f"[green]Response:[/green] {result['response']}\n")
        console.print("-" * 60 + "\n")

    return results


def compare_models(
    model_a_path: str,
    model_b_path: str,
    prompts: list = None,
    max_new_tokens: int = 80
):
    """
    Compare outputs from two different models.

    Args:
        model_a_path: First model path
        model_b_path: Second model path
        prompts: List of prompts to test (uses defaults if None)
        max_new_tokens: Maximum tokens to generate

    Returns:
        list: Comparison results for each prompt
    """
    from src.auto_bot_tuner.utils.model_loading import load_model_and_tokenizer
    from src.auto_bot_tuner.evaluation import compare_model_outputs

    # Default prompts if none provided
    if prompts is None:
        prompts = [
            "What is the capital of France?",
            "Explain quantum computing:",
            "Write a haiku about programming:"
        ]

    console.print(f"\n[cyan]Loading first model from {model_a_path}...[/cyan]")
    model_a, tokenizer_a, device = load_model_and_tokenizer(model_a_path, use_lora=False)
    console.print("[green]✓ First model loaded![/green]")

    console.print(f"\n[cyan]Loading second model from {model_b_path}...[/cyan]")
    model_b, tokenizer_b, _ = load_model_and_tokenizer(model_b_path, use_lora=False)
    console.print("[green]✓ Second model loaded![/green]")

    console.print("\n[cyan]Generating comparisons...[/cyan]\n")
    results = compare_model_outputs(
        model_a=model_a,
        model_b=model_b,
        tokenizer=tokenizer_a,  # Assume same tokenizer
        prompts=prompts,
        model_a_name=model_a_path.split("/")[-1],
        model_b_name=model_b_path.split("/")[-1],
        device=device,
        max_new_tokens=max_new_tokens
    )

    console.print("\n[bold green]Model Comparison:[/bold green]\n")
    for i, result in enumerate(results, 1):
        console.print(f"[bold]Prompt {i}:[/bold] {result['prompt']}\n")

        model_a_name = model_a_path.split("/")[-1]
        model_b_name = model_b_path.split("/")[-1]

        console.print(f"[blue]{model_a_name}:[/blue]")
        console.print(f"  {result[f'{model_a_name}_response']}\n")

        console.print(f"[magenta]{model_b_name}:[/magenta]")
        console.print(f"  {result[f'{model_b_name}_response']}\n")

        console.print("-" * 60 + "\n")

    return results


def main():
    """Main entry point for evaluation command."""
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate trained models")
    subparsers = parser.add_subparsers(dest="command", help="Evaluation command")

    # Perplexity evaluation
    perplexity_parser = subparsers.add_parser("perplexity", help="Compute perplexity on dataset")
    perplexity_parser.add_argument("--model", type=str, required=True, help="Model path")
    perplexity_parser.add_argument("--dataset", type=str, required=True, help="Dataset name")
    perplexity_parser.add_argument("--config", type=str, help="Dataset config")
    perplexity_parser.add_argument("--max-samples", type=int, default=100, help="Max samples")

    # Generation quality
    generation_parser = subparsers.add_parser("generation", help="Test generation quality")
    generation_parser.add_argument("--model", type=str, required=True, help="Model path")
    generation_parser.add_argument("--prompt", type=str, action="append", help="Test prompt (can specify multiple)")
    generation_parser.add_argument("--max-tokens", type=int, default=100, help="Max new tokens")
    generation_parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")

    # Model comparison
    compare_parser = subparsers.add_parser("compare", help="Compare two models")
    compare_parser.add_argument("--model-a", type=str, required=True, help="First model path")
    compare_parser.add_argument("--model-b", type=str, required=True, help="Second model path")
    compare_parser.add_argument("--prompt", type=str, action="append", help="Test prompt (can specify multiple)")
    compare_parser.add_argument("--max-tokens", type=int, default=80, help="Max new tokens")

    args = parser.parse_args()

    try:
        if args.command == "perplexity":
            evaluate_perplexity(
                model_path=args.model,
                dataset_name=args.dataset,
                dataset_config=args.config,
                max_samples=args.max_samples
            )
        elif args.command == "generation":
            test_generation_quality(
                model_path=args.model,
                prompts=args.prompt,
                max_new_tokens=args.max_tokens,
                temperature=args.temperature
            )
        elif args.command == "compare":
            compare_models(
                model_a_path=args.model_a,
                model_b_path=args.model_b,
                prompts=args.prompt,
                max_new_tokens=args.max_tokens
            )
        else:
            parser.print_help()
    except Exception as e:
        console.print(f"\n[bold red]✗ Error: {e}[/bold red]")
        import traceback
        console.print(f"[red]{traceback.format_exc()}[/red]")
        sys.exit(1)


if __name__ == "__main__":
    main()
