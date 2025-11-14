"""
Reward Model training command implementation.

This module provides functions for training reward models that predict human preferences.
"""

import sys
from pathlib import Path
from rich.console import Console

# Add src directory to Python path for imports
src_path = Path(__file__).parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

console = Console()


def train_reward_quick(model_choice: str, dataset_choice: str):
    """
    Quick reward model training with pre-configured settings.

    Args:
        model_choice: Base model name or path
        dataset_choice: Preference dataset identifier

    Returns:
        dict: Training results
    """
    from src.auto_bot_tuner.rlhf import (
        create_reward_model_from_pretrained,
        RewardModelTrainer,
        RewardModelConfig,
        RewardModelDataset,
        load_reward_dataset,
        validate_reward_dataset
    )
    from transformers import AutoTokenizer
    from src.auto_bot_tuner.utils.model_loading import get_device

    console.print("\n[cyan]Loading tokenizer...[/cyan]")
    tokenizer = AutoTokenizer.from_pretrained(model_choice)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    console.print("\n[cyan]Creating reward model...[/cyan]")
    device = get_device()

    reward_model = create_reward_model_from_pretrained(
        model_name=model_choice,
        tokenizer=tokenizer,
        freeze_base=False,
        device=device
    )
    console.print(f"[green]✓ Reward model created on {device}[/green]")

    console.print("\n[cyan]Loading preference dataset...[/cyan]")
    raw_dataset = load_reward_dataset(
        dataset_name=dataset_choice,
        split="train",
        max_samples=1000
    )

    validate_reward_dataset(raw_dataset)
    console.print("[green]✓ Dataset validated[/green]")

    train_dataset = RewardModelDataset(
        dataset=raw_dataset,
        tokenizer=tokenizer,
        max_length=512
    )

    console.print(f"[green]✓ Loaded {len(train_dataset)} preference pairs[/green]")

    # Configure training
    config = RewardModelConfig(
        learning_rate=1e-5,
        batch_size=4,
        gradient_accumulation_steps=4,
        num_epochs=1,
        warmup_steps=100,
        logging_steps=10,
        save_steps=500,
        output_dir="checkpoints/reward_model/quick-training",
        margin=0.0
    )

    # Create trainer
    trainer = RewardModelTrainer(
        model=reward_model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        config=config
    )

    console.print("\n[bold green]Starting reward model training...[/bold green]\n")
    results = trainer.train()

    console.print("\n[bold green]✓ Reward model training completed![/bold green]")
    console.print(f"  Total steps: {results['total_steps']}")
    console.print(f"  Best accuracy: {results.get('best_eval_accuracy', 'N/A')}")
    console.print(f"  Checkpoints saved to: {config.output_dir}")

    return results


def train_reward_standard(
    model_choice: str,
    dataset_choice: str,
    learning_rate: float = 1e-5,
    batch_size: int = 4,
    grad_accum: int = 4,
    epochs: int = 1,
    margin: float = 0.0,
    max_samples: int = 0,
    warmup_steps: int = 100,
    freeze_base: bool = False
):
    """
    Standard reward model training with configurable parameters.

    Args:
        model_choice: Base model name or path
        dataset_choice: Preference dataset identifier
        learning_rate: Learning rate
        batch_size: Training batch size
        grad_accum: Gradient accumulation steps
        epochs: Number of epochs
        margin: Ranking loss margin
        max_samples: Maximum samples (0 for all)
        warmup_steps: Number of warmup steps
        freeze_base: Whether to freeze base model (train only value head)

    Returns:
        dict: Training results
    """
    from src.auto_bot_tuner.rlhf import (
        create_reward_model_from_pretrained,
        RewardModelTrainer,
        RewardModelConfig,
        RewardModelDataset,
        load_reward_dataset,
        validate_reward_dataset
    )
    from transformers import AutoTokenizer
    from src.auto_bot_tuner.utils.model_loading import get_device

    console.print("\n[cyan]Loading tokenizer...[/cyan]")
    tokenizer = AutoTokenizer.from_pretrained(model_choice)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    console.print("\n[cyan]Creating reward model...[/cyan]")
    device = get_device()

    reward_model = create_reward_model_from_pretrained(
        model_name=model_choice,
        tokenizer=tokenizer,
        freeze_base=freeze_base,
        device=device
    )
    console.print(f"[green]✓ Reward model created[/green]")

    console.print("\n[cyan]Loading preference dataset...[/cyan]")
    raw_dataset = load_reward_dataset(
        dataset_name=dataset_choice,
        split="train",
        max_samples=max_samples if max_samples > 0 else None
    )

    validate_reward_dataset(raw_dataset)

    train_dataset = RewardModelDataset(
        dataset=raw_dataset,
        tokenizer=tokenizer,
        max_length=512
    )

    console.print(f"[green]✓ Loaded {len(train_dataset)} preference pairs[/green]")

    # Configure training
    config = RewardModelConfig(
        learning_rate=learning_rate,
        batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        num_epochs=epochs,
        warmup_steps=warmup_steps,
        logging_steps=10,
        save_steps=500,
        output_dir="checkpoints/reward_model/standard-training",
        margin=margin
    )

    # Create trainer
    trainer = RewardModelTrainer(
        model=reward_model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        config=config
    )

    console.print("\n[bold green]Starting reward model training...[/bold green]\n")
    results = trainer.train()

    console.print("\n[bold green]✓ Reward model training completed![/bold green]")
    console.print(f"  Total steps: {results['total_steps']}")
    console.print(f"  Best accuracy: {results.get('best_eval_accuracy', 'N/A')}")
    console.print(f"  Checkpoints saved to: {config.output_dir}")

    return results


def main():
    """Main entry point for reward model training command."""
    import argparse

    parser = argparse.ArgumentParser(description="Train a reward model to predict human preferences")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Base model name or path"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Preference dataset name (e.g., Anthropic/hh-rlhf)"
    )
    parser.add_argument(
        "--preset",
        type=str,
        choices=["quick", "standard"],
        default="standard",
        help="Training preset"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-5,
        help="Learning rate"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size"
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=0.0,
        help="Ranking loss margin"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Maximum samples (0 for all)"
    )
    parser.add_argument(
        "--freeze-base",
        action="store_true",
        help="Freeze base model (train only value head)"
    )

    args = parser.parse_args()

    try:
        if args.preset == "quick":
            train_reward_quick(args.model, args.dataset)
        else:
            train_reward_standard(
                model_choice=args.model,
                dataset_choice=args.dataset,
                learning_rate=args.learning_rate,
                batch_size=args.batch_size,
                margin=args.margin,
                max_samples=args.max_samples,
                freeze_base=args.freeze_base
            )
    except Exception as e:
        console.print(f"\n[bold red]✗ Training failed: {e}[/bold red]")
        import traceback
        console.print(f"[red]{traceback.format_exc()}[/red]")
        sys.exit(1)


if __name__ == "__main__":
    main()
