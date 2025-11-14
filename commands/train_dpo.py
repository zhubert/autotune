"""
Direct Preference Optimization (DPO) command implementation.

This module provides functions for DPO training with different preset configurations.
"""

import sys
from pathlib import Path
from rich.console import Console

# Add src directory to Python path for imports
src_path = Path(__file__).parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

console = Console()


def train_dpo_quick(model_choice: str, dataset_choice: str):
    """
    Quick DPO training with pre-configured settings.

    Args:
        model_choice: Model path or HuggingFace identifier (should be SFT-tuned)
        dataset_choice: Preference dataset identifier

    Returns:
        dict: Training results
    """
    from src.auto_bot_tuner.utils.model_loading import load_model_and_tokenizer
    from src.auto_bot_tuner.dpo import (
        DPOTrainer,
        DPOConfig,
        PreferenceDataset,
        load_preference_dataset,
        create_reference_model,
        validate_preference_dataset
    )

    console.print("\n[cyan]Loading policy model...[/cyan]")
    policy_model, tokenizer, device = load_model_and_tokenizer(
        model_path=model_choice,
        use_lora=True
    )

    console.print("\n[cyan]Creating reference model (frozen copy)...[/cyan]")
    reference_model = create_reference_model(policy_model, device)
    console.print("[green]✓ Reference model created[/green]")

    console.print("\n[cyan]Loading preference dataset...[/cyan]")
    raw_dataset = load_preference_dataset(
        dataset_name=dataset_choice,
        split="train",
        max_samples=1000
    )

    # Validate dataset format
    validate_preference_dataset(raw_dataset)
    console.print(f"[green]✓ Dataset validated[/green]")

    train_dataset = PreferenceDataset(
        dataset=raw_dataset,
        tokenizer=tokenizer,
        max_length=512
    )

    console.print(f"[green]✓ Loaded {len(train_dataset)} preference pairs[/green]")

    # Configure training
    config = DPOConfig(
        beta=0.1,
        learning_rate=5e-7,
        batch_size=4,
        gradient_accumulation_steps=4,
        num_epochs=1,
        warmup_steps=50,
        logging_steps=10,
        save_steps=500,
        output_dir="checkpoints/dpo/quick-training",
    )

    # Create trainer
    trainer = DPOTrainer(
        policy_model=policy_model,
        reference_model=reference_model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        config=config
    )

    console.print("\n[bold green]Starting DPO training...[/bold green]\n")
    results = trainer.train()

    console.print("\n[bold green]✓ DPO training completed![/bold green]")
    console.print(f"  Total steps: {results['total_steps']}")
    console.print(f"  Best accuracy: {results['best_eval_accuracy']}")
    console.print(f"  Checkpoints saved to: {config.output_dir}")

    return results


def train_dpo_standard(
    model_choice: str,
    dataset_choice: str,
    beta: float = 0.1,
    label_smoothing: float = 0.0,
    learning_rate: float = 5e-7,
    batch_size: int = 4,
    grad_accum: int = 4,
    epochs: int = 1,
    max_samples: int = 0,
    warmup_steps: int = 50,
    use_lora: bool = True
):
    """
    Standard DPO training with configurable parameters.

    Args:
        model_choice: Model path or HuggingFace identifier
        dataset_choice: Preference dataset identifier
        beta: KL penalty strength (typical: 0.1-0.5)
        label_smoothing: Label smoothing (0 = no smoothing)
        learning_rate: Learning rate
        batch_size: Training batch size
        grad_accum: Gradient accumulation steps
        epochs: Number of epochs
        max_samples: Maximum samples (0 for all)
        warmup_steps: Number of warmup steps
        use_lora: Whether to use LoRA

    Returns:
        dict: Training results
    """
    from src.auto_bot_tuner.utils.model_loading import load_model_and_tokenizer
    from src.auto_bot_tuner.dpo import (
        DPOTrainer,
        DPOConfig,
        PreferenceDataset,
        load_preference_dataset,
        create_reference_model,
        validate_preference_dataset
    )

    console.print("\n[cyan]Loading policy model...[/cyan]")
    policy_model, tokenizer, device = load_model_and_tokenizer(
        model_path=model_choice,
        use_lora=use_lora
    )

    console.print("\n[cyan]Creating reference model...[/cyan]")
    reference_model = create_reference_model(policy_model, device)
    console.print("[green]✓ Reference model created[/green]")

    console.print("\n[cyan]Loading preference dataset...[/cyan]")
    raw_dataset = load_preference_dataset(
        dataset_name=dataset_choice,
        split="train",
        max_samples=max_samples if max_samples > 0 else None
    )

    validate_preference_dataset(raw_dataset)

    train_dataset = PreferenceDataset(
        dataset=raw_dataset,
        tokenizer=tokenizer,
        max_length=512
    )

    console.print(f"[green]✓ Loaded {len(train_dataset)} preference pairs[/green]")

    # Configure training
    config = DPOConfig(
        beta=beta,
        label_smoothing=label_smoothing,
        learning_rate=learning_rate,
        batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        num_epochs=epochs,
        warmup_steps=warmup_steps,
        logging_steps=10,
        save_steps=500,
        output_dir="checkpoints/dpo/standard-training",
    )

    # Create trainer
    trainer = DPOTrainer(
        policy_model=policy_model,
        reference_model=reference_model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        config=config
    )

    console.print("\n[bold green]Starting DPO training...[/bold green]\n")
    results = trainer.train()

    console.print("\n[bold green]✓ DPO training completed![/bold green]")
    console.print(f"  Total steps: {results['total_steps']}")
    console.print(f"  Best accuracy: {results.get('best_eval_accuracy', 'N/A')}")
    console.print(f"  Checkpoints saved to: {config.output_dir}")

    return results


def main():
    """Main entry point for DPO training command."""
    import argparse

    parser = argparse.ArgumentParser(description="Train model using Direct Preference Optimization")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model name or checkpoint path (should be SFT-tuned)"
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
        "--beta",
        type=float,
        default=0.1,
        help="KL penalty strength (typical: 0.1-0.5)"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=5e-7,
        help="Learning rate"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Maximum samples (0 for all)"
    )
    parser.add_argument(
        "--no-lora",
        action="store_true",
        help="Disable LoRA"
    )

    args = parser.parse_args()

    try:
        if args.preset == "quick":
            train_dpo_quick(args.model, args.dataset)
        else:
            train_dpo_standard(
                model_choice=args.model,
                dataset_choice=args.dataset,
                beta=args.beta,
                learning_rate=args.learning_rate,
                batch_size=args.batch_size,
                max_samples=args.max_samples,
                use_lora=not args.no_lora
            )
    except Exception as e:
        console.print(f"\n[bold red]✗ Training failed: {e}[/bold red]")
        import traceback
        console.print(f"[red]{traceback.format_exc()}[/red]")
        sys.exit(1)


if __name__ == "__main__":
    main()
