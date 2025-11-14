"""
Supervised Fine-Tuning (SFT) command implementation.

This module provides functions for SFT training with different preset configurations.
"""

import sys
from pathlib import Path
from rich.console import Console

# Add src directory to Python path for imports
src_path = Path(__file__).parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

console = Console()


def train_sft_quick(model_choice: str, dataset_choice: str):
    """
    Quick SFT training with pre-configured settings optimized for fast experimentation.

    Args:
        model_choice: HuggingFace model name or local path
        dataset_choice: HuggingFace dataset identifier

    Returns:
        dict: Training results including total_steps, epochs_completed, and output_dir
    """
    from src.auto_bot_tuner.utils.model_loading import load_model_and_tokenizer
    from src.auto_bot_tuner.sft import (
        SFTTrainer,
        SFTConfig,
        InstructionDataset,
        load_instruction_dataset
    )

    console.print("\n[cyan]Loading model...[/cyan]")
    model, tokenizer, device = load_model_and_tokenizer(
        model_path=model_choice,
        use_lora=True
    )

    console.print("\n[cyan]Loading dataset...[/cyan]")
    raw_dataset = load_instruction_dataset(
        dataset_name=dataset_choice,
        split="train",
        max_samples=1000
    )

    # Detect dataset format
    format_type = "alpaca" if "instruction" in raw_dataset.column_names else "chat"
    console.print(f"[green]Detected dataset format: {format_type}[/green]")

    train_dataset = InstructionDataset(
        dataset=raw_dataset,
        tokenizer=tokenizer,
        max_length=512,
        format_type=format_type
    )

    console.print(f"[green]✓ Loaded {len(train_dataset)} training samples[/green]")

    # Configure training
    config = SFTConfig(
        learning_rate=2e-5,
        batch_size=4,
        gradient_accumulation_steps=4,
        num_epochs=3,
        warmup_steps=100,
        logging_steps=10,
        save_steps=500,
        output_dir="checkpoints/sft/quick-training",
    )

    # Create trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        config=config
    )

    console.print("\n[bold green]Starting training...[/bold green]\n")
    results = trainer.train()

    console.print("\n[bold green]✓ Training completed![/bold green]")
    console.print(f"  Total steps: {results['total_steps']}")
    console.print(f"  Epochs: {results['epochs_completed']}")
    console.print(f"  Checkpoints saved to: {config.output_dir}")

    return results


def train_sft_standard(
    model_choice: str,
    dataset_choice: str,
    batch_size: int = 4,
    grad_accum: int = 4,
    learning_rate: float = 2e-5,
    epochs: int = 3,
    max_samples: int = 0,
    use_lora: bool = True
):
    """
    Standard SFT training with configurable parameters.

    Args:
        model_choice: HuggingFace model name or local path
        dataset_choice: HuggingFace dataset identifier
        batch_size: Training batch size
        grad_accum: Gradient accumulation steps
        learning_rate: Learning rate for optimizer
        epochs: Number of training epochs
        max_samples: Maximum samples to use (0 for all)
        use_lora: Whether to use LoRA for parameter-efficient training

    Returns:
        dict: Training results
    """
    from src.auto_bot_tuner.utils.model_loading import load_model_and_tokenizer
    from src.auto_bot_tuner.sft import (
        SFTTrainer,
        SFTConfig,
        InstructionDataset,
        load_instruction_dataset
    )

    console.print("\n[cyan]Loading model...[/cyan]")
    model, tokenizer, device = load_model_and_tokenizer(
        model_path=model_choice,
        use_lora=use_lora
    )

    console.print("\n[cyan]Loading dataset...[/cyan]")
    raw_dataset = load_instruction_dataset(
        dataset_name=dataset_choice,
        split="train",
        max_samples=max_samples if max_samples > 0 else None
    )

    format_type = "alpaca" if "instruction" in raw_dataset.column_names else "chat"
    console.print(f"[green]Detected dataset format: {format_type}[/green]")

    train_dataset = InstructionDataset(
        dataset=raw_dataset,
        tokenizer=tokenizer,
        max_length=512,
        format_type=format_type
    )

    console.print(f"[green]✓ Loaded {len(train_dataset)} training samples[/green]")

    # Configure training
    config = SFTConfig(
        learning_rate=learning_rate,
        batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        num_epochs=epochs,
        warmup_steps=100,
        logging_steps=10,
        save_steps=500,
        output_dir="checkpoints/sft/standard-training",
    )

    # Create trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        config=config
    )

    console.print("\n[bold green]Starting training...[/bold green]\n")
    results = trainer.train()

    console.print("\n[bold green]✓ Training completed![/bold green]")
    console.print(f"  Total steps: {results['total_steps']}")
    console.print(f"  Epochs: {results['epochs_completed']}")
    console.print(f"  Checkpoints saved to: {config.output_dir}")

    return results


def train_sft_custom(
    model_choice: str,
    dataset_choice: str,
    learning_rate: float = 2e-5,
    batch_size: int = 4,
    grad_accum: int = 4,
    epochs: int = 3,
    max_samples: int = 0,
    warmup_steps: int = 100,
    max_grad_norm: float = 1.0,
    weight_decay: float = 0.01,
    adam_beta1: float = 0.9,
    adam_beta2: float = 0.999,
    logging_steps: int = 10,
    save_steps: int = 1000,
    save_total_limit: int = 3,
    output_dir: str = "checkpoints/sft/custom-training",
    use_lora: bool = True,
    fp16: bool = False,
    bf16: bool = False
):
    """
    Custom SFT training with full parameter configuration.

    Args:
        model_choice: HuggingFace model name or local path
        dataset_choice: HuggingFace dataset identifier
        learning_rate: Learning rate for optimizer
        batch_size: Training batch size
        grad_accum: Gradient accumulation steps
        epochs: Number of training epochs
        max_samples: Maximum samples to use (0 for all)
        warmup_steps: Number of warmup steps
        max_grad_norm: Maximum gradient norm for clipping
        weight_decay: Weight decay for regularization
        adam_beta1: Adam beta1 parameter
        adam_beta2: Adam beta2 parameter
        logging_steps: Log every N steps
        save_steps: Save checkpoint every N steps
        save_total_limit: Maximum number of checkpoints to keep
        output_dir: Directory to save checkpoints
        use_lora: Whether to use LoRA
        fp16: Use FP16 mixed precision
        bf16: Use BF16 mixed precision

    Returns:
        dict: Training results
    """
    from src.auto_bot_tuner.utils.model_loading import load_model_and_tokenizer
    from src.auto_bot_tuner.sft import (
        SFTTrainer,
        SFTConfig,
        InstructionDataset,
        load_instruction_dataset
    )

    console.print("\n[cyan]Loading model...[/cyan]")
    model, tokenizer, device = load_model_and_tokenizer(
        model_path=model_choice,
        use_lora=use_lora
    )

    console.print("\n[cyan]Loading dataset...[/cyan]")
    raw_dataset = load_instruction_dataset(
        dataset_name=dataset_choice,
        split="train",
        max_samples=max_samples if max_samples > 0 else None
    )

    format_type = "alpaca" if "instruction" in raw_dataset.column_names else "chat"
    console.print(f"[green]Detected dataset format: {format_type}[/green]")

    train_dataset = InstructionDataset(
        dataset=raw_dataset,
        tokenizer=tokenizer,
        max_length=512,
        format_type=format_type
    )

    console.print(f"[green]✓ Loaded {len(train_dataset)} training samples[/green]")

    # Configure training with all custom parameters
    config = SFTConfig(
        learning_rate=learning_rate,
        batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        num_epochs=epochs,
        warmup_steps=warmup_steps,
        max_grad_norm=max_grad_norm,
        weight_decay=weight_decay,
        adam_beta1=adam_beta1,
        adam_beta2=adam_beta2,
        logging_steps=logging_steps,
        save_steps=save_steps,
        save_total_limit=save_total_limit,
        output_dir=output_dir,
        fp16=fp16,
        bf16=bf16,
    )

    # Create trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        config=config
    )

    console.print("\n[bold green]Starting training...[/bold green]\n")
    results = trainer.train()

    console.print("\n[bold green]✓ Training completed![/bold green]")
    console.print(f"  Total steps: {results['total_steps']}")
    console.print(f"  Epochs: {results['epochs_completed']}")
    console.print(f"  Checkpoints saved to: {config.output_dir}")

    return results


def resume_sft_training(
    checkpoint_path: str,
    dataset_choice: str,
    additional_epochs: int = 1
):
    """
    Resume SFT training from a checkpoint.

    Args:
        checkpoint_path: Path to checkpoint directory
        dataset_choice: HuggingFace dataset identifier
        additional_epochs: Number of additional epochs to train

    Returns:
        dict: Training results
    """
    import torch
    from pathlib import Path
    import json
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from src.auto_bot_tuner.sft import (
        SFTTrainer,
        SFTConfig,
        InstructionDataset,
        load_instruction_dataset
    )

    checkpoint_dir = Path(checkpoint_path)

    console.print("\n[cyan]Loading checkpoint...[/cyan]")

    # Load model and tokenizer from checkpoint
    model = AutoModelForCausalLM.from_pretrained(checkpoint_dir)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)

    # Get device
    from src.auto_bot_tuner.utils.model_loading import get_device
    device = get_device()
    model = model.to(device)

    console.print(f"[green]✓ Loaded model from {checkpoint_dir}[/green]")

    # Load training configuration
    config_path = checkpoint_dir.parent / "training_config.json"
    if config_path.exists():
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        console.print("[green]✓ Loaded training configuration[/green]")
    else:
        console.print("[yellow]Warning: training_config.json not found, using defaults[/yellow]")
        config_dict = {}

    # Load training state
    state_path = checkpoint_dir / "training_state.pt"
    training_state = torch.load(state_path, map_location=device)
    console.print(f"[green]✓ Loaded training state (step {training_state['global_step']})[/green]")

    # Load dataset
    console.print("\n[cyan]Loading dataset...[/cyan]")
    raw_dataset = load_instruction_dataset(
        dataset_name=dataset_choice,
        split="train",
        max_samples=config_dict.get('max_samples', None)
    )

    format_type = "alpaca" if "instruction" in raw_dataset.column_names else "chat"
    train_dataset = InstructionDataset(
        dataset=raw_dataset,
        tokenizer=tokenizer,
        max_length=512,
        format_type=format_type
    )
    console.print(f"[green]✓ Loaded {len(train_dataset)} training samples[/green]")

    # Create config with updated epochs
    original_epochs = config_dict.get('num_epochs', 3)
    config = SFTConfig(
        learning_rate=config_dict.get('learning_rate', 2e-5),
        batch_size=config_dict.get('batch_size', 4),
        gradient_accumulation_steps=config_dict.get('gradient_accumulation_steps', 4),
        num_epochs=additional_epochs,
        warmup_steps=config_dict.get('warmup_steps', 100),
        logging_steps=config_dict.get('logging_steps', 10),
        save_steps=config_dict.get('save_steps', 1000),
        output_dir=str(checkpoint_dir.parent / "resumed"),
        weight_decay=config_dict.get('weight_decay', 0.01),
        adam_beta1=config_dict.get('adam_beta1', 0.9),
        adam_beta2=config_dict.get('adam_beta2', 0.999),
        max_grad_norm=config_dict.get('max_grad_norm', 1.0),
        save_total_limit=config_dict.get('save_total_limit', 3),
        fp16=config_dict.get('fp16', False),
        bf16=config_dict.get('bf16', False),
    )

    # Create trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        config=config
    )

    # Restore training state
    trainer.global_step = training_state['global_step']
    trainer.epoch = training_state['epoch']
    trainer.best_eval_loss = training_state.get('best_eval_loss', float('inf'))
    trainer.optimizer.load_state_dict(training_state['optimizer_state'])
    trainer.scheduler.load_state_dict(training_state['scheduler_state'])

    console.print(f"\n[bold green]Resuming training from step {trainer.global_step}...[/bold green]\n")
    results = trainer.train()

    console.print("\n[bold green]✓ Training completed![/bold green]")
    console.print(f"  Total steps: {results['total_steps']}")
    console.print(f"  Epochs: {results['epochs_completed']}")
    console.print(f"  Checkpoints saved to: {config.output_dir}")

    return results


def main():
    """
    Main entry point for the SFT training command.

    This is called when using the command directly from the CLI.
    """
    import argparse

    parser = argparse.ArgumentParser(description="Train model using Supervised Fine-Tuning")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model name or path (e.g., gpt2, gpt2-medium)"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name (e.g., yahma/alpaca-cleaned)"
    )
    parser.add_argument(
        "--preset",
        type=str,
        choices=["quick", "standard"],
        default="standard",
        help="Training preset (quick or standard)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size"
    )
    parser.add_argument(
        "--grad-accum",
        type=int,
        default=4,
        help="Gradient accumulation steps"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-5,
        help="Learning rate"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of epochs"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Maximum samples to use (0 for all)"
    )
    parser.add_argument(
        "--no-lora",
        action="store_true",
        help="Disable LoRA (train full model)"
    )

    args = parser.parse_args()

    try:
        if args.preset == "quick":
            train_sft_quick(args.model, args.dataset)
        else:
            train_sft_standard(
                model_choice=args.model,
                dataset_choice=args.dataset,
                batch_size=args.batch_size,
                grad_accum=args.grad_accum,
                learning_rate=args.learning_rate,
                epochs=args.epochs,
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
