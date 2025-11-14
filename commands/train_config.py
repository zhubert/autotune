"""
Config-based training command implementation.

This module provides functions for training models from YAML configuration files.
Useful for batch jobs, automated workflows, and reproducible experiments.
"""

import sys
from pathlib import Path
from rich.console import Console
import yaml

# Add src directory to Python path for imports
src_path = Path(__file__).parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

console = Console()


def train_sft_from_config(config_path: str):
    """
    Train SFT model from YAML config file.

    Args:
        config_path: Path to YAML configuration file

    Returns:
        dict: Training results
    """
    from src.auto_bot_tuner.sft import (
        SFTTrainer, SFTConfig, InstructionDataset, load_instruction_dataset
    )
    from src.auto_bot_tuner.utils.model_loading import load_model_and_tokenizer

    console.print("\n[bold cyan]═══════════════════════════════════════════════════[/bold cyan]")
    console.print("[bold cyan]  Supervised Fine-Tuning (SFT) - Config Mode[/bold cyan]")
    console.print("[bold cyan]═══════════════════════════════════════════════════[/bold cyan]\n")

    # Load config
    console.print(f"[cyan]Loading configuration from: {config_path}[/cyan]")
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    console.print("[green]✓ Config loaded![/green]")

    # Display config summary
    console.print("\n[bold]Configuration Summary:[/bold]")
    console.print(f"  Model: {cfg['model']['name']}")
    console.print(f"  Dataset: {cfg['dataset']['name']}")
    console.print(f"  Max samples: {cfg['dataset'].get('max_samples', 'all')}")
    console.print(f"  Output dir: {cfg['training']['output_dir']}")
    console.print(f"  Epochs: {cfg['training']['num_epochs']}")
    console.print(f"  Batch size: {cfg['training']['batch_size']}")

    # Load model
    console.print("\n[cyan][1/3] Loading model...[/cyan]")
    model, tokenizer, device = load_model_and_tokenizer(
        model_path=cfg['model']['name'],
        use_lora=cfg['model'].get('use_lora', True)
    )
    console.print("[green]✓ Model loaded![/green]")

    # Load dataset
    console.print("\n[cyan][2/3] Loading dataset...[/cyan]")
    raw_dataset = load_instruction_dataset(
        dataset_name=cfg['dataset']['name'],
        split="train",
        max_samples=cfg['dataset'].get('max_samples')
    )

    dataset = InstructionDataset(
        dataset=raw_dataset,
        tokenizer=tokenizer,
        max_length=cfg['dataset'].get('max_length', 512),
        format_type=cfg['dataset'].get('format_type', 'auto')
    )
    console.print(f"[green]✓ Dataset loaded! ({len(dataset)} samples)[/green]")

    # Configure training
    console.print("\n[cyan][3/3] Starting training...[/cyan]")
    train_cfg = SFTConfig(**cfg['training'])

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        config=train_cfg
    )

    results = trainer.train()

    console.print("\n[bold green]═══════════════════════════════════════════════════[/bold green]")
    console.print("[bold green]  Training Complete![/bold green]")
    console.print("[bold green]═══════════════════════════════════════════════════[/bold green]")
    console.print(f"[dim]Model saved to: {train_cfg.output_dir}[/dim]\n")

    return results


def train_dpo_from_config(config_path: str):
    """
    Train DPO model from YAML config file.

    Args:
        config_path: Path to YAML configuration file

    Returns:
        dict: Training results
    """
    from src.auto_bot_tuner.dpo import (
        DPOTrainer, DPOConfig, PreferenceDataset,
        load_preference_dataset, create_reference_model
    )
    from src.auto_bot_tuner.utils.model_loading import load_model_and_tokenizer

    console.print("\n[bold cyan]═══════════════════════════════════════════════════[/bold cyan]")
    console.print("[bold cyan]  Direct Preference Optimization (DPO) - Config Mode[/bold cyan]")
    console.print("[bold cyan]═══════════════════════════════════════════════════[/bold cyan]\n")

    # Load config
    console.print(f"[cyan]Loading configuration from: {config_path}[/cyan]")
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    console.print("[green]✓ Config loaded![/green]")

    # Display config summary
    console.print("\n[bold]Configuration Summary:[/bold]")
    console.print(f"  Model: {cfg['model']['name']}")
    console.print(f"  Dataset: {cfg['dataset']['name']}")
    console.print(f"  Max samples: {cfg['dataset'].get('max_samples', 'all')}")
    console.print(f"  Output dir: {cfg['training']['output_dir']}")
    console.print(f"  Beta: {cfg['training']['beta']}")

    # Load policy model
    console.print("\n[cyan][1/4] Loading policy model...[/cyan]")
    policy_model, tokenizer, device = load_model_and_tokenizer(
        model_path=cfg['model']['name'],
        use_lora=cfg['model'].get('use_lora', True)
    )
    console.print("[green]✓ Policy model loaded![/green]")

    # Create reference model
    console.print("\n[cyan][2/4] Creating reference model...[/cyan]")
    reference_model = create_reference_model(policy_model, device)
    console.print("[green]✓ Reference model created![/green]")

    # Load dataset
    console.print("\n[cyan][3/4] Loading preference dataset...[/cyan]")
    raw_dataset = load_preference_dataset(
        dataset_name=cfg['dataset']['name'],
        split="train",
        max_samples=cfg['dataset'].get('max_samples')
    )

    dataset = PreferenceDataset(
        dataset=raw_dataset,
        tokenizer=tokenizer,
        max_length=cfg['dataset'].get('max_length', 512)
    )
    console.print(f"[green]✓ Dataset loaded! ({len(dataset)} samples)[/green]")

    # Configure training
    console.print("\n[cyan][4/4] Starting DPO training...[/cyan]")
    train_cfg = DPOConfig(**cfg['training'])

    trainer = DPOTrainer(
        policy_model=policy_model,
        reference_model=reference_model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        config=train_cfg
    )

    results = trainer.train()

    console.print("\n[bold green]═══════════════════════════════════════════════════[/bold green]")
    console.print("[bold green]  DPO Training Complete![/bold green]")
    console.print("[bold green]═══════════════════════════════════════════════════[/bold green]")
    console.print(f"[dim]Model saved to: {train_cfg.output_dir}[/dim]\n")

    return results


def train_reward_model_from_config(config_path: str):
    """
    Train reward model from YAML config file.

    Args:
        config_path: Path to YAML configuration file

    Returns:
        dict: Training results
    """
    from src.auto_bot_tuner.rlhf import (
        RewardModelTrainer, RewardModelConfig, ComparisonDataset,
        load_comparison_dataset, create_reward_model_from_pretrained
    )
    from src.auto_bot_tuner.utils.model_loading import load_model_and_tokenizer

    console.print("\n[bold cyan]═══════════════════════════════════════════════════[/bold cyan]")
    console.print("[bold cyan]  Reward Model Training - Config Mode[/bold cyan]")
    console.print("[bold cyan]═══════════════════════════════════════════════════[/bold cyan]\n")

    # Load config
    console.print(f"[cyan]Loading configuration from: {config_path}[/cyan]")
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    console.print("[green]✓ Config loaded![/green]")

    # Display config summary
    console.print("\n[bold]Configuration Summary:[/bold]")
    console.print(f"  Model: {cfg['model']['name']}")
    console.print(f"  Dataset: {cfg['dataset']['name']}")
    console.print(f"  Max samples: {cfg['dataset'].get('max_samples', 'all')}")
    console.print(f"  Output dir: {cfg['training']['output_dir']}")
    console.print(f"  Freeze base: {cfg['model'].get('freeze_base', False)}")

    # Load base model
    console.print("\n[cyan][1/4] Loading base model...[/cyan]")
    base_model, tokenizer, device = load_model_and_tokenizer(
        model_path=cfg['model']['name'],
        use_lora=False  # Add LoRA when creating reward model
    )
    console.print("[green]✓ Base model loaded![/green]")

    # Create reward model
    console.print("\n[cyan][2/4] Creating reward model...[/cyan]")
    reward_model = create_reward_model_from_pretrained(
        base_model=base_model,
        tokenizer=tokenizer,
        freeze_base=cfg['model'].get('freeze_base', False),
        use_lora=cfg['model'].get('use_lora', True)
    )
    console.print("[green]✓ Reward model created![/green]")

    # Load dataset
    console.print("\n[cyan][3/4] Loading comparison dataset...[/cyan]")
    raw_dataset = load_comparison_dataset(
        dataset_name=cfg['dataset']['name'],
        split="train",
        max_samples=cfg['dataset'].get('max_samples')
    )

    dataset = ComparisonDataset(
        dataset=raw_dataset,
        tokenizer=tokenizer,
        max_length=cfg['dataset'].get('max_length', 512)
    )
    console.print(f"[green]✓ Dataset loaded! ({len(dataset)} samples)[/green]")

    # Configure training
    console.print("\n[cyan][4/4] Starting reward model training...[/cyan]")
    train_cfg = RewardModelConfig(**cfg['training'])

    trainer = RewardModelTrainer(
        model=reward_model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        config=train_cfg
    )

    results = trainer.train()

    console.print("\n[bold green]═══════════════════════════════════════════════════[/bold green]")
    console.print("[bold green]  Reward Model Training Complete![/bold green]")
    console.print("[bold green]═══════════════════════════════════════════════════[/bold green]")
    console.print(f"[dim]Model saved to: {train_cfg.output_dir}[/dim]\n")

    return results


def train_ppo_from_config(config_path: str):
    """
    Train PPO model from YAML config file.

    Args:
        config_path: Path to YAML configuration file

    Returns:
        dict: Training results
    """
    from src.auto_bot_tuner.rlhf import (
        PPOTrainer, PPOConfig, PromptDataset,
        load_prompt_dataset, create_reward_model_from_pretrained
    )
    from src.auto_bot_tuner.utils.model_loading import load_model_and_tokenizer

    console.print("\n[bold cyan]═══════════════════════════════════════════════════[/bold cyan]")
    console.print("[bold cyan]  PPO Training - Config Mode[/bold cyan]")
    console.print("[bold cyan]═══════════════════════════════════════════════════[/bold cyan]\n")

    # Load config
    console.print(f"[cyan]Loading configuration from: {config_path}[/cyan]")
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    console.print("[green]✓ Config loaded![/green]")

    # Display config summary
    console.print("\n[bold]Configuration Summary:[/bold]")
    console.print(f"  Policy model: {cfg['model']['policy_model']}")
    console.print(f"  Reward model: {cfg['model']['reward_model']}")
    console.print(f"  Dataset: {cfg['dataset']['name']}")
    console.print(f"  Max samples: {cfg['dataset'].get('max_samples', 'all')}")
    console.print(f"  Output dir: {cfg['training']['output_dir']}")

    # Load policy model
    console.print("\n[cyan][1/4] Loading policy model...[/cyan]")
    policy_model, tokenizer, device = load_model_and_tokenizer(
        model_path=cfg['model']['policy_model'],
        use_lora=cfg['model'].get('use_lora', True)
    )
    console.print("[green]✓ Policy model loaded![/green]")

    # Load reward model
    console.print("\n[cyan][2/4] Loading reward model...[/cyan]")
    reward_base_model, _, _ = load_model_and_tokenizer(
        model_path=cfg['model']['reward_model'],
        use_lora=False
    )
    reward_model = create_reward_model_from_pretrained(
        base_model=reward_base_model,
        tokenizer=tokenizer,
        freeze_base=True,
        use_lora=False
    )
    console.print("[green]✓ Reward model loaded![/green]")

    # Load dataset
    console.print("\n[cyan][3/4] Loading prompt dataset...[/cyan]")
    raw_dataset = load_prompt_dataset(
        dataset_name=cfg['dataset']['name'],
        split="train",
        max_samples=cfg['dataset'].get('max_samples')
    )

    dataset = PromptDataset(
        dataset=raw_dataset,
        tokenizer=tokenizer,
        max_length=cfg['dataset'].get('max_length', 512)
    )
    console.print(f"[green]✓ Dataset loaded! ({len(dataset)} samples)[/green]")

    # Configure training
    console.print("\n[cyan][4/4] Starting PPO training...[/cyan]")
    train_cfg = PPOConfig(**cfg['training'])

    trainer = PPOTrainer(
        policy_model=policy_model,
        reward_model=reward_model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        config=train_cfg
    )

    results = trainer.train()

    console.print("\n[bold green]═══════════════════════════════════════════════════[/bold green]")
    console.print("[bold green]  PPO Training Complete![/bold green]")
    console.print("[bold green]═══════════════════════════════════════════════════[/bold green]")
    console.print(f"[dim]Model saved to: {train_cfg.output_dir}[/dim]\n")

    return results


def main():
    """Main entry point for config-based training command."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Train models from YAML configuration files"
    )
    parser.add_argument(
        "config",
        type=str,
        help="Path to YAML config file"
    )
    parser.add_argument(
        "--type",
        type=str,
        choices=["sft", "dpo", "reward", "ppo"],
        help="Training type (auto-detected from config path if not specified)"
    )

    args = parser.parse_args()

    # Auto-detect training type from path if not specified
    training_type = args.type
    if not training_type:
        config_name = Path(args.config).stem.lower()
        if "sft" in config_name:
            training_type = "sft"
        elif "dpo" in config_name:
            training_type = "dpo"
        elif "reward" in config_name:
            training_type = "reward"
        elif "ppo" in config_name:
            training_type = "ppo"
        else:
            parser.error(
                "Could not auto-detect training type from filename. "
                "Please specify --type (sft, dpo, reward, or ppo)"
            )

    console.print(f"\n[bold]Training type:[/bold] {training_type.upper()}")
    console.print(f"[bold]Config file:[/bold] {args.config}")

    try:
        # Run appropriate training
        if training_type == "sft":
            train_sft_from_config(args.config)
        elif training_type == "dpo":
            train_dpo_from_config(args.config)
        elif training_type == "reward":
            train_reward_model_from_config(args.config)
        elif training_type == "ppo":
            train_ppo_from_config(args.config)
    except Exception as e:
        console.print(f"\n[bold red]✗ Training failed: {e}[/bold red]")
        import traceback
        console.print(f"[red]{traceback.format_exc()}[/red]")
        sys.exit(1)


if __name__ == "__main__":
    main()
