"""
Interactive CLI menu handlers

This module provides the interactive menu system that delegates to command implementations.
Following the pattern from the transformer project where the interactive CLI calls into
standalone command modules.
"""

import sys
from pathlib import Path

# Add src directory to Python path for imports
src_path = Path(__file__).parent
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

import questionary
from questionary import Style
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

console = Console()

custom_style = Style([
    ('qmark', 'fg:#5f87ff bold'),
    ('question', 'bold'),
    ('answer', 'fg:#5fff5f bold'),
    ('pointer', 'fg:#5f87ff bold'),
    ('highlighted', 'fg:#5f87ff bold'),
    ('selected', 'fg:#5fff5f'),
    ('separator', 'fg:#6c6c6c'),
    ('instruction', 'fg:#858585'),
])


def print_banner():
    """Print welcome banner"""
    banner = Text()
    banner.append("🎓 LLM Post-Training", style="bold cyan")
    banner.append("\n")
    banner.append("Learn how to fine-tune language models", style="dim")

    console.print(Panel(
        banner,
        border_style="cyan",
        padding=(1, 2)
    ))


def interactive_main():
    """
    Main entry point for interactive mode.

    Displays the main menu in a loop until user exits.
    """
    while True:
        try:
            main_menu()
        except KeyboardInterrupt:
            console.print("\n\n👋 Goodbye!", style="cyan")
            sys.exit(0)


def _build_context_menu(tracker):
    """
    Build menu options based on current state.

    Returns a smart, adaptive menu that only shows what makes sense.
    """
    from src.auto_bot_tuner.utils.model_loading import list_downloaded_models

    choices = []

    # Check what we have
    downloaded_models = list_downloaded_models()
    has_models = len(downloaded_models) > 0

    # Check training state
    has_sft = False
    has_reward = False
    has_training_in_progress = False

    if tracker:
        stats = tracker.registry.get_stats()
        has_sft = stats.get('ready_for_dpo', 0) > 0 or stats.get('ready_for_rlhf', 0) > 0
        has_reward = stats.get('reward_models', 0) > 0
        has_training_in_progress = stats.get('training', 0) > 0

    # Show a compact status line
    status_parts = []
    if has_models:
        status_parts.append(f"{len(downloaded_models)} model(s)")
    if has_sft:
        status_parts.append("SFT ready")
    if has_reward:
        status_parts.append("reward model")

    if status_parts:
        console.print(f"[dim]Status: {', '.join(status_parts)}[/dim]\n")

    # Priority 1: If training is in progress, show that first
    if has_training_in_progress:
        choices.append("⏳ Check Training Progress")
        console.print("[yellow]⚡ Training in progress![/yellow]\n")

    # Priority 2: If no models, downloading is the obvious first step
    if not has_models:
        console.print("[cyan]💡 Get started by downloading a pre-trained model[/cyan]\n")
        choices.append("📥 Download Pre-trained Model")
        choices.append("📥 Download Datasets")
    else:
        # Priority 3: Training options based on progress
        choices.append("📚 Train: Supervised Fine-Tuning (SFT)")

        if has_sft:
            choices.append("🎯 Train: DPO (Preference Alignment)")

            if has_reward:
                choices.append("🚀 Train: RLHF with PPO")
            else:
                choices.append("🏆 Train: Reward Model (for RLHF)")

        # Separator for non-training actions
        choices.append("─" * 40)

        # Testing and evaluation
        choices.append("💬 Chat with a Model")
        choices.append("📊 Evaluate Models")

        if tracker and tracker.registry.get_all_models():
            choices.append("🗂️  Browse Trained Models")

        # Utility options
        choices.append("─" * 40)
        choices.append("📥 Download Model/Dataset")

    choices.append("❌ Exit")

    return choices


def main_menu():
    """
    Display context-aware main menu.

    Menu options adapt based on:
    - Whether models are downloaded
    - Training progress and prerequisites
    - Available checkpoints
    """
    print_banner()

    # Initialize progress tracker
    try:
        from wizard import ProgressTracker
        tracker = ProgressTracker()
    except Exception as e:
        # Progress tracking is optional - continue with menu if it fails
        console.print(f"[dim]Progress tracking unavailable: {e}[/dim]\n")
        tracker = None

    # Build context-aware menu
    choices = _build_context_menu(tracker)

    choice = questionary.select(
        "What would you like to do?",
        choices=choices,
        style=custom_style
    ).ask()

    if choice is None or choice == "❌ Exit":
        console.print("\n👋 Goodbye!", style="cyan")
        sys.exit(0)

    # Route to appropriate handler
    if choice.startswith("─"):
        # Separator selected - ignore
        return
    elif "Download Pre-trained Model" in choice:
        download_model_menu()
    elif "Download Model/Dataset" in choice:
        download_combined_menu()
    elif "Download Datasets" in choice:
        download_data_menu()
    elif "Check Training Progress" in choice:
        check_training_progress(tracker)
    elif "SFT" in choice or "Supervised" in choice:
        sft_menu()
    elif "Reward Model" in choice:
        reward_model_menu()
    elif "RLHF" in choice:
        rlhf_menu()
    elif "DPO" in choice:
        dpo_menu()
    elif "Evaluate" in choice:
        evaluate_menu()
    elif "Chat" in choice:
        chat_menu()
    elif "Browse" in choice:
        browse_models_menu()


def check_training_progress(tracker):
    """Display current training progress"""
    if not tracker:
        console.print("[yellow]Progress tracking unavailable[/yellow]")
        input("\nPress Enter to continue...")
        return

    active = tracker.get_active_training()
    if not active:
        console.print("[yellow]No training currently in progress[/yellow]")
        input("\nPress Enter to continue...")
        return

    from wizard import render_active_training
    console.print()
    render_active_training(active)
    console.print()

    input("\nPress Enter to continue...")


def download_model_menu():
    """Menu for downloading pre-trained models"""
    console.print("\n[bold cyan]📥 Download Pre-trained Model[/bold cyan]\n")

    console.print(
        "[dim]Pre-trained models will be downloaded from HuggingFace "
        "and cached locally.[/dim]\n"
    )

    models = [
        {
            "name": "Llama 3.2 1B (Recommended)",
            "id": "meta-llama/Llama-3.2-1B",
            "size": "~2.5 GB",
            "desc": "Meta's latest 1B model - great balance of size and quality"
        },
        {
            "name": "Llama 3.2 3B",
            "id": "meta-llama/Llama-3.2-3B",
            "size": "~6 GB",
            "desc": "Larger model, better performance, needs more VRAM"
        },
        {
            "name": "GPT-2 Medium",
            "id": "gpt2-medium",
            "size": "~1.5 GB",
            "desc": "Smaller OpenAI model - fastest to experiment with"
        },
        {
            "name": "Qwen 2.5 1.5B",
            "id": "Qwen/Qwen2.5-1.5B",
            "size": "~3 GB",
            "desc": "Alibaba's model - excellent instruction following"
        },
    ]

    # Create table showing model options
    table = Table(title="Available Models", show_header=True)
    table.add_column("Model", style="cyan", no_wrap=True)
    table.add_column("Size", style="magenta")
    table.add_column("Description", style="white")

    for model in models:
        table.add_row(model["name"], model["size"], model["desc"])

    console.print(table)
    console.print()

    choices = [m["name"] for m in models] + ["← Back to Main Menu"]

    choice = questionary.select(
        "Which model would you like to download?",
        choices=choices,
        style=custom_style
    ).ask()

    if choice is None or "Back" in choice:
        return

    # Find selected model
    selected = next(m for m in models if m["name"] == choice)

    # Delegate to command
    from commands import download
    download.download_model(selected["id"])

    input("\nPress Enter to continue...")


def sft_menu():
    """Menu for Supervised Fine-Tuning"""
    console.print("\n[bold cyan]📚 Supervised Fine-Tuning (SFT)[/bold cyan]\n")

    choices = [
        "🚀 Quick Training (Small dataset, fast)",
        "⚙️  Standard Training (Default settings)",
        "🔧 Custom Training (Configure everything)",
        "▶️  Resume Training",
        "← Back to Main Menu"
    ]

    choice = questionary.select(
        "What would you like to do?",
        choices=choices,
        style=custom_style
    ).ask()

    if choice is None or "Back" in choice:
        return

    if "Quick" in choice:
        quick_sft_training()
    elif "Standard" in choice:
        standard_sft_training()
    elif "Custom" in choice:
        custom_sft_training()
    elif "Resume" in choice:
        resume_sft_training()


def quick_sft_training():
    """Quick training mode with sensible defaults."""
    console.print("\n[bold cyan]🚀 Quick SFT Training[/bold cyan]\n")
    console.print("This mode uses pre-configured settings optimized for quick experimentation.\n")

    # Select model
    console.print("[cyan]Select a model:[/cyan]")
    from src.auto_bot_tuner.utils.model_loading import list_downloaded_models

    downloaded_models = list_downloaded_models()

    if not downloaded_models:
        # Fallback: allow user to input HuggingFace model ID or default options
        use_custom = questionary.confirm(
            "No local models found. Do you want to use a HuggingFace model ID?",
            default=True,
            style=custom_style
        ).ask()

        if not use_custom:
            return

        model_choice = questionary.text(
            "Model ID or path:",
            default="gpt2",
            style=custom_style
        ).ask()

        if not model_choice:
            return
    else:
        # Show model selection menu
        choices = [f"models/{model}" for model in downloaded_models]
        choices.extend([
            "Use HuggingFace model ID or custom path",
            "Back to menu"
        ])

        model_choice = questionary.select(
            "Model:",
            choices=choices,
            style=custom_style
        ).ask()

        if model_choice == "Back to menu" or model_choice is None:
            return

        if model_choice == "Use HuggingFace model ID or custom path":
            model_choice = questionary.text(
                "Model ID or path:",
                default="gpt2",
                style=custom_style
            ).ask()

            if not model_choice:
                return

    # Select dataset
    console.print("\n[cyan]Select a dataset:[/cyan]")
    dataset_choice = questionary.select(
        "Dataset:",
        choices=[
            "yahma/alpaca-cleaned",
            "OpenAssistant/oasst1",
            "tatsu-lab/alpaca",
            "Back to menu"
        ],
        style=custom_style
    ).ask()

    if dataset_choice == "Back to menu" or dataset_choice is None:
        return

    # Confirm settings
    console.print("\n[bold yellow]Quick Training Configuration:[/bold yellow]")
    console.print(f"  Model: {model_choice}")
    console.print(f"  Dataset: {dataset_choice}")
    console.print(f"  Batch size: 4")
    console.print(f"  Gradient accumulation: 4 (effective batch size: 16)")
    console.print(f"  Learning rate: 2e-5")
    console.print(f"  Epochs: 3")
    console.print(f"  Max samples: 1000 (for quick testing)")
    console.print(f"  LoRA: Enabled (r=16)")

    confirm = questionary.confirm(
        "\nStart training?",
        default=True,
        style=custom_style
    ).ask()

    if not confirm:
        return

    # Delegate to command
    try:
        from commands import train_sft
        train_sft.train_sft_quick(model_choice, dataset_choice)
    except Exception as e:
        console.print(f"\n[bold red]✗ Training failed: {e}[/bold red]")
        import traceback
        console.print(f"[red]{traceback.format_exc()}[/red]")

    input("\nPress Enter to continue...")


def standard_sft_training():
    """Standard training mode with more configuration options."""
    console.print("\n[bold cyan]📋 Standard SFT Training[/bold cyan]\n")
    console.print("This mode allows you to configure key training parameters.\n")

    # Select model
    console.print("[cyan]Select a model:[/cyan]")
    from src.auto_bot_tuner.utils.model_loading import list_downloaded_models

    downloaded_models = list_downloaded_models()

    if not downloaded_models:
        # Fallback: allow user to input HuggingFace model ID or default options
        use_custom = questionary.confirm(
            "No local models found. Do you want to use a HuggingFace model ID?",
            default=True,
            style=custom_style
        ).ask()

        if not use_custom:
            return

        model_choice = questionary.text(
            "Model ID or path:",
            default="gpt2-medium",
            style=custom_style
        ).ask()

        if not model_choice:
            return
    else:
        # Show model selection menu
        choices = [f"models/{model}" for model in downloaded_models]
        choices.extend([
            "Use HuggingFace model ID or custom path",
            "Back to menu"
        ])

        model_choice = questionary.select(
            "Model:",
            choices=choices,
            style=custom_style
        ).ask()

        if model_choice == "Back to menu" or model_choice is None:
            return

        if model_choice == "Use HuggingFace model ID or custom path":
            model_choice = questionary.text(
                "Model ID or path:",
                default="gpt2-medium",
                style=custom_style
            ).ask()

            if not model_choice:
                return

    dataset_choice = questionary.text(
        "Dataset name (HuggingFace identifier):",
        default="yahma/alpaca-cleaned",
        style=custom_style
    ).ask()

    if dataset_choice is None:
        return

    batch_size = questionary.text(
        "Batch size:",
        default="4",
        style=custom_style
    ).ask()

    grad_accum = questionary.text(
        "Gradient accumulation steps:",
        default="4",
        style=custom_style
    ).ask()

    learning_rate = questionary.text(
        "Learning rate:",
        default="2e-5",
        style=custom_style
    ).ask()

    epochs = questionary.text(
        "Number of epochs:",
        default="3",
        style=custom_style
    ).ask()

    max_samples = questionary.text(
        "Max samples (0 for all):",
        default="0",
        style=custom_style
    ).ask()

    use_lora = questionary.confirm(
        "Use LoRA?",
        default=True,
        style=custom_style
    ).ask()

    # Confirm settings
    console.print("\n[bold yellow]Training Configuration:[/bold yellow]")
    console.print(f"  Model: {model_choice}")
    console.print(f"  Dataset: {dataset_choice}")
    console.print(f"  Batch size: {batch_size}")
    console.print(f"  Gradient accumulation: {grad_accum}")
    console.print(f"  Learning rate: {learning_rate}")
    console.print(f"  Epochs: {epochs}")
    console.print(f"  Max samples: {max_samples if int(max_samples) > 0 else 'All'}")
    console.print(f"  LoRA: {'Enabled' if use_lora else 'Disabled'}")

    confirm = questionary.confirm(
        "\nStart training?",
        default=True,
        style=custom_style
    ).ask()

    if not confirm:
        return

    # Delegate to command
    try:
        from commands import train_sft
        train_sft.train_sft_standard(
            model_choice=model_choice,
            dataset_choice=dataset_choice,
            batch_size=int(batch_size),
            grad_accum=int(grad_accum),
            learning_rate=float(learning_rate),
            epochs=int(epochs),
            max_samples=int(max_samples),
            use_lora=use_lora
        )
    except Exception as e:
        console.print(f"\n[bold red]✗ Training failed: {e}[/bold red]")
        import traceback
        console.print(f"[red]{traceback.format_exc()}[/red]")

    input("\nPress Enter to continue...")


def custom_sft_training():
    """Custom training mode with full configuration options."""
    console.print("\n[bold cyan]⚙️  Custom SFT Training[/bold cyan]\n")
    console.print("This mode allows you to configure all training parameters.\n")

    # Select model
    console.print("[cyan]Select a model:[/cyan]")
    from src.auto_bot_tuner.utils.model_loading import list_downloaded_models

    downloaded_models = list_downloaded_models()

    if not downloaded_models:
        # Fallback: allow user to input HuggingFace model ID or default options
        use_custom = questionary.confirm(
            "No local models found. Do you want to use a HuggingFace model ID?",
            default=True,
            style=custom_style
        ).ask()

        if not use_custom:
            return

        model_choice = questionary.text(
            "Model ID or path:",
            default="gpt2-medium",
            style=custom_style
        ).ask()

        if not model_choice:
            return
    else:
        # Show model selection menu
        choices = [f"models/{model}" for model in downloaded_models]
        choices.extend([
            "Use HuggingFace model ID or custom path",
            "Back to menu"
        ])

        model_choice = questionary.select(
            "Model:",
            choices=choices,
            style=custom_style
        ).ask()

        if model_choice == "Back to menu" or model_choice is None:
            return

        if model_choice == "Use HuggingFace model ID or custom path":
            model_choice = questionary.text(
                "Model ID or path:",
                default="gpt2-medium",
                style=custom_style
            ).ask()

            if not model_choice:
                return

    dataset_choice = questionary.text(
        "Dataset name (HuggingFace identifier):",
        default="yahma/alpaca-cleaned",
        style=custom_style
    ).ask()
    if dataset_choice is None:
        return

    # Core hyperparameters
    learning_rate = questionary.text(
        "Learning rate:",
        default="2e-5",
        style=custom_style
    ).ask()

    batch_size = questionary.text(
        "Batch size:",
        default="4",
        style=custom_style
    ).ask()

    grad_accum = questionary.text(
        "Gradient accumulation steps:",
        default="4",
        style=custom_style
    ).ask()

    epochs = questionary.text(
        "Number of epochs:",
        default="3",
        style=custom_style
    ).ask()

    max_samples = questionary.text(
        "Max samples (0 for all):",
        default="0",
        style=custom_style
    ).ask()

    # Advanced parameters
    warmup_steps = questionary.text(
        "Warmup steps:",
        default="100",
        style=custom_style
    ).ask()

    max_grad_norm = questionary.text(
        "Max gradient norm (for clipping):",
        default="1.0",
        style=custom_style
    ).ask()

    weight_decay = questionary.text(
        "Weight decay:",
        default="0.01",
        style=custom_style
    ).ask()

    # Optimization parameters
    adam_beta1 = questionary.text(
        "Adam beta1:",
        default="0.9",
        style=custom_style
    ).ask()

    adam_beta2 = questionary.text(
        "Adam beta2:",
        default="0.999",
        style=custom_style
    ).ask()

    # Logging and checkpointing
    logging_steps = questionary.text(
        "Logging steps:",
        default="10",
        style=custom_style
    ).ask()

    save_steps = questionary.text(
        "Save checkpoint every N steps:",
        default="1000",
        style=custom_style
    ).ask()

    save_total_limit = questionary.text(
        "Maximum checkpoints to keep:",
        default="3",
        style=custom_style
    ).ask()

    output_dir = questionary.text(
        "Output directory:",
        default="checkpoints/sft/custom-training",
        style=custom_style
    ).ask()

    # LoRA and precision
    use_lora = questionary.confirm(
        "Use LoRA?",
        default=True,
        style=custom_style
    ).ask()

    use_fp16 = questionary.confirm(
        "Use FP16 mixed precision?",
        default=False,
        style=custom_style
    ).ask()

    use_bf16 = questionary.confirm(
        "Use BF16 mixed precision?",
        default=False,
        style=custom_style
    ).ask()

    # Confirm settings
    console.print("\n[bold yellow]Custom Training Configuration:[/bold yellow]")
    console.print(f"  Model: {model_choice}")
    console.print(f"  Dataset: {dataset_choice}")
    console.print(f"  Learning rate: {learning_rate}")
    console.print(f"  Batch size: {batch_size}")
    console.print(f"  Gradient accumulation: {grad_accum}")
    console.print(f"  Epochs: {epochs}")
    console.print(f"  Max samples: {max_samples if int(max_samples) > 0 else 'All'}")
    console.print(f"  Warmup steps: {warmup_steps}")
    console.print(f"  Max grad norm: {max_grad_norm}")
    console.print(f"  Weight decay: {weight_decay}")
    console.print(f"  Adam betas: ({adam_beta1}, {adam_beta2})")
    console.print(f"  Logging steps: {logging_steps}")
    console.print(f"  Save steps: {save_steps}")
    console.print(f"  Save limit: {save_total_limit}")
    console.print(f"  Output dir: {output_dir}")
    console.print(f"  LoRA: {'Enabled' if use_lora else 'Disabled'}")
    console.print(f"  FP16: {'Enabled' if use_fp16 else 'Disabled'}")
    console.print(f"  BF16: {'Enabled' if use_bf16 else 'Disabled'}")

    confirm = questionary.confirm(
        "\nStart training?",
        default=True,
        style=custom_style
    ).ask()

    if not confirm:
        return

    # Delegate to command
    try:
        from commands import train_sft
        train_sft.train_sft_custom(
            model_choice=model_choice,
            dataset_choice=dataset_choice,
            learning_rate=float(learning_rate),
            batch_size=int(batch_size),
            grad_accum=int(grad_accum),
            epochs=int(epochs),
            max_samples=int(max_samples),
            warmup_steps=int(warmup_steps),
            max_grad_norm=float(max_grad_norm),
            weight_decay=float(weight_decay),
            adam_beta1=float(adam_beta1),
            adam_beta2=float(adam_beta2),
            logging_steps=int(logging_steps),
            save_steps=int(save_steps),
            save_total_limit=int(save_total_limit),
            output_dir=output_dir,
            use_lora=use_lora,
            fp16=use_fp16,
            bf16=use_bf16
        )
    except Exception as e:
        console.print(f"\n[bold red]✗ Training failed: {e}[/bold red]")
        import traceback
        console.print(f"[red]{traceback.format_exc()}[/red]")

    input("\nPress Enter to continue...")


def resume_sft_training():
    """Resume training from a checkpoint."""
    console.print("\n[bold cyan]🔄 Resume SFT Training[/bold cyan]\n")
    console.print("This will resume training from a saved checkpoint.\n")

    # Ask for checkpoint path
    checkpoint_path = questionary.text(
        "Checkpoint directory path:",
        default="checkpoints/sft/checkpoint-1000",
        style=custom_style
    ).ask()

    if checkpoint_path is None:
        return

    # Verify checkpoint exists
    from pathlib import Path
    checkpoint_dir = Path(checkpoint_path)
    if not checkpoint_dir.exists():
        console.print(f"[bold red]✗ Checkpoint not found: {checkpoint_path}[/bold red]")
        input("\nPress Enter to continue...")
        return

    # Check for training state file
    training_state = checkpoint_dir / "training_state.pt"
    if not training_state.exists():
        console.print(f"[bold red]✗ Training state not found in checkpoint[/bold red]")
        console.print(f"[yellow]The checkpoint may be incomplete or invalid.[/yellow]")
        input("\nPress Enter to continue...")
        return

    # Get dataset for continued training
    dataset_choice = questionary.text(
        "Dataset name (HuggingFace identifier):",
        default="yahma/alpaca-cleaned",
        style=custom_style
    ).ask()

    if dataset_choice is None:
        return

    # Additional epochs to train
    additional_epochs = questionary.text(
        "Additional epochs to train:",
        default="1",
        style=custom_style
    ).ask()

    # Confirm
    console.print("\n[bold yellow]Resume Training Configuration:[/bold yellow]")
    console.print(f"  Checkpoint: {checkpoint_path}")
    console.print(f"  Dataset: {dataset_choice}")
    console.print(f"  Additional epochs: {additional_epochs}")

    confirm = questionary.confirm(
        "\nResume training?",
        default=True,
        style=custom_style
    ).ask()

    if not confirm:
        return

    # Delegate to command
    try:
        from commands import train_sft
        train_sft.resume_sft_training(
            checkpoint_path=checkpoint_path,
            dataset_choice=dataset_choice,
            additional_epochs=int(additional_epochs)
        )
    except Exception as e:
        console.print(f"\n[bold red]✗ Resume failed: {e}[/bold red]")
        import traceback
        console.print(f"[red]{traceback.format_exc()}[/red]")

    input("\nPress Enter to continue...")


def dpo_menu():
    """Menu for Direct Preference Optimization"""
    console.print("\n[bold cyan]🎯 Direct Preference Optimization (DPO)[/bold cyan]\n")

    console.print("DPO optimizes models using preference data without requiring RLHF.\n")

    choices = [
        "1. Quick DPO training",
        "2. Standard DPO training",
        "3. Back to main menu"
    ]

    choice = questionary.select(
        "Select an option:",
        choices=choices,
        style=custom_style
    ).ask()

    if choice is None or "Back" in choice:
        return

    if "Quick" in choice:
        quick_dpo_training()
    elif "Standard" in choice:
        standard_dpo_training()


def quick_dpo_training():
    """Quick DPO training with sensible defaults."""
    console.print("\n[bold cyan]🎯 Quick DPO Training[/bold cyan]\n")
    console.print("This mode uses pre-configured DPO settings for quick experimentation.\n")

    # Model selection
    console.print("[cyan]Select base model (should be SFT-tuned):[/cyan]")
    model_choice = questionary.select(
        "Model:",
        choices=[
            "gpt2",
            "gpt2-medium",
            "Use custom checkpoint path",
            "Back to menu"
        ],
        style=custom_style
    ).ask()

    if model_choice == "Back to menu" or model_choice is None:
        return

    if "custom" in model_choice.lower():
        model_choice = questionary.text(
            "Enter checkpoint path:",
            default="checkpoints/sft/final",
            style=custom_style
        ).ask()

    # Dataset selection
    console.print("\n[cyan]Select preference dataset:[/cyan]")
    dataset_choice = questionary.select(
        "Dataset:",
        choices=[
            "Anthropic/hh-rlhf",
            "lvwerra/stack-exchange-paired",
            "Back to menu"
        ],
        style=custom_style
    ).ask()

    if dataset_choice == "Back to menu" or dataset_choice is None:
        return

    # Confirm settings
    console.print("\n[bold yellow]Quick DPO Configuration:[/bold yellow]")
    console.print(f"  Model: {model_choice}")
    console.print(f"  Dataset: {dataset_choice}")
    console.print(f"  Beta: 0.1 (KL penalty strength)")
    console.print(f"  Batch size: 4")
    console.print(f"  LoRA: Enabled")

    confirm = questionary.confirm(
        "\nStart DPO training?",
        default=True,
        style=custom_style
    ).ask()

    if not confirm:
        return

    # Delegate to command
    try:
        from commands import train_dpo
        train_dpo.train_dpo_quick(model_choice, dataset_choice)
    except Exception as e:
        console.print(f"\n[bold red]✗ Training failed: {e}[/bold red]")
        import traceback
        console.print(f"[red]{traceback.format_exc()}[/red]")

    input("\nPress Enter to continue...")


def standard_dpo_training():
    """Standard DPO training with configurable parameters."""
    console.print("\n[bold cyan]📋 Standard DPO Training[/bold cyan]\n")
    console.print("This mode allows you to configure DPO training parameters.\n")

    # Model selection
    console.print("[cyan]Select base model (should be SFT-tuned):[/cyan]")
    model_choice = questionary.select(
        "Model:",
        choices=[
            "gpt2",
            "gpt2-medium",
            "Use custom checkpoint path",
            "Back to menu"
        ],
        style=custom_style
    ).ask()

    if model_choice == "Back to menu" or model_choice is None:
        return

    if "custom" in model_choice.lower():
        model_choice = questionary.text(
            "Enter checkpoint path:",
            default="checkpoints/sft/final",
            style=custom_style
        ).ask()

    # Dataset selection
    console.print("\n[cyan]Select preference dataset:[/cyan]")
    dataset_choice = questionary.select(
        "Dataset:",
        choices=[
            "Anthropic/hh-rlhf",
            "lvwerra/stack-exchange-paired",
            "Back to menu"
        ],
        style=custom_style
    ).ask()

    if dataset_choice == "Back to menu" or dataset_choice is None:
        return

    # DPO-specific parameters
    beta = questionary.text(
        "Beta (KL penalty strength, typical: 0.1-0.5):",
        default="0.1",
        style=custom_style
    ).ask()

    label_smoothing = questionary.text(
        "Label smoothing (0 = no smoothing):",
        default="0.0",
        style=custom_style
    ).ask()

    # Training hyperparameters
    learning_rate = questionary.text(
        "Learning rate:",
        default="5e-7",
        style=custom_style
    ).ask()

    batch_size = questionary.text(
        "Batch size:",
        default="4",
        style=custom_style
    ).ask()

    grad_accum = questionary.text(
        "Gradient accumulation steps:",
        default="4",
        style=custom_style
    ).ask()

    epochs = questionary.text(
        "Number of epochs:",
        default="1",
        style=custom_style
    ).ask()

    max_samples = questionary.text(
        "Max samples (0 for all):",
        default="0",
        style=custom_style
    ).ask()

    # Advanced options
    warmup_steps = questionary.text(
        "Warmup steps:",
        default="50",
        style=custom_style
    ).ask()

    use_lora = questionary.confirm(
        "Use LoRA?",
        default=True,
        style=custom_style
    ).ask()

    # Confirm settings
    console.print("\n[bold yellow]Standard DPO Configuration:[/bold yellow]")
    console.print(f"  Model: {model_choice}")
    console.print(f"  Dataset: {dataset_choice}")
    console.print(f"  Beta: {beta}")
    console.print(f"  Label smoothing: {label_smoothing}")
    console.print(f"  Learning rate: {learning_rate}")
    console.print(f"  Batch size: {batch_size}")
    console.print(f"  Gradient accumulation: {grad_accum}")
    console.print(f"  Epochs: {epochs}")
    console.print(f"  Max samples: {max_samples if int(max_samples) > 0 else 'All'}")
    console.print(f"  Warmup steps: {warmup_steps}")
    console.print(f"  LoRA: {'Enabled' if use_lora else 'Disabled'}")

    confirm = questionary.confirm(
        "\nStart DPO training?",
        default=True,
        style=custom_style
    ).ask()

    if not confirm:
        return

    # Delegate to command
    try:
        from commands import train_dpo
        train_dpo.train_dpo_standard(
            model_choice=model_choice,
            dataset_choice=dataset_choice,
            beta=float(beta),
            label_smoothing=float(label_smoothing),
            learning_rate=float(learning_rate),
            batch_size=int(batch_size),
            grad_accum=int(grad_accum),
            epochs=int(epochs),
            max_samples=int(max_samples),
            warmup_steps=int(warmup_steps),
            use_lora=use_lora
        )
    except Exception as e:
        console.print(f"\n[bold red]✗ Training failed: {e}[/bold red]")
        import traceback
        console.print(f"[red]{traceback.format_exc()}[/red]")

    input("\nPress Enter to continue...")


def reward_model_menu():
    """Menu for Reward Model training"""
    console.print("\n[bold cyan]🏆 Train Reward Model[/bold cyan]\n")

    console.print("Train a reward model to predict human preferences.\n")

    choices = [
        "1. Quick reward model training",
        "2. Standard reward model training",
        "3. Back to main menu"
    ]

    choice = questionary.select(
        "Select an option:",
        choices=choices,
        style=custom_style
    ).ask()

    if choice is None or "Back" in choice:
        return

    if "Quick" in choice:
        quick_reward_model_training()
    elif "Standard" in choice:
        standard_reward_model_training()


def quick_reward_model_training():
    """Quick reward model training"""
    console.print("\n[bold cyan]🏆 Quick Reward Model Training[/bold cyan]\n")

    # Model selection
    console.print("[cyan]Select base model:[/cyan]")
    model_choice = questionary.select(
        "Model:",
        choices=[
            "gpt2",
            "gpt2-medium",
            "Use custom checkpoint",
            "Back to menu"
        ],
        style=custom_style
    ).ask()

    if model_choice == "Back to menu" or model_choice is None:
        return

    if "custom" in model_choice.lower():
        model_choice = questionary.text(
            "Enter model path or HF identifier:",
            default="gpt2-medium",
            style=custom_style
        ).ask()

    # Dataset selection
    console.print("\n[cyan]Select preference dataset:[/cyan]")
    dataset_choice = questionary.select(
        "Dataset:",
        choices=[
            "Anthropic/hh-rlhf",
            "Dahoas/rm-static",
            "lvwerra/stack-exchange-paired",
            "Back to menu"
        ],
        style=custom_style
    ).ask()

    if dataset_choice == "Back to menu" or dataset_choice is None:
        return

    # Confirm
    console.print("\n[bold yellow]Quick Reward Model Configuration:[/bold yellow]")
    console.print(f"  Base model: {model_choice}")
    console.print(f"  Dataset: {dataset_choice}")
    console.print(f"  Batch size: 4")
    console.print(f"  Learning rate: 1e-5")

    confirm = questionary.confirm(
        "\nStart training?",
        default=True,
        style=custom_style
    ).ask()

    if not confirm:
        return

    # Delegate to command
    try:
        from commands import train_reward
        train_reward.train_reward_quick(model_choice, dataset_choice)
    except Exception as e:
        console.print(f"\n[bold red]✗ Training failed: {e}[/bold red]")
        import traceback
        console.print(f"[red]{traceback.format_exc()}[/red]")

    input("\nPress Enter to continue...")


def standard_reward_model_training():
    """Standard reward model training with configurable parameters."""
    console.print("\n[bold cyan]📋 Standard Reward Model Training[/bold cyan]\n")
    console.print("This mode allows you to configure reward model training parameters.\n")

    # Model selection
    console.print("[cyan]Select base model:[/cyan]")
    model_choice = questionary.select(
        "Model:",
        choices=[
            "gpt2",
            "gpt2-medium",
            "Use custom checkpoint",
            "Back to menu"
        ],
        style=custom_style
    ).ask()

    if model_choice == "Back to menu" or model_choice is None:
        return

    if "custom" in model_choice.lower():
        model_choice = questionary.text(
            "Enter model path or HF identifier:",
            default="gpt2-medium",
            style=custom_style
        ).ask()

    # Dataset selection
    console.print("\n[cyan]Select preference dataset:[/cyan]")
    dataset_choice = questionary.select(
        "Dataset:",
        choices=[
            "Anthropic/hh-rlhf",
            "Dahoas/rm-static",
            "lvwerra/stack-exchange-paired",
            "Back to menu"
        ],
        style=custom_style
    ).ask()

    if dataset_choice == "Back to menu" or dataset_choice is None:
        return

    # Training hyperparameters
    learning_rate = questionary.text(
        "Learning rate:",
        default="1e-5",
        style=custom_style
    ).ask()

    batch_size = questionary.text(
        "Batch size:",
        default="4",
        style=custom_style
    ).ask()

    grad_accum = questionary.text(
        "Gradient accumulation steps:",
        default="4",
        style=custom_style
    ).ask()

    epochs = questionary.text(
        "Number of epochs:",
        default="1",
        style=custom_style
    ).ask()

    # Reward-specific parameters
    margin = questionary.text(
        "Ranking loss margin:",
        default="0.0",
        style=custom_style
    ).ask()

    max_samples = questionary.text(
        "Max samples (0 for all):",
        default="0",
        style=custom_style
    ).ask()

    warmup_steps = questionary.text(
        "Warmup steps:",
        default="100",
        style=custom_style
    ).ask()

    freeze_base = questionary.confirm(
        "Freeze base model (train only value head)?",
        default=False,
        style=custom_style
    ).ask()

    # Confirm
    console.print("\n[bold yellow]Standard Reward Model Configuration:[/bold yellow]")
    console.print(f"  Base model: {model_choice}")
    console.print(f"  Dataset: {dataset_choice}")
    console.print(f"  Learning rate: {learning_rate}")
    console.print(f"  Batch size: {batch_size}")
    console.print(f"  Gradient accumulation: {grad_accum}")
    console.print(f"  Epochs: {epochs}")
    console.print(f"  Margin: {margin}")
    console.print(f"  Max samples: {max_samples if int(max_samples) > 0 else 'All'}")
    console.print(f"  Warmup steps: {warmup_steps}")
    console.print(f"  Freeze base: {'Yes' if freeze_base else 'No'}")

    confirm = questionary.confirm(
        "\nStart training?",
        default=True,
        style=custom_style
    ).ask()

    if not confirm:
        return

    # Delegate to command
    try:
        from commands import train_reward
        train_reward.train_reward_standard(
            model_choice=model_choice,
            dataset_choice=dataset_choice,
            learning_rate=float(learning_rate),
            batch_size=int(batch_size),
            grad_accum=int(grad_accum),
            epochs=int(epochs),
            margin=float(margin),
            max_samples=int(max_samples),
            warmup_steps=int(warmup_steps),
            freeze_base=freeze_base
        )
    except Exception as e:
        console.print(f"\n[bold red]✗ Training failed: {e}[/bold red]")
        import traceback
        console.print(f"[red]{traceback.format_exc()}[/red]")

    input("\nPress Enter to continue...")


def rlhf_menu():
    """Menu for RLHF with PPO"""
    console.print("\n[bold cyan]🚀 RLHF with PPO[/bold cyan]\n")

    console.print(
        "RLHF (Reinforcement Learning from Human Feedback) uses PPO to optimize\n"
        "a language model according to a reward model trained on human preferences.\n"
    )

    choices = [
        "1. Quick PPO training (recommended defaults)",
        "2. Standard PPO training (configure parameters)",
        "3. Back to main menu"
    ]

    choice = questionary.select(
        "Select training mode:",
        choices=choices,
        style=custom_style
    ).ask()

    if choice is None or "Back" in choice:
        return

    if "Quick" in choice:
        quick_ppo_training()
    elif "Standard" in choice:
        standard_ppo_training()


def evaluate_menu():
    """Menu for model evaluation"""
    console.print("\n[bold cyan]📊 Evaluate Models[/bold cyan]\n")

    console.print("Evaluate trained models with various metrics.\n")

    choices = [
        "1. Compute perplexity on dataset",
        "2. Test generation quality",
        "3. Compare two models",
        "4. Back to main menu"
    ]

    choice = questionary.select(
        "Select an option:",
        choices=choices,
        style=custom_style
    ).ask()

    if choice is None or "Back" in choice:
        return

    if "perplexity" in choice.lower():
        evaluate_perplexity()
    elif "generation" in choice.lower():
        test_generation_quality()
    elif "Compare" in choice:
        compare_models()


def evaluate_perplexity():
    """Evaluate model perplexity"""
    console.print("\n[bold cyan]📈 Compute Perplexity[/bold cyan]\n")
    console.print("Compute perplexity on a dataset to measure model performance.\n")

    # Model selection
    console.print("[cyan]Select a model to evaluate:[/cyan]")
    from src.auto_bot_tuner.utils.model_loading import list_downloaded_models

    downloaded_models = list_downloaded_models()

    if not downloaded_models:
        use_custom = questionary.confirm(
            "No local models found. Do you want to use a HuggingFace model ID?",
            default=True,
            style=custom_style
        ).ask()

        if not use_custom:
            return

        model_path = questionary.text(
            "Model ID or path:",
            default="gpt2",
            style=custom_style
        ).ask()

        if not model_path:
            return
    else:
        choices = [f"models/{model}" for model in downloaded_models]
        choices.extend([
            "Use HuggingFace model ID or custom path",
            "Back to menu"
        ])

        choice = questionary.select(
            "Model:",
            choices=choices,
            style=custom_style
        ).ask()

        if choice == "Back to menu" or choice is None:
            return

        if choice == "Use HuggingFace model ID or custom path":
            model_path = questionary.text(
                "Model ID or path:",
                default="gpt2",
                style=custom_style
            ).ask()

            if not model_path:
                return
        else:
            model_path = choice

    # Dataset selection
    dataset_choice = questionary.text(
        "Dataset name (HuggingFace identifier):",
        default="wikitext",
        style=custom_style
    ).ask()

    if dataset_choice is None:
        return

    # Dataset config (optional)
    dataset_config = questionary.text(
        "Dataset config (leave blank if none):",
        default="wikitext-2-raw-v1",
        style=custom_style
    ).ask()

    if dataset_config == "":
        dataset_config = None

    # Max samples
    max_samples = questionary.text(
        "Max samples to evaluate (0 for all):",
        default="100",
        style=custom_style
    ).ask()

    # Confirm
    console.print("\n[bold yellow]Evaluation Configuration:[/bold yellow]")
    console.print(f"  Model: {model_path}")
    console.print(f"  Dataset: {dataset_choice}")
    if dataset_config:
        console.print(f"  Config: {dataset_config}")
    console.print(f"  Max samples: {max_samples if int(max_samples) > 0 else 'All'}")

    confirm = questionary.confirm(
        "\nStart evaluation?",
        default=True,
        style=custom_style
    ).ask()

    if not confirm:
        return

    # Delegate to command
    try:
        from commands import evaluate
        evaluate.evaluate_perplexity(
            model_path=model_path,
            dataset_name=dataset_choice,
            dataset_config=dataset_config,
            max_samples=int(max_samples)
        )
    except Exception as e:
        console.print(f"\n[bold red]✗ Evaluation failed: {e}[/bold red]")
        import traceback
        console.print(f"[red]{traceback.format_exc()}[/red]")

    input("\nPress Enter to continue...")


def test_generation_quality():
    """Test generation quality"""
    console.print("\n[bold cyan]✨ Test Generation Quality[/bold cyan]\n")
    console.print("Test model generation quality on sample prompts.\n")

    # Model selection
    console.print("[cyan]Select a model to test:[/cyan]")
    from src.auto_bot_tuner.utils.model_loading import list_downloaded_models

    downloaded_models = list_downloaded_models()

    if not downloaded_models:
        use_custom = questionary.confirm(
            "No local models found. Do you want to use a HuggingFace model ID?",
            default=True,
            style=custom_style
        ).ask()

        if not use_custom:
            return

        model_path = questionary.text(
            "Model ID or path:",
            default="gpt2",
            style=custom_style
        ).ask()

        if not model_path:
            return
    else:
        choices = [f"models/{model}" for model in downloaded_models]
        choices.extend([
            "Use HuggingFace model ID or custom path",
            "Back to menu"
        ])

        choice = questionary.select(
            "Model:",
            choices=choices,
            style=custom_style
        ).ask()

        if choice == "Back to menu" or choice is None:
            return

        if choice == "Use HuggingFace model ID or custom path":
            model_path = questionary.text(
                "Model ID or path:",
                default="gpt2",
                style=custom_style
            ).ask()

            if not model_path:
                return
        else:
            model_path = choice

    # Generation parameters
    max_new_tokens = questionary.text(
        "Max new tokens to generate:",
        default="100",
        style=custom_style
    ).ask()

    temperature = questionary.text(
        "Temperature (0.1-2.0):",
        default="0.7",
        style=custom_style
    ).ask()

    # Option to use custom prompts
    use_custom_prompts = questionary.confirm(
        "Do you want to provide custom prompts? (default prompts will be used if no)",
        default=False,
        style=custom_style
    ).ask()

    prompts = None
    if use_custom_prompts:
        console.print("\n[cyan]Enter your prompts (one per line, press Ctrl+D or enter blank line when done):[/cyan]")
        prompts = []
        while True:
            try:
                prompt = questionary.text(
                    f"Prompt {len(prompts) + 1} (or leave blank to finish):",
                    style=custom_style
                ).ask()

                if not prompt:
                    break
                prompts.append(prompt)
            except (EOFError, KeyboardInterrupt):
                break

        if not prompts:
            prompts = None

    # Confirm
    console.print("\n[bold yellow]Generation Configuration:[/bold yellow]")
    console.print(f"  Model: {model_path}")
    console.print(f"  Max new tokens: {max_new_tokens}")
    console.print(f"  Temperature: {temperature}")
    console.print(f"  Prompts: {'Custom' if prompts else 'Default'}")

    confirm = questionary.confirm(
        "\nStart generation?",
        default=True,
        style=custom_style
    ).ask()

    if not confirm:
        return

    # Delegate to command
    try:
        from commands import evaluate
        evaluate.test_generation_quality(
            model_path=model_path,
            prompts=prompts,
            max_new_tokens=int(max_new_tokens),
            temperature=float(temperature)
        )
    except Exception as e:
        console.print(f"\n[bold red]✗ Generation failed: {e}[/bold red]")
        import traceback
        console.print(f"[red]{traceback.format_exc()}[/red]")

    input("\nPress Enter to continue...")


def compare_models():
    """Compare two models"""
    console.print("\n[bold cyan]⚖️  Compare Two Models[/bold cyan]\n")
    console.print("Compare outputs from two different models on the same prompts.\n")

    from src.auto_bot_tuner.utils.model_loading import list_downloaded_models

    # Helper function to select a model
    def select_model(prompt_text: str, default_id: str = "gpt2"):
        downloaded_models = list_downloaded_models()

        if not downloaded_models:
            use_custom = questionary.confirm(
                f"{prompt_text}: No local models found. Use a HuggingFace model ID?",
                default=True,
                style=custom_style
            ).ask()

            if not use_custom:
                return None

            model_path = questionary.text(
                "Model ID or path:",
                default=default_id,
                style=custom_style
            ).ask()

            return model_path
        else:
            choices = [f"models/{model}" for model in downloaded_models]
            choices.extend([
                "Use HuggingFace model ID or custom path",
                "Back to menu"
            ])

            choice = questionary.select(
                f"{prompt_text}:",
                choices=choices,
                style=custom_style
            ).ask()

            if choice == "Back to menu" or choice is None:
                return None

            if choice == "Use HuggingFace model ID or custom path":
                model_path = questionary.text(
                    "Model ID or path:",
                    default=default_id,
                    style=custom_style
                ).ask()

                return model_path
            else:
                return choice

    # Select first model
    console.print("[cyan]Select the first model:[/cyan]")
    model_a_path = select_model("First model", "gpt2")

    if not model_a_path:
        return

    # Select second model
    console.print("\n[cyan]Select the second model:[/cyan]")
    model_b_path = select_model("Second model", "gpt2-medium")

    if not model_b_path:
        return

    # Generation parameters
    max_new_tokens = questionary.text(
        "Max new tokens to generate:",
        default="80",
        style=custom_style
    ).ask()

    # Option to use custom prompts
    use_custom_prompts = questionary.confirm(
        "Do you want to provide custom prompts? (default prompts will be used if no)",
        default=False,
        style=custom_style
    ).ask()

    prompts = None
    if use_custom_prompts:
        console.print("\n[cyan]Enter your prompts (one per line, press Ctrl+D or enter blank line when done):[/cyan]")
        prompts = []
        while True:
            try:
                prompt = questionary.text(
                    f"Prompt {len(prompts) + 1} (or leave blank to finish):",
                    style=custom_style
                ).ask()

                if not prompt:
                    break
                prompts.append(prompt)
            except (EOFError, KeyboardInterrupt):
                break

        if not prompts:
            prompts = None

    # Confirm
    console.print("\n[bold yellow]Comparison Configuration:[/bold yellow]")
    console.print(f"  Model A: {model_a_path}")
    console.print(f"  Model B: {model_b_path}")
    console.print(f"  Max new tokens: {max_new_tokens}")
    console.print(f"  Prompts: {'Custom' if prompts else 'Default'}")

    confirm = questionary.confirm(
        "\nStart comparison?",
        default=True,
        style=custom_style
    ).ask()

    if not confirm:
        return

    # Delegate to command
    try:
        from commands import evaluate
        evaluate.compare_models(
            model_a_path=model_a_path,
            model_b_path=model_b_path,
            prompts=prompts,
            max_new_tokens=int(max_new_tokens)
        )
    except Exception as e:
        console.print(f"\n[bold red]✗ Comparison failed: {e}[/bold red]")
        import traceback
        console.print(f"[red]{traceback.format_exc()}[/red]")

    input("\nPress Enter to continue...")


def chat_menu():
    """Menu for chatting with models"""
    console.print("\n[bold cyan]💬 Chat with Model[/bold cyan]\n")

    # Get list of downloaded models
    from src.auto_bot_tuner.utils.model_loading import list_downloaded_models

    downloaded_models = list_downloaded_models()

    if not downloaded_models:
        console.print("[yellow]No models found in the models/ directory.[/yellow]")
        console.print("[dim]You can download models from the main menu or specify a HuggingFace model ID.[/dim]\n")

        use_custom = questionary.confirm(
            "Do you want to use a HuggingFace model ID instead?",
            default=True,
            style=custom_style
        ).ask()

        if not use_custom:
            return

        model_path = questionary.text(
            "Model ID or path:",
            default="gpt2",
            style=custom_style
        ).ask()

        if model_path is None:
            return
    else:
        # Show model selection menu
        console.print("[cyan]Select a model to chat with:[/cyan]\n")

        choices = [f"models/{model}" for model in downloaded_models]
        choices.append("Use HuggingFace model ID or custom path")
        choices.append("← Back to Main Menu")

        choice = questionary.select(
            "Model:",
            choices=choices,
            style=custom_style
        ).ask()

        if choice is None or "Back" in choice:
            return

        if "HuggingFace" in choice:
            model_path = questionary.text(
                "Model ID or path:",
                default="gpt2",
                style=custom_style
            ).ask()

            if model_path is None:
                return
        else:
            model_path = choice

    # Delegate to command
    try:
        from commands import chat
        chat.chat_with_model(model_path)
    except Exception as e:
        console.print(f"\n[bold red]✗ Error: {e}[/bold red]")
        import traceback
        console.print(f"[red]{traceback.format_exc()}[/red]")

    input("\nPress Enter to continue...")


def quick_ppo_training():
    """Quick PPO training with recommended defaults"""
    console.print("\n[bold cyan]🚀 Quick PPO Training[/bold cyan]\n")

    console.print(
        "[dim]This will train a model using PPO (RLHF) with recommended defaults.\n"
        "You'll need a trained SFT model and reward model checkpoint.[/dim]\n"
    )

    # Model selection
    console.print("[cyan]Select SFT model (policy):[/cyan]")
    model_choice = questionary.select(
        "SFT Model:",
        choices=[
            "checkpoints/sft/final (trained checkpoint)",
            "gpt2 (use base model - for demo only)",
            "gpt2-medium",
            "Back to menu"
        ],
        style=custom_style
    ).ask()

    if model_choice == "Back to menu" or model_choice is None:
        return

    # Extract model path
    if "(trained checkpoint)" in model_choice:
        sft_model = "checkpoints/sft/final"
    elif "(use base model" in model_choice:
        sft_model = "gpt2"
    else:
        sft_model = model_choice

    # Reward model selection
    console.print("\n[cyan]Select reward model:[/cyan]")
    reward_choice = questionary.select(
        "Reward Model:",
        choices=[
            "checkpoints/reward/final (trained checkpoint)",
            "gpt2 (create fresh reward model - for demo only)",
            "Back to menu"
        ],
        style=custom_style
    ).ask()

    if reward_choice == "Back to menu" or reward_choice is None:
        return

    if "(trained checkpoint)" in reward_choice:
        reward_model = "checkpoints/reward/final"
    else:
        reward_model = "gpt2"

    # Dataset selection for prompts
    console.print("\n[cyan]Select dataset for prompts:[/cyan]")
    dataset_choice = questionary.select(
        "Dataset:",
        choices=[
            "Anthropic/hh-rlhf",
            "Dahoas/rm-static",
            "OpenAssistant/oasst1",
            "Back to menu"
        ],
        style=custom_style
    ).ask()

    if dataset_choice == "Back to menu" or dataset_choice is None:
        return

    # Confirm
    console.print("\n[bold yellow]Quick PPO Configuration:[/bold yellow]")
    console.print(f"  SFT Model: {sft_model}")
    console.print(f"  Reward Model: {reward_model}")
    console.print(f"  Dataset: {dataset_choice}")
    console.print(f"  Batch size: 4")
    console.print(f"  PPO epochs: 4")
    console.print(f"  Number of rollouts: 100")
    console.print(f"  Learning rate: 1e-6")

    confirm = questionary.confirm(
        "\nStart training?",
        default=True,
        style=custom_style
    ).ask()

    if not confirm:
        return

    # Delegate to command
    try:
        from commands import train_ppo
        train_ppo.train_ppo_quick(sft_model, reward_model, dataset_choice)
    except Exception as e:
        console.print(f"\n[bold red]✗ Training failed: {e}[/bold red]")
        import traceback
        console.print(f"[red]{traceback.format_exc()}[/red]")

    input("\nPress Enter to continue...")


def standard_ppo_training():
    """Standard PPO training with configurable parameters"""
    console.print("\n[bold cyan]🚀 Standard PPO Training[/bold cyan]\n")
    console.print("This mode allows you to configure PPO training parameters.\n")

    # Model selection
    console.print("[cyan]Select SFT model (policy):[/cyan]")
    model_choice = questionary.select(
        "SFT Model:",
        choices=[
            "checkpoints/sft/final (trained checkpoint)",
            "gpt2 (use base model - for demo only)",
            "gpt2-medium",
            "Custom path",
            "Back to menu"
        ],
        style=custom_style
    ).ask()

    if model_choice == "Back to menu" or model_choice is None:
        return

    if "Custom" in model_choice:
        model_choice = questionary.text(
            "Enter SFT model path:",
            default="checkpoints/sft/final",
            style=custom_style
        ).ask()
    elif "(trained checkpoint)" in model_choice:
        model_choice = "checkpoints/sft/final"
    elif "(use base model" in model_choice:
        model_choice = "gpt2"

    # Reward model selection
    console.print("\n[cyan]Select reward model:[/cyan]")
    reward_choice = questionary.select(
        "Reward Model:",
        choices=[
            "checkpoints/reward/final (trained checkpoint)",
            "gpt2 (create fresh - for demo only)",
            "Custom path",
            "Back to menu"
        ],
        style=custom_style
    ).ask()

    if reward_choice == "Back to menu" or reward_choice is None:
        return

    if "Custom" in reward_choice:
        reward_model = questionary.text(
            "Enter reward model path:",
            default="checkpoints/reward/final",
            style=custom_style
        ).ask()
    elif "(trained checkpoint)" in reward_choice:
        reward_model = "checkpoints/reward/final"
    else:
        reward_model = "gpt2"

    # Dataset selection
    console.print("\n[cyan]Select dataset for prompts:[/cyan]")
    dataset_choice = questionary.select(
        "Dataset:",
        choices=[
            "Anthropic/hh-rlhf",
            "Dahoas/rm-static",
            "OpenAssistant/oasst1",
            "Back to menu"
        ],
        style=custom_style
    ).ask()

    if dataset_choice == "Back to menu" or dataset_choice is None:
        return

    # PPO-specific parameters
    clip_ratio = questionary.text(
        "PPO clip ratio (epsilon):",
        default="0.2",
        style=custom_style
    ).ask()

    vf_coef = questionary.text(
        "Value function coefficient:",
        default="0.5",
        style=custom_style
    ).ask()

    entropy_coef = questionary.text(
        "Entropy coefficient:",
        default="0.01",
        style=custom_style
    ).ask()

    kl_coef = questionary.text(
        "KL penalty coefficient:",
        default="0.1",
        style=custom_style
    ).ask()

    # Training parameters
    learning_rate = questionary.text(
        "Learning rate:",
        default="1e-6",
        style=custom_style
    ).ask()

    batch_size = questionary.text(
        "Batch size (for rollouts):",
        default="4",
        style=custom_style
    ).ask()

    ppo_epochs = questionary.text(
        "PPO epochs per rollout:",
        default="4",
        style=custom_style
    ).ask()

    num_rollouts = questionary.text(
        "Number of rollouts:",
        default="100",
        style=custom_style
    ).ask()

    # Generation parameters
    max_new_tokens = questionary.text(
        "Max new tokens per generation:",
        default="128",
        style=custom_style
    ).ask()

    temperature = questionary.text(
        "Temperature:",
        default="1.0",
        style=custom_style
    ).ask()

    # Confirm
    console.print("\n[bold yellow]Standard PPO Configuration:[/bold yellow]")
    console.print(f"  SFT Model: {model_choice}")
    console.print(f"  Reward Model: {reward_model}")
    console.print(f"  Dataset: {dataset_choice}")
    console.print(f"  Clip ratio: {clip_ratio}")
    console.print(f"  Value coef: {vf_coef}")
    console.print(f"  Entropy coef: {entropy_coef}")
    console.print(f"  KL coef: {kl_coef}")
    console.print(f"  Learning rate: {learning_rate}")
    console.print(f"  Batch size: {batch_size}")
    console.print(f"  PPO epochs: {ppo_epochs}")
    console.print(f"  Num rollouts: {num_rollouts}")
    console.print(f"  Max new tokens: {max_new_tokens}")
    console.print(f"  Temperature: {temperature}")

    confirm = questionary.confirm(
        "\nStart training?",
        default=True,
        style=custom_style
    ).ask()

    if not confirm:
        return

    # Delegate to command
    try:
        from commands import train_ppo
        train_ppo.train_ppo_standard(
            sft_model=model_choice,
            reward_model_path=reward_model,
            dataset_name=dataset_choice,
            clip_ratio=float(clip_ratio),
            vf_coef=float(vf_coef),
            entropy_coef=float(entropy_coef),
            kl_coef=float(kl_coef),
            learning_rate=float(learning_rate),
            batch_size=int(batch_size),
            ppo_epochs=int(ppo_epochs),
            num_rollouts=int(num_rollouts),
            max_new_tokens=int(max_new_tokens),
            temperature=float(temperature)
        )
    except Exception as e:
        console.print(f"\n[bold red]✗ Training failed: {e}[/bold red]")
        import traceback
        console.print(f"[red]{traceback.format_exc()}[/red]")

    input("\nPress Enter to continue...")


def download_data_menu():
    """Menu for downloading datasets"""
    console.print("\n[bold cyan]📥 Download Datasets[/bold cyan]\n")

    datasets = [
        {
            "name": "Alpaca (Instruction Tuning)",
            "id": "tatsu-lab/alpaca",
            "size": "~52K examples",
            "desc": "Classic instruction following dataset"
        },
        {
            "name": "Alpaca Cleaned (Instruction Tuning)",
            "id": "yahma/alpaca-cleaned",
            "size": "~52K examples",
            "desc": "Cleaned version of Alpaca dataset"
        },
        {
            "name": "OpenAssistant (Preference Data)",
            "id": "OpenAssistant/oasst1",
            "size": "~161K messages",
            "desc": "Human preferences for RLHF/DPO"
        },
        {
            "name": "Anthropic HH-RLHF",
            "id": "Anthropic/hh-rlhf",
            "size": "~169K comparisons",
            "desc": "Helpful and harmless preferences"
        },
        {
            "name": "Dahoas RM Static",
            "id": "Dahoas/rm-static",
            "size": "~76K pairs",
            "desc": "Reward model training dataset"
        },
        {
            "name": "Stack Exchange Paired",
            "id": "lvwerra/stack-exchange-paired",
            "size": "~300K pairs",
            "desc": "Stack Exchange preference pairs"
        },
    ]

    table = Table(title="Available Datasets", show_header=True)
    table.add_column("Dataset", style="cyan")
    table.add_column("Size", style="magenta")
    table.add_column("Description", style="white")

    for ds in datasets:
        table.add_row(ds["name"], ds["size"], ds["desc"])

    console.print(table)
    console.print()

    # Select dataset
    choices = [ds["name"] for ds in datasets] + ["← Back to Main Menu"]

    choice = questionary.select(
        "Which dataset would you like to download?",
        choices=choices,
        style=custom_style
    ).ask()

    if choice is None or "Back" in choice:
        return

    # Find selected dataset
    selected = next(ds for ds in datasets if ds["name"] == choice)

    # Ask about cache location
    console.print(f"\n[cyan]Dataset will be downloaded and cached locally.[/cyan]")
    console.print(f"[dim]HuggingFace datasets are cached in: ~/.cache/huggingface/datasets[/dim]\n")

    # Ask about splits
    console.print("[cyan]Select split(s) to download:[/cyan]")
    split_choice = questionary.select(
        "Split:",
        choices=[
            "All splits (train, test, validation)",
            "Train only",
            "Test only",
            "Validation only",
            "Back to menu"
        ],
        style=custom_style
    ).ask()

    if split_choice is None or "Back" in split_choice:
        return

    # Determine splits to download
    if "All" in split_choice:
        splits = None  # Download all
    elif "Train" in split_choice:
        splits = ["train"]
    elif "Test" in split_choice:
        splits = ["test"]
    elif "Validation" in split_choice:
        splits = ["validation"]

    # Confirm download
    console.print("\n[bold yellow]Download Configuration:[/bold yellow]")
    console.print(f"  Dataset: {selected['name']}")
    console.print(f"  ID: {selected['id']}")
    console.print(f"  Size: {selected['size']}")
    console.print(f"  Splits: {split_choice}")

    confirm = questionary.confirm(
        "\nStart download?",
        default=True,
        style=custom_style
    ).ask()

    if not confirm:
        return

    # Download dataset
    try:
        from datasets import load_dataset

        console.print(f"\n[cyan]Downloading {selected['id']}...[/cyan]")
        console.print("[dim]This may take a few minutes depending on dataset size...[/dim]\n")

        if splits is None:
            # Download all splits
            dataset = load_dataset(selected['id'])
            console.print(f"\n[bold green]✓ Dataset downloaded successfully![/bold green]")
            console.print(f"  Available splits: {list(dataset.keys())}")
            for split_name, split_data in dataset.items():
                console.print(f"  - {split_name}: {len(split_data)} examples")
        else:
            # Download specific split
            for split in splits:
                try:
                    dataset = load_dataset(selected['id'], split=split)
                    console.print(f"\n[bold green]✓ {split} split downloaded successfully![/bold green]")
                    console.print(f"  Examples: {len(dataset)}")
                except Exception as e:
                    console.print(f"[yellow]Warning: Could not download {split} split: {e}[/yellow]")

        console.print(f"\n[green]Dataset cached at: ~/.cache/huggingface/datasets/{selected['id'].replace('/', '___')}[/green]")

    except Exception as e:
        console.print(f"\n[bold red]✗ Download failed: {e}[/bold red]")
        import traceback
        console.print(f"[red]{traceback.format_exc()}[/red]")

    input("\nPress Enter to continue...")


def download_combined_menu():
    """Combined menu for downloading models or datasets"""
    console.print("\n[bold cyan]📥 Download Resources[/bold cyan]\n")

    choice = questionary.select(
        "What would you like to download?",
        choices=[
            "Pre-trained Model (GPT-2, Llama, Qwen)",
            "Dataset (Alpaca, HH-RLHF, etc.)",
            "← Back to Main Menu"
        ],
        style=custom_style
    ).ask()

    if choice is None or "Back" in choice:
        return
    elif "Model" in choice:
        download_model_menu()
    elif "Dataset" in choice:
        download_data_menu()


def browse_models_menu():
    """
    Browse all registered models in the model registry.

    Shows a tree view of all models organized by base model,
    with their training lineage, capabilities, and checkpoints.
    """
    try:
        from wizard import ProgressTracker, render_model_browser
        tracker = ProgressTracker()

        console.print("\n")
        render_model_browser(tracker.registry)

        # Offer actions
        console.print("[bold]Actions:[/bold]")
        console.print("  [1] View model details")
        console.print("  [2] Back to main menu")

        action = questionary.select(
            "Select an action:",
            choices=["View model details", "Back to main menu"],
            style=custom_style
        ).ask()

        if action and "View" in action:
            # Get list of models
            models = tracker.registry.get_all_models()
            if models:
                model_choices = [m.id for m in models] + ["← Back"]

                selected = questionary.select(
                    "Select a model:",
                    choices=model_choices,
                    style=custom_style
                ).ask()

                if selected and selected != "← Back":
                    from wizard import render_model_details
                    model = tracker.registry.get_model(selected)
                    render_model_details(model)
                    input("\nPress Enter to continue...")

    except Exception as e:
        console.print(f"\n[bold red]✗ Error browsing models: {e}[/bold red]")
        import traceback
        console.print(f"[red]{traceback.format_exc()}[/red]")

    input("\nPress Enter to continue...")
