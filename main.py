#!/usr/bin/env python3
"""
LLM Post-Training - Command-Line Interface

Educational project for learning how post-training works:
- Supervised Fine-Tuning (SFT)
- Reward Model Training
- RLHF with PPO
- Direct Preference Optimization (DPO)

Run without arguments for interactive mode, or use subcommands for CLI mode.
"""

import sys
import argparse
from pathlib import Path


def create_parser():
    """
    Create the argument parser with all subcommands.

    Returns:
        argparse.ArgumentParser: Configured argument parser
    """
    parser = argparse.ArgumentParser(
        description="LLM Post-Training - Educational toolkit for fine-tuning language models",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # ===== SFT Command =====
    sft_parser = subparsers.add_parser(
        "train-sft",
        help="Supervised Fine-Tuning",
        description="Train model using Supervised Fine-Tuning"
    )
    sft_parser.add_argument("--model", type=str, required=True, help="Model name or path")
    sft_parser.add_argument("--dataset", type=str, required=True, help="Dataset name")
    sft_parser.add_argument("--preset", type=str, choices=["quick", "standard"], default="standard")
    sft_parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    sft_parser.add_argument("--grad-accum", type=int, default=4, help="Gradient accumulation")
    sft_parser.add_argument("--learning-rate", type=float, default=2e-5, help="Learning rate")
    sft_parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    sft_parser.add_argument("--max-samples", type=int, default=0, help="Max samples (0 for all)")
    sft_parser.add_argument("--no-lora", action="store_true", help="Disable LoRA")

    # ===== DPO Command =====
    dpo_parser = subparsers.add_parser(
        "train-dpo",
        help="Direct Preference Optimization",
        description="Train model using DPO"
    )
    dpo_parser.add_argument("--model", type=str, required=True, help="Model checkpoint path")
    dpo_parser.add_argument("--dataset", type=str, required=True, help="Preference dataset")
    dpo_parser.add_argument("--preset", type=str, choices=["quick", "standard"], default="standard")
    dpo_parser.add_argument("--beta", type=float, default=0.1, help="KL penalty strength")
    dpo_parser.add_argument("--learning-rate", type=float, default=5e-7, help="Learning rate")
    dpo_parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    dpo_parser.add_argument("--max-samples", type=int, default=0, help="Max samples")
    dpo_parser.add_argument("--no-lora", action="store_true", help="Disable LoRA")

    # ===== Reward Model Command =====
    reward_parser = subparsers.add_parser(
        "train-reward",
        help="Train Reward Model",
        description="Train a reward model to predict human preferences"
    )
    reward_parser.add_argument("--model", type=str, required=True, help="Base model")
    reward_parser.add_argument("--dataset", type=str, required=True, help="Preference dataset")
    reward_parser.add_argument("--preset", type=str, choices=["quick", "standard"], default="standard")
    reward_parser.add_argument("--learning-rate", type=float, default=1e-5, help="Learning rate")
    reward_parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    reward_parser.add_argument("--margin", type=float, default=0.0, help="Ranking loss margin")
    reward_parser.add_argument("--max-samples", type=int, default=0, help="Max samples")
    reward_parser.add_argument("--freeze-base", action="store_true", help="Freeze base model")

    # ===== Evaluate Command =====
    eval_parser = subparsers.add_parser(
        "evaluate",
        help="Evaluate models",
        description="Evaluate trained models"
    )
    eval_subparsers = eval_parser.add_subparsers(dest="eval_command", help="Evaluation type")

    # Perplexity
    perplexity_parser = eval_subparsers.add_parser("perplexity", help="Compute perplexity")
    perplexity_parser.add_argument("--model", type=str, required=True, help="Model path")
    perplexity_parser.add_argument("--dataset", type=str, required=True, help="Dataset name")
    perplexity_parser.add_argument("--config", type=str, help="Dataset config")
    perplexity_parser.add_argument("--max-samples", type=int, default=100, help="Max samples")

    # Generation quality
    generation_parser = eval_subparsers.add_parser("generation", help="Test generation quality")
    generation_parser.add_argument("--model", type=str, required=True, help="Model path")
    generation_parser.add_argument("--prompt", type=str, action="append", help="Test prompt")
    generation_parser.add_argument("--max-tokens", type=int, default=100, help="Max new tokens")
    generation_parser.add_argument("--temperature", type=float, default=0.7, help="Temperature")

    # Compare models
    compare_parser = eval_subparsers.add_parser("compare", help="Compare two models")
    compare_parser.add_argument("--model-a", type=str, required=True, help="First model")
    compare_parser.add_argument("--model-b", type=str, required=True, help="Second model")
    compare_parser.add_argument("--prompt", type=str, action="append", help="Test prompt")
    compare_parser.add_argument("--max-tokens", type=int, default=80, help="Max new tokens")

    # ===== Chat Command =====
    chat_parser = subparsers.add_parser(
        "chat",
        help="Chat with a model",
        description="Interactive chat with a trained model"
    )
    chat_parser.add_argument("model", type=str, help="Model path or identifier")
    chat_parser.add_argument("--max-tokens", type=int, default=150, help="Max tokens per response")
    chat_parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")

    # ===== Download Command =====
    download_parser = subparsers.add_parser(
        "download",
        help="Download models",
        description="Download models and datasets"
    )
    download_subparsers = download_parser.add_subparsers(dest="download_command", help="Download type")

    # Download model
    model_dl_parser = download_subparsers.add_parser("model", help="Download a model")
    model_dl_parser.add_argument("model_id", type=str, help="HuggingFace model identifier")

    # List models
    download_subparsers.add_parser("list", help="List downloaded models")

    # ===== Convert Command =====
    convert_parser = subparsers.add_parser(
        "convert",
        help="Convert and export checkpoints",
        description="Convert checkpoints between formats"
    )
    convert_subparsers = convert_parser.add_subparsers(dest="convert_command", help="Conversion type")

    # Merge LoRA
    merge_parser = convert_subparsers.add_parser("merge-lora", help="Merge LoRA with base model")
    merge_parser.add_argument("--base-model", type=str, required=True, help="Base model path")
    merge_parser.add_argument("--lora-checkpoint", type=str, required=True, help="LoRA checkpoint path")
    merge_parser.add_argument("--output", type=str, required=True, help="Output directory")
    merge_parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda", "mps"])

    # Export checkpoint
    export_parser = convert_subparsers.add_parser("export", help="Export checkpoint to format")
    export_parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint path")
    export_parser.add_argument("--output", type=str, required=True, help="Output directory")
    export_parser.add_argument("--format", type=str, default="safetensors",
                               choices=["safetensors", "pytorch", "gguf"])

    # ===== Train from Config Command =====
    config_parser = subparsers.add_parser(
        "train-config",
        help="Train from YAML config",
        description="Train models using YAML configuration files"
    )
    config_parser.add_argument("config", type=str, help="Path to YAML config file")
    config_parser.add_argument("--type", type=str, choices=["sft", "dpo", "reward", "ppo"],
                               help="Training type (auto-detected if not specified)")

    return parser


def main():
    """
    Main entry point.

    If no arguments provided, launches interactive CLI.
    Otherwise, routes to appropriate command based on arguments.
    """
    # If no arguments, launch interactive mode
    if len(sys.argv) == 1:
        from src.interactive import interactive_main
        interactive_main()
        return

    # Parse command-line arguments
    parser = create_parser()
    args = parser.parse_args()

    # Route to appropriate command
    try:
        if args.command == "train-sft":
            from commands import train_sft
            if args.preset == "quick":
                train_sft.train_sft_quick(args.model, args.dataset)
            else:
                train_sft.train_sft_standard(
                    model_choice=args.model,
                    dataset_choice=args.dataset,
                    batch_size=args.batch_size,
                    grad_accum=args.grad_accum,
                    learning_rate=args.learning_rate,
                    epochs=args.epochs,
                    max_samples=args.max_samples,
                    use_lora=not args.no_lora
                )

        elif args.command == "train-dpo":
            from commands import train_dpo
            if args.preset == "quick":
                train_dpo.train_dpo_quick(args.model, args.dataset)
            else:
                train_dpo.train_dpo_standard(
                    model_choice=args.model,
                    dataset_choice=args.dataset,
                    beta=args.beta,
                    learning_rate=args.learning_rate,
                    batch_size=args.batch_size,
                    max_samples=args.max_samples,
                    use_lora=not args.no_lora
                )

        elif args.command == "train-reward":
            from commands import train_reward
            if args.preset == "quick":
                train_reward.train_reward_quick(args.model, args.dataset)
            else:
                train_reward.train_reward_standard(
                    model_choice=args.model,
                    dataset_choice=args.dataset,
                    learning_rate=args.learning_rate,
                    batch_size=args.batch_size,
                    margin=args.margin,
                    max_samples=args.max_samples,
                    freeze_base=args.freeze_base
                )

        elif args.command == "evaluate":
            from commands import evaluate
            if args.eval_command == "perplexity":
                evaluate.evaluate_perplexity(
                    model_path=args.model,
                    dataset_name=args.dataset,
                    dataset_config=args.config,
                    max_samples=args.max_samples
                )
            elif args.eval_command == "generation":
                evaluate.test_generation_quality(
                    model_path=args.model,
                    prompts=args.prompt,
                    max_new_tokens=args.max_tokens,
                    temperature=args.temperature
                )
            elif args.eval_command == "compare":
                evaluate.compare_models(
                    model_a_path=args.model_a,
                    model_b_path=args.model_b,
                    prompts=args.prompt,
                    max_new_tokens=args.max_tokens
                )
            else:
                parser.parse_args(["evaluate", "--help"])

        elif args.command == "chat":
            from commands import chat
            chat.chat_with_model(
                model_path=args.model,
                max_new_tokens=args.max_tokens,
                temperature=args.temperature
            )

        elif args.command == "download":
            from commands import download
            if args.download_command == "model":
                download.download_model(args.model_id)
            elif args.download_command == "list":
                download.list_downloaded_models()
            else:
                parser.parse_args(["download", "--help"])

        elif args.command == "convert":
            from commands import convert
            if args.convert_command == "merge-lora":
                convert.merge_lora_checkpoint(
                    base_model_path=args.base_model,
                    lora_checkpoint_path=args.lora_checkpoint,
                    output_path=args.output,
                    device=args.device
                )
            elif args.convert_command == "export":
                convert.export_checkpoint(
                    checkpoint_path=args.checkpoint,
                    output_path=args.output,
                    format=args.format
                )
            else:
                parser.parse_args(["convert", "--help"])

        elif args.command == "train-config":
            from commands import train_config
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

            if training_type == "sft":
                train_config.train_sft_from_config(args.config)
            elif training_type == "dpo":
                train_config.train_dpo_from_config(args.config)
            elif training_type == "reward":
                train_config.train_reward_model_from_config(args.config)
            elif training_type == "ppo":
                train_config.train_ppo_from_config(args.config)

        else:
            parser.print_help()

    except KeyboardInterrupt:
        from rich.console import Console
        console = Console()
        console.print("\n\n[cyan]Interrupted by user[/cyan]")
        sys.exit(0)
    except Exception as e:
        from rich.console import Console
        console = Console()
        console.print(f"\n[bold red]Error: {e}[/bold red]")
        import traceback
        console.print(f"[red]{traceback.format_exc()}[/red]")
        sys.exit(1)


if __name__ == "__main__":
    main()
