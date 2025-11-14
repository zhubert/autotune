"""
Checkpoint conversion command implementation.

This module provides functions for converting and exporting model checkpoints.
"""

import sys
from pathlib import Path
from rich.console import Console

# Add src directory to Python path for imports
src_path = Path(__file__).parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

console = Console()


def merge_lora_checkpoint(
    base_model_path: str,
    lora_checkpoint_path: str,
    output_path: str,
    device: str = "cpu"
):
    """
    Merge LoRA adapter with base model to create standalone model.

    Args:
        base_model_path: Path to base model or HuggingFace model ID
        lora_checkpoint_path: Path to LoRA checkpoint directory
        output_path: Where to save merged model
        device: Device to load models on

    Returns:
        bool: True if conversion succeeded, False otherwise
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    try:
        console.print(f"\n[cyan]Loading base model from: {base_model_path}[/cyan]")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.bfloat16,
            device_map=device
        )
        console.print("[green]✓ Base model loaded![/green]")

        console.print(f"\n[cyan]Loading LoRA adapter from: {lora_checkpoint_path}[/cyan]")
        model = PeftModel.from_pretrained(base_model, lora_checkpoint_path)
        console.print("[green]✓ LoRA adapter loaded![/green]")

        console.print("\n[cyan]Merging LoRA weights with base model...[/cyan]")
        merged_model = model.merge_and_unload()
        console.print("[green]✓ Weights merged![/green]")

        console.print(f"\n[cyan]Saving merged model to: {output_path}[/cyan]")
        Path(output_path).mkdir(parents=True, exist_ok=True)
        merged_model.save_pretrained(output_path)

        # Also save tokenizer for convenience
        console.print("[cyan]Saving tokenizer...[/cyan]")
        tokenizer = AutoTokenizer.from_pretrained(base_model_path)
        tokenizer.save_pretrained(output_path)

        console.print("\n[bold green]✓ Conversion complete![/bold green]")
        console.print(f"[dim]Merged model saved to: {output_path}[/dim]")
        console.print("[dim]You can now use this model without LoRA dependencies[/dim]\n")

        return True

    except Exception as e:
        console.print(f"\n[bold red]✗ Conversion failed: {e}[/bold red]")
        import traceback
        console.print(f"[red]{traceback.format_exc()}[/red]")
        return False


def export_checkpoint(
    checkpoint_path: str,
    output_path: str,
    format: str = "safetensors"
):
    """
    Export checkpoint to different format.

    Args:
        checkpoint_path: Path to checkpoint directory
        output_path: Where to save exported model
        format: Export format (safetensors, pytorch, gguf)

    Returns:
        bool: True if export succeeded, False otherwise
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    try:
        console.print(f"\n[cyan]Loading checkpoint from: {checkpoint_path}[/cyan]")
        model = AutoModelForCausalLM.from_pretrained(checkpoint_path)
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
        console.print("[green]✓ Checkpoint loaded![/green]")

        console.print(f"\n[cyan]Exporting to {format} format...[/cyan]")
        Path(output_path).mkdir(parents=True, exist_ok=True)

        if format == "safetensors":
            model.save_pretrained(output_path, safe_serialization=True)
            tokenizer.save_pretrained(output_path)
        elif format == "pytorch":
            model.save_pretrained(output_path, safe_serialization=False)
            tokenizer.save_pretrained(output_path)
        elif format == "gguf":
            console.print("[yellow]⚠ GGUF export requires llama.cpp tools[/yellow]")
            console.print("[dim]Please use llama.cpp's convert.py script manually[/dim]")
            return False
        else:
            console.print(f"[red]✗ Unknown format: {format}[/red]")
            return False

        console.print(f"\n[bold green]✓ Export complete![/bold green]")
        console.print(f"[dim]Model exported to: {output_path}[/dim]\n")

        return True

    except Exception as e:
        console.print(f"\n[bold red]✗ Export failed: {e}[/bold red]")
        import traceback
        console.print(f"[red]{traceback.format_exc()}[/red]")
        return False


def main():
    """Main entry point for convert command."""
    import argparse

    parser = argparse.ArgumentParser(description="Convert and export model checkpoints")
    subparsers = parser.add_subparsers(dest="command", help="Conversion command")

    # Merge LoRA checkpoint
    merge_parser = subparsers.add_parser(
        "merge-lora",
        help="Merge LoRA adapter with base model"
    )
    merge_parser.add_argument(
        "--base-model",
        type=str,
        required=True,
        help="Base model path or HuggingFace ID (e.g., 'gpt2', 'meta-llama/Llama-3.2-1B')"
    )
    merge_parser.add_argument(
        "--lora-checkpoint",
        type=str,
        required=True,
        help="Path to LoRA checkpoint directory (e.g., 'checkpoints/sft/example/best')"
    )
    merge_parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for merged model"
    )
    merge_parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda", "mps"],
        help="Device to load models on (default: cpu)"
    )

    # Export checkpoint
    export_parser = subparsers.add_parser(
        "export",
        help="Export checkpoint to different format"
    )
    export_parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to checkpoint directory"
    )
    export_parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for exported model"
    )
    export_parser.add_argument(
        "--format",
        type=str,
        default="safetensors",
        choices=["safetensors", "pytorch", "gguf"],
        help="Export format (default: safetensors)"
    )

    args = parser.parse_args()

    try:
        if args.command == "merge-lora":
            merge_lora_checkpoint(
                base_model_path=args.base_model,
                lora_checkpoint_path=args.lora_checkpoint,
                output_path=args.output,
                device=args.device
            )
        elif args.command == "export":
            export_checkpoint(
                checkpoint_path=args.checkpoint,
                output_path=args.output,
                format=args.format
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
