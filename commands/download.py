"""
Download command implementation.

This module provides functions for downloading models and datasets.
"""

import sys
from pathlib import Path
from rich.console import Console

# Add src directory to Python path for imports
src_path = Path(__file__).parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

console = Console()


def download_model(model_id: str):
    """
    Download a pre-trained model from HuggingFace.

    Args:
        model_id: HuggingFace model identifier (e.g., meta-llama/Llama-3.2-1B)

    Returns:
        bool: True if download succeeded, False otherwise
    """
    from src.auto_bot_tuner.utils.model_loading import download_model as dl_model

    console.print(f"\n[cyan]Downloading {model_id}...[/cyan]")
    console.print(f"[dim]Model will be cached locally[/dim]\n")

    success = dl_model(model_id)

    if success:
        console.print(f"\n[green]✓ Successfully downloaded {model_id}![/green]")
        console.print(f"[dim]Model saved to: models/{model_id.split('/')[-1]}[/dim]\n")
    else:
        console.print(f"\n[red]✗ Failed to download model[/red]\n")

    return success


def list_downloaded_models():
    """
    List all downloaded models in the models/ directory.

    Returns:
        list: List of model names
    """
    from src.auto_bot_tuner.utils.model_loading import list_downloaded_models as list_models

    models = list_models()

    if models:
        console.print("\n[bold cyan]Downloaded Models:[/bold cyan]\n")
        for i, model in enumerate(models, 1):
            console.print(f"  {i}. {model}")
        console.print()
    else:
        console.print("\n[yellow]No models found in models/ directory[/yellow]\n")

    return models


def main():
    """Main entry point for download command."""
    import argparse

    parser = argparse.ArgumentParser(description="Download models and datasets")
    subparsers = parser.add_subparsers(dest="command", help="Download command")

    # Download model
    model_parser = subparsers.add_parser("model", help="Download a pre-trained model")
    model_parser.add_argument(
        "model_id",
        type=str,
        help="HuggingFace model identifier (e.g., gpt2, meta-llama/Llama-3.2-1B)"
    )

    # List models
    subparsers.add_parser("list", help="List downloaded models")

    args = parser.parse_args()

    try:
        if args.command == "model":
            download_model(args.model_id)
        elif args.command == "list":
            list_downloaded_models()
        else:
            parser.print_help()
    except Exception as e:
        console.print(f"\n[bold red]✗ Error: {e}[/bold red]")
        import traceback
        console.print(f"[red]{traceback.format_exc()}[/red]")
        sys.exit(1)


if __name__ == "__main__":
    main()
