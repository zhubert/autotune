"""
Model Loading Utilities

Functions for downloading and loading pre-trained models from HuggingFace.

Key Concepts:
-------------
1. **Model Hub**: HuggingFace hosts thousands of pre-trained models
2. **Automatic Downloads**: transformers library handles downloading and caching
3. **Local Caching**: Models are cached in ~/.cache/huggingface/
4. **Device Mapping**: Automatically place model on GPU/CPU

Educational Notes:
-----------------
When you download a model, you get:
- Model weights (.bin or .safetensors files)
- Config file (architecture details)
- Tokenizer files (vocabulary and special tokens)

For post-training, we need:
- The base model (for SFT, DPO)
- Sometimes a copy for the reference model (RLHF)
"""

from pathlib import Path
from typing import Optional
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig
)
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()


def get_device() -> str:
    """
    Detect the best available device for training.

    Returns:
        str: Device string ('cuda', 'mps', or 'cpu')

    Educational Note:
    ----------------
    Device selection affects training speed dramatically:
    - CUDA (NVIDIA): Fastest, best library support
    - ROCm (AMD): Fast, uses torch.cuda API via HIP
    - MPS (Apple Silicon): Good for M1/M2/M3 Macs
    - CPU: Slowest, but always works
    """
    if torch.cuda.is_available():
        # Works for both NVIDIA (CUDA) and AMD (ROCm)
        device = "cuda"
        device_name = torch.cuda.get_device_name(0)
        console.print(f"[green]✓ Using GPU: {device_name}[/green]")
    elif torch.backends.mps.is_available():
        device = "mps"
        console.print("[green]✓ Using Apple Silicon GPU (MPS)[/green]")
    else:
        device = "cpu"
        console.print("[yellow]⚠ Using CPU (slow, consider using a GPU)[/yellow]")

    return device


def download_model(model_id: str, save_path: Optional[Path] = None) -> bool:
    """
    Download a pre-trained model from HuggingFace.

    Args:
        model_id: HuggingFace model ID (e.g., "meta-llama/Llama-3.2-1B")
        save_path: Optional path to save model (defaults to models/{model_name})

    Returns:
        bool: True if successful, False otherwise

    Educational Note:
    ----------------
    This function downloads three key components:
    1. Model weights - The actual neural network parameters
    2. Configuration - Architecture details (layers, heads, etc.)
    3. Tokenizer - Converts text to/from token IDs

    The model is cached automatically by transformers library,
    so subsequent loads will be instant!

    Why we need this:
    - Can't train from scratch (too expensive)
    - Pre-trained models already understand language
    - Post-training adapts them to specific tasks
    """
    try:
        # Determine save path
        if save_path is None:
            model_name = model_id.split("/")[-1]
            save_path = Path("models") / model_name

        save_path.mkdir(parents=True, exist_ok=True)

        console.print(f"\n[cyan]Downloading model: {model_id}[/cyan]")
        console.print(f"[dim]This will be cached for future use[/dim]\n")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            # Download config
            task1 = progress.add_task("Downloading config...", total=None)
            config = AutoConfig.from_pretrained(model_id)
            config.save_pretrained(save_path)
            progress.update(task1, completed=True)

            # Download tokenizer
            task2 = progress.add_task("Downloading tokenizer...", total=None)
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            tokenizer.save_pretrained(save_path)
            progress.update(task2, completed=True)

            # Download model weights
            task3 = progress.add_task("Downloading model weights (this may take a while)...", total=None)
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                dtype=torch.bfloat16,  # Use bfloat16 to save memory
                device_map="auto",  # Automatically place on best device
                low_cpu_mem_usage=True,  # Optimize memory usage
            )
            model.save_pretrained(save_path)
            progress.update(task3, completed=True)

        # Print model info
        console.print(f"\n[green]✓ Model downloaded successfully![/green]")
        console.print(f"[dim]Saved to: {save_path}[/dim]\n")

        # Show model stats
        num_params = sum(p.numel() for p in model.parameters())
        console.print(f"[cyan]Model Statistics:[/cyan]")
        console.print(f"  Parameters: {num_params:,} ({num_params/1e9:.2f}B)")
        console.print(f"  Vocab size: {config.vocab_size:,}")
        console.print(f"  Hidden size: {config.hidden_size}")
        console.print(f"  Layers: {config.num_hidden_layers}")
        console.print(f"  Attention heads: {config.num_attention_heads}")

        return True

    except Exception as e:
        console.print(f"\n[red]✗ Error downloading model: {e}[/red]")
        return False


def list_downloaded_models() -> list[str]:
    """
    List all models downloaded in the local models directory.

    Returns:
        list[str]: List of model names (directory names in models/)

    Educational Note:
    ----------------
    This function scans the local models directory to find
    all downloaded models. Models can be:
    - Downloaded via the download_model() function
    - Manually placed in the models/ directory
    - Training checkpoints saved during fine-tuning
    """
    models_dir = Path("models")

    if not models_dir.exists():
        return []

    # Find all directories in models/ (excluding hidden files)
    model_dirs = [
        d.name for d in models_dir.iterdir()
        if d.is_dir() and not d.name.startswith('.')
    ]

    return sorted(model_dirs)


def load_model_and_tokenizer(
    model_path: str,
    device: Optional[str] = None,
    use_lora: bool = False
):
    """
    Load a model and tokenizer for training or inference.

    Args:
        model_path: Path to model (local path or HuggingFace ID)
        device: Device to load model on (auto-detected if None)
        use_lora: Whether to prepare model for LoRA training

    Returns:
        tuple: (model, tokenizer, device)

    Educational Note:
    ----------------
    LoRA (Low-Rank Adaptation):
    - Only trains a small number of additional parameters
    - Original model weights stay frozen
    - Reduces memory usage by 3-10x
    - Slightly worse than full fine-tuning, but much more practical

    Why LoRA matters:
    - Full fine-tuning a 1B model needs ~12GB just for gradients
    - LoRA might only need ~2GB
    - Enables training larger models on consumer hardware
    """
    if device is None:
        device = get_device()

    console.print(f"\n[cyan]Loading model: {model_path}[/cyan]")

    try:
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        # Ensure tokenizer has pad token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            console.print("[yellow]⚠ Set pad_token = eos_token[/yellow]")

        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
            device_map="auto",
            low_cpu_mem_usage=True,
        )

        if use_lora:
            # Prepare for LoRA training
            console.print("[cyan]Preparing model for LoRA training...[/cyan]")

            from peft import LoraConfig, get_peft_model, TaskType

            # LoRA Configuration
            # Educational note: LoRA (Low-Rank Adaptation) adds small trainable matrices
            # to the model's attention layers instead of fine-tuning all parameters.
            # This dramatically reduces memory usage and training time.
            #
            # Key parameters:
            # - r: Rank of the update matrices (typical: 8-64, higher = more capacity)
            # - lora_alpha: Scaling factor (typical: 16-32)
            # - target_modules: Which layers to adapt (attention layers are most effective)
            # - lora_dropout: Dropout for regularization
            # - fan_in_fan_out: GPT-2 uses Conv1D which transposes weights, so we need True
            lora_config = LoraConfig(
                r=16,  # Rank of LoRA matrices
                lora_alpha=32,  # Scaling factor (typically 2*r)
                target_modules=["c_attn", "c_proj"],  # GPT-2 attention layers (c_attn = QKV, c_proj = output)
                lora_dropout=0.05,  # Dropout for regularization
                bias="none",  # Don't adapt bias parameters
                task_type=TaskType.CAUSAL_LM,  # Causal language modeling
                fan_in_fan_out=True,  # Required for GPT-2's Conv1D layers
            )

            # Wrap model with LoRA adapters
            model = get_peft_model(model, lora_config)

            # Print trainable parameters
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in model.parameters())
            trainable_pct = 100 * trainable_params / total_params

            console.print(f"[green]✓ LoRA adapters added[/green]")
            console.print(f"  Trainable parameters: {trainable_params:,} ({trainable_pct:.2f}% of total)")
            console.print(f"  Total parameters: {total_params:,}")

        console.print(f"[green]✓ Model loaded successfully![/green]")

        return model, tokenizer, device

    except Exception as e:
        console.print(f"[red]✗ Error loading model: {e}[/red]")
        raise
