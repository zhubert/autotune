"""
PPO (RLHF) training command implementation.

This module provides functions for training models with PPO using reward models.
"""

import sys
from pathlib import Path
from rich.console import Console

# Add src directory to Python path for imports
src_path = Path(__file__).parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

console = Console()


def prepare_prompts_from_dataset(dataset_name: str, max_prompts: int = 500):
    """
    Extract prompts from a preference dataset.

    Args:
        dataset_name: HuggingFace dataset name
        max_prompts: Maximum number of prompts to extract

    Returns:
        List of prompt strings
    """
    from datasets import load_dataset

    console.print(f"[cyan]Loading prompts from {dataset_name}...[/cyan]")

    try:
        dataset = load_dataset(dataset_name, split="train")
    except Exception as e:
        console.print(f"[yellow]Warning: Could not load dataset: {e}[/yellow]")
        console.print("[yellow]Using fallback prompts...[/yellow]")
        return [
            "Human: What is the meaning of life?\n\nAssistant:",
            "Human: How can I be more productive?\n\nAssistant:",
            "Human: Explain quantum computing simply.\n\nAssistant:",
        ] * (max_prompts // 3)

    prompts = []
    for item in dataset:
        # Extract prompt from various dataset formats
        if "chosen" in item:
            # hh-rlhf format
            text = item["chosen"]
            if "\n\nAssistant:" in text:
                prompt = text.split("\n\nAssistant:")[0] + "\n\nAssistant:"
                prompts.append(prompt.strip())
        elif "prompt" in item:
            # Generic prompt format
            prompts.append(item["prompt"])

        if len(prompts) >= max_prompts:
            break

    if len(prompts) == 0:
        console.print("[yellow]Warning: No prompts extracted, using fallback[/yellow]")
        prompts = [
            "Human: What is the meaning of life?\n\nAssistant:",
            "Human: How can I be more productive?\n\nAssistant:",
        ] * (max_prompts // 2)

    console.print(f"[green]✓ Extracted {len(prompts)} prompts[/green]")
    return prompts


def train_ppo_quick(sft_model: str, reward_model_path: str, dataset_name: str):
    """
    Quick PPO training with pre-configured settings.

    Args:
        sft_model: SFT model name or path (policy model)
        reward_model_path: Reward model checkpoint path
        dataset_name: Dataset to extract prompts from

    Returns:
        dict: Training results
    """
    import torch
    from src.auto_bot_tuner.rlhf import (
        PPOTrainer,
        PPOConfig,
        PromptDataset,
        create_reference_model,
        create_reward_model_from_pretrained,
        create_value_network_from_policy,
    )
    from src.auto_bot_tuner.utils.model_loading import load_model_and_tokenizer, get_device

    console.print("\n[bold cyan]Starting PPO (RLHF) Training[/bold cyan]\n")

    # Get device
    device = get_device()
    console.print(f"[cyan]Using device: {device}[/cyan]")

    # 1. Load policy model (SFT checkpoint)
    console.print("\n[cyan]Loading policy model (SFT)...[/cyan]")
    policy_model, tokenizer, _ = load_model_and_tokenizer(
        sft_model,
        use_lora=True,
        device=device
    )

    # Set pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        policy_model.config.pad_token_id = tokenizer.eos_token_id

    console.print(f"[green]✓ Loaded policy model: {sft_model}[/green]")

    # 2. Create reference model
    console.print("\n[cyan]Creating reference model...[/cyan]")
    reference_model = create_reference_model(policy_model)
    console.print("[green]✓ Reference model created (frozen)[/green]")

    # 3. Load reward model
    console.print("\n[cyan]Loading reward model...[/cyan]")
    try:
        reward_model = create_reward_model_from_pretrained(
            reward_model_path,
            tokenizer,
            freeze_base=True
        )
        reward_model = reward_model.to(device)
        console.print(f"[green]✓ Loaded reward model: {reward_model_path}[/green]")
    except Exception as e:
        console.print(f"[yellow]Warning: Could not load reward model: {e}[/yellow]")
        console.print("[yellow]Creating fresh reward model (not recommended for real training)[/yellow]")
        reward_model = create_reward_model_from_pretrained(
            sft_model,
            tokenizer,
            freeze_base=True
        )
        reward_model = reward_model.to(device)

    # 4. Create value network
    console.print("\n[cyan]Creating value network...[/cyan]")
    value_network = create_value_network_from_policy(
        policy_model,
        freeze_base=False
    )
    value_network = value_network.to(device)
    console.print("[green]✓ Value network created[/green]")

    # 5. Prepare prompts
    prompts = prepare_prompts_from_dataset(dataset_name, max_prompts=500)

    # Create prompt dataset
    prompt_dataset = PromptDataset(
        prompts=prompts,
        tokenizer=tokenizer,
        max_length=256
    )
    console.print(f"[green]✓ Created dataset with {len(prompt_dataset)} prompts[/green]")

    # 6. Configure PPO
    console.print("\n[cyan]Configuring PPO trainer...[/cyan]")
    config = PPOConfig(
        # PPO parameters
        clip_ratio=0.2,
        vf_coef=0.5,
        entropy_coef=0.01,
        kl_coef=0.1,

        # Training parameters
        learning_rate=1e-6,
        batch_size=4,
        mini_batch_size=2,
        ppo_epochs=4,
        num_rollouts=100,

        # Generation
        max_new_tokens=128,
        temperature=1.0,
        top_p=0.9,

        # Logging
        logging_steps=1,
        save_steps=10,
        output_dir="checkpoints/ppo",

        # Mixed precision
        bf16=torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False
    )

    console.print("[green]✓ Configuration created[/green]")

    # 7. Create trainer
    console.print("\n[cyan]Creating PPO trainer...[/cyan]")
    trainer = PPOTrainer(
        policy_model=policy_model,
        value_network=value_network,
        reward_model=reward_model,
        reference_model=reference_model,
        tokenizer=tokenizer,
        prompt_dataset=prompt_dataset,
        config=config
    )
    console.print("[green]✓ Trainer created[/green]")

    # 8. Train
    console.print("\n[bold green]Starting training...[/bold green]\n")
    trainer.train()

    console.print("\n[bold green]✓ Training complete![/bold green]")
    console.print(f"[cyan]Model saved to: {config.output_dir}[/cyan]\n")

    return {
        "success": True,
        "output_dir": config.output_dir
    }


def train_ppo_standard(
    sft_model: str,
    reward_model_path: str,
    dataset_name: str,
    clip_ratio: float = 0.2,
    vf_coef: float = 0.5,
    entropy_coef: float = 0.01,
    kl_coef: float = 0.1,
    learning_rate: float = 1e-6,
    batch_size: int = 4,
    ppo_epochs: int = 4,
    num_rollouts: int = 100,
    max_new_tokens: int = 128,
    temperature: float = 1.0
):
    """
    Standard PPO training with configurable parameters.

    Args:
        sft_model: SFT model name or path (policy model)
        reward_model_path: Reward model checkpoint path
        dataset_name: Dataset to extract prompts from
        clip_ratio: PPO clip ratio (epsilon)
        vf_coef: Value function coefficient
        entropy_coef: Entropy coefficient
        kl_coef: KL penalty coefficient
        learning_rate: Learning rate
        batch_size: Batch size for rollouts
        ppo_epochs: PPO epochs per rollout
        num_rollouts: Number of rollouts
        max_new_tokens: Max tokens to generate
        temperature: Sampling temperature

    Returns:
        dict: Training results
    """
    import torch
    from src.auto_bot_tuner.rlhf import (
        PPOTrainer,
        PPOConfig,
        PromptDataset,
        create_reference_model,
        create_reward_model_from_pretrained,
        create_value_network_from_policy,
    )
    from src.auto_bot_tuner.utils.model_loading import load_model_and_tokenizer, get_device

    console.print("\n[bold cyan]Starting PPO (RLHF) Training[/bold cyan]\n")

    # Get device
    device = get_device()
    console.print(f"[cyan]Using device: {device}[/cyan]")

    # 1. Load policy model (SFT checkpoint)
    console.print("\n[cyan]Loading policy model (SFT)...[/cyan]")
    policy_model, tokenizer, _ = load_model_and_tokenizer(
        sft_model,
        use_lora=True,
        device=device
    )

    # Set pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        policy_model.config.pad_token_id = tokenizer.eos_token_id

    console.print(f"[green]✓ Loaded policy model: {sft_model}[/green]")

    # 2. Create reference model
    console.print("\n[cyan]Creating reference model...[/cyan]")
    reference_model = create_reference_model(policy_model)
    console.print("[green]✓ Reference model created (frozen)[/green]")

    # 3. Load reward model
    console.print("\n[cyan]Loading reward model...[/cyan]")
    try:
        reward_model = create_reward_model_from_pretrained(
            reward_model_path,
            tokenizer,
            freeze_base=True
        )
        reward_model = reward_model.to(device)
        console.print(f"[green]✓ Loaded reward model: {reward_model_path}[/green]")
    except Exception as e:
        console.print(f"[yellow]Warning: Could not load reward model: {e}[/yellow]")
        console.print("[yellow]Creating fresh reward model (not recommended for real training)[/yellow]")
        reward_model = create_reward_model_from_pretrained(
            sft_model,
            tokenizer,
            freeze_base=True
        )
        reward_model = reward_model.to(device)

    # 4. Create value network
    console.print("\n[cyan]Creating value network...[/cyan]")
    value_network = create_value_network_from_policy(
        policy_model,
        freeze_base=False
    )
    value_network = value_network.to(device)
    console.print("[green]✓ Value network created[/green]")

    # 5. Prepare prompts
    prompts = prepare_prompts_from_dataset(dataset_name, max_prompts=500)

    # Create prompt dataset
    prompt_dataset = PromptDataset(
        prompts=prompts,
        tokenizer=tokenizer,
        max_length=256
    )
    console.print(f"[green]✓ Created dataset with {len(prompt_dataset)} prompts[/green]")

    # 6. Configure PPO
    console.print("\n[cyan]Configuring PPO trainer...[/cyan]")
    config = PPOConfig(
        # PPO parameters
        clip_ratio=clip_ratio,
        vf_coef=vf_coef,
        entropy_coef=entropy_coef,
        kl_coef=kl_coef,

        # Training parameters
        learning_rate=learning_rate,
        batch_size=batch_size,
        mini_batch_size=2,
        ppo_epochs=ppo_epochs,
        num_rollouts=num_rollouts,

        # Generation
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=0.9,

        # Logging
        logging_steps=1,
        save_steps=10,
        output_dir="checkpoints/ppo/standard",

        # Mixed precision
        bf16=torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False
    )

    console.print("[green]✓ Configuration created[/green]")

    # 7. Create trainer
    console.print("\n[cyan]Creating PPO trainer...[/cyan]")
    trainer = PPOTrainer(
        policy_model=policy_model,
        value_network=value_network,
        reward_model=reward_model,
        reference_model=reference_model,
        tokenizer=tokenizer,
        prompt_dataset=prompt_dataset,
        config=config
    )
    console.print("[green]✓ Trainer created[/green]")

    # 8. Train
    console.print("\n[bold green]Starting training...[/bold green]\n")
    trainer.train()

    console.print("\n[bold green]✓ Training complete![/bold green]")
    console.print(f"[cyan]Model saved to: {config.output_dir}[/cyan]\n")

    return {
        "success": True,
        "output_dir": config.output_dir
    }
