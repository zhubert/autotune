"""
Interactive chat command implementation.

This module provides a multi-turn chat interface for conversing with models.
"""

import sys
from pathlib import Path
from rich.console import Console

# Add src directory to Python path for imports
src_path = Path(__file__).parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

console = Console()


def chat_with_model(model_path: str, max_new_tokens: int = 150, temperature: float = 0.7):
    """
    Start an interactive chat session with a model.

    Args:
        model_path: Model path or HuggingFace identifier
        max_new_tokens: Maximum tokens to generate per response
        temperature: Sampling temperature for generation

    Returns:
        None (runs interactive loop until user exits)
    """
    from src.auto_bot_tuner.utils.model_loading import load_model_and_tokenizer
    from src.auto_bot_tuner.evaluation import generate_response, GenerationConfig

    console.print(f"\n[cyan]Loading model from {model_path}...[/cyan]")
    model, tokenizer, device = load_model_and_tokenizer(model_path, use_lora=False)
    console.print("[green]✓ Model loaded![/green]\n")

    # Generation settings
    console.print("[bold yellow]Chat Mode[/bold yellow]")
    console.print("Type your messages and press Enter.")
    console.print("Commands: 'quit' or 'exit' to stop, 'clear' to reset\n")
    console.print("-" * 60)

    config = GenerationConfig(
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=0.9,
        do_sample=True,
        repetition_penalty=1.2,
        no_repeat_ngram_size=3
    )

    conversation_history = []

    while True:
        user_input = input("\nYou: ").strip()

        if user_input.lower() in ['quit', 'exit']:
            console.print("\n[cyan]Goodbye![/cyan]")
            break

        if user_input.lower() == 'clear':
            conversation_history = []
            console.print("\n[cyan]Conversation cleared![/cyan]")
            continue

        if not user_input:
            continue

        # Build prompt with conversation history
        # Check if tokenizer has a chat template (e.g., Qwen models)
        if hasattr(tokenizer, 'chat_template') and tokenizer.chat_template is not None:
            # Use the tokenizer's chat template
            messages = []

            # Add conversation history
            for i in range(0, len(conversation_history), 2):
                if i + 1 < len(conversation_history):
                    user_msg = conversation_history[i].replace("User: ", "")
                    assistant_msg = conversation_history[i + 1].replace("Assistant: ", "")
                    messages.append({"role": "user", "content": user_msg})
                    messages.append({"role": "assistant", "content": assistant_msg})

            # Add current user input
            messages.append({"role": "user", "content": user_input})

            # Apply chat template
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            # Fallback to simple format for models without chat templates
            if conversation_history:
                prompt = "\n".join(conversation_history) + f"\nUser: {user_input}\nAssistant:"
            else:
                prompt = f"User: {user_input}\nAssistant:"

        # Generate response
        response = generate_response(model, tokenizer, prompt, config, device)

        # Update conversation history
        conversation_history.append(f"User: {user_input}")
        conversation_history.append(f"Assistant: {response}")

        # Keep only last 5 exchanges
        if len(conversation_history) > 10:
            conversation_history = conversation_history[-10:]

        console.print(f"\n[bold green]Assistant:[/bold green] {response}")
        print("-" * 60)


def main():
    """Main entry point for chat command."""
    import argparse

    parser = argparse.ArgumentParser(description="Chat with a trained model")
    parser.add_argument(
        "model",
        type=str,
        help="Model path or HuggingFace identifier"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=150,
        help="Maximum tokens per response"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature"
    )

    args = parser.parse_args()

    try:
        chat_with_model(
            model_path=args.model,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature
        )
    except Exception as e:
        console.print(f"\n[bold red]✗ Error: {e}[/bold red]")
        import traceback
        console.print(f"[red]{traceback.format_exc()}[/red]")
        sys.exit(1)


if __name__ == "__main__":
    main()
