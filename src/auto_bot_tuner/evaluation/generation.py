"""
Text generation utilities for model evaluation and interactive use.
"""

import torch
from typing import Optional, List, Union
from dataclasses import dataclass


@dataclass
class GenerationConfig:
    """Configuration for text generation."""

    max_new_tokens: int = 100
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    do_sample: bool = True
    num_return_sequences: int = 1
    repetition_penalty: float = 1.0
    length_penalty: float = 1.0
    no_repeat_ngram_size: int = 0


def generate_text(
    model,
    tokenizer,
    prompt: str,
    config: Optional[GenerationConfig] = None,
    device: Optional[torch.device] = None
) -> str:
    """
    Generate text from a prompt.

    Args:
        model: Language model
        tokenizer: Tokenizer
        prompt: Input prompt
        config: Generation configuration
        device: Device to run on

    Returns:
        Generated text (full sequence including prompt)
    """
    if config is None:
        config = GenerationConfig()

    if device is None:
        device = next(model.parameters()).device

    model.eval()

    # Tokenize prompt
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=config.max_new_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
            top_k=config.top_k,
            do_sample=config.do_sample,
            num_return_sequences=config.num_return_sequences,
            repetition_penalty=config.repetition_penalty,
            length_penalty=config.length_penalty,
            no_repeat_ngram_size=config.no_repeat_ngram_size,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    # Decode
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return generated_text


def generate_response(
    model,
    tokenizer,
    prompt: str,
    config: Optional[GenerationConfig] = None,
    device: Optional[torch.device] = None
) -> str:
    """
    Generate only the response part (excluding the prompt).

    Args:
        model: Language model
        tokenizer: Tokenizer
        prompt: Input prompt
        config: Generation configuration
        device: Device to run on

    Returns:
        Generated response (prompt removed)
    """
    if config is None:
        config = GenerationConfig()

    if device is None:
        device = next(model.parameters()).device

    model.eval()

    # Tokenize prompt
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    prompt_length = inputs["input_ids"].shape[1]

    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=config.max_new_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
            top_k=config.top_k,
            do_sample=config.do_sample,
            num_return_sequences=config.num_return_sequences,
            repetition_penalty=config.repetition_penalty,
            length_penalty=config.length_penalty,
            no_repeat_ngram_size=config.no_repeat_ngram_size,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    # Decode only the generated tokens (skip the prompt tokens)
    response_tokens = outputs[0][prompt_length:]
    response = tokenizer.decode(response_tokens, skip_special_tokens=True)

    # Clean up the response - stop at conversational markers that indicate
    # the model is hallucinating the next turn of the conversation
    stop_markers = [
        "\nUser:",
        "\nHuman:",
        "\nAssistant:",
        "\n\nUser:",
        "\n\nHuman:",
        "\n\nAssistant:",
        "<|im_start|>",  # Qwen special tokens
        "<|im_end|>",
    ]

    for marker in stop_markers:
        if marker in response:
            response = response.split(marker)[0]
            break

    # Remove common system prompt patterns that might leak through
    system_prompt_patterns = [
        "You are an AI assistant. Provide a detailed answer so user don't need to search outside to understand the answer.",
        "You are a helpful assistant.",
        "You are an AI assistant.",
    ]

    for pattern in system_prompt_patterns:
        if response.startswith(pattern):
            response = response[len(pattern):].strip()
        # Also check for the pattern anywhere in the first 200 chars
        if pattern in response[:200]:
            response = response.replace(pattern, "").strip()

    return response.strip()


def batch_generate(
    model,
    tokenizer,
    prompts: List[str],
    config: Optional[GenerationConfig] = None,
    device: Optional[torch.device] = None,
    show_progress: bool = True
) -> List[str]:
    """
    Generate responses for multiple prompts.

    Args:
        model: Language model
        tokenizer: Tokenizer
        prompts: List of prompts
        config: Generation configuration
        device: Device to run on
        show_progress: Whether to show progress bar

    Returns:
        List of generated responses
    """
    if config is None:
        config = GenerationConfig()

    if device is None:
        device = next(model.parameters()).device

    model.eval()

    responses = []

    iterator = prompts
    if show_progress:
        from tqdm import tqdm
        iterator = tqdm(prompts, desc="Generating")

    for prompt in iterator:
        response = generate_response(model, tokenizer, prompt, config, device)
        responses.append(response)

    return responses


def generate_with_streaming(
    model,
    tokenizer,
    prompt: str,
    config: Optional[GenerationConfig] = None,
    device: Optional[torch.device] = None,
    callback: Optional[callable] = None
):
    """
    Generate text with streaming output.

    This generates tokens one at a time and calls a callback function
    with each new token, enabling real-time display.

    Args:
        model: Language model
        tokenizer: Tokenizer
        prompt: Input prompt
        config: Generation configuration
        device: Device to run on
        callback: Function called with each generated token

    Yields:
        Generated tokens one at a time
    """
    if config is None:
        config = GenerationConfig()

    if device is None:
        device = next(model.parameters()).device

    model.eval()

    # Tokenize prompt
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    # Generate tokens one at a time
    generated_ids = input_ids.clone()

    for _ in range(config.max_new_tokens):
        with torch.no_grad():
            outputs = model(generated_ids)
            logits = outputs.logits[:, -1, :]

            # Apply temperature
            logits = logits / config.temperature

            # Apply top-k filtering
            if config.top_k > 0:
                top_k_logits, top_k_indices = torch.topk(logits, config.top_k)
                logits = torch.full_like(logits, float('-inf'))
                logits.scatter_(1, top_k_indices, top_k_logits)

            # Apply top-p (nucleus) filtering
            if config.top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(
                    torch.softmax(sorted_logits, dim=-1), dim=-1
                )

                # Remove tokens with cumulative probability above threshold
                sorted_indices_to_remove = cumulative_probs > config.top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                logits[indices_to_remove] = float('-inf')

            # Sample next token
            if config.do_sample:
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)

            # Check for EOS
            if next_token.item() == tokenizer.eos_token_id:
                break

            # Append to generated sequence
            generated_ids = torch.cat([generated_ids, next_token], dim=-1)

            # Decode the new token
            token_text = tokenizer.decode(next_token[0], skip_special_tokens=True)

            # Call callback if provided
            if callback:
                callback(token_text)

            yield token_text

    # Return full generated text
    full_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return full_text


def interactive_generate(
    model,
    tokenizer,
    device: Optional[torch.device] = None,
    config: Optional[GenerationConfig] = None
):
    """
    Interactive text generation loop.

    Allows the user to enter prompts and see generated responses
    in a loop until they exit.

    Args:
        model: Language model
        tokenizer: Tokenizer
        device: Device to run on
        config: Generation configuration
    """
    if config is None:
        config = GenerationConfig()

    if device is None:
        device = next(model.parameters()).device

    print("Interactive Generation Mode")
    print("Type 'quit' or 'exit' to stop")
    print("Type 'config' to adjust generation settings")
    print("-" * 50)

    while True:
        prompt = input("\nPrompt: ").strip()

        if prompt.lower() in ['quit', 'exit']:
            print("Goodbye!")
            break

        if prompt.lower() == 'config':
            print("\nCurrent settings:")
            print(f"  max_new_tokens: {config.max_new_tokens}")
            print(f"  temperature: {config.temperature}")
            print(f"  top_p: {config.top_p}")
            print(f"  top_k: {config.top_k}")
            continue

        if not prompt:
            continue

        print("\nGenerating...\n")
        response = generate_response(model, tokenizer, prompt, config, device)
        print(f"Response: {response}\n")
        print("-" * 50)
