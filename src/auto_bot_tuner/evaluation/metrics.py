"""
Evaluation metrics for language models.

Provides functions to compute perplexity, generation quality, and other
evaluation metrics for assessing model performance.
"""

import torch
import torch.nn.functional as F
from typing import Optional, List, Dict
from datasets import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm


def compute_perplexity(
    model,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    labels: Optional[torch.Tensor] = None
) -> float:
    """
    Compute perplexity for a batch of sequences.

    Perplexity measures how well the model predicts the test data.
    Lower perplexity = better model.

    Perplexity = exp(cross_entropy_loss)

    Args:
        model: Language model
        input_ids: Input token IDs of shape (batch_size, seq_len)
        attention_mask: Attention mask
        labels: Target labels (if None, uses input_ids shifted by 1)

    Returns:
        Perplexity score (scalar)

    Educational Note:
        Perplexity can be interpreted as the effective vocabulary size
        the model is confused over. A perplexity of 100 means the model
        is as confused as if it had to choose uniformly from 100 words.
    """
    model.eval()

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits

        # If no labels provided, use input_ids shifted for next-token prediction
        if labels is None:
            labels = input_ids.clone()

        # Shift logits and labels for next-token prediction
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        # Flatten
        shift_logits = shift_logits.view(-1, shift_logits.size(-1))
        shift_labels = shift_labels.view(-1)

        # Compute cross-entropy loss
        if attention_mask is not None:
            # Create mask for shifted labels
            shift_mask = attention_mask[..., 1:].contiguous().view(-1)
            # Only compute loss on non-padding tokens
            loss = F.cross_entropy(
                shift_logits[shift_mask.bool()],
                shift_labels[shift_mask.bool()],
                reduction='mean'
            )
        else:
            loss = F.cross_entropy(shift_logits, shift_labels, reduction='mean')

        perplexity = torch.exp(loss).item()

    return perplexity


def compute_perplexity_on_dataset(
    model,
    dataset: Dataset,
    tokenizer,
    batch_size: int = 8,
    max_length: int = 512,
    text_column: str = "text",
    device: Optional[torch.device] = None
) -> Dict[str, float]:
    """
    Compute perplexity on a full dataset.

    Args:
        model: Language model
        dataset: HuggingFace dataset
        tokenizer: Tokenizer
        batch_size: Batch size for evaluation
        max_length: Maximum sequence length
        text_column: Name of text column in dataset
        device: Device to run on

    Returns:
        Dictionary with perplexity and other metrics
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    model.to(device)

    # Tokenize dataset
    def tokenize_function(examples):
        return tokenizer(
            examples[text_column],
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors="pt"
        )

    # Process in batches
    total_loss = 0.0
    total_tokens = 0
    num_batches = 0

    for i in tqdm(range(0, len(dataset), batch_size), desc="Computing perplexity"):
        batch = dataset[i:i + batch_size]

        # Tokenize batch
        texts = batch[text_column] if isinstance(batch[text_column], list) else [batch[text_column]]
        encoded = tokenizer(
            texts,
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors="pt"
        )

        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            # Compute loss
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()
            shift_mask = attention_mask[..., 1:].contiguous()

            # Flatten
            shift_logits = shift_logits.view(-1, shift_logits.size(-1))
            shift_labels = shift_labels.view(-1)
            shift_mask = shift_mask.view(-1)

            # Compute loss on non-padding tokens
            loss = F.cross_entropy(
                shift_logits[shift_mask.bool()],
                shift_labels[shift_mask.bool()],
                reduction='sum'
            )

            total_loss += loss.item()
            total_tokens += shift_mask.sum().item()
            num_batches += 1

    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss)).item()

    return {
        "perplexity": perplexity,
        "loss": avg_loss,
        "total_tokens": total_tokens,
        "num_samples": len(dataset)
    }


def evaluate_generation_quality(
    model,
    tokenizer,
    prompts: List[str],
    device: Optional[torch.device] = None,
    max_new_tokens: int = 100,
    temperature: float = 0.7,
    top_p: float = 0.9
) -> List[Dict[str, str]]:
    """
    Evaluate generation quality on a list of prompts.

    Args:
        model: Language model
        tokenizer: Tokenizer
        prompts: List of prompts to generate from
        device: Device to run on
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_p: Nucleus sampling threshold

    Returns:
        List of dictionaries with prompts and generated responses
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    model.to(device)

    results = []

    for prompt in tqdm(prompts, desc="Generating responses"):
        # Tokenize prompt
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id
            )

        # Decode
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = generated_text[len(prompt):].strip()

        results.append({
            "prompt": prompt,
            "response": response,
            "full_text": generated_text
        })

    return results


def compare_model_outputs(
    model_a,
    model_b,
    tokenizer,
    prompts: List[str],
    model_a_name: str = "Model A",
    model_b_name: str = "Model B",
    device: Optional[torch.device] = None,
    **generation_kwargs
) -> List[Dict[str, str]]:
    """
    Compare outputs from two models on the same prompts.

    Args:
        model_a: First model
        model_b: Second model
        tokenizer: Tokenizer (shared)
        prompts: List of prompts
        model_a_name: Name for model A
        model_b_name: Name for model B
        device: Device to run on
        **generation_kwargs: Additional generation arguments

    Returns:
        List of comparison dictionaries
    """
    if device is None:
        device = next(model_a.parameters()).device

    model_a.eval()
    model_b.eval()
    model_a.to(device)
    model_b.to(device)

    # Default generation settings
    gen_kwargs = {
        "max_new_tokens": 100,
        "temperature": 0.7,
        "top_p": 0.9,
        "do_sample": True
    }
    gen_kwargs.update(generation_kwargs)

    results = []

    for prompt in tqdm(prompts, desc="Comparing models"):
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        # Generate from model A
        with torch.no_grad():
            outputs_a = model_a.generate(
                **inputs,
                **gen_kwargs,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id
            )
            response_a = tokenizer.decode(outputs_a[0], skip_special_tokens=True)[len(prompt):].strip()

        # Generate from model B
        with torch.no_grad():
            outputs_b = model_b.generate(
                **inputs,
                **gen_kwargs,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id
            )
            response_b = tokenizer.decode(outputs_b[0], skip_special_tokens=True)[len(prompt):].strip()

        results.append({
            "prompt": prompt,
            f"{model_a_name}_response": response_a,
            f"{model_b_name}_response": response_b
        })

    return results


def compute_diversity_metrics(generated_texts: List[str]) -> Dict[str, float]:
    """
    Compute diversity metrics for generated texts.

    Measures:
    - Distinct-1: Ratio of unique unigrams
    - Distinct-2: Ratio of unique bigrams
    - Average length: Mean number of tokens

    Args:
        generated_texts: List of generated text samples

    Returns:
        Dictionary with diversity metrics
    """
    from collections import Counter

    all_unigrams = []
    all_bigrams = []
    total_tokens = 0

    for text in generated_texts:
        tokens = text.split()
        total_tokens += len(tokens)

        # Unigrams
        all_unigrams.extend(tokens)

        # Bigrams
        bigrams = [f"{tokens[i]}_{tokens[i+1]}" for i in range(len(tokens)-1)]
        all_bigrams.extend(bigrams)

    # Compute metrics
    distinct_1 = len(set(all_unigrams)) / len(all_unigrams) if all_unigrams else 0
    distinct_2 = len(set(all_bigrams)) / len(all_bigrams) if all_bigrams else 0
    avg_length = total_tokens / len(generated_texts) if generated_texts else 0

    return {
        "distinct_1": distinct_1,
        "distinct_2": distinct_2,
        "avg_length": avg_length,
        "unique_unigrams": len(set(all_unigrams)),
        "unique_bigrams": len(set(all_bigrams))
    }


def compute_model_size_info(model) -> Dict[str, int]:
    """
    Compute model size information.

    Args:
        model: PyTorch model

    Returns:
        Dictionary with parameter counts
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "frozen_parameters": total_params - trainable_params,
        "trainable_percentage": 100 * trainable_params / total_params if total_params > 0 else 0
    }
