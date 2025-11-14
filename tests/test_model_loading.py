"""
Tests for model loading utilities.
"""

import pytest
import torch
from transformers import AutoTokenizer

from src.auto_bot_tuner.utils.model_loading import (
    load_model_and_tokenizer,
    get_device
)


class TestGetDevice:
    """Test get_device function."""

    def test_device_detection(self):
        """Test that device is detected."""
        device_str = get_device()

        assert isinstance(device_str, str)
        assert device_str in ["cpu", "cuda", "mps"]

    def test_cuda_preference(self):
        """Test CUDA preference when available."""
        device = get_device()

        # If CUDA is available, should prefer it
        if torch.cuda.is_available():
            assert device.type == "cuda"

    def test_cpu_fallback(self):
        """Test CPU fallback."""
        device_str = get_device()

        # Should always return a valid device
        assert device_str in ["cpu", "cuda", "mps"]


class TestLoadModelAndTokenizer:
    """Test load_model_and_tokenizer function."""

    def test_basic_loading(self):
        """Test basic model and tokenizer loading."""
        model, tokenizer, device = load_model_and_tokenizer("gpt2", use_lora=False)

        # Check model
        assert model is not None
        assert hasattr(model, "forward")
        assert hasattr(model, "generate")

        # Check tokenizer
        assert tokenizer is not None
        assert hasattr(tokenizer, "encode")
        assert hasattr(tokenizer, "decode")

        # Check device
        assert isinstance(device, str)
        assert device in ["cpu", "cuda", "mps"]

    def test_pad_token_set(self):
        """Test that pad token is set to eos token if missing."""
        model, tokenizer, device = load_model_and_tokenizer("gpt2", use_lora=False)

        # GPT-2 doesn't have pad_token by default, should be set to eos_token
        assert tokenizer.pad_token is not None
        assert tokenizer.pad_token == tokenizer.eos_token
        assert tokenizer.pad_token_id is not None

    def test_model_on_correct_device(self):
        """Test that model is loaded on the correct device."""
        model, tokenizer, device = load_model_and_tokenizer("gpt2", use_lora=False)

        # Get a parameter to check device
        first_param = next(model.parameters())
        # device is a string, convert to torch.device for comparison
        device_obj = torch.device(device)
        assert first_param.device.type == device_obj.type

    def test_model_trainable(self):
        """Test that model parameters are trainable by default."""
        model, tokenizer, device = load_model_and_tokenizer("gpt2", use_lora=False)

        # Most parameters should be trainable
        trainable_params = sum(p.requires_grad for p in model.parameters())
        assert trainable_params > 0

    def test_tokenizer_consistency(self):
        """Test that tokenizer encode/decode works."""
        model, tokenizer, device = load_model_and_tokenizer("gpt2", use_lora=False)

        text = "Hello world"
        tokens = tokenizer.encode(text)
        decoded = tokenizer.decode(tokens)

        assert isinstance(tokens, list)
        assert isinstance(decoded, str)
        assert "Hello" in decoded or "hello" in decoded


# Note: apply_lora is handled internally by load_model_and_tokenizer with use_lora=True


class TestLoadWithLora:
    """Test loading model with LoRA directly."""

    def test_load_with_lora(self):
        """Test loading model with LoRA enabled."""
        model, tokenizer, device = load_model_and_tokenizer("gpt2", use_lora=True)

        # Check that LoRA is applied
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)

        # Should have both frozen and trainable parameters
        assert trainable_params > 0
        assert frozen_params > 0
        assert total_params == trainable_params + frozen_params

    def test_lora_reduces_trainable_params(self):
        """Test that LoRA significantly reduces trainable parameters."""
        # Load without LoRA
        model_full, tokenizer, device = load_model_and_tokenizer("gpt2", use_lora=False)
        full_trainable = sum(p.numel() for p in model_full.parameters() if p.requires_grad)

        # Load with LoRA
        model_lora, tokenizer, device = load_model_and_tokenizer("gpt2", use_lora=True)
        lora_trainable = sum(p.numel() for p in model_lora.parameters() if p.requires_grad)

        # LoRA should have far fewer trainable parameters
        assert lora_trainable < full_trainable
        # Should be less than 10% of original
        assert lora_trainable < full_trainable * 0.1

    def test_lora_model_still_functional(self):
        """Test that LoRA model can still run inference."""
        model, tokenizer, device = load_model_and_tokenizer("gpt2", use_lora=True)

        # Try a forward pass
        input_ids = torch.tensor([[1, 2, 3, 4, 5]]).to(device)

        with torch.no_grad():
            outputs = model(input_ids)

        assert hasattr(outputs, "logits")
        assert outputs.logits.shape[0] == 1
        assert outputs.logits.shape[1] == 5

    def test_lora_model_can_train(self):
        """Test that LoRA model can be trained."""
        model, tokenizer, device = load_model_and_tokenizer("gpt2", use_lora=True)

        optimizer = torch.optim.Adam(
            [p for p in model.parameters() if p.requires_grad], lr=1e-4
        )

        # Create dummy data
        input_ids = torch.tensor([[1, 2, 3, 4, 5]]).to(device)

        # Training step
        model.train()
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss

        assert loss is not None
        assert loss.requires_grad

        # Backward pass should work
        loss.backward()

        # Check that gradients exist for trainable params
        trainable_params_with_grad = sum(
            p.grad is not None for p in model.parameters() if p.requires_grad
        )
        assert trainable_params_with_grad > 0


class TestModelLoadingIntegration:
    """Integration tests for model loading."""

    def test_full_loading_pipeline(self):
        """Test complete loading pipeline."""
        # Load model
        model, tokenizer, device = load_model_and_tokenizer("gpt2", use_lora=False)

        # Verify tokenizer setup
        assert tokenizer.pad_token is not None

        # Test encoding
        text = "Test sentence"
        tokens = tokenizer(text, return_tensors="pt", padding=True)
        input_ids = tokens["input_ids"].to(device)
        attention_mask = tokens["attention_mask"].to(device)

        # Test forward pass
        model.eval()
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)

        assert outputs.logits.shape[0] == 1
        assert outputs.logits.shape[-1] == model.config.vocab_size

    def test_generate_with_loaded_model(self):
        """Test text generation with loaded model."""
        model, tokenizer, device = load_model_and_tokenizer("gpt2", use_lora=False)

        prompt = "Hello"
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        model.eval()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=5,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id
            )

        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        assert isinstance(generated_text, str)
        assert len(generated_text) > 0

    def test_lora_training_step(self):
        """Test a full training step with LoRA."""
        model, tokenizer, device = load_model_and_tokenizer("gpt2", use_lora=True)

        # Setup optimizer for LoRA params only
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=5e-5
        )

        # Create batch
        texts = ["Hello world", "This is a test"]
        batch = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=32
        ).to(device)

        # Training step
        model.train()
        outputs = model(**batch, labels=batch["input_ids"])
        loss = outputs.loss

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Verify training happened
        assert torch.isfinite(loss).all()
