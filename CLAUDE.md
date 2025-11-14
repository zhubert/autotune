# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Educational LLM post-training toolkit. **Code is extensively documented with educational comments** - read the implementation files for "why" explanations, mathematical formulas, and concept descriptions.

## Essential Commands

```bash
# Setup
make install              # NVIDIA/CPU
make install-rocm         # AMD GPUs (Linux)

# Testing
make test                                           # All tests
uv run pytest tests/test_sft_dataset.py -v         # Specific file
uv run pytest tests/test_sft_loss.py::test_name -v # Single test

# Training
python main.py                                     # Interactive menu
python main.py train-config configs/sft_gpt2_quick.yaml
python main.py train-sft --model gpt2 --dataset yahma/alpaca-cleaned --preset quick
```

## Architecture

**Module organization:**
- `src/auto_bot_tuner/{method}/` - Self-contained implementations (trainer.py, loss.py, dataset.py)
- `commands/` - CLI handlers called by both interactive and CLI modes
- `main.py` - Routes to interactive (`src/interactive.py`) or CLI mode

**All training methods follow same pattern:** trainer handles optimization loop, loss returns `(tensor, metrics_dict)`, dataset handles format conversion.

## Critical Implementation Details

### Reference Models (DPO/RLHF)
- Create with `create_reference_model(policy_model, device)` - returns frozen deep copy
- Reference model runs inference only, never trains
- Used for KL penalty: compare policy vs reference log probs

### Loss Masking (SFT)
- Labels with value -100 are ignored (prompt tokens)
- Only assistant responses contribute to loss
- Pattern: `F.cross_entropy(..., ignore_index=-100)`

### Model Loading
- **Always use** `load_model_and_tokenizer()` from `src/auto_bot_tuner/utils/model_loading.py`
- Auto-sets `pad_token = eos_token` if missing (required for batching)
- LoRA: rank=16, alpha=32, targets `["c_attn", "c_proj"]` for GPT-2 (c_attn=QKV, c_proj=output)

### RLHF Pipeline
- Requires 4 models: policy (trains), value network (trains), reward model (frozen), reference (frozen)
- Two-phase training: (1) generate rollouts with policy, (2) train on rollouts with PPO
- Value network: same architecture as policy but scalar head instead of vocab head
- Reward model: base LM + linear layer to scalar

### Dataset Formats
**SFT:** Alpaca (`instruction`, `input`, `output`) or Chat (`messages` with `role`/`content`)
**Preference (DPO/Reward):** Must have `prompt`, `chosen`, `rejected`

## Code Modification Rules

**Educational codebase - clarity over performance:**
- Preserve "why" comments explaining concepts/math
- Avoid premature optimization
- Use descriptive names over clever code
- Add educational docstrings to new functions

**Consistency requirements:**
- Loss functions return `(loss_tensor, metrics_dict)`
- Loss masking uses labels = -100 for ignored tokens
- Config dataclasses need parameter comments
- New trainers follow `src/auto_bot_tuner/sft/trainer.py` structure

**Testing pattern:**
- Use GPT-2 tokenizer + synthetic data (no downloads)
- Verify shapes, gradients, mathematical properties

## Common Code Patterns

```python
# Model loading (centralizes device detection, pad token setup, LoRA)
from src.auto_bot_tuner.utils.model_loading import load_model_and_tokenizer
model, tokenizer, device = load_model_and_tokenizer("gpt2", use_lora=True)

# Reference model for DPO/RLHF
from src.auto_bot_tuner.dpo import create_reference_model
ref_model = create_reference_model(policy_model, device)  # Frozen deep copy

# Reward model (adds scalar head to base LM)
from src.auto_bot_tuner.rlhf import create_reward_model_from_pretrained
reward_model = create_reward_model_from_pretrained("gpt2", tokenizer)

# Value network (same arch as policy, scalar head)
from src.auto_bot_tuner.rlhf import create_value_network_from_policy
value_net = create_value_network_from_policy(policy_model)

# Dataset loading
from datasets import load_dataset
from src.auto_bot_tuner.sft import InstructionDataset
raw = load_dataset("yahma/alpaca-cleaned", split="train")
dataset = InstructionDataset(raw, tokenizer, max_length=512, format_type="alpaca")
```

## Project Structure
```
src/auto_bot_tuner/
├── sft/           # Supervised Fine-Tuning
├── dpo/           # Direct Preference Optimization
├── rlhf/          # RLHF with PPO + Reward Models
├── evaluation/    # Metrics (perplexity, generation quality)
└── utils/         # Model loading, device detection

commands/          # CLI handlers (train_sft.py, train_dpo.py, etc.)
configs/           # YAML training configs
tests/             # 68 tests (pytest)
```
