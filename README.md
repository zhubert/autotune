# Autotune

[![Tests](https://img.shields.io/badge/tests-210%20passing-success)](tests/)
[![Python](https://img.shields.io/badge/python-3.12%2B-blue)]()
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

An educational toolkit for fine-tuning and aligning language models. Learn how post-training works by implementing SFT, DPO, and RLHF from scratch with extensive documentation.

## What is Post-Training?

**Post-training** transforms a base language model (which just predicts tokens) into a helpful assistant that follows instructions and aligns with human preferences.

This toolkit implements four key techniques:

- **Supervised Fine-Tuning (SFT)** - Teach models to follow instructions
- **Reward Modeling** - Train models to predict human preferences
- **Direct Preference Optimization (DPO)** - Align models without reinforcement learning
- **RLHF with PPO** - Use reinforcement learning to optimize for human preferences

All implementations include extensive educational comments explaining the "why" behind each technique.

## Quick Start

### Installation

```bash
# Install UV (fast Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
make install              # NVIDIA/CPU
make install-rocm         # AMD GPUs (Linux)
```

### Using the Interactive CLI

Launch the interactive menu:

```bash
python main.py
```

The CLI walks you through:
1. Downloading pre-trained models (GPT-2, Llama, etc.)
2. Training with SFT, DPO, or RLHF
3. Evaluating and chatting with your models

No command-line flags to memorize - just follow the prompts!

### Using as a Library

```python
from src.auto_bot_tuner.sft import SFTTrainer, SFTConfig, InstructionDataset
from src.auto_bot_tuner.utils.model_loading import load_model_and_tokenizer

# Load model with LoRA for efficient training
model, tokenizer, device = load_model_and_tokenizer("gpt2", use_lora=True)

# Prepare instruction dataset
dataset = InstructionDataset(your_dataset, tokenizer)

# Train
trainer = SFTTrainer(model, tokenizer, dataset, config=SFTConfig())
trainer.train()
```

See the `configs/` directory for example training configurations.

## Project Structure

```
src/auto_bot_tuner/
├── sft/                    # Supervised Fine-Tuning
├── dpo/                    # Direct Preference Optimization
├── rlhf/                   # RLHF with PPO + Reward Models
├── evaluation/             # Metrics and generation tools
└── utils/                  # Model loading, device detection

commands/                   # CLI command handlers
configs/                    # Example training configurations
tests/                      # 210 passing tests
```

## Learning Path

Work through the techniques in order:

**1. Supervised Fine-Tuning (SFT)** - Start here! Teach a base model to follow instructions using the Alpaca dataset.

**2. Reward Modeling** - Train a model to predict human preferences from comparison data.

**3. Direct Preference Optimization (DPO)** - Align models directly on preference pairs without needing a separate reward model.

**4. RLHF with PPO** - Use reinforcement learning to optimize for human preferences with the full pipeline.

Each implementation includes detailed comments explaining the math and reasoning.

## Key Papers

- [Training language models to follow instructions](https://arxiv.org/abs/2203.02155) (InstructGPT/RLHF)
- [Direct Preference Optimization](https://arxiv.org/abs/2305.18290) (DPO)
- [Proximal Policy Optimization](https://arxiv.org/abs/1707.06347) (PPO)
- [LoRA: Low-Rank Adaptation](https://arxiv.org/abs/2106.09685) (Efficient fine-tuning)

## Running Tests

```bash
make test
```

## License

MIT License - see LICENSE file for details.
