"""
Prerequisite checking system

Validates that required models and components are available before
starting a new training stage.
"""

from typing import List
from .models import PrereqResult, Model
from .model_registry import ModelRegistry


class PrerequisiteChecker:
    """Check if user has required components for a training method"""

    def __init__(self, registry: ModelRegistry):
        self.registry = registry

    def check_sft_prerequisites(self) -> PrereqResult:
        """SFT has no prerequisites - just needs a base model"""
        return PrereqResult(
            satisfied=True,
            message="✓ SFT can be started with any base model",
            suggestions=[
                "Download a pre-trained model (Llama, GPT-2, Qwen)",
                "Or use any HuggingFace model ID"
            ]
        )

    def check_dpo_prerequisites(self) -> PrereqResult:
        """DPO requires an SFT-tuned model"""
        sft_models = self.registry.get_models_ready_for("dpo")

        if not sft_models:
            return PrereqResult(
                satisfied=False,
                missing=["sft_model"],
                message="DPO requires an SFT-tuned model as starting point",
                suggestions=[
                    "Train SFT model first (30-60 min guided wizard)",
                    "Use base model anyway (not recommended, poor results)"
                ],
                recommendations=[
                    "Start with SFT training to create instruction-following model",
                    "Then use DPO to align it to human preferences"
                ]
            )

        return PrereqResult(
            satisfied=True,
            available_models=sft_models,
            message=f"✓ Found {len(sft_models)} SFT model(s) ready for DPO",
            recommendations=[
                f"Use your most recent SFT model: {sft_models[0].id}",
                "Or select a different SFT model from the list"
            ]
        )

    def check_reward_model_prerequisites(self) -> PrereqResult:
        """Reward model training needs a base model"""
        # Reward models can be trained from scratch or from SFT models
        sft_models = self.registry.get_models_by_method("sft")

        if sft_models:
            return PrereqResult(
                satisfied=True,
                available_models=sft_models,
                message="✓ Can use base model or existing SFT model",
                suggestions=[
                    "Use SFT model for better reward model (recommended)",
                    "Or start from base model"
                ]
            )

        return PrereqResult(
            satisfied=True,
            message="✓ Can train reward model from base model",
            suggestions=[
                "Download a base model (GPT-2, Llama, etc.)",
                "Reward model will learn to predict preferences"
            ]
        )

    def check_rlhf_prerequisites(self) -> PrereqResult:
        """RLHF requires both SFT model AND reward model"""
        sft_models = self.registry.get_models_ready_for("rlhf")
        reward_models = [
            m for m in self.registry.get_all_models()
            if m.model_type == "reward_model" and m.status.value == "ready"
        ]

        missing = []
        messages = []

        if not sft_models:
            missing.append("sft_model")
            messages.append("❌ No SFT model available")
        else:
            messages.append(f"✓ Found {len(sft_models)} SFT model(s)")

        if not reward_models:
            missing.append("reward_model")
            messages.append("❌ No reward model available")
        else:
            messages.append(f"✓ Found {len(reward_models)} reward model(s)")

        if missing:
            suggestions = []
            if "sft_model" in missing:
                suggestions.append("Train SFT model first (~30-60 min)")
            if "reward_model" in missing:
                suggestions.append("Train reward model (~45-60 min)")

            # Add alternative
            if "reward_model" in missing and "sft_model" not in missing:
                suggestions.append("Or try DPO instead (simpler, no reward model needed)")

            return PrereqResult(
                satisfied=False,
                missing=missing,
                message=f"RLHF requires: {', '.join(missing)}",
                suggestions=suggestions,
                recommendations=[
                    "Train missing components using the guided wizards",
                    "RLHF is the most complex method - consider DPO for simpler alternative"
                ]
            )

        # Check compatibility (same base model)
        compatible_pairs = []
        for sft_model in sft_models:
            compatible_reward = self.registry.find_compatible_reward_model(sft_model.id)
            if compatible_reward:
                compatible_pairs.append((sft_model, compatible_reward))

        if not compatible_pairs:
            return PrereqResult(
                satisfied=False,
                missing=["compatible_models"],
                message="SFT and reward models must share the same base model",
                available_models=sft_models,
                suggestions=[
                    "Train a reward model using the same base as your SFT model",
                    "Or use DPO which doesn't need a reward model"
                ]
            )

        return PrereqResult(
            satisfied=True,
            available_models=[pair[0] for pair in compatible_pairs],
            message=f"✓ All RLHF prerequisites satisfied! Found {len(compatible_pairs)} compatible pair(s)",
            recommendations=[
                f"Use: {compatible_pairs[0][0].id} (policy) + {compatible_pairs[0][1].id} (reward)",
                "Or select different compatible models"
            ]
        )

    def check_for_method(self, method: str) -> PrereqResult:
        """Check prerequisites for any method"""
        if method == "sft":
            return self.check_sft_prerequisites()
        elif method == "dpo":
            return self.check_dpo_prerequisites()
        elif method == "reward_model":
            return self.check_reward_model_prerequisites()
        elif method == "rlhf":
            return self.check_rlhf_prerequisites()
        else:
            return PrereqResult(
                satisfied=False,
                message=f"Unknown method: {method}"
            )

    def get_suggested_model(self, method: str) -> Model:
        """Get the best suggested model for a method"""
        result = self.check_for_method(method)

        if result.satisfied and result.available_models:
            # Return the most recent model
            return result.available_models[0]

        return None

    def get_compatible_reward_model(self, policy_model_id: str) -> Model:
        """Get a compatible reward model for a policy model"""
        return self.registry.find_compatible_reward_model(policy_model_id)

    def validate_model_for_method(
        self,
        model_id: str,
        method: str
    ) -> PrereqResult:
        """Validate that a specific model can be used for a method"""
        model = self.registry.get_model(model_id)

        if not model:
            return PrereqResult(
                satisfied=False,
                message=f"Model {model_id} not found"
            )

        if method == "dpo":
            if model.has_completed_method("sft"):
                return PrereqResult(
                    satisfied=True,
                    message=f"✓ {model_id} is ready for DPO",
                    available_models=[model]
                )
            else:
                return PrereqResult(
                    satisfied=False,
                    message=f"❌ {model_id} needs SFT training first",
                    missing=["sft"]
                )

        elif method == "rlhf":
            if not model.has_completed_method("sft"):
                return PrereqResult(
                    satisfied=False,
                    message=f"❌ {model_id} needs SFT training first",
                    missing=["sft"]
                )

            reward_model = self.get_compatible_reward_model(model_id)
            if not reward_model:
                return PrereqResult(
                    satisfied=False,
                    message=f"❌ No compatible reward model for {model_id}",
                    missing=["reward_model"]
                )

            return PrereqResult(
                satisfied=True,
                message=f"✓ {model_id} is ready for RLHF",
                available_models=[model, reward_model]
            )

        return PrereqResult(satisfied=True)
