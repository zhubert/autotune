"""
Recommendation engine for suggesting next training steps

Analyzes the current state and suggests intelligent next actions.
"""

from typing import List, Optional
from datetime import datetime, timedelta

from .models import Recommendations, Action, TrainingSession
from .model_registry import ModelRegistry


class RecommendationEngine:
    """Generate contextual next-step suggestions"""

    def __init__(self, registry: ModelRegistry):
        self.registry = registry

    def generate_recommendations(self) -> Recommendations:
        """Analyze progress and suggest next actions"""

        # Check for models currently training
        training_models = self.registry.get_training_models()
        if training_models:
            return self._recommend_check_training(training_models[0])

        # Check for recently completed models
        recent_completed = self._get_recent_completed_models(hours=2)
        if recent_completed:
            return self._recommend_post_training_actions(recent_completed[0])

        # Check what models are ready for next stage
        ready_dpo = self.registry.get_models_ready_for("dpo")
        ready_rlhf = self.registry.get_models_ready_for("rlhf")

        if ready_dpo:
            return self._recommend_dpo_training(ready_dpo)

        if ready_rlhf:
            return self._recommend_rlhf_training(ready_rlhf)

        # No models at all - recommend starting
        all_models = self.registry.get_all_models()
        if not all_models:
            return self._recommend_getting_started()

        # Have models but nothing ready for next stage
        return self._recommend_explore_options()

    def _recommend_check_training(self, model) -> Recommendations:
        """Training in progress - suggest checking on it"""
        # Get the in-progress lineage entry
        in_progress = None
        for entry in model.lineage:
            if entry.status == "in_progress":
                in_progress = entry
                break

        if in_progress:
            pct = 0
            if in_progress.total_steps > 0:
                pct = (in_progress.current_step / in_progress.total_steps) * 100

            return Recommendations(
                priority="high",
                suggested_action="check_training",
                suggested_action_title="Check Training Progress",
                reason=f"Model '{model.id}' is currently training ({pct:.0f}% complete)",
                alternatives=[
                    Action(
                        id="view_logs",
                        title="View Training Logs",
                        description="Check detailed progress and metrics"
                    ),
                    Action(
                        id="train_different",
                        title="Start Different Training",
                        description="Train another model in parallel"
                    )
                ]
            )

        return self._recommend_general_actions()

    def _recommend_post_training_actions(self, model) -> Recommendations:
        """Just finished training - suggest validation"""
        latest_method = model.lineage[-1].method

        if latest_method == "sft":
            return Recommendations(
                priority="medium",
                suggested_action="test_sft_model",
                suggested_action_title=f"Test Your SFT Model: {model.id}",
                reason="You just completed SFT training! See how well it follows instructions.",
                alternatives=[
                    Action(
                        id="evaluate_perplexity",
                        title="Evaluate Perplexity",
                        description="Measure model quality on test dataset"
                    ),
                    Action(
                        id="start_dpo",
                        title="Continue to DPO",
                        description="Align model to human preferences"
                    ),
                    Action(
                        id="train_reward_model",
                        title="Train Reward Model",
                        description="Prepare for RLHF training"
                    )
                ]
            )

        elif latest_method == "reward_model":
            # Check if there are SFT models for RLHF
            sft_models = self.registry.get_models_ready_for("rlhf")

            return Recommendations(
                priority="medium",
                suggested_action="start_rlhf",
                suggested_action_title="Ready for RLHF!",
                reason=f"You completed reward model training. {'RLHF is now available!' if sft_models else 'Train an SFT model to enable RLHF.'}",
                alternatives=[
                    Action(
                        id="test_reward_model",
                        title="Test Reward Model",
                        description="Verify it ranks responses correctly"
                    ),
                    Action(
                        id="train_sft" if not sft_models else "try_dpo_instead",
                        title="Train SFT Model" if not sft_models else "Try DPO Instead",
                        description="Needed for RLHF" if not sft_models else "Simpler alternative to RLHF"
                    )
                ]
            )

        elif latest_method == "dpo":
            return Recommendations(
                priority="medium",
                suggested_action="compare_sft_dpo",
                suggested_action_title=f"Compare Before/After DPO: {model.id}",
                reason="DPO training complete! Compare with the base SFT model.",
                alternatives=[
                    Action(
                        id="chat_with_model",
                        title="Chat with DPO Model",
                        description="Test the preference-aligned responses"
                    ),
                    Action(
                        id="evaluate_both",
                        title="Evaluate Both Models",
                        description="Run formal evaluation metrics"
                    )
                ]
            )

        elif latest_method == "rlhf":
            return Recommendations(
                priority="medium",
                suggested_action="celebrate_rlhf",
                suggested_action_title=f"RLHF Training Complete! {model.id}",
                reason="You've completed the full RLHF pipeline!",
                alternatives=[
                    Action(
                        id="compare_all",
                        title="Compare Full Pipeline",
                        description="Compare base → SFT → RLHF improvements"
                    ),
                    Action(
                        id="chat_with_model",
                        title="Chat with RLHF Model",
                        description="Test the fully aligned model"
                    )
                ]
            )

        return self._recommend_general_actions()

    def _recommend_dpo_training(self, ready_models: List) -> Recommendations:
        """Models ready for DPO"""
        return Recommendations(
            priority="medium",
            suggested_action="start_dpo",
            suggested_action_title="Train DPO Model",
            reason=f"You have {len(ready_models)} SFT model(s) ready for preference alignment",
            alternatives=[
                Action(
                    id="train_reward_model",
                    title="Train Reward Model Instead",
                    description="Prepare for RLHF (more complex but more control)"
                ),
                Action(
                    id="evaluate_sft",
                    title="Evaluate SFT Models",
                    description="Test current models before further training"
                ),
                Action(
                    id="train_new_sft",
                    title="Train New SFT Model",
                    description="Try different dataset or base model"
                )
            ]
        )

    def _recommend_rlhf_training(self, ready_models: List) -> Recommendations:
        """Models ready for RLHF"""
        return Recommendations(
            priority="high",
            suggested_action="start_rlhf",
            suggested_action_title="Start RLHF Training",
            reason=f"You have all prerequisites for RLHF! {len(ready_models)} model(s) ready",
            alternatives=[
                Action(
                    id="try_dpo_first",
                    title="Try DPO First",
                    description="Simpler alternative to test before RLHF"
                ),
                Action(
                    id="evaluate_current",
                    title="Evaluate Current Models",
                    description="Benchmark before RLHF training"
                )
            ]
        )

    def _recommend_getting_started(self) -> Recommendations:
        """No models yet - recommend starting"""
        return Recommendations(
            priority="high",
            suggested_action="start_sft",
            suggested_action_title="Start Your First Training",
            reason="Get started with Supervised Fine-Tuning (SFT) - the foundation of post-training",
            alternatives=[
                Action(
                    id="learning_path",
                    title="Follow Learning Path",
                    description="Guided step-by-step tutorial through all methods"
                ),
                Action(
                    id="download_model",
                    title="Download Pre-trained Model",
                    description="Get Llama, GPT-2, or Qwen for training"
                )
            ]
        )

    def _recommend_explore_options(self) -> Recommendations:
        """Have models but nothing obvious to do next"""
        stats = self.registry.get_stats()

        return Recommendations(
            priority="low",
            suggested_action="explore_models",
            suggested_action_title="Explore Your Models",
            reason=f"You have {stats['total_models']} model(s). Browse and test them!",
            alternatives=[
                Action(
                    id="chat_with_model",
                    title="Chat with a Model",
                    description="Test any of your trained models"
                ),
                Action(
                    id="train_new",
                    title="Train New Model",
                    description="Try different dataset or base model"
                ),
                Action(
                    id="compare_models",
                    title="Compare Models",
                    description="Evaluate differences between your models"
                )
            ]
        )

    def _recommend_general_actions(self) -> Recommendations:
        """General fallback recommendations"""
        stats = self.registry.get_stats()

        return Recommendations(
            priority="low",
            suggested_action="browse_registry",
            suggested_action_title="Browse Model Registry",
            reason=f"View your {stats['total_models']} model(s) and choose an action",
            alternatives=[
                Action(
                    id="train_new",
                    title="Start New Training",
                    description="Train a new model"
                ),
                Action(
                    id="evaluate",
                    title="Evaluate Models",
                    description="Run benchmarks on your models"
                )
            ]
        )

    def _get_recent_completed_models(self, hours: int = 2):
        """Get models completed within the last N hours"""
        cutoff = datetime.now() - timedelta(hours=hours)
        recent = []

        for model in self.registry.get_all_models():
            if model.status.value != "ready":
                continue

            # Check latest lineage entry
            if model.lineage:
                latest = model.lineage[-1]
                if latest.completed_at:
                    completed = datetime.fromisoformat(latest.completed_at)
                    if completed > cutoff:
                        recent.append(model)

        # Sort by completion time, most recent first
        recent.sort(
            key=lambda m: m.lineage[-1].completed_at if m.lineage[-1].completed_at else "",
            reverse=True
        )

        return recent
