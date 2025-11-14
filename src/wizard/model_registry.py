"""
Model registry for tracking multiple models and their lineage

The registry maintains a database of all trained models, their training history,
and relationships between models (e.g., which DPO model came from which SFT model).
"""

import json
from pathlib import Path
from typing import Optional, List, Dict
from datetime import datetime

from .models import (
    Model, LineageEntry, ModelStatus, TrainingSession
)


class ModelRegistry:
    """Track and manage multiple models at different training stages"""

    def __init__(self, progress_file: str = ".llm-training-progress.json"):
        self.progress_file = Path(progress_file)
        self.state = self.load()

    def load(self) -> dict:
        """Load progress state from disk"""
        if not self.progress_file.exists():
            return self._create_empty_state()

        try:
            with open(self.progress_file, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            # If file is corrupted, create new state
            return self._create_empty_state()

    def save(self):
        """Save progress state to disk"""
        # Ensure parent directory exists
        self.progress_file.parent.mkdir(parents=True, exist_ok=True)

        with open(self.progress_file, 'w') as f:
            json.dump(self.state, f, indent=2)

    def _create_empty_state(self) -> dict:
        """Create empty progress state"""
        return {
            "version": "2.0",
            "models": {},
            "training_sessions": [],
            "training_pipelines": {},
            "model_comparisons": []
        }

    def register_model(
        self,
        model_id: str,
        base_model: str,
        training_id: str,
        method: str,
        checkpoint_path: str,
        dataset: Optional[str] = None,
        parent_model_id: Optional[str] = None,
        tags: List[str] = None,
        notes: str = "",
        model_type: str = "generative",
        metrics: Dict[str, float] = None
    ) -> Model:
        """Register a new model or add to lineage of existing model"""

        lineage_entry = LineageEntry(
            method=method,
            training_id=training_id,
            checkpoint=checkpoint_path,
            parent=parent_model_id,
            dataset=dataset,
            completed_at=datetime.now().isoformat(),
            status="completed",
            metrics=metrics or {}
        )

        if model_id in self.state["models"]:
            # Add to existing model's lineage
            model_data = self.state["models"][model_id]
            model_data["lineage"].append(lineage_entry.to_dict())
            model_data["status"] = ModelStatus.READY.value
            model = Model.from_dict(model_data)
        else:
            # Create new model
            model = Model(
                id=model_id,
                base_model=base_model,
                created_at=datetime.now().isoformat(),
                lineage=[lineage_entry],
                tags=tags or [],
                notes=notes,
                status=ModelStatus.READY,
                model_type=model_type
            )
            self.state["models"][model_id] = model.to_dict()

        # Update capabilities based on training methods
        self._update_capabilities(model_id)

        self.save()
        return model

    def mark_model_training(
        self,
        model_id: str,
        training_id: str,
        method: str,
        checkpoint_path: str,
        dataset: Optional[str] = None,
        total_steps: int = 0
    ):
        """Mark a model as currently training"""
        lineage_entry = LineageEntry(
            method=method,
            training_id=training_id,
            checkpoint=checkpoint_path,
            dataset=dataset,
            status="in_progress",
            current_step=0,
            total_steps=total_steps
        )

        if model_id in self.state["models"]:
            model_data = self.state["models"][model_id]
            model_data["lineage"].append(lineage_entry.to_dict())
            model_data["status"] = ModelStatus.TRAINING.value
        else:
            # This shouldn't happen, but handle it
            raise ValueError(f"Cannot mark unknown model {model_id} as training")

        self.save()

    def update_training_progress(
        self,
        model_id: str,
        training_id: str,
        current_step: int,
        metrics: Dict[str, float] = None
    ):
        """Update progress of a training session"""
        if model_id not in self.state["models"]:
            return

        model_data = self.state["models"][model_id]

        # Find the lineage entry with this training_id
        for entry in model_data["lineage"]:
            if entry.get("training_id") == training_id:
                entry["current_step"] = current_step
                if metrics:
                    entry["metrics"].update(metrics)
                break

        self.save()

    def complete_model_training(
        self,
        model_id: str,
        training_id: str,
        final_metrics: Dict[str, float] = None
    ):
        """Mark a model's training as completed"""
        if model_id not in self.state["models"]:
            return

        model_data = self.state["models"][model_id]

        # Find and update the lineage entry
        for entry in model_data["lineage"]:
            if entry.get("training_id") == training_id:
                entry["status"] = "completed"
                entry["completed_at"] = datetime.now().isoformat()
                if final_metrics:
                    entry["metrics"].update(final_metrics)
                break

        model_data["status"] = ModelStatus.READY.value
        self._update_capabilities(model_id)
        self.save()

    def _update_capabilities(self, model_id: str):
        """Update model capabilities based on completed training"""
        model_data = self.state["models"][model_id]
        capabilities = set()

        for entry in model_data["lineage"]:
            if entry.get("status") == "completed":
                method = entry["method"]
                if method == "sft":
                    capabilities.add("instruction-following")
                elif method == "dpo":
                    capabilities.add("preference-aligned")
                elif method == "rlhf":
                    capabilities.add("preference-aligned")
                    capabilities.add("rlhf-optimized")
                elif method == "reward_model":
                    capabilities.add("reward-scoring")

        model_data["capabilities"] = list(capabilities)

    def get_model(self, model_id: str) -> Optional[Model]:
        """Get a specific model by ID"""
        if model_id not in self.state["models"]:
            return None
        return Model.from_dict(self.state["models"][model_id])

    def get_all_models(self) -> List[Model]:
        """Get all registered models"""
        return [
            Model.from_dict(data)
            for data in self.state["models"].values()
        ]

    def get_models_by_base(self, base_model: str) -> List[Model]:
        """Get all models derived from a specific base model"""
        return [
            Model.from_dict(data)
            for data in self.state["models"].values()
            if data["base_model"] == base_model
        ]

    def get_models_by_method(self, method: str) -> List[Model]:
        """Get all models that have completed a specific training method"""
        models = []
        for model_data in self.state["models"].values():
            if any(
                entry["method"] == method and entry.get("status") == "completed"
                for entry in model_data["lineage"]
            ):
                models.append(Model.from_dict(model_data))
        return models

    def get_models_ready_for(self, next_method: str) -> List[Model]:
        """Get models ready for the next training stage"""

        prerequisites = {
            "dpo": ["sft"],
            "rlhf": ["sft"]
        }

        required_methods = prerequisites.get(next_method, [])

        models = []
        for model_data in self.state["models"].values():
            # Skip reward models for generative training
            if model_data.get("model_type") == "reward_model":
                continue

            # Skip models currently training
            if model_data.get("status") == ModelStatus.TRAINING.value:
                continue

            # Check if model has all prerequisites
            completed_methods = {
                entry["method"]
                for entry in model_data["lineage"]
                if entry.get("status") == "completed"
            }

            if all(req in completed_methods for req in required_methods):
                # Check if already did this method
                if next_method not in completed_methods:
                    models.append(Model.from_dict(model_data))

        return models

    def get_training_models(self) -> List[Model]:
        """Get all models currently in training"""
        return [
            Model.from_dict(data)
            for data in self.state["models"].values()
            if data.get("status") == ModelStatus.TRAINING.value
        ]

    def find_compatible_reward_model(
        self,
        policy_model_id: str
    ) -> Optional[Model]:
        """Find a reward model compatible with a policy model (same base)"""

        policy_model_data = self.state["models"].get(policy_model_id)
        if not policy_model_data:
            return None

        policy_base = policy_model_data["base_model"]

        # Look for reward models with same base
        for model_data in self.state["models"].values():
            if (
                model_data.get("model_type") == "reward_model" and
                model_data["base_model"] == policy_base and
                model_data["status"] == ModelStatus.READY.value
            ):
                return Model.from_dict(model_data)

        return None

    def get_models_by_tag(self, tag: str) -> List[Model]:
        """Get all models with a specific tag"""
        return [
            Model.from_dict(data)
            for data in self.state["models"].values()
            if tag in data.get("tags", [])
        ]

    def add_tag(self, model_id: str, tag: str):
        """Add a tag to a model"""
        if model_id in self.state["models"]:
            tags = self.state["models"][model_id].get("tags", [])
            if tag not in tags:
                tags.append(tag)
                self.state["models"][model_id]["tags"] = tags
                self.save()

    def remove_tag(self, model_id: str, tag: str):
        """Remove a tag from a model"""
        if model_id in self.state["models"]:
            tags = self.state["models"][model_id].get("tags", [])
            if tag in tags:
                tags.remove(tag)
                self.state["models"][model_id]["tags"] = tags
                self.save()

    def update_notes(self, model_id: str, notes: str):
        """Update notes for a model"""
        if model_id in self.state["models"]:
            self.state["models"][model_id]["notes"] = notes
            self.save()

    def delete_model(self, model_id: str):
        """Remove a model from the registry (doesn't delete files)"""
        if model_id in self.state["models"]:
            del self.state["models"][model_id]
            self.save()

    def get_stats(self) -> Dict[str, int]:
        """Get registry statistics"""
        total = len(self.state["models"])
        training = len(self.get_training_models())
        ready_dpo = len(self.get_models_ready_for("dpo"))
        ready_rlhf = len(self.get_models_ready_for("rlhf"))

        reward_models = sum(
            1 for data in self.state["models"].values()
            if data.get("model_type") == "reward_model"
        )

        return {
            "total_models": total,
            "training": training,
            "ready_for_dpo": ready_dpo,
            "ready_for_rlhf": ready_rlhf,
            "reward_models": reward_models
        }
