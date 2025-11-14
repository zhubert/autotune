"""
Main progress tracking coordinator

Combines all tracking components and provides high-level interface.
"""

import json
from pathlib import Path
from typing import Optional, List, Dict
from datetime import datetime

from .models import (
    UserProfile, TrainingSession, TrainingStatus,
    LearningPhase, PhaseStatus
)
from .model_registry import ModelRegistry
from .checkpoint_detector import CheckpointDetector
from .prerequisites import PrerequisiteChecker
from .recommendations import RecommendationEngine


class ProgressTracker:
    """Central coordinator for progress tracking"""

    def __init__(self, progress_file: str = ".llm-training-progress.json"):
        self.progress_file = Path(progress_file)
        self.state = self.load()

        # Initialize components
        self.registry = ModelRegistry(progress_file)
        self.detector = CheckpointDetector()
        self.prereq_checker = PrerequisiteChecker(self.registry)
        self.recommender = RecommendationEngine(self.registry)

    def load(self) -> dict:
        """Load progress state from disk"""
        if not self.progress_file.exists():
            return self._create_empty_state()

        try:
            with open(self.progress_file, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return self._create_empty_state()

    def save(self):
        """Save progress state to disk"""
        self.progress_file.parent.mkdir(parents=True, exist_ok=True)

        with open(self.progress_file, 'w') as f:
            json.dump(self.state, f, indent=2)

    def _create_empty_state(self) -> dict:
        """Create empty progress state"""
        return {
            "version": "2.0",
            "user_profile": UserProfile().to_dict(),
            "models": {},
            "training_sessions": [],
            "learning_phases": {
                "phase_1_sft": LearningPhase(
                    phase_id="phase_1_sft",
                    status=PhaseStatus.AVAILABLE
                ).to_dict(),
                "phase_2_reward": LearningPhase(
                    phase_id="phase_2_reward",
                    status=PhaseStatus.LOCKED,
                    unlock_requirements=["phase_1_sft"]
                ).to_dict(),
                "phase_3_dpo": LearningPhase(
                    phase_id="phase_3_dpo",
                    status=PhaseStatus.LOCKED,
                    unlock_requirements=["phase_1_sft"]
                ).to_dict(),
                "phase_4_rlhf": LearningPhase(
                    phase_id="phase_4_rlhf",
                    status=PhaseStatus.LOCKED,
                    unlock_requirements=["phase_1_sft", "phase_2_reward"]
                ).to_dict()
            },
            "training_pipelines": {},
            "model_comparisons": [],
            "achievements": []
        }

    # User Profile Methods

    def get_user_profile(self) -> UserProfile:
        """Get user profile"""
        return UserProfile.from_dict(self.state.get("user_profile", {}))

    def update_user_profile(self, **kwargs):
        """Update user profile fields"""
        profile = self.get_user_profile()
        for key, value in kwargs.items():
            if hasattr(profile, key):
                setattr(profile, key, value)
        self.state["user_profile"] = profile.to_dict()
        self.save()

    # Training Session Methods

    def start_training_session(
        self,
        method: str,
        config: dict,
        model_id: Optional[str] = None,
        total_steps: int = 0
    ) -> str:
        """Start a new training session"""
        session_id = f"{method}_{int(datetime.now().timestamp())}"

        session = TrainingSession(
            id=session_id,
            method=method,
            status=TrainingStatus.IN_PROGRESS,
            started_at=datetime.now().isoformat(),
            model_id=model_id,
            config=config,
            total_steps=total_steps
        )

        self.state["training_sessions"].append(session.to_dict())

        # Create training lock
        self.detector.create_training_lock(method, session)

        self.save()
        return session_id

    def update_training_progress(
        self,
        session_id: str,
        current_step: int,
        metrics: Dict[str, float] = None
    ):
        """Update training session progress"""
        for session_data in self.state["training_sessions"]:
            if session_data["id"] == session_id:
                session_data["current_step"] = current_step
                if metrics:
                    if "metrics" not in session_data:
                        session_data["metrics"] = {}
                    session_data["metrics"].update(metrics)

                # Also update model registry if model_id is set
                model_id = session_data.get("model_id")
                if model_id:
                    self.registry.update_training_progress(
                        model_id, session_id, current_step, metrics
                    )

                break

        self.save()

    def complete_training_session(
        self,
        session_id: str,
        checkpoint_path: str,
        final_metrics: Dict[str, float] = None
    ):
        """Mark training session as completed"""
        session = None
        for session_data in self.state["training_sessions"]:
            if session_data["id"] == session_id:
                session_data["status"] = TrainingStatus.COMPLETED.value
                session_data["completed_at"] = datetime.now().isoformat()
                session_data["checkpoint_path"] = checkpoint_path
                if final_metrics:
                    session_data["results"] = final_metrics
                session = TrainingSession.from_dict(session_data)
                break

        if session:
            # Remove training lock
            self.detector.remove_training_lock(session.method)

            # Update model registry if model_id is set
            if session.model_id:
                self.registry.complete_model_training(
                    session.model_id, session_id, final_metrics
                )

            # Update learning phases if applicable
            self._update_learning_phases(session.method)

        self.save()

    def fail_training_session(self, session_id: str, error: str):
        """Mark training session as failed"""
        for session_data in self.state["training_sessions"]:
            if session_data["id"] == session_id:
                session_data["status"] = TrainingStatus.FAILED.value
                session_data["completed_at"] = datetime.now().isoformat()
                session_data["error_message"] = error

                # Remove training lock
                method = session_data.get("method")
                if method:
                    self.detector.remove_training_lock(method)
                break

        self.save()

    def get_active_training(self) -> Optional[TrainingSession]:
        """Get currently active training session"""
        for session_data in self.state["training_sessions"]:
            if session_data["status"] == TrainingStatus.IN_PROGRESS.value:
                return TrainingSession.from_dict(session_data)
        return None

    def get_recent_sessions(self, limit: int = 5) -> List[TrainingSession]:
        """Get recent training sessions"""
        sessions = [
            TrainingSession.from_dict(s)
            for s in self.state["training_sessions"]
        ]
        # Sort by started_at, most recent first
        sessions.sort(key=lambda s: s.started_at, reverse=True)
        return sessions[:limit]

    # Learning Phase Methods

    def _update_learning_phases(self, completed_method: str):
        """Update learning phase status based on completed training"""
        phase_map = {
            "sft": "phase_1_sft",
            "reward_model": "phase_2_reward",
            "dpo": "phase_3_dpo",
            "rlhf": "phase_4_rlhf"
        }

        phase_id = phase_map.get(completed_method)
        if not phase_id:
            return

        phases = self.state.get("learning_phases", {})
        if phase_id in phases:
            phases[phase_id]["status"] = PhaseStatus.COMPLETED.value
            phases[phase_id]["completed_at"] = datetime.now().isoformat()

            # Unlock dependent phases
            self._unlock_dependent_phases(phase_id)

    def _unlock_dependent_phases(self, completed_phase_id: str):
        """Unlock phases that depend on the completed phase"""
        phases = self.state.get("learning_phases", {})

        for phase_id, phase_data in phases.items():
            if phase_data["status"] == PhaseStatus.LOCKED.value:
                requirements = phase_data.get("unlock_requirements", [])
                # Check if all requirements are met
                all_met = all(
                    phases.get(req, {}).get("status") == PhaseStatus.COMPLETED.value
                    for req in requirements
                )
                if all_met:
                    phase_data["status"] = PhaseStatus.AVAILABLE.value

    def get_learning_phase(self, phase_id: str) -> Optional[LearningPhase]:
        """Get a specific learning phase"""
        phases = self.state.get("learning_phases", {})
        if phase_id in phases:
            return LearningPhase.from_dict(phases[phase_id])
        return None

    def get_next_learning_phase(self) -> Optional[str]:
        """Get the next available learning phase"""
        phases = self.state.get("learning_phases", {})
        phase_order = ["phase_1_sft", "phase_2_reward", "phase_3_dpo", "phase_4_rlhf"]

        for phase_id in phase_order:
            if phase_id in phases:
                status = phases[phase_id].get("status")
                if status in [PhaseStatus.AVAILABLE.value, PhaseStatus.IN_PROGRESS.value]:
                    return phase_id

        return None

    # Statistics and Summary

    def get_stats(self) -> Dict:
        """Get overall progress statistics"""
        registry_stats = self.registry.get_stats()

        sessions = self.state.get("training_sessions", [])
        completed_sessions = sum(
            1 for s in sessions
            if s.get("status") == TrainingStatus.COMPLETED.value
        )

        phases = self.state.get("learning_phases", {})
        completed_phases = sum(
            1 for p in phases.values()
            if p.get("status") == PhaseStatus.COMPLETED.value
        )

        return {
            **registry_stats,
            "total_training_sessions": len(sessions),
            "completed_sessions": completed_sessions,
            "completed_phases": completed_phases,
            "total_phases": len(phases)
        }

    def sync_with_filesystem(self):
        """Scan filesystem and update registry with any missing checkpoints"""
        all_checkpoints = self.detector.scan_all_checkpoints()

        # Check if there are checkpoints not in the registry
        # This would happen if user manually copied checkpoints or state file was deleted

        for method, checkpoints in all_checkpoints.items():
            for ckpt in checkpoints:
                # Check if this checkpoint is already tracked
                is_tracked = False
                for model in self.registry.get_all_models():
                    for entry in model.lineage:
                        if entry.checkpoint == ckpt.path:
                            is_tracked = True
                            break
                    if is_tracked:
                        break

                # If not tracked and has metadata, could register it
                # For now, just report it
                if not is_tracked and ckpt.has_metadata:
                    print(f"Found untracked checkpoint: {ckpt.path}")
