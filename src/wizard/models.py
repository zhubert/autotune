"""
Data models for progress tracking system

These dataclasses represent the state of training progress, model lineage,
and training pipelines.
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum


class TrainingStatus(str, Enum):
    """Status of a training session"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    ABANDONED = "abandoned"


class ModelStatus(str, Enum):
    """Status of a model"""
    READY = "ready"
    TRAINING = "training"
    FAILED = "failed"


class PhaseStatus(str, Enum):
    """Status of a learning phase"""
    LOCKED = "locked"
    AVAILABLE = "available"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"


class ExperienceLevel(str, Enum):
    """User experience level"""
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    EXPERT = "expert"


@dataclass
class LineageEntry:
    """Single entry in a model's training lineage"""
    method: str  # sft, dpo, reward_model, rlhf
    training_id: str
    checkpoint: str
    parent: Optional[str] = None
    dataset: Optional[str] = None
    completed_at: Optional[str] = None
    status: str = "completed"
    current_step: Optional[int] = None
    total_steps: Optional[int] = None
    metrics: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Model:
    """Represents a trained model with lineage"""
    id: str
    base_model: str
    created_at: str
    lineage: List[LineageEntry]
    tags: List[str] = field(default_factory=list)
    notes: str = ""
    status: ModelStatus = ModelStatus.READY
    capabilities: List[str] = field(default_factory=list)
    model_type: str = "generative"  # generative or reward_model

    def to_dict(self) -> dict:
        data = asdict(self)
        data["status"] = self.status.value
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "Model":
        """Create Model from dictionary"""
        # Convert lineage entries
        lineage = [
            LineageEntry(**entry) if isinstance(entry, dict) else entry
            for entry in data.get("lineage", [])
        ]

        return cls(
            id=data["id"],
            base_model=data["base_model"],
            created_at=data["created_at"],
            lineage=lineage,
            tags=data.get("tags", []),
            notes=data.get("notes", ""),
            status=ModelStatus(data.get("status", "ready")),
            capabilities=data.get("capabilities", []),
            model_type=data.get("model_type", "generative")
        )

    def get_latest_checkpoint(self) -> Optional[str]:
        """Get the most recent checkpoint path"""
        if not self.lineage:
            return None
        return self.lineage[-1].checkpoint

    def has_completed_method(self, method: str) -> bool:
        """Check if model has completed a specific training method"""
        return any(
            entry.method == method and entry.status == "completed"
            for entry in self.lineage
        )

    def get_method_checkpoint(self, method: str) -> Optional[str]:
        """Get checkpoint for a specific training method"""
        for entry in reversed(self.lineage):
            if entry.method == method and entry.status == "completed":
                return entry.checkpoint
        return None


@dataclass
class TrainingSession:
    """Represents a training session"""
    id: str
    method: str
    status: TrainingStatus
    started_at: str
    model_id: Optional[str] = None
    completed_at: Optional[str] = None
    current_step: int = 0
    total_steps: int = 0
    config: Dict[str, Any] = field(default_factory=dict)
    results: Dict[str, Any] = field(default_factory=dict)
    checkpoint_path: Optional[str] = None
    error_message: Optional[str] = None

    def to_dict(self) -> dict:
        data = asdict(self)
        data["status"] = self.status.value
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "TrainingSession":
        return cls(
            id=data["id"],
            method=data["method"],
            status=TrainingStatus(data["status"]),
            started_at=data["started_at"],
            model_id=data.get("model_id"),
            completed_at=data.get("completed_at"),
            current_step=data.get("current_step", 0),
            total_steps=data.get("total_steps", 0),
            config=data.get("config", {}),
            results=data.get("results", {}),
            checkpoint_path=data.get("checkpoint_path"),
            error_message=data.get("error_message")
        )

    @property
    def progress_pct(self) -> float:
        """Get progress percentage"""
        if self.total_steps == 0:
            return 0.0
        return (self.current_step / self.total_steps) * 100

    @property
    def duration_minutes(self) -> Optional[float]:
        """Get duration in minutes if completed"""
        if not self.completed_at:
            return None
        start = datetime.fromisoformat(self.started_at)
        end = datetime.fromisoformat(self.completed_at)
        return (end - start).total_seconds() / 60


@dataclass
class PipelineStage:
    """Single stage in a training pipeline"""
    stage: str
    status: str = "pending"
    model_id: Optional[str] = None
    requires: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Pipeline:
    """Multi-stage training pipeline"""
    id: str
    name: str
    description: str
    stages: List[PipelineStage]
    base_model: Optional[str] = None
    created_at: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Pipeline":
        stages = [
            PipelineStage(**stage) if isinstance(stage, dict) else stage
            for stage in data.get("stages", [])
        ]
        return cls(
            id=data["id"],
            name=data["name"],
            description=data["description"],
            stages=stages,
            base_model=data.get("base_model"),
            created_at=data.get("created_at")
        )

    def get_current_stage(self) -> Optional[PipelineStage]:
        """Get the current/next stage to execute"""
        for stage in self.stages:
            if stage.status in ["pending", "in_progress"]:
                return stage
        return None

    def is_complete(self) -> bool:
        """Check if all stages are completed"""
        return all(stage.status == "completed" for stage in self.stages)


@dataclass
class LearningPhase:
    """Learning path phase status"""
    phase_id: str
    status: PhaseStatus
    training_id: Optional[str] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    unlock_requirements: List[str] = field(default_factory=list)
    quiz_results: Dict[str, bool] = field(default_factory=dict)

    def to_dict(self) -> dict:
        data = asdict(self)
        data["status"] = self.status.value
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "LearningPhase":
        return cls(
            phase_id=data["phase_id"],
            status=PhaseStatus(data["status"]),
            training_id=data.get("training_id"),
            started_at=data.get("started_at"),
            completed_at=data.get("completed_at"),
            unlock_requirements=data.get("unlock_requirements", []),
            quiz_results=data.get("quiz_results", {})
        )


@dataclass
class UserProfile:
    """User profile and preferences"""
    experience_level: ExperienceLevel = ExperienceLevel.BEGINNER
    preferred_mode: str = "guided"
    learning_path_enabled: bool = True
    completed_tutorials: List[str] = field(default_factory=list)
    last_active: Optional[str] = None

    def to_dict(self) -> dict:
        data = asdict(self)
        data["experience_level"] = self.experience_level.value
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "UserProfile":
        return cls(
            experience_level=ExperienceLevel(data.get("experience_level", "beginner")),
            preferred_mode=data.get("preferred_mode", "guided"),
            learning_path_enabled=data.get("learning_path_enabled", True),
            completed_tutorials=data.get("completed_tutorials", []),
            last_active=data.get("last_active")
        )


@dataclass
class CheckpointInfo:
    """Information about a checkpoint on disk"""
    path: str
    created_at: str
    size_mb: float
    has_metadata: bool
    has_model: bool
    metadata: Dict[str, Any] = field(default_factory=dict)
    training_id: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Action:
    """Recommended action"""
    id: str
    title: str
    description: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Recommendations:
    """Training recommendations"""
    priority: str  # high, medium, low
    suggested_action: str
    suggested_action_title: str
    reason: str
    alternatives: List[Action] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class PrereqResult:
    """Result of prerequisite check"""
    satisfied: bool
    missing: List[str] = field(default_factory=list)
    message: str = ""
    suggestions: List[str] = field(default_factory=list)
    available_models: List[Model] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        data = asdict(self)
        data["available_models"] = [m.to_dict() for m in self.available_models]
        return data
