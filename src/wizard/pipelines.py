"""
Pipeline management system

Manages multi-stage training pipelines (e.g., SFT → DPO → RLHF).
"""

from typing import Optional, List
from datetime import datetime

from .models import Pipeline, PipelineStage
from .model_registry import ModelRegistry
from .prerequisites import PrerequisiteChecker


class PipelineManager:
    """Manage multi-stage training pipelines"""

    def __init__(self, registry: ModelRegistry):
        self.registry = registry
        self.prereq_checker = PrerequisiteChecker(registry)
        self.pipelines = registry.state.get("training_pipelines", {})

    def create_pipeline(
        self,
        name: str,
        description: str,
        stages: List[str],
        base_model: Optional[str] = None
    ) -> Pipeline:
        """Create a new training pipeline"""

        pipeline_id = f"pipeline_{int(datetime.now().timestamp())}"

        pipeline = Pipeline(
            id=pipeline_id,
            name=name,
            description=description,
            stages=[
                PipelineStage(stage=stage, status="pending")
                for stage in stages
            ],
            base_model=base_model,
            created_at=datetime.now().isoformat()
        )

        self.pipelines[pipeline_id] = pipeline.to_dict()
        self.registry.state["training_pipelines"] = self.pipelines
        self.registry.save()

        return pipeline

    def get_pipeline(self, pipeline_id: str) -> Optional[Pipeline]:
        """Get a pipeline by ID"""
        if pipeline_id in self.pipelines:
            return Pipeline.from_dict(self.pipelines[pipeline_id])
        return None

    def list_pipelines(self) -> List[Pipeline]:
        """List all pipelines"""
        return [Pipeline.from_dict(p) for p in self.pipelines.values()]

    def get_active_pipelines(self) -> List[Pipeline]:
        """Get pipelines that are not complete"""
        return [
            Pipeline.from_dict(p)
            for p in self.pipelines.values()
            if not self._is_pipeline_complete(p)
        ]

    def _is_pipeline_complete(self, pipeline_data: dict) -> bool:
        """Check if pipeline is complete"""
        return all(
            stage["status"] == "completed"
            for stage in pipeline_data.get("stages", [])
        )

    def update_stage_status(
        self,
        pipeline_id: str,
        stage_index: int,
        status: str,
        model_id: Optional[str] = None
    ):
        """Update status of a pipeline stage"""
        if pipeline_id in self.pipelines:
            stages = self.pipelines[pipeline_id]["stages"]
            if 0 <= stage_index < len(stages):
                stages[stage_index]["status"] = status
                if model_id:
                    stages[stage_index]["model_id"] = model_id
                self.registry.save()

    def get_next_stage(self, pipeline_id: str) -> Optional[tuple]:
        """Get the next pending stage in a pipeline"""
        pipeline = self.get_pipeline(pipeline_id)
        if not pipeline:
            return None

        for i, stage in enumerate(pipeline.stages):
            if stage.status == "pending":
                return (i, stage)

        return None

    def can_start_stage(self, pipeline_id: str, stage_index: int) -> bool:
        """Check if a stage can be started"""
        pipeline = self.get_pipeline(pipeline_id)
        if not pipeline:
            return False

        # Check if previous stage is complete
        if stage_index > 0:
            prev_stage = pipeline.stages[stage_index - 1]
            if prev_stage.status != "completed":
                return False

        return True

    def complete_stage(
        self,
        pipeline_id: str,
        stage_index: int,
        model_id: str
    ):
        """Mark a pipeline stage as completed"""
        self.update_stage_status(pipeline_id, stage_index, "completed", model_id)

    def fail_stage(self, pipeline_id: str, stage_index: int):
        """Mark a pipeline stage as failed"""
        self.update_stage_status(pipeline_id, stage_index, "failed")

    def delete_pipeline(self, pipeline_id: str):
        """Delete a pipeline"""
        if pipeline_id in self.pipelines:
            del self.pipelines[pipeline_id]
            self.registry.save()

    # Predefined pipeline templates

    @staticmethod
    def get_template(template_name: str) -> dict:
        """Get predefined pipeline template"""

        templates = {
            "full_rlhf": {
                "name": "Full RLHF Pipeline",
                "description": "Complete pipeline: SFT → Reward Model → RLHF",
                "stages": ["sft", "reward_model", "rlhf"]
            },
            "dpo_simple": {
                "name": "Simple DPO Pipeline",
                "description": "Lightweight pipeline: SFT → DPO",
                "stages": ["sft", "dpo"]
            },
            "comparison": {
                "name": "DPO vs RLHF Comparison",
                "description": "Train both methods: SFT → [DPO + Reward Model → RLHF]",
                "stages": ["sft", "dpo", "reward_model", "rlhf"]
            },
            "sft_only": {
                "name": "SFT Only",
                "description": "Just supervised fine-tuning",
                "stages": ["sft"]
            }
        }

        return templates.get(template_name, {})

    def create_from_template(
        self,
        template_name: str,
        base_model: Optional[str] = None
    ) -> Optional[Pipeline]:
        """Create pipeline from a template"""

        template = self.get_template(template_name)
        if not template:
            return None

        return self.create_pipeline(
            name=template["name"],
            description=template["description"],
            stages=template["stages"],
            base_model=base_model
        )

    def get_pipeline_progress(self, pipeline_id: str) -> dict:
        """Get progress statistics for a pipeline"""
        pipeline = self.get_pipeline(pipeline_id)
        if not pipeline:
            return {}

        total_stages = len(pipeline.stages)
        completed = sum(1 for s in pipeline.stages if s.status == "completed")
        in_progress = sum(1 for s in pipeline.stages if s.status == "in_progress")
        failed = sum(1 for s in pipeline.stages if s.status == "failed")

        return {
            "total_stages": total_stages,
            "completed": completed,
            "in_progress": in_progress,
            "failed": failed,
            "pending": total_stages - completed - in_progress - failed,
            "progress_pct": (completed / total_stages * 100) if total_stages > 0 else 0,
            "is_complete": completed == total_stages
        }
