"""
Checkpoint detection and scanning system

Automatically detects training checkpoints on disk and extracts metadata.
"""

import json
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime
import os

from .models import CheckpointInfo, TrainingSession, TrainingStatus


class CheckpointDetector:
    """Scans filesystem for training artifacts and updates progress"""

    def __init__(self, checkpoints_dir: str = "checkpoints"):
        self.checkpoints_dir = Path(checkpoints_dir)

    def scan_all_checkpoints(self) -> Dict[str, List[CheckpointInfo]]:
        """Find all checkpoint directories organized by method"""
        checkpoints = {
            "sft": self._scan_directory("sft"),
            "reward": self._scan_directory("reward"),
            "dpo": self._scan_directory("dpo"),
            "rlhf": self._scan_directory("rlhf")
        }
        return checkpoints

    def _scan_directory(self, method: str) -> List[CheckpointInfo]:
        """Scan a checkpoint directory and extract metadata"""
        method_dir = self.checkpoints_dir / method

        if not method_dir.exists():
            return []

        checkpoints = []
        for ckpt_dir in method_dir.iterdir():
            if not ckpt_dir.is_dir():
                continue

            # Skip hidden directories
            if ckpt_dir.name.startswith('.'):
                continue

            info = self._extract_checkpoint_info(ckpt_dir)
            if info:
                checkpoints.append(info)

        # Sort by creation time, newest first
        return sorted(checkpoints, key=lambda x: x.created_at, reverse=True)

    def _extract_checkpoint_info(self, ckpt_dir: Path) -> Optional[CheckpointInfo]:
        """Extract information from a checkpoint directory"""
        try:
            # Get creation time
            created_at = datetime.fromtimestamp(
                ckpt_dir.stat().st_ctime
            ).isoformat()

            # Calculate directory size
            size_mb = self._get_dir_size_mb(ckpt_dir)

            # Check for various metadata files
            metadata_file = ckpt_dir / "training_metadata.json"
            config_file = ckpt_dir / "config.json"
            trainer_state = ckpt_dir / "trainer_state.json"

            metadata = {}
            training_id = None

            # Try to load metadata
            if metadata_file.exists():
                try:
                    with open(metadata_file) as f:
                        metadata = json.load(f)
                        training_id = metadata.get("training_id")
                except json.JSONDecodeError:
                    pass

            # Try config file
            if config_file.exists() and not metadata:
                try:
                    with open(config_file) as f:
                        metadata["config"] = json.load(f)
                except json.JSONDecodeError:
                    pass

            # Check for PyTorch model files
            has_model = self._has_pytorch_model(ckpt_dir)

            return CheckpointInfo(
                path=str(ckpt_dir),
                created_at=created_at,
                size_mb=size_mb,
                has_metadata=metadata_file.exists(),
                has_model=has_model,
                metadata=metadata,
                training_id=training_id
            )

        except Exception as e:
            # Skip checkpoints that cause errors
            print(f"Warning: Could not scan {ckpt_dir}: {e}")
            return None

    def _get_dir_size_mb(self, directory: Path) -> float:
        """Calculate total size of directory in MB"""
        total_size = 0
        for entry in directory.rglob('*'):
            if entry.is_file():
                try:
                    total_size += entry.stat().st_size
                except (OSError, FileNotFoundError):
                    # Skip files we can't access
                    pass
        return total_size / (1024 * 1024)  # Convert to MB

    def _has_pytorch_model(self, directory: Path) -> bool:
        """Check if directory contains PyTorch model files"""
        # Look for common PyTorch model file patterns
        patterns = [
            "*.pt",
            "*.pth",
            "pytorch_model.bin",
            "model.safetensors",
            "adapter_model.bin",  # LoRA
            "adapter_model.safetensors"  # LoRA
        ]

        for pattern in patterns:
            if list(directory.glob(pattern)):
                return True

        return False

    def detect_running_training(self) -> Optional[TrainingSession]:
        """Check if any training is currently running"""
        # Look for lock files
        for method in ["sft", "reward", "dpo", "rlhf"]:
            lock_file = self.checkpoints_dir / method / ".training_lock"
            if lock_file.exists():
                try:
                    with open(lock_file) as f:
                        data = json.load(f)
                        return TrainingSession.from_dict(data)
                except (json.JSONDecodeError, IOError):
                    # Stale lock file, remove it
                    lock_file.unlink()

        return None

    def create_training_lock(
        self,
        method: str,
        session: TrainingSession
    ):
        """Create a lock file for running training"""
        lock_dir = self.checkpoints_dir / method
        lock_dir.mkdir(parents=True, exist_ok=True)

        lock_file = lock_dir / ".training_lock"
        with open(lock_file, 'w') as f:
            json.dump(session.to_dict(), f, indent=2)

    def remove_training_lock(self, method: str):
        """Remove training lock file"""
        lock_file = self.checkpoints_dir / method / ".training_lock"
        if lock_file.exists():
            lock_file.unlink()

    def find_latest_checkpoint(self, method: str) -> Optional[CheckpointInfo]:
        """Find the most recent checkpoint for a method"""
        checkpoints = self._scan_directory(method)
        if checkpoints:
            return checkpoints[0]  # Already sorted by date
        return None

    def find_checkpoint_by_id(
        self,
        method: str,
        training_id: str
    ) -> Optional[CheckpointInfo]:
        """Find a checkpoint by training ID"""
        checkpoints = self._scan_directory(method)

        for ckpt in checkpoints:
            if ckpt.training_id == training_id:
                return ckpt

            # Also check if directory name matches
            if training_id in ckpt.path:
                return ckpt

        return None

    def get_checkpoint_age_str(self, checkpoint: CheckpointInfo) -> str:
        """Get human-readable age of checkpoint"""
        created = datetime.fromisoformat(checkpoint.created_at)
        now = datetime.now()
        delta = now - created

        if delta.days > 0:
            if delta.days == 1:
                return "1 day ago"
            elif delta.days < 7:
                return f"{delta.days} days ago"
            elif delta.days < 30:
                weeks = delta.days // 7
                return f"{weeks} week{'s' if weeks > 1 else ''} ago"
            else:
                months = delta.days // 30
                return f"{months} month{'s' if months > 1 else ''} ago"

        hours = delta.seconds // 3600
        if hours > 0:
            return f"{hours} hour{'s' if hours > 1 else ''} ago"

        minutes = delta.seconds // 60
        if minutes > 0:
            return f"{minutes} minute{'s' if minutes > 1 else ''} ago"

        return "just now"

    def validate_checkpoint(self, checkpoint: CheckpointInfo) -> Dict[str, bool]:
        """Validate that a checkpoint has all necessary files"""
        ckpt_path = Path(checkpoint.path)

        checks = {
            "directory_exists": ckpt_path.exists(),
            "has_model_files": checkpoint.has_model,
            "has_metadata": checkpoint.has_metadata,
            "has_config": (ckpt_path / "config.json").exists(),
            "has_tokenizer": (ckpt_path / "tokenizer_config.json").exists()
        }

        checks["is_valid"] = all([
            checks["directory_exists"],
            checks["has_model_files"]
        ])

        return checks
