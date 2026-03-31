"""JSON-based checkpointing: save/load/resume per-sample experiment progress."""

import json
import logging
import os
import tempfile
from typing import Dict, List

logger = logging.getLogger(__name__)


class Checkpoint:
    """Manages experiment checkpoints for resumable runs.

    Saves progress after every completed training sample using atomic writes
    (write to temp file, then os.replace to final path).
    """

    def __init__(self, checkpoint_path: str, config_id: str) -> None:
        """Initialize checkpoint manager.

        Args:
            checkpoint_path: Path to checkpoint JSON file.
            config_id: Experiment configuration ID.
        """
        self.checkpoint_path = checkpoint_path
        self.config_id = config_id
        self._data: Dict = {
            "config_id": config_id,
            "completed_train_samples": [],
            "last_completed_index": -1,
        }

    def exists(self) -> bool:
        """Check if a checkpoint file exists."""
        return os.path.exists(self.checkpoint_path)

    def load(self) -> bool:
        """Load existing checkpoint from disk.

        Returns:
            True if loaded successfully, False otherwise.
        """
        if not self.exists():
            return False

        try:
            with open(self.checkpoint_path, "r", encoding="utf-8") as f:
                loaded = json.load(f)

            if loaded.get("config_id") != self.config_id:
                logger.warning(
                    f"Checkpoint config_id mismatch: "
                    f"expected={self.config_id}, found={loaded.get('config_id')}. "
                    f"Ignoring checkpoint."
                )
                return False

            self._data = loaded
            n = len(self._data.get("completed_train_samples", []))
            last = self._data.get("last_completed_index", -1)
            logger.info(
                f"Loaded checkpoint: {n} completed samples, last_completed_index={last}"
            )
            return True

        except (json.JSONDecodeError, KeyError, OSError) as e:
            logger.warning(f"Failed to load checkpoint: {e}. Starting fresh.")
            return False

    def save_sample(self, sample_result: Dict) -> None:
        """Append a completed sample result and save to disk atomically.

        Args:
            sample_result: Dict with sample results (must include 'index').
        """
        self._data["completed_train_samples"].append(sample_result)
        self._data["last_completed_index"] = sample_result.get("index", -1)
        self._atomic_write()

    def _atomic_write(self) -> None:
        """Write checkpoint to disk using a temp file + os.replace for atomicity."""
        dir_path = os.path.dirname(self.checkpoint_path) or "."
        os.makedirs(dir_path, exist_ok=True)

        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                suffix=".tmp",
                dir=dir_path,
                delete=False,
                encoding="utf-8",
            ) as tmp_file:
                json.dump(self._data, tmp_file, indent=2, default=str)
                tmp_path = tmp_file.name

            os.replace(tmp_path, self.checkpoint_path)
            logger.debug(f"Checkpoint saved: {self.checkpoint_path}")

        except OSError as e:
            logger.error(f"Failed to write checkpoint: {e}")
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)

    @property
    def last_completed_index(self) -> int:
        """Return the index of the last completed training sample."""
        return self._data.get("last_completed_index", -1)

    @property
    def completed_samples(self) -> List[Dict]:
        """Return list of all completed sample results."""
        return self._data.get("completed_train_samples", [])

    def delete(self) -> None:
        """Delete the checkpoint file on successful experiment completion."""
        if self.exists():
            try:
                os.unlink(self.checkpoint_path)
                logger.info(f"Checkpoint deleted: {self.checkpoint_path}")
            except OSError as e:
                logger.warning(f"Failed to delete checkpoint: {e}")
