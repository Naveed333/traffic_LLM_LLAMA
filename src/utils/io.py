"""Save/load results, JSON serialization, atomic file writes."""

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy arrays and scalars."""

    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


def save_results(results: Dict, output_path: str) -> None:
    """Save experiment results to a JSON file using atomic write.

    Args:
        results: Results dictionary (may contain numpy arrays/scalars).
        output_path: Full path to the output JSON file.
    """
    dir_path = os.path.dirname(output_path) or "."
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
            json.dump(results, tmp_file, indent=2, cls=NumpyEncoder)
            tmp_path = tmp_file.name

        os.replace(tmp_path, output_path)
        logger.info(f"Results saved to: {output_path}")

    except OSError as e:
        logger.error(f"Failed to save results to {output_path}: {e}")
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise


def load_results(input_path: str) -> Optional[Dict]:
    """Load results from a JSON file.

    Args:
        input_path: Path to the JSON file.

    Returns:
        Loaded dictionary, or None if loading fails.
    """
    try:
        with open(input_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, OSError) as e:
        logger.error(f"Failed to load results from {input_path}: {e}")
        return None


def ensure_dir(path: str) -> None:
    """Create directory if it doesn't exist."""
    Path(path).mkdir(parents=True, exist_ok=True)
