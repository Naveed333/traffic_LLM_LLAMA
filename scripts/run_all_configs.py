#!/usr/bin/env python3
"""Run all 6 TrafficLLM experimental configs sequentially.

Manages vLLM server lifecycle: stops/starts server as needed, batches configs
by model to minimize server restarts (all 3 context windows for 8B, then all 3 for 1B).

Usage:
    python scripts/run_all_configs.py --config configs/default.yaml --data-dir ./data/milan/
"""

import argparse
import logging
import os
import subprocess
import sys
import time

import requests
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.logger import setup_logging

logger = logging.getLogger(__name__)

VLLM_HEALTH_URL = "http://localhost:8000/v1/models"
VLLM_STARTUP_TIMEOUT = 300  # seconds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run all 6 TrafficLLM configs sequentially.")
    parser.add_argument("--config", default="configs/default.yaml", help="YAML config path.")
    parser.add_argument("--data-dir", help="Override data directory.")
    parser.add_argument("--output-dir", default="./results/", help="Results output directory.")
    parser.add_argument("--force-restart", action="store_true", help="Ignore checkpoints.")
    parser.add_argument("--skip-configs", nargs="+", default=[], help="Config IDs to skip.")
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def stop_vllm_server(port: int = 8000) -> None:
    """Stop any running vLLM server on the given port."""
    try:
        result = subprocess.run(
            ["lsof", "-ti", f"tcp:{port}"],
            capture_output=True, text=True,
        )
        pids = result.stdout.strip().split("\n")
        for pid in pids:
            if pid.strip():
                logger.info(f"Killing process {pid} on port {port}.")
                subprocess.run(["kill", "-9", pid], check=False)
        time.sleep(3)
    except Exception as e:
        logger.warning(f"Error stopping vLLM server: {e}")


def start_vllm_server(model_id: str, max_model_len: int) -> subprocess.Popen:
    """Launch vLLM server as a subprocess."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    start_script = os.path.join(script_dir, "start_vllm.sh")

    cmd = ["bash", start_script, model_id, str(max_model_len)]
    logger.info(f"Starting vLLM: {' '.join(cmd)}")

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return proc


def wait_for_vllm(timeout: int = VLLM_STARTUP_TIMEOUT) -> bool:
    """Poll vLLM health endpoint until ready or timeout."""
    logger.info(f"Waiting for vLLM server (timeout={timeout}s)...")
    elapsed = 0
    while elapsed < timeout:
        try:
            response = requests.get(VLLM_HEALTH_URL, timeout=5)
            if response.status_code == 200:
                logger.info("vLLM server is ready.")
                return True
        except Exception:
            pass
        time.sleep(5)
        elapsed += 5
    logger.error("vLLM server did not become ready in time.")
    return False


def run_experiment(
    config_path: str,
    config_id: str,
    data_dir: str,
    output_dir: str,
    force_restart: bool,
) -> bool:
    """Run a single experiment config via subprocess."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    script = os.path.join(script_dir, "run_experiment.py")

    cmd = [
        sys.executable, script,
        "--config", config_path,
        "--config-id", config_id,
        "--data-dir", data_dir,
        "--output-dir", output_dir,
    ]
    if force_restart:
        cmd.append("--force-restart")

    logger.info(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        logger.error(f"Experiment {config_id} failed with return code {result.returncode}.")
        return False
    return True


def group_by_model(experiments: list) -> dict:
    """Group experiment configs by model_id to minimize server restarts."""
    groups = {}
    for exp in experiments:
        model = exp["model_id"]
        groups.setdefault(model, []).append(exp)
    return groups


def main() -> None:
    args = parse_args()
    setup_logging(level="INFO", log_dir="./logs/", config_id="run_all")

    cfg = load_config(args.config)
    data_dir = args.data_dir or cfg["dataset"]["data_dir"]

    experiments = cfg["experiments"]
    grouped = group_by_model(experiments)

    logger.info(f"Running {len(experiments)} configs across {len(grouped)} models.")
    logger.info(f"Skipping: {args.skip_configs}")

    failed_configs = []
    current_model = None

    for model_id, exps in grouped.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"Model: {model_id} — {len(exps)} configs")
        logger.info(f"{'='*60}")

        # Start vLLM once per model (use largest context window for flexibility)
        max_context = max(e["context_window"] for e in exps)

        if current_model != model_id:
            stop_vllm_server()
            proc = start_vllm_server(model_id, max_context)
            if not wait_for_vllm():
                logger.error(f"Failed to start vLLM for model {model_id}. Skipping all its configs.")
                failed_configs.extend([e["config_id"] for e in exps])
                continue
            current_model = model_id

        # Run each context window config for this model
        # Sort by context window descending (largest first, already loaded)
        for exp in sorted(exps, key=lambda e: e["context_window"], reverse=True):
            config_id = exp["config_id"]

            if config_id in args.skip_configs:
                logger.info(f"Skipping config: {config_id}")
                continue

            logger.info(f"\n--- Config: {config_id} ---")

            # For smaller context windows, we don't need to restart the server
            # (vLLM serves at max_model_len; smaller prompts just won't use all of it)
            # But we need to track context_window in the algorithm, not in vLLM here.

            success = run_experiment(
                config_path=args.config,
                config_id=config_id,
                data_dir=data_dir,
                output_dir=args.output_dir,
                force_restart=args.force_restart,
            )

            if not success:
                failed_configs.append(config_id)

    # ── Final report ─────────────────────────────────────────────────────────────
    stop_vllm_server()

    logger.info(f"\n{'='*60}")
    logger.info("All experiments complete.")
    if failed_configs:
        logger.warning(f"Failed configs: {failed_configs}")
    else:
        logger.info("All configs completed successfully.")
    logger.info(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
