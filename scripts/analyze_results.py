#!/usr/bin/env python3
"""Load results from all 6 configs and generate comparison tables and plots.

Usage:
    python scripts/analyze_results.py --results-dir ./results/ --output-dir ./analysis/
"""

import argparse
import csv
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.io import load_results
from src.utils.logger import setup_logging

logger = logging.getLogger(__name__)

CONFIG_ORDER = ["8B_128K", "8B_64K", "8B_4K", "1B_128K", "1B_64K", "1B_4K"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze and visualize TrafficLLM experiment results.")
    parser.add_argument("--results-dir", default="./results/", help="Directory with result JSON files.")
    parser.add_argument("--output-dir", default="./analysis/", help="Directory to save analysis outputs.")
    return parser.parse_args()


def load_all_results(results_dir: str) -> Dict[str, dict]:
    """Load all available result JSON files."""
    results = {}
    for config_id in CONFIG_ORDER:
        path = os.path.join(results_dir, f"{config_id}_results.json")
        data = load_results(path)
        if data:
            results[config_id] = data
            logger.info(f"Loaded: {config_id}")
        else:
            logger.warning(f"Missing results for: {config_id} (expected at {path})")
    return results


def compute_summary(results: Dict[str, dict]) -> List[Dict]:
    """Compute summary statistics per config."""
    rows = []
    for config_id in CONFIG_ORDER:
        if config_id not in results:
            continue

        data = results[config_id]
        test_phase = data.get("test_phase", {})
        train_phase = data.get("training_phase", {})

        avg_mae = test_phase.get("avg_mae", float("nan"))
        avg_mse = test_phase.get("avg_mse", float("nan"))

        # Compute average refinement iterations
        per_sample_train = train_phase.get("per_sample", [])
        avg_iters = 0.0
        context_exceeded_pct = 0.0

        if per_sample_train:
            iters = [s.get("iterations_completed", 0) for s in per_sample_train]
            exceeded = [1 if s.get("context_window_exceeded", False) else 0 for s in per_sample_train]
            avg_iters = sum(iters) / len(iters)
            context_exceeded_pct = 100.0 * sum(exceeded) / len(exceeded)

        rows.append({
            "Config": config_id,
            "Model": data.get("model_id", ""),
            "Context": data.get("context_window", ""),
            "Avg MAE": f"{avg_mae:.4f}" if avg_mae == avg_mae else "N/A",
            "Avg MSE": f"{avg_mse:.4f}" if avg_mse == avg_mse else "N/A",
            "Avg Iterations": f"{avg_iters:.2f}",
            "Context Exceeded (%)": f"{context_exceeded_pct:.1f}",
            "Test Samples": test_phase.get("num_evaluated", 0),
        })

    return rows


def print_table(rows: List[Dict]) -> None:
    """Print results as a formatted table."""
    if not rows:
        print("No results to display.")
        return

    headers = list(rows[0].keys())
    col_widths = {h: max(len(h), max(len(str(r[h])) for r in rows)) for h in headers}

    header_line = " | ".join(h.ljust(col_widths[h]) for h in headers)
    separator = "-+-".join("-" * col_widths[h] for h in headers)

    print("\n" + "=" * len(header_line))
    print("TrafficLLM Experimental Results")
    print("=" * len(header_line))
    print(header_line)
    print(separator)
    for row in rows:
        print(" | ".join(str(row[h]).ljust(col_widths[h]) for h in headers))
    print("=" * len(header_line) + "\n")


def save_csv(rows: List[Dict], output_path: str) -> None:
    """Save summary table as CSV."""
    if not rows:
        return
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    logger.info(f"Saved CSV: {output_path}")


def generate_plots(results: Dict[str, dict], output_dir: str) -> None:
    """Generate matplotlib plots: bar charts, heatmap, convergence line plot."""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        logger.warning("matplotlib/numpy not available. Skipping plot generation.")
        return

    os.makedirs(output_dir, exist_ok=True)

    configs = [c for c in CONFIG_ORDER if c in results]
    if not configs:
        logger.warning("No results for plotting.")
        return

    maes = []
    mses = []
    for c in configs:
        tp = results[c].get("test_phase", {})
        maes.append(tp.get("avg_mae", 0.0) or 0.0)
        mses.append(tp.get("avg_mse", 0.0) or 0.0)

    # ── 1. Bar chart: MAE by config ─────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(configs, maes, color=["#2196F3" if "8B" in c else "#FF5722" for c in configs])
    ax.set_title("Average MAE by Configuration", fontsize=14)
    ax.set_xlabel("Configuration")
    ax.set_ylabel("Average MAE")
    ax.bar_label(bars, fmt="%.3f", padding=3)
    plt.xticks(rotation=15)
    plt.tight_layout()
    path = os.path.join(output_dir, "mae_by_config.png")
    plt.savefig(path, dpi=150)
    plt.close()
    logger.info(f"Saved: {path}")

    # ── 2. Bar chart: MSE by config ─────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(configs, mses, color=["#4CAF50" if "8B" in c else "#9C27B0" for c in configs])
    ax.set_title("Average MSE by Configuration", fontsize=14)
    ax.set_xlabel("Configuration")
    ax.set_ylabel("Average MSE")
    ax.bar_label(bars, fmt="%.2f", padding=3)
    plt.xticks(rotation=15)
    plt.tight_layout()
    path = os.path.join(output_dir, "mse_by_config.png")
    plt.savefig(path, dpi=150)
    plt.close()
    logger.info(f"Saved: {path}")

    # ── 3. Heatmap: Model × Context → MAE ──────────────────────────────────────
    models = ["Llama-3.1-8B", "Llama-3.2-1B"]
    contexts = ["128K", "64K", "4K"]
    mae_matrix = np.zeros((len(models), len(contexts)))

    config_map = {
        "8B_128K": (0, 0), "8B_64K": (0, 1), "8B_4K": (0, 2),
        "1B_128K": (1, 0), "1B_64K": (1, 1), "1B_4K": (1, 2),
    }

    for cfg_id, (mi, ci) in config_map.items():
        if cfg_id in results:
            mae_matrix[mi][ci] = results[cfg_id].get("test_phase", {}).get("avg_mae", 0.0) or 0.0

    fig, ax = plt.subplots(figsize=(8, 4))
    im = ax.imshow(mae_matrix, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(len(contexts)))
    ax.set_xticklabels(contexts)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models)
    ax.set_title("Average MAE: Model Size × Context Window", fontsize=13)
    plt.colorbar(im, ax=ax, label="Avg MAE")
    for mi in range(len(models)):
        for ci in range(len(contexts)):
            ax.text(ci, mi, f"{mae_matrix[mi][ci]:.3f}", ha="center", va="center", fontsize=10)
    plt.tight_layout()
    path = os.path.join(output_dir, "mae_heatmap.png")
    plt.savefig(path, dpi=150)
    plt.close()
    logger.info(f"Saved: {path}")

    # ── 4. Line plot: MAE convergence across refinement iterations ──────────────
    fig, ax = plt.subplots(figsize=(10, 5))

    for cfg_id in configs:
        per_sample = results[cfg_id].get("training_phase", {}).get("per_sample", [])
        if not per_sample:
            continue

        # Compute average MAE at each iteration index
        max_iters = max(len(s.get("mae_history", [])) for s in per_sample)
        if max_iters == 0:
            continue

        avg_mae_per_iter = []
        for it in range(max_iters):
            values = [
                s["mae_history"][it]
                for s in per_sample
                if len(s.get("mae_history", [])) > it
            ]
            avg_mae_per_iter.append(np.mean(values) if values else 0.0)

        ax.plot(range(max_iters), avg_mae_per_iter, marker="o", label=cfg_id)

    ax.set_title("MAE Convergence Across Refinement Iterations (Training)", fontsize=13)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Average MAE")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    path = os.path.join(output_dir, "mae_convergence.png")
    plt.savefig(path, dpi=150)
    plt.close()
    logger.info(f"Saved: {path}")


def main() -> None:
    args = parse_args()
    setup_logging(level="INFO")

    logger.info(f"Loading results from: {args.results_dir}")
    results = load_all_results(args.results_dir)

    if not results:
        logger.error("No results found. Run experiments first.")
        sys.exit(1)

    # Summary table
    summary_rows = compute_summary(results)
    print_table(summary_rows)

    # Save CSV
    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, "results_summary.csv")
    save_csv(summary_rows, csv_path)

    # Generate plots
    generate_plots(results, args.output_dir)

    logger.info(f"Analysis complete. Outputs saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
