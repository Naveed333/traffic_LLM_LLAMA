#!/usr/bin/env python3
"""Main entry point: run one TrafficLLM experimental config (model + context window).

Usage:
    python scripts/run_experiment.py \
        --config configs/default.yaml \
        --config-id 8B_128K \
        --data-dir ./data/milan/ \
        --max-iterations 3 \
        --output-dir ./results/
"""

import argparse
import datetime
import os
import random
import sys
import time

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.algorithm.trafficllm import TrafficLLM
from src.data.loader import load_daily_files, get_available_cell_ids
from src.data.preprocessor import preprocess_base_station
from src.data.splitter import train_test_split, select_source_example
from src.llm.client import LLMClient
from src.prompts.builder import build_pexam
from src.utils.checkpoint import Checkpoint
from src.utils.io import save_results, ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run TrafficLLM experiment for a single model+context config."
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--config-id", required=True)
    parser.add_argument("--data-dir")
    parser.add_argument("--max-iterations", type=int)
    parser.add_argument("--output-dir", default="./results/")
    parser.add_argument("--force-restart", action="store_true")
    parser.add_argument("--log-dir")
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def find_experiment_config(experiments: list, config_id: str) -> dict:
    for exp in experiments:
        if exp["config_id"] == config_id:
            return exp
    raise ValueError(
        f"Config ID '{config_id}' not found. Available: {[e['config_id'] for e in experiments]}"
    )


def select_best_pexam(
    source_x, source_y, source_input_dates, source_target_dates,
    train_results, target_x_train, target_y_train,
    target_input_dates_train, target_target_dates_train,
) -> str:
    if not train_results:
        x_ex, y_ex, in_d, out_d = select_source_example(
            source_x, source_y, source_input_dates, source_target_dates
        )
        return build_pexam(x_ex, y_ex, in_d, out_d)

    best_idx = None
    best_mae = float("inf")

    for result in train_results:
        idx = result.get("index", -1)
        mae_history = result.get("mae_history", [])
        final_mae = mae_history[-1] if mae_history else float("inf")

        if final_mae < best_mae and result.get("final_prediction") is not None:
            best_mae = final_mae
            best_idx = idx

    if best_idx is None or best_idx >= len(target_x_train):
        x_ex, y_ex, in_d, out_d = select_source_example(
            source_x, source_y, source_input_dates, source_target_dates
        )
        return build_pexam(x_ex, y_ex, in_d, out_d)

    print(f"Best training example: index={best_idx}, MAE={best_mae:.4f}")

    x_best = target_x_train[best_idx]
    y_best_pred = np.array(train_results[best_idx]["final_prediction"])
    in_date_best = target_input_dates_train[best_idx]
    out_date_best = target_target_dates_train[best_idx]

    return build_pexam(x_best, y_best_pred, in_date_best, out_date_best)


def main() -> None:
    args = parse_args()

    np.random.seed(42)
    random.seed(42)

    cfg = load_config(args.config)
    exp_cfg = find_experiment_config(cfg["experiments"], args.config_id)

    data_dir = args.data_dir or cfg["dataset"]["data_dir"]
    target_cells = cfg["dataset"]["target_cells"]
    spatial_method = cfg["dataset"].get("spatial_method", "sum")
    max_iterations = args.max_iterations or cfg["algorithm"]["max_refinement_iterations"]
    log_dir = args.log_dir or cfg["logging"]["log_dir"]

    model_id = exp_cfg["model_id"]
    context_window = exp_cfg["context_window"]

    print(f"Starting experiment: config_id={args.config_id}, model={model_id}, context={context_window}")
    print(f"Data dir: {data_dir}")
    print(f"Base station cells ({len(target_cells)}): {target_cells}")
    print(f"Spatial aggregation method: {spatial_method}")

    ensure_dir(args.output_dir)

    checkpoint_path = os.path.join(args.output_dir, f"{args.config_id}_checkpoint.json")
    checkpoint = Checkpoint(checkpoint_path=checkpoint_path, config_id=args.config_id)

    if not args.force_restart and checkpoint.load():
        print(f"Resuming from checkpoint. Last completed index: {checkpoint.last_completed_index}")

    prompt_log_path = None
    if cfg["logging"].get("save_all_prompts", False):
        ensure_dir(log_dir)
        prompt_log_path = os.path.join(log_dir, f"{args.config_id}_prompts.jsonl")

    print("Loading dataset...")
    start_time = time.time()
    raw_df = load_daily_files(data_dir)

    available_cells = get_available_cell_ids(raw_df)
    print(f"Available cells: {len(available_cells)} total")

    print(f"Preprocessing base station ({len(target_cells)} cells)...")
    tgt_x, tgt_y, tgt_in_dates, tgt_tgt_dates = preprocess_base_station(
        raw_df,
        cell_ids=target_cells,
        target_column=cfg["dataset"]["prediction_target"],
        spatial_method=spatial_method,
    )

    train_data, test_data = train_test_split(
        tgt_x, tgt_y, tgt_in_dates, tgt_tgt_dates,
        train_ratio=cfg["dataset"]["train_ratio"],
    )
    x_train, y_train, in_dates_train, tgt_dates_train = train_data
    x_test, y_test, in_dates_test, tgt_dates_test = test_data

    print(f"Train samples: {len(x_train)}, Test samples: {len(x_test)}")

    pexam_x = x_train if len(x_train) > 0 else tgt_x
    pexam_y = y_train if len(y_train) > 0 else tgt_y
    pexam_in_dates = in_dates_train if len(in_dates_train) > 0 else tgt_in_dates
    pexam_tgt_dates = tgt_dates_train if len(tgt_dates_train) > 0 else tgt_tgt_dates
    x_ex, y_ex, ex_in_date, ex_out_date = select_source_example(
        pexam_x, pexam_y, pexam_in_dates, pexam_tgt_dates
    )
    initial_pexam = build_pexam(x_ex, y_ex, ex_in_date, ex_out_date)

    llm_client = LLMClient(
        base_url=cfg["vllm"]["base_url"],
        api_key=cfg["vllm"]["api_key"],
        model_id=model_id,
        temperature=cfg["algorithm"]["temperature"],
        save_all_prompts=cfg["logging"].get("save_all_prompts", False),
        prompt_log_path=prompt_log_path,
    )

    trafficllm = TrafficLLM(
        llm_client=llm_client,
        context_window=context_window,
        max_refinement_iterations=max_iterations,
        convergence_threshold=cfg["algorithm"]["convergence_threshold"],
        max_validation_iterations=cfg["algorithm"]["max_validation_iterations"],
        max_parse_retries=cfg["algorithm"]["max_parse_retries"],
    )

    print("=== TRAINING PHASE ===")

    completed_results = {r["index"]: r for r in checkpoint.completed_samples}
    train_results = []

    for i in range(len(x_train)):
        if i in completed_results:
            train_results.append(completed_results[i])
            continue

        if i <= checkpoint.last_completed_index and not args.force_restart:
            continue

        print(f"Train sample {i + 1}/{len(x_train)}: {in_dates_train[i]} → {tgt_dates_train[i]}")

        try:
            result = trafficllm.run_refinement(
                x_t=x_train[i],
                y_t=y_train[i],
                pexam=initial_pexam,
                input_date=in_dates_train[i],
                target_date=tgt_dates_train[i],
            )
            result["index"] = i
            train_results.append(result)
            checkpoint.save_sample(result)

            mae_final = result["mae_history"][-1] if result["mae_history"] else float("nan")
            print(f"Train sample {i} done: MAE={mae_final:.4f}, iterations={result['iterations_completed']}")

        except Exception as e:
            print(f"ERROR: Train sample {i} failed: {e}")
            continue

    best_pexam = select_best_pexam(
        pexam_x, pexam_y, pexam_in_dates, pexam_tgt_dates,
        train_results, x_train, y_train, in_dates_train, tgt_dates_train,
    )

    best_train_idx = None
    best_train_mae = float("inf")
    for r in train_results:
        mae_h = r.get("mae_history", [])
        final_mae = mae_h[-1] if mae_h else float("inf")
        if final_mae < best_train_mae and r.get("final_prediction") is not None:
            best_train_mae = final_mae
            best_train_idx = r.get("index")

    print("=== TEST PHASE ===")
    test_samples = list(zip(x_test, y_test, in_dates_test, tgt_dates_test))
    test_results = trafficllm.evaluate_test(test_samples, best_pexam)

    elapsed = time.time() - start_time

    final_results = {
        "config_id": args.config_id,
        "model_id": model_id,
        "context_window": context_window,
        "target_cells": target_cells,
        "spatial_method": spatial_method,
        "num_train_samples": len(x_train),
        "num_test_samples": len(x_test),
        "training_phase": {
            "per_sample": train_results,
            "best_example_index": best_train_idx,
            "best_example_mae": best_train_mae if best_train_mae < float("inf") else None,
        },
        "test_phase": {
            "avg_mae": test_results["avg_mae"],
            "avg_mse": test_results["avg_mse"],
            "per_sample": test_results["per_sample"],
            "num_evaluated": test_results["num_evaluated"],
            "num_skipped": test_results["num_skipped"],
        },
        "metadata": {
            "timestamp": datetime.datetime.now().isoformat(),
            "total_llm_calls": llm_client.total_calls,
            "total_runtime_seconds": elapsed,
            "vllm_params": {
                "max_model_len": context_window,
                "dtype": "float16",
            },
        },
    }

    output_path = os.path.join(args.output_dir, f"{args.config_id}_results.json")
    save_results(final_results, output_path)

    checkpoint.delete()

    print(
        f"Experiment complete: config_id={args.config_id}, "
        f"avg_test_mae={test_results['avg_mae']:.4f}, "
        f"avg_test_mse={test_results['avg_mse']:.4f}, "
        f"runtime={elapsed:.1f}s, "
        f"total_llm_calls={llm_client.total_calls}"
    )


if __name__ == "__main__":
    main()
