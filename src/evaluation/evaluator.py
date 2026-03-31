"""Run test evaluation (single-pass, no refinement) for TrafficLLM."""

from typing import Dict, List, Tuple

import numpy as np

from .metrics import compute_mae, compute_mse


class Evaluator:
    """Evaluates TrafficLLM on test samples using single-pass prediction (no refinement)."""

    def __init__(self, predictor) -> None:
        self.predictor = predictor

    def evaluate(
        self,
        test_samples: List[Tuple[np.ndarray, np.ndarray, str, str]],
        best_pexam: str,
    ) -> Dict:
        results = []
        skipped = 0

        for idx, (x_t, y_t, input_date, target_date) in enumerate(test_samples):
            print(f"Test sample {idx + 1}/{len(test_samples)}: {input_date} → {target_date}")

            try:
                y_hat = self.predictor.predict(best_pexam, x_t, input_date, target_date)

                if y_hat is None:
                    skipped += 1
                    continue

                mae = compute_mae(y_t, y_hat)
                mse = compute_mse(y_t, y_hat)

                results.append({
                    "index": idx,
                    "input_date": input_date,
                    "target_date": target_date,
                    "mae": mae,
                    "mse": mse,
                    "prediction": y_hat.tolist() if isinstance(y_hat, np.ndarray) else y_hat,
                    "ground_truth": y_t.tolist() if isinstance(y_t, np.ndarray) else list(y_t),
                })

                print(f"  MAE={mae:.4f}, MSE={mse:.4f}")

            except Exception as e:
                print(f"ERROR: Test sample {idx} failed: {e}")
                skipped += 1
                continue

        if not results:
            print("ERROR: No test samples evaluated successfully.")
            return {
                "avg_mae": float("nan"),
                "avg_mse": float("nan"),
                "per_sample": [],
                "num_evaluated": 0,
                "num_skipped": skipped,
            }

        avg_mae = float(np.mean([r["mae"] for r in results]))
        avg_mse = float(np.mean([r["mse"] for r in results]))

        print(f"Test complete: avg_mae={avg_mae:.4f}, avg_mse={avg_mse:.4f}, evaluated={len(results)}, skipped={skipped}")

        return {
            "avg_mae": avg_mae,
            "avg_mse": avg_mse,
            "per_sample": results,
            "num_evaluated": len(results),
            "num_skipped": skipped,
        }
