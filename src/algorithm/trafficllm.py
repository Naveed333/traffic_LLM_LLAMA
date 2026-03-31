"""Main orchestrator for TrafficLLM: Algorithm 1 full loop."""

from typing import Dict, List, Tuple

import numpy as np

from ..evaluation.metrics import compute_mae, compute_mse, mae_converged
from .feedback import FeedbackGenerator
from .predictor import Predictor
from .refiner import Refiner


class TrafficLLM:
    """Orchestrates the full TrafficLLM Algorithm 1 loop."""

    def __init__(
        self,
        llm_client,
        context_window: int,
        max_refinement_iterations: int = 3,
        convergence_threshold: float = 0.001,
        max_validation_iterations: int = 3,
        max_parse_retries: int = 3,
    ) -> None:
        self.context_window = context_window
        self.max_iterations = max_refinement_iterations
        self.convergence_threshold = convergence_threshold

        self.predictor = Predictor(llm_client=llm_client, max_parse_retries=max_parse_retries)
        self.feedback_gen = FeedbackGenerator(
            llm_client=llm_client,
            max_validation_iterations=max_validation_iterations,
            convergence_threshold=convergence_threshold,
        )
        self.refiner = Refiner(
            llm_client=llm_client,
            context_window=context_window,
            max_parse_retries=max_parse_retries,
        )

    def run_refinement(
        self,
        x_t: np.ndarray,
        y_t: np.ndarray,
        pexam: str,
        input_date: str,
        target_date: str,
    ) -> Dict:
        result = {
            "input_date": input_date,
            "target_date": target_date,
            "final_prediction": None,
            "iterations_completed": 0,
            "mae_history": [],
            "all_predictions": [],
            "context_window_exceeded": False,
        }

        y_hat_0 = self.predictor.predict(pexam, x_t, input_date, target_date)

        if y_hat_0 is None:
            print(f"ERROR: Initial prediction failed for {input_date} → {target_date}. Skipping.")
            return result

        initial_mae = compute_mae(y_t, y_hat_0)
        result["all_predictions"].append(y_hat_0.tolist())
        result["mae_history"].append(initial_mae)
        result["final_prediction"] = y_hat_0.tolist()

        print(f"  Initial MAE: {initial_mae:.4f}")

        history: List[Tuple[np.ndarray, str, str]] = []
        current_pred = y_hat_0
        previous_mae = initial_mae

        for i in range(self.max_iterations):
            pfeed_i = self.feedback_gen.generate(y_t, current_pred, input_date, target_date)
            prefine_i = self.refiner.build_refinement_instruction(target_date)

            candidate_history = history + [(current_pred, pfeed_i, prefine_i)]

            if not self.refiner.check_fits_context(x_t, input_date, target_date, candidate_history):
                print(f"  Context window exceeded at iteration {i + 1}. Stopping early.")
                result["context_window_exceeded"] = True
                break

            history = candidate_history
            new_pred = self.refiner.refine(x_t, input_date, target_date, history)

            if new_pred is None:
                print(f"  Refinement failed at iteration {i + 1}. Stopping.")
                break

            current_mae = compute_mae(y_t, new_pred)
            result["all_predictions"].append(new_pred.tolist())
            result["mae_history"].append(current_mae)
            result["iterations_completed"] = i + 1
            result["final_prediction"] = new_pred.tolist()

            print(f"  Iteration {i + 1}: MAE={current_mae:.4f} (was {previous_mae:.4f})")

            if mae_converged(current_mae, previous_mae, self.convergence_threshold):
                print(f"  Converged at iteration {i + 1}.")
                current_pred = new_pred
                previous_mae = current_mae
                break

            previous_mae = current_mae
            current_pred = new_pred

        return result

    def evaluate_test(
        self,
        test_samples: List[Tuple[np.ndarray, np.ndarray, str, str]],
        best_pexam: str,
    ) -> Dict:
        from ..evaluation.evaluator import Evaluator
        evaluator = Evaluator(predictor=self.predictor)
        return evaluator.evaluate(test_samples, best_pexam)
