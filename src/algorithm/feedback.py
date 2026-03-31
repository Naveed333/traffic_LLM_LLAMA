"""Step 2a: Feedback generation for TrafficLLM (fully programmatic — no LLM calls)."""

from typing import Tuple

import numpy as np
from scipy.optimize import curve_fit

from ..evaluation.metrics import compute_mae
from ..prompts.builder import build_pfeed_assembled


def _fit_sinusoidal(values: np.ndarray) -> str:
    t = np.arange(24, dtype=float)

    try:
        def sine_func(t, a, w, p, c):
            return a * np.sin(w * t + p) + c

        c0 = float(np.mean(values))
        a0 = float((np.max(values) - np.min(values)) / 2)
        popt, _ = curve_fit(
            sine_func, t, values,
            p0=[a0, 2 * np.pi / 24, 0, c0],
            maxfev=5000,
        )
        a, w, p, c = popt
        return f"a={a:.3f}*sin({w:.4f}*t+{p:.4f}) + {c:.3f}"
    except Exception:
        return f"mean={float(np.mean(values)):.3f}, std={float(np.std(values)):.3f}"


class FeedbackGenerator:
    """Generates feedback for the refinement loop (Algorithm 1, Step 2a)."""

    def __init__(
        self,
        llm_client=None,
        max_validation_iterations: int = 3,
        convergence_threshold: float = 0.001,
        max_tokens: int = 512,
    ) -> None:
        self.convergence_threshold = convergence_threshold

    def generate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        input_date: str,
        target_date: str,
    ) -> str:
        mae_value = compute_mae(y_true, y_pred)
        f_act_params = _fit_sinusoidal(y_true)
        f_pred_params = _fit_sinusoidal(y_pred)

        method_summary = (
            "in-context learning with iterative self-refinement based on historical traffic patterns"
        )

        return build_pfeed_assembled(
            mae_value=mae_value,
            f_act_params=f_act_params,
            f_pred_params=f_pred_params,
            method_summary=method_summary,
        )
