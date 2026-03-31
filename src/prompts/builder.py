"""Build prompt components for TrafficLLM: pexam, pinput, pques, pfeed, prefine."""

import logging
from typing import List, Tuple

import numpy as np

from .templates import (
    PEXAM_TEMPLATE,
    PFEED_ASSEMBLED_TEMPLATE,
    PFEED_MAE_TEMPLATE,
    PFEED_PERIODIC_TEMPLATE,
    PINPUT_TEMPLATE,
    PREFINE_TEMPLATE,
    PQUES_TEMPLATE,
    REFINEMENT_HISTORY_INPUT,
    REFINEMENT_HISTORY_PRED,
    VALIDATION_CORRECT_TEMPLATE,
    VALIDATION_REVIEW_TEMPLATE,
)

logger = logging.getLogger(__name__)


def format_values(values: np.ndarray, decimals: int = 4) -> str:
    """Format a numpy array as a comma-separated string with fixed decimals."""
    return ", ".join(f"{v:.{decimals}f}" for v in values)


def build_pexam(
    x_example: np.ndarray,
    y_example: np.ndarray,
    example_input_date: str,
    example_output_date: str,
) -> str:
    """Build the demonstration example prompt (pexam) from source cell data.

    Args:
        x_example: 24-element input day vector.
        y_example: 24-element output day vector (known ground truth from source).
        example_input_date: Date string for input day.
        example_output_date: Date string for output/predicted day.

    Returns:
        Formatted pexam string.
    """
    return PEXAM_TEMPLATE.format(
        input_values=format_values(x_example),
        input_date=example_input_date,
        output_values=format_values(y_example),
        output_date=example_output_date,
    )


def build_pinput(x_t: np.ndarray, input_date: str) -> str:
    """Build the input prompt (pinput) for the current prediction sample."""
    return PINPUT_TEMPLATE.format(
        input_values=format_values(x_t),
        input_date=input_date,
    )


def build_pques(input_date: str, target_date: str) -> str:
    """Build the question prompt (pques) asking for next-day prediction."""
    return PQUES_TEMPLATE.format(
        input_date=input_date,
        target_date=target_date,
    )


def build_initial_prompt(
    pexam: str,
    x_t: np.ndarray,
    input_date: str,
    target_date: str,
) -> str:
    """Concatenate pexam + pinput + pques for initial prediction (Eq. 3).

    Args:
        pexam: Demonstration example prompt.
        x_t: Current sample's input day vector.
        input_date: Current sample's input date.
        target_date: Current sample's target date.

    Returns:
        Full prompt string for initial LLM call.
    """
    pinput = build_pinput(x_t, input_date)
    pques = build_pques(input_date, target_date)
    return f"{pexam}\n\n{pinput}\n\n{pques}"


# def build_pfeed_mae_prompt(
#     y_true: np.ndarray,
#     y_pred: np.ndarray,
#     target_date: str,
# ) -> str:
#     """Build the MAE feedback query prompt."""
#     return PFEED_MAE_TEMPLATE.format(
#         target_date=target_date,
#         ground_truth=format_values(y_true),
#         predicted=format_values(y_pred),
#     )


# def build_pfeed_periodic_prompt(y_true: np.ndarray, y_pred: np.ndarray) -> str:
#     """Build the sine/cosine fitting feedback query prompt."""
#     return PFEED_PERIODIC_TEMPLATE.format(
#         ground_truth=format_values(y_true),
#         predicted=format_values(y_pred),
#     )


def build_validation_review_prompt(previous_answers: str) -> str:
    """Build the validation review prompt."""
    return VALIDATION_REVIEW_TEMPLATE.format(previous_answers=previous_answers)


def build_validation_correct_prompt() -> str:
    """Build the validation correction prompt."""
    return VALIDATION_CORRECT_TEMPLATE


def build_pfeed_assembled(
    mae_value: float,
    f_act_params: str,
    f_pred_params: str,
    method_summary: str = "in-context learning with iterative refinement",
) -> str:
    """Assemble all four feedback components into pfeed,i string.

    Args:
        mae_value: The authoritative (programmatic) MAE value.
        f_act_params: String representation of fitted ground-truth function.
        f_pred_params: String representation of fitted prediction function.
        method_summary: Description of current prediction approach.

    Returns:
        Assembled pfeed string.
    """
    return PFEED_ASSEMBLED_TEMPLATE.format(
        mae_value=f"{mae_value:.4f}",
        f_act_params=f_act_params,
        f_pred_params=f_pred_params,
        method_summary=method_summary,
    )


def build_prefine(target_date: str) -> str:
    """Build the refinement instruction prompt (prefine,i)."""
    return PREFINE_TEMPLATE.format(target_date=target_date)


def build_growing_prompt(
    x_t: np.ndarray,
    input_date: str,
    target_date: str,
    history: List[Tuple[np.ndarray, str, str]],
) -> str:
    """Build the growing refinement prompt (Equation 5).

    Accumulates ALL previous history: x[t] ⊕ ŷ₀ ⊕ pfeed,0 ⊕ prefine,0 ⊕ ...

    Args:
        x_t: Current sample's input vector.
        input_date: Input date string.
        target_date: Target date string.
        history: List of (prediction, pfeed, prefine) tuples accumulated so far.

    Returns:
        Full growing prompt string for refinement LLM call.
    """
    parts = []

    # x[t]: input traffic
    parts.append(
        REFINEMENT_HISTORY_INPUT.format(
            input_values=format_values(x_t),
            input_date=input_date,
        )
    )

    # Append each historical (ŷᵢ, pfeed,i, prefine,i)
    for pred, pfeed, prefine in history:
        parts.append(
            REFINEMENT_HISTORY_PRED.format(
                pred_values=format_values(pred),
                target_date=target_date,
            )
        )
        parts.append(pfeed)
        parts.append(prefine)

    return "\n\n".join(parts)
