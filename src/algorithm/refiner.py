"""Step 2b: Prediction refinement using growing prompt (Equation 5)."""

from typing import List, Optional, Tuple

import numpy as np

from ..llm.token_counter import fits_in_context
from ..prompts.builder import build_growing_prompt, build_prefine
from ..prompts.parser import parse_prediction, validate_prediction_format
from ..prompts.templates import RETRY_CLARIFICATION_TEMPLATE


class Refiner:
    """Handles prediction refinement using the growing prompt (Algorithm 1, Step 2b)."""

    def __init__(
        self,
        llm_client,
        context_window: int,
        max_parse_retries: int = 3,
        max_tokens: int = 512,
    ) -> None:
        self.llm = llm_client
        self.context_window = context_window
        self.max_parse_retries = max_parse_retries
        self.max_tokens = max_tokens

    def build_refinement_instruction(self, target_date: str) -> str:
        return build_prefine(target_date)

    def check_fits_context(
        self,
        x_t: np.ndarray,
        input_date: str,
        target_date: str,
        history: List[Tuple[np.ndarray, str, str]],
    ) -> bool:
        prompt = build_growing_prompt(x_t, input_date, target_date, history)
        return fits_in_context(prompt, self.context_window, max_new_tokens=self.max_tokens)

    def refine(
        self,
        x_t: np.ndarray,
        input_date: str,
        target_date: str,
        history: List[Tuple[np.ndarray, str, str]],
    ) -> Optional[np.ndarray]:
        prompt = build_growing_prompt(x_t, input_date, target_date, history)

        for attempt in range(self.max_parse_retries):
            response = self.llm.generate(prompt, max_tokens=self.max_tokens)

            if self.llm.is_context_exceeded(response):
                print("WARNING: Context exceeded during refinement.")
                return None

            if not response:
                if attempt < self.max_parse_retries - 1:
                    prompt = prompt + "\n\n" + RETRY_CLARIFICATION_TEMPLATE
                continue

            values = parse_prediction(response)
            if values is not None and validate_prediction_format(values):
                return np.array(values, dtype=float)

            print(f"WARNING: Failed to parse refinement (attempt {attempt + 1}).")

            if attempt < self.max_parse_retries - 1:
                prompt = prompt + "\n\n" + RETRY_CLARIFICATION_TEMPLATE

        print(f"ERROR: Failed to get valid refinement after {self.max_parse_retries} attempts ({input_date} → {target_date}).")
        return None
