"""Step 1: Initial prediction via pexam ⊕ pinput ⊕ pques (Equation 3)."""

from typing import Optional

import numpy as np

from ..prompts.builder import build_initial_prompt
from ..prompts.parser import parse_prediction, validate_prediction_format
from ..prompts.templates import RETRY_CLARIFICATION_TEMPLATE


class Predictor:
    """Handles initial traffic prediction (Algorithm 1, Step 1)."""

    def __init__(
        self,
        llm_client,
        max_parse_retries: int = 3,
        max_tokens: int = 512,
    ) -> None:
        self.llm = llm_client
        self.max_parse_retries = max_parse_retries
        self.max_tokens = max_tokens

    def predict(
        self,
        pexam: str,
        x_t: np.ndarray,
        input_date: str,
        target_date: str,
    ) -> Optional[np.ndarray]:
        prompt = build_initial_prompt(pexam, x_t, input_date, target_date)

        for attempt in range(self.max_parse_retries):
            response = self.llm.generate(prompt, max_tokens=self.max_tokens)

            if self.llm.is_context_exceeded(response):
                print("WARNING: Context exceeded during initial prediction.")
                return None

            if not response:
                if attempt < self.max_parse_retries - 1:
                    prompt = prompt + "\n\n" + RETRY_CLARIFICATION_TEMPLATE
                continue

            values = parse_prediction(response)

            if values is not None and validate_prediction_format(values):
                return np.array(values, dtype=float)

            print(f"WARNING: Failed to parse prediction (attempt {attempt + 1}/{self.max_parse_retries}).")

            if attempt < self.max_parse_retries - 1:
                prompt = prompt + "\n\n" + RETRY_CLARIFICATION_TEMPLATE

        print(f"ERROR: Failed to get valid prediction after {self.max_parse_retries} attempts ({input_date} → {target_date}).")
        return None
