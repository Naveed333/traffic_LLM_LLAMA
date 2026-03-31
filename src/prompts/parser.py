"""Parse LLM responses: extract 24 traffic values, MAE, and sine/cosine parameters."""

import logging
import math
import re
from typing import List, Optional

logger = logging.getLogger(__name__)


def parse_prediction(response: str) -> Optional[List[float]]:
    """Extract exactly 24 comma-separated float values from LLM response.

    Handles formats:
    - [1.0, 2.0, 3.0, ...]  (bracketed list)
    - 1.0, 2.0, 3.0, ...    (plain comma-separated)
    - numbers on separate lines
    - mixed whitespace/newline delimiters

    Args:
        response: Raw LLM response string.

    Returns:
        List of exactly 24 floats, or None if parsing fails.
    """
    if not response or not response.strip():
        logger.warning("Empty response received.")
        return None

    number_pattern = r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"

    # Strategy 1: Find content inside square brackets
    bracket_match = re.search(r"\[([^\[\]]+)\]", response, re.DOTALL)
    if bracket_match:
        candidate = bracket_match.group(1)
        result = _extract_floats(candidate)
        if result and len(result) == 24:
            return result

    # Strategy 2: Find all numbers and check count
    all_numbers = re.findall(number_pattern, response)
    floats = []
    for num_str in all_numbers:
        try:
            floats.append(float(num_str))
        except ValueError:
            pass

    if len(floats) == 24:
        return floats
    elif len(floats) > 24:
        logger.debug(f"Found {len(floats)} numbers, taking last 24.")
        return floats[-24:]

    # Strategy 3: Look for newline-separated numbers
    lines = [line.strip() for line in response.split("\n") if line.strip()]
    line_floats = []
    for line in lines:
        nums = re.findall(number_pattern, line)
        for n in nums:
            try:
                line_floats.append(float(n))
            except ValueError:
                pass

    if len(line_floats) == 24:
        return line_floats

    logger.warning(
        f"Could not extract exactly 24 values from response. "
        f"Found {len(floats)} numbers. Response (truncated): {response[:200]}"
    )
    return None


def _extract_floats(text: str) -> Optional[List[float]]:
    """Helper: extract all floats from a text string."""
    number_pattern = r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"
    matches = re.findall(number_pattern, text)
    try:
        return [float(m) for m in matches]
    except ValueError:
        return None


def parse_mae(response: str) -> Optional[float]:
    """Extract MAE value from LLM response.

    Handles patterns like:
    - "The MAE is: 12.34"
    - "MAE = 12.34"
    - "MAE is 12.34"
    - "mean absolute error: 12.34"

    Args:
        response: Raw LLM response string.

    Returns:
        Float MAE value or None if not found.
    """
    if not response:
        return None

    patterns = [
        r"MAE\s+is[:\s]+([+-]?\d*\.?\d+)",
        r"MAE\s*=\s*([+-]?\d*\.?\d+)",
        r"The MAE is[:\s]+([+-]?\d*\.?\d+)",
        r"mean absolute error[:\s]+([+-]?\d*\.?\d+)",
        r"MAE[:\s]+([+-]?\d*\.?\d+)",
    ]

    for pattern in patterns:
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                continue

    logger.debug(f"Could not parse MAE from response: {response[:200]}")
    return None


def parse_sine_cosine(response: str) -> Optional[str]:
    """Extract fitted function parameters from LLM response.

    Returns the raw string containing function descriptions for inclusion
    in the feedback prompt.

    Args:
        response: Raw LLM response string.

    Returns:
        String with function parameters, or a fallback descriptor.
    """
    if not response:
        return "f(t) = unknown"

    patterns = [
        r"f_act\s*=\s*([^\n]+)",
        r"f_pred\s*=\s*([^\n]+)",
        r"(?:sine|cosine|sin|cos)\s+function[:\s]+([^\n]+)",
        r"fitted function[:\s]+([^\n]+)",
    ]

    found_parts = []
    for pattern in patterns:
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            found_parts.append(match.group(0).strip())

    if found_parts:
        return " | ".join(found_parts)

    return response.strip()[:200] if response.strip() else "f(t) = unknown"


def validate_prediction_format(prediction: List[float], expected_count: int = 24) -> bool:
    """Check that prediction has the expected number of finite values.

    Args:
        prediction: List of predicted values.
        expected_count: Expected number of values (default: 24).

    Returns:
        True if format is valid, False otherwise.
    """
    if not prediction:
        return False
    if len(prediction) != expected_count:
        return False
    if any(math.isnan(v) or math.isinf(v) for v in prediction):
        return False
    return True
