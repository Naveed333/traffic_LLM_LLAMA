"""Token counting utilities for context window management."""

import logging

logger = logging.getLogger(__name__)

# Rough Llama tokenizer heuristic: ~4 characters per token
CHARS_PER_TOKEN = 4


def estimate_token_count(text: str) -> int:
    """Estimate token count using character-based heuristic.

    Uses the rough approximation of 4 characters per token,
    which is reasonable for Llama-family models.

    Args:
        text: Input text string.

    Returns:
        Estimated token count.
    """
    return max(1, len(text) // CHARS_PER_TOKEN)


def fits_in_context(
    prompt: str,
    context_window: int,
    safety_margin: float = 0.90,
    max_new_tokens: int = 1024,
) -> bool:
    """Check if a prompt fits within the context window.

    Args:
        prompt: The prompt text to check.
        context_window: Maximum context window size in tokens.
        safety_margin: Use at most this fraction of the context window.
        max_new_tokens: Reserve this many tokens for the response.

    Returns:
        True if prompt fits, False if it would exceed the limit.
    """
    estimated = estimate_token_count(prompt)
    effective_limit = int(context_window * safety_margin) - max_new_tokens

    if estimated > effective_limit:
        logger.debug(
            f"Token check: estimated {estimated} tokens > limit {effective_limit} "
            f"(context_window={context_window}, margin={safety_margin})."
        )
        return False
    return True


def get_token_budget(
    context_window: int,
    safety_margin: float = 0.90,
    max_new_tokens: int = 1024,
) -> int:
    """Calculate available token budget for prompts.

    Args:
        context_window: Maximum context window size.
        safety_margin: Safety factor (use at most this fraction).
        max_new_tokens: Tokens to reserve for response.

    Returns:
        Token budget available for the prompt.
    """
    return int(context_window * safety_margin) - max_new_tokens
