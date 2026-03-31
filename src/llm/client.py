"""HTTP client for local vLLM server (no OpenAI SDK dependency)."""

import json
import time
from typing import Optional

import requests

_CONTEXT_EXCEEDED_SENTINEL = "__CONTEXT_LENGTH_EXCEEDED__"


class LLMClient:
    """Direct HTTP wrapper around a local vLLM /v1/chat/completions endpoint."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model_id: str,
        temperature: float = 0.0,
        save_all_prompts: bool = False,
        prompt_log_path: Optional[str] = None,
    ) -> None:
        self.model_id = model_id
        self.temperature = temperature
        self.save_all_prompts = save_all_prompts
        self.prompt_log_path = prompt_log_path
        self._call_count = 0

        self.endpoint = base_url.rstrip("/") + "/chat/completions"

        self.session = requests.Session()
        self.session.headers.update({
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        })

        print(f"LLMClient initialized: model={model_id}, endpoint={self.endpoint}")

    def generate(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: int = 1024,
        max_retries: int = 3,
        timeout: float = 120.0,
    ) -> str:
        temp = temperature if temperature is not None else self.temperature
        self._call_count += 1
        call_id = self._call_count

        payload = {
            "model": self.model_id,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temp,
            "max_tokens": max_tokens,
        }

        last_error = None
        for attempt in range(max_retries):
            try:
                resp = self.session.post(self.endpoint, json=payload, timeout=timeout)

                if resp.status_code == 400:
                    body = resp.json().get("error", {})
                    msg = str(body.get("message", "")).lower()
                    if any(k in msg for k in ("context", "length", "token")):
                        return _CONTEXT_EXCEEDED_SENTINEL
                    return ""

                if resp.status_code != 200:
                    raise requests.HTTPError(f"HTTP {resp.status_code}: {resp.text[:200]}")

                data = resp.json()
                result = data["choices"][0]["message"]["content"] or ""

                if self.save_all_prompts:
                    self._log_prompt_response(call_id, prompt, result)

                return result

            except requests.Timeout as e:
                last_error = e
                wait = 2 ** attempt
                print(f"WARNING: LLM call #{call_id} timed out (attempt {attempt + 1}). Retrying in {wait}s.")
                time.sleep(wait)

            except requests.ConnectionError as e:
                last_error = e
                wait = 2 ** attempt
                print(f"WARNING: LLM call #{call_id} connection error (attempt {attempt + 1}). Retrying in {wait}s.")
                time.sleep(wait)

            except requests.HTTPError as e:
                last_error = e
                wait = 2 ** attempt
                print(f"WARNING: LLM call #{call_id} HTTP error (attempt {attempt + 1}). Retrying in {wait}s.")
                time.sleep(wait)

            except Exception as e:
                last_error = e
                print(f"ERROR: LLM call #{call_id} unexpected error: {e}")
                break

        print(f"ERROR: LLM call #{call_id} failed after {max_retries} attempts. Last error: {last_error}")
        return ""

    def is_context_exceeded(self, response: str) -> bool:
        return response == _CONTEXT_EXCEEDED_SENTINEL

    @property
    def total_calls(self) -> int:
        return self._call_count

    def _log_prompt_response(self, call_id: int, prompt: str, response: str) -> None:
        if not self.prompt_log_path:
            return
        import os
        entry = {
            "call_id": call_id,
            "timestamp": time.time(),
            "prompt": prompt,
            "response": response,
        }
        os.makedirs(os.path.dirname(self.prompt_log_path), exist_ok=True)
        with open(self.prompt_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
