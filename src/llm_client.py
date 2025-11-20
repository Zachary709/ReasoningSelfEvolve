from __future__ import annotations

from typing import Optional

import requests

from src.config import DEFAULT_VLLM_BASE_URL


class LocalLLM:
    """
    Thin client for a local vLLM OpenAI-compatible server.
    """

    def __init__(
        self,
        model_path: str,
        api_base: str = DEFAULT_VLLM_BASE_URL,
        max_new_tokens: int = 4096,
        temperature: float = 0.15,
        top_p: float = 0.9,
        dry_run: bool = False,
        request_timeout: Optional[float] = None,
    ) -> None:
        self.model_path = model_path
        self.api_base = api_base.rstrip("/")
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.dry_run = dry_run
        self.request_timeout = request_timeout

    def generate(self, prompt: str) -> str:
        if self.dry_run:
            return f"[DRY-RUN OUTPUT] Prompt preview: {prompt[:200]}..."

        url = f"{self.api_base}/v1/completions"
        payload = {
            "model": self.model_path,
            "prompt": prompt,
            "max_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
        }
        try:
            response = requests.post(url, json=payload, timeout=self.request_timeout)
            response.raise_for_status()
            data = response.json()
        except requests.RequestException as exc:
            raise RuntimeError(f"[LLM] Failed to call vLLM endpoint: {exc}") from exc

        choices = data.get("choices")
        if not choices:
            raise RuntimeError(f"[LLM] vLLM endpoint returned no choices: {data}")

        text = choices[0].get("text") or choices[0].get("message", {}).get("content")
        if text is None:
            raise RuntimeError(f"[LLM] Unable to parse generation result: {data}")
        return text.strip()

