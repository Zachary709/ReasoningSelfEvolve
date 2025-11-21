from __future__ import annotations

import logging
import re
from typing import Optional

import requests
from transformers import AutoTokenizer

from src.utils.config import DEFAULT_VLLM_BASE_URL

logger = logging.getLogger(__name__)


class LocalLLM:
    """
    Thin client for a local vLLM OpenAI-compatible server.
    """

    def __init__(
        self,
        model_path: str,
        api_base: str = DEFAULT_VLLM_BASE_URL,
        max_new_tokens: int = 4096,
        max_context_length: int = 30000,
        temperature: float = 0.15,
        top_p: float = 0.9,
        dry_run: bool = False,
        request_timeout: Optional[float] = None,
    ) -> None:
        self.model_path = model_path
        self.api_base = api_base.rstrip("/")
        self.max_new_tokens = max_new_tokens
        self.max_context_length = max_context_length
        self.temperature = temperature
        self.top_p = top_p
        self.dry_run = dry_run
        self.request_timeout = request_timeout
        self._tokenizer: Optional[AutoTokenizer] = None

    def _get_tokenizer(self) -> AutoTokenizer:
        """Lazy load tokenizer to avoid loading it if not needed."""
        if self._tokenizer is None:
            try:
                self._tokenizer = AutoTokenizer.from_pretrained(
                    self.model_path, trust_remote_code=True
                )
            except Exception as e:
                logger.warning(
                    f"Failed to load tokenizer from {self.model_path}: {e}. "
                    "Token length checking will be disabled."
                )
        return self._tokenizer

    def _count_tokens(self, text: str) -> int:
        """Count tokens in text using the model's tokenizer."""
        tokenizer = self._get_tokenizer()
        if tokenizer is None:
            # Fallback: rough estimate using character count (roughly 4 chars per token)
            return len(text) // 4
        return len(tokenizer.encode(text, add_special_tokens=False))

    def _truncate_prompt(self, prompt: str, max_tokens: int) -> str:
        """Truncate prompt to fit within max_tokens limit, keeping the beginning."""
        tokenizer = self._get_tokenizer()
        if tokenizer is None:
            # Fallback: truncate by characters (rough estimate)
            estimated_chars = max_tokens * 4
            if len(prompt) <= estimated_chars:
                return prompt
            logger.warning(
                f"Prompt too long ({len(prompt)} chars), truncating to ~{estimated_chars} chars "
                "(tokenizer not available, using rough estimate)"
            )
            truncated = prompt[:estimated_chars]
            # Add truncation note, but account for its length
            note = "\n\n[Prompt truncated due to length limit]"
            note_chars = len(note)
            if len(truncated) + note_chars > estimated_chars:
                truncated = truncated[:estimated_chars - note_chars]
            return truncated + note

        # Tokenize and truncate properly
        tokens = tokenizer.encode(prompt, add_special_tokens=False)
        if len(tokens) <= max_tokens:
            return prompt

        logger.warning(
            f"Prompt has {len(tokens)} tokens, truncating to {max_tokens} tokens "
            f"(max context length: {self.max_context_length})"
        )

        # Reserve some tokens for truncation note
        truncation_note = "\n\n[Prompt truncated due to length limit]"
        note_tokens = len(tokenizer.encode(truncation_note, add_special_tokens=False))
        available_tokens = max_tokens - note_tokens
        if available_tokens <= 0:
            # If note itself is too long, just truncate without note
            available_tokens = max_tokens
            truncation_note = ""

        # Keep the first available_tokens tokens
        truncated_tokens = tokens[:available_tokens]
        truncated_prompt = tokenizer.decode(truncated_tokens, skip_special_tokens=True)
        truncated_prompt += truncation_note
        return truncated_prompt

    def generate(
        self, 
        prompt: str, 
        stop: Optional[list[str]] = None,
        max_new_tokens_override: Optional[int] = None
    ) -> str:
        if self.dry_run:
            return f"[DRY-RUN OUTPUT] Prompt preview: {prompt[:200]}..."

        # Calculate prompt token count
        input_token_count = self._count_tokens(prompt)
        
        # Calculate maximum input tokens
        # Strategy: prioritize input tokens, then allocate remaining space for output
        # Reserve at least 100 tokens for response, but don't reserve more than necessary
        min_output_tokens = 100
        buffer_tokens = 50  # Safety buffer
        
        # Maximum input tokens: context_length - min_output - buffer
        max_input_tokens = self.max_context_length - min_output_tokens - buffer_tokens
        
        # Truncate prompt if necessary
        if input_token_count > max_input_tokens:
            original_count = input_token_count
            prompt = self._truncate_prompt(prompt, max_input_tokens)
            input_token_count = self._count_tokens(prompt)
            logger.warning(
                f"Prompt truncated from {original_count} to {input_token_count} tokens "
                f"(max context length: {self.max_context_length})"
            )

        # Use override if provided, otherwise use default
        max_new_tokens_to_use = max_new_tokens_override if max_new_tokens_override is not None else self.max_new_tokens
        
        # Calculate effective max_new_tokens based on actual prompt length
        # Use remaining space after prompt and buffer
        effective_max_new_tokens = min(
            max_new_tokens_to_use,
            self.max_context_length - input_token_count - buffer_tokens
        )
        if effective_max_new_tokens < min_output_tokens:
            effective_max_new_tokens = min_output_tokens
            logger.warning(
                f"Prompt is very long ({input_token_count} tokens), "
                f"limiting response to {effective_max_new_tokens} tokens"
            )

        url = f"{self.api_base}/v1/completions"
        payload = {
            "model": self.model_path,
            "prompt": prompt,
            "max_tokens": effective_max_new_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
        }
        
        # Add stop tokens if provided
        if stop is not None:
            payload["stop"] = stop
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
        
        # Handle empty response
        text = text.strip()
        if not text:
            logger.warning(
                f"[LLM] Received empty response from model. "
                f"Finish reason: {choices[0].get('finish_reason')}, "
                f"Usage: {data.get('usage', {})}"
            )
            # Return a placeholder message instead of raising an error
            return "[Empty response from model]"
        
        # For verification calls (when max_new_tokens_override is set), truncate output after verdict
        # Strategy: Find the first occurrence of \boxed{0} or \boxed{1} and keep everything
        # up to and including the verdict, then truncate everything after it
        # This ensures the verdict is always preserved even if model continues generating
        if max_new_tokens_override is not None:
            # Check if output contains \boxed{0} or \boxed{1}
            verdict_match = re.search(r"\\boxed\{([01])\}", text)
            if verdict_match:
                # Find the start and end positions of the verdict
                verdict_start = verdict_match.start()
                verdict_end = verdict_match.end()
                
                # Find the start of the line containing the verdict
                line_start = text.rfind("\n", 0, verdict_start)
                if line_start == -1:
                    line_start = 0
                else:
                    line_start += 1  # Include the newline character
                
                # Find the end of the line containing the verdict
                line_end = text.find("\n", verdict_end)
                if line_end != -1:
                    # Keep everything up to and including the line with verdict
                    text = text[:line_end].strip()
                else:
                    # No newline after verdict, keep everything up to and including verdict
                    text = text[:verdict_end].strip()
            # If no verdict found, keep the full output (don't truncate)
        
        return text

