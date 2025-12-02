from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional, Sequence, Union

from openai import OpenAI
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
        api_key: Optional[str] = None,
        max_new_tokens: int = 4096,
        max_context_length: int = 30000,
        temperature: float = 0.15,
        top_p: float = 0.9,
        dry_run: bool = False,
        request_timeout: Optional[float] = None,
        enable_thinking: bool = True,
    ) -> None:
        self.model_path = model_path
        self.api_base = api_base.rstrip("/")
        self.api_key = api_key or "EMPTY"
        self.max_new_tokens = max_new_tokens
        self.max_context_length = max_context_length
        self.temperature = temperature
        self.top_p = top_p
        self.dry_run = dry_run
        self.request_timeout = request_timeout
        self._tokenizer: Optional[AutoTokenizer] = None
        self._client = OpenAI(
            base_url=f"{self.api_base}/v1",
            api_key=self.api_key,
        )
        self.enable_thinking = enable_thinking

    def _get_tokenizer(self) -> AutoTokenizer:
        """Lazy load tokenizer to avoid loading it if not needed."""
        if self._tokenizer is None:
            try:
                self._tokenizer = AutoTokenizer.from_pretrained(
                    self.model_path, 
                    torch_dtype="auto",
                    device_map="auto",
                    trust_remote_code=True
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

    def _messages_to_text(self, messages: Sequence[Dict[str, str]]) -> str:
        """Flatten chat messages into text for approximate token counting."""
        tokenizer = self._get_tokenizer()
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self.enable_thinking # Switches between thinking and non-thinking modes. Default is True.
        )
        return text

    def generate(
        self,
        prompt: Union[str, Sequence[Dict[str, str]]],
        stop: Optional[list[str]] = None,
        max_new_tokens_override: Optional[int] = None,
    ) -> str:
        if self.dry_run:
            preview_text = (
                prompt if isinstance(prompt, str) else self._messages_to_text(prompt)
            )
            return f"[DRY-RUN OUTPUT] Prompt preview: {preview_text[:200]}..."

        if isinstance(prompt, str):
            messages: List[Dict[str, str]] = [{"role": "system", "content": "You are a math problem solver. You are given a math problem and you need to solve it."}, {"role": "user", "content": prompt}]
        else:
            messages = list(prompt)
        prompt = self._messages_to_text(messages)

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

        

        client = (
            self._client.with_options(timeout=self.request_timeout)
            if self.request_timeout is not None
            else self._client
        )
        try:
            response = client.chat.completions.create(
                model=self.model_path,
                messages=messages,
                max_tokens=effective_max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                # extra_body={"enable_thinking": False},
                stop=stop,
            )
        except Exception as exc:
            raise RuntimeError(f"[LLM] Failed to call OpenAI client: {exc}") from exc

        choices = response.choices
        if not choices:
            raise RuntimeError("[LLM] OpenAI client returned no choices.")

        text = getattr(choices[0], "text", None) or getattr(
            getattr(choices[0], "message", None), "content", None
        )
        if text is None:
            raise RuntimeError("[LLM] Unable to parse generation result.")
        
        # Handle empty response
        text = text.strip()
        if not text:
            logger.warning(
                f"[LLM] Received empty response from model. "
                f"Finish reason: {getattr(choices[0], 'finish_reason', None)}, "
                f"Usage: {getattr(response, 'usage', {})}"
            )
            # Return a placeholder message instead of raising an error
            return "[Empty response from model]"
        
        # output usage
        usage = response.usage
        logger.info(f"[LLM] Usage: {usage}")
        # print(f"[LLM] Usage: {usage}")

        return text

