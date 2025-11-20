from __future__ import annotations

import logging
import re
from typing import Any, Dict, Optional

from src.data_loader import ProblemRecord
from src.llm_client import LocalLLM
from src.prompts import PromptBuilder


class SelfEvolvingSolver:
    def __init__(
        self,
        llm: LocalLLM,
        prompt_builder: PromptBuilder,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.llm = llm
        self.prompts = prompt_builder
        self.logger = logger or logging.getLogger(__name__)

    def solve(self, record: ProblemRecord, rounds: int = 2) -> Dict[str, Any]:
        history = []
        self._log(self._divider("Problem"))
        self._log(f"Solving problem {record.problem_id}")
        self._log("Problem statement:\n%s", record.prompt)

        solution_prompt = self.prompts.solution(record.prompt)
        self._log(self._divider("Initial Solution"))
        self._log("Generating initial solution...")
        solution_text = self.llm.generate(solution_prompt)
        solution_body = self._extract_section(solution_text, "solution") or solution_text
        self._log("Initial solution LLM output:\n%s", solution_text)
        history.append(
            {
                "role": "solution",
                "prompt": solution_prompt,
                "response": solution_text,
                "solution_body": solution_body,
            }
        )

        self._log(self._divider("Initial Verification"))
        self._log("Running initial verification...")
        verification_prompt = self.prompts.verification(record.prompt, solution_body)
        verification_text = self.llm.generate(verification_prompt)
        self._log("Initial verification LLM output:\n%s", verification_text)
        history.append(
            {"role": "verification", "prompt": verification_prompt, "response": verification_text}
        )

        verdict = self._extract_verdict(verification_text)
        self._log(f"Initial verification verdict: {verdict}")

        for round_idx in range(2, rounds + 1):
            if verdict == 1:
                break
            epoch_label = f"epoch {round_idx}/{rounds}"
            self._log(self._divider(f"Refinement {epoch_label}"))
            self._log(f"Entering refinement {epoch_label}...")
            refinement_prompt = self.prompts.refinement(
                record.prompt, solution_body, verification_text
            )
            solution_text = self.llm.generate(refinement_prompt)
            solution_body = self._extract_section(solution_text, "solution") or solution_text
            self._log(f"Solution LLM output ({epoch_label}):\n%s", solution_text)
            history.append(
                {
                    "role": f"solution_round_{round_idx}",
                    "prompt": refinement_prompt,
                    "response": solution_text,
                    "solution_body": solution_body,
                }
            )

            self._log(self._divider(f"Verification {epoch_label}"))
            self._log(f"Running verification ({epoch_label})...")
            verification_prompt = self.prompts.verification(record.prompt, solution_body)
            verification_text = self.llm.generate(verification_prompt)
            self._log(f"Verification LLM output ({epoch_label}):\n%s", verification_text)
            history.append(
                {
                    "role": f"verification_round_{round_idx}",
                    "prompt": verification_prompt,
                    "response": verification_text,
                }
            )
            verdict = self._extract_verdict(verification_text)
            self._log(f"Verification verdict ({epoch_label}): {verdict}")

        final_answer = self._extract_boxed_answer(solution_body)
        self._log(self._divider("Final Verdict"))
        self._log(f"Final verdict: {verdict}, Final Answer: {final_answer}")
        return {
            "problem_id": record.problem_id,
            "problem": record.prompt,
            "final_solution": solution_text,
            "final_solution_body": solution_body,
            "final_verification": verification_text,
            "verdict": verdict,
            "boxed_answer": final_answer,
            "history": history,
        }

    @staticmethod
    def _extract_verdict(text: str) -> Optional[int]:
        match = re.search(r"\\boxed\{([01])\}", text)
        if match:
            return int(match.group(1))
        return None

    @staticmethod
    def _extract_boxed_answer(text: str) -> Optional[str]:
        match = re.search(r"\\boxed\{([^}]+)\}", text)
        if match:
            return match.group(1).strip()
        return None

    @staticmethod
    def _extract_section(text: str, tag: str) -> Optional[str]:
        pattern = rf"<{tag}>(.*?)</{tag}>"
        match = re.search(pattern, text, flags=re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return None

    def _log(self, message: str, *args: Any) -> None:
        if self.logger:
            self.logger.info(message, *args)

    @staticmethod
    def _divider(label: str) -> str:
        line = "~" * 30
        return f"{line} {label} {line}"

