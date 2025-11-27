"""
Self-evolving solver pipeline for AIME 2024 style problems.

Usage example:

    python -m src.self_evolving_solver --dry-run
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.utils.config import (
    DEFAULT_MODEL_PATH,
    DEFAULT_VLLM_BASE_URL,
    parse_run_config,
)
from src.utils.data_loader import ProblemRecord, load_problem
from src.utils.logging_utils import configure_logging
from src.llm.llm_client import LocalLLM
from src.prompts.prompts import PromptBuilder
from src.solver.solver_engine import SelfEvolvingSolver


def _resolve_log_path(config: Dict[str, object]) -> str | None:
    raw_path = config.get("log_path") or config.get("output")
    if not raw_path:
        return None
    log_path = Path(raw_path)
    if log_path.suffix != ".log":
        log_path = log_path.with_suffix(".log")
    return str(log_path)


def build_solver(config: Dict[str, object]) -> Tuple[SelfEvolvingSolver, ProblemRecord, int]:
    questions_dir_raw = config.get("questions_dir")
    if questions_dir_raw is None:
        questions_dir = PROJECT_ROOT / "questions"
    else:
        questions_dir = Path(questions_dir_raw)
    questions_file = str(config.get("questions_file", "aime2024_questions.txt"))
    model_path = str(config.get("model", DEFAULT_MODEL_PATH))
    api_base = str(config.get("api_base", DEFAULT_VLLM_BASE_URL))
    problem_id = config.get("problem_id")
    rounds = int(config.get("rounds", 2))
    log_path_str = _resolve_log_path(config)
    dry_run = bool(config.get("dry_run", False))
    max_new_tokens = int(config.get("max_new_tokens", 4096))
    max_context_length = int(config.get("max_context_length", 30000))
    temperature = float(config.get("temperature", 0.15))
    top_p = float(config.get("top_p", 0.9))
    verification_max_new_tokens = config.get("verification_max_new_tokens")
    max_report_tokens = int(config.get("max_report_tokens", 10000))
    if verification_max_new_tokens is not None:
        verification_max_new_tokens = int(verification_max_new_tokens)

    logger = configure_logging(log_path_str)

    record = load_problem(questions_dir, problem_id, questions_file)
    llm = LocalLLM(
        model_path=model_path,
        api_base=api_base,
        max_new_tokens=max_new_tokens,
        max_context_length=max_context_length,
        temperature=temperature,
        top_p=top_p,
        dry_run=dry_run,
    )
    solver = SelfEvolvingSolver(
        llm=llm, 
        prompt_builder=PromptBuilder(), 
        logger=logger,
        max_new_tokens=max_new_tokens,
        verification_max_new_tokens=verification_max_new_tokens,
        max_report_tokens=max_report_tokens
    )
    return solver, record, rounds


def main() -> None:
    config = parse_run_config()
    solver, record, rounds = build_solver(config)
    solver.solve(record, rounds=rounds)
    solver.logger.info("Solving pipeline completed.")


if __name__ == "__main__":
    main()

