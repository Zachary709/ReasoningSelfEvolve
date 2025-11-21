from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

DEFAULT_LOG_PATH = Path("/home/zhangdw/ReasoningSelfEvolve/outputs/self_evolving_solver.log")


def configure_logging(log_path_str: Optional[str]) -> logging.Logger:
    logger = logging.getLogger("self_evolving_solver")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    log_path = Path(log_path_str) if log_path_str else DEFAULT_LOG_PATH
    if log_path.suffix != ".log":
        log_path = log_path.with_suffix(".log")
    log_path.parent.mkdir(parents=True, exist_ok=True)

    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.info(f"Logging to {log_path}")

    return logger

