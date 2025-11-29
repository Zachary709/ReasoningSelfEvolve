from __future__ import annotations

import argparse
import yaml
from pathlib import Path
from typing import Any, Dict

DEFAULT_CONFIG_PATH = Path("/home/zhangdw/ReasoningSelfEvolve/config/solver_config.yaml")
DEFAULT_DATASET_PATH = Path(
    "/home/zhangdw/ReasoningSelfEvolve/questions/aime2024_questions.txt"
)
DEFAULT_MODEL_PATH = "/home/zhangdw/models/Qwen/Qwen3-8B"
DEFAULT_VLLM_BASE_URL = "http://0.0.0.0:8000"


def load_config(config_path: Path) -> Dict[str, Any]:
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found at {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_run_config() -> Dict[str, Any]:
    parser = argparse.ArgumentParser(description="Self-evolving solver runner (config-based).")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help=f"Path to YAML config file (default: {DEFAULT_CONFIG_PATH}).",
    )
    args = parser.parse_args()
    return load_config(args.config)

