"""将 AIME 2024 题目按题干和答案拆分成纯文本文件."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT_DIR / "data" / "AIME_2024" / "aime_2024_problems.parquet"
OUTPUT_DIR = ROOT_DIR / "questions"


def _to_single_line(text: str) -> str:
    """压缩多余空白，避免换行影响后续拼接。"""
    return " ".join(str(text).split())


def export_questions_and_answers(
    data_path: Path = DATA_PATH, output_dir: Path = OUTPUT_DIR
) -> tuple[Path, Path]:
    """
    从 Parquet 文件导出题目与答案到两个 txt 文件。

    返回写入后的 (questions_path, answers_path)。
    """
    df = pd.read_parquet(data_path)
    required = {"ID", "Problem", "Answer"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"缺少必要列: {sorted(missing)}")

    output_dir.mkdir(parents=True, exist_ok=True)
    questions_path = output_dir / "aime2024_questions.txt"
    answers_path = output_dir / "aime2024_answers.txt"

    questions_lines = []
    answers_lines = []

    for row in df.itertuples(index=False):
        questions_lines.append(f"{row.ID}\t{_to_single_line(row.Problem)}")
        answers_lines.append(f"{row.ID}\t{_to_single_line(row.Answer)}")

    questions_path.write_text("\n".join(questions_lines), encoding="utf-8")
    answers_path.write_text("\n".join(answers_lines), encoding="utf-8")
    return questions_path, answers_path


if __name__ == "__main__":
    q_path, a_path = export_questions_and_answers()
    print(f"题目已写入: {q_path}")
    print(f"答案已写入: {a_path}")

