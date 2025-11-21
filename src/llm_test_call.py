from __future__ import annotations

import argparse
import os
import sys
import textwrap

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.llm.llm_client import LocalLLM
from src.utils.config import DEFAULT_MODEL_PATH, DEFAULT_VLLM_BASE_URL
·


def main() -> None:
    
    model_path = "/home/zhangdw/models/DeepSeek/DeepSeek-R1-0528-Qwen3-8B"
    api_base = "http://0.0.0.0:8000"
    max_new_tokens = 10000
    temperature = 0.15
    top_p = 0.9
    dry_run = False
    question = "你是一个数学专家，请解决以下问题：矩形纸张$ABCD$的长度$AB=8，宽度$BC=6。将纸张折叠，使$A$角与$C$角完全重合。计算折痕的长度（折线）。 注意：在给出结果后必须输出<end_of_solution>"
    client = LocalLLM(
        model_path=model_path,
        api_base=api_base,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        dry_run=dry_run,
    )

    print("=" * 40)
    print("问题：")
    print(textwrap.fill(question, width=80))
    print("=" * 40)

    try:
        answer = client.generate(question, stop=["<end_of_solution>"])
    except Exception as exc:
        print(f"[LLM 调用失败] {exc}")
        return

    print("回答：")
    print(textwrap.fill(answer, width=80))


if __name__ == "__main__":
    main()

