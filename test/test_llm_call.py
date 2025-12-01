from __future__ import annotations

import argparse
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.llm.llm_client import LocalLLM
from src.utils.config import DEFAULT_MODEL_PATH, DEFAULT_VLLM_BASE_URL


SYSTEM_PROMPT = """
You are a gold medalist in a math competition. The user will provide you with a question and does not require you to answer or provide an answer directly. 
"""


USER_PROMPT = """
Let $N$ be the greatest four-digit positive integer with the property that whenever one of its digits is changed to $1$, the resulting number is divisible by $7$. Let $Q$ and $R$ be the quotient and remainder, respectively, when $N$ is divided by $1000$. Find $Q+R$.

Please carefully analyze the existing conditions for this problem and point out what the first step should be based on the existing conditions.
"""



def main() -> None:
    
    model_path = "/home/zhangdw/models/Qwen/Qwen3-8B"
    api_base = "http://0.0.0.0:8000"
    max_context_length = 32768
    max_new_tokens = 32768
    temperature = 0.0
    top_p = 1.0
    dry_run = False
    texts = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": USER_PROMPT}]
    client = LocalLLM(
        model_path=model_path,
        api_base=api_base,
        max_new_tokens=max_new_tokens,
        max_context_length=max_context_length,
        temperature=temperature,
        top_p=top_p,
        dry_run=dry_run,
    )

    print("=" * 40)
    print("问题：")
    print(texts[1]["content"])
    print("=" * 40)

    try:
        answer = client.generate(texts)
    except Exception as exc:
        print(f"[LLM 调用失败] {exc}")
        return

    print("回答：")
    print(answer)


if __name__ == "__main__":
    main()

