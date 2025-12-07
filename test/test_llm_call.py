from __future__ import annotations

import argparse
import os
import sys
from collections import Counter
from typing import List, Tuple

import matplotlib.pyplot as plt
from transformers import AutoTokenizer

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.llm.llm_client import LocalLLM
from src.utils.config import DEFAULT_MODEL_PATH, DEFAULT_VLLM_BASE_URL

SYSTEM_PROMPT_CN = """### Core Instructions ###

问题本身并不重要：您的主要目标是分析当前数学问题所给出的条件具体有哪些，问题里所需要求得的答案并不重要，任何所要直接求出答案的尝试均会视为对命令的违反。相反，需要关注的是题目给出的条件，详细列出题目给出了什么条件。
给出条件的诚实性：如果你已经给出了所能找到的所有条件，请直接说出已经找到了所有的条件，不要尝试猜测或编造看似正确的条件。给出的条件的要求如下：
1. 建立问题中数学对象的临界性质。
2. 对所有数学使用TeX：所有数学变量、表达式和关系都必须用TeX分隔符括起来（例如，“设$n$为整数。”）。

###输出格式###

您的回复必须按照以下部分的确切顺序进行组织。

1. 摘要

简要概述你的发现。

2.已知的数学条件

详细列出题目给出的所有数学条件。

###自我纠正说明###

在最终确定输出之前，请仔细查看您的“已知的数学条件”，以确保它们干净、严谨，并严格遵守上述所有说明。验证每个语句是否直接有助于最终连贯的数学论证。"""


SYSTEM_PROMPT_EN1 = """### Core Instructions ###

The problem itself is not important: your main goal is to analyze the specific conditions given by the current mathematical problem, and the answer required in the problem is not important. Any attempt to directly find the answer will be considered a violation of the command. On the contrary, it is important to pay attention to the conditions given by the question and list in detail what conditions the question provides.
Honesty in providing conditions: If you have provided all the conditions that can be found, please state directly that all the conditions have been found, and do not attempt to guess or fabricate seemingly correct conditions. The requirements for the given conditions are as follows:
1. Establish the critical properties of mathematical objects in the problem.
2. Use TeX for all mathematics: All mathematical variables, expressions, and relationships must be enclosed in TeX separators (for example, "Let $n $be an integer. ”）.

###Output format###

Your response must be organized in the exact order of the following sections.

1. Abstract

Provide a brief overview of your findings.

2. Known mathematical conditions

List in detail all the mathematical conditions given in the problem.

###Self correction instructions###

Before finalizing the output, please carefully review your 'known mathematical conditions' to ensure they are clean, rigorous, and strictly adhere to all the above instructions. Verify whether each statement directly contributes to the final coherent mathematical argument. """


SYSTEM_PROMPT_EN = '''
### Core Instructions ###

*   **Rigor is Paramount:** Your primary goal is to produce a complete and rigorously justified solution. Every step in your solution must be logically sound and clearly explained. A correct final answer derived from flawed or incomplete reasoning is considered a failure.
*   **Honesty About Completeness:** If you cannot find a complete solution, you must **not** guess or create a solution that appears correct but contains hidden flaws or justification gaps. Instead, you should present only significant partial results that you can rigorously prove. A partial result is considered significant if it represents a substantial advancement toward a full solution. Examples include:
    *   Proving a key lemma.
    *   Fully resolving one or more cases within a logically sound case-based proof.
    *   Establishing a critical property of the mathematical objects in the problem.
    *   For an optimization problem, proving an upper or lower bound without proving that this bound is achievable.
*   **Use TeX for All Mathematics:** All mathematical variables, expressions, and relations must be enclosed in TeX delimiters (e.g., `Let $n$ be an integer.`).

### Output Format ###

Your response MUST be structured into the following sections, in this exact order.

**1. Summary**

Provide a concise overview of your findings. This section must contain two parts:

*   **a. Verdict:** State clearly whether you have found a complete solution or a partial solution.
    *   **For a complete solution:** State the final answer, e.g., "I have successfully solved the problem. The final answer is..."
    *   **For a partial solution:** State the main rigorous conclusion(s) you were able to prove, e.g., "I have not found a complete solution, but I have rigorously proven that..."
*   **b. Method Sketch:** Present a high-level, conceptual outline of your solution. This sketch should allow an expert to understand the logical flow of your argument without reading the full detail. It should include:
    *   A narrative of your overall strategy.
    *   The full and precise mathematical statements of any key lemmas or major intermediate results.
    *   If applicable, describe any key constructions or case splits that form the backbone of your argument.

**2. Detailed Solution**

Present the full, step-by-step mathematical proof. Each step must be logically justified and clearly explained. The level of detail should be sufficient for an expert to verify the correctness of your reasoning without needing to fill in any gaps. This section must contain ONLY the complete, rigorous proof, free of any internal commentary, alternative approaches, or failed attempts.

### Self-Correction Instruction ###

Before finalizing your output, carefully review your "Method Sketch" and "Detailed Solution" to ensure they are clean, rigorous, and strictly adhere to all instructions provided above. Verify that every statement contributes directly to the final, coherent mathematical argument.
'''


USER_PROMPT = r"""
### Problem ###

Every morning Aya goes for a $9$-kilometer-long walk and stops at a coffee shop afterwards. When she walks at a constant speed of $s$ kilometers per hour, the walk takes her 4 hours, including $t$ minutes spent in the coffee shop. When she walks $s+2$ kilometers per hour, the walk takes her 2 hours and 24 minutes, including $t$ minutes spent in the coffee shop. Suppose Aya walks at $s+ rac{1}{2}$ kilometers per hour. Find the number of minutes the walk takes her, including the $t$ minutes spent in the coffee shop.
"""


def format_token_label(tokenizer: AutoTokenizer, token: str) -> str:
    """Render tokenizer-specific tokens into a human-readable label."""
    try:
        readable = tokenizer.convert_tokens_to_string([token])
    except Exception:
        readable = token

    readable = readable.replace("\r", "\\r").replace("\n", "\\n").replace("\t", "\\t")

    if readable == "":
        return repr(token)

    if readable.strip() == "":
        # Token is purely whitespace; expose each space.
        return "␠" * len(readable)

    leading_spaces = len(readable) - len(readable.lstrip(" "))
    trailing_spaces = len(readable) - len(readable.rstrip(" "))
    core = readable.strip(" ")
    label = f"{'␠' * leading_spaces}{core}{'␠' * trailing_spaces}"
    return label



def main() -> None:
    
    model_path = "/home/zhangdw/models/Qwen/Qwen3-8B"
    api_base = "http://0.0.0.0:8000"
    max_context_length = 1000
    max_new_tokens = 32768
    temperature = 0.6
    top_p = 0.95
    dry_run = False
    texts = [{"role": "system", "content": SYSTEM_PROMPT_EN}, {"role": "user", "content": USER_PROMPT}]
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
    # print(texts)
    print("=" * 40)

    try:
        answer = client.generate(texts)
    except Exception as exc:
        print(f"[LLM 调用失败] {exc}")
        return

    print("回答：")
    print(answer)

    # Token-level diagnostics for the generated text
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
    )
    tokens = tokenizer.tokenize(answer)
    counts = Counter(tokens)
    total_tokens = sum(counts.values())

    print("\nTop 20 tokens (count | percentage):")
    for tok, cnt in counts.most_common(20):
        pct = (cnt / total_tokens) * 100 if total_tokens else 0
        label = format_token_label(tokenizer, tok)
        print(f"{label:>15}: {cnt:>5} | {pct:5.2f}%")

    # Visualize the distribution of the most frequent tokens
    def plot_top_tokens(token_counts: Counter, top_k: int = 20) -> Tuple[List[str], List[int]]:
        top_items = token_counts.most_common(top_k)
        labels = [format_token_label(tokenizer, item[0]) for item in top_items]
        values = [item[1] for item in top_items]
        return labels, values

    labels, values = plot_top_tokens(counts, top_k=20)
    os.makedirs("outputs", exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(range(len(values)), values, color="#4C72B0")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Count")
    ax.set_title("Top token frequencies (generation output)")
    fig.tight_layout()
    output_path = os.path.join("outputs", "token_distribution.png")
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Token distribution plot saved to: {output_path}")


if __name__ == "__main__":
    main()

