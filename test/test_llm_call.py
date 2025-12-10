from __future__ import annotations

import os
import sys
from collections import Counter
from pathlib import Path

from transformers import AutoTokenizer

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.llm.llm_client import LocalLLM
from src.utils.config import DEFAULT_MODEL_PATH, DEFAULT_VLLM_BASE_URL
from src.utils.data_loader import load_problem
from src.utils.visualization import format_token_label, plot_token_distribution, plot_log_binned_tokens
from src.prompts.prompts import PROMPT_SOLUTION_SYSTEM

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






def get_all_problem_ids() -> list:
    """生成所有 AIME 2024 题目 ID 列表"""
    problem_ids = []
    # 2024-I-1 到 2024-I-15
    for i in range(1, 16):
        problem_ids.append(f"2024-I-{i}")
    # 2024-II-1 到 2024-II-15
    for i in range(1, 16):
        problem_ids.append(f"2024-II-{i}")
    return problem_ids


def process_single_problem(
    problem_id: str,
    client: LocalLLM,
    tokenizer: AutoTokenizer,
    questions_dir: Path,
) -> None:
    """处理单个题目：生成回答并可视化 token 分布"""
    print("\n" + "=" * 60)
    print(f"正在处理题目: {problem_id}")
    print("=" * 60)
    
    # 加载题目
    try:
        problem = load_problem(questions_dir, problem_id=problem_id)
    except Exception as exc:
        print(f"[加载题目失败] {problem_id}: {exc}")
        return
    
    user_prompt = f"### Problem ###\n\n{problem.prompt}"
    texts = [{"role": "system", "content": PROMPT_SOLUTION_SYSTEM}, {"role": "user", "content": user_prompt}]
    
    print("问题：")
    print(texts[1]["content"])
    print("-" * 40)
    
    # 生成回答
    try:
        answer = client.generate(texts)
    except Exception as exc:
        print(f"[LLM 调用失败] {problem_id}: {exc}")
        return
    
    print("回答：")
    print(answer)
    
    # Token 统计
    tokens = tokenizer.tokenize(answer)
    counts = Counter(tokens)
    total_tokens = sum(counts.values())
    
    print(f"\nTop 20 tokens (count | percentage) for {problem_id}:")
    for tok, cnt in counts.most_common(20):
        pct = (cnt / total_tokens) * 100 if total_tokens else 0
        label = format_token_label(tokenizer, tok)
        print(f"{label:>15}: {cnt:>5} | {pct:5.2f}%")
    
    # 可视化
    image_dir = os.path.join(PROJECT_ROOT, "images", problem_id)
    
    output_path = plot_token_distribution(counts, tokenizer, image_dir, problem_id)
    if output_path:
        print(f"Token distribution plot saved to: {output_path}")
    
    output_path2 = plot_log_binned_tokens(counts, tokenizer, image_dir, problem_id)
    if output_path2:
        print(f"Log-binned token distribution plot saved to: {output_path2}")


def main() -> None:
    model_path = "/home/zhangdw/models/Qwen/Qwen3-8B"
    api_base = "http://0.0.0.0:8000"
    max_context_length = 32768
    max_new_tokens = 32768
    temperature = 0.6
    top_p = 0.95
    dry_run = False
    
    questions_dir = Path(PROJECT_ROOT) / "questions"
    
    # 初始化 LLM 客户端（只初始化一次）
    client = LocalLLM(
        model_path=model_path,
        api_base=api_base,
        max_new_tokens=max_new_tokens,
        max_context_length=max_context_length,
        temperature=temperature,
        top_p=top_p,
        dry_run=dry_run,
    )
    
    # 初始化 tokenizer（只初始化一次）
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
    )
    
    # 获取所有题目 ID
    problem_ids = get_all_problem_ids()
    total_problems = len(problem_ids)
    
    print(f"共 {total_problems} 道题目需要处理")
    print("题目列表:", problem_ids)
    
    # 遍历所有题目
    for idx, problem_id in enumerate(problem_ids, 1):
        print(f"\n[{idx}/{total_problems}] ", end="")
        process_single_problem(problem_id, client, tokenizer, questions_dir)
    
    print("\n" + "=" * 60)
    print("所有题目处理完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()

