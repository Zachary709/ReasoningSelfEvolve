#!/usr/bin/env python3
"""
测试 base model 在 AIME 2024 问题集上的表现。
使用 qwen_math 模块提供的数学结果判断功能来评估答案正确性。
"""

from pathlib import Path
from typing import Dict, List, Tuple, Any
import sys

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.llm.llm_client import LocalLLM
from src.prompts.prompts import PromptBuilder
from src.utils.qwen_math import compute_score
from src.utils.config import DEFAULT_MODEL_PATH, DEFAULT_VLLM_BASE_URL
from src.utils.data_loader import load_problem, ProblemRecord


def load_all_problems(
    questions_dir: Path, questions_file: str = "aime2024_questions.txt"
) -> List[ProblemRecord]:
    """
    使用 data_loader 中的方法加载所有问题。
    
    Args:
        questions_dir: questions 文件夹路径
        questions_file: 问题文件名
    
    Returns:
        List of ProblemRecord 对象
    """
    questions_path = questions_dir / questions_file
    if not questions_path.exists():
        raise FileNotFoundError(f"Questions file not found at {questions_path}")
    
    # 先读取所有问题ID
    problem_ids = []
    with questions_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t", 1)
            if len(parts) == 2:
                problem_id = parts[0]
                problem_ids.append(problem_id)
    
    if not problem_ids:
        raise ValueError(f"No problems found in {questions_path}")
    
    # 使用 load_problem 加载每个问题
    problems = []
    for problem_id in sorted(problem_ids):
        record = load_problem(questions_dir, problem_id, questions_file)
        problems.append(record)
    
    return problems


def test_base_model(
    questions_dir: Path,
    questions_file: str = "aime2024_questions.txt",
    model_path: str = DEFAULT_MODEL_PATH,
    api_base: str = DEFAULT_VLLM_BASE_URL,
    max_new_tokens: int = 4096,
    temperature: float = 0.1,
    top_p: float = 0.9,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """
    测试 base model 在问题集上的表现。
    
    Args:
        questions_dir: questions 文件夹路径
        questions_file: 问题文件名
        model_path: 模型路径
        api_base: API base URL
        max_new_tokens: 最大生成token数
        temperature: 温度参数
        top_p: top_p参数
        dry_run: 是否为dry run模式
    
    Returns:
        包含测试结果的字典
    """
    # 使用 data_loader 中的方法加载所有问题
    print(f"正在从 {questions_dir} 加载问题文件: {questions_file}")
    problems = load_all_problems(questions_dir, questions_file)
    print(f"共加载 {len(problems)} 个问题")
    
    # 初始化模型和提示构建器
    print(f"\n初始化模型: {model_path}")
    print(f"API base: {api_base}")
    llm = LocalLLM(
        model_path=model_path,
        api_base=api_base,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        dry_run=dry_run,
    )
    prompt_builder = PromptBuilder()
    
    # 测试结果
    results = []
    correct_count = 0
    total_count = 0
    
    print("\n开始测试...")
    print("=" * 80)
    
    for idx, record in enumerate(problems, 1):
        problem_id = record.problem_id
        problem_text = record.prompt
        ground_truth = record.answer
        
        print(f"\n[{idx}/{len(problems)}] 问题 ID: {problem_id}")
        print(f"问题: {problem_text[:100]}..." if len(problem_text) > 100 else f"问题: {problem_text}")
        
        if ground_truth is None:
            print("  警告: 没有标准答案，跳过")
            results.append({
                "problem_id": problem_id,
                "correct": None,
                "score": None,
                "ground_truth": None,
                "solution": None,
            })
            continue
        
        print(f"标准答案: {ground_truth}")
        
        try:
            # 构建提示
            messages = prompt_builder.solution(problem_text)
            
            # 生成解答
            print("  正在生成解答...")
            solution_text = llm.generate(messages, max_new_tokens_override=max_new_tokens)
            
            # 评估答案
            print("  正在评估答案...")
            score = compute_score(
                data_source="aime2024",
                solution_str=solution_text,
                ground_truth=ground_truth,
            )
            
            is_correct = score > 0.5
            if is_correct:
                correct_count += 1
            total_count += 1
            
            print(f"  结果: {'✓ 正确' if is_correct else '✗ 错误'} (score: {score})")
            
            results.append({
                "problem_id": problem_id,
                "correct": is_correct,
                "score": score,
                "ground_truth": ground_truth,
                "solution": solution_text[:500] + "..." if len(solution_text) > 500 else solution_text,
            })
            
        except Exception as e:
            print(f"  错误: {e}")
            results.append({
                "problem_id": problem_id,
                "correct": False,
                "score": 0.0,
                "ground_truth": ground_truth,
                "solution": f"Error: {str(e)}",
            })
            total_count += 1
    
    # 统计结果
    accuracy = correct_count / total_count if total_count > 0 else 0.0
    
    print("\n" + "=" * 80)
    print("\n测试完成!")
    print(f"总问题数: {len(problems)}")
    print(f"有标准答案的问题数: {total_count}")
    print(f"正确答案数: {correct_count}")
    print(f"准确率: {accuracy:.2%} ({correct_count}/{total_count})")
    
    return {
        "total_problems": len(problems),
        "total_with_answer": total_count,
        "correct_count": correct_count,
        "accuracy": accuracy,
        "results": results,
    }


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="测试 base model 在 AIME 2024 问题集上的表现")
    parser.add_argument(
        "--questions_dir",
        type=Path,
        default=PROJECT_ROOT / "questions",
        help="questions 文件夹路径",
    )
    parser.add_argument(
        "--questions_file",
        type=str,
        default="aime2024_questions.txt",
        help="问题文件名",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL_PATH,
        help="模型路径",
    )
    parser.add_argument(
        "--api_base",
        type=str,
        default=DEFAULT_VLLM_BASE_URL,
        help="API base URL",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=32768,
        help="最大生成token数",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="温度参数",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=1.0,
        help="top_p参数",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        default=False,
        help="Dry run模式（不实际调用模型）",
    )
    
    args = parser.parse_args()
    
    # 运行测试
    results = test_base_model(
        questions_dir=args.questions_dir,
        questions_file=args.questions_file,
        model_path=args.model,
        api_base=args.api_base,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()

