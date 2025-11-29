#!/usr/bin/env python3
"""
测试 self-evolving solver 在 AIME 2024 问题集上的表现。
使用 qwen_math 模块提供的数学结果判断功能来评估答案正确性。
"""

from pathlib import Path
from typing import Dict, List, Any
import sys
import logging

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.llm.llm_client import LocalLLM
from src.prompts.prompts import PromptBuilder
from src.utils.qwen_math import compute_score
from src.utils.config import DEFAULT_MODEL_PATH, DEFAULT_VLLM_BASE_URL
from src.utils.data_loader import load_problem, ProblemRecord
from src.solver.solver_engine import SelfEvolvingSolver
from src.utils.logging_utils import configure_logging


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


def test_self_evolving_solver(
    questions_dir: Path,
    questions_file: str = "aime2024_questions.txt",
    model_path: str = DEFAULT_MODEL_PATH,
    api_base: str = DEFAULT_VLLM_BASE_URL,
    max_new_tokens: int = 4096,
    max_context_length: int = 30000,
    temperature: float = 0.1,
    top_p: float = 0.9,
    rounds: int = 2,
    verification_max_new_tokens: int = None,
    max_report_tokens: int = 10000,
    dry_run: bool = False,
    log_path: Path = None,
) -> Dict[str, Any]:
    """
    测试 self-evolving solver 在问题集上的表现。
    
    Args:
        questions_dir: questions 文件夹路径
        questions_file: 问题文件名
        model_path: 模型路径
        api_base: API base URL
        max_new_tokens: 最大生成token数
        max_context_length: 最大上下文长度
        temperature: 温度参数
        top_p: top_p参数
        rounds: 迭代轮数
        verification_max_new_tokens: 验证阶段最大生成token数
        max_report_tokens: 最大报告token数
        dry_run: 是否为dry run模式
        log_path: 日志文件路径（可选）
    
    Returns:
        包含测试结果的字典
    """
    # 使用 data_loader 中的方法加载所有问题
    print(f"正在从 {questions_dir} 加载问题文件: {questions_file}")
    problems = load_all_problems(questions_dir, questions_file)
    print(f"共加载 {len(problems)} 个问题")
    
    # 初始化模型
    print(f"\n初始化模型: {model_path}")
    print(f"API base: {api_base}")
    print(f"迭代轮数: {rounds}")
    llm = LocalLLM(
        model_path=model_path,
        api_base=api_base,
        max_new_tokens=max_new_tokens,
        max_context_length=max_context_length,
        temperature=temperature,
        top_p=top_p,
        dry_run=dry_run,
    )
    
    # 配置 logger（可选）
    logger = None
    if log_path:
        logger = configure_logging(str(log_path))
    else:
        # 创建一个简单的控制台 logger
        logger = logging.getLogger("test_self_evolving_solver")
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
            handler.setFormatter(formatter)
            logger.addHandler(handler)
    
    # 创建 solver
    prompt_builder = PromptBuilder()
    solver = SelfEvolvingSolver(
        llm=llm,
        prompt_builder=prompt_builder,
        logger=logger,
        max_new_tokens=max_new_tokens,
        verification_max_new_tokens=verification_max_new_tokens,
        max_report_tokens=max_report_tokens,
    )
    
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
                "boxed_answer": None,
                "verdict": None,
            })
            continue
        
        print(f"标准答案: {ground_truth}")
        
        try:
            # 使用 solver 求解
            print(f"  正在使用 self-evolving solver 求解（{rounds} 轮迭代）...")
            solve_result = solver.solve(record, rounds=rounds)
            
            # 提取结果（使用 solve 实际返回的字段）
            # solve 返回的字段：problem_id, problem, final_solution, final_solution_body, 
            # final_verification, verdict, boxed_answer, history
            final_solution = solve_result.get("final_solution", "")
            
            # 评估答案
            print("  正在评估答案...")
            # 优先使用 final_solution_body，其次使用 final_solution，最后使用 boxed_answer
            solution_str = final_solution
            score = compute_score(
                data_source="aime2024",
                solution_str=solution_str,
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
                "solution": final_solution[:500] + "..." if len(final_solution) > 500 else final_solution,
            })
            
        except Exception as e:
            print(f"  错误: {e}")
            import traceback
            traceback.print_exc()
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
    
    parser = argparse.ArgumentParser(description="测试 self-evolving solver 在 AIME 2024 问题集上的表现")
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
        default="/home/zhangdw/models/Qwen/Qwen3-8B",
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
        default=20000,
        help="最大生成token数",
    )
    parser.add_argument(
        "--max_context_length",
        type=int,
        default=32768,
        help="最大上下文长度",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="温度参数",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=1.0,
        help="top_p参数",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=5,
        help="迭代轮数",
    )
    parser.add_argument(
        "--verification_max_new_tokens",
        type=int,
        default=25000,
        help="验证阶段最大生成token数",
    )
    parser.add_argument(
        "--max_report_tokens",
        type=int,
        default=5000,
        help="最大报告token数",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        default=False,
        help="Dry run模式（不实际调用模型）",
    )
    parser.add_argument(
        "--log_path",
        type=Path,
        default="/home/zhangdw/ReasoningSelfEvolve/outputs/latest_run.log",
        help="日志文件路径（可选）",
    )
    
    args = parser.parse_args()
    
    # 运行测试
    results = test_self_evolving_solver(
        questions_dir=args.questions_dir,
        questions_file=args.questions_file,
        model_path=args.model,
        api_base=args.api_base,
        max_new_tokens=args.max_new_tokens,
        max_context_length=args.max_context_length,
        temperature=args.temperature,
        top_p=args.top_p,
        rounds=args.rounds,
        verification_max_new_tokens=args.verification_max_new_tokens,
        max_report_tokens=args.max_report_tokens,
        dry_run=args.dry_run,
        log_path=args.log_path,
    )


if __name__ == "__main__":
    main()

