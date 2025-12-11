#!/usr/bin/env python3
"""
测试 base model 在 AIME 2024 问题集上的表现。
使用 qwen_math 模块提供的数学结果判断功能来评估答案正确性。
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from collections import Counter
import sys

import yaml
from transformers import AutoTokenizer

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.llm.llm_client import LocalLLM
from src.prompts.prompts import PromptBuilder
from src.utils.qwen_math import compute_score
from src.utils.config import DEFAULT_MODEL_PATH, DEFAULT_VLLM_BASE_URL
from src.utils.data_loader import load_problem, load_all_problems, ProblemRecord

# 默认配置文件路径
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "test_base_model_config.yaml"


def format_token_label(tokenizer: AutoTokenizer, token: str) -> str:
    """
    格式化 token 标签用于显示。
    将特殊字符转换为可读形式。
    
    Args:
        tokenizer: tokenizer 实例
        token: 原始 token 字符串
    
    Returns:
        格式化后的标签字符串
    """
    # 替换常见的特殊字符
    label = token
    label = label.replace("Ġ", "_")      # 空格前缀
    label = label.replace("Ċ", "\\n")    # 换行符
    label = label.replace("ĉ", "\\t")    # 制表符
    label = label.replace("$", "\\$")    # 美元符号
    return label


def save_solution_and_token_stats(
    problem_id: str,
    solution_text: str,
    tokenizer: AutoTokenizer,
    project_root: Path,
) -> Dict[str, Any]:
    """
    保存 solution.txt 和 token_stats.json 到 outputs 文件夹。
    
    Args:
        problem_id: 问题 ID
        solution_text: 生成的解答文本
        tokenizer: 用于 tokenize 的 tokenizer
        project_root: 项目根目录
    
    Returns:
        包含 token 统计信息的字典
    """
    # 创建输出目录
    output_dir = project_root / "outputs" / problem_id
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存 solution.txt
    solution_path = output_dir / "solution.txt"
    with open(solution_path, "w", encoding="utf-8") as f:
        f.write(solution_text)
    print(f"    Solution saved to: {solution_path}")
    
    # Token 统计
    tokens = tokenizer.tokenize(solution_text)
    counts = Counter(tokens)
    total_tokens_count = sum(counts.values())
    
    # 构建 token_stats 字典
    token_stats = {
        "problem_id": problem_id,
        "total_tokens": total_tokens_count,
        "unique_tokens": len(counts),
        "token_counts": dict(counts.most_common()),  # 按频率排序的完整 token 计数
        "top_20_tokens": [
            {
                "token": tok,
                "label": format_token_label(tokenizer, tok),
                "count": cnt,
                "percentage": round((cnt / total_tokens_count) * 100, 4) if total_tokens_count else 0
            }
            for tok, cnt in counts.most_common(20)
        ],
    }
    
    # 保存 token_stats.json
    token_stats_path = output_dir / "token_stats.json"
    with open(token_stats_path, "w", encoding="utf-8") as f:
        json.dump(token_stats, f, ensure_ascii=False, indent=2)
    print(f"    Token stats saved to: {token_stats_path}")
    
    return token_stats


def load_config(config_path: Optional[Path] = None) -> dict:
    """
    从 YAML 配置文件加载配置。
    
    Args:
        config_path: 配置文件路径，默认为 config/test_base_model_config.yaml
    
    Returns:
        配置字典
    """
    if config_path is None:
        config_path = DEFAULT_CONFIG_PATH
    
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    return config


def test_base_model(
    questions_dir: Path,
    questions_file: str = "aime2024_questions.txt",
    model_path: str = DEFAULT_MODEL_PATH,
    api_base: str = DEFAULT_VLLM_BASE_URL,
    max_new_tokens: int = 32768,
    temperature: float = 0.6,
    top_p: float = 0.95,
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
    
    # 初始化 tokenizer（用于可视化）
    print("初始化 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
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
            
            # 保存 solution 和 token 统计
            print("  正在保存结果...")
            save_solution_and_token_stats(
                problem_id=problem_id,
                solution_text=solution_text,
                tokenizer=tokenizer,
                project_root=PROJECT_ROOT,
            )
            
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
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="配置文件路径",
    )
    parser.add_argument(
        "--questions_dir",
        type=Path,
        default=None,
        help="questions 文件夹路径（覆盖配置文件）",
    )
    parser.add_argument(
        "--questions_file",
        type=str,
        default=None,
        help="问题文件名（覆盖配置文件）",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="模型路径（覆盖配置文件）",
    )
    parser.add_argument(
        "--api_base",
        type=str,
        default=None,
        help="API base URL（覆盖配置文件）",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=None,
        help="最大生成token数（覆盖配置文件）",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="温度参数（覆盖配置文件）",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=None,
        help="top_p参数（覆盖配置文件）",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        default=None,
        help="Dry run模式（覆盖配置文件）",
    )
    
    args = parser.parse_args()
    
    # 加载配置文件
    print(f"加载配置文件: {args.config}")
    config = load_config(args.config)
    
    # 命令行参数覆盖配置文件
    questions_dir = args.questions_dir if args.questions_dir is not None else Path(config.get("questions_dir", PROJECT_ROOT / "questions"))
    questions_file = args.questions_file if args.questions_file is not None else config.get("questions_file", "aime2024_questions.txt")
    model_path = args.model if args.model is not None else config.get("model", DEFAULT_MODEL_PATH)
    api_base = args.api_base if args.api_base is not None else config.get("api_base", DEFAULT_VLLM_BASE_URL)
    max_new_tokens = args.max_new_tokens if args.max_new_tokens is not None else config.get("max_new_tokens", 32768)
    temperature = args.temperature if args.temperature is not None else config.get("temperature", 0.6)
    top_p = args.top_p if args.top_p is not None else config.get("top_p", 0.95)
    dry_run = args.dry_run if args.dry_run is not None else config.get("dry_run", False)
    
    print(f"配置参数:")
    print(f"  questions_dir: {questions_dir}")
    print(f"  questions_file: {questions_file}")
    print(f"  model: {model_path}")
    print(f"  api_base: {api_base}")
    print(f"  max_new_tokens: {max_new_tokens}")
    print(f"  temperature: {temperature}")
    print(f"  top_p: {top_p}")
    print(f"  dry_run: {dry_run}")
    
    # 运行测试
    results = test_base_model(
        questions_dir=questions_dir,
        questions_file=questions_file,
        model_path=model_path,
        api_base=api_base,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        dry_run=dry_run,
    )


if __name__ == "__main__":
    main()

