#!/usr/bin/env python3
"""
测试 base model 在 AIME 2024 问题集上的表现。
使用 qwen_math 模块提供的数学结果判断功能来评估答案正确性。
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Set
from collections import Counter
from datetime import datetime
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

# 测试输出目录（使用项目根目录下的 outputs/base_model）
TEST_OUTPUTS_DIR = PROJECT_ROOT / "outputs" / "base_model"


class OutputManager:
    """
    输出文件管理器，支持流式写入。
    输出到 test/outputs/{session_id}/run.out，session_id 格式为 "MM-DD_HH-MM"。
    """
    
    def __init__(self, output_dir: Path = TEST_OUTPUTS_DIR, session_id: Optional[str] = None):
        """
        初始化输出管理器。
        
        Args:
            output_dir: 输出根目录路径
            session_id: 会话 ID（格式为 MM-DD_HH-MM），为 None 则使用当前时间
        """
        self.start_time = datetime.now()
        
        # 生成或使用 session_id
        if session_id:
            self.session_id = session_id
        else:
            self.session_id = self.start_time.strftime("%m-%d_%H-%M")
        
        # 创建 session 子文件夹
        self.session_dir = output_dir / self.session_id
        self.session_dir.mkdir(parents=True, exist_ok=True)
        
        # 文件名固定为 run.out
        self.filename = "run.out"
        self.filepath = self.session_dir / self.filename
        
        # 清空或创建文件（覆盖模式）
        with open(self.filepath, "w", encoding="utf-8") as f:
            f.write(f"# 测试开始时间: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"# 会话 ID: {self.session_id}\n")
            f.write("=" * 80 + "\n\n")
        
        print(f"输出文件: {self.filepath}")
    
    def write(self, content: str, flush: bool = True):
        """
        流式写入内容到输出文件。
        
        Args:
            content: 要写入的内容
            flush: 是否立即刷新到磁盘
        """
        with open(self.filepath, "a", encoding="utf-8") as f:
            f.write(content)
            if flush:
                f.flush()
                os.fsync(f.fileno())  # 确保写入磁盘
    
    def writeln(self, content: str = "", flush: bool = True):
        """写入一行内容"""
        self.write(content + "\n", flush)
    
    def write_problem_start(self, idx: int, total: int, problem_id: str, problem_text: str, ground_truth: Optional[str]):
        """写入问题开始标记"""
        self.writeln()
        self.writeln(f"[{idx}/{total}] 问题 ID: {problem_id}")
        self.writeln(f"问题: {problem_text[:200]}..." if len(problem_text) > 200 else f"问题: {problem_text}")
        if ground_truth:
            self.writeln(f"标准答案: {ground_truth}")
    
    def write_problem_result(self, problem_id: str, is_correct: Optional[bool], score: Optional[float], 
                             solution_preview: str = "", error: str = ""):
        """写入问题结果"""
        if error:
            self.writeln(f"  错误: {error}")
        elif is_correct is not None:
            status = "✓ 正确" if is_correct else "✗ 错误"
            self.writeln(f"  结果: {status} (score: {score})")
            if solution_preview:
                self.writeln(f"  解答预览: {solution_preview[:300]}...")
        else:
            self.writeln(f"  警告: 没有标准答案，跳过")
        self.writeln("-" * 40)
    
    def write_summary(self, total_problems: int, total_with_answer: int, correct_count: int, accuracy: float):
        """写入测试总结"""
        self.writeln()
        self.writeln("=" * 80)
        self.writeln("测试完成!")
        self.writeln(f"总问题数: {total_problems}")
        self.writeln(f"有标准答案的问题数: {total_with_answer}")
        self.writeln(f"正确答案数: {correct_count}")
        self.writeln(f"准确率: {accuracy:.2%} ({correct_count}/{total_with_answer})")
        self.writeln(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


class ProgressTracker:
    """
    进度跟踪器，支持题级别的断点重连。
    进度文件保存到 test/outputs/{session_id}/progress.json。
    """
    
    def __init__(self, output_dir: Path = TEST_OUTPUTS_DIR, session_id: Optional[str] = None):
        """
        初始化进度跟踪器。
        
        Args:
            output_dir: 输出根目录路径
            session_id: 会话标识（用于断点重连），格式为 "MM-DD_HH-MM"
        """
        # 如果提供了 session_id，使用它；否则使用当前时间
        if session_id:
            self.session_id = session_id
        else:
            self.session_id = datetime.now().strftime("%m-%d_%H-%M")
        
        # 创建 session 子文件夹
        self.session_dir = output_dir / self.session_id
        self.session_dir.mkdir(parents=True, exist_ok=True)
        
        # 进度文件固定为 progress.json
        self.progress_file = self.session_dir / "progress.json"
        
        # 加载已完成的问题
        self.completed_problems: Set[str] = set()
        self.results: Dict[str, Dict[str, Any]] = {}
        self._load_progress()
    
    def _load_progress(self):
        """从进度文件加载已完成的问题"""
        if self.progress_file.exists():
            try:
                with open(self.progress_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.completed_problems = set(data.get("completed_problems", []))
                    self.results = data.get("results", {})
                    print(f"已恢复进度: {len(self.completed_problems)} 个问题已完成")
            except (json.JSONDecodeError, IOError) as e:
                print(f"警告: 无法加载进度文件 {self.progress_file}: {e}")
                self.completed_problems = set()
                self.results = {}
    
    def _sort_problem_ids(self, problem_ids: List[str]) -> List[str]:
        """
        对题目 ID 进行排序。
        题目 ID 格式为 "2024-I-1"、"2024-II-15" 等，按年份、卷号（I 在 II 之前）、题号排序。
        """
        def sort_key(problem_id: str):
            parts = problem_id.split("-")
            if len(parts) >= 3:
                year = parts[0]
                volume = 0 if parts[1] == "I" else 1  # I 在 II 之前
                try:
                    num = int(parts[2])
                except ValueError:
                    num = 0
                return (year, volume, num)
            return (problem_id, 0, 0)
        
        return sorted(problem_ids, key=sort_key)
    
    def _save_progress(self):
        """保存进度到文件"""
        # 生成 correctness 字典：每个题目的答案正确性
        correctness = {}
        for problem_id, problem_data in self.results.items():
            correctness[problem_id] = problem_data.get("correct")
        
        # 对 completed_problems 和 results 进行排序
        sorted_completed = self._sort_problem_ids(list(self.completed_problems))
        sorted_correctness = {k: correctness[k] for k in self._sort_problem_ids(list(correctness.keys()))}
        sorted_results = {k: self.results[k] for k in self._sort_problem_ids(list(self.results.keys()))}
        
        data = {
            "session_id": self.session_id,
            "completed_problems": sorted_completed,
            "correctness": sorted_correctness,
            "results": sorted_results,
            "last_update": datetime.now().isoformat(),
        }
        with open(self.progress_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
    
    def is_completed(self, problem_id: str) -> bool:
        """检查问题是否已完成"""
        return problem_id in self.completed_problems
    
    def mark_completed(self, problem_id: str, result: Dict[str, Any]):
        """
        标记问题已完成并保存结果。
        
        Args:
            problem_id: 问题 ID
            result: 问题的测试结果
        """
        self.completed_problems.add(problem_id)
        self.results[problem_id] = result
        self._save_progress()  # 立即保存，防止中断丢失
    
    def get_result(self, problem_id: str) -> Optional[Dict[str, Any]]:
        """获取已完成问题的结果"""
        return self.results.get(problem_id)
    
    def get_stats(self) -> Dict[str, int]:
        """获取当前统计信息"""
        correct_count = sum(1 for r in self.results.values() if r.get("correct") is True)
        total_with_answer = sum(1 for r in self.results.values() if r.get("correct") is not None)
        return {
            "completed": len(self.completed_problems),
            "correct": correct_count,
            "total_with_answer": total_with_answer,
        }


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


def save_logprobs(
    problem_id: str,
    logprobs_data: Optional[List[Dict[str, Any]]],
    project_root: Path,
    session_id: Optional[str] = None,
) -> Optional[Path]:
    """
    保存 logprobs 数据到 outputs 文件夹。
    
    Args:
        problem_id: 问题 ID
        logprobs_data: logprobs 数据列表，每个元素包含 token, logprob, 可选的 top_logprobs
        project_root: 项目根目录
        session_id: 会话 ID（日期格式 MM-DD_HH-MM），用于创建子文件夹
    
    Returns:
        保存的文件路径，如果 logprobs_data 为 None 则返回 None
    """
    if logprobs_data is None:
        print(f"    警告: 没有 logprobs 数据可保存")
        return None
    
    # 创建输出目录：outputs/base_model/session_id/problem_id
    if session_id:
        output_dir = project_root / "outputs" / "base_model" / session_id / problem_id
    else:
        output_dir = project_root / "outputs" / "base_model" / problem_id
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存 logprobs.json
    logprobs_path = output_dir / "logprobs.json"
    
    # 构建保存的数据结构
    logprobs_output = {
        "problem_id": problem_id,
        "total_tokens": len(logprobs_data),
        "logprobs": logprobs_data,
    }
    
    with open(logprobs_path, "w", encoding="utf-8") as f:
        json.dump(logprobs_output, f, ensure_ascii=False, indent=2)
    
    print(f"    Logprobs saved to: {logprobs_path}")
    return logprobs_path


def save_solution_and_token_stats(
    problem_id: str,
    solution_text: str,
    tokenizer: AutoTokenizer,
    project_root: Path,
    session_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    保存 solution.txt 和 token_stats.json 到 outputs 文件夹。
    
    Args:
        problem_id: 问题 ID
        solution_text: 生成的解答文本
        tokenizer: 用于 tokenize 的 tokenizer
        project_root: 项目根目录
        session_id: 会话 ID（日期格式 MM-DD_HH-MM），用于创建子文件夹
    
    Returns:
        包含 token 统计信息的字典
    """
    # 创建输出目录：outputs/base_model/session_id/problem_id
    if session_id:
        output_dir = project_root / "outputs" / "base_model" / session_id / problem_id
    else:
        output_dir = project_root / "outputs" / "base_model" / problem_id
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
    top_logprobs: Optional[int] = 20,
    dry_run: bool = False,
    resume_session: Optional[str] = None,
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
        top_logprobs: 每个位置返回的 top logprobs 数量，为 0 或 None 则不获取 logprobs
        dry_run: 是否为dry run模式
        resume_session: 断点重连的会话 ID（格式为 "MM-DD_HH-MM"），为 None 则开始新会话
    
    Returns:
        包含测试结果的字典
    """
    # 初始化输出管理器和进度跟踪器
    if resume_session:
        # 断点重连模式：使用已有的会话
        progress_tracker = ProgressTracker(session_id=resume_session)
        # 输出追加到已有文件
        output_manager = OutputManager.__new__(OutputManager)
        output_manager.session_id = resume_session
        output_manager.session_dir = TEST_OUTPUTS_DIR / resume_session
        output_manager.session_dir.mkdir(parents=True, exist_ok=True)
        output_manager.filename = "run.out"
        output_manager.filepath = output_manager.session_dir / output_manager.filename
        output_manager.start_time = datetime.now()
        # 追加恢复标记
        output_manager.write(f"\n\n# 断点恢复时间: {output_manager.start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        output_manager.writeln("=" * 80)
        print(f"断点重连模式: 恢复会话 {resume_session}")
    else:
        # 新会话模式
        output_manager = OutputManager()
        progress_tracker = ProgressTracker(session_id=output_manager.session_id)
    
    # 使用 data_loader 中的方法加载所有问题
    print(f"正在从 {questions_dir} 加载问题文件: {questions_file}")
    output_manager.writeln(f"正在从 {questions_dir} 加载问题文件: {questions_file}")
    problems = load_all_problems(questions_dir, questions_file)
    print(f"共加载 {len(problems)} 个问题")
    output_manager.writeln(f"共加载 {len(problems)} 个问题")
    
    # 初始化模型和提示构建器
    print(f"\n初始化模型: {model_path}")
    print(f"API base: {api_base}")
    output_manager.writeln(f"\n模型: {model_path}")
    output_manager.writeln(f"API base: {api_base}")
    output_manager.writeln(f"参数: temperature={temperature}, top_p={top_p}, max_new_tokens={max_new_tokens}")
    
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
    skipped_count = 0
    
    # 从进度跟踪器恢复已有统计
    if resume_session:
        stats = progress_tracker.get_stats()
        skipped_count = stats["completed"]
        print(f"已跳过 {skipped_count} 个已完成的问题")
        output_manager.writeln(f"已跳过 {skipped_count} 个已完成的问题")
    
    print("\n开始测试...")
    print("=" * 80)
    output_manager.writeln("\n开始测试...")
    output_manager.writeln("=" * 80)
    
    for idx, record in enumerate(problems, 1):
        problem_id = record.problem_id
        problem_text = record.prompt
        ground_truth = record.answer
        
        # 检查是否已完成（断点重连）
        if progress_tracker.is_completed(problem_id):
            cached_result = progress_tracker.get_result(problem_id)
            results.append(cached_result)
            if cached_result.get("correct") is True:
                correct_count += 1
            if cached_result.get("correct") is not None:
                total_count += 1
            print(f"\n[{idx}/{len(problems)}] 问题 ID: {problem_id} - 已完成，跳过")
            continue
        
        print(f"\n[{idx}/{len(problems)}] 问题 ID: {problem_id}")
        print(f"问题: {problem_text[:100]}..." if len(problem_text) > 100 else f"问题: {problem_text}")
        
        # 流式写入问题开始
        output_manager.write_problem_start(idx, len(problems), problem_id, problem_text, ground_truth)
        
        if ground_truth is None:
            print("  警告: 没有标准答案，跳过")
            result = {
                "problem_id": problem_id,
                "correct": None,
                "score": None,
                "ground_truth": None,
                "solution": None,
            }
            results.append(result)
            progress_tracker.mark_completed(problem_id, result)
            output_manager.write_problem_result(problem_id, None, None)
            continue
        
        print(f"标准答案: {ground_truth}")
        
        try:
            # 构建提示
            messages = prompt_builder.solution(problem_text)
            
            # 生成解答（根据配置决定是否获取 logprobs）
            print("  正在生成解答...")
            return_logprobs = top_logprobs is not None and top_logprobs > 0
            
            if return_logprobs:
                generation_result = llm.generate(
                    messages, 
                    max_new_tokens_override=max_new_tokens,
                    return_logprobs=True,
                    top_logprobs=top_logprobs,
                )
                # 解析生成结果
                solution_text = generation_result["text"]
                logprobs_data = generation_result.get("logprobs")
            else:
                # 不获取 logprobs，直接返回文本
                solution_text = llm.generate(
                    messages, 
                    max_new_tokens_override=max_new_tokens,
                )
                logprobs_data = None
            
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
                session_id=progress_tracker.session_id,
            )
            
            # 保存 logprobs
            save_logprobs(
                problem_id=problem_id,
                logprobs_data=logprobs_data,
                project_root=PROJECT_ROOT,
                session_id=progress_tracker.session_id,
            )
            
            result = {
                "problem_id": problem_id,
                "correct": is_correct,
                "score": score,
                "ground_truth": ground_truth,
                "solution": solution_text[:500] + "..." if len(solution_text) > 500 else solution_text,
            }
            results.append(result)
            
            # 流式写入结果并保存进度（题级别断点）
            output_manager.write_problem_result(problem_id, is_correct, score, solution_text[:200])
            progress_tracker.mark_completed(problem_id, result)
            
        except Exception as e:
            print(f"  错误: {e}")
            result = {
                "problem_id": problem_id,
                "correct": False,
                "score": 0.0,
                "ground_truth": ground_truth,
                "solution": f"Error: {str(e)}",
            }
            results.append(result)
            total_count += 1
            
            # 流式写入错误并保存进度
            output_manager.write_problem_result(problem_id, False, 0.0, error=str(e))
            progress_tracker.mark_completed(problem_id, result)
    
    # 统计结果
    accuracy = correct_count / total_count if total_count > 0 else 0.0
    
    print("\n" + "=" * 80)
    print("\n测试完成!")
    print(f"总问题数: {len(problems)}")
    print(f"有标准答案的问题数: {total_count}")
    print(f"正确答案数: {correct_count}")
    print(f"准确率: {accuracy:.2%} ({correct_count}/{total_count})")
    
    # 写入测试总结
    output_manager.write_summary(len(problems), total_count, correct_count, accuracy)
    
    return {
        "total_problems": len(problems),
        "total_with_answer": total_count,
        "correct_count": correct_count,
        "accuracy": accuracy,
        "results": results,
        "session_id": progress_tracker.session_id,
    }


def main():
    """主函数：从配置文件读取参数并运行测试"""
    # 加载配置文件
    print(f"加载配置文件: {DEFAULT_CONFIG_PATH}")
    config = load_config(DEFAULT_CONFIG_PATH)
    
    # 从配置文件读取参数
    questions_dir = Path(config.get("questions_dir", PROJECT_ROOT / "questions"))
    questions_file = config.get("questions_file", "aime2024_questions.txt")
    model_path = config.get("model", DEFAULT_MODEL_PATH)
    api_base = config.get("api_base", DEFAULT_VLLM_BASE_URL)
    max_new_tokens = config.get("max_new_tokens", 32768)
    temperature = config.get("temperature", 0.6)
    top_p = config.get("top_p", 0.95)
    top_logprobs = config.get("top_logprobs", 5)
    # 处理 top_logprobs 为 null/None/0 的情况
    if top_logprobs is None or top_logprobs == 0:
        top_logprobs = None
    dry_run = config.get("dry_run", False)
    resume_session = config.get("resume", None)
    if resume_session is not None and str(resume_session).lower() == "none":
        resume_session = None
    
    print(f"配置参数:")
    print(f"  questions_dir: {questions_dir}")
    print(f"  questions_file: {questions_file}")
    print(f"  model: {model_path}")
    print(f"  api_base: {api_base}")
    print(f"  max_new_tokens: {max_new_tokens}")
    print(f"  temperature: {temperature}")
    print(f"  top_p: {top_p}")
    print(f"  top_logprobs: {top_logprobs}")
    print(f"  dry_run: {dry_run}")
    print(f"  resume: {resume_session}")
    
    # 运行测试
    results = test_base_model(
        questions_dir=questions_dir,
        questions_file=questions_file,
        model_path=model_path,
        api_base=api_base,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        top_logprobs=top_logprobs,
        dry_run=dry_run,
        resume_session=resume_session,
    )
    
    # 输出会话 ID，便于断点重连
    print(f"\n会话 ID: {results.get('session_id')}")
    print(f"如需断点重连，请在配置文件中设置: resume: {results.get('session_id')}")


if __name__ == "__main__":
    main()

