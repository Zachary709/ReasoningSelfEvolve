#!/usr/bin/env python3
"""
测试 self-evolving solver 在 AIME 2024 问题集上的表现。
使用 qwen_math 模块提供的数学结果判断功能来评估答案正确性。
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Set
from collections import Counter
from datetime import datetime
import sys
import logging

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
from src.solver.solver_engine import SelfEvolvingSolver
from src.utils.logging_utils import configure_logging
from src.utils.visualization import plot_round_accuracy

# 默认配置文件路径
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "test_self_evolve_config.yaml"

# 测试输出目录（使用项目根目录下的 outputs/self_evolve）
TEST_OUTPUTS_DIR = PROJECT_ROOT / "outputs" / "self_evolve"


class OutputManager:
    """
    输出文件管理器，支持流式写入。
    输出到 outputs/self_evolve/{session_id}/run.out，session_id 格式为 "MM-DD_HH-MM"。
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
    
    def write_round_result(self, problem_id: str, round_num: int, is_correct: Optional[bool], score: Optional[float], 
                           solution_preview: str = "", error: str = ""):
        """写入轮次结果"""
        if error:
            self.writeln(f"  Round {round_num} 错误: {error}")
        elif is_correct is not None:
            status = "✓ 正确" if is_correct else "✗ 错误"
            self.writeln(f"  Round {round_num} 结果: {status} (score: {score})")
            if solution_preview:
                self.writeln(f"  解答预览: {solution_preview[:200]}...")
    
    def write_problem_result(self, problem_id: str, final_round: int, is_correct: Optional[bool], score: Optional[float], 
                             solution_preview: str = "", error: str = ""):
        """写入问题最终结果"""
        if error:
            self.writeln(f"  最终错误: {error}")
        elif is_correct is not None:
            status = "✓ 正确" if is_correct else "✗ 错误"
            self.writeln(f"  最终结果 (Round {final_round}): {status} (score: {score})")
            if solution_preview:
                self.writeln(f"  解答预览: {solution_preview[:300]}...")
        else:
            self.writeln(f"  警告: 没有标准答案，跳过")
        self.writeln("-" * 40)
    
    def write_summary(self, total_problems: int, total_with_answer: int, correct_count: int, accuracy: float, rounds: int):
        """写入测试总结"""
        self.writeln()
        self.writeln("=" * 80)
        self.writeln("测试完成!")
        self.writeln(f"总问题数: {total_problems}")
        self.writeln(f"有标准答案的问题数: {total_with_answer}")
        self.writeln(f"正确答案数: {correct_count}")
        self.writeln(f"准确率: {accuracy:.2%} ({correct_count}/{total_with_answer})")
        self.writeln(f"迭代轮数: {rounds}")
        self.writeln(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


class ProgressTracker:
    """
    进度跟踪器，支持题级别+轮次级别的断点重连。
    进度文件保存到 outputs/self_evolve/{session_id}/progress.json。
    路径结构：outputs/self_evolve/{session_id}/{problem_id}/{round}/...
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
        
        # 加载已完成的问题和轮次
        self.completed_problems: Set[str] = set()  # 完全完成的问题
        self.problem_rounds: Dict[str, int] = {}  # 每个问题已完成的最后一轮
        self.results: Dict[str, Dict[str, Any]] = {}
        self._load_progress()
    
    def _load_progress(self):
        """从进度文件加载已完成的问题和轮次"""
        if self.progress_file.exists():
            try:
                with open(self.progress_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.completed_problems = set(data.get("completed_problems", []))
                    self.problem_rounds = data.get("problem_rounds", {})
                    self.results = data.get("results", {})
                    print(f"已恢复进度: {len(self.completed_problems)} 个问题已完全完成")
                    if self.problem_rounds:
                        in_progress = [f"{p}(round {r})" for p, r in self.problem_rounds.items() if p not in self.completed_problems]
                        if in_progress:
                            print(f"进行中的问题: {', '.join(in_progress)}")
            except (json.JSONDecodeError, IOError) as e:
                print(f"警告: 无法加载进度文件 {self.progress_file}: {e}")
                self.completed_problems = set()
                self.problem_rounds = {}
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
        # 生成 correctness 字典：每个题目的每个 round 的答案正确性
        correctness = {}
        for problem_id, problem_data in self.results.items():
            problem_correctness = {}
            # 添加每个 round 的正确性
            if "rounds" in problem_data:
                for round_num, round_data in problem_data["rounds"].items():
                    problem_correctness[f"round_{round_num}"] = round_data.get("correct")
            # 添加最终结果的正确性
            if "final" in problem_data:
                problem_correctness["final"] = problem_data["final"].get("correct")
            correctness[problem_id] = problem_correctness
        
        # 对 completed_problems 和 problem_rounds 进行排序
        sorted_completed = self._sort_problem_ids(list(self.completed_problems))
        sorted_problem_rounds = {k: self.problem_rounds[k] for k in self._sort_problem_ids(list(self.problem_rounds.keys()))}
        sorted_correctness = {k: correctness[k] for k in self._sort_problem_ids(list(correctness.keys()))}
        sorted_results = {k: self.results[k] for k in self._sort_problem_ids(list(self.results.keys()))}
        
        data = {
            "session_id": self.session_id,
            "completed_problems": sorted_completed,
            "problem_rounds": sorted_problem_rounds,
            "correctness": sorted_correctness,
            "results": sorted_results,
            "last_update": datetime.now().isoformat(),
        }
        with open(self.progress_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
    
    def is_completed(self, problem_id: str) -> bool:
        """检查问题是否已完全完成（所有轮次）"""
        return problem_id in self.completed_problems
    
    def get_last_completed_round(self, problem_id: str) -> int:
        """获取问题已完成的最后一轮，返回 -1 表示没有完成任何轮次"""
        return self.problem_rounds.get(problem_id, -1)
    
    def mark_round_completed(self, problem_id: str, round_num: int, round_result: Dict[str, Any]):
        """
        标记问题的某一轮已完成。
        
        Args:
            problem_id: 问题 ID
            round_num: 轮次号（0 表示初始输出）
            round_result: 该轮的结果
        """
        self.problem_rounds[problem_id] = round_num
        # 更新或创建问题的结果记录
        if problem_id not in self.results:
            self.results[problem_id] = {"rounds": {}}
        self.results[problem_id]["rounds"][str(round_num)] = round_result
        self._save_progress()
    
    def mark_completed(self, problem_id: str, result: Dict[str, Any]):
        """
        标记问题已完全完成并保存最终结果。
        
        Args:
            problem_id: 问题 ID
            result: 问题的最终测试结果
        """
        self.completed_problems.add(problem_id)
        if problem_id not in self.results:
            self.results[problem_id] = {}
        self.results[problem_id]["final"] = result
        self._save_progress()
    
    def get_result(self, problem_id: str) -> Optional[Dict[str, Any]]:
        """获取已完成问题的最终结果"""
        if problem_id in self.results:
            return self.results[problem_id].get("final")
        return None
    
    def get_round_result(self, problem_id: str, round_num: int) -> Optional[Dict[str, Any]]:
        """获取问题某一轮的结果"""
        if problem_id in self.results and "rounds" in self.results[problem_id]:
            return self.results[problem_id]["rounds"].get(str(round_num))
        return None
    
    def get_stats(self) -> Dict[str, int]:
        """获取当前统计信息"""
        correct_count = sum(1 for r in self.results.values() 
                          if r.get("final", {}).get("correct") is True)
        total_with_answer = sum(1 for r in self.results.values() 
                               if r.get("final", {}).get("correct") is not None)
        return {
            "completed": len(self.completed_problems),
            "correct": correct_count,
            "total_with_answer": total_with_answer,
        }


def calculate_round_accuracy(correctness: Dict[str, Dict[str, Any]]) -> tuple:
    """
    计算每一轮的平均准确率
    
    Args:
        correctness: 正确性字典，格式为 {problem_id: {round_x: bool, ...}}
    
    Returns:
        (每轮准确率列表, 最大轮数)
    """
    # 找出每个题目的轮数和对应的正确性
    problem_rounds = {}
    for problem_id, rounds_data in correctness.items():
        # 提取 round_X 的数据（排除 final）
        round_results = {}
        for key, value in rounds_data.items():
            if key.startswith('round_'):
                round_num = int(key.split('_')[1])
                round_results[round_num] = value
        if round_results:
            problem_rounds[problem_id] = round_results
    
    if not problem_rounds:
        return [], 0
    
    # 找出最大轮数
    max_round = max(max(rounds.keys()) for rounds in problem_rounds.values())
    
    # 计算每一轮的平均准确率
    round_accuracies = []
    for round_num in range(max_round + 1):  # 从 round_0 到 round_max
        correct_count = 0
        total_count = len(problem_rounds)
        
        for problem_id, rounds_data in problem_rounds.items():
            if round_num in rounds_data:
                # 该题目有这一轮的数据
                if rounds_data[round_num]:
                    correct_count += 1
            else:
                # 该题目没有这一轮的数据，使用最后一轮的结果
                last_round = max(rounds_data.keys())
                if rounds_data[last_round]:
                    correct_count += 1
        
        accuracy = correct_count / total_count
        round_accuracies.append(accuracy)
    
    return round_accuracies, max_round


def format_token_label(tokenizer: AutoTokenizer, token: str) -> str:
    """
    格式化 token 标签用于显示。
    将特殊字符转换为可读形式。
    """
    label = token
    label = label.replace("Ġ", "_")
    label = label.replace("Ċ", "\\n")
    label = label.replace("ĉ", "\\t")
    label = label.replace("$", "\\$")
    return label


def save_solution_and_token_stats(
    problem_id: str,
    solution_text: str,
    tokenizer: AutoTokenizer,
    session_dir: Path,
    round_num: int,
) -> Dict[str, Any]:
    """
    保存 solution.txt 和 token_stats.json 到 outputs 文件夹。
    路径结构：{session_dir}/{problem_id}/{round}/...
    
    Args:
        problem_id: 问题 ID
        solution_text: 生成的解答文本
        tokenizer: 用于 tokenize 的 tokenizer
        session_dir: 会话目录路径
        round_num: 轮次号（0 表示初始输出）
    
    Returns:
        包含 token 统计信息的字典
    """
    # 创建输出目录：{session_dir}/{problem_id}/{round}
    output_dir = session_dir / problem_id / str(round_num)
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
        "round": round_num,
        "total_tokens": total_tokens_count,
        "unique_tokens": len(counts),
        "token_counts": dict(counts.most_common()),
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


def save_history(
    problem_id: str,
    history_entry: Dict[str, Any],
    session_dir: Path,
    round_num: int,
) -> Path:
    """
    保存当前轮次的求解历史到文件（只保存当前轮的输出）。
    
    Args:
        problem_id: 问题 ID
        history_entry: 当前轮次的历史记录（单个条目）
        session_dir: 会话目录路径
        round_num: 当前轮次
    
    Returns:
        保存的文件路径
    """
    output_dir = session_dir / problem_id / str(round_num)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 只保存当前轮的信息，不包含 logprobs（logprobs 单独保存）
    history_to_save = {
        "role": history_entry.get("role", ""),
        "prompt": history_entry.get("prompt", ""),
        "response": history_entry.get("response", ""),
        "solution_body": history_entry.get("solution_body", ""),
    }
    
    history_path = output_dir / "history.json"
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history_to_save, f, ensure_ascii=False, indent=2)
    
    return history_path


def save_logprobs(
    problem_id: str,
    logprobs_data: Optional[List[Dict[str, Any]]],
    session_dir: Path,
    round_num: int,
) -> Optional[Path]:
    """
    保存 logprobs 数据到 outputs 文件夹。
    
    Args:
        problem_id: 问题 ID
        logprobs_data: logprobs 数据列表
        session_dir: 会话目录路径
        round_num: 轮次号
    
    Returns:
        保存的文件路径，如果 logprobs_data 为 None 则返回 None
    """
    if logprobs_data is None:
        print(f"    警告: Round {round_num} 没有 logprobs 数据可保存")
        return None
    
    output_dir = session_dir / problem_id / str(round_num)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logprobs_path = output_dir / "logprobs.json"
    
    logprobs_output = {
        "problem_id": problem_id,
        "round": round_num,
        "total_tokens": len(logprobs_data),
        "logprobs": logprobs_data,
    }
    
    with open(logprobs_path, "w", encoding="utf-8") as f:
        json.dump(logprobs_output, f, ensure_ascii=False, indent=2)
    
    print(f"    Logprobs saved to: {logprobs_path}")
    return logprobs_path


def load_config(config_path: Optional[Path] = None) -> dict:
    """
    从 YAML 配置文件加载配置。
    
    Args:
        config_path: 配置文件路径，默认为 config/test_self_evolve_config.yaml
    
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


def test_self_evolving_solver(
    questions_dir: Path,
    questions_file: str = "aime2024_questions.txt",
    model_path: str = DEFAULT_MODEL_PATH,
    api_base: str = DEFAULT_VLLM_BASE_URL,
    max_new_tokens: int = 20000,
    max_context_length: int = 32768,
    temperature: float = 0.0,
    top_p: float = 1.0,
    rounds: int = 5,
    verification_max_new_tokens: int = 5000,
    max_report_tokens: int = 5000,
    top_logprobs: Optional[int] = 20,
    dry_run: bool = False,
    log_path: Optional[Path] = None,
    resume_session: Optional[str] = None,
    selected_problems: Optional[List[str]] = None,
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
        top_logprobs: 每个位置返回的 top logprobs 数量，为 0 或 None 则不获取 logprobs
        dry_run: 是否为dry run模式
        log_path: 日志文件路径（可选）
        resume_session: 断点重连的会话 ID
        selected_problems: 指定要解决的题目 ID 列表，为 None 则解决所有题目
    
    Returns:
        包含测试结果的字典
    """
    # 初始化输出管理器和进度跟踪器
    if resume_session:
        # 断点重连模式
        progress_tracker = ProgressTracker(session_id=resume_session)
        output_manager = OutputManager.__new__(OutputManager)
        output_manager.session_id = resume_session
        output_manager.session_dir = TEST_OUTPUTS_DIR / resume_session
        output_manager.session_dir.mkdir(parents=True, exist_ok=True)
        output_manager.filename = "run.out"
        output_manager.filepath = output_manager.session_dir / output_manager.filename
        output_manager.start_time = datetime.now()
        output_manager.write(f"\n\n# 断点恢复时间: {output_manager.start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        output_manager.writeln("=" * 80)
        print(f"断点重连模式: 恢复会话 {resume_session}")
    else:
        # 新会话模式
        output_manager = OutputManager()
        progress_tracker = ProgressTracker(session_id=output_manager.session_id)
    
    # 加载问题
    print(f"正在从 {questions_dir} 加载问题文件: {questions_file}")
    output_manager.writeln(f"正在从 {questions_dir} 加载问题文件: {questions_file}")
    all_problems = load_all_problems(questions_dir, questions_file)
    print(f"共加载 {len(all_problems)} 个问题")
    output_manager.writeln(f"共加载 {len(all_problems)} 个问题")
    
    # 根据 selected_problems 过滤问题
    if selected_problems is not None and len(selected_problems) > 0:
        selected_set = set(selected_problems)
        problems = [p for p in all_problems if p.problem_id in selected_set]
        # 检查是否有无效的问题 ID
        found_ids = {p.problem_id for p in problems}
        invalid_ids = selected_set - found_ids
        if invalid_ids:
            print(f"警告: 以下指定的问题 ID 未找到: {invalid_ids}")
            output_manager.writeln(f"警告: 以下指定的问题 ID 未找到: {invalid_ids}")
        print(f"已选择 {len(problems)} 个问题: {[p.problem_id for p in problems]}")
        output_manager.writeln(f"已选择 {len(problems)} 个问题: {[p.problem_id for p in problems]}")
        
        # 强制重新运行指定的题目：清除这些题目的进度记录和输出文件
        cleared_problems = []
        for problem_id in found_ids:
            if progress_tracker.is_completed(problem_id) or problem_id in progress_tracker.problem_rounds:
                cleared_problems.append(problem_id)
                # 从进度记录中移除
                progress_tracker.completed_problems.discard(problem_id)
                progress_tracker.problem_rounds.pop(problem_id, None)
                progress_tracker.results.pop(problem_id, None)
                # 删除对应的输出目录
                problem_output_dir = progress_tracker.session_dir / problem_id
                if problem_output_dir.exists():
                    import shutil
                    shutil.rmtree(problem_output_dir)
                    print(f"  已删除输出目录: {problem_output_dir}")
        
        if cleared_problems:
            # 保存更新后的进度
            progress_tracker._save_progress()
            print(f"已清除 {len(cleared_problems)} 个指定题目的历史记录，将重新运行: {cleared_problems}")
            output_manager.writeln(f"已清除 {len(cleared_problems)} 个指定题目的历史记录，将重新运行: {cleared_problems}")
    else:
        problems = all_problems
        print(f"将解决所有 {len(problems)} 个问题")
        output_manager.writeln(f"将解决所有 {len(problems)} 个问题")
    
    # 初始化模型
    print(f"\n初始化模型: {model_path}")
    print(f"API base: {api_base}")
    print(f"迭代轮数: {rounds}")
    output_manager.writeln(f"\n模型: {model_path}")
    output_manager.writeln(f"API base: {api_base}")
    output_manager.writeln(f"参数: temperature={temperature}, top_p={top_p}, max_new_tokens={max_new_tokens}")
    output_manager.writeln(f"迭代轮数: {rounds}")
    
    llm = LocalLLM(
        model_path=model_path,
        api_base=api_base,
        max_new_tokens=max_new_tokens,
        max_context_length=max_context_length,
        temperature=temperature,
        top_p=top_p,
        dry_run=dry_run,
    )
    
    # 配置 logger
    logger = None
    if log_path:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        logger = configure_logging(str(log_path))
    else:
        logger = logging.getLogger("test_self_evolving_solver")
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
            handler.setFormatter(formatter)
            logger.addHandler(handler)
    
    # 初始化 tokenizer
    print("初始化 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
    )
    
    # 创建 solver
    prompt_builder = PromptBuilder()
    return_logprobs = top_logprobs is not None and top_logprobs > 0
    solver = SelfEvolvingSolver(
        llm=llm,
        prompt_builder=prompt_builder,
        logger=logger,
        max_new_tokens=max_new_tokens,
        verification_max_new_tokens=verification_max_new_tokens,
        max_report_tokens=max_report_tokens,
        return_logprobs=return_logprobs,
        top_logprobs=top_logprobs if return_logprobs else 20,
    )
    
    # 测试结果
    results = []
    correct_count = 0
    total_count = 0
    
    # 从进度跟踪器恢复已有统计
    if resume_session:
        stats = progress_tracker.get_stats()
        print(f"已跳过 {stats['completed']} 个已完成的问题")
        output_manager.writeln(f"已跳过 {stats['completed']} 个已完成的问题")
    
    print("\n开始测试...")
    print("=" * 80)
    output_manager.writeln("\n开始测试...")
    output_manager.writeln("=" * 80)
    
    for idx, record in enumerate(problems, 1):
        problem_id = record.problem_id
        problem_text = record.prompt
        ground_truth = record.answer
        
        # 检查是否已完成
        if progress_tracker.is_completed(problem_id):
            cached_result = progress_tracker.get_result(problem_id)
            results.append(cached_result)
            if cached_result and cached_result.get("correct") is True:
                correct_count += 1
            if cached_result and cached_result.get("correct") is not None:
                total_count += 1
            print(f"\n[{idx}/{len(problems)}] 问题 ID: {problem_id} - 已完成，跳过")
            continue
        
        print(f"\n[{idx}/{len(problems)}] 问题 ID: {problem_id}")
        print(f"问题: {problem_text[:100]}..." if len(problem_text) > 100 else f"问题: {problem_text}")
        
        output_manager.write_problem_start(idx, len(problems), problem_id, problem_text, ground_truth)
        
        if ground_truth is None:
            print("  警告: 没有标准答案，跳过")
            result = {
                "problem_id": problem_id,
                "correct": None,
                "score": None,
                "ground_truth": None,
                "solution": None,
                "final_round": None,
            }
            results.append(result)
            progress_tracker.mark_completed(problem_id, result)
            output_manager.write_problem_result(problem_id, 0, None, None)
            continue
        
        print(f"标准答案: {ground_truth}")
        
        try:
            # 使用 solver 求解
            print(f"  正在使用 self-evolving solver 求解（{rounds} 轮迭代）...")
            solve_result = solver.solve(record, rounds=rounds)
            
            # 提取历史记录并按轮次保存
            history = solve_result.get("history", [])
            
            # 保存每一轮的结果
            # Round 0: 初始解答
            # Round 1-N: 每轮改进后的解答
            current_round = 0
            for entry in history:
                role = entry.get("role", "")
                
                if role == "solution" or role.startswith("solution_round_"):
                    solution_text = entry.get("response", "")
                    solution_body = entry.get("solution_body", "")
                    solution_logprobs = entry.get("logprobs")
                    
                    # 计算分数
                    score = compute_score(
                        data_source="aime2024",
                        solution_str=solution_text,
                        ground_truth=ground_truth,
                    )
                    is_correct = score > 0.5
                    
                    # 保存这一轮的结果（solution_text 包含完整输出，含 think 内容）
                    save_solution_and_token_stats(
                        problem_id=problem_id,
                        solution_text=solution_text,
                        tokenizer=tokenizer,
                        session_dir=progress_tracker.session_dir,
                        round_num=current_round,
                    )
                    
                    # 保存历史（只保存当前轮的输出）
                    save_history(
                        problem_id=problem_id,
                        history_entry=entry,
                        session_dir=progress_tracker.session_dir,
                        round_num=current_round,
                    )
                    
                    # 保存 logprobs（仅 solver 的，不包含 verifier）
                    save_logprobs(
                        problem_id=problem_id,
                        logprobs_data=solution_logprobs,
                        session_dir=progress_tracker.session_dir,
                        round_num=current_round,
                    )
                    
                    round_result = {
                        "round": current_round,
                        "correct": is_correct,
                        "score": score,
                        "solution_preview": solution_body[:500] if solution_body else "",
                    }
                    
                    progress_tracker.mark_round_completed(problem_id, current_round, round_result)
                    output_manager.write_round_result(problem_id, current_round, is_correct, score, 
                                                      solution_body[:200] if solution_body else "")
                    
                    print(f"    Round {current_round}: {'✓ 正确' if is_correct else '✗ 错误'} (score: {score})")
                    
                    current_round += 1
            
            # 最终结果
            final_solution = solve_result.get("final_solution", "")
            final_score = compute_score(
                data_source="aime2024",
                solution_str=final_solution,
                ground_truth=ground_truth,
            )
            final_is_correct = final_score > 0.5
            final_round = current_round - 1 if current_round > 0 else 0
            
            if final_is_correct:
                correct_count += 1
            total_count += 1
            
            print(f"  最终结果 (Round {final_round}): {'✓ 正确' if final_is_correct else '✗ 错误'} (score: {final_score})")
            
            result = {
                "problem_id": problem_id,
                "correct": final_is_correct,
                "score": final_score,
                "ground_truth": ground_truth,
                "solution": final_solution[:500] + "..." if len(final_solution) > 500 else final_solution,
                "final_round": final_round,
                "boxed_answer": solve_result.get("boxed_answer"),
            }
            results.append(result)
            
            output_manager.write_problem_result(problem_id, final_round, final_is_correct, final_score, 
                                                final_solution[:200])
            progress_tracker.mark_completed(problem_id, result)
            
        except Exception as e:
            print(f"  错误: {e}")
            import traceback
            traceback.print_exc()
            result = {
                "problem_id": problem_id,
                "correct": False,
                "score": 0.0,
                "ground_truth": ground_truth,
                "solution": f"Error: {str(e)}",
                "final_round": -1,
            }
            results.append(result)
            total_count += 1
            
            output_manager.write_problem_result(problem_id, -1, False, 0.0, error=str(e))
            progress_tracker.mark_completed(problem_id, result)
    
    # 统计结果
    accuracy = correct_count / total_count if total_count > 0 else 0.0
    
    print("\n" + "=" * 80)
    print("\n测试完成!")
    print(f"总问题数: {len(problems)}")
    print(f"有标准答案的问题数: {total_count}")
    print(f"正确答案数: {correct_count}")
    print(f"准确率: {accuracy:.2%} ({correct_count}/{total_count})")
    
    output_manager.write_summary(len(problems), total_count, correct_count, accuracy, rounds)
    
    # 从 progress_tracker 获取 correctness 数据并画图
    # 重新加载进度文件以获取完整的 correctness 数据
    if progress_tracker.progress_file.exists():
        with open(progress_tracker.progress_file, "r", encoding="utf-8") as f:
            progress_data = json.load(f)
            correctness = progress_data.get("correctness", {})
            
            if correctness:
                round_accuracies, max_round = calculate_round_accuracy(correctness)
                if round_accuracies:
                    output_path = progress_tracker.session_dir / "round_accuracy.png"
                    plot_round_accuracy(round_accuracies, progress_tracker.session_id, str(output_path))
                    print(f"\n每轮准确率图已保存到: {output_path}")
                    print("每轮准确率:")
                    for i, acc in enumerate(round_accuracies):
                        print(f"  Round {i}: {acc:.2%}")
    
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
    max_new_tokens = config.get("max_new_tokens", 20000)
    max_context_length = config.get("max_context_length", 32768)
    temperature = config.get("temperature", 0.0)
    top_p = config.get("top_p", 1.0)
    rounds = config.get("rounds", 5)
    verification_max_new_tokens = config.get("verification_max_new_tokens", 5000)
    max_report_tokens = config.get("max_report_tokens", 5000)
    top_logprobs = config.get("top_logprobs", 20)
    # 处理 top_logprobs 为 null/None/0 的情况
    if top_logprobs is None or top_logprobs == 0:
        top_logprobs = None
    dry_run = config.get("dry_run", False)
    log_path_str = config.get("log_path")
    log_path = Path(log_path_str) if log_path_str else None
    resume_session = config.get("resume", None)
    if resume_session is not None and str(resume_session).lower() == "none":
        resume_session = None
    selected_problems = config.get("selected_problems", None)
    # 处理 selected_problems 为 null/None 或空列表的情况
    if selected_problems is None or (isinstance(selected_problems, list) and len(selected_problems) == 0):
        selected_problems = None
    
    print(f"配置参数:")
    print(f"  questions_dir: {questions_dir}")
    print(f"  questions_file: {questions_file}")
    print(f"  model: {model_path}")
    print(f"  api_base: {api_base}")
    print(f"  max_new_tokens: {max_new_tokens}")
    print(f"  max_context_length: {max_context_length}")
    print(f"  temperature: {temperature}")
    print(f"  top_p: {top_p}")
    print(f"  rounds: {rounds}")
    print(f"  verification_max_new_tokens: {verification_max_new_tokens}")
    print(f"  max_report_tokens: {max_report_tokens}")
    print(f"  top_logprobs: {top_logprobs}")
    print(f"  dry_run: {dry_run}")
    print(f"  log_path: {log_path}")
    print(f"  resume: {resume_session}")
    print(f"  selected_problems: {selected_problems}")
    
    # 运行测试
    results = test_self_evolving_solver(
        questions_dir=questions_dir,
        questions_file=questions_file,
        model_path=model_path,
        api_base=api_base,
        max_new_tokens=max_new_tokens,
        max_context_length=max_context_length,
        temperature=temperature,
        top_p=top_p,
        rounds=rounds,
        verification_max_new_tokens=verification_max_new_tokens,
        max_report_tokens=max_report_tokens,
        top_logprobs=top_logprobs,
        dry_run=dry_run,
        log_path=log_path,
        resume_session=resume_session,
        selected_problems=selected_problems,
    )
    
    # 输出会话 ID
    print(f"\n会话 ID: {results.get('session_id')}")
    print(f"如需断点重连，请在配置文件中设置: resume: {results.get('session_id')}")


if __name__ == "__main__":
    main()

