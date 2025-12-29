"""
可视化 LLM 输出过程中每个 token 的熵变化。
支持两种模式：
1. base_model: 从 outputs/base_model/{session_id}/{problem_id}/logprobs.json 读取
2. self_evolve: 从 outputs/self_evolve/{session_id}/{problem_id}/{round}/logprobs.json 读取
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import yaml

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 默认配置文件路径
DEFAULT_CONFIG_PATH = project_root / "config" / "entropy_config.yaml"


class EntropyVisualizer:
    """
    从预保存的 logprobs.json 文件计算并可视化每个 token 的熵。
    支持 base_model 和 self_evolve 两种模式。
    """

    def __init__(self, mode: str = "base_model", outputs_dir: Optional[Path] = None) -> None:
        """
        初始化熵可视化器。
        
        Args:
            mode: 运行模式，"base_model" 或 "self_evolve"
            outputs_dir: outputs 目录路径，默认根据 mode 自动设置
        """
        self.mode = mode
        if outputs_dir:
            self.outputs_dir = outputs_dir
        else:
            if mode == "self_evolve":
                self.outputs_dir = project_root / "outputs" / "self_evolve"
            else:
                self.outputs_dir = project_root / "outputs" / "base_model"

    def _calculate_entropy(self, logprobs_list: List[Dict[str, float]]) -> float:
        """
        根据 top logprobs 计算熵。
        
        H = -∑ p(x) * log(p(x))
        
        Args:
            logprobs_list: top_logprobs 列表，每个元素包含 token 和 logprob
        
        Returns:
            熵值 (以 nats 为单位)
        """
        if not logprobs_list:
            return 0.0

        # 将 logprobs 转换为概率
        logprobs = [item["logprob"] for item in logprobs_list]
        probs = [math.exp(lp) for lp in logprobs]
        
        # 归一化（因为我们只有 top-k，需要归一化）
        total_prob = sum(probs)
        if total_prob > 0:
            probs = [p / total_prob for p in probs]
        
        # 计算熵
        entropy = 0.0
        for p in probs:
            if p > 0:
                entropy -= p * math.log(p)

        return entropy

    def load_logprobs(
        self, 
        session_id: str, 
        problem_id: str,
        round_num: Optional[int] = None,
    ) -> Tuple[List[str], List[float], List[Dict]]:
        """
        从 logprobs.json 文件加载数据并计算熵。
        
        Args:
            session_id: 会话 ID
            problem_id: 问题 ID
            round_num: 轮次号（仅 self_evolve 模式需要）
        
        Returns:
            (token 列表, 熵列表, 原始 logprobs 数据)
        """
        if self.mode == "self_evolve":
            if round_num is None:
                raise ValueError("self_evolve 模式需要指定 round_num")
            logprobs_path = self.outputs_dir / session_id / problem_id / str(round_num) / "logprobs.json"
        else:
            logprobs_path = self.outputs_dir / session_id / problem_id / "logprobs.json"
        
        if not logprobs_path.exists():
            raise FileNotFoundError(f"logprobs 文件不存在: {logprobs_path}")
        
        print(f"正在加载 logprobs 文件: {logprobs_path}")
        
        with open(logprobs_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        logprobs_data = data.get("logprobs", [])
        total_tokens = data.get("total_tokens", len(logprobs_data))
        
        print(f"加载完成，共 {total_tokens} 个 token")
        
        tokens_list: List[str] = []
        entropies: List[float] = []
        
        for token_info in logprobs_data:
            token = token_info.get("token", "")
            tokens_list.append(token)
            
            # 获取 top_logprobs 用于计算熵
            top_logprobs = token_info.get("top_logprobs", [])
            
            # 如果没有 top_logprobs，使用当前 token 的 logprob
            if not top_logprobs:
                # 单个 token 的熵为 0（确定性选择）
                entropies.append(0.0)
            else:
                entropy = self._calculate_entropy(top_logprobs)
                entropies.append(entropy)
        
        return tokens_list, entropies, logprobs_data

    def visualize_entropy(
        self,
        tokens: List[str],
        entropies: List[float],
        title: str = "Token Entropy During Generation",
        save_path: Optional[str] = None,
        max_display_tokens: int = 100,
    ) -> Optional[float]:
        """
        可视化 token 熵的变化。
        
        Args:
            tokens: token 列表
            entropies: 对应的熵列表
            title: 图表标题
            save_path: 保存路径，如果为 None 则显示图表
            max_display_tokens: x 轴最多显示的 token 数量
        
        Returns:
            平均熵值，如果没有数据则返回 None
        """
        if not tokens or not entropies:
            print("没有数据可视化")
            return None

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        
        n_tokens = len(tokens)
        x = np.arange(n_tokens)
        
        # 上图：熵的折线图
        ax1.plot(x, entropies, 'b-', linewidth=1, alpha=0.7)
        ax1.fill_between(x, entropies, alpha=0.3)
        ax1.set_xlabel('Token Index', fontsize=12)
        ax1.set_ylabel('Entropy (nats)', fontsize=12)
        ax1.set_title(title, fontsize=14)
        ax1.grid(True, alpha=0.3)
        
        # 添加统计信息
        mean_entropy = np.mean(entropies)
        max_entropy = np.max(entropies)
        min_entropy = np.min(entropies)
        ax1.axhline(y=mean_entropy, color='r', linestyle='--', label=f'Mean: {mean_entropy:.3f}')
        ax1.legend(loc='upper right')
        
        # 标注高熵点
        threshold = mean_entropy + np.std(entropies)
        high_entropy_indices = [i for i, e in enumerate(entropies) if e > threshold]
        if high_entropy_indices:
            ax1.scatter(high_entropy_indices, [entropies[i] for i in high_entropy_indices], 
                       color='red', s=20, zorder=5, label='High Entropy')

        # 下图：滑动窗口平均熵
        window_size = min(20, n_tokens // 5) if n_tokens > 20 else 1
        if window_size > 1:
            smoothed = np.convolve(entropies, np.ones(window_size)/window_size, mode='valid')
            x_smoothed = np.arange(window_size//2, len(smoothed) + window_size//2)
            ax2.plot(x_smoothed, smoothed, 'g-', linewidth=2, label=f'Smoothed (window={window_size})')
            ax2.fill_between(x_smoothed, smoothed, alpha=0.3, color='green')
        else:
            ax2.plot(x, entropies, 'g-', linewidth=2)
            ax2.fill_between(x, entropies, alpha=0.3, color='green')
        
        ax2.set_xlabel('Token Index', fontsize=12)
        ax2.set_ylabel('Smoothed Entropy (nats)', fontsize=12)
        ax2.set_title('Smoothed Entropy (Moving Average)', fontsize=14)
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='upper right')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"图表已保存到: {save_path}")
        else:
            plt.show()
        
        # 关闭图表，释放内存
        plt.close(fig)
        
        # 打印统计信息
        print("\n" + "=" * 50)
        print("熵统计信息:")
        print(f"  总 token 数: {n_tokens}")
        print(f"  平均熵: {mean_entropy:.4f} nats")
        print(f"  最大熵: {max_entropy:.4f} nats (token index: {np.argmax(entropies)})")
        print(f"  最小熵: {min_entropy:.4f} nats (token index: {np.argmin(entropies)})")
        print(f"  标准差: {np.std(entropies):.4f}")
        print("=" * 50)
        
        return mean_entropy

    def visualize_summary(
        self,
        problem_ids: List[str],
        mean_entropies: List[float],
        save_path: str,
        title: str = "Mean Entropy by Problem",
    ) -> None:
        """
        绘制所有 problem 的平均熵汇总图。
        
        Args:
            problem_ids: problem ID 列表
            mean_entropies: 对应的平均熵列表
            save_path: 保存路径
            title: 图表标题
        """
        if not problem_ids or not mean_entropies:
            print("没有数据可绘制汇总图")
            return
        
        fig, ax = plt.subplots(figsize=(14, 6))
        
        x = np.arange(len(problem_ids))
        bars = ax.bar(x, mean_entropies, color='steelblue', alpha=0.8, edgecolor='navy')
        
        # 设置 x 轴标签
        ax.set_xticks(x)
        ax.set_xticklabels(problem_ids, rotation=45, ha='right', fontsize=10)
        
        ax.set_xlabel('Problem ID', fontsize=12)
        ax.set_ylabel('Mean Entropy (nats)', fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.grid(True, alpha=0.3, axis='y')
        
        # 添加平均线
        overall_mean = np.mean(mean_entropies)
        ax.axhline(y=overall_mean, color='red', linestyle='--', linewidth=2, 
                   label=f'Overall Mean: {overall_mean:.4f}')
        ax.legend(loc='upper right')
        
        # 在柱子上方显示数值
        for bar, entropy in zip(bars, mean_entropies):
            height = bar.get_height()
            ax.annotate(f'{entropy:.3f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=8, rotation=90)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n汇总图已保存到: {save_path}")
        
        # 关闭图表，释放内存
        plt.close(fig)

    def visualize_rounds_comparison(
        self,
        session_id: str,
        problem_id: str,
        rounds: List[int],
        save_path: str,
    ) -> None:
        """
        可视化同一问题不同轮次的熵对比（仅 self_evolve 模式）。
        
        Args:
            session_id: 会话 ID
            problem_id: 问题 ID
            rounds: 轮次列表
            save_path: 保存路径
        """
        if self.mode != "self_evolve":
            print("rounds_comparison 仅支持 self_evolve 模式")
            return
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        mean_entropies = []
        valid_rounds = []
        
        for round_num in rounds:
            try:
                tokens, entropies, _ = self.load_logprobs(session_id, problem_id, round_num)
                if entropies:
                    mean_entropies.append(np.mean(entropies))
                    valid_rounds.append(round_num)
            except FileNotFoundError:
                continue
        
        if not valid_rounds:
            print(f"没有找到任何轮次的 logprobs 数据: {problem_id}")
            return
        
        x = np.arange(len(valid_rounds))
        bars = ax.bar(x, mean_entropies, color='steelblue', alpha=0.8, edgecolor='navy')
        
        ax.set_xticks(x)
        ax.set_xticklabels([f'Round {r}' for r in valid_rounds])
        ax.set_xlabel('Round', fontsize=12)
        ax.set_ylabel('Mean Entropy (nats)', fontsize=12)
        ax.set_title(f'Mean Entropy by Round - Problem {problem_id}', fontsize=14)
        ax.grid(True, alpha=0.3, axis='y')
        
        for bar, entropy in zip(bars, mean_entropies):
            height = bar.get_height()
            ax.annotate(f'{entropy:.3f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"轮次对比图已保存到: {save_path}")
        plt.close(fig)


def detect_mode(input_path: str) -> str:
    """
    根据输入路径自动检测模式。
    
    Args:
        input_path: 输入路径字符串
    
    Returns:
        "base_model" 或 "self_evolve"
    """
    if "self_evolve" in input_path:
        return "self_evolve"
    elif "base_model" in input_path:
        return "base_model"
    else:
        # 默认为 base_model
        return "base_model"


def get_available_rounds(session_dir: Path, problem_id: str) -> List[int]:
    """
    获取某个问题下所有可用的轮次。
    
    Args:
        session_dir: 会话目录
        problem_id: 问题 ID
    
    Returns:
        轮次列表（已排序）
    """
    problem_dir = session_dir / problem_id
    if not problem_dir.exists():
        return []
    
    rounds = []
    for item in problem_dir.iterdir():
        if item.is_dir() and item.name.isdigit():
            rounds.append(int(item.name))
    
    return sorted(rounds)


def load_config(config_path: Optional[Path] = None) -> dict:
    """
    从 YAML 配置文件加载配置。
    
    Args:
        config_path: 配置文件路径
    
    Returns:
        配置字典
    """
    if config_path is None:
        config_path = DEFAULT_CONFIG_PATH
    
    if not config_path.exists():
        # 返回默认配置
        return {
            "input_path": str(project_root / "outputs" / "base_model"),
            "session_id": None,
            "skip_last_tokens": 1000,
        }
    
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    return config


def main():
    """主函数：遍历 session 下所有 problem 并生成熵可视化"""
    # 加载配置
    config = load_config()
    
    # 获取输入路径并自动检测模式
    input_path = config.get("input_path", str(project_root / "outputs" / "base_model"))
    mode = detect_mode(input_path)
    session_id = config.get("session_id")
    skip_last_tokens = config.get("skip_last_tokens", 1000)
    
    print("=" * 50)
    print(f"熵可视化分析")
    print(f"  模式: {mode}")
    print(f"  输入路径: {input_path}")
    print(f"  Session ID: {session_id}")
    print(f"  跳过最后 token 数: {skip_last_tokens}")
    print("=" * 50)
    print()
    
    # 设置输出目录
    if mode == "self_evolve":
        outputs_dir = project_root / "outputs" / "self_evolve"
    else:
        outputs_dir = project_root / "outputs" / "base_model"
    
    # 如果没有指定 session_id，使用输入路径中的
    if session_id is None:
        # 尝试从输入路径解析 session_id
        input_path_obj = Path(input_path)
        if input_path_obj.exists() and input_path_obj.is_dir():
            # 假设输入路径格式为 .../outputs/{mode}/{session_id}
            session_id = input_path_obj.name
        else:
            # 获取最新的 session
            session_dirs = [d for d in outputs_dir.iterdir() if d.is_dir()]
            if session_dirs:
                session_dirs.sort(key=lambda d: d.stat().st_mtime, reverse=True)
                session_id = session_dirs[0].name
            else:
                print(f"错误: 没有找到任何 session 目录: {outputs_dir}")
                return
    
    print(f"使用 Session ID: {session_id}")
    
    # 创建熵可视化器
    visualizer = EntropyVisualizer(mode=mode, outputs_dir=outputs_dir)
    
    # 获取 session 目录
    session_dir = outputs_dir / session_id
    if not session_dir.exists():
        print(f"错误: session 目录不存在: {session_dir}")
        print(f"请确保已运行相应的测试脚本生成 logprobs 数据")
        return
    
    # 遍历 session 目录下所有子文件夹（每个子文件夹是一个 problem）
    problem_dirs = [d for d in session_dir.iterdir() if d.is_dir()]
    # 过滤掉非问题目录（如可能存在的其他文件夹）
    problem_dirs = [d for d in problem_dirs if "-" in d.name]
    # 按 problem_id 排序
    problem_dirs.sort(key=lambda d: (d.name.split("-")[1], int(d.name.split("-")[-1])))
    
    if not problem_dirs:
        print(f"错误: session 目录下没有 problem 子文件夹: {session_dir}")
        return
    
    print(f"找到 {len(problem_dirs)} 个 problem 目录")
    print()
    
    # 统计信息
    success_count = 0
    skip_count = 0
    error_count = 0
    
    # 收集每个 problem 的平均熵
    problem_ids_for_summary: List[str] = []
    mean_entropies_for_summary: List[float] = []
    
    if mode == "self_evolve":
        # Self-evolve 模式：遍历每个问题的每个轮次
        for problem_dir in problem_dirs:
            problem_id = problem_dir.name
            
            # 获取该问题的所有可用轮次
            available_rounds = get_available_rounds(session_dir, problem_id)
            
            if not available_rounds:
                print(f"[跳过] {problem_id}: 没有找到任何轮次")
                skip_count += 1
                continue
            
            print(f"\n{'=' * 50}")
            print(f"处理 Problem: {problem_id} (轮次: {available_rounds})")
            print("=" * 50)
            
            # 收集每轮的平均熵用于对比
            round_mean_entropies = []
            
            for round_num in available_rounds:
                round_dir = problem_dir / str(round_num)
                logprobs_file = round_dir / "logprobs.json"
                
                # 检查熵图是否已存在，如果存在则跳过
                entropy_png_path = round_dir / f"entropy_{problem_id.replace('-', '_')}_round{round_num}.png"
                if entropy_png_path.exists():
                    print(f"  [跳过] Round {round_num}: 熵图已存在 ({entropy_png_path.name})")
                    skip_count += 1
                    continue
                
                if not logprobs_file.exists():
                    print(f"  [跳过] Round {round_num}: 没有 logprobs.json 文件")
                    continue
                
                print(f"\n  处理 Round {round_num}")
                
                try:
                    tokens, entropies, raw_data = visualizer.load_logprobs(
                        session_id=session_id,
                        problem_id=problem_id,
                        round_num=round_num,
                    )
                    
                    # 跳过最后的 token
                    if skip_last_tokens > 0 and len(tokens) > skip_last_tokens:
                        tokens = tokens[:-skip_last_tokens]
                        entropies = entropies[:-skip_last_tokens]
                    
                    if tokens and entropies:
                        save_path = str(round_dir / f"entropy_{problem_id.replace('-', '_')}_round{round_num}.png")
                        
                        mean_entropy = visualizer.visualize_entropy(
                            tokens=tokens,
                            entropies=entropies,
                            title=f"Token Entropy - Problem {problem_id} Round {round_num}",
                            save_path=save_path,
                        )
                        success_count += 1
                        
                        if mean_entropy is not None:
                            round_mean_entropies.append(mean_entropy)
                    else:
                        print(f"  [警告] Round {round_num}: 没有有效的 logprobs 数据")
                        skip_count += 1
                        
                except Exception as e:
                    print(f"  [错误] Round {round_num}: {e}")
                    error_count += 1
            
            # 如果有多个轮次，生成轮次对比图
            if len(round_mean_entropies) > 1:
                comparison_save_path = str(problem_dir / f"entropy_rounds_comparison_{problem_id.replace('-', '_')}.png")
                visualizer.visualize_rounds_comparison(
                    session_id=session_id,
                    problem_id=problem_id,
                    rounds=available_rounds,
                    save_path=comparison_save_path,
                )
            
            # 使用最后一轮的平均熵作为该问题的代表
            if round_mean_entropies:
                problem_ids_for_summary.append(problem_id)
                mean_entropies_for_summary.append(round_mean_entropies[-1])
    
    else:
        # Base model 模式：直接遍历每个问题
        for problem_dir in problem_dirs:
            problem_id = problem_dir.name
            
            # 检查熵图是否已存在，如果存在则跳过
            entropy_png_path = problem_dir / f"entropy_{problem_id.replace('-', '_')}.png"
            if entropy_png_path.exists():
                print(f"[跳过] {problem_id}: 熵图已存在 ({entropy_png_path.name})")
                skip_count += 1
                continue
            
            logprobs_file = problem_dir / "logprobs.json"
            
            if not logprobs_file.exists():
                print(f"[跳过] {problem_id}: 没有 logprobs.json 文件")
                skip_count += 1
                continue
            
            print(f"\n{'=' * 50}")
            print(f"处理 Problem: {problem_id}")
            print("=" * 50)
            
            try:
                tokens, entropies, raw_data = visualizer.load_logprobs(
                    session_id=session_id,
                    problem_id=problem_id,
                )
                
                # 跳过最后的 token
                if skip_last_tokens > 0 and len(tokens) > skip_last_tokens:
                    tokens = tokens[:-skip_last_tokens]
                    entropies = entropies[:-skip_last_tokens]

                if tokens and entropies:
                    save_path = str(problem_dir / f"entropy_{problem_id.replace('-', '_')}.png")
                    
                    mean_entropy = visualizer.visualize_entropy(
                        tokens=tokens,
                        entropies=entropies,
                        title=f"Token Entropy - Problem {problem_id}",
                        save_path=save_path,
                    )
                    success_count += 1
                    
                    if mean_entropy is not None:
                        problem_ids_for_summary.append(problem_id)
                        mean_entropies_for_summary.append(mean_entropy)
                else:
                    print(f"[警告] {problem_id}: 没有有效的 logprobs 数据")
                    skip_count += 1
                    
            except Exception as e:
                print(f"[错误] {problem_id}: {e}")
                error_count += 1
    
    # 绘制汇总图
    if problem_ids_for_summary and mean_entropies_for_summary:
        summary_save_path = str(session_dir / f"entropy_summary_{mode}.png")
        visualizer.visualize_summary(
            problem_ids=problem_ids_for_summary,
            mean_entropies=mean_entropies_for_summary,
            save_path=summary_save_path,
            title=f"Mean Entropy by Problem - Session {session_id} ({mode})",
        )
    
    # 打印总结
    print("\n" + "=" * 50)
    print("处理完成!")
    print(f"  模式: {mode}")
    print(f"  成功: {success_count}")
    print(f"  跳过: {skip_count}")
    print(f"  错误: {error_count}")
    print(f"  总计: {len(problem_dirs)}")
    print("=" * 50)


if __name__ == "__main__":
    main()
