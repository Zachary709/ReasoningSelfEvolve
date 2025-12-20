"""
可视化 LLM 输出过程中每个 token 的熵变化。
从 outputs/{session_id}/{problem_id}/logprobs.json 读取预先保存的 logprobs 数据。
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class EntropyVisualizer:
    """
    从预保存的 logprobs.json 文件计算并可视化每个 token 的熵。
    """

    def __init__(self) -> None:
        """
        初始化熵可视化器。
        
        Args:
            outputs_dir: outputs 目录路径，默认为项目根目录下的 outputs/
        """
        self.outputs_dir = project_root / "outputs"

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
        
        # entropy = entropy / max(probs)

        return entropy

    def load_logprobs(
        self, 
        session_id: str, 
        problem_id: str
    ) -> Tuple[List[str], List[float], List[Dict]]:
        """
        从 logprobs.json 文件加载数据并计算熵。
        
        Args:
            session_id: 会话 ID
            problem_id: 问题 ID
        
        Returns:
            (token 列表, 熵列表, 原始 logprobs 数据)
        """
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
        
        # 显示高熵 token
        # if high_entropy_indices:
        #     print("\n高熵 token (熵 > mean + std):")
        #     for idx in high_entropy_indices[:10]:  # 最多显示 10 个
        #         token = tokens[idx]
        #         entropy = entropies[idx]
        #         # 处理特殊字符显示
        #         display_token = repr(token) if token.strip() == '' else token
        #         print(f"  [{idx}] {display_token}: {entropy:.4f}")
        #     if len(high_entropy_indices) > 10:
        #         print(f"  ... 还有 {len(high_entropy_indices) - 10} 个高熵 token")
        
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
        
        # 打印统计信息
        # print("\n" + "=" * 50)
        # print("汇总统计信息:")
        # print(f"  问题数量: {len(problem_ids)}")
        # print(f"  总体平均熵: {overall_mean:.4f} nats")
        # print(f"  最高平均熵: {np.max(mean_entropies):.4f} nats ({problem_ids[np.argmax(mean_entropies)]})")
        # print(f"  最低平均熵: {np.min(mean_entropies):.4f} nats ({problem_ids[np.argmin(mean_entropies)]})")
        # print(f"  标准差: {np.std(mean_entropies):.4f}")
        # print("=" * 50)


def main():
    """主函数：遍历 session 下所有 problem 并生成熵可视化"""
    # 使用默认配置
    session_id = "12-20_21-30"
    
    print("=" * 50)
    print(f"熵可视化分析")
    print(f"  Session ID: {session_id}")
    print("=" * 50)
    print()
    
    # 创建熵可视化器
    visualizer = EntropyVisualizer()
    
    # 获取 session 目录
    session_dir = visualizer.outputs_dir / session_id
    if not session_dir.exists():
        print(f"错误: session 目录不存在: {session_dir}")
        print(f"请确保已运行 test_base_model.py 生成 logprobs 数据")
        return
    
    # 遍历 session 目录下所有子文件夹（每个子文件夹是一个 problem）
    problem_dirs = [d for d in session_dir.iterdir() if d.is_dir()]
    # 按 problem_id 排序：先按中间部分（如 I, II），再按最后的数字
    # 例如 2024-I-1, 2024-I-2, ..., 2024-I-15, 2024-II-1, ...
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
    
    for problem_dir in problem_dirs:
        problem_id = problem_dir.name
        logprobs_file = problem_dir / "logprobs.json"
        
        # 检查是否存在 logprobs.json
        if not logprobs_file.exists():
            print(f"[跳过] {problem_id}: 没有 logprobs.json 文件")
            skip_count += 1
            continue
        
        print(f"\n{'=' * 50}")
        print(f"处理 Problem: {problem_id}")
        print("=" * 50)
        
        try:
            # 加载 logprobs 并计算熵
            tokens, entropies, raw_data = visualizer.load_logprobs(
                session_id=session_id,
                problem_id=problem_id,
            )
            
            tokens = tokens[:-1000]
            entropies = entropies[:-1000]

            # 可视化
            if tokens and entropies:
                save_path = str(problem_dir / f"entropy_{problem_id.replace('-', '_')}.png")
                
                mean_entropy = visualizer.visualize_entropy(
                    tokens=tokens,
                    entropies=entropies,
                    title=f"Token Entropy - Problem {problem_id}",
                    save_path=save_path,
                )
                success_count += 1
                
                # 收集平均熵用于汇总图
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
        summary_save_path = str(session_dir / "entropy_summary.png")
        visualizer.visualize_summary(
            problem_ids=problem_ids_for_summary,
            mean_entropies=mean_entropies_for_summary,
            save_path=summary_save_path,
            title=f"Mean Entropy by Problem - Session {session_id}",
        )
    
    # 打印总结
    print("\n" + "=" * 50)
    print("处理完成!")
    print(f"  成功: {success_count}")
    print(f"  跳过: {skip_count}")
    print(f"  错误: {error_count}")
    print(f"  总计: {len(problem_dirs)}")
    print("=" * 50)


if __name__ == "__main__":
    main()
