"""可视化相关工具函数"""
from __future__ import annotations

import os
from collections import Counter
from typing import List, Tuple

import matplotlib.pyplot as plt
from transformers import AutoTokenizer


def format_token_label(tokenizer: AutoTokenizer, token: str) -> str:
    """Render tokenizer-specific tokens into a human-readable label."""
    try:
        readable = tokenizer.convert_tokens_to_string([token])
    except Exception:
        readable = token

    readable = readable.replace("\r", "\\r").replace("\n", "\\n").replace("\t", "\\t")
    
    # 转义 $ 符号，避免 matplotlib 把它当作 LaTeX 公式解析
    readable = readable.replace("$", r"\$")

    if readable == "":
        return repr(token)

    if readable.strip() == "":
        # Token is purely whitespace; use underscore to represent spaces (ASCII safe)
        return "_" * len(readable)

    leading_spaces = len(readable) - len(readable.lstrip(" "))
    trailing_spaces = len(readable) - len(readable.rstrip(" "))
    core = readable.strip(" ")
    # 用下划线表示空格，避免字体不支持的 Unicode 符号
    label = f"{'_' * leading_spaces}{core}{'_' * trailing_spaces}"
    return label


def plot_token_distribution(
    token_counts: Counter,
    tokenizer: AutoTokenizer,
    output_dir: str,
    problem_id: str,
) -> str:
    """
    绘制所有 token 频率分布图（对数刻度 Y 轴）。
    
    Args:
        token_counts: token 计数器
        tokenizer: 用于格式化 token 标签的 tokenizer
        output_dir: 输出目录
        problem_id: 问题 ID，用于图表标题
    
    Returns:
        保存的图片路径
    """
    # 获取所有 token，按频率降序排列
    all_items = token_counts.most_common()
    labels = [format_token_label(tokenizer, item[0]) for item in all_items]
    values = [item[1] for item in all_items]
    
    if not values:
        return ""
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 根据 token 数量动态调整图表宽度
    fig_width = max(20, len(labels) * 0.3)  # 每个 token 大约 0.3 英寸宽
    fig, ax = plt.subplots(figsize=(fig_width, 8))
    ax.bar(range(len(values)), values, color="#4C72B0")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90, ha="center", fontsize=6)  # 垂直旋转，字体缩小
    ax.set_yscale('log')  # 使用对数刻度
    ax.set_ylabel("Count (log scale)")
    ax.set_title(f"All token frequencies - Log Scale ({problem_id})")
    fig.tight_layout()
    
    output_path = os.path.join(output_dir, "token_distribution.png")
    fig.savefig(output_path)
    plt.close(fig)
    
    return output_path


def plot_log_binned_tokens(
    token_counts: Counter,
    tokenizer: AutoTokenizer,
    output_dir: str,
    problem_id: str,
) -> str:
    """
    绘制对数分箱的 token 频率图。
    第1个柱子：第1多的token，第2个柱子：第2-3多，第3个柱子：第4-7多，以此类推。
    
    Args:
        token_counts: token 计数器
        tokenizer: 用于格式化 token 标签的 tokenizer（保留接口一致性）
        output_dir: 输出目录
        problem_id: 问题 ID，用于图表标题
    
    Returns:
        保存的图片路径
    """
    # 获取所有 token 的值，按频率降序排列
    all_items = token_counts.most_common()
    values = [item[1] for item in all_items]
    
    if not values:
        return ""
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    bin_labels = []
    bin_values = []
    
    idx = 0
    bin_num = 0
    while idx < len(values):
        # 每个区间的大小是 2^bin_num
        bin_size = 2 ** bin_num
        start_idx = idx
        end_idx = min(idx + bin_size, len(values))
        
        # 计算该区间内所有 token 的总出现次数
        bin_sum = sum(values[start_idx:end_idx])
        bin_values.append(bin_sum)
        
        # 生成区间标签（使用排名，从1开始）
        start_rank = start_idx + 1
        end_rank = end_idx
        if start_rank == end_rank:
            bin_labels.append(f"#{start_rank}")
        else:
            bin_labels.append(f"#{start_rank}-{end_rank}")
        
        idx = end_idx
        bin_num += 1
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(range(len(bin_values)), bin_values, color="#E07B39")
    ax.set_xticks(range(len(bin_labels)))
    ax.set_xticklabels(bin_labels, rotation=45, ha="right", fontsize=8)
    ax.set_xlabel("Token rank range (log-binned)")
    ax.set_ylabel("Total count in bin")
    ax.set_title(f"Token frequencies by log-binned rank ({problem_id})")
    fig.tight_layout()
    
    output_path = os.path.join(output_dir, "token_distribution_log_binned.png")
    fig.savefig(output_path)
    plt.close(fig)
    
    return output_path


def plot_round_accuracy(
    round_accuracies: List[float],
    session_id: str,
    output_path: str,
) -> str:
    """
    绘制准确率-轮数图
    
    Args:
        round_accuracies: 每轮的准确率列表
        session_id: 会话ID
        output_path: 输出文件路径
    
    Returns:
        保存的图片路径
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    rounds = list(range(len(round_accuracies)))
    
    # 绘制折线图
    ax.plot(rounds, round_accuracies, marker='o', linewidth=2, markersize=8, 
            color='#2E86AB', markerfacecolor='#A23B72', markeredgecolor='#2E86AB')
    
    # 在每个点上标注准确率值
    for i, acc in enumerate(round_accuracies):
        ax.annotate(f'{acc:.1%}', (i, acc), textcoords="offset points", 
                    xytext=(0, 10), ha='center', fontsize=10)
    
    # 设置坐标轴
    ax.set_xlabel('Round', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title(f'Self-Evolve Round Accuracy\n(Session: {session_id})', fontsize=14)
    
    # 设置 x 轴刻度
    ax.set_xticks(rounds)
    ax.set_xticklabels([f'Round {i}' for i in rounds])
    
    # 设置 y 轴范围和格式
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    
    # 添加网格
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # 调整布局
    fig.tight_layout()
    
    # 保存图片
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    return output_path

