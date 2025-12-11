#!/usr/bin/env python3
"""
可视化测试结果的工具模块。
用于从 outputs 目录读取数据，生成 token 分布图并保存到 images 目录。
"""

import os
import json
from collections import Counter
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Any
import sys

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.data_loader import load_problem
from src.utils.qwen_math import compute_score


def load_token_stats(
    problem_id: str,
    project_root: Optional[Path] = None,
) -> Optional[dict]:
    """
    从 outputs 文件夹加载单个问题的 token 统计信息。
    
    Args:
        problem_id: 问题 ID
        project_root: 项目根目录，默认为 PROJECT_ROOT
    
    Returns:
        包含 token 统计信息的字典，如果文件不存在则返回 None
    """
    if project_root is None:
        project_root = PROJECT_ROOT
    
    # 直接从 outputs 文件夹读取已保存的 token 统计数据
    token_stats_path = os.path.join(project_root, "outputs", problem_id, "token_stats.json")
    
    if not os.path.exists(token_stats_path):
        print(f"  Token stats file not found: {token_stats_path}")
        return None
    
    with open(token_stats_path, "r", encoding="utf-8") as f:
        token_stats = json.load(f)
    
    return token_stats


def visualize_all_from_outputs(
    project_root: Optional[Path] = None,
) -> None:
    """
    从 outputs 文件夹读取所有已保存的 token 统计数据。
    
    Args:
        project_root: 项目根目录
    """
    if project_root is None:
        project_root = PROJECT_ROOT
    
    outputs_dir = os.path.join(project_root, "outputs")
    
    if not os.path.exists(outputs_dir):
        print(f"Outputs directory not found: {outputs_dir}")
        return
    
    # 获取所有问题 ID（子文件夹名）
    problem_ids = [
        d for d in os.listdir(outputs_dir)
        if os.path.isdir(os.path.join(outputs_dir, d))
    ]
    
    if not problem_ids:
        print("No problem outputs found.")
        return
    
    # 排序
    problem_ids.sort(key=lambda x: (x.split("-")[1], int(x.split("-")[-1])))
    
    print(f"Found {len(problem_ids)} problems in outputs directory.")
    
    for idx, problem_id in enumerate(problem_ids, 1):
        print(f"\n[{idx}/{len(problem_ids)}] Processing {problem_id}...")
        load_token_stats(problem_id, project_root)
    
    print("\n" + "=" * 60)
    print("All visualizations completed!")
    print("=" * 60)


def get_log_binned_data(token_counts: Counter) -> Tuple[List[str], List[int]]:
    """
    计算对数分箱的 token 频率数据。
    第1个柱子：第1多的token，第2个柱子：第2-3多，第3个柱子：第4-7多，以此类推。
    
    Args:
        token_counts: token 计数器
    
    Returns:
        (bin_labels, bin_values) 元组
    """
    all_items = token_counts.most_common()
    values = [item[1] for item in all_items]
    
    if not values:
        return [], []
    
    bin_labels = []
    bin_values = []
    
    idx = 0
    bin_num = 0
    while idx < len(values):
        bin_size = 2 ** bin_num
        start_idx = idx
        end_idx = min(idx + bin_size, len(values))
        
        bin_sum = sum(values[start_idx:end_idx])
        bin_values.append(bin_sum)
        
        start_rank = start_idx + 1
        end_rank = end_idx
        if start_rank == end_rank:
            bin_labels.append(f"#{start_rank}")
        else:
            bin_labels.append(f"#{start_rank}-{end_rank}")
        
        idx = end_idx
        bin_num += 1
    
    return bin_labels, bin_values


def collect_html_token_stats(
    project_root: Optional[Path] = None,
) -> List[Dict]:
    """
    获取 token 统计信息，便于 html 绘图。
    直接读取已保存的 token_stats.json 文件，不再重新计算 token。
    
    Args:
        project_root: 项目根目录
    
    Returns:
        包含所有问题 token 统计信息的列表
    """
    if project_root is None:
        project_root = PROJECT_ROOT
    
    outputs_dir = os.path.join(project_root, "outputs")
    questions_dir = Path(project_root) / "questions"
    
    if not os.path.exists(outputs_dir):
        print(f"Outputs directory not found: {outputs_dir}")
        return []
    
    # 获取所有问题 ID（子文件夹名）
    problem_ids = [
        d for d in os.listdir(outputs_dir)
        if os.path.isdir(os.path.join(outputs_dir, d))
    ]
    
    if not problem_ids:
        print("No problem outputs found.")
        return []
    
    # 排序
    problem_ids.sort(key=lambda x: (x.split("-")[1], int(x.split("-")[-1])))
    
    print(f"Found {len(problem_ids)} problems in outputs directory.")
    
    all_stats = []
    
    for idx, problem_id in enumerate(problem_ids, 1):
        # 使用 load_token_stats 加载 token 统计数据
        token_data = load_token_stats(problem_id, project_root)
        
        if token_data is None:
            print(f"  [{idx}/{len(problem_ids)}] Skipping {problem_id} - token_stats.json not found")
            continue
        
        # 从 token_counts 构建 Counter 用于计算 bin 数据
        token_counts = Counter(token_data.get("token_counts", {}))
        bin_labels, bin_values = get_log_binned_data(token_counts)
        
        # 读取 solution 文本用于判断正确性和获取长度
        solution_path = os.path.join(project_root, "outputs", problem_id, "solution.txt")
        solution_text = ""
        if os.path.exists(solution_path):
            with open(solution_path, "r", encoding="utf-8") as f:
                solution_text = f.read()

        # 读取标准答案并判定正确性
        is_correct: Optional[bool] = None
        try:
            record = load_problem(questions_dir, problem_id)
            if record.answer is not None:
                score = compute_score("aime", solution_text, record.answer)
                is_correct = bool(score == 1.0)
            else:
                print(f"    No reference answer found for {problem_id}, skip correctness check.")
        except Exception as e:
            print(f"    Failed to load answer for {problem_id}: {e}")

        stats = {
            "problem_id": problem_id,
            "total_tokens": token_data.get("total_tokens", 0),
            "unique_tokens": token_data.get("unique_tokens", 0),
            "bin_labels": bin_labels,
            "bin_values": bin_values,
            "solution_length": len(solution_text),
            "is_correct": is_correct,
        }
        all_stats.append(stats)
    
    return all_stats


def generate_interactive_html(
    all_stats: List[Dict],
    output_dir: str,
    filename: str = "token_distribution_interactive.html",
) -> str:
    """
    生成交互式 HTML 页面，包含所有30个问题的对数分箱图。
    实现鼠标悬停高亮功能。
    
    Args:
        all_stats: 所有问题的统计信息列表
        output_dir: 输出目录
        filename: 输出文件名
    
    Returns:
        保存的 HTML 文件路径
    """
    if not all_stats:
        return ""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 生成颜色
    n_problems = len(all_stats)
    colors = [mcolors.to_hex(plt.cm.tab20(i / n_problems)) for i in range(n_problems)]
    
    # 准备 JavaScript 数据
    js_data = []
    for i, stats in enumerate(all_stats):
        js_data.append({
            "problem_id": stats["problem_id"],
            "total_tokens": stats["total_tokens"],
            "unique_tokens": stats["unique_tokens"],
            "solution_length": stats["solution_length"],
            "bin_labels": stats["bin_labels"],
            "bin_values": stats["bin_values"],
            "color": colors[i],
            "is_correct": stats.get("is_correct"),  # 答案正确性
        })
    
    html_content = f'''<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Token Distribution - Interactive Visualization</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Segoe UI', 'PingFang SC', 'Microsoft YaHei', sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
            min-height: 100vh;
            color: #e8e8e8;
        }}
        
        .container {{
            max-width: 1800px;
            margin: 0 auto;
            padding: 20px;
        }}
        
        h1 {{
            text-align: center;
            margin-bottom: 30px;
            font-size: 2.2em;
            background: linear-gradient(90deg, #00d4ff, #7b2cbf);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            text-shadow: 0 0 30px rgba(0, 212, 255, 0.3);
        }}
        
        .layout {{
            display: grid;
            grid-template-columns: 280px 1fr;
            gap: 25px;
        }}
        
        .sidebar {{
            background: rgba(255, 255, 255, 0.05);
            border-radius: 16px;
            padding: 20px;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.1);
            max-height: 85vh;
            overflow-y: auto;
        }}
        
        .sidebar h2 {{
            font-size: 1.2em;
            margin-bottom: 15px;
            color: #00d4ff;
            border-bottom: 2px solid #00d4ff;
            padding-bottom: 10px;
        }}
        
        .problem-item {{
            padding: 12px 15px;
            margin: 8px 0;
            border-radius: 10px;
            cursor: pointer;
            transition: all 0.3s ease;
            border-left: 4px solid transparent;
            background: rgba(255, 255, 255, 0.03);
        }}
        
        .problem-item:hover {{
            background: rgba(255, 255, 255, 0.12);
            transform: translateX(5px);
            box-shadow: 0 4px 15px rgba(0, 212, 255, 0.2);
        }}
        
        .problem-item.active {{
            background: rgba(0, 212, 255, 0.2);
            border-left-color: #00d4ff;
            box-shadow: 0 0 20px rgba(0, 212, 255, 0.3);
        }}
        
        .problem-name {{
            font-weight: 600;
            font-size: 1em;
            margin-bottom: 5px;
        }}
        
        .problem-stats {{
            font-size: 0.75em;
            color: #aaa;
        }}
        
        .main-content {{
            display: flex;
            flex-direction: column;
            gap: 25px;
        }}
        
        .chart-container {{
            background: rgba(255, 255, 255, 0.05);
            border-radius: 16px;
            padding: 25px;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.1);
        }}
        
        .chart-title {{
            font-size: 1.3em;
            margin-bottom: 15px;
            color: #00d4ff;
        }}
        
        .single-chart {{
            height: 350px;
        }}
        
        .all-charts-grid {{
            display: grid;
            grid-template-columns: repeat(6, 1fr);
            gap: 15px;
        }}
        
        .mini-chart-wrapper {{
            background: rgba(255, 255, 255, 0.03);
            border-radius: 12px;
            padding: 10px;
            cursor: pointer;
            transition: all 0.3s ease;
            border: 2px solid transparent;
        }}
        
        .mini-chart-wrapper:hover {{
            background: rgba(255, 255, 255, 0.08);
            transform: scale(1.02);
        }}
        
        .mini-chart-wrapper.highlighted {{
            border-color: #00d4ff;
            box-shadow: 0 0 25px rgba(0, 212, 255, 0.4);
            background: rgba(0, 212, 255, 0.1);
        }}
        
        .mini-chart-title {{
            font-size: 0.85em;
            font-weight: 600;
            margin-bottom: 8px;
            text-align: center;
            color: #ddd;
        }}
        
        .mini-chart {{
            height: 120px;
        }}
        
        .info-panel {{
            background: rgba(255, 255, 255, 0.05);
            border-radius: 16px;
            padding: 20px;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.1);
        }}
        
        .info-grid {{
            display: grid;
            grid-template-columns: repeat(5, 1fr);
            gap: 15px;
        }}
        
        .info-card {{
            background: rgba(0, 212, 255, 0.1);
            border-radius: 10px;
            padding: 15px;
            text-align: center;
        }}
        
        .info-value {{
            font-size: 1.8em;
            font-weight: 700;
            color: #00d4ff;
        }}
        
        .info-label {{
            font-size: 0.85em;
            color: #aaa;
            margin-top: 5px;
        }}
        
        .correct {{
            color: #00ff88 !important;
        }}
        
        .incorrect {{
            color: #ff4757 !important;
        }}
        
        .unknown {{
            color: #aaa !important;
        }}
        
        .correctness-badge {{
            display: inline-block;
            padding: 2px 8px;
            border-radius: 12px;
            font-size: 0.7em;
            font-weight: 600;
            margin-left: 8px;
            vertical-align: middle;
        }}
        
        .correctness-badge.correct {{
            background: rgba(0, 255, 136, 0.2);
            border: 1px solid #00ff88;
        }}
        
        .correctness-badge.incorrect {{
            background: rgba(255, 71, 87, 0.2);
            border: 1px solid #ff4757;
        }}
        
        .correctness-badge.unknown {{
            background: rgba(170, 170, 170, 0.2);
            border: 1px solid #aaa;
        }}
        
        .mini-chart-wrapper.correct-border {{
            border-color: #00ff88;
        }}
        
        .mini-chart-wrapper.incorrect-border {{
            border-color: #ff4757;
        }}
        
        .correctness-indicator {{
            text-align: center;
            font-size: 0.75em;
            margin-top: 5px;
            font-weight: 600;
        }}
        
        ::-webkit-scrollbar {{
            width: 8px;
        }}
        
        ::-webkit-scrollbar-track {{
            background: rgba(255, 255, 255, 0.05);
            border-radius: 4px;
        }}
        
        ::-webkit-scrollbar-thumb {{
            background: rgba(0, 212, 255, 0.5);
            border-radius: 4px;
        }}
        
        ::-webkit-scrollbar-thumb:hover {{
            background: rgba(0, 212, 255, 0.7);
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🔬 Token Distribution Analysis</h1>
        
        <div class="layout">
            <div class="sidebar">
                <h2>📋 问题列表</h2>
                <div id="problem-list"></div>
            </div>
            
            <div class="main-content">
                <div class="info-panel">
                    <div class="info-grid">
                        <div class="info-card">
                            <div class="info-value" id="current-problem">-</div>
                            <div class="info-label">当前问题</div>
                        </div>
                        <div class="info-card">
                            <div class="info-value" id="total-tokens">-</div>
                            <div class="info-label">总 Token 数</div>
                        </div>
                        <div class="info-card">
                            <div class="info-value" id="unique-tokens">-</div>
                            <div class="info-label">唯一 Token 数</div>
                        </div>
                        <div class="info-card">
                            <div class="info-value" id="solution-length">-</div>
                            <div class="info-label">解答长度 (字符)</div>
                        </div>
                        <div class="info-card" id="correctness-card">
                            <div class="info-value" id="is-correct">-</div>
                            <div class="info-label">答案正确性</div>
                        </div>
                    </div>
                </div>
                
                <div class="chart-container">
                    <div class="chart-title">📊 选中问题的对数分箱图</div>
                    <div class="single-chart">
                        <canvas id="mainChart"></canvas>
                    </div>
                </div>
                
                <div class="chart-container">
                    <div class="chart-title">🗂️ 所有问题概览 (点击或悬停查看详情)</div>
                    <div class="all-charts-grid" id="all-charts-grid"></div>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        const problemData = {json.dumps(js_data, ensure_ascii=False)};
        
        let mainChart = null;
        let miniCharts = [];
        let currentIndex = 0;
        
        function formatNumber(num) {{
            if (num >= 1000) {{
                return (num / 1000).toFixed(1) + 'K';
            }}
            return num.toString();
        }}
        
        function getCorrectnessInfo(isCorrect) {{
            if (isCorrect === true) {{
                return {{ text: '✓ 正确', className: 'correct' }};
            }} else if (isCorrect === false) {{
                return {{ text: '✗ 错误', className: 'incorrect' }};
            }} else {{
                return {{ text: '? 未知', className: 'unknown' }};
            }}
        }}
        
        function updateInfoPanel(data) {{
            document.getElementById('current-problem').textContent = data.problem_id;
            document.getElementById('total-tokens').textContent = formatNumber(data.total_tokens);
            document.getElementById('unique-tokens').textContent = formatNumber(data.unique_tokens);
            document.getElementById('solution-length').textContent = formatNumber(data.solution_length);
            
            const correctnessInfo = getCorrectnessInfo(data.is_correct);
            const correctnessEl = document.getElementById('is-correct');
            correctnessEl.textContent = correctnessInfo.text;
            correctnessEl.className = 'info-value ' + correctnessInfo.className;
            
            // 更新卡片背景颜色
            const card = document.getElementById('correctness-card');
            card.style.background = data.is_correct === true ? 'rgba(0, 255, 136, 0.15)' : 
                                    data.is_correct === false ? 'rgba(255, 71, 87, 0.15)' : 
                                    'rgba(0, 212, 255, 0.1)';
        }}
        
        function createMainChart(data) {{
            const ctx = document.getElementById('mainChart').getContext('2d');
            
            if (mainChart) {{
                mainChart.destroy();
            }}
            
            mainChart = new Chart(ctx, {{
                type: 'bar',
                data: {{
                    labels: data.bin_labels,
                    datasets: [{{
                        label: 'Token Count',
                        data: data.bin_values,
                        backgroundColor: data.color + 'CC',
                        borderColor: data.color,
                        borderWidth: 2,
                        borderRadius: 6,
                    }}]
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {{
                        legend: {{
                            display: false
                        }},
                        tooltip: {{
                            backgroundColor: 'rgba(0, 0, 0, 0.8)',
                            titleColor: '#00d4ff',
                            bodyColor: '#fff',
                            padding: 12,
                            cornerRadius: 8,
                        }}
                    }},
                    scales: {{
                        x: {{
                            grid: {{
                                color: 'rgba(255, 255, 255, 0.1)'
                            }},
                            ticks: {{
                                color: '#aaa',
                                font: {{ size: 11 }}
                            }}
                        }},
                        y: {{
                            grid: {{
                                color: 'rgba(255, 255, 255, 0.1)'
                            }},
                            ticks: {{
                                color: '#aaa',
                                font: {{ size: 11 }}
                            }}
                        }}
                    }}
                }}
            }});
        }}
        
        function createMiniChart(containerId, data) {{
            const ctx = document.getElementById(containerId).getContext('2d');
            
            return new Chart(ctx, {{
                type: 'bar',
                data: {{
                    labels: data.bin_labels,
                    datasets: [{{
                        data: data.bin_values,
                        backgroundColor: data.color + 'AA',
                        borderColor: data.color,
                        borderWidth: 1,
                        borderRadius: 3,
                    }}]
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {{
                        legend: {{ display: false }},
                        tooltip: {{ enabled: false }}
                    }},
                    scales: {{
                        x: {{ display: false }},
                        y: {{ display: false }}
                    }}
                }}
            }});
        }}
        
        function selectProblem(index) {{
            currentIndex = index;
            const data = problemData[index];
            
            // 更新信息面板
            updateInfoPanel(data);
            
            // 更新主图表
            createMainChart(data);
            
            // 更新侧边栏高亮
            document.querySelectorAll('.problem-item').forEach((item, i) => {{
                item.classList.toggle('active', i === index);
            }});
            
            // 更新小图高亮
            document.querySelectorAll('.mini-chart-wrapper').forEach((wrapper, i) => {{
                wrapper.classList.toggle('highlighted', i === index);
            }});
        }}
        
        function init() {{
            // 创建问题列表
            const listContainer = document.getElementById('problem-list');
            problemData.forEach((data, index) => {{
                const item = document.createElement('div');
                item.className = 'problem-item';
                const correctnessInfo = getCorrectnessInfo(data.is_correct);
                const badgeHtml = `<span class="correctness-badge ${{correctnessInfo.className}}">${{correctnessInfo.text}}</span>`;
                item.innerHTML = `
                    <div class="problem-name" style="border-left: 4px solid ${{data.color}}; padding-left: 10px;">${{data.problem_id}}${{badgeHtml}}</div>
                    <div class="problem-stats">Tokens: ${{formatNumber(data.total_tokens)}} | Unique: ${{formatNumber(data.unique_tokens)}}</div>
                `;
                item.addEventListener('click', () => selectProblem(index));
                item.addEventListener('mouseenter', () => selectProblem(index));
                listContainer.appendChild(item);
            }});
            
            // 创建小图网格
            const gridContainer = document.getElementById('all-charts-grid');
            problemData.forEach((data, index) => {{
                const wrapper = document.createElement('div');
                const correctnessInfo = getCorrectnessInfo(data.is_correct);
                const borderClass = data.is_correct === true ? 'correct-border' : 
                                    data.is_correct === false ? 'incorrect-border' : '';
                wrapper.className = 'mini-chart-wrapper ' + borderClass;
                wrapper.innerHTML = `
                    <div class="mini-chart-title">${{data.problem_id}}</div>
                    <div class="mini-chart">
                        <canvas id="miniChart${{index}}"></canvas>
                    </div>
                    <div class="correctness-indicator ${{correctnessInfo.className}}">${{correctnessInfo.text}}</div>
                `;
                wrapper.addEventListener('click', () => selectProblem(index));
                wrapper.addEventListener('mouseenter', () => selectProblem(index));
                gridContainer.appendChild(wrapper);
            }});
            
            // 创建所有小图表
            problemData.forEach((data, index) => {{
                miniCharts.push(createMiniChart(`miniChart${{index}}`, data));
            }});
            
            // 选择第一个问题
            selectProblem(0);
        }}
        
        init();
    </script>
</body>
</html>
'''
    
    output_path = os.path.join(output_dir, filename)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    
    return output_path


def generate_html_visualizations(
    project_root: Optional[Path] = None,
) -> None:
    """
    生成所有可视化：
    1. 收集所有问题的 token 统计信息（从已保存的 token_stats.json 读取）
    2. 生成交互式 HTML 页面
    
    Args:
        project_root: 项目根目录
    """
    if project_root is None:
        project_root = PROJECT_ROOT
    
    print("=" * 60)
    print("Collecting token statistics from all problems...")
    print("=" * 60)
    
    all_stats = collect_html_token_stats(project_root)
    
    if not all_stats:
        print("No statistics collected. Exiting.")
        return
    
    print(f"\nCollected statistics for {len(all_stats)} problems.")
    
    output_dir = os.path.join(project_root, "images")
    
    # 生成交互式 HTML
    print("\nGenerating interactive HTML...")
    html_path = generate_interactive_html(all_stats, output_dir)
    if html_path:
        print(f"  Interactive HTML saved to: {html_path}")
    
    print("\n" + "=" * 60)
    print("All visualizations completed!")
    print("=" * 60)


def main():
    # visualize_all_from_outputs()

    """主函数：从 outputs 读取数据并生成交互式 HTML"""
    # 生成交互式 HTML（直接从 token_stats.json 读取数据，无需 tokenizer）
    # generate_html_visualizations()



    data1 = load_token_stats("2024-I-1")
    data2 = load_token_stats("2024-I-2")

    tokens1 = data1["token_counts"].keys()
    tokens2 = data2["token_counts"].keys()

    # 交集
    common_tokens = set(tokens1) & set(tokens2)
    
    tokens1 = tokens1 - common_tokens
    for token in tokens1:
        # 将特殊字符转换为可读形式
        label = token.replace("Ġ", "").replace("Ċ", "").replace("ĉ", "")
        print(label)



if __name__ == "__main__":
    main()

