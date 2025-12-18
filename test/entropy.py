"""
可视化 LLM 输出过程中每个 token 的熵变化。
"""
from __future__ import annotations

import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import math
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from openai import OpenAI
from transformers import AutoTokenizer

from src.utils.data_loader import load_problem
from src.utils.config import DEFAULT_VLLM_BASE_URL, DEFAULT_MODEL_PATH


class EntropyVisualizer:
    """
    计算并可视化 LLM 生成过程中每个 token 的熵。
    """

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_PATH,
        api_base: str = DEFAULT_VLLM_BASE_URL,
        temperature: float = 0.6,
        top_p: float = 0.95,
        max_new_tokens: int = 4096,
        top_logprobs: int = 20,  # 获取 top-k 的 logprobs 用于熵计算
    ) -> None:
        self.model_path = model_path
        self.api_base = api_base.rstrip("/")
        self.temperature = temperature
        self.top_p = top_p
        self.max_new_tokens = max_new_tokens
        self.top_logprobs = top_logprobs

        self._client = OpenAI(
            base_url=f"{self.api_base}/v1",
            api_key="EMPTY",
        )
        self._tokenizer: Optional[AutoTokenizer] = None

    def _get_tokenizer(self) -> AutoTokenizer:
        """懒加载 tokenizer"""
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True
            )
        return self._tokenizer

    def _apply_chat_template(self, prompt: str) -> List[Dict[str, str]]:
        """将 prompt 转换为 chat messages 格式"""
        messages = [
            {"role": "system", "content": "You are a math problem solver. You are given a math problem and you need to solve it."},
            {"role": "user", "content": prompt}
        ]
        return messages

    def _calculate_entropy(self, logprobs_dict: Dict[str, float]) -> float:
        """
        根据 top logprobs 计算熵。
        
        H = -∑ p(x) * log(p(x))
        
        Args:
            logprobs_dict: token -> logprob 的字典
        
        Returns:
            熵值 (以 nats 为单位)
        """
        if not logprobs_dict:
            return 0.0

        # 将 logprobs 转换为概率
        logprobs = list(logprobs_dict.values())
        probs = [math.exp(lp) for lp in logprobs]
        print(probs)
        
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

    def generate_with_entropy(
        self, 
        prompt: str
    ) -> Tuple[str, List[str], List[float]]:
        """
        生成文本并收集每个 token 的熵。
        
        Args:
            prompt: 输入 prompt
        
        Returns:
            (生成的文本, token 列表, 熵列表)
        """
        messages = self._apply_chat_template(prompt)
        
        print(f"正在调用 LLM (temperature={self.temperature}, top_p={self.top_p})...")
        print("-" * 50)
        
        tokens_list: List[str] = []
        entropies: List[float] = []
        generated_text = ""
        
        # 使用 stream 模式获取 logprobs
        try:
            response = self._client.chat.completions.create(
                model=self.model_path,
                messages=messages,
                max_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                logprobs=True,
                top_logprobs=self.top_logprobs,
                stream=True,
            )
            
            for chunk in response:
                if chunk.choices and chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    generated_text += content
                    print(content, end="", flush=True)
                    
                    # 获取 logprobs
                    if hasattr(chunk.choices[0], 'logprobs') and chunk.choices[0].logprobs:
                        logprobs_content = chunk.choices[0].logprobs.content
                        if logprobs_content:
                            for token_info in logprobs_content:
                                token = token_info.token
                                tokens_list.append(token)
                                
                                # 收集 top logprobs
                                top_lps = {}
                                if token_info.top_logprobs:
                                    for tlp in token_info.top_logprobs:
                                        top_lps[tlp.token] = tlp.logprob
                                
                                # 计算熵
                                entropy = self._calculate_entropy(top_lps)
                                entropies.append(entropy)
                                return 1, 2, 3
            
            print("\n" + "-" * 50)
            
        except Exception as e:
            print(f"\n调用 LLM 出错: {e}")
            raise

        return generated_text, tokens_list, entropies

    def visualize_entropy(
        self,
        tokens: List[str],
        entropies: List[float],
        title: str = "Token Entropy During Generation",
        save_path: Optional[str] = None,
        max_display_tokens: int = 100,
    ) -> None:
        """
        可视化 token 熵的变化。
        
        Args:
            tokens: token 列表
            entropies: 对应的熵列表
            title: 图表标题
            save_path: 保存路径，如果为 None 则显示图表
            max_display_tokens: x 轴最多显示的 token 数量
        """
        if not tokens or not entropies:
            print("没有数据可视化")
            return

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
        if high_entropy_indices:
            print("\n高熵 token (熵 > mean + std):")
            for idx in high_entropy_indices[:10]:  # 最多显示 10 个
                token = tokens[idx]
                entropy = entropies[idx]
                # 处理特殊字符显示
                display_token = repr(token) if token.strip() == '' else token
                print(f"  [{idx}] {display_token}: {entropy:.4f}")
            if len(high_entropy_indices) > 10:
                print(f"  ... 还有 {len(high_entropy_indices) - 10} 个高熵 token")


def main():
    """主函数"""
    # 配置参数
    questions_dir = project_root / "questions"
    problem_id = "2024-I-11"
    
    # 加载问题
    print("=" * 50)
    print(f"加载问题: {problem_id}")
    print("=" * 50)
    
    problem = load_problem(questions_dir, problem_id)
    print(f"问题 ID: {problem.problem_id}")
    print(f"问题内容: {problem.prompt[:200]}..." if len(problem.prompt) > 200 else f"问题内容: {problem.prompt}")
    if problem.answer:
        print(f"参考答案: {problem.answer}")
    print()
    
    # 创建熵可视化器
    visualizer = EntropyVisualizer(
        model_path=DEFAULT_MODEL_PATH,
        api_base=DEFAULT_VLLM_BASE_URL,
        temperature=0.6,
        top_p=0.95,
        max_new_tokens=32000,
        top_logprobs=20,  # 获取 top-20 logprobs 用于熵计算
    )
    
    # 生成并获取熵
    print("开始生成...")
    generated_text, tokens, entropies = visualizer.generate_with_entropy(problem.prompt)
    
    return

    # 可视化
    if tokens and entropies:
        # 确保输出目录存在
        output_dir = project_root / "test" / "outputs"
        output_dir.mkdir(parents=True, exist_ok=True)
        save_path = str(output_dir / f"entropy_{problem_id.replace('-', '_')}.png")
        visualizer.visualize_entropy(
            tokens=tokens,
            entropies=entropies,
            title=f"Token Entropy - Problem {problem_id}",
            save_path=save_path,
        )
    else:
        print("未能获取 logprobs 数据，无法计算熵")


if __name__ == "__main__":
    main()
