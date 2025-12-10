from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


@dataclass
class ProblemRecord:
    problem_id: str
    prompt: str
    answer: Optional[str] = None


def load_problem(questions_dir: Path, problem_id: Optional[str] = None, questions_file: str = "aime2024_questions.txt") -> ProblemRecord:
    """
    从 questions 文件夹下的 txt 文件加载问题。
    
    Args:
        questions_dir: questions 文件夹路径
        problem_id: 问题ID，如果为 None 则随机选择
        questions_file: 问题文件名，默认为 aime2024_questions.txt
    
    Returns:
        ProblemRecord 对象
    """
    questions_path = questions_dir / questions_file
    if not questions_path.exists():
        raise FileNotFoundError(f"Questions file not found at {questions_path}")
    
    # 读取所有问题
    questions_dict = {}
    with questions_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t", 1)
            if len(parts) == 2:
                q_id, problem_text = parts
                questions_dict[q_id] = problem_text
    
    if not questions_dict:
        raise ValueError(f"No problems found in {questions_path}")
    
    # 选择问题
    if problem_id is not None:
        if problem_id not in questions_dict:
            raise ValueError(f"Problem ID {problem_id} not found in dataset.")
        selected_id = problem_id
    else:
        selected_id = random.choice(list(questions_dict.keys()))
    
    # 尝试加载答案
    answer = None
    # 根据问题文件名生成对应的答案文件名
    if "questions" in questions_file:
        answers_file = questions_file.replace("questions", "answers")
    else:
        # 如果文件名不包含 "questions"，则尝试添加 "_answers"
        base_name = questions_file.replace(".txt", "")
        answers_file = f"{base_name}_answers.txt"
    answers_path = questions_dir / answers_file
    if answers_path.exists():
        with answers_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("\t", 1)
                if len(parts) == 2:
                    a_id, answer_text = parts
                    if a_id == selected_id:
                        answer = answer_text
                        break
    
    return ProblemRecord(problem_id=selected_id, prompt=questions_dict[selected_id], answer=answer)


def load_all_problems(
    questions_dir: Path, questions_file: str = "aime2024_questions.txt"
) -> List[ProblemRecord]:
    """
    加载问题文件中的所有问题。
    
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
    for problem_id in problem_ids:
        record = load_problem(questions_dir, problem_id, questions_file)
        problems.append(record)
    
    return problems

