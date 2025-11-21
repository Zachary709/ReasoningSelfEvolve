from __future__ import annotations

PROMPT_SOLUTION_TEMPLATE = """You are an IMO gold medalist mathematician. Think step by step before answering.

Solve the following problem, explain your reasoning clearly, and ensure your final answer is strictly in the format: \\boxed{{...}}.

Problem:
{problem}
"""

PROMPT_VERIFICATION_TEMPLATE = """You are an independent verifier. Examine the candidate solution line by line, without assuming its final answer is correct. Re-derive every key step, point out any logical gaps or numerical mistakes, and state clearly whether the solution is trustworthy.

Your report must include:
- A concise recap of the critical steps you checked.
- Any issues you found, with a brief explanation; if multiple, list them all.

PROBLEM:
{problem}

CANDIDATE SOLUTION:
{solution}

Now, please verify the candidate solution.
"""

PROMPT_REFINEMENT_TEMPLATE = """
You are an IMO gold medalist mathematician. Think step by step before answering.

Given your previous solution and the verification report's feedback, re-solve the problem carefully from scratch. Address every issue raised, confirm each derivation, and only present an answer you have fully justified. Ensure your final answer is strictly in the format: \\boxed{{...}}.

PROBLEM:
{problem}

PREVIOUS SOLUTION:
{solution}

VERIFICATION REPORT:
{verification}

Now, please give your new solution.
"""


class PromptBuilder:
    def solution(self, problem: str) -> str:
        return PROMPT_SOLUTION_TEMPLATE.format(problem=problem)

    def verification(self, problem: str, solution: str) -> str:
        return PROMPT_VERIFICATION_TEMPLATE.format(problem=problem, solution=solution)

    def refinement(self, problem: str, solution: str, verification: str) -> str:
        return PROMPT_REFINEMENT_TEMPLATE.format(
            problem=problem, solution=solution, verification=verification
        )

