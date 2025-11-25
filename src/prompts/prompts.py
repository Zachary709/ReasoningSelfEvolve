from __future__ import annotations

from typing import Dict, List

Message = Dict[str, str]

PROMPT_SOLUTION_SYSTEM = (
    '''You are an IMO Mathematics Competition competitor. There will be a problem for you to solve.
    1. Think step by step before answering. 
    2. Solve the following problem.
    3. Ensure your final answer is strictly in the format: \\boxed{{...}}.'''
)

PROMPT_SOLUTION_USER = """
Problem:
{problem}

Now, please solve the problem.
"""

PROMPT_VERIFICATION_SYSTEM = (
    '''You are a judge for the IMO Mathematics Competition. There will be an answer given by the contestant for you to evaluate.
    1. The answer may be incomplete, in which case please focus on each step rather than the missing parts.
    2. Check line by line whether there are any issues with the answer given by the contestant. 
    3. If there are any issues, explain the reasons for the issues, and indicate the optimization direction. 
    4. If the solution is trustworthy, state clearly that the solution is trustworthy.'''
)

PROMPT_VERIFICATION_USER = """
Problem:
{problem}

Answer given by the contestant:
{solution}

Now, please examine the answer given by the contestant."""

PROMPT_REFINEMENT_SYSTEM = (
    '''You are an IMO Mathematics Competition competitor. 
    1. Think step by step before answering. 
    2. Given your previous solution and the judge's feedback, re-solve the problem carefully from scratch. 
    3. Address every issue raised, confirm each derivation, and only present an answer you have fully justified. 
    4. Ensure your final answer is strictly in the format: \\boxed{{...}}.'''
)

PROMPT_REFINEMENT_USER = """
Problem:
{problem}

Previous solution:
{solution}

Verification report:
{verification}

Now, please give your new solution."""


class PromptBuilder:
    @staticmethod
    def _build_messages(system_prompt: str, user_prompt: str) -> List[Message]:
        return [
            {"role": "system", "content": system_prompt.strip()},
            {"role": "user", "content": user_prompt.strip()},
        ]

    def solution(self, problem: str) -> List[Message]:
        return self._build_messages(
            PROMPT_SOLUTION_SYSTEM,
            PROMPT_SOLUTION_USER.format(problem=problem),
        )

    def verification(self, problem: str, solution: str) -> List[Message]:
        return self._build_messages(
            PROMPT_VERIFICATION_SYSTEM,
            PROMPT_VERIFICATION_USER.format(problem=problem, solution=solution),
        )

    def refinement(
        self, problem: str, solution: str, verification: str
    ) -> List[Message]:
        return self._build_messages(
            PROMPT_REFINEMENT_SYSTEM,
            PROMPT_REFINEMENT_USER.format(
                problem=problem, solution=solution, verification=verification
            ),
        )

