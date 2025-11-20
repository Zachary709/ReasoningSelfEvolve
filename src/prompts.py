from __future__ import annotations

PROMPT_SOLUTION_TEMPLATE = """You are an IMO gold medalist mathematician. Think step by step before answering.

Solve the following problem strictly using this format:
<think>
private reasoning, plans, checks. Keep it detailed yet concise.
</think>
<solution>
public explanation plus computations and end with: Final Answer: \\boxed{{...}}
</solution>

Do not produce text outside these tags, and never output multiple <solution> blocks.

Problem:
{problem}
"""

PROMPT_VERIFICATION_TEMPLATE = """Verify the given solution step by step to check correctness. Provide a short verification report, containing the key points of the solution and any errors found. Finally, put your judgement strictly in the format: \\boxed{{1}} if correct, or \\boxed{{0}} if incorrect.

Problem:
{problem}

Candidate solution:
{solution}
"""

PROMPT_REFINEMENT_TEMPLATE = """Given your previous solution and verification report, reconsider the problem carefully and provide a corrected solution.

Respond strictly using the format:
<think>
updated private reasoning, including analysis of the verification feedback.
</think>
<solution>
revised public explanation ending with Final Answer: \\boxed{{...}}
</solution>

Do not output anything outside these tags.

Problem:
{problem}

Previous solution:
{solution}

Verification report:
{verification}
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

