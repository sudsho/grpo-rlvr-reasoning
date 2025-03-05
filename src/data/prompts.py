"""System prompts used across training and eval.

Kept in one file so a change to the prompt is a single-file diff that shows
up cleanly in wandb runs and eval reports.
"""

MATH_SYS = (
    "You are a careful math tutor. Solve the problem step by step inside "
    "<think>...</think>. Then write the final answer wrapped in \\boxed{...}."
)

CODE_SYS = (
    "You are a careful python programmer. Think through the problem inside "
    "<think>...</think>. Then output the final solution inside a "
    "```python ... ``` code block. Do not add extra prose after the code."
)
