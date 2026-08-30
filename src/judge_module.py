from __future__ import annotations
from openai import OpenAI
from src.types import JudgeEvalResult


class JudgeEvaluator:
    def __init__(
            self,
            model: str = "gpt-5.4-mini",
            client: OpenAI | None = None,
    ):
        self.model = model
        self.client = client or OpenAI()

    def evaluate(
            self,
            query_id: str,
            answer_text: str,
            gold_judgment: str,
            gold_label: str | None = None,
    ) -> JudgeEvalResult:
        prompt = f"""### Instruction:
You are evaluating a model response for a conflict-aware personal assistant task.
The model response is a free-text answer. It should clearly state whether the user's request can be carried out, and explain why.

You must evaluate two things in order:

Step 1. Decision correctness
- Gold Label indicates the correct decision.
- If Gold Label is "non_conflict", the correct response should proceed, agree, allow, or say the request can be carried out.
- If Gold Label is "conflict", the correct response should refuse, warn, block, or say the request should not be carried out as requested.
- If the model response gives the wrong decision, output FAIL.
- If the model response is ambiguous about whether it will proceed or refuse, output FAIL.

Step 2. Rationale correctness
Only evaluate the rationale if the decision is correct.
Compare the model response with the Gold Rationale.
The response does not need exact wording, but it must capture the same core reason.

Use the following labels:

PASS:
The model response conveys the correct decision, and the rationale captures the same core reason as the Gold Rationale.
Minor wording differences are acceptable.
Extra helpful details are acceptable if they do not change or distort the core reason.

WRONG_RATIONALE:
The model response conveys the correct decision, but the rationale is not fully correct.
This includes cases where the rationale:
- misses key evidence from the Gold Rationale
- is too vague to fully support the decision
- mentions only a secondary reason while missing the main reason
- includes irrelevant or weak justifications
- adds unsupported details that make the reasoning less faithful, while the final decision is still correct

FAIL:
The model response conveys the wrong decision.
Also output FAIL if the response is ambiguous about the decision.
Also output FAIL if the decision is correct but the rationale gives the opposite, unsupported, or substantially wrong reason.

Important:
- Decision correctness has priority.
- If the decision is wrong, output FAIL even if some rationale sounds plausible.
- Do not require exact wording.
- Do not penalize conversational phrasing or helpful suggestions.
- Only output one label: PASS, WRONG_RATIONALE, or FAIL.
- Do not include any explanation.

### Gold Label:
{gold_label}

### Gold Rationale:
{gold_judgment}

### Model Response:
{answer_text}

### Label:
"""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an evaluation judge. "
                        "Output only one label: PASS, WRONG_RATIONALE, or FAIL."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
        )

        judge_label = (response.choices[0].message.content or "").strip().upper()

        if judge_label not in {"PASS", "WRONG_RATIONALE", "FAIL"}:
            judge_label = "FAIL"

        return JudgeEvalResult(
            query_id=query_id,
            judge=judge_label,
            expected_decision=None,
            decision_correct=None,
            gold_label=gold_label or "",
        )
