from __future__ import annotations
import json
import re
from typing import Any
from src.types import QueryViews


class MultiViewQueryGenerator:
    def __init__(
            self,
            client: Any,
            model: str = "Qwen/Qwen3-4B-Instruct-2507",
            num_counter: int = 3,
            backend: str = "openai_chat",
            temperature: float = 0.6,
            max_output_tokens: int = 300,
    ) -> None:
        self.client = client
        self.model = model
        self.num_counter = num_counter
        self.backend = backend
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens

    def generate(
            self,
            query_id: str,
            query: str,
            ref_date: str | None = None,
            conflict_plan: dict | None = None,
    ) -> QueryViews:
        query = query.strip()
        if not query:
            raise ValueError("query must not be empty")

        result = self._generate_views(
            query=query,
            ref_date=ref_date or "",
            conflict_plan=conflict_plan or {},
        )

        return QueryViews(
            query_id=query_id,
            original=query,
            reduced="",
            counter=result["counter"][: self.num_counter],
        )

    def _generate_views(
            self,
            query: str,
            ref_date: str,
            conflict_plan: dict | None = None,
    ) -> dict[str, Any]:
        prompt = self._build_prompt(query, ref_date, conflict_plan=conflict_plan or {})

        if self.backend == "openai_responses":
            data = self._call_openai_responses(prompt)
        elif self.backend == "openai_chat":
            data = self._call_openai_chat(prompt)
        else:
            raise ValueError(
                f"Unsupported backend: {self.backend}. "
                f"Choose from ['openai_responses', 'openai_chat']"
            )

        counter = [q.strip() for q in data["counter"] if isinstance(q, str) and q.strip()]

        if not counter:
            raise ValueError("generated counter queries are empty")

        return {"counter": counter}

    def _call_openai_responses(self, prompt: str) -> dict[str, Any]:
        response = self.client.responses.create(
            model=self.model,
            input=prompt,
            text={
                "format": {
                    "type": "json_schema",
                    "name": "query_views",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "counter": {
                                "type": "array",
                                "items": {"type": "string"},
                                "minItems": 1,
                                "maxItems": self.num_counter,
                            },
                        },
                        "required": ["counter"],
                        "additionalProperties": False,
                    },
                }
            },
        )
        return json.loads(response.output_text)

    def _call_openai_chat(self, prompt: str) -> dict[str, Any]:
        response = self.client.chat.completions.create(
            model=self.model,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a query reformulation assistant. "
                        "Return valid JSON only with key 'counter'."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt + "\n\nReturn JSON.",
                },
            ],
        )

        content = response.choices[0].message.content
        if content is None:
            raise ValueError("model returned empty content")

        return self._robust_json_loads(content)

    def _robust_json_loads(self, text: str) -> dict[str, Any]:
        text = text.strip()

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, flags=re.DOTALL)
        if fenced:
            return json.loads(fenced.group(1))

        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(text[start:end + 1])

        raise ValueError(f"failed to parse JSON from model output:\n{text}")

    def _build_prompt(
            self,
            query: str,
            ref_date: str,
            conflict_plan: dict | None = None,
    ) -> str:
        ref_date_block = ""
        if ref_date:
            ref_date_block = f"""\
The reference date is {ref_date}.
When the query contains relative temporal expressions such as today, tomorrow, tonight,
this Wednesday, next Friday, or this weekend, interpret them with respect to the reference date
when generating counter views.
Do not add extra explanation about the temporal resolution.
"""

        conflict_block = ""
        if conflict_plan and conflict_plan.get("conflict_dimensions"):
            dims = conflict_plan["conflict_dimensions"]
            lines = []
            for d in dims:
                anchors = ", ".join(d.get("key_anchors", []))
                lines.append(
                    f"  - [{d.get('dimension', '?')}] {d.get('description', '')} "
                    f"(key anchors: {anchors})"
                )
            risk = conflict_plan.get("risk_summary", "")
            conflict_block = (
                    "Conflict analysis (use this to guide counter query generation):\n"
                    + "\n".join(lines)
                    + (f"\nRisk summary: {risk}" if risk else "")
                    + "\n"
            )

        return f"""\
### Instruction:
You generate retrieval-oriented counter queries for conflict-aware retrieval.
Return JSON with exactly one field: "counter".

{ref_date_block}

{conflict_block}
### Definitions:
* counter
A list of retrieval queries aimed at finding information that could block, constrain, or conflict with the request.
These should not be simple paraphrases of the original query.
They should help retrieve evidence about conflict-bearing factors such as schedule, availability, timing, commitments, access constraints, coordination constraints, and so on.
If a conflict analysis is provided above, use it to generate counter queries that specifically target those conflict dimensions and anchors. Each counter query should be distinct and targeted.
Avoid near-duplicate counter queries.

* Rules:
- Output valid JSON only
- Do not include explanations
- "counter": list of {self.num_counter} strings

Request: {query}
"""
