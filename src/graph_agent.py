from __future__ import annotations

import json
import pickle
import re
from pathlib import Path
from typing import Any

from openai import OpenAI


class DocGraph:
    def __init__(self, graph_path: str | Path) -> None:
        with open(graph_path, "rb") as f:
            self._graph: dict[str, list[tuple[str, float]]] = pickle.load(f)

    def neighbors(self, doc_id: str, top_k: int | None = None) -> list[tuple[str, float]]:
        nbrs = self._graph.get(doc_id, [])
        return nbrs[:top_k] if top_k else nbrs


_FILTER_USER = """\
### Instruction:
You are an evidence selection agent for conflict-aware personal assistant retrieval.

You will receive:
1. A user request
2. A list of candidate documents collected through retrieval and graph expansion

Your task is NOT to make the final decision about the request.

Your task is to select the documents that are MOST LIKELY to help a downstream assistant determine whether the request can or cannot be carried out.

* A document may be useful even if:
- it does not explicitly mention a conflict
- it only provides partial information
- it contains habits, routines, preferences, schedules, relationships, locations, or behavioral context
- its relevance may only become clear after combining it with other documents

Focus on retaining documents that may later become important evidence.

* Respond with a JSON object:
{{
  "selected_doc_ids": ["doc_id_1", "doc_id_2", ...]
}}

* Selection priorities:
1. Documents containing constraints, commitments, obligations, availability, schedules, routines, or resource limitations relevant to the request.
2. Documents containing personal preferences, behavioral tendencies, habits, sensitivities, or compatibility information relevant to the request.
3. Documents that clarify people, places, timing, recurring activities, dependencies, or social relationships mentioned or implied in the request.
4. Documents that may serve as bridge evidence connecting the request to other potentially relevant facts.

* Rules:
- Select exactly {top_n} doc_ids if at least {top_n} candidate documents are provided.
- Order selected_doc_ids from most useful to least useful.
- Prefer documents with concrete, person-specific, time-specific, place-specific, or state-specific information.
- Prefer potentially decision-relevant evidence over topical similarity.
- Do NOT require a document to explicitly prove feasibility or conflict in order to keep it.
- Avoid documents that are purely topical and provide no useful contextual evidence.
- Only use doc_ids from the provided list.
- Return valid JSON only. No explanation. No markdown fences.

### Request:
{query}

### Candidate documents ({n_docs} total):
{docs_text}

Select the top {top_n} doc_ids that are most useful as evidence for deciding whether the request can be carried out.
Respond with JSON only.

### Response:
"""

_PLANNER_USER = """\
### Instruction:
You are a traversal guidance agent for conflict-aware personal assistant retrieval.

You will receive a user request.
Your task is to identify retrieval directions that are likely to uncover useful evidence for downstream conflict reasoning.

* Focus on context directions that may help later retrieval discover:
- constraints
- commitments
- routines
- preferences
- availability
- compatibility
- dependencies
- supporting feasibility evidence
- contextual information that may become important when combined with other documents

Useful directions do NOT need to directly imply a conflict.
Indirect or partial contextual evidence may still be important later.

Be selective.
Only include directions that are plausibly useful for retrieving decision-relevant evidence for this specific request.
Avoid generic directions that are not grounded in the request.

* Context directions may involve:
- schedules, timing, recurring routines, or prior commitments
- people involved, relationships, coordination, or availability
- locations, transportation, access, or travel
- preferences, sensitivities, habits, or behavioral tendencies
- physical state, health, mobility, or energy constraints
- resources, reservations, permissions, or required preparation
- contextual information that helps interpret ambiguous or underspecified parts of the request

* Respond with a JSON object:
{{
  "conflict_dimensions": [
    {{
      "dimension": "<short free-form context direction>",
      "description": "<one concrete sentence about WHY this direction is useful for this request>",
      "key_anchors": ["<specific name, time, place, or role from the request>"]
    }}
  ],
  "risk_summary": "<one sentence on the most useful retrieval focus>"
}}

* Rules:
- Select only 1 to 3 directions. If only one clearly applies, return just one.
- Each direction must be grounded in something specific from the request.
- Do not include speculative directions with no plausible retrieval value.
- Include directions that may support feasibility as well as directions that may reveal constraints.
- key_anchors must contain exact terms from the request, not generic paraphrases.
- Return valid JSON only. No explanation. No markdown fences.

### Request:
{query}

### Reference date: {ref_date}

Identify decision-relevant context directions that would help graph traversal \
select useful documents for judging this request later. \
Return JSON only.\

### Response:
"""


class DocGraphHopAgent:
    def __init__(
            self,
            doc_graph: DocGraph,
            max_hops: int = 5,
            neighbors_per_doc: int = 3,
    ) -> None:
        self.doc_graph = doc_graph
        self.max_hops = max_hops
        self.neighbors_per_doc = neighbors_per_doc

    def run(
            self,
            query: str,
            seed_doc_ids: list[str],
            all_docs: dict[str, str],
    ) -> dict[str, Any]:
        collected: list[str] = list(seed_doc_ids)
        visited: set[str] = set(seed_doc_ids)
        hop_trace: list[dict[str, Any]] = []

        for hop in range(self.max_hops):
            new_docs: list[str] = []
            for doc_id in collected:
                for nbr_id, _ in self.doc_graph.neighbors(doc_id, top_k=self.neighbors_per_doc):
                    if nbr_id not in visited and nbr_id in all_docs:
                        visited.add(nbr_id)
                        new_docs.append(nbr_id)

            if not new_docs:
                break

            collected.extend(new_docs)
            print(f"  [hop {hop}] total={len(collected)} (+{len(new_docs)} new)", flush=True)
            hop_trace.append({"hop": hop, "n_collected": len(collected), "n_new": len(new_docs)})

        return {
            "collected_doc_ids": collected,
            "hop_trace": hop_trace,
        }


class ConflictPlannerAgent:
    def __init__(
            self,
            client: OpenAI,
            model: str = "Qwen/Qwen3-4B-Instruct-2507",
            temperature: float = 0.6,
            max_tokens: int = 512,
    ) -> None:
        self.client = client
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def plan(
            self,
            query: str,
            ref_date: str = "",
    ) -> dict:
        user_msg = _PLANNER_USER.format(query=query, ref_date=ref_date or "unknown")
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": user_msg},
                ],
            )
            content = response.choices[0].message.content or ""
            parsed = _safe_parse(content, fallback={})
            if "conflict_dimensions" in parsed:
                return parsed
        except Exception as e:
            print(f"  [planner] LLM error: {e} — returning empty plan", flush=True)

        return {"conflict_dimensions": [], "risk_summary": ""}



class ConflictFilterAgent:
    def __init__(
            self,
            client: OpenAI,
            model: str = "gpt-5.4-mini",
            top_n: int = 10,
            temperature: float = 0.0,
    ) -> None:
        self.client = client
        self.model = model
        self.top_n = top_n
        self.temperature = temperature

    def filter(
            self,
            query: str,
            doc_ids: list[str],
            all_docs: dict[str, str],
    ) -> list[str]:
        if len(doc_ids) <= self.top_n:
            return doc_ids

        parts: list[str] = []
        for i, doc_id in enumerate(doc_ids, start=1):
            text = all_docs.get(doc_id, "")
            parts.append(f"[{i}] (doc_id={doc_id})\n{text}")
        docs_text = "\n\n".join(parts)

        user_msg = _FILTER_USER.format(
            query=query,
            n_docs=len(doc_ids),
            docs_text=docs_text,
            top_n=self.top_n
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "user", "content": user_msg},
                ],
            )
            content = response.choices[0].message.content or ""
            parsed = _safe_parse(content, fallback={"selected_doc_ids": []})
            selected = parsed.get("selected_doc_ids", [])

            valid = set(doc_ids)
            filtered = [d for d in selected if d in valid]

            if filtered:
                return filtered[:self.top_n]
        except Exception as e:
            print(f"  [filter] LLM error: {e} — falling back to top-{self.top_n}", flush=True)

        return doc_ids[:self.top_n]


def _safe_parse(text: str, fallback: dict[str, Any]) -> dict[str, Any]:
    text = text.strip()

    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        try:
            return json.loads(fenced.group(1))
        except json.JSONDecodeError:
            pass

    s, e = text.find("{"), text.rfind("}")
    if s != -1 and e > s:
        try:
            return json.loads(text[s:e + 1])
        except json.JSONDecodeError:
            pass

    return fallback
