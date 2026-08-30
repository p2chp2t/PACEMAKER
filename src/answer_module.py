from __future__ import annotations

from openai import OpenAI
from src.types import ScoredDoc


class AnswerGenerator:
    def __init__(
            self,
            model: str = "gpt-5.4-mini",
            max_docs: int = 10,
            client: OpenAI | None = None,
            backend: str = "openai_chat",
    ):
        self.model = model
        self.max_docs = max_docs
        self.client = client or OpenAI()
        self.backend = backend

    def generate(
            self,
            query_id: str,
            query: str,
            docs: list[ScoredDoc],
            persona_name: str,
            ref_date: str,
    ) -> str:
        selected_docs = docs[: self.max_docs]
        context_text = self._build_context_text(selected_docs)
        prompt = self._build_prompt(
            query=query,
            persona_name=persona_name,
            ref_date=ref_date,
            context_text=context_text,
        )

        if self.backend == "openai_responses":
            return self._call_openai_responses(prompt)
        elif self.backend == "openai_chat":
            return self._call_openai_chat(prompt)
        else:
            raise ValueError(
                f"Unsupported backend: {self.backend}. "
                f"Choose from ['openai_responses', 'openai_chat']"
            )

    def _build_context_text(self, docs: list[ScoredDoc]) -> str:
        if not docs:
            return "(No retrieved facts)"
        return "\n".join(doc.text for doc in docs)

    def _build_prompt(
            self,
            query: str,
            persona_name: str,
            ref_date: str,
            context_text: str,
    ) -> str:
        return f"""### Instruction:
You are the personal assistant for {persona_name}.
Today is {ref_date}.
The information below consists of facts you already know about {persona_name} and the world they live in.
Answer the query as their assistant, using those facts when relevant.

Act as an agent who can take actions in that world, coordinate with people and systems there, and speak directly to {persona_name}.
Use what you know to decide whether this request should be carried out.

Write a response of 2 to 4 sentences that clearly states whether the request can or cannot be carried out, and explains why based on the known facts.

Rules:
- Clearly state whether the request can or cannot be carried out.
- Explain why, based only on the provided facts and the query.
- Speak directly as the assistant to {persona_name}.
- Do not mention "context", "provided information", or "dataset".
- Do not refuse with the reason of "no permission", assume you are able to take any action as an assistant for {persona_name}.
- Do not use bullet points, JSON, or any labels.
- Keep the response to 2 to 4 sentences.

### Context:
{context_text}

### Query:
{query}

### Response:
"""

    def _call_openai_responses(self, prompt: str) -> str:
        response = self.client.responses.create(
            model=self.model,
            input=prompt,
        )
        return response.output_text.strip()

    def _call_openai_chat(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "user", "content": prompt},
            ],
        )
        content = response.choices[0].message.content
        if content is None:
            raise ValueError("model returned empty content")
        return content.strip()
