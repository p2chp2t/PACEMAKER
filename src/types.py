from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Literal, Protocol


@dataclass
class Document:
    doc_id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class QueryViews:
    query_id: str
    original: str
    reduced: str
    counter: list[str] = field(default_factory=list)

    def all_views(self) -> list[tuple[str, str]]:
        views: list[tuple[str, str]] = [
            ("original", self.original),
        ]
        views.extend([("counter", q) for q in self.counter if q.strip()])
        return views

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RetrievedDoc:
    doc_id: str
    text: str
    score: float
    rank: int
    source: str
    view_type: str
    query_view: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ScoredDoc:
    doc_id: str
    text: str
    fusion_score: float
    component_scores: dict[str, float] = field(default_factory=dict)
    matched_views: list[str] = field(default_factory=list)
    matched_sources: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AnswerResult:
    query_id: str
    decision: str
    explanation: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class JudgeEvalResult:
    query_id: str
    judge: str
    expected_decision: str = ""
    decision_correct: bool | None = None
    gold_label: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
