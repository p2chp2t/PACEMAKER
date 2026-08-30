from __future__ import annotations
from collections import defaultdict
from src.types import RetrievedDoc, ScoredDoc


class WeightedRRFFusion:
    def __init__(
            self,
            k: int = 30,
            view_weights: dict[str, float] | None = None,
    ):
        self.k = k
        self.view_weights = view_weights or {
            "original": 1.0,
            "counter": 1.2,
        }

    def fuse(self, results: list[RetrievedDoc], top_k: int) -> list[ScoredDoc]:
        score_map: dict[str, float] = defaultdict(float)
        text_map: dict[str, str] = {}
        metadata_map: dict[str, dict] = {}
        component_map: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        view_map: dict[str, set[str]] = defaultdict(set)
        source_map: dict[str, set[str]] = defaultdict(set)

        for item in results:
            weight = self.view_weights.get(item.view_type, 1.0)
            contrib = weight * (1.0 / (self.k + item.rank))

            score_map[item.doc_id] += contrib
            text_map[item.doc_id] = item.text
            metadata_map[item.doc_id] = item.metadata

            comp_key = f"{item.view_type}:{item.source}"
            component_map[item.doc_id][comp_key] += contrib
            view_map[item.doc_id].add(item.view_type)
            source_map[item.doc_id].add(item.source)

        scored_docs: list[ScoredDoc] = []
        for doc_id, fusion_score in score_map.items():
            scored_docs.append(
                ScoredDoc(
                    doc_id=doc_id,
                    text=text_map[doc_id],
                    fusion_score=float(fusion_score),
                    component_scores=dict(component_map[doc_id]),
                    matched_views=sorted(view_map[doc_id]),
                    matched_sources=sorted(source_map[doc_id]),
                    metadata=metadata_map[doc_id],
                )
            )

        scored_docs.sort(key=lambda x: x.fusion_score, reverse=True)
        return scored_docs[:top_k]
