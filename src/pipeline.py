from __future__ import annotations

from src.fusion_module import WeightedRRFFusion
from src.graph_agent import DocGraph, DocGraphHopAgent, ConflictFilterAgent, ConflictPlannerAgent
from src.types import QueryViews, RetrievedDoc, ScoredDoc


class RetrievalPipeline:
    def __init__(
            self,
            query_generator,
            dense_retriever,
            sparse_retriever,
            fusion: WeightedRRFFusion,
            doc_graph: DocGraph,
            hop_agent: DocGraphHopAgent,
            filter_agent: ConflictFilterAgent,
            planner_agent: ConflictPlannerAgent | None = None,
            ref_date: str | None = None,
    ) -> None:
        self.query_generator = query_generator
        self.dense_retriever = dense_retriever
        self.sparse_retriever = sparse_retriever
        self.fusion = fusion
        self.doc_graph = doc_graph
        self.hop_agent = hop_agent
        self.filter_agent = filter_agent
        self.planner_agent = planner_agent
        self.ref_date = ref_date

    def run_one(
            self,
            query_id: str,
            query: str,
            top_k_per_view: int = 10,
            final_top_k: int = 20,
    ) -> dict:
        conflict_plan: dict = {}
        if self.planner_agent is not None:
            conflict_plan = self.planner_agent.plan(
                query=query,
                ref_date=self.ref_date or "",
            )
            dims = [d.get("dimension", "?") for d in conflict_plan.get("conflict_dimensions", [])]
            print(f"  [planner] dimensions: {dims}", flush=True)

        query_views: QueryViews = self.query_generator.generate(
            query_id=query_id,
            query=query,
            ref_date=self.ref_date,
            conflict_plan=conflict_plan,
        )

        all_docs: dict[str, str] = {
            doc.doc_id: doc.text
            for doc in self.dense_retriever.documents
        }

        all_hits: list[RetrievedDoc] = []
        for view_type, view_text in query_views.all_views():
            all_hits.extend(self.dense_retriever.retrieve(
                query_view=view_text,
                view_type=view_type,
                top_k=top_k_per_view,
            ))
            all_hits.extend(self.sparse_retriever.retrieve(
                query_view=view_text,
                view_type=view_type,
                top_k=top_k_per_view,
            ))

        fused: list[ScoredDoc] = self.fusion.fuse(
            results=all_hits,
            top_k=final_top_k,
        )
        fused_doc_ids: list[str] = [doc.doc_id for doc in fused]
        print(f"  [fused] {len(fused_doc_ids)} docs from RRF fusion", flush=True)

        seed_doc_ids: list[str] = self.filter_agent.filter(
            query=query,
            doc_ids=fused_doc_ids,
            all_docs=all_docs,
        )
        print(f"  [seed] {len(fused_doc_ids)} -> {len(seed_doc_ids)} docs after pre-hop filtering", flush=True)

        collected: list[str] = list(seed_doc_ids)
        visited: set[str] = set(seed_doc_ids)
        hop_trace: list[dict] = []

        for hop in range(self.hop_agent.max_hops):
            new_docs: list[str] = []
            for doc_id in collected:
                for nbr_id, _ in self.hop_agent.doc_graph.neighbors(
                        doc_id, top_k=self.hop_agent.neighbors_per_doc
                ):
                    if nbr_id not in visited and nbr_id in all_docs:
                        visited.add(nbr_id)
                        new_docs.append(nbr_id)

            if not new_docs:
                break

            collected.extend(new_docs)
            hop_trace.append({"hop": hop, "n_collected": len(collected), "n_new": len(new_docs)})
            print(f"  [hop {hop}] total={len(collected)} (+{len(new_docs)} new)", flush=True)

        agent_result = {"collected_doc_ids": collected, "hop_trace": hop_trace}
        collected_doc_ids: list[str] = collected
        print(f"  [collected] {len(collected_doc_ids)} docs after hop+filter", flush=True)

        filtered_doc_ids = self.filter_agent.filter(
            query=query,
            doc_ids=collected_doc_ids,
            all_docs=all_docs,
        )
        print(f"  [filter] {len(collected_doc_ids)} -> {len(filtered_doc_ids)} docs (final)", flush=True)

        fused_score_map = {doc.doc_id: doc for doc in fused}
        final_results: list[ScoredDoc] = []
        for doc_id in filtered_doc_ids:
            text = all_docs.get(doc_id, "")
            if not text:
                continue
            scored = fused_score_map.get(doc_id)
            final_results.append(
                ScoredDoc(
                    doc_id=doc_id,
                    text=text,
                    fusion_score=scored.fusion_score if scored else 0.0,
                    component_scores=scored.component_scores if scored else {},
                    matched_views=scored.matched_views if scored else [],
                    matched_sources=scored.matched_sources if scored else ["graph"],
                    metadata=scored.metadata if scored else {},
                )
            )

        return {
            "query_id": query_id,
            "query": query,
            "query_views": query_views.to_dict(),
            "fused_doc_ids": fused_doc_ids,
            "seed_doc_ids": seed_doc_ids,
            "raw_results": [doc.to_dict() for doc in fused],
            "fused_results": [r.to_dict() for r in final_results],
            "collected_doc_ids": collected_doc_ids,
            "filtered_doc_ids": filtered_doc_ids,
            "hop_trace": agent_result["hop_trace"],
        }

    def run_batch(
            self,
            queries: list[dict],
            top_k_per_view: int = 10,
            final_top_k: int = 20,
    ) -> list[dict]:
        return [
            self.run_one(
                query_id=row["query_id"],
                query=row["query"],
                top_k_per_view=top_k_per_view,
                final_top_k=final_top_k,
            )
            for row in queries
        ]


class FullPipeline:
    def __init__(self, retrieval_pipeline: RetrievalPipeline, answer_generator) -> None:
        self.retrieval_pipeline = retrieval_pipeline
        self.answer_generator = answer_generator

    def run_one(self, query_id, query, persona_name, ref_date,
                top_k_per_view=10, final_top_k=20) -> dict:
        retrieval_output = self.retrieval_pipeline.run_one(
            query_id=query_id, query=query,
            top_k_per_view=top_k_per_view, final_top_k=final_top_k,
        )
        answer = self.answer_generator.generate(
            query_id=query_id, query=query,
            docs=retrieval_output["fused_results"],
            persona_name=persona_name, ref_date=ref_date,
        )
        return {**retrieval_output, "answer": answer.to_dict()}

    def run_batch(self, queries, persona_name, ref_date,
                  top_k_per_view=10, final_top_k=20) -> list[dict]:
        return [
            self.run_one(
                query_id=row["query_id"], query=row["query"],
                persona_name=persona_name, ref_date=ref_date,
                top_k_per_view=top_k_per_view, final_top_k=final_top_k,
            )
            for row in queries
        ]
