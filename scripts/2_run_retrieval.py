from __future__ import annotations

import argparse
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.data_utils import load_corpus, load_query_bundle, save_json
from src.fusion_module import WeightedRRFFusion
from src.graph_agent import DocGraph, DocGraphHopAgent, ConflictFilterAgent, ConflictPlannerAgent
from src.pipeline import RetrievalPipeline
from src.query_module import MultiViewQueryGenerator
from src.retrieval_module import DenseRetriever, SparseRetriever


def process_case(
        case_dir: Path,
        index_root: Path,
        out_root: Path,
        dense_model: str,
        query_view_model: str,
        query_view_backend: str,
        query_view_base_url: str,
        query_view_temperature: float,
        query_view_max_output_tokens: int,
        top_k_per_view: int,
        final_top_k: int,
        num_counter: int,
        max_hops: int,
        neighbors_per_doc: int,
        filter_model: str,
        filter_base_url: str,
        filter_top_n: int,
) -> dict:
    case_name = case_dir.name
    corpus_path = case_dir / "corpus.json"
    queries_path = case_dir / "queries.json"
    index_dir = index_root / case_name
    doc_graph_path = index_dir / "doc_graph.pkl"
    case_out_dir = out_root / case_name
    out_path = case_out_dir / "retrieval_results.json"

    for required, label in [
        (corpus_path, "corpus file"),
        (queries_path, "queries file"),
        (index_dir, "index dir"),
        (doc_graph_path, "doc_graph.pkl"),
    ]:
        if not Path(required).exists():
            return {"case_name": case_name, "status": "skipped",
                    "reason": f"missing {label}: {required}"}

    documents = load_corpus(str(corpus_path))
    if not documents:
        return {"case_name": case_name, "status": "skipped",
                "reason": f"no documents in {corpus_path}"}

    query_bundle = load_query_bundle(str(queries_path))
    queries = query_bundle.get("queries", [])
    ref_date = (query_bundle.get("timeline") or {}).get("ref_date")

    if not queries:
        return {"case_name": case_name, "status": "skipped",
                "reason": f"no queries in {queries_path}"}

    if query_view_base_url:
        _api_key = os.getenv("VLLM_API_KEY", "EMPTY")
        qv_client = OpenAI(api_key=_api_key, base_url=query_view_base_url)
    else:
        qv_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    if filter_base_url:
        _filter_api_key = os.getenv("VLLM_API_KEY", "EMPTY")
        filter_client = OpenAI(api_key=_filter_api_key, base_url=filter_base_url)
    else:
        filter_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    query_generator = MultiViewQueryGenerator(
        client=qv_client,
        model=query_view_model,
        num_counter=num_counter,
        backend=query_view_backend,
        temperature=query_view_temperature,
        max_output_tokens=query_view_max_output_tokens,
    )

    dense = DenseRetriever(
        documents=documents,
        model_name=dense_model,
        index_path=index_dir / "dense.faiss",
        id_map_path=index_dir / "dense_doc_ids.pkl",
    )

    sparse = SparseRetriever(
        documents=documents,
        bm25_path=index_dir / "bm25.pkl",
        id_map_path=index_dir / "sparse_doc_ids.pkl",
    )

    fusion = WeightedRRFFusion(
        k=60,
        view_weights={"original": 1.0, "counter": 1.2},
    )

    doc_graph = DocGraph(graph_path=doc_graph_path)

    hop_agent = DocGraphHopAgent(
        doc_graph=doc_graph,
        max_hops=max_hops,
        neighbors_per_doc=neighbors_per_doc,
    )

    filter_agent = ConflictFilterAgent(
        client=filter_client,
        model=filter_model,
        top_n=filter_top_n,
        temperature=0.0,
    )

    planner_agent = ConflictPlannerAgent(
        client=filter_client,
        model=filter_model,
        temperature=0.6,
    )

    pipeline = RetrievalPipeline(
        query_generator=query_generator,
        dense_retriever=dense,
        sparse_retriever=sparse,
        fusion=fusion,
        doc_graph=doc_graph,
        hop_agent=hop_agent,
        filter_agent=filter_agent,
        planner_agent=planner_agent,
        ref_date=ref_date,
    )

    outputs = []
    n_queries = len(queries)
    for i, row in enumerate(queries, start=1):
        qid = row["query_id"]
        print(f"[{case_name}] query {i}/{n_queries} ({qid})", flush=True)
        result = pipeline.run_one(
            query_id=qid,
            query=row["query"],
            top_k_per_view=top_k_per_view,
            final_top_k=final_top_k,
        )
        n_docs = len(result.get("fused_results", []))
        print(f"[{case_name}] query {i}/{n_queries} ({qid}) done | docs={n_docs}", flush=True)
        outputs.append(result)

    case_out_dir.mkdir(parents=True, exist_ok=True)
    save_json(str(out_path), outputs)

    return {
        "case_name": case_name,
        "status": "ok",
        "num_documents": len(documents),
        "num_queries": len(queries),
        "out_path": str(out_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--index_root", type=str, required=True)
    parser.add_argument("--out_root", type=str, required=True)

    parser.add_argument("--query_view_model", type=str, default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--query_view_backend", type=str, default="openai_chat")
    parser.add_argument("--query_view_base_url", type=str, default="")
    parser.add_argument("--query_view_temperature", type=float, default=0.0)
    parser.add_argument("--query_view_max_output_tokens", type=int, default=300)
    parser.add_argument("--num_counter", type=int, default=3)

    parser.add_argument("--dense_model", type=str, default="Qwen/Qwen3-Embedding-8B")
    parser.add_argument("--top_k_per_view", type=int, default=10)
    parser.add_argument("--final_top_k", type=int, default=20)

    parser.add_argument("--max_hops", type=int, default=5)
    parser.add_argument("--neighbors_per_doc", type=int, default=3)
    parser.add_argument("--filter_model", type=str, default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--filter_base_url", type=str, default="",
                        help="vLLM base URL for ConflictFilterAgent (e.g. http://localhost:8001/v1)")
    parser.add_argument("--filter_top_n", type=int, default=10)

    parser.add_argument("--workers", type=int, default=1)

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    index_root = Path(args.index_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    case_dirs = sorted([p for p in data_dir.iterdir() if p.is_dir()])
    if not case_dirs:
        raise ValueError(f"No subdirectories in data_dir: {data_dir}")

    print(f"[INFO] {len(case_dirs)} cases | workers={args.workers}")
    print(f"[INFO] dense_model={args.dense_model}")
    print(f"[INFO] max_hops={args.max_hops} | neighbors_per_doc={args.neighbors_per_doc}")

    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                process_case,
                case_dir, index_root, out_root,
                args.dense_model,
                args.query_view_model, args.query_view_backend,
                args.query_view_base_url, args.query_view_temperature,
                args.query_view_max_output_tokens,
                args.top_k_per_view, args.final_top_k, args.num_counter,
                args.max_hops, args.neighbors_per_doc,
                args.filter_model, args.filter_base_url, args.filter_top_n,
            ): case_dir
            for case_dir in case_dirs
        }
        for future in as_completed(futures):
            case_name = futures[future].name
            try:
                result = future.result()
                results.append(result)
                if result["status"] == "ok":
                    print(f"[OK] {case_name}: {result['num_queries']} queries -> {result['out_path']}")
                else:
                    print(f"[SKIP] {case_name}: {result['reason']}")
            except Exception as e:
                import traceback
                print(f"[ERROR] {case_name}: {e}")
                traceback.print_exc()
                results.append({"case_name": case_name, "status": "error", "reason": str(e)})

    num_ok = sum(r["status"] == "ok" for r in results)
    num_skip = sum(r["status"] == "skipped" for r in results)
    num_err = sum(r["status"] == "error" for r in results)
    print(f"\n[DONE] ok={num_ok} skip={num_skip} error={num_err}")


if __name__ == "__main__":
    main()
