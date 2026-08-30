from __future__ import annotations

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.data_utils import load_corpus
from src.retrieval_module import DenseRetriever, SparseRetriever


def process_case(
        case_dir: Path,
        index_root: Path,
        dense_model: str,
        knn_k: int,
        batch_size: int,
) -> dict:
    case_name = case_dir.name
    corpus_path = case_dir / "corpus.json"

    if not corpus_path.exists():
        return {"case_name": case_name, "status": "skipped", "reason": f"missing corpus: {corpus_path}"}

    documents = load_corpus(str(corpus_path))
    if not documents:
        return {"case_name": case_name, "status": "skipped", "reason": f"no documents in {corpus_path}"}

    out_dir = index_root / case_name
    out_dir.mkdir(parents=True, exist_ok=True)

    dense_index_path = out_dir / "dense.faiss"
    dense_id_map_path = out_dir / "dense_doc_ids.pkl"
    sparse_bm25_path = out_dir / "bm25.pkl"
    sparse_id_map_path = out_dir / "sparse_doc_ids.pkl"
    doc_graph_path = out_dir / "doc_graph.pkl"

    print(f"[{case_name}] building dense index + doc graph (k={knn_k})...", flush=True)
    DenseRetriever.build_and_save(
        documents=documents,
        model_name=dense_model,
        index_path=dense_index_path,
        id_map_path=dense_id_map_path,
        graph_path=doc_graph_path,
        knn_k=knn_k,
        batch_size=batch_size,
    )

    print(f"[{case_name}] building sparse (BM25) index...", flush=True)
    SparseRetriever.build_and_save(
        documents=documents,
        bm25_path=sparse_bm25_path,
        id_map_path=sparse_id_map_path,
    )

    return {
        "case_name": case_name,
        "status": "ok",
        "num_documents": len(documents),
        "dense_index": str(dense_index_path),
        "doc_graph": str(doc_graph_path),
        "sparse_index": str(sparse_bm25_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--index_root", type=str, required=True)
    parser.add_argument("--dense_model", type=str, default="Qwen/Qwen3-Embedding-8B")
    parser.add_argument("--knn_k", type=int, default=10,
                        help="Number of nearest neighbors per doc in the doc graph")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Encoding batch size (reduce if OOM)")
    parser.add_argument("--workers", type=int, default=1,
                        help="Parallel cases (keep 1 if GPU memory is tight)")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    index_root = Path(args.index_root)
    index_root.mkdir(parents=True, exist_ok=True)

    case_dirs = sorted([p for p in data_dir.iterdir() if p.is_dir()])
    if not case_dirs:
        raise ValueError(f"No subdirectories in data_dir: {data_dir}")

    print(f"[INFO] {len(case_dirs)} cases | workers={args.workers}")
    print(f"[INFO] dense_model={args.dense_model} | knn_k={args.knn_k}")

    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                process_case,
                case_dir,
                index_root,
                args.dense_model,
                args.knn_k,
                args.batch_size,
            ): case_dir
            for case_dir in case_dirs
        }
        for future in as_completed(futures):
            case_name = futures[future].name
            try:
                result = future.result()
                results.append(result)
                if result["status"] == "ok":
                    print(f"[OK] {case_name}: {result['num_documents']} docs | "
                          f"graph={result['doc_graph']}")
                else:
                    print(f"[SKIP] {case_name}: {result['reason']}")
            except Exception as e:
                print(f"[ERROR] {case_name}: {e}")
                results.append({"case_name": case_name, "status": "error", "reason": str(e)})

    num_ok = sum(r["status"] == "ok" for r in results)
    num_skip = sum(r["status"] == "skipped" for r in results)
    num_err = sum(r["status"] == "error" for r in results)
    print(f"\n[INFO] All index builds complete. ok={num_ok} skip={num_skip} error={num_err}")


if __name__ == "__main__":
    main()
