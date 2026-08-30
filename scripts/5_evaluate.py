from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.data_utils import load_json, load_queries, save_json
from src.eval_utils import evaluate_judge_evals, evaluate_retrieval


def build_gold_maps(queries: list[dict]) -> tuple[dict[str, list[str]], dict[str, str]]:
    gold_doc_map: dict[str, list[str]] = {}
    gold_decision_map: dict[str, str] = {}

    for row in queries:
        query_id = row["query_id"]
        gold_doc_map[query_id] = row.get("gold_doc_ids", [])

        label = row.get("label")
        if label == "conflict":
            gold_decision_map[query_id] = "NO"
        elif label == "non_conflict":
            gold_decision_map[query_id] = "YES"

    return gold_doc_map, gold_decision_map


def process_case(
        case_dir: Path,
        retrieval_root: Path,
        out_root: Path,
        answer_root: Path | None = None,
        judge_root: Path | None = None,
        ks: list[int] | None = None,
) -> dict:
    case_name = case_dir.name
    queries_path = case_dir / "queries.json"
    retrieval_results_path = retrieval_root / case_name / "retrieval_results.json"

    # Kept for backward compatibility with old scripts.
    answer_results_path = (
        answer_root / case_name / "answer_results.json"
        if answer_root is not None
        else None
    )

    judge_eval_results_path = (
        judge_root / case_name / "judge_eval_results.json"
        if judge_root is not None
        else None
    )

    case_out_dir = out_root / case_name
    out_path = case_out_dir / "eval_report.json"

    if ks is None:
        ks = [5, 10]

    if not queries_path.exists():
        return {
            "case_name": case_name,
            "status": "skipped",
            "reason": f"missing queries file: {queries_path}",
        }

    if not retrieval_results_path.exists():
        return {
            "case_name": case_name,
            "status": "skipped",
            "reason": f"missing retrieval results: {retrieval_results_path}",
        }

    queries = load_queries(str(queries_path))
    retrieval_results = load_json(str(retrieval_results_path))

    judge_eval_results = (
        load_json(str(judge_eval_results_path))
        if judge_eval_results_path is not None and judge_eval_results_path.exists()
        else None
    )

    gold_doc_map, _ = build_gold_maps(queries)

    report: dict = {}

    report["retrieval"] = evaluate_retrieval(
        predictions=retrieval_results,
        gold_map=gold_doc_map,
        ks=ks,
    )

    if judge_eval_results is not None:
        report["judge_eval"] = evaluate_judge_evals(judge_eval_results)

    case_out_dir.mkdir(parents=True, exist_ok=True)
    save_json(str(out_path), report)

    return {
        "case_name": case_name,
        "status": "ok",
        "has_judge_eval": judge_eval_results is not None,
        "out_path": str(out_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--retrieval_root", type=str, required=True)
    parser.add_argument("--answer_root", type=str, default=None)
    parser.add_argument("--judge_eval_root", type=str, default=None)
    parser.add_argument("--out_root", type=str, required=True)
    parser.add_argument("--workers", type=int, default=8)

    parser.add_argument(
        "--ks",
        type=int,
        nargs="+",
        default=[5, 10],
        help="Cutoffs for retrieval evaluation metrics.",
    )

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    retrieval_root = Path(args.retrieval_root)
    answer_root = Path(args.answer_root) if args.answer_root else None
    judge_root = Path(args.judge_eval_root) if args.judge_eval_root else None
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    case_dirs = sorted([p for p in data_dir.iterdir() if p.is_dir()])
    if not case_dirs:
        raise ValueError(f"No subdirectories found in data_dir: {data_dir}")

    print(f"[INFO] Found {len(case_dirs)} case directories under {data_dir}")
    print(f"[INFO] Using {args.workers} workers")
    print(f"[INFO] Retrieval eval ks: {args.ks}")

    results = []

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                process_case,
                case_dir,
                retrieval_root,
                out_root,
                answer_root,
                judge_root,
                args.ks,
            ): case_dir
            for case_dir in case_dirs
        }

        for future in as_completed(futures):
            case_dir = futures[future]
            case_name = case_dir.name

            try:
                result = future.result()
                results.append(result)

                if result["status"] == "ok":
                    print(
                        f"[OK] {case_name}: saved eval report "
                        f"(judge_eval={result['has_judge_eval']}) "
                        f"-> {result['out_path']}"
                    )
                else:
                    print(f"[SKIP] {case_name}: {result['reason']}")

            except Exception as e:
                print(f"[ERROR] {case_name}: {e}")
                results.append(
                    {
                        "case_name": case_name,
                        "status": "error",
                        "reason": str(e),
                    }
                )

    num_ok = sum(r["status"] == "ok" for r in results)
    num_skip = sum(r["status"] == "skipped" for r in results)
    num_error = sum(r["status"] == "error" for r in results)

    print("\n[INFO] All evaluations complete.")
    print(f"[INFO] Success: {num_ok}")
    print(f"[INFO] Skipped: {num_skip}")
    print(f"[INFO] Errors: {num_error}")


if __name__ == "__main__":
    main()
