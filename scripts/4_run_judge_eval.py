from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from pathlib import Path
import sys

from openai import OpenAI

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.data_utils import load_json, load_queries, save_json
from src.judge_module import JudgeEvaluator


def process_case(
        case_dir: Path,
        answer_root: Path,
        out_root: Path,
        model: str,
        api_key: str,
) -> dict:
    case_name = case_dir.name
    queries_path = case_dir / "queries.json"
    answer_results_path = answer_root / case_name / "answer_results.json"
    case_out_dir = out_root / case_name
    out_path = case_out_dir / "judge_eval_results.json"

    if not queries_path.exists():
        return {
            "case_name": case_name,
            "status": "skipped",
            "reason": f"missing queries file: {queries_path}",
        }

    if not answer_results_path.exists():
        return {
            "case_name": case_name,
            "status": "skipped",
            "reason": f"missing answer results: {answer_results_path}",
        }

    answer_results = load_json(str(answer_results_path))
    queries = load_queries(str(queries_path))
    query_lookup = {q["query_id"]: q for q in queries}

    client = OpenAI(api_key=api_key) if api_key else OpenAI()
    evaluator = JudgeEvaluator(model=model, client=client)

    outputs: list[dict] = []

    for row in answer_results:
        query_id = row.get("query_id")
        query = query_lookup.get(query_id, {}).get("query", "") if query_id else ""

        if query_id not in query_lookup:
            outputs.append(
                {
                    **row,
                    "judge_eval": {
                        "query_id": query_id,
                        "judge": "FAIL",
                        "expected_decision": None,
                        "decision_correct": False,
                        "gold_label": "",
                        "error": f"query_id not found in queries.json: {query_id}",
                    },
                }
            )
            continue

        q = query_lookup[query_id]

        judge_eval = evaluator.evaluate(
            query_id=query_id,
            answer_text=row.get("answer", ""),
            gold_judgment=q.get("judgment", ""),
            gold_label=q.get("label"),
        )

        outputs.append(
            {
                **row,
                "judge_eval": judge_eval.to_dict(),
            }
        )

    case_out_dir.mkdir(parents=True, exist_ok=True)
    save_json(str(out_path), outputs)

    return {
        "case_name": case_name,
        "status": "ok",
        "num_queries": len(outputs),
        "out_path": str(out_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--answer_root", type=str, required=True)
    parser.add_argument("--model", type=str, default="gpt-5.4-mini")
    parser.add_argument("--out_root", type=str, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--api_key",
        type=str,
        default="",
        help="OpenAI API key for judge eval. Can also be set via OPENAI_API_KEY env var.",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    answer_root = Path(args.answer_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    api_key = args.api_key or os.getenv("OPENAI_API_KEY", "")

    case_dirs = sorted([p for p in data_dir.iterdir() if p.is_dir()])
    if not case_dirs:
        raise ValueError(f"No subdirectories found in data_dir: {data_dir}")

    print(f"[INFO] Found {len(case_dirs)} case directories under {data_dir}")
    print(f"[INFO] Using {args.workers} workers")

    results = []

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                process_case,
                case_dir,
                answer_root,
                out_root,
                args.model,
                api_key,
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
                        f"[OK] {case_name}: saved judge eval results "
                        f"({result['num_queries']} queries) -> {result['out_path']}"
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

    print("\n[INFO] All judge eval runs complete.")
    print(f"[INFO] Success: {num_ok}")
    print(f"[INFO] Skipped: {num_skip}")
    print(f"[INFO] Errors: {num_error}")


if __name__ == "__main__":
    main()
