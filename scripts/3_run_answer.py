from __future__ import annotations

import argparse
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.answer_module import AnswerGenerator
from src.data_utils import load_json, load_query_bundle, save_json
from src.types import ScoredDoc

OPENAI_API_KEY = "" # Set your OpenAI API key here or use environment variable OPENAI_API_KEY. For vLLM, you can use a dummy key like "EMPTY" or set VLLM_API_KEY.


def load_persona_name(profile_path: Path) -> str:
    if not profile_path.exists():
        raise ValueError(f"Profile file not found: {profile_path}")
    profile = load_json(str(profile_path))
    try:
        persona_name = profile["spec"]["demographic"]["full_name"]
    except KeyError as e:
        raise ValueError(f"Fail to extract persona name from {profile_path}") from e
    return persona_name


def build_client(
        api_key: str | None = None,
        base_url: str | None = None,
) -> OpenAI:
    kwargs = {}

    if api_key:
        kwargs["api_key"] = api_key

    if base_url:
        kwargs["base_url"] = base_url

    return OpenAI(**kwargs)


def process_case(
        case_dir: Path,
        retrieval_root: Path,
        profile_dir: Path,
        out_root: Path,
        model: str,
        max_docs: int,
        backend: str,
        api_key: str | None,
        base_url: str | None,
) -> dict:
    case_name = case_dir.name
    queries_path = case_dir / "queries.json"
    retrieval_results_path = retrieval_root / case_name / "retrieval_results.json"
    profile_path = profile_dir / f"{case_name}.json"
    case_out_dir = out_root / case_name
    out_path = case_out_dir / "answer_results.json"

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

    if not profile_path.exists():
        return {
            "case_name": case_name,
            "status": "skipped",
            "reason": f"missing profile file: {profile_path}",
        }

    retrieval_results = load_json(str(retrieval_results_path))
    query_bundle = load_query_bundle(str(queries_path))
    timeline = query_bundle.get("timeline", {})
    ref_date = timeline.get("ref_date")

    if ref_date is None:
        return {
            "case_name": case_name,
            "status": "skipped",
            "reason": f"missing timeline.ref_date in {queries_path}",
        }

    persona_name = load_persona_name(profile_path)

    client = build_client(
        api_key=api_key,
        base_url=base_url,
    )

    answer_generator = AnswerGenerator(
        model=model,
        max_docs=max_docs,
        client=client,
        backend=backend,
    )

    outputs: list[dict] = []

    for row in retrieval_results:
        docs = [ScoredDoc(**d) for d in row["fused_results"]]

        answer = answer_generator.generate(
            query_id=row["query_id"],
            query=row["query"],
            docs=docs,
            persona_name=persona_name,
            ref_date=ref_date,
        )

        outputs.append(
            {
                **row,
                "answer": answer,
            }
        )

    case_out_dir.mkdir(parents=True, exist_ok=True)
    save_json(str(out_path), outputs)

    return {
        "case_name": case_name,
        "status": "ok",
        "persona_name": persona_name,
        "num_queries": len(outputs),
        "out_path": str(out_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--retrieval_root", type=str, required=True)
    parser.add_argument("--profile_dir", type=str, required=True)
    parser.add_argument("--out_root", type=str, required=True)
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--max_docs", type=int, default=10)

    parser.add_argument(
        "--backend",
        type=str,
        default="openai_chat",
        choices=["openai_responses", "openai_chat"],
    )

    parser.add_argument(
        "--api_key",
        type=str,
        default=OPENAI_API_KEY,
        help="OpenAI API key or dummy key for vLLM. If omitted, environment variable is used.",
    )
    parser.add_argument(
        "--base_url",
        type=str,
        default=None,
        help="Optional base URL for OpenAI-compatible servers like vLLM, e.g. http://localhost:8001/v1",
    )

    parser.add_argument("--workers", type=int, default=4)

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    retrieval_root = Path(args.retrieval_root)
    profile_dir = Path(args.profile_dir)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    api_key = args.api_key or os.environ.get("VLLM_API_KEY", "EMPTY")

    if args.backend == "openai_responses" and args.base_url:
        print(
            "[WARN] You set --backend openai_responses with a custom --base_url. "
            "For vLLM/Qwen/Llama servers, openai_chat is usually safer."
        )

    if args.backend == "openai_responses" and not api_key:
        raise ValueError(
            "No API key found. Provide --api_key or set OPENAI_API_KEY "
            "when using backend=openai_responses."
        )

    case_dirs = sorted([p for p in data_dir.iterdir() if p.is_dir()])
    if not case_dirs:
        raise ValueError(f"No subdirectories found in data_dir: {data_dir}")

    print(f"[INFO] Found {len(case_dirs)} case directories under {data_dir}")
    print(f"[INFO] Using {args.workers} workers")
    print(f"[INFO] Model: {args.model}")
    print(f"[INFO] Backend: {args.backend}")
    print(f"[INFO] Base URL: {args.base_url or '(default OpenAI)'}")

    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                process_case,
                case_dir,
                retrieval_root,
                profile_dir,
                out_root,
                args.model,
                args.max_docs,
                args.backend,
                api_key,
                args.base_url,
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
                        f"[OK] {case_name}: saved answer results "
                        f"({result['num_queries']} queries, persona={result['persona_name']}) "
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

    print("\n[INFO] All answer runs complete.")
    print(f"[INFO] Success: {num_ok}")
    print(f"[INFO] Skipped: {num_skip}")
    print(f"[INFO] Errors: {num_error}")


if __name__ == "__main__":
    main()
