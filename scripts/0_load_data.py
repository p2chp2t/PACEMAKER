from __future__ import annotations
import argparse
import json
from collections import defaultdict
from pathlib import Path
from datasets import load_dataset


DATASET_NAME = "p2chp2t/pace"
TIMELINE_FIELDS = {"window_start", "ref_date", "window_end"}


def save_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def extract_timeline(row):
    return {
        "window_start": row["window_start"],
        "ref_date": row["ref_date"],
        "window_end": row["window_end"],
    }


def load_queries(output_dir: Path):
    dataset = load_dataset(
        DATASET_NAME,
        "queries",
        split="test",
    )

    grouped = defaultdict(list)
    timelines = {}

    for row in dataset:
        case_id = row["case_id"]
        timelines[case_id] = extract_timeline(row)
        query = {
            k: v
            for k, v in row.items()
            if k != "case_id" and k not in TIMELINE_FIELDS
        }
        grouped[case_id].append(query)

    for case_id, queries in grouped.items():
        save_json(
            output_dir / "kb" / case_id / "queries.json",
            {
                "timeline": timelines[case_id],
                "queries": queries,
            },
        )


def load_corpus(output_dir: Path):
    dataset = load_dataset(
        DATASET_NAME,
        "corpus",
        split="test",
    )

    grouped = defaultdict(list)
    timelines = {}

    for row in dataset:
        case_id = row["case_id"]
        timelines[case_id] = extract_timeline(row)
        document = {
            k: v
            for k, v in row.items()
            if k != "case_id" and k not in TIMELINE_FIELDS
        }
        grouped[case_id].append(document)

    for case_id, corpus in grouped.items():
        save_json(
            output_dir / "kb" / case_id / "corpus.json",
            {
                "timeline": timelines[case_id],
                "corpus": corpus,
            },
        )


def load_profiles(output_dir: Path):
    dataset = load_dataset(
        DATASET_NAME,
        "profiles",
        split="test",
    )

    for row in dataset:
        case_id = row["case_id"]
        profile = {
            k: v
            for k, v in row.items()
            if k != "case_id"
        }
        save_json(
            output_dir / "profile" / f"{case_id}.json",
            profile,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("./data/pace"),
    )
    args = parser.parse_args()

    print("Downloading PACE from Hugging Face...")
    load_queries(args.output_dir)
    load_corpus(args.output_dir)
    load_profiles(args.output_dir)

    print(f"PACE is successfully loaded at: {args.output_dir}")


if __name__ == "__main__":
    main()