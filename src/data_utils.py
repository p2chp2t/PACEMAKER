from __future__ import annotations
import json
from pathlib import Path
from typing import Any
from src.types import Document


def load_json(path: str | Path) -> Any:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str | Path, data: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def save_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_label(label: str | None) -> str | None:
    if label is None:
        return None
    label = label.strip().lower()
    if label in {"conflict"}:
        return "conflict"
    if label in {"non_conflict", "non-conflict", "non conflict", "no_conflict", "no-conflict", "no conflict"}:
        return "non_conflict"
    return label


def load_corpus_bundle(path: str | Path) -> dict[str, Any]:
    data = load_json(path)
    timeline = data.get("timeline", [])
    documents: list[Document] = []
    if isinstance(data, dict) and "corpus" in data:
        rows = data["corpus"]
    elif isinstance(data, list):
        rows = data
    else:
        raise ValueError(f"Unsupported corpus format: {path}")

    for row in rows:
        metadata = {k: v for k, v in row.items() if k not in {"doc_id", "text"}}
        metadata["label"] = normalize_label(metadata["label"]) if "label" in metadata else None
        documents.append(
            Document(
                doc_id=row["doc_id"],
                text=row["text"],
                metadata=metadata,
            )
        )
    return {
        "timeline": timeline,
        "documents": documents,
    }


def load_corpus(path: str | Path) -> list[Document]:
    return load_corpus_bundle(path)["documents"]


def load_query_bundle(path: str | Path) -> dict[str, Any]:
    data = load_json(path)
    timeline = data.get("timeline", [])
    queries: list[dict[str, Any]] = []
    if isinstance(data, dict) and "queries" in data:
        rows = data["queries"]
    elif isinstance(data, list):
        rows = data
    else:
        raise ValueError(f"Unsupported query format: {path}")

    for i, row in enumerate(rows):
        query_id = row.get("query_id", f"q{i:03d}")
        query_text = row.get("query", row.get("text"))
        if not query_text:
            raise ValueError(f"Missing query text at row {i}")
        item = {
            "query_id": query_id,
            "query": query_text,
            **{k: v for k, v in row.items() if k not in {"query_id", "query", "text"}},
        }
        queries.append(item)
    return {
        "timeline": timeline,
        "queries": queries,
    }


def load_queries(path: str | Path) -> list[dict[str, Any]]:
    return load_query_bundle(path)["queries"]
