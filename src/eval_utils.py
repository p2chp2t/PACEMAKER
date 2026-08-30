from __future__ import annotations


def hit_at_k(pred_doc_ids: list[str], gold_doc_ids: list[str], k: int) -> float:
    pred_topk = pred_doc_ids[:k]
    gold_set = set(gold_doc_ids)
    if not gold_set:
        return 0.0
    return 1.0 if any(doc_id in gold_set for doc_id in pred_topk) else 0.0


def recall_at_k(pred_doc_ids: list[str], gold_doc_ids: list[str], k: int) -> float:
    gold_set = set(gold_doc_ids)
    if not gold_set:
        return 0.0
    pred_topk = set(pred_doc_ids[:k])
    return len(pred_topk & gold_set) / len(gold_set)


def precision_at_k(pred_doc_ids: list[str], gold_doc_ids: list[str], k: int) -> float:
    if k <= 0:
        return 0.0

    pred_topk = pred_doc_ids[:k]
    if not pred_topk:
        return 0.0

    gold_set = set(gold_doc_ids)
    if not gold_set:
        return 0.0

    return len(set(pred_topk) & gold_set) / len(pred_topk)


def all_gold_at_k(pred_doc_ids: list[str], gold_doc_ids: list[str], k: int) -> float:
    gold_set = set(gold_doc_ids)
    if not gold_set:
        return 0.0

    pred_topk = set(pred_doc_ids[:k])
    return 1.0 if gold_set.issubset(pred_topk) else 0.0


def mrr(pred_doc_ids: list[str], gold_doc_ids: list[str]) -> float:
    gold_set = set(gold_doc_ids)
    if not gold_set:
        return 0.0

    for rank, doc_id in enumerate(pred_doc_ids, start=1):
        if doc_id in gold_set:
            return 1.0 / rank

    return 0.0


def evaluate_retrieval(
        predictions: list[dict],
        gold_map: dict[str, list[str]],
        ks: list[int] | None = None,
) -> dict[str, float]:
    if ks is None:
        ks = [5, 10]

    metrics: dict[str, float] = {}

    for k in ks:
        metrics[f"hit@{k}"] = 0.0
        metrics[f"recall@{k}"] = 0.0
        metrics[f"precision@{k}"] = 0.0
        metrics[f"all_gold@{k}"] = 0.0

    metrics["mrr"] = 0.0

    if not predictions:
        return metrics

    n = len(predictions)

    for row in predictions:
        query_id = row["query_id"]
        pred_doc_ids = [d["doc_id"] for d in row["fused_results"]]
        gold_doc_ids = gold_map.get(query_id, [])

        for k in ks:
            metrics[f"hit@{k}"] += hit_at_k(pred_doc_ids, gold_doc_ids, k)
            metrics[f"recall@{k}"] += recall_at_k(pred_doc_ids, gold_doc_ids, k)
            metrics[f"precision@{k}"] += precision_at_k(pred_doc_ids, gold_doc_ids, k)
            metrics[f"all_gold@{k}"] += all_gold_at_k(pred_doc_ids, gold_doc_ids, k)

        metrics["mrr"] += mrr(pred_doc_ids, gold_doc_ids)

    for key in metrics:
        metrics[key] /= n

    return metrics


def evaluate_judge_evals(judge_results: list[dict]) -> dict:
    total = len(judge_results)

    pass_count = sum(
        1 for r in judge_results
        if r["judge_eval"]["judge"] == "PASS"
    )
    wrong_count = sum(
        1 for r in judge_results
        if r["judge_eval"]["judge"] == "WRONG_RATIONALE"
    )
    fail_count = sum(
        1 for r in judge_results
        if r["judge_eval"]["judge"] == "FAIL"
    )

    conflict_rows = [
        r for r in judge_results
        if r["judge_eval"].get("gold_label") == "conflict"
    ]
    non_conflict_rows = [
        r for r in judge_results
        if r["judge_eval"].get("gold_label") == "non_conflict"
    ]

    def count_label(rows: list[dict], label: str) -> int:
        return sum(
            1 for r in rows
            if r["judge_eval"]["judge"] == label
        )

    conflict_pass = count_label(conflict_rows, "PASS")
    conflict_wrong = count_label(conflict_rows, "WRONG_RATIONALE")
    conflict_fail = count_label(conflict_rows, "FAIL")

    non_conflict_pass = count_label(non_conflict_rows, "PASS")
    non_conflict_wrong = count_label(non_conflict_rows, "WRONG_RATIONALE")
    non_conflict_fail = count_label(non_conflict_rows, "FAIL")

    pass_rate = pass_count / total if total else 0.0
    wrong_rate = wrong_count / total if total else 0.0
    fail_rate = fail_count / total if total else 0.0

    return {
        "TOTAL": total,
        "TOTAL_CONFLICT": len(conflict_rows),
        "TOTAL_NON_CONFLICT": len(non_conflict_rows),

        "PASS": pass_count,
        "WRONG_RATIONALE": wrong_count,
        "FAIL": fail_count,

        "CONFLICT_PASS": conflict_pass,
        "CONFLICT_WRONG_RATIONALE": conflict_wrong,
        "CONFLICT_FAIL": conflict_fail,

        "NON_CONFLICT_PASS": non_conflict_pass,
        "NON_CONFLICT_WRONG_RATIONALE": non_conflict_wrong,
        "NON_CONFLICT_FAIL": non_conflict_fail,

        "PASS_rate": pass_rate,
        "WRONG_RATIONALE_rate": wrong_rate,
        "FAIL_rate": fail_rate,
    }
