from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_reports(eval_root: Path) -> list[tuple[str, dict]]:
    reports = []
    for p in sorted(eval_root.glob("*/eval_report.json")):
        case_name = p.parent.name
        with open(p) as f:
            reports.append((case_name, json.load(f)))
    return reports


def _is_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _sort_metric_keys(keys: list[str]) -> list[str]:
    prefix_order = {
        "hit": 0,
        "recall": 1,
        "precision": 2,
        "all_gold": 3,
        "mrr": 4,
    }

    def key_fn(k: str) -> tuple[int, int, str]:
        if "@" in k:
            prefix, cutoff = k.split("@", 1)
            try:
                cutoff_i = int(cutoff)
            except ValueError:
                cutoff_i = 10 ** 9
            return (prefix_order.get(prefix, 100), cutoff_i, k)

        return (prefix_order.get(k, 100), 0, k)

    return sorted(keys, key=key_fn)


def _collect_retrieval_keys(reports: list[tuple[str, dict]]) -> list[str]:
    keys: set[str] = set()

    for _, report in reports:
        retrieval = report.get("retrieval")
        if not isinstance(retrieval, dict):
            continue

        for k, v in retrieval.items():
            if _is_number(v):
                keys.add(k)

    return _sort_metric_keys(list(keys))


def _safe_rate(num: int, den: int) -> float:
    return round(num / den, 4) if den else 0.0


def _get_int(d: dict, key: str) -> int:
    return int(d.get(key, 0) or 0)


def _add_simple_rates(
        target: dict,
        *,
        prefix: str,
        total: int,
        correct: int,
        wrong: int,
        fail: int,
) -> None:
    target[f"{prefix}pass_rate"] = _safe_rate(correct, total)
    target[f"{prefix}wrong_rate"] = _safe_rate(wrong, total)
    target[f"{prefix}fail_rate"] = _safe_rate(fail, total)


def aggregate(reports: list[tuple[str, dict]]) -> dict:
    retrieval_keys = _collect_retrieval_keys(reports)

    ret_sums: dict[str, float] = {k: 0.0 for k in retrieval_keys}
    ret_counts: dict[str, int] = {k: 0 for k in retrieval_keys}

    judge_counts: dict[str, int] = {
        "TOTAL": 0,
        "TOTAL_CONFLICT": 0,
        "TOTAL_NON_CONFLICT": 0,

        "PASS": 0,
        "WRONG_RATIONALE": 0,
        "FAIL": 0,

        "CONFLICT_PASS": 0,
        "CONFLICT_WRONG_RATIONALE": 0,
        "CONFLICT_FAIL": 0,

        "NON_CONFLICT_PASS": 0,
        "NON_CONFLICT_WRONG_RATIONALE": 0,
        "NON_CONFLICT_FAIL": 0,
    }

    ret_n = 0
    judge_n = 0
    per_case = []

    for case_name, report in reports:
        row: dict = {"case": case_name}

        retrieval = report.get("retrieval")
        if isinstance(retrieval, dict):
            ret_n += 1

            for k in retrieval_keys:
                if k in retrieval and _is_number(retrieval[k]):
                    v = float(retrieval[k])
                    ret_sums[k] += v
                    ret_counts[k] += 1
                    row[k] = round(v, 4)

        judge_eval = report.get("judge_eval")
        if isinstance(judge_eval, dict):
            judge_n += 1

            pass_correct = _get_int(judge_eval, "PASS")
            wrong = _get_int(judge_eval, "WRONG_RATIONALE")
            fail = _get_int(judge_eval, "FAIL")
            total = _get_int(judge_eval, "TOTAL")
            if total == 0:
                total = pass_correct + wrong + fail

            conflict_pass = _get_int(judge_eval, "CONFLICT_PASS")
            conflict_wrong = _get_int(judge_eval, "CONFLICT_WRONG_RATIONALE")
            conflict_fail = _get_int(judge_eval, "CONFLICT_FAIL")
            total_conflict = _get_int(judge_eval, "TOTAL_CONFLICT")
            if total_conflict == 0:
                total_conflict = conflict_pass + conflict_wrong + conflict_fail

            non_conflict_pass = _get_int(judge_eval, "NON_CONFLICT_PASS")
            non_conflict_wrong = _get_int(judge_eval, "NON_CONFLICT_WRONG_RATIONALE")
            non_conflict_fail = _get_int(judge_eval, "NON_CONFLICT_FAIL")
            total_non_conflict = _get_int(judge_eval, "TOTAL_NON_CONFLICT")
            if total_non_conflict == 0:
                total_non_conflict = (
                        non_conflict_pass
                        + non_conflict_wrong
                        + non_conflict_fail
                )

            case_counts = {
                "TOTAL": total,
                "TOTAL_CONFLICT": total_conflict,
                "TOTAL_NON_CONFLICT": total_non_conflict,

                "PASS": pass_correct,
                "WRONG_RATIONALE": wrong,
                "FAIL": fail,

                "CONFLICT_PASS": conflict_pass,
                "CONFLICT_WRONG_RATIONALE": conflict_wrong,
                "CONFLICT_FAIL": conflict_fail,

                "NON_CONFLICT_PASS": non_conflict_pass,
                "NON_CONFLICT_WRONG_RATIONALE": non_conflict_wrong,
                "NON_CONFLICT_FAIL": non_conflict_fail,
            }

            for k, v in case_counts.items():
                judge_counts[k] += v
                row[k] = v

            _add_simple_rates(
                row,
                prefix="",
                total=total,
                correct=pass_correct,
                wrong=wrong,
                fail=fail,
            )
            _add_simple_rates(
                row,
                prefix="conflict_",
                total=total_conflict,
                correct=conflict_pass,
                wrong=conflict_wrong,
                fail=conflict_fail,
            )
            _add_simple_rates(
                row,
                prefix="non_conflict_",
                total=total_non_conflict,
                correct=non_conflict_pass,
                wrong=non_conflict_wrong,
                fail=non_conflict_fail,
            )

        per_case.append(row)

    macro: dict = {}

    if ret_n:
        macro["retrieval"] = {
            k: round(ret_sums[k] / ret_counts[k], 4) if ret_counts[k] else 0.0
            for k in retrieval_keys
        }

    if judge_n:
        total = judge_counts["TOTAL"]
        total_conflict = judge_counts["TOTAL_CONFLICT"]
        total_non_conflict = judge_counts["TOTAL_NON_CONFLICT"]

        pass_correct = judge_counts["PASS"]
        wrong = judge_counts["WRONG_RATIONALE"]
        fail = judge_counts["FAIL"]

        conflict_pass = judge_counts["CONFLICT_PASS"]
        conflict_wrong = judge_counts["CONFLICT_WRONG_RATIONALE"]
        conflict_fail = judge_counts["CONFLICT_FAIL"]

        non_conflict_pass = judge_counts["NON_CONFLICT_PASS"]
        non_conflict_wrong = judge_counts["NON_CONFLICT_WRONG_RATIONALE"]
        non_conflict_fail = judge_counts["NON_CONFLICT_FAIL"]

        judge_macro = {**judge_counts}

        _add_simple_rates(
            judge_macro,
            prefix="",
            total=total,
            correct=pass_correct,
            wrong=wrong,
            fail=fail,
        )

        _add_simple_rates(
            judge_macro,
            prefix="conflict_",
            total=total_conflict,
            correct=conflict_pass,
            wrong=conflict_wrong,
            fail=conflict_fail,
        )

        _add_simple_rates(
            judge_macro,
            prefix="non_conflict_",
            total=total_non_conflict,
            correct=non_conflict_pass,
            wrong=non_conflict_wrong,
            fail=non_conflict_fail,
        )

        macro["judge_eval"] = judge_macro

    return {
        "num_cases": len(reports),
        "num_cases_with_retrieval": ret_n,
        "num_cases_with_judge_eval": judge_n,
        "macro_avg": macro,
        "per_case": per_case,
    }


def pretty_print(result: dict) -> None:
    n = result["num_cases"]
    macro = result["macro_avg"]

    print(f"\n{'=' * 72}")
    print(f"  Aggregate Report  ({n} cases)")
    print(f"{'=' * 72}")

    if "retrieval" in macro:
        r = macro["retrieval"]
        print("\n[Retrieval]  (macro average across cases)")
        for k in _sort_metric_keys(list(r.keys())):
            print(f"  {k:<18}: {r[k]:.4f}")

    if "judge_eval" in macro:
        j = macro["judge_eval"]

        print("\n[Judge Eval]  (micro counts across all queries)")
        print(f"  TOTAL              : {j['TOTAL']:>5}")
        print(f"  TOTAL_CONFLICT     : {j['TOTAL_CONFLICT']:>5}")
        print(f"  TOTAL_NON_CONFLICT : {j['TOTAL_NON_CONFLICT']:>5}")
        print(
            f"  PASS               : {j['PASS']:>5}  |  "
            f"WRONG_RATIONALE : {j['WRONG_RATIONALE']:>5}  |  "
            f"FAIL : {j['FAIL']:>5}"
        )

        print("\n[Judge Eval]  (overall rates)")
        print(f"  pass_rate       : {j['pass_rate']:.4f}")
        print(f"  wrong_rate         : {j['wrong_rate']:.4f}")
        print(f"  fail_rate          : {j['fail_rate']:.4f}")

        print("\n[Judge Eval]  (by type)")
        print("  [conflict]")
        print(f"    pass_rate     : {j['conflict_pass_rate']:.4f}")
        print(f"    wrong_rate       : {j['conflict_wrong_rate']:.4f}")
        print(f"    fail_rate        : {j['conflict_fail_rate']:.4f}")

        print("  [non_conflict]")
        print(f"    pass_rate     : {j['non_conflict_pass_rate']:.4f}")
        print(f"    wrong_rate       : {j['non_conflict_wrong_rate']:.4f}")
        print(f"    fail_rate        : {j['non_conflict_fail_rate']:.4f}")

        print("\n  *pass_rate = PASS / TOTAL")
        print("  *wrong_rate   = WRONG_RATIONALE / TOTAL")
        print("  *fail_rate    = FAIL / TOTAL")

    print()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--eval_root",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
    )
    args = parser.parse_args()

    eval_root = Path(args.eval_root)
    reports = load_reports(eval_root)

    if not reports:
        print(f"[ERROR] Not found eval_report.json: {eval_root}")
        return

    print(f"[INFO] {len(reports)} case loaded")

    result = aggregate(reports)
    pretty_print(result)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"[INFO] SAVE: {out_path}")


if __name__ == "__main__":
    main()
