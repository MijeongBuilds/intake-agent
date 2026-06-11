"""
run.py — runs the agent's CLASSIFY step on every case in golden_dataset/
and prints the result next to the ground truth.

HOW TO RUN (from the agent/ folder):
    .venv/bin/python run.py
"""

import json
from pathlib import Path

from agent_engine import check_readability, classify, extract, evaluate_gates, MODEL

GOLDEN = Path(__file__).parent / "golden_dataset"


def _mark(ok: bool) -> str:
    return "✅" if ok else "❌"


def _check(label: str, got, expected) -> None:
    """Print one extracted field next to its ground-truth value with a pass mark."""
    print(f"    {label:<26}: {str(got):<40} {_mark(got == expected)}  (gt: {expected})")


def main() -> None:
    print(f"\nModel: {MODEL}")
    print("=" * 60)

    for case_file in sorted(GOLDEN.glob("case_*.json")):
        case = json.load(case_file.open())
        gt = case["ground_truth"]
        tenant = case["input_metadata"]["tenant_id"]

        # --- 0. pre-classify readability gate (halt unreadable before any LLM call) ---
        readability = check_readability(case["input_metadata"], tenant)
        if not readability["readable"]:
            print(f"\n{case['case_id']}")
            print(f"  ⛔ HALTED — {readability['reason']}")
            print(f"  ROUTING    : Unreadable Queue  (expected: {gt['expected_processing']['routing_target']})")
            continue

        # --- 1. classify ---
        result = classify(case["input_text"], tenant_id=tenant)

        cls_ok = result.predicted_class.value == gt["predicted_class"]
        ae_ok = result.adverse_event_flag == gt["adverse_event_flag"]

        print(f"\n{case['case_id']}")
        print(f"  class      : {result.predicted_class.value}  {_mark(cls_ok)}  (expected: {gt['predicted_class']})")
        print(f"  AE flag    : {result.adverse_event_flag}  {_mark(ae_ok)}  (expected: {gt['adverse_event_flag']})")
        print(f"  confidence : {result.class_confidence:.2f}   conflict margin: {result.classification_conflict_margin:.2f}")

        # --- 2. extract (always — full draft for every path) ---
        common, record = extract(case["input_text"], result, case["input_metadata"])

        print("  EXTRACT — common metadata:")
        gt_common = gt["common_metadata"]
        for field in ("customer_org", "reporter_name", "reporter_type",
                      "contact_email", "contact_phone", "country_code",
                      "language", "product_mentioned"):
            got = getattr(common, field)
            got = got.value if hasattr(got, "value") else got
            _check(field, got, gt_common.get(field))

        if record is not None:
            print(f"  EXTRACT — {record.record_type} record:")
            gt_record = gt.get("on_label_record") or gt.get("ae_record") or {}
            for field, value in record.model_dump().items():
                if field == "record_type":
                    continue
                got = value.value if hasattr(value, "value") else value
                exp = gt_record.get(field, "—")
                _check(field, got, exp)

        # --- 3. 7-gate compliance unit -> terminal decision ---
        gates = evaluate_gates(result, common, record, case["input_metadata"], tenant)
        gt_proc = gt["expected_processing"]
        print("  GATES — terminal decision:")
        _check("catalog_match_result", gates["catalog_match_result"], gt_proc.get("catalog_match_result"))
        _check("hcp_system_id", gates["hcp_system_id"], gt_proc.get("hcp_system_id"))
        _check("record_status", gates["record_status"], gt_proc.get("record_status"))
        _check("routing_target", gates["routing_target"], gt_proc.get("routing_target"))
        for g in gates["failed_gates"]:
            print(f"               - {g}")

    print("\n" + "=" * 60)
    print("Done.\n")


if __name__ == "__main__":
    main()
