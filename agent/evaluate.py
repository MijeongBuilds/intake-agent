"""
evaluate.py — the eval harness.

Two jobs:
  1. run_case()  — run the full pipeline on ONE document. The single shared runner;
     the Streamlit app caches a thin wrapper around this.
  2. evaluate_all() + aggregate_metrics() — score EVERY golden case and compute the
     PROVEN-IF metrics collectively.

AUTO-SCALES: reads every golden_dataset/case_*.json. Drop in a new case file and it is
included in the next eval — no code changes as the dataset grows.

Run as a CLI for a quick scorecard:
    .venv/bin/python evaluate.py
"""

import json
from pathlib import Path

from pydantic import ValidationError

from agent_engine import check_readability, classify, extract, evaluate_gates, MODEL
from graders import grade_case

GOLDEN = Path(__file__).parent / "golden_dataset"

# ============================================================
# Field comparison — the SINGLE source of truth (shared with the UI)
# ============================================================
# Free-text / list fields judged by meaning, not string equality. Excluded from the
# exact-match accuracy counts (scored separately / by hand / LLM-judge).
SEMANTIC_FIELDS = {
    "inquiry_summary_text", "clinical_justification", "adverse_symptoms_list",
    "seriousness_rationale", "off_label_indication", "proposed_study_title",
}
_MISSING = object()

META_FIELDS = ["customer_org", "reporter_name", "reporter_type", "contact_email",
               "contact_phone", "country_code", "language", "product_mentioned"]
TERMINAL_FIELDS = ["catalog_match_result", "hcp_system_id", "record_status", "routing_target"]


def _norm(v):
    return v.strip().casefold() if isinstance(v, str) else v


def field_status(field: str, got, expected) -> str:
    """pass | fail | semantic | nogt — casing/whitespace-insensitive for exact fields."""
    if expected is _MISSING:
        return "nogt"
    if field in SEMANTIC_FIELDS:
        return "semantic"
    return "pass" if _norm(got) == _norm(expected) else "fail"


# ============================================================
# The pipeline runner (shared by the app + the harness)
# ============================================================

def run_case(input_text: str, input_metadata: dict, model: str = MODEL,
             ae_extract_version: str = None) -> dict:
    """check_readability -> classify -> extract -> 7-gate unit. Returns JSON-able dicts.

    `ae_extract_version` selects the AE-extract prompt variant for Compare/Sweep; None = current.
    """
    tenant_id = input_metadata["tenant_id"]

    readability = check_readability(input_metadata, tenant_id)
    if not readability["readable"]:
        return {"halted": True, "readability": readability,
                "routing_target": "Unreadable Queue", "record_status": "Pending Review"}

    classification = None
    try:
        classification = classify(input_text, tenant_id=tenant_id, model=model)
        common, record = extract(input_text, classification, input_metadata, model=model,
                                 ae_extract_version=ae_extract_version)
        gates = evaluate_gates(classification, common, record, input_metadata, tenant_id)
    except NotImplementedError as e:
        # The predicted class has no automated extractor yet -> fail SAFE to a human (MIS),
        # never crash. (Classification succeeded; we just don't auto-build a record for this
        # class.) The common metadata extracted before the dispatch is KEPT — the reviewer
        # still gets reporter/product/contact pre-filled; only the record stays manual.
        unbuilt_common = getattr(e, "common", None)
        return {
            "halted": False,
            "unbuilt_class": True,
            "readability": readability,
            "classification": classification.model_dump(mode="json") if classification is not None else None,
            "common": unbuilt_common.model_dump(mode="json") if unbuilt_common is not None else None,
            "record": None,
            "record_type": None,
            "gates": {
                "routing_target": "MIS Queue",
                "record_status": "Pending Review",
                "catalog_match_result": None,
                "hcp_system_id": None,
                "failed_gates": [f"Extraction: class '{classification.predicted_class.value}' has no automated extractor yet — routed to human review."],
                "gates": [],
                "day_zero": None,
                "regulatory_due_date": None,
                "reporting_regime": None,
            },
        }
    except ValidationError:
        # Gate 7 (schema validation), WIRED: the model returned a record that breaks the
        # Pydantic contract. Extraction is deterministic (temp 0), so a retry would reproduce
        # the same failure — we don't gamble, we fail SAFE to a human, never crash.
        # SAFETY PRECEDENCE: if the AE flag was already set, this is a safety case — route to
        # PV (with an incomplete draft for the human to finish), NOT to the general MIS queue.
        # A detected adverse event must never be downgraded to MIS just because its draft broke.
        is_ae = classification is not None and classification.adverse_event_flag
        return {
            "halted": False,
            "schema_failed": True,
            "readability": readability,
            "classification": classification.model_dump(mode="json") if classification is not None else None,
            "common": None,
            "record": None,
            "record_type": None,
            "gates": {
                "routing_target": "PV Queue" if is_ae else "MIS Queue",
                "record_status": "Pending Review",
                "catalog_match_result": None,
                "hcp_system_id": None,
                "failed_gates": [
                    ("Gate 7: schema validation failed — the AI draft was incomplete; "
                     + ("adverse event was detected, so routed to PV for a human to complete the ICSR."
                        if is_ae else "routed to human review."))
                ],
                "gates": [],
                "day_zero": None,
                "regulatory_due_date": None,
                "reporting_regime": None,
            },
        }
    return {
        "halted": False,
        "readability": readability,
        "classification": classification.model_dump(mode="json"),
        "common": common.model_dump(mode="json"),
        "record": record.model_dump(mode="json") if record is not None else None,
        "record_type": record.record_type if record is not None else None,
        "gates": gates,
    }


# ============================================================
# Per-case evaluation (compare a run against ground truth)
# ============================================================

def evaluate_case(case: dict, result: dict, model: str = MODEL, grade: bool = True) -> dict:
    gt = case["ground_truth"]
    proc = gt["expected_processing"]
    e = {"case_id": case.get("case_id", case["input_metadata"]["document_id"]),
         "halted": result.get("halted", False)}

    if e["halted"]:
        e.update({
            "terminal_pred": result["routing_target"],
            "terminal_gt": proc.get("routing_target"),
            "terminal_correct": result["routing_target"] == proc.get("routing_target"),
            "auto_approved": False, "confidence": None, "class_correct": None,
            "ae_gt": None, "ae_pred": None, "record_fully_correct": None,
            "metadata_results": [], "all_results": [], "graders": [],
        })
        return e

    if result.get("schema_failed"):
        g = result["gates"]
        e.update({
            "terminal_pred": g["routing_target"],
            "terminal_gt": proc.get("routing_target"),
            "terminal_correct": g["routing_target"] == proc.get("routing_target"),
            "auto_approved": False, "confidence": None, "class_correct": None,
            "ae_gt": gt.get("adverse_event_flag"), "ae_pred": None, "record_fully_correct": None,
            "metadata_results": [], "all_results": [], "graders": [],
        })
        return e

    if result.get("unbuilt_class"):
        g = result["gates"]
        c = result.get("classification") or {}
        e.update({
            "terminal_pred": g["routing_target"],
            "terminal_gt": proc.get("routing_target"),
            "terminal_correct": g["routing_target"] == proc.get("routing_target"),
            "predicted_class": c.get("predicted_class"), "gt_class": gt.get("predicted_class"),
            "class_correct": (c.get("predicted_class") == gt.get("predicted_class")) if c else None,
            "auto_approved": False, "confidence": c.get("class_confidence"),
            "ae_gt": gt.get("adverse_event_flag"), "ae_pred": c.get("adverse_event_flag"),
            "record_fully_correct": None, "metadata_results": [], "all_results": [], "graders": [],
        })
        return e

    cls = result["classification"]
    g = result["gates"]
    e["predicted_class"] = cls["predicted_class"]
    e["gt_class"] = gt["predicted_class"]
    # classification_exempt: genuinely out-of-taxonomy cases (e.g. case_013 storage Q) have no
    # clean class. They're scored on routing (→ human), NOT on the class label — so class_correct
    # is None here and is excluded from classification-accuracy.
    e["class_correct"] = None if gt.get("classification_exempt") else (cls["predicted_class"] == gt["predicted_class"])
    e["ae_pred"] = cls["adverse_event_flag"]
    e["ae_gt"] = gt["adverse_event_flag"]
    e["confidence"] = cls["class_confidence"]
    e["terminal_pred"] = g["routing_target"]
    e["terminal_gt"] = proc.get("routing_target")
    e["terminal_correct"] = g["routing_target"] == proc.get("routing_target")
    e["auto_approved"] = g["routing_target"] == "Auto-Approve"

    class_results = [
        ("predicted_class", field_status("predicted_class", cls["predicted_class"], gt.get("predicted_class", _MISSING))),
        ("adverse_event_flag", field_status("adverse_event_flag", cls["adverse_event_flag"], gt.get("adverse_event_flag", _MISSING))),
        ("out_of_scope", field_status("out_of_scope", cls["out_of_scope"], gt.get("out_of_scope", _MISSING))),
    ]
    # gt may lack common_metadata when a case is an "unbuilt-class" scenario (no metadata GT) but
    # the model still classified it into a built class — fall back to {} so it scores as no-GT, not a crash.
    gt_meta = gt.get("common_metadata") or {}
    meta_results = [(f, field_status(f, (result["common"] or {}).get(f), gt_meta.get(f, _MISSING)))
                    for f in META_FIELDS]
    record_results = []
    if result["record"]:
        gt_rec = gt.get("on_label_record") or gt.get("off_label_record") or gt.get("ae_record") or {}
        for f, v in result["record"].items():
            if f == "record_type":
                continue
            record_results.append((f, field_status(f, v, gt_rec.get(f, _MISSING))))
    terminal_results = [(f, field_status(f, g.get(f), proc.get(f, _MISSING))) for f in TERMINAL_FIELDS]

    all_results = class_results + meta_results + record_results + terminal_results
    e["metadata_results"] = meta_results
    e["all_results"] = all_results
    # "record fully correct" = no exact-field mismatch (semantic/nogt don't count against).
    e["record_fully_correct"] = (all(s != "fail" for _, s in all_results)
                                 and any(s == "pass" for _, s in all_results))

    # LLM-as-judge graders for the free-text fields exact-match can't score.
    e["graders"] = grade_case(case["input_text"], result["common"], result["record"], model) if grade else []
    return e


def evaluate_all(model: str = MODEL, grade: bool = True, ae_extract_version: str = None) -> list:
    """Run + score every golden case. Auto-discovers case_*.json."""
    evals = []
    for case_file in sorted(GOLDEN.glob("case_*.json")):
        case = json.loads(case_file.read_text())
        result = run_case(case["input_text"], case["input_metadata"], model, ae_extract_version)
        evals.append(evaluate_case(case, result, model, grade))
    return evals


# ============================================================
# Aggregate metrics — the PROVEN-IF scorecard
# ============================================================

def _frac(bools):
    bools = [b for b in bools if b is not None]
    return (sum(1 for b in bools if b) / len(bools)) if bools else None


def _calibration(nonhalted):
    edges = [(0.90, 1.01), (0.80, 0.90), (0.0, 0.80)]
    out = []
    for lo, hi in edges:
        b = [e for e in nonhalted if e["confidence"] is not None and lo <= e["confidence"] < hi]
        out.append({"range": f"{lo:.2f}–{min(hi, 1.0):.2f}", "n": len(b),
                    "accuracy": _frac([e["class_correct"] for e in b])})
    return out


def aggregate_metrics(evals: list) -> dict:
    n = len(evals)
    nonhalted = [e for e in evals if not e["halted"]]
    ae_gt = [e for e in nonhalted if e["ae_gt"]]
    ae_pred = [e for e in nonhalted if e["ae_pred"]]
    auto = [e for e in evals if e.get("auto_approved")]
    human = [e for e in nonhalted if not e.get("auto_approved")]
    meta_flags = [s == "pass" for e in human for _, s in e["metadata_results"] if s in ("pass", "fail")]

    by_terminal = {}
    for e in evals:
        by_terminal[e["terminal_pred"]] = by_terminal.get(e["terminal_pred"], 0) + 1

    return {
        "n": n, "n_nonhalted": len(nonhalted),
        "classification_accuracy": _frac([e["class_correct"] for e in nonhalted]),
        "terminal_accuracy": _frac([e["terminal_correct"] for e in evals]),
        "ae_recall": _frac([e["ae_pred"] for e in ae_gt]), "ae_recall_n": len(ae_gt),
        "ae_precision": _frac([e["ae_gt"] for e in ae_pred]), "ae_precision_n": len(ae_pred),
        "auto_approve_precision": _frac([e["record_fully_correct"] for e in auto]), "auto_approve_n": len(auto),
        "coverage": (len(auto) / n) if n else None, "coverage_n": len(auto),
        "metadata_accuracy": _frac(meta_flags), "metadata_n": len(meta_flags),
        "calibration": _calibration(nonhalted),
        "by_terminal": by_terminal,
        "graders": _grader_summary(evals),
        "evals": evals,
    }


def _grader_summary(evals: list) -> dict:
    """Roll the per-case LLM-judge verdicts into per-grader pass-rate / avg score."""
    by_grader = {}
    for e in evals:
        for g in e.get("graders", []):
            by_grader.setdefault(g["grader"], []).append(g)
    out = {}
    for name, rs in by_grader.items():
        if rs[0]["type"] == "passfail":
            out[name] = {"type": "passfail", "n": len(rs),
                         "pass_rate": _frac([r["passed"] for r in rs])}
        else:
            out[name] = {"type": "score", "n": len(rs),
                         "avg": (sum(r["score"] for r in rs) / len(rs)),
                         "pass_rate": _frac([r["passed"] for r in rs]),
                         "threshold": rs[0].get("threshold")}
    return out


# ============================================================
# Compare / Sweep — cost, success criteria, and per-config disk cache
# ============================================================
import time
from prompts import EXTRACT_PROMPT_VERSION

EVAL_CACHE = Path(__file__).parent / "eval_cache"

# Approx public list price, USD per 1M tokens (input, output). Used for the cost axis only.
PRICE = {
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-haiku-4-5-20251001": (1.0, 5.0),
    "gpt-4o": (2.5, 10.0),
    "gpt-4o-mini": (0.15, 0.60),
}


def success_criteria(m: dict) -> list:
    """The GO / NO-GO bar — what every config is judged against (not raw %)."""
    g = m.get("graders", {})
    pr = lambda name: g[name]["pass_rate"] if name in g else None
    rows = [
        ("AE recall (hard floor)", m["ae_recall"], 1.00),
        ("Classification accuracy", m["classification_accuracy"], 0.90),
        ("Auto-approve precision", m["auto_approve_precision"], 0.99),
        ("Faithfulness (no hallucination)", pr("faithfulness"), 1.00),
        ("Symptom coverage (AE)", pr("symptom_coverage"), 1.00),
        ("Metadata draft accuracy", m["metadata_accuracy"], 0.95),
    ]
    return [{"name": n, "value": v, "target": t,
             "pass": (v is not None and v >= t - 1e-9)} for n, v, t in rows]


def _cost_block(usage: dict, model: str, latency_s: float, n: int) -> dict:
    intok = sum(u.get("input_tokens", 0) for u in usage.values())
    outtok = sum(u.get("output_tokens", 0) for u in usage.values())
    pin, pout = PRICE.get(model, (None, None))
    usd = (intok / 1e6 * pin + outtok / 1e6 * pout) if pin is not None else None
    return {"latency_s": round(latency_s, 1), "latency_per_case": round(latency_s / n, 1) if n else None,
            "input_tokens": intok, "output_tokens": outtok,
            "est_usd": round(usd, 4) if usd is not None else None,
            "est_usd_per_case": round(usd / n, 4) if (usd is not None and n) else None}


def evaluate(model: str = MODEL, ae_extract_version: str = None, grade: bool = True) -> dict:
    """One-shot: run + score + aggregate, with cost/latency + success criteria attached."""
    t0 = time.perf_counter()
    try:  # capture token usage across all LLM calls in the run (langchain-core >= 0.3.x)
        from langchain_core.callbacks import get_usage_metadata_callback
        with get_usage_metadata_callback() as cb:
            evals = evaluate_all(model, grade, ae_extract_version)
        usage = dict(cb.usage_metadata)
    except ImportError:
        evals = evaluate_all(model, grade, ae_extract_version)
        usage = {}
    latency = time.perf_counter() - t0

    m = aggregate_metrics(evals)
    m["model"] = model
    m["ae_extract_version"] = ae_extract_version or EXTRACT_PROMPT_VERSION
    m["cost"] = _cost_block(usage, model, latency, m["n"])
    m["criteria"] = success_criteria(m)
    return m


def config_key(model: str, ae_extract_version: str = None) -> str:
    return f"{model}__ae-{ae_extract_version or EXTRACT_PROMPT_VERSION}"


def evaluate_cached(model: str = MODEL, ae_extract_version: str = None, force: bool = False) -> dict:
    """Disk-cached evaluate() — view a full report without re-running. force=True re-runs live."""
    EVAL_CACHE.mkdir(exist_ok=True)
    f = EVAL_CACHE / f"{config_key(model, ae_extract_version)}.json"
    if f.exists() and not force:
        return json.loads(f.read_text())
    m = evaluate(model, ae_extract_version)
    f.write_text(json.dumps(m, indent=2, default=str))
    return m


def list_cached_configs() -> list:
    """[(model, ae_extract_version)] for every single-run eval report on disk (skips __var)."""
    if not EVAL_CACHE.exists():
        return []
    out = []
    for f in sorted(EVAL_CACHE.glob("*.json")):
        if "__var" in f.name:
            continue
        try:
            m = json.loads(f.read_text())
            out.append((m.get("model"), m.get("ae_extract_version")))
        except Exception:
            pass
    return out


# ============================================================
# Variance — run each config K times, report median / mean ± range
# ============================================================
import statistics

# The scalar metrics we track across runs (key -> getter from a full metrics dict).
VAR_METRICS = {
    "classification_accuracy": lambda m: m["classification_accuracy"],
    "terminal_accuracy": lambda m: m["terminal_accuracy"],
    "ae_recall": lambda m: m["ae_recall"],
    "auto_approve_precision": lambda m: m["auto_approve_precision"],
    "metadata_accuracy": lambda m: m["metadata_accuracy"],
    "faithfulness": lambda m: (m["graders"].get("faithfulness") or {}).get("pass_rate"),
    "symptom_coverage": lambda m: (m["graders"].get("symptom_coverage") or {}).get("pass_rate"),
    "est_usd": lambda m: m["cost"]["est_usd"],
    "latency_s": lambda m: m["cost"]["latency_s"],
}


def _run_scalars(m: dict) -> dict:
    return {k: get(m) for k, get in VAR_METRICS.items()}


def evaluate_variance(model: str = MODEL, ae_extract_version: str = None, k: int = 3,
                      force: bool = False) -> dict:
    """Run the config K times -> {runs: [scalar dicts]}. Seeds run 1 from the existing
    single-run cache (current prompts) to save one full eval, then runs the rest live."""
    EVAL_CACHE.mkdir(exist_ok=True)
    f = EVAL_CACHE / f"{config_key(model, ae_extract_version)}__var{k}.json"
    if f.exists() and not force:
        return json.loads(f.read_text())
    runs = []
    single = EVAL_CACHE / f"{config_key(model, ae_extract_version)}.json"
    if single.exists() and not force:
        runs.append(_run_scalars(json.loads(single.read_text())))   # reuse the committed single run
    while len(runs) < k:
        runs.append(_run_scalars(evaluate(model, ae_extract_version)))
    payload = {"model": model, "ae_extract_version": ae_extract_version or EXTRACT_PROMPT_VERSION,
               "k": k, "runs": runs}
    f.write_text(json.dumps(payload, indent=2, default=str))
    return payload


def variance_stats(payload: dict) -> dict:
    """Per-metric {median, mean, min, max, n} across the runs."""
    out = {}
    for key in VAR_METRICS:
        vals = [r[key] for r in payload["runs"] if r.get(key) is not None]
        out[key] = ({"median": statistics.median(vals), "mean": sum(vals) / len(vals),
                     "min": min(vals), "max": max(vals), "n": len(vals)} if vals else None)
    return out


def has_variance(model: str = MODEL, ae_extract_version: str = None, k: int = 3) -> bool:
    return (EVAL_CACHE / f"{config_key(model, ae_extract_version)}__var{k}.json").exists()


# ============================================================
# CLI scorecard
# ============================================================

def _pct(x):
    return "—" if x is None else f"{x * 100:.0f}%"


def _main():
    m = evaluate()
    print(f"\nEval over {m['n']} golden cases (model={MODEL})")
    print("=" * 56)
    print(f"  Classification accuracy : {_pct(m['classification_accuracy'])}   (n={m['n_nonhalted']})")
    print(f"  Terminal routing acc    : {_pct(m['terminal_accuracy'])}   (n={m['n']})")
    print(f"  Auto-approve precision  : {_pct(m['auto_approve_precision'])}   target ≥99%  (n={m['auto_approve_n']})")
    print(f"  Coverage (auto-approved): {_pct(m['coverage'])}   ({m['coverage_n']}/{m['n']})")
    print(f"  AE recall (HARD FLOOR)  : {_pct(m['ae_recall'])}   target 100%  (n={m['ae_recall_n']})")
    print(f"  AE precision            : {_pct(m['ae_precision'])}   (n={m['ae_precision_n']})")
    print(f"  Metadata draft accuracy : {_pct(m['metadata_accuracy'])}   target ≥95%  (n={m['metadata_n']} fields)")
    print("  Calibration:")
    for b in m["calibration"]:
        print(f"     conf {b['range']}:  n={b['n']}  acc={_pct(b['accuracy'])}")
    print(f"  By terminal route       : {m['by_terminal']}")
    print("  LLM-judge graders:")
    for name, s in m["graders"].items():
        if s["type"] == "passfail":
            print(f"     {name:<18}: pass {_pct(s['pass_rate'])}   (n={s['n']})")
        else:
            print(f"     {name:<18}: avg {s['avg']:.1f}/5  pass {_pct(s['pass_rate'])} (≥{s['threshold']})  (n={s['n']})")
    print("=" * 56)


if __name__ == "__main__":
    _main()
