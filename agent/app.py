"""
app.py — Intake Agent Evaluation Console.

Two tabs:
  📊 Eval Dashboard  — the PROVEN-IF metrics across the WHOLE golden set (the bet, measured).
  🔬 Case Inspector  — trace ONE document through every pipeline stage vs ground truth.

The pipeline + metric logic live in evaluate.py (shared with the CLI). This file is UI only.

HOW TO RUN (from the agent/ folder):
    .venv/bin/streamlit run app.py
"""

import html
import json
from pathlib import Path

import pandas as pd
import streamlit as st

import evaluate as ev
from config import get_thresholds
from prompts import PROMPT_VERSION, EXTRACT_PROMPT_VERSION, AE_EXTRACT_VERSION_INFO, AE_EXTRACT_PROMPTS
import prompt_store
from prompt_registry import PROMPT_REGISTRY
import difflib

GOLDEN = Path(__file__).parent / "golden_dataset"
MODELS = ["claude-sonnet-4-6", "claude-haiku-4-5-20251001", "gpt-4o", "gpt-4o-mini"]

# Agent notes — known / expected divergences worth keeping in front of you.
AGENT_NOTES = [
    "**Free-text fields are semantic, not exact.** `inquiry_summary_text` and "
    "`adverse_symptoms_list` are judged by meaning — a wording/casing difference is NOT a failure. "
    "They're marked 🟦 and reviewed by hand (or by the LLM-judge eval).",
    "**`ae_seriousness` is a borderline judgment.** Judged vs the 5 ICH serious criteria + the "
    "Important Medical Event exception. For case_021 the cardiac symptoms make it a defensible "
    "*Serious* (the golden label was updated from a coin-flip Non-Serious). Routing is unchanged "
    "either way — Gate 4 sends every AE to PV.",
    "**All 7 gates run as one unit AFTER extraction.** No separate early router — extraction always "
    "runs first, so every path (Auto-Approve / MIS / PV / Unreadable) carries the AI's full draft "
    "into its queue. Gate 4 (AE) is a policy override → PV; any other gate fails → MIS; all green → Auto-Approve.",
    "**`is_valid_icsr` is computed in code**, not trusted from the LLM — derived from the four legal "
    "pillars in the classifier output.",
]


# ============================================================
# Cached compute (LLM calls cached so widget clicks don't re-run them)
# ============================================================

@st.cache_data(show_spinner=False)
def run_pipeline(input_text: str, input_metadata: dict, model: str) -> dict:
    return ev.run_case(input_text, input_metadata, model)


@st.cache_data(show_spinner=False)
def load_eval(model: str) -> dict:
    return ev.evaluate(model)



# ============================================================
# Display helpers
# ============================================================

def _fmt(v) -> str:
    if v is None:
        return "None"
    if isinstance(v, list):
        return ", ".join(str(x) for x in v) if v else "[]"
    return str(v)


def _pct(x) -> str:
    return "—" if x is None else f"{x * 100:.0f}%"


STATUS_BADGE = {
    "pass": "✅ Match",
    "fail": "❌ Mismatch",
    "semantic": "🟦 Semantic · text, judged by meaning",
    "pending": "🟦 On track · pending gates",
    "nogt": "⚪ No ground truth",
}


def comparison_rows(fields, got_dict, gt_dict):
    rows = []
    for f in fields:
        got = got_dict.get(f)
        expected = gt_dict.get(f, ev._MISSING) if gt_dict is not None else ev._MISSING
        rows.append({
            "Field": f,
            "Extracted": _fmt(got),
            "Ground truth": "—" if expected is ev._MISSING else _fmt(expected),
            "code": ev.field_status(f, got, expected),
        })
    return rows


def tally(rows) -> tuple:
    passes = sum(1 for r in rows if r["code"] == "pass")
    exact = sum(1 for r in rows if r["code"] in ("pass", "fail"))
    return passes, exact


TABLE_CSS = """
<style>
table.cmp { width:100%; border-collapse:collapse; table-layout:fixed; margin:0.25rem 0 0.5rem; }
table.cmp th, table.cmp td { border:1px solid #e8e8e8; padding:8px 11px; text-align:left;
    vertical-align:top; white-space:normal; word-break:break-word; overflow-wrap:anywhere; font-size:0.9rem; }
table.cmp th { background:#f6f8fa; font-weight:600; }
table.cmp col.c-field { width:22%; }
table.cmp col.c-val { width:30%; }
table.cmp col.c-status { width:18%; }
</style>
"""


def render_table(rows) -> None:
    body = ""
    for r in rows:
        badge = STATUS_BADGE.get(r["code"], r["code"])
        body += ("<tr>"
                 f"<td><b>{html.escape(str(r['Field']))}</b></td>"
                 f"<td>{html.escape(str(r['Extracted']))}</td>"
                 f"<td>{html.escape(str(r['Ground truth']))}</td>"
                 f"<td>{badge}</td></tr>")
    st.markdown(
        "<table class='cmp'>"
        "<colgroup><col class='c-field'><col class='c-val'><col class='c-val'><col class='c-status'></colgroup>"
        "<tr><th>Field</th><th>Extracted</th><th>Ground truth</th><th>Status</th></tr>"
        f"{body}</table>",
        unsafe_allow_html=True,
    )


def render_gates(gates) -> None:
    body = ""
    for gg in gates:
        badge = "✅ Pass" if gg["passed"] else "❌ Fail"
        check = html.escape(str(gg["detail"]))
        if not gg["passed"] and gg.get("hint"):
            check += (f"<br><span style='color:#b00020;font-size:0.85em'>↳ why: "
                      f"{html.escape(gg['hint'])}</span>")
        body += ("<tr>"
                 f"<td><b>{html.escape(gg['id'])}</b> · {html.escape(gg['name'])}</td>"
                 f"<td>{check}</td>"
                 f"<td>{badge}</td></tr>")
    st.markdown(
        "<table class='cmp'>"
        "<colgroup><col class='c-field'><col class='c-val'><col class='c-status'></colgroup>"
        "<tr><th>Gate</th><th>Check</th><th>Status</th></tr>"
        f"{body}</table>",
        unsafe_allow_html=True,
    )


# ============================================================
# Pipeline flow visual (Epic C) — lights up the path THIS case took
# ============================================================

_PIPE_FLOW = {
    "intake": "Intake",
    "read": "Readability (OCR)",
    "classify": "Classify",
    "extract": "Extract (full draft)",
    "gates": "7-Gate Unit",
}
_PIPE_TERM = {
    "unread": ("Unreadable Queue", "#555555"),
    "pv": ("PV Queue", "#c0392b"),
    "mis": ("MIS Queue", "#b8860b"),
    "auto": ("Auto-Approve", "#1e7e34"),
}
_PIPE_EDGES = [
    ("intake", "read", ""),
    ("read", "unread", "OCR < 0.75"),
    ("read", "classify", "readable"),
    ("classify", "extract", ""),
    ("extract", "gates", ""),
    ("gates", "pv", "AE · G4"),
    ("gates", "mis", "gate fail"),
    ("gates", "auto", "all pass"),
]


def pipeline_dot(res: dict) -> str:
    """Build a Graphviz DOT string for the pipeline, highlighting the path this case took."""
    if res.get("halted"):
        active_nodes = {"intake", "read", "unread"}
        active_edges = {("intake", "read"), ("read", "unread")}
        gate_label = _PIPE_FLOW["gates"]
    else:
        target = res["gates"]["routing_target"]
        term = {"PV Queue": "pv", "MIS Queue": "mis", "Auto-Approve": "auto"}.get(target, "mis")
        active_nodes = {"intake", "read", "classify", "extract", "gates", term}
        active_edges = {("intake", "read"), ("read", "classify"), ("classify", "extract"),
                        ("extract", "gates"), ("gates", term)}
        nfail = sum(1 for gg in res["gates"]["gates"] if not gg["passed"])
        gate_label = f"7-Gate Unit ({nfail} failed)" if nfail else "7-Gate Unit (all pass)"

    out = ['digraph G {', 'rankdir=LR; bgcolor="transparent";',
           'node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=11, margin="0.14,0.08"];',
           'edge [fontname="Helvetica", fontsize=9];']

    for nid, lbl in _PIPE_FLOW.items():
        label = gate_label if nid == "gates" else lbl
        if nid in active_nodes:
            out.append(f'{nid} [label="{label}", fillcolor="#d6e4ff", color="#1f4e8c", penwidth=2];')
        else:
            out.append(f'{nid} [label="{label}", fillcolor="#f4f4f4", color="#dcdcdc", fontcolor="#a8a8a8"];')

    for nid, (lbl, col) in _PIPE_TERM.items():
        if nid in active_nodes:
            out.append(f'{nid} [label="{lbl}", fillcolor="{col}", color="{col}", fontcolor="white", penwidth=2];')
        else:
            out.append(f'{nid} [label="{lbl}", fillcolor="#f4f4f4", color="#dcdcdc", fontcolor="#a8a8a8"];')

    for a, b, lbl in _PIPE_EDGES:
        if (a, b) in active_edges:
            out.append(f'{a} -> {b} [label="{lbl}", color="#1f4e8c", penwidth=2.4, fontcolor="#1f4e8c"];')
        else:
            out.append(f'{a} -> {b} [label="{lbl}", color="#dcdcdc", style=dashed, fontcolor="#c4c4c4"];')

    out.append("}")
    return "\n".join(out)


def render_pipeline(res: dict) -> None:
    st.markdown("#### 🔀 Pipeline path  ·  *the route this document took*")
    st.graphviz_chart(pipeline_dot(res), use_container_width=True)
    st.caption("**OCR tiers (Gate 1):**  < 0.75 → Unreadable Queue (can't process)  ·  "
               "0.75–0.90 → process (classify + extract) but **force human review**  ·  "
               "≥ 0.90 → eligible for auto-approve.")
    if not res.get("halted"):
        failed = [g for g in res["gates"]["gates"] if not g["passed"]]
        if failed:
            st.markdown("**Why it didn't auto-approve:**")
            for g in failed:
                st.markdown(f"- **{g['id']} · {g['name']}** — {g.get('hint', '')}")


# ============================================================
# TAB 1 — Eval Dashboard (collective)
# ============================================================

def _target_mark(value, target):
    if value is None:
        return "—"
    return f"{'✅' if value >= target else '⚠️'} target {target * 100:.0f}%"


def render_dashboard(model: str) -> None:
    st.subheader("📊 Eval Dashboard — the bet, measured across the golden set")
    st.caption("**Riskiest assumption:** the model is *accurate* AND *calibrated* enough to gate its own "
               "work (auto-commit vs human). The infra is known engineering and the model is "
               "parameterized — so this is the only thing that must be *proven*, not assumed.")

    if st.button("▶ Run / re-run full eval across the golden set", type="primary", key="run_eval"):
        load_eval.clear()                      # clicking RE-RUNS (clear the cached result)
        st.session_state["eval_ran"] = True
    if not st.session_state.get("eval_ran"):
        st.info("Hit **Run full eval** to score every golden case and compute the PROVEN-IF metrics.")
        return

    with st.spinner(f"Running {model} across the golden set…"):
        m = load_eval(model)

    st.warning(f"⚖️ **n = {m['n']} cases.** Read this as the evaluation **framework + method**, not a "
               "statistically settled number — it **auto-scales** as golden cases are added (drop a "
               "`case_*.json` in and it's included). Method over decimals.")
    st.caption(f"🔖 **Audit stamp** (on every record, per 21 CFR Part 11): model `{model}` · "
               f"prompts `{PROMPT_VERSION}` / `{EXTRACT_PROMPT_VERSION}`. See the prompt changelog for version history.")

    # --- PROVEN-IF scorecard ---
    st.markdown("#### PROVEN-IF scorecard")
    a = st.columns(3)
    a[0].metric("Auto-approve precision", _pct(m["auto_approve_precision"]),
                _target_mark(m["auto_approve_precision"], 0.99), delta_color="off",
                help=f"Record fully correct among auto-approved. Target ≥99%. n={m['auto_approve_n']}")
    a[1].metric("Coverage (auto-approved)", _pct(m["coverage"]),
                f"{m['coverage_n']}/{m['n']} cases", delta_color="off",
                help="Efficiency — how many qualify for no-human auto-commit. Report actual.")
    a[2].metric("AE recall — HARD FLOOR", _pct(m["ae_recall"]),
                _target_mark(m["ae_recall"], 1.0), delta_color="off",
                help=f"Of true adverse events, fraction flagged. Target 100%, never compromise. n={m['ae_recall_n']}")
    b = st.columns(3)
    b[0].metric("AE precision", _pct(m["ae_precision"]), "guardrail vs PV overload", delta_color="off",
                help=f"Of flagged AEs, fraction real. Report actual; sharpen prompt if PV floods. n={m['ae_precision_n']}")
    b[1].metric("Metadata draft accuracy", _pct(m["metadata_accuracy"]),
                _target_mark(m["metadata_accuracy"], 0.95), delta_color="off",
                help=f"Human-review path — quality of the reviewer's starting draft. Target ≥95%. n={m['metadata_n']} fields")
    b[2].metric("Classification accuracy", _pct(m["classification_accuracy"]),
                f"{m['n_nonhalted']} classified", delta_color="off",
                help="Predicted class vs ground truth.")

    st.divider()

    # --- precision × coverage quadrant ---
    st.markdown("#### Precision × Coverage")
    prec, cov = m["auto_approve_precision"], m["coverage"]
    if prec is None or cov is None:
        st.caption("No auto-approved cases yet to place on the quadrant.")
    else:
        ph, ch = prec >= 0.99, cov >= 0.5
        quad = {(True, True): "✅ **the win** (top-left) — auto-approves a lot, and they're right",
                (True, False): "⚠️ safe but low automation — escalates most",
                (False, True): "❌ **dangerous** — auto-approving wrong records",
                (False, False): "❌ worst case"}[(ph, ch)]
        st.markdown(f"Current position → **precision {_pct(prec)} · coverage {_pct(cov)}** → {quad}")
        st.caption("Coverage target is 'report actual'; the 0.5 split here is illustrative for the quadrant.")

    st.divider()

    # --- confidence calibration ---
    st.markdown("#### Confidence calibration")
    st.table(pd.DataFrame([{"Confidence range": c["range"], "n": c["n"], "Class accuracy": _pct(c["accuracy"])}
                           for c in m["calibration"]]))
    st.caption("Calibrated = accuracy tracks confidence (e.g. 0.85 ≈ 85% correct). The gate threshold "
               "(0.90) relies on this — if a 0.93 bucket scores 70%, the gate is mis-set.")

    # --- breakdown by terminal route ---
    st.markdown("#### By terminal route")
    st.table(pd.DataFrame([{"Terminal route": k, "Cases": v} for k, v in sorted(m["by_terminal"].items())]))

    # --- LLM-as-judge graders (the free-text fields exact-match can't score) ---
    graders = m.get("graders", {})
    if graders:
        st.divider()
        st.markdown("#### 🧑‍⚖️ LLM-as-judge graders")
        st.caption("The hybrid eval's second half: free-text fields (`inquiry_summary_text`, "
                   "`adverse_symptoms_list`) and grounding can't be scored by exact match — so a "
                   "single-responsibility **LLM-judge** reads the source document + the extracted value "
                   "and returns a verdict **with reasoning** (auditable). Pass/fail for safety-critical, 1–5 for quality.")
        labels = {"faithfulness": "Faithfulness · no hallucination",
                  "symptom_coverage": "Symptom coverage · AE",
                  "summary_accuracy": "Summary accuracy"}
        gcols = st.columns(len(graders))
        for i, (name, s) in enumerate(graders.items()):
            if s["type"] == "passfail":
                gcols[i].metric(labels.get(name, name), _pct(s["pass_rate"]),
                                f"pass rate · n={s['n']}", delta_color="off")
            else:
                gcols[i].metric(labels.get(name, name), f"{s['avg']:.1f}/5",
                                f"{_pct(s['pass_rate'])} pass (≥{s['threshold']}) · n={s['n']}", delta_color="off")
        st.caption("⚠️ The judge LLM has its own biases — in production, spot-check graders against human review.")


# ============================================================
# TAB 2 — Case Inspector (per-case pipeline trace)
# ============================================================

def render_case_inspector(model: str) -> None:
    # --- inputs live in THIS tab (one Run button per context, no sidebar overlap) ---
    cases = {p.name: json.loads(p.read_text()) for p in sorted(GOLDEN.glob("case_*.json"))}
    mode = st.radio("Document source", ["Golden case", "Custom paste"], horizontal=True, key="ci_mode")
    if mode == "Golden case":
        case_name = st.selectbox("Case", list(cases.keys()), key="ci_case")
        case = cases[case_name]
        input_text = case["input_text"]
        input_metadata = case["input_metadata"]
        gt = case["ground_truth"]
        scenario = case.get("scenario", "")
        labeling_notes = case.get("labeling_notes", [])
    else:
        input_text = st.text_area("Paste document text", height=160, key="ci_text",
                                  placeholder="Paste an inbound document here…")
        cc1, cc2, cc3 = st.columns(3)
        ocr_sim = cc1.slider("OCR confidence", 0.0, 1.0, 1.0, 0.01, key="ci_ocr",
                             help="Drag below 0.75 to trip the pre-classify Unreadable halt (no LLM spent).")
        channel = cc2.selectbox("Source channel", ["Web Portal", "Email", "Network API"], key="ci_chan")
        rdate = cc3.date_input("Received date", key="ci_date")
        input_metadata = {"document_id": "CUSTOM_001", "tenant_id": "TENANT_BIO_99",
                          "source_channel": channel, "received_date": str(rdate), "ocr_confidence": ocr_sim}
        gt = None
        scenario = "Custom document (no ground truth)"
        labeling_notes = []

    if st.button("▶ Run pipeline on this document", type="primary", key="ci_run"):
        if not input_text.strip():
            st.warning("No document text to process.")
        else:
            with st.spinner(f"Running {model}…"):
                st.session_state["result"] = run_pipeline(input_text, input_metadata, model)
                st.session_state["ctx"] = {
                    "input_text": input_text, "input_metadata": input_metadata, "gt": gt,
                    "scenario": scenario, "labeling_notes": labeling_notes, "model": model,
                }

    st.divider()

    if "result" not in st.session_state:
        st.info("Pick a case (or paste a document) above and hit **Run pipeline**.")
        return

    res = st.session_state["result"]
    ctx = st.session_state["ctx"]
    gt = ctx["gt"]
    th = get_thresholds(ctx["input_metadata"]["tenant_id"])

    # --- pre-classify halt: unreadable document (no LLM spent) ---
    if res.get("halted"):
        rb = res["readability"]
        st.error("⛔ **HALTED at the pre-classify readability gate.** The document is unreadable, so "
                 "**no LLM call was spent** — routed straight to the Unreadable Queue for rescan / remediation.")
        h1, h2 = st.columns(2)
        h1.metric("Terminal decision", "Unreadable Queue")
        h2.metric("OCR confidence", f"{rb['ocr_confidence']:.2f}",
                  f"{rb['ocr_confidence'] - th['ocr_unreadable_below']:+.2f} vs {th['ocr_unreadable_below']} floor",
                  delta_color="off")
        st.markdown(f"- {rb['reason']}")
        render_pipeline(res)
        st.divider()
        st.subheader("0 · Input")
        if ctx["scenario"]:
            st.caption(ctx["scenario"])
        st.text_area("Document text", ctx["input_text"], height=200, disabled=True)
        st.table(pd.DataFrame(ctx["input_metadata"].items(), columns=["Field", "Value"]))
        st.caption("Classify → Extract → Gates were all skipped — the document never reached the model.")
        return

    cls = res["classification"]
    gt_class_ok = gt is not None and cls["predicted_class"] == gt["predicted_class"]
    gt_ae_ok = gt is not None and cls["adverse_event_flag"] == gt["adverse_event_flag"]

    class_badge = "✅ matches ground truth" if gt_class_ok else ("❌ differs from ground truth" if gt else "")
    if cls.get("out_of_scope"):
        st.markdown(f"**Predicted class:**  🚫 **OUT OF SCOPE** (spam/off-topic; closest weighed: {cls['predicted_class']})  ·  {class_badge}")
    else:
        st.markdown(f"**Predicted class:**  {cls['predicted_class']}  ·  {class_badge}")
    c2, c3, c4 = st.columns(3)
    c2.metric("AE flag", str(cls["adverse_event_flag"]), ("✅" if gt_ae_ok else ("❌" if gt else "—")), delta_color="off")
    c3.metric("Terminal decision", res["gates"]["routing_target"])
    c4.metric("Model", ctx["model"].replace("claude-", ""))

    render_pipeline(res)

    st.divider()

    # 0 — INPUT
    st.subheader("0 · Input")
    if ctx["scenario"]:
        st.caption(ctx["scenario"])
    ic1, ic2 = st.columns([3, 2])
    with ic1:
        st.text_area("Document text", ctx["input_text"], height=260, disabled=True)
    with ic2:
        st.markdown("**Ingestion envelope** (system-known, never guessed)")
        st.table(pd.DataFrame(ctx["input_metadata"].items(), columns=["Field", "Value"]))

    st.divider()

    # 1 — CLASSIFY
    st.subheader("1 · Classifier output")
    conf = cls["class_confidence"]
    margin = cls["classification_conflict_margin"]
    conf_gate = th["classification_confidence_min"]
    margin_gate = th["conflict_margin_min"]
    m1, m2, m3 = st.columns(3)
    m1.metric("Class confidence", f"{conf:.2f}", f"{conf - conf_gate:+.2f} vs {conf_gate} gate",
              help="How sure the model is of the chosen class (0–1).")
    m2.metric("Conflict margin (top1−top2)", f"{margin:.2f}", "informational — not a gate", delta_color="off",
              help="Gap between the top class score and the runner-up. Computed from the per-class scores. "
                   "Shown for audit; the old conflict gate (Gate 3) was retired as redundant with confidence (Gate 2).")
    m3.metric("Scope", "Out of Scope" if cls["out_of_scope"] else "In Scope")
    st.caption(f"Confidence & margin are **computed in code from the model's per-class scores** (not self-reported): "
               f"confidence = top class score, margin = top1−top2. A confidence below the gate "
               f"(**{conf_gate}**, Gate 2) sends the case to human review (MIS Queue). Margin is informational — "
               f"Gate 3 was retired (it never fired independently of Gate 2; the two signals are coupled).")
    rationale = cls.get("classification_rationale")
    if rationale:
        st.info(f"🧠 **Why this class & confidence:** {rationale}")
    class_fields = ["predicted_class", "adverse_event_flag", "out_of_scope",
                    "has_patient_pillar", "has_reporter_pillar", "has_product_pillar", "has_event_pillar"]
    class_rows = comparison_rows(class_fields, cls, gt)
    render_table(class_rows)

    st.divider()

    # 2 — EXTRACT (always runs — full draft for every path)
    st.subheader("2 · Extracted common metadata")
    meta_fields = ["customer_org", "reporter_name", "reporter_type", "contact_email",
                   "contact_phone", "country_code", "language", "product_mentioned"]
    gt_common = gt["common_metadata"] if gt else None
    meta_rows = comparison_rows(meta_fields, res["common"], gt_common)
    render_table(meta_rows)

    st.subheader("2b · Extracted transactional record")
    if res["record"] is None:
        st.info("This class produces no transactional record (e.g. Legal/Reg, Public FAQ).")
        rec_rows = []
    else:
        st.caption(f"Record type: **{res['record_type']}**")
        gt_record = (gt.get("on_label_record") or gt.get("off_label_record") or gt.get("ae_record")) if gt else None
        rec_fields = [f for f in res["record"] if f != "record_type"]
        rec_rows = comparison_rows(rec_fields, res["record"], gt_record)
        render_table(rec_rows)

    st.divider()

    # 3 — 8-GATE COMPLIANCE UNIT -> TERMINAL DECISION
    st.subheader("3 · 7-gate compliance unit → terminal decision  ·  *deterministic, no LLM*")
    st.caption("After extraction, all 7 gates run as ONE unit. **Gate 4 (AE)** is a policy override → PV; "
               "any other gate fails → MIS; all green → Auto-Approve. Catalog (G5) + HCP (G6) are resolved "
               "for every **in-scope** path, so the MIS/PV queue receives the AI's full draft — never a blank form.")
    if cls.get("out_of_scope"):
        st.error("🚫 **OUT OF SCOPE — not a medical inquiry (likely spam / off-topic).** "
                 "A reviewer can dismiss this at a glance — no need to read the field details. "
                 f"(Closest class the model weighed: *{cls['predicted_class']}*, but it flagged the doc out-of-scope. "
                 "We don't enrich out-of-scope docs, so catalog/HCP are left unresolved.)")
    g = res["gates"]
    gc1, gc2 = st.columns(2)
    gc1.metric("Catalog match (Gate 5)", g["catalog_match_result"] or "— no match")
    gc2.metric("HCP link (Gate 6)", g["hcp_system_id"] or "— no match")
    render_gates(g["gates"])

    target = g["routing_target"]
    tbanner = {"Auto-Approve": st.success, "MIS Queue": st.warning,
               "PV Queue": st.error, "Unreadable Queue": st.error}.get(target, st.info)
    tbanner(f"**Terminal decision: {target}**  ·  record_status: **{g['record_status']}**")
    for f in g["failed_gates"]:
        st.markdown(f"- {f}")
    st.caption(f"🔖 Audit stamp: model `{ctx['model']}` · prompts `{PROMPT_VERSION}` / `{EXTRACT_PROMPT_VERSION}` "
               "(stamped into the record's ProcessingDecision for traceability).")

    final_rows = []
    if gt is not None:
        final_fields = ["catalog_match_result", "hcp_system_id", "record_status", "routing_target"]
        got_final = {k: g[k] for k in final_fields}
        final_rows = comparison_rows(final_fields, got_final, gt["expected_processing"])
        st.markdown("**Extracted vs golden ground truth:**")
        render_table(final_rows)

    if gt is not None:
        p = sum(tally(rows)[0] for rows in (class_rows, meta_rows, rec_rows, final_rows))
        ntot = sum(tally(rows)[1] for rows in (class_rows, meta_rows, rec_rows, final_rows))
        st.metric("Exact-match fields passing (whole pipeline)", f"{p}/{ntot}",
                  help="Semantic (🟦) and no-ground-truth (⚪) fields are excluded from this count.")

    st.divider()

    # 4 — NOTES
    st.subheader("4 · Notes")
    with st.expander("🧠 Agent notes — known / expected divergences", expanded=True):
        for nnote in AGENT_NOTES:
            st.markdown(f"- {nnote}")
    if ctx["labeling_notes"]:
        with st.expander("🏷️ Ground-truth labeling notes (from the golden case)", expanded=False):
            for nnote in ctx["labeling_notes"]:
                st.markdown(f"- {nnote}")



# ============================================================
# Compare / Sweep — model × prompt-version bake-off
# ============================================================

SHORT_MODEL = {"claude-sonnet-4-6": "Sonnet 4.6", "claude-haiku-4-5-20251001": "Haiku 4.5",
               "gpt-4o": "GPT-4o", "gpt-4o-mini": "GPT-4o mini"}


def _cfg_label(model: str, ver: str) -> str:
    return f"{SHORT_MODEL.get(model, model)} · {AE_EXTRACT_VERSION_INFO.get(ver, {}).get('label', ver)}"


def _smart_labels(configs: list) -> list:
    """Label each config by only the dimension(s) that VARY across the set.
    Comparing two prompts on the same model → labels are just the prompt versions (no model noise)."""
    multi_model = len({m for m, _ in configs}) > 1
    multi_ver = len({v for _, v in configs}) > 1
    out = []
    for m, v in configs:
        parts = []
        if multi_model:
            parts.append(SHORT_MODEL.get(m, m))
        if multi_ver:
            parts.append(AE_EXTRACT_VERSION_INFO.get(v, {}).get("label", v))
        if not parts:  # single config (nothing varies) → show both
            parts = [SHORT_MODEL.get(m, m), AE_EXTRACT_VERSION_INFO.get(v, {}).get("label", v)]
        out.append(" · ".join(parts))
    return out


def _render_prompt_diff(old: str, new: str, old_label: str, new_label: str) -> None:
    """Line-level colored diff: red = only in old, green = only in new (the changed prompt lines)."""
    diff = list(difflib.ndiff(old.splitlines(), new.splitlines()))
    rows = []
    for ln in diff:
        if ln.startswith("? "):
            continue
        txt = html.escape(ln[2:])
        if ln.startswith("- "):
            rows.append(f'<div style="background:#fdecec;color:#a3262c;padding:1px 6px">− {txt}</div>')
        elif ln.startswith("+ "):
            rows.append(f'<div style="background:#eaf6ec;color:#1f7a37;padding:1px 6px">+ {txt}</div>')
        else:
            rows.append(f'<div style="color:#7a8699;padding:1px 6px">&nbsp;&nbsp;{txt}</div>')
    st.caption(f"🔴 removed in {old_label}  ·  🟢 added in {new_label}")
    st.markdown(f'<div style="font-family:ui-monospace,Menlo,monospace;font-size:0.74rem;'
                f'border:1px solid #e6eaf0;border-radius:8px;overflow:auto;max-height:340px">'
                + "".join(rows) + "</div>", unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def _eval_cached(model: str, ver: str, force: bool) -> dict:
    return ev.evaluate_cached(model, ver, force=force)


CRITERIA_HELP = {
    "AE recall (hard floor)": "Of true adverse events, the share the agent flagged (flagged ÷ all true AEs). Must be 100% — a missed AE is a safety failure.",
    "Classification accuracy": "Share of documents given the correct class (correct ÷ classified). Target ≥90%.",
    "Auto-approve precision": "Of cases auto-approved with no human, the share fully correct (correct ÷ auto-approved). Target ≥99% before any no-human commit.",
    "Faithfulness (no hallucination)": "Share of records where every extracted value is supported by the source (LLM-judge). Below 100% = at least one unsupported value.",
    "Symptom coverage (AE)": "Share of AE records that capture every symptom and invent none (LLM-judge). Target 100%.",
    "Metadata draft accuracy": "Share of metadata fields matching ground truth on the human-review path (correct ÷ scored fields). Target ≥95%.",
}


def render_compare(_model: str) -> None:
    st.subheader("⚖️ Model Bake-off")
    st.caption("Which **model** wins? Same golden set, same **current prompts** (prompt iteration lives in "
               "📜 Prompt Library) — **accuracy first, then cost**, judged against the success criteria.")

    # Model selection ONLY — the prompt is pinned to the latest version. Comparing prompts
    # is a separate, model-free concern (the Prompt Library).
    pinned_ver = EXTRACT_PROMPT_VERSION
    models_sel = st.multiselect("Models to compare", MODELS, default=MODELS,
                                format_func=lambda m: SHORT_MODEL.get(m, m))
    if not models_sel:
        st.info("Pick at least one model.")
        return
    st.caption(f"Prompt pinned to the current version (**{AE_EXTRACT_VERSION_INFO[pinned_ver]['label']}** + the rest "
               "of the live prompts).")
    configs = [(m, pinned_ver) for m in models_sel]

    cached = set(ev.list_cached_configs())
    n_uncached = sum(1 for cfg in configs if cfg not in cached)
    tag = f" · {n_uncached} need a live run" if n_uncached else " · all cached (instant)"
    run = st.button(f"▶ Load / run {len(configs)} model(s){tag}", type="primary", key="run_cmp")
    force = st.checkbox("Force re-run live (refresh cache)", value=False, key="cmp_force")
    if run:
        st.session_state["cmp_ran"] = True
    if not st.session_state.get("cmp_ran"):
        st.info("Pick models, then **Load / run**. Cached load instantly; new ones run live once and cache.")
        return

    results = []
    for (m, v) in configs:
        if force or (m, v) not in cached:
            with st.spinner(f"Running {_cfg_label(m, v)} live across the golden set…"):
                res = ev.evaluate_cached(m, v, force=True)
        else:
            res = _eval_cached(m, v, False)
        results.append(((m, v), res))
    labels = _smart_labels([cfg for cfg, _ in results])

    # --- aggregation selector: single run vs median / mean±range of ×3 runs ---
    agg = st.radio("Show", ["Single run", "Median of 3 runs", "Mean ± range of 3 runs"],
                   horizontal=True, key="cmp_agg",
                   help="Single = one pass (a point estimate). Median/Mean±range need the ×3 variance run.")
    var_mode = agg != "Single run"
    if var_mode and not all(ev.has_variance(m, v) for (m, v) in configs):
        st.warning("No ×3 variance cached for every selected model yet. Run **Build-Variance.command** "
                   "(or `python build_variance.py`), then come back. Showing the single run for now.")
        var_mode = False
    # Per-config: stats across the 3 runs (var_mode) keyed by VAR_METRICS.
    stats_by = {cfg: (ev.variance_stats(ev.evaluate_variance(*cfg)) if var_mode else None)
                for cfg, _ in results}

    def _mv(cfg, res, key):  # numeric value for criteria/logic under the current mode
        if var_mode:
            s = stats_by[cfg].get(key)
            return s["median" if "Median" in agg else "mean"] if s else None
        return ev._run_scalars(res).get(key)

    def _fmt(cfg, res, key, kind="pct"):  # display string under the current mode
        def one(v):
            if v is None:
                return "—"
            return (f"{v*100:.0f}%" if kind == "pct" else
                    f"${v:.4f}" if kind == "usd" else f"{v:.0f}s")
        if not var_mode:
            return one(ev._run_scalars(res).get(key))
        s = stats_by[cfg].get(key)
        if not s:
            return "—"
        if "Median" in agg:
            return one(s["median"])
        lo, hi = (one(s["min"]).rstrip("%s$"), one(s["max"]).rstrip("%s$"))
        return f"{one(s['mean'])}  ({lo}–{hi})"

    # --- GO / NO-GO (criteria computed on the chosen aggregate) ---
    st.markdown("#### Success criteria — GO / NO-GO")
    def _synth(cfg, res):
        return {"ae_recall": _mv(cfg, res, "ae_recall"),
                "classification_accuracy": _mv(cfg, res, "classification_accuracy"),
                "auto_approve_precision": _mv(cfg, res, "auto_approve_precision"),
                "metadata_accuracy": _mv(cfg, res, "metadata_accuracy"),
                "graders": {"faithfulness": {"pass_rate": _mv(cfg, res, "faithfulness")},
                            "symptom_coverage": {"pass_rate": _mv(cfg, res, "symptom_coverage")}}}
    crits = [ev.success_criteria(_synth(cfg, res)) for (cfg, res) in results]
    crit_names = [c["name"] for c in crits[0]]
    rows = []
    for i, name in enumerate(crit_names):
        row = {"Criterion": f"{name}  (≥{crits[0][i]['target']:.0%})"}
        for lab, cl in zip(labels, crits):
            c = cl[i]
            row[lab] = ("✅" if c["pass"] else "⚠️") + f" {_pct(c['value']) if c['value'] is not None else '—'}"
        rows.append(row)
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    st.caption(f"✅ meets the bar · ⚠️ below it. {'Pass/fail judged on the ' + agg.lower() + '.' if var_mode else 'Single-run point estimate.'}")
    with st.expander("ⓘ What each criterion means & how it's computed"):
        for name in crit_names:
            st.markdown(f"**{name}** — {CRITERIA_HELP.get(name, '—')}")

    # --- metrics + cost ---
    st.markdown("#### Metrics & cost")
    metric_rows = [  # (label, VAR_METRICS key, kind)
        ("Classification accuracy", "classification_accuracy", "pct"),
        ("Terminal routing accuracy", "terminal_accuracy", "pct"),
        ("AE recall (floor 100%)", "ae_recall", "pct"),
        ("Auto-approve precision", "auto_approve_precision", "pct"),
        ("Metadata accuracy", "metadata_accuracy", "pct"),
        ("Faithfulness — no hallucination", "faithfulness", "pct"),
        ("Symptom coverage (AE)", "symptom_coverage", "pct"),
        ("Latency — full set (s)", "latency_s", "lat"),
        ("Est. cost — full set (USD)", "est_usd", "usd"),
    ]
    table = []
    for name, key, kind in metric_rows:
        row = {"Metric": name}
        for lab, (cfg, res) in zip(labels, results):
            row[lab] = _fmt(cfg, res, key, kind)
        table.append(row)
    st.dataframe(pd.DataFrame(table), hide_index=True, use_container_width=True)
    n_runs = f" · median/range over {stats_by[results[0][0]]['ae_recall']['n']} runs" if (var_mode and stats_by[results[0][0]].get('ae_recall')) else ""
    st.caption(f"⚖️ n={results[0][1]['n']} cases{n_runs}. Cost = est. list price; latency from the live run.")

    # --- where it went wrong: every error type, only failing cases ---
    st.markdown("#### 🔎 Where it went wrong — every error, by case")
    st.caption("Only cases with a problem appear. Covers misclassification, misrouting, metadata mismatches, "
               "and the LLM-judge graders (faithfulness, symptom coverage, summary). Click a case for the detail.")
    for lab, (_, res) in zip(labels, results):
        problems = []
        for e in res["evals"]:
            errs = []
            if e.get("class_correct") is False:
                errs.append(f"**Misclassified** — predicted `{e.get('predicted_class')}`, expected `{e.get('gt_class')}`")
            if e.get("terminal_correct") is False:
                errs.append(f"**Misrouted** — went to `{e.get('terminal_pred')}`, expected `{e.get('terminal_gt')}`")
            meta_fail = [f for f, s in e.get("metadata_results", []) if s == "fail"]
            if meta_fail:
                errs.append("**Metadata mismatch** — " + ", ".join(meta_fail))
            for gg in e.get("graders", []):
                if not gg.get("passed"):
                    nm = gg["grader"].replace("_", " ")
                    extra = f" (score {gg.get('score')}/5)" if gg["grader"] == "summary_accuracy" else ""
                    errs.append(f"**{nm.title()}{extra}** — {gg.get('reasoning', '')}")
            if errs:
                problems.append((e["case_id"], e, errs))
        if not problems:
            st.markdown(f"**{lab}** — ✅ clean: no errors across any dimension")
            continue
        st.markdown(f"**{lab}** — ⚠️ {len(problems)} case(s) with an issue:")
        for cid, e, errs in problems:
            with st.expander(f"⚠️ {cid} · {e.get('predicted_class', '—')} → {e.get('terminal_pred', '—')} · {len(errs)} issue(s)"):
                for er in errs:
                    st.markdown("- " + er)
                st.caption("Open this case in the 🔬 Case Inspector to see it against the source.")

    # --- per-case diff between two models ---
    if len(results) >= 2:
        st.markdown("#### What differs between two models")
        st.caption("Pick two models to see which cases they handle differently (same prompt) — class, route, "
                   "or faithfulness verdict.")
        pick = st.multiselect("Compare two models", labels, default=labels[:2],
                              key="cmp_flip", max_selections=2)
        if len(pick) == 2:
            ia, ib = labels.index(pick[0]), labels.index(pick[1])
            # Column headers: re-label from ONLY the two picked configs, so two same-model
            # configs read as just the prompt versions (drop the redundant model) even when
            # the broader bake-off spans both models.
            col_a, col_b = _smart_labels([results[ia][0], results[ib][0]])
            ea = {e["case_id"]: e for e in results[ia][1]["evals"]}
            eb = {e["case_id"]: e for e in results[ib][1]["evals"]}

            def faith(e):
                for gg in e.get("graders", []):
                    if gg["grader"] == "faithfulness":
                        return "pass" if gg["passed"] else "❌ FAIL"
                return "—"

            diff = []
            for cid in sorted(ea):
                a, b = ea[cid], eb.get(cid, {})
                ka = (a.get("predicted_class"), a.get("terminal_pred"), faith(a))
                kb = (b.get("predicted_class"), b.get("terminal_pred"), faith(b))
                if ka != kb:
                    diff.append({"Case": cid,
                                 col_a: f"{ka[0]} → {ka[1]} · faith {ka[2]}",
                                 col_b: f"{kb[0]} → {kb[1]} · faith {kb[2]}"})
            if diff:
                st.dataframe(pd.DataFrame(diff), hide_index=True, use_container_width=True)
                st.caption("Only cases that differ are shown — class, route, or faithfulness verdict.")
            else:
                st.success("No case-level differences (class, route, and faithfulness identical across these two).")

    # Prompts are pinned to the current version here; iterating/diffing prompts is a
    # separate, model-free concern in the Prompt Library.
    st.caption("All models above run the **current prompts** (latest version). To iterate or diff prompt "
               "versions, see **📜 Prompt Library** (model-independent).")


# ============================================================
# Prompt Library — every prompt, versioned & tracked (one place)
# ============================================================

def render_prompts(_model: str) -> None:
    # Safety net: capture any prompt edited since its last snapshot (so nothing is ever lost,
    # even if prompts.py was edited directly). No change → no-op. Runs once per session.
    if not st.session_state.get("_prompts_synced"):
        newly = prompt_store.sync()
        st.session_state["_prompts_synced"] = True
        if newly:
            st.toast(f"📜 Auto-captured {len(newly)} changed prompt version(s) — annotate the reason in the Library.")
    st.subheader("📜 Prompt Library — versioned & diff-able")
    st.caption("Every prompt the agent uses, in one place. We store the **full text of each version**, so any two are "
               "truly comparable. A prompt shows multiple versions **only where we kept the older text** — otherwise "
               "it's just the current version. We never show a version number we can't reproduce.")
    st.caption("New versions are saved by **`prompt_store.save_version(text, what, why)`** whenever a prompt changes — "
               "so this stays real as the agent evolves. Records carry the pipeline stamp "
               "(`classify_v3` / `extract_v7`) for the 21 CFR Part 11 audit trail.")

    # Version-history overview — every REAL stored version across all prompts, newest first.
    st.markdown("#### Version history")
    name_by_id = {p["id"]: p["name"] for p in PROMPT_REGISTRY}
    vh = []
    for p in PROMPT_REGISTRY:
        for v in prompt_store.versions(p["id"]):
            vh.append({"Prompt": name_by_id[p["id"]], "Version": f"v{v['version']}",
                       "Date": v["date"], "What changed": v["change"], "Why changed": v["why"]})
    vh.sort(key=lambda r: (r["Date"], r["Prompt"], r["Version"]), reverse=True)
    st.dataframe(pd.DataFrame(vh), hide_index=True, use_container_width=True)
    st.caption("Only versions whose full text we kept appear here — most prompts have just their current version; "
               "the AE extractor has the kept v1→v2 (clinical → verbatim) A/B. Expand a prompt below to read its text or diff versions.")
    st.divider()

    for p in PROMPT_REGISTRY:
        vs = prompt_store.versions(p["id"])
        if not vs:
            continue
        n = len(vs)
        tag = f"{n} versions · 🔬 diff-able" if n >= 2 else "1 version (current)"
        with st.expander(f"{p['stage']}  ·  {p['name']}  —  {tag}"):
            st.markdown(f"*{p['purpose']}*")

            if n >= 2:
                # Table A — the real version history (only versions whose text we kept)
                st.markdown("**Table A — Version history**")
                hist = [{"Version": f"v{v['version']}", "Date": v["date"],
                         "What changed": v["change"], "Why changed": v["why"]} for v in vs]
                st.dataframe(pd.DataFrame(hist), hide_index=True, use_container_width=True)

                # Table B — pick two versions → red/green diff
                st.markdown("**Table B — Compare two versions** *(pick 2 → red/green diff)*")
                opts = [f"v{v['version']}" for v in vs]
                pick = st.multiselect("versions", opts, default=[opts[-2], opts[-1]],
                                      key=f"plib_{p['id']}", max_selections=2, label_visibility="collapsed")
                if len(pick) == 2:
                    by = {f"v{v['version']}": v for v in vs}
                    a, b = sorted((by[pick[0]], by[pick[1]]), key=lambda v: v["version"])  # old → new
                    _render_prompt_diff(a["text"], b["text"],
                                        f"v{a['version']} · {a['change']}", f"v{b['version']} · {b['change']}")
                else:
                    st.caption("Pick exactly two versions to see the diff.")
            else:
                st.caption("Only one version is stored, so there's nothing to diff yet. "
                           "The next change will save a new version here automatically.")

            st.markdown(f"**Current prompt text (v{vs[-1]['version']}):**")
            st.code(vs[-1]["text"], language=None)


# ============================================================
# Page + sidebar + tabs
# ============================================================

st.set_page_config(page_title="Intake Agent — Evaluation Console", layout="wide")
st.title("🔬 Document Intake Agent — Evaluation Console")
st.caption("Collective eval across the golden set + per-case pipeline trace, validated against ground truth.")
st.markdown(TABLE_CSS, unsafe_allow_html=True)

with st.sidebar:
    st.header("Controls")
    model = st.selectbox("Model", MODELS, index=0,
                         help="Used by both views — the Sonnet ↔ Haiku bake-off swap.")
    st.caption("**📊 Eval Dashboard** — PROVEN-IF metrics across the whole golden set.\n\n"
               "**🔬 Case Inspector** — trace one document through every pipeline stage.")

# Persistent view switcher (survives reruns — clicking a Run button won't bounce the view).
# NOTE: the reviewer experience lives in the separate Lovable app (Vigilant.AI) — this
# console is the PM/eval surface, so no Reviewer Workspace tab here.
view = st.radio("View", ["📊 Eval Dashboard", "⚖️ Compare / Sweep", "📜 Prompt Library",
                         "🔬 Case Inspector"],
                horizontal=True, label_visibility="collapsed", key="view")

st.divider()

if view == "🔬 Case Inspector":
    render_case_inspector(model)
elif view == "⚖️ Compare / Sweep":
    render_compare(model)
elif view == "📜 Prompt Library":
    render_prompts(model)
else:
    render_dashboard(model)
