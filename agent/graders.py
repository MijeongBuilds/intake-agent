"""
graders.py — LLM-as-judge graders for the free-text / semantic fields that exact
match can't score.

Each grader is a single-responsibility LLM call that reads the SOURCE DOCUMENT + the
agent's output and returns a verdict WITH step-by-step reasoning (auditable). Template
(from the AI Eval Lab): "Your ONLY job is X. You are NOT judging Y." → steps → reasoning
→ verdict. Pass/fail for safety-critical; 1-5 for quality.

Three graders:
  - faithfulness     (pass/fail)  : every extracted value is grounded in the document
  - symptom_coverage (pass/fail)  : adverse_symptoms_list captures ALL symptoms, none invented
  - summary_accuracy (1-5, ≥4)    : inquiry_summary_text faithfully captures the inquiry

CAVEAT: the judge LLM has its own biases — spot-check against human judgment.
"""

from pydantic import BaseModel, Field

from agent_engine import get_llm, MODEL


# ---- structured verdicts ----

class PassFailVerdict(BaseModel):
    reasoning: str          # step-by-step (the audit trail)
    passed: bool


class ScoreVerdict(BaseModel):
    reasoning: str
    score: int = Field(ge=1, le=5)


def _judge(system: str, user: str, schema, model: str):
    # temperature=0 judge for reproducibility (via get_llm).
    return get_llm(model).with_structured_output(schema).invoke(
        [("system", system), ("human", user)])


def _doc_block(document_text: str) -> str:
    return f'SOURCE DOCUMENT:\n"""\n{document_text}\n"""'


# ============================================================
# 1. faithfulness — every extracted value is grounded in the document
# ============================================================

FAITHFULNESS_SYSTEM = """You are a faithfulness checker for a pharmaceutical document-intake agent. Your ONLY job is to verify that every value in the EXTRACTED RECORD is supported by the SOURCE DOCUMENT. You are NOT judging completeness, tone, or whether values match an answer key — only whether each value is grounded in the document (no hallucination, no contradiction).

Steps:
1. List each concrete value in the extracted record (names, products, symptoms, dates, sections, etc.).
2. Cross-reference each against the source document.
3. Flag HALLUCINATIONS (a value appearing nowhere in the document) and CONTRADICTIONS (a value conflicting with the document).

Outcome:
- passed=true: every concrete value is supported by the document. Zero hallucinations, zero contradictions.
- passed=false: ANY hallucinated or contradicted value.

Rules:
- Be strict — a single invented value is a fail.
- Legitimate derivations are NOT hallucinations if grounded: phone normalized to E.164, patient name reduced to initials, country inferred from a stated address, reporter_type inferred from a stated credential.
- A value of None/empty is never a hallucination."""


def grade_faithfulness(document_text: str, content: dict, model: str = MODEL) -> dict:
    fields = "\n".join(f"- {k}: {v}" for k, v in content.items() if v not in (None, [], ""))
    user = f"{_doc_block(document_text)}\n\nEXTRACTED RECORD (judge these values):\n{fields}"
    v = _judge(FAITHFULNESS_SYSTEM, user, PassFailVerdict, model)
    return {"grader": "faithfulness", "type": "passfail",
            "passed": v.passed, "reasoning": v.reasoning}


# ============================================================
# 2. symptom_coverage — all AE symptoms captured, none invented
# ============================================================

SYMPTOM_COVERAGE_SYSTEM = """You are a symptom-coverage checker for adverse-event (AE) reports. Your ONLY job is to verify that the extracted adverse_symptoms_list captures EVERY adverse symptom described in the source document, and invents none. You are NOT judging seriousness, wording, or other fields — only completeness (recall) and no-invention (precision) of the symptom list.

This is safety-critical: a MISSED symptom is the worst failure (AE recall floor).

Steps:
1. From the document, list every adverse symptom / reaction the patient experiences (plain terms).
2. Compare against the extracted list. Match by MEANING, not exact words ("racing heartbeat" = "tachycardia").
3. Flag any document symptom MISSING from the list, and any list item INVENTED (not in the document).

Outcome:
- passed=true: every document symptom is represented (recall) AND none invented (precision).
- passed=false: ANY missed symptom OR any invented symptom."""


def grade_symptom_coverage(document_text: str, symptoms_list, model: str = MODEL) -> dict:
    sl = ", ".join(symptoms_list) if symptoms_list else "(empty)"
    user = f"{_doc_block(document_text)}\n\nEXTRACTED adverse_symptoms_list:\n{sl}"
    v = _judge(SYMPTOM_COVERAGE_SYSTEM, user, PassFailVerdict, model)
    return {"grader": "symptom_coverage", "type": "passfail",
            "passed": v.passed, "reasoning": v.reasoning}


# ============================================================
# 3. summary_accuracy — does the summary capture the actual inquiry (1-5)
# ============================================================

SUMMARY_ACCURACY_SYSTEM = """You are a summary-accuracy evaluator for medical-information inquiry records. Your ONLY job is to rate how faithfully the extracted inquiry_summary_text captures the ACTUAL inquiry in the source document. You are NOT judging tone or length — only accuracy and completeness of the summary relative to what the document is asking.

Score 1-5:
5 - Captures the exact inquiry precisely and completely.
4 - Captures the inquiry with a minor omission or imprecision.
3 - Roughly right but misses an important aspect, or is vague.
2 - Partially wrong or misses the main point.
1 - Wrong inquiry, or fabricates the ask.

Give brief reasoning, then the score."""

SUMMARY_THRESHOLD = 4


def grade_summary_accuracy(document_text: str, summary_text: str, model: str = MODEL) -> dict:
    user = f"{_doc_block(document_text)}\n\nEXTRACTED inquiry_summary_text:\n{summary_text}"
    v = _judge(SUMMARY_ACCURACY_SYSTEM, user, ScoreVerdict, model)
    return {"grader": "summary_accuracy", "type": "score", "score": v.score,
            "passed": v.score >= SUMMARY_THRESHOLD, "threshold": SUMMARY_THRESHOLD,
            "reasoning": v.reasoning}


# ============================================================
# dispatch — run the applicable graders for one extracted case
# ============================================================

# common-metadata content fields the LLM actually read (not system/envelope).
_CONTENT_META = ["customer_org", "reporter_name", "reporter_type", "contact_email",
                 "contact_phone", "country_code", "language", "product_mentioned"]

# Record fields that are NOT extracted-from-document, so faithfulness must NOT judge them:
# retrieval results, code-derived flags, in-prompt judgments, and reasoning/labels.
_NOT_FROM_DOCUMENT = {
    "record_type",           # constant label
    "suggested_srd_match",   # SRD retrieval (grounded in the registry, not the doc)
    "is_valid_icsr",         # derived in code from the 4 pillars
    "ae_seriousness",        # judgment vs the 5 ICH criteria, not a stated fact
    "seriousness_rationale", # the judge's reasoning, not an extracted fact
}


def grade_case(document_text: str, common: dict, record, model: str = MODEL) -> list:
    """Run every grader that applies to this extracted case. Returns a list of verdicts."""
    results = []

    # faithfulness — judge ONLY values extracted from the document (common content + the
    # extracted record fields), excluding retrieval / code-derived / judgment fields.
    content = {k: common.get(k) for k in _CONTENT_META}
    if record:
        content.update({k: v for k, v in record.items() if k not in _NOT_FROM_DOCUMENT})
    results.append(grade_faithfulness(document_text, content, model))

    if record:
        if "adverse_symptoms_list" in record:
            results.append(grade_symptom_coverage(document_text, record["adverse_symptoms_list"], model))
        if "inquiry_summary_text" in record:
            results.append(grade_summary_accuracy(document_text, record["inquiry_summary_text"], model))

    return results
