"""
agent_engine.py — the intake agent.

Built one node at a time. Right now it contains the CLASSIFY step:
  document text  ->  Claude  ->  validated ClassifierOutput (class + AE flag + confidence)

The model is a CONFIG variable (MODEL) so the bake-off is a one-word swap, not a rebuild.
"""

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from pydantic import BaseModel

from schemas import (
    ClassifierOutput,
    CommonMetadata,
    ExtractedMetadata,
    ClassifierScores,
    DocClass,
    OnLabelRecord,
    OffLabelRecord,
    AEReportRecord,
    Seriousness,
    TransactionalRecord,
)
from prompts import (
    CLASSIFY_SYSTEM_PROMPT,
    build_classify_user_message,
    EXTRACT_METADATA_SYSTEM_PROMPT,
    build_extract_metadata_user_message,
    ONLABEL_EXTRACT_SYSTEM_PROMPT,
    build_onlabel_extract_user_message,
    OFFLABEL_EXTRACT_SYSTEM_PROMPT,
    build_offlabel_extract_user_message,
    AE_EXTRACT_SYSTEM_PROMPT,
    get_ae_extract_prompt,
    AE_SERIOUSNESS_SYSTEM_PROMPT,
    build_ae_extract_user_message,
)
from config import get_taxonomy, get_thresholds, get_reporting_deadlines, get_srds, get_catalog, get_hcps
from typing import Optional
from datetime import date, timedelta

# override=True: Claude Code's shell pre-sets ANTHROPIC_API_KEY="" which would otherwise win.
load_dotenv(override=True)

# --- model config (swap this one string for the bake-off) ---
MODEL = "claude-sonnet-4-6"


def get_llm(model: str = MODEL, temperature: float = 0):
    # temperature=0 -> deterministic, reproducible (needed for calibration). The seriousness
    # self-consistency vote overrides this with temperature>0 so its N samples diverge.
    # Provider-agnostic: Anthropic models use ChatAnthropic; anything else (gpt-*, o-*) routes
    # to ChatOpenAI so the bake-off can compare across providers. (Needs langchain-openai +
    # OPENAI_API_KEY for the OpenAI path.)
    if not model.startswith("claude"):
        from langchain_openai import ChatOpenAI  # optional dependency
        return ChatOpenAI(model=model, temperature=temperature, max_tokens=1024)
    return ChatAnthropic(model=model, temperature=temperature, max_tokens=1024)


# ============================================================
# PRE-CLASSIFY READABILITY GATE (Gate 1, early half) — NO LLM
# ============================================================
# Runs BEFORE classify. If the document is unreadable (OCR below the tenant's
# floor), halt immediately and route to the Unreadable queue — don't spend an
# LLM call on junk text. Readable docs proceed to classify -> extract -> gates.
# (Production: OCR is per-page; here ocr_confidence is the worst-page score.)

def check_readability(input_metadata: dict, tenant_id: str) -> dict:
    """Return {readable, ocr_confidence, reason}. readable=False => halt before classify."""
    th = get_thresholds(tenant_id)
    ocr = input_metadata.get("ocr_confidence")
    if ocr is not None and ocr < th["ocr_unreadable_below"]:
        return {
            "readable": False,
            "ocr_confidence": ocr,
            "reason": (f"Gate 1 (pre-classify): OCR {ocr:.2f} < {th['ocr_unreadable_below']} "
                       f"— unreadable; halted before classification (no LLM spent)"),
        }
    return {"readable": True, "ocr_confidence": ocr, "reason": None}


def classify(document_text: str, tenant_id: str, model: str = MODEL) -> ClassifierOutput:
    """One LLM call -> per-class fit scores -> a ClassifierOutput with DERIVED routing numbers.

    The model scores every class (0-1); the CODE then computes the predicted_class (argmax),
    class_confidence (top score), and classification_conflict_margin (top1 - top2). This makes
    the routing-critical numbers *computed*, not self-reported — the reliable signal (the old
    self-reported confidence/margin fluttered run-to-run). AE flag / scope / pillars stay LLM judgments.

    The tenant's taxonomy is loaded from configs/<tenant_id>.yaml at runtime — same code for
    every customer; only the config differs.
    """
    taxonomy = get_taxonomy(tenant_id)
    # The company's product portfolio (brand names) — so the classifier can flag an inquiry
    # SOLELY about another company's drug as out-of-scope (classify_v4). Bounded + stable;
    # the full catalog (IDs, exact resolution) stays a downstream tool at Gate 5.
    portfolio = [p["brand_name"] for p in get_catalog(tenant_id)]
    llm = get_llm(model).with_structured_output(ClassifierScores)
    raw: ClassifierScores = llm.invoke([
        ("system", CLASSIFY_SYSTEM_PROMPT),
        ("human", build_classify_user_message(document_text, taxonomy, portfolio)),
    ])

    ranked = sorted(raw.class_scores, key=lambda c: c.fit_score, reverse=True)
    top = ranked[0].fit_score
    second = ranked[1].fit_score if len(ranked) > 1 else 0.0

    # Out-of-scope sentinel: if the doc isn't a medical inquiry at all (spam/marketing/wrong-company),
    # the predicted_class becomes the explicit "Out of scope" value rather than the noisy argmax of
    # the 10 real classes. (The top real class is still kept in class_scores for the reject-option view.)
    predicted = DocClass.OUT_OF_SCOPE if raw.out_of_scope else ranked[0].doc_class

    return ClassifierOutput(
        predicted_class=predicted,
        class_confidence=top,
        classification_conflict_margin=round(top - second, 4),
        classification_rationale=raw.classification_rationale,
        class_scores=ranked,
        adverse_event_flag=raw.adverse_event_flag,
        out_of_scope=raw.out_of_scope,
        has_patient_pillar=raw.has_patient_pillar,
        has_reporter_pillar=raw.has_reporter_pillar,
        has_product_pillar=raw.has_product_pillar,
        has_event_pillar=raw.has_event_pillar,
    )


# NOTE: there is NO separate pre-extraction router anymore. All 7 gates run
# together AFTER extraction, as one unit, in evaluate_gates() (below). Order:
#   classify -> extract (full draft, always) -> evaluate_gates -> terminal decision.
# This guarantees every path (Auto-Approve / MIS / PV) carries the AI's full draft.


# ============================================================
# EXTRACT — common metadata + the per-class transactional record
# ============================================================

def extract_metadata(document_text: str, input_metadata: dict,
                     model: str = MODEL) -> CommonMetadata:
    """LLM extracts the document-body fields; the ingestion ENVELOPE supplies the
    rest. The model never guesses tenant_id / channel / dates the system already
    knows — it only reads what's in the document.
    """
    llm = get_llm(model).with_structured_output(ExtractedMetadata)
    messages = [
        ("system", EXTRACT_METADATA_SYSTEM_PROMPT),
        ("human", build_extract_metadata_user_message(document_text)),
    ]
    extracted: ExtractedMetadata = llm.invoke(messages)

    # Merge: envelope (system-known) + extracted (document-read) -> full contract.
    return CommonMetadata(
        tenant_id=input_metadata["tenant_id"],
        case_id=input_metadata.get("case_id", input_metadata["document_id"]),
        document_id=input_metadata["document_id"],
        source_channel=input_metadata["source_channel"],
        received_date=input_metadata["received_date"],
        **extracted.model_dump(),
    )


def _extract_on_label(document_text: str, tenant_id: str, model: str) -> OnLabelRecord:
    # Pass only id/topic/section of the tenant's SRDs (not the response text) so the
    # model can SUGGEST a pre-approved doc — retrieval, never authoring.
    srds = get_srds(tenant_id)
    llm = get_llm(model).with_structured_output(OnLabelRecord)
    messages = [
        ("system", ONLABEL_EXTRACT_SYSTEM_PROMPT),
        ("human", build_onlabel_extract_user_message(document_text, srds)),
    ]
    return llm.invoke(messages)


class _SeriousnessVote(BaseModel):
    seriousness_rationale: str
    ae_seriousness: Seriousness


# Self-consistency: seriousness is the ONE field that is both unstable (borderline AE cases
# flip between runs) and safety-critical, so we judge it best-of-N instead of single-shot.
SERIOUSNESS_VOTES = 3          # majority of N
SERIOUSNESS_TEMPERATURE = 0.8  # >0 so the N samples explore different reasoning paths (temp 0 = identical, no vote)


def _judge_seriousness(document_text: str, model: str = MODEL,
                       n: int = SERIOUSNESS_VOTES) -> tuple:
    """Sample the ICH seriousness verdict N times at temperature>0 and take the majority.
    Returns (Seriousness, rationale stamped with the vote tally for the audit trail).
    """
    llm = get_llm(model, temperature=SERIOUSNESS_TEMPERATURE).with_structured_output(_SeriousnessVote)
    msg = [("system", AE_SERIOUSNESS_SYSTEM_PROMPT),
           ("human", build_ae_extract_user_message(document_text))]
    votes = [llm.invoke(msg) for _ in range(n)]
    serious_n = sum(1 for v in votes if v.ae_seriousness == Seriousness.SERIOUS)
    verdict = Seriousness.SERIOUS if serious_n * 2 > n else Seriousness.NON_SERIOUS
    winner = next(v for v in votes if v.ae_seriousness == verdict)
    tally = f"[self-consistency: {serious_n}/{n} Serious] "
    return verdict, tally + (winner.seriousness_rationale or "")


def _extract_off_label(document_text: str, model: str) -> OffLabelRecord:
    llm = get_llm(model).with_structured_output(OffLabelRecord)
    messages = [
        ("system", OFFLABEL_EXTRACT_SYSTEM_PROMPT),
        ("human", build_offlabel_extract_user_message(document_text)),
    ]
    return llm.invoke(messages)


def _extract_ae(document_text: str, classification: ClassifierOutput,
                model: str, ae_extract_version: Optional[str] = None) -> AEReportRecord:
    llm = get_llm(model).with_structured_output(AEReportRecord)
    messages = [
        ("system", get_ae_extract_prompt(ae_extract_version)),
        ("human", build_ae_extract_user_message(document_text)),
    ]
    record: AEReportRecord = llm.invoke(messages)

    # Seriousness is unstable on borderline cases -> override the single-shot value with a
    # best-of-N self-consistency vote (applied ONLY here, not blanket across all fields).
    record.ae_seriousness, record.seriousness_rationale = _judge_seriousness(document_text, model)

    # is_valid_icsr is a DERIVED field — recompute it deterministically from the
    # four legal pillars rather than trusting the LLM's self-report.
    record.is_valid_icsr = (
        classification.has_patient_pillar
        and classification.has_reporter_pillar
        and classification.has_product_pillar
        and classification.has_event_pillar
    )
    return record


# Classes 9 (Legal/Reg) and 10 (Public FAQ) produce NO record -> None.
NO_RECORD_CLASSES = {DocClass.LEGAL_REG, DocClass.PUBLIC_FAQ, DocClass.OUT_OF_SCOPE}


class UnbuiltClassError(NotImplementedError):
    """The predicted class has no automated record extractor yet.

    Subclasses NotImplementedError (existing fail-safe catches keep working) and
    carries the already-extracted CommonMetadata, so the human still gets the
    reporter/product/contact pre-filled — only the class-specific record stays manual.
    """

    def __init__(self, message: str, common=None):
        super().__init__(message)
        self.common = common


def extract_record(document_text: str, classification: ClassifierOutput,
                   tenant_id: str, model: str = MODEL,
                   ae_extract_version: Optional[str] = None) -> Optional[TransactionalRecord]:
    """Dispatch to the per-class extractor for the predicted class.

    On-Label + AE are implemented (the two golden cases). The remaining record
    classes raise NotImplementedError so the gap is loud, not silent.
    """
    cls = classification.predicted_class

    if cls in NO_RECORD_CLASSES:
        return None
    if cls is DocClass.ON_LABEL:
        return _extract_on_label(document_text, tenant_id, model)
    if cls is DocClass.OFF_LABEL:
        return _extract_off_label(document_text, model)
    if cls is DocClass.ADVERSE_EVENT:
        return _extract_ae(document_text, classification, model, ae_extract_version)

    raise NotImplementedError(f"extract_record not yet built for class: {cls.value}")


def extract(document_text: str, classification: ClassifierOutput,
            input_metadata: dict, model: str = MODEL,
            ae_extract_version: Optional[str] = None) -> tuple:
    """Full extract node: (CommonMetadata, transactional_record-or-None).

    Runs regardless of early routing — an AE bound for the PV queue still needs
    its draft ICSR built for the human reviewer. `ae_extract_version` selects the
    AE-extract prompt variant (Compare/Sweep); None = current.
    """
    common = extract_metadata(document_text, input_metadata, model)
    try:
        record = extract_record(document_text, classification, input_metadata["tenant_id"],
                                model, ae_extract_version)
    except UnbuiltClassError:
        raise
    except NotImplementedError as e:
        # Don't throw away the common metadata we already extracted — the human
        # reviewer still gets a pre-filled draft for everything but the record.
        raise UnbuiltClassError(str(e), common=common) from e
    return common, record


# ============================================================
# REFERENCE-DATA RESOLVERS (Gate 5 + Gate 6 entity linking — NO LLM)
# ============================================================

def resolve_catalog_match(product_mentioned: Optional[str], tenant_id: str) -> Optional[str]:
    """Gate 5: link the extracted brand name to an ACTIVE catalog entry -> catalog_id, else None."""
    if not product_mentioned:
        return None
    norm = product_mentioned.strip().casefold()
    for p in get_catalog(tenant_id):
        if p["brand_name"].strip().casefold() == norm and p.get("is_active", True):
            return p["catalog_id"]
    return None


def reporting_deadline(day_zero: Optional[date], seriousness, active_study: bool,
                       deadlines: dict) -> tuple:
    """Compute the regulatory reporting due date + regime from Day 0 (MedInfo intake).

    The clock starts when the file hits Medical Information — NOT when the case is
    escalated to PV. Post-marketing spontaneous is the default; an active study (or the
    product used as a comparator/concomitant) switches to the clinical-trial (SUSAR) clock.
    Returns (due_date | None, regime | None). Only AE cases (seriousness set) get a date.
    """
    if day_zero is None or seriousness is None:
        return None, None
    serious = getattr(seriousness, "value", seriousness) == "Serious"
    if active_study:
        regime = "Clinical Trial (SUSAR)"
        # fatal/life-threatening = clinical_trial_life_threatening (7d) — not separately detected in MVP.
        days = deadlines["clinical_trial_serious"] if serious else None  # non-serious SUSAR isn't expedited
    else:
        regime = "Post-Marketing (Spontaneous)"
        days = deadlines["post_marketing_serious"] if serious else deadlines["post_marketing_non_serious"]
    due = day_zero + timedelta(days=days) if days is not None else None
    return due, regime


def resolve_hcp_link(reporter_name: Optional[str], tenant_id: str) -> Optional[str]:
    """Gate 6: link the reporter to the HCP registry -> hcp_system_id, else None.

    Consumers are intentionally absent from the registry, so they resolve to None
    (which is correct — a consumer report can't auto-approve on the HCP gate).
    """
    if not reporter_name:
        return None
    norm = reporter_name.strip().casefold()
    for h in get_hcps(tenant_id):
        if h["name"].strip().casefold() == norm:
            return h["hcp_system_id"]
    return None


# ============================================================
# 8-GATE COMPLIANCE UNIT -> terminal decision (NO LLM)
# ============================================================
# Runs AFTER extraction, as ONE unit:
#   classify -> extract (full draft, always) -> THIS -> terminal decision + route.
# Because extraction always runs first, EVERY path (Auto-Approve / MIS / PV) carries
# the AI's full draft (metadata + record + resolved catalog/HCP) into its queue —
# the human reviews a pre-filled draft, never a blank form.

# Plain-English meaning of a FAILED gate — shown to reviewers so they know where to focus.
GATE_HINTS = {
    "Gate 1": "scan/text quality too low to fully trust",
    "Gate 2": "the model isn't confident which class this is",
    "Gate 4": "a patient safety event (adverse event) was detected",
    "Gate 5": "the product isn't one we make (not in the catalog)",
    "Gate 6": "the reporter isn't a recognized HCP in our registry",
    "Gate 7": "the extracted data didn't fit the required structure",
    "Gate 8": "this isn't an in-scope medical inquiry",
}


def evaluate_gates(classification: ClassifierOutput, common: CommonMetadata,
                   record, input_metadata: dict, tenant_id: str) -> dict:
    """Run all 8 compliance gates as one unit and return the terminal decision.

    Returns:
      routing_target       : "Auto-Approve" | "MIS Queue" | "PV Queue"
      record_status        : "Approved" | "Pending Review"
      catalog_match_result : resolved catalog_id or None  (Gate 5)
      hcp_system_id        : resolved hcp id or None       (Gate 6)
      failed_gates         : the gate strings that drove the route (audit list)
      gates                : per-gate [{id, name, passed, detail}] for display
    """
    th = get_thresholds(tenant_id)

    # Entity linking — resolved for every IN-SCOPE case (draft enrichment on all paths).
    # Out-of-scope docs (spam/off-topic) are NOT enriched: a spam blast that name-drops a real
    # product shouldn't show a clean catalog/HCP match — it's not a real inquiry. (Gate 8 routes it.)
    if classification.out_of_scope:
        catalog_id = hcp_id = None
    else:
        catalog_id = resolve_catalog_match(common.product_mentioned, tenant_id)
        hcp_id = resolve_hcp_link(common.reporter_name, tenant_id)
    ocr = input_metadata.get("ocr_confidence")

    # Day 0 = MedInfo intake date -> start the regulatory reporting clock (AE cases only).
    rd = input_metadata.get("received_date")
    day_zero = date.fromisoformat(rd) if isinstance(rd, str) else rd
    due_date, regime = reporting_deadline(
        day_zero,
        getattr(record, "ae_seriousness", None),
        getattr(common, "active_study_flag", False),
        get_reporting_deadlines(tenant_id),
    )

    gates = []

    def add(gid, name, passed, detail):
        gates.append({"id": gid, "name": name, "passed": passed, "detail": detail,
                      "hint": GATE_HINTS.get(gid, "")})

    # Gate 1 — text quality (OCR)
    if ocr is None:
        add("Gate 1", "Text quality (OCR)", True, "digital-native (no OCR)")
    elif ocr < th["ocr_unreadable_below"]:
        add("Gate 1", "Text quality (OCR)", False, f"unreadable: OCR {ocr:.2f} < {th['ocr_unreadable_below']}")
    elif ocr < th["ocr_auto_approve_min"]:
        add("Gate 1", "Text quality (OCR)", False, f"low: OCR {ocr:.2f} < {th['ocr_auto_approve_min']}")
    else:
        add("Gate 1", "Text quality (OCR)", True, f"OCR {ocr:.2f} ≥ {th['ocr_auto_approve_min']}")

    # Gate 2 — classification confidence
    c_ok = classification.class_confidence >= th["classification_confidence_min"]
    add("Gate 2", "Classification confidence", c_ok,
        f"{classification.class_confidence:.2f} {'≥' if c_ok else '<'} {th['classification_confidence_min']}")

    # Gate 3 (Conflict margin) — RETIRED. Eval proved it is redundant with Gate 2: confidence
    # and margin are coupled (a tight margin forces the top score down), so Gate 3 never fires
    # independently of Gate 2. Removing it changes zero routing decisions. The margin is still
    # COMPUTED (classification_conflict_margin) and shown for audit — we just no longer gate on it.
    # IDs 4-8 kept as-is (no renumber) to preserve audit-trail continuity.

    # Gate 4 — adverse-event policy (any AE -> PV, regardless of the other gates)
    ae = classification.adverse_event_flag
    add("Gate 4", "Adverse-event policy", not ae,
        "AE flag set -> PV (never auto-approve)" if ae else "no adverse event")

    oos = classification.out_of_scope

    # Gate 5 — catalog match
    add("Gate 5", "Catalog match", catalog_id is not None,
        catalog_id if catalog_id else
        ("not checked — document is out of scope" if oos else f"'{common.product_mentioned}' not in active catalog"))

    # Gate 6 — HCP link
    add("Gate 6", "HCP link", hcp_id is not None,
        hcp_id if hcp_id else
        ("not checked — document is out of scope" if oos else f"'{common.reporter_name}' not in HCP registry"))

    # Gate 7 — schema validation (Pydantic-enforced at extraction; valid if we reached here)
    add("Gate 7", "Schema validation", True, "Pydantic-valid at extraction")

    # Gate 8 — in scope
    add("Gate 8", "In scope", not classification.out_of_scope,
        "out of scope (no class matched)" if classification.out_of_scope else "in scope")

    # --- terminal decision ---
    # failed_gates ALWAYS lists EVERY gate that did not pass — so the reviewer (PV or MIS)
    # sees exactly where to focus. On the PV path this matters: Gate 4 forces the route, but
    # other failures (low OCR, no catalog match, consumer reporter, etc.) still inform review.
    failed = [f"{g['id']}: {g['name']} — {g['detail']}" for g in gates if not g["passed"]]
    # Gate 4 is a POLICY OVERRIDE: any adverse event -> PV, regardless of other gates.
    if ae:
        target, status = "PV Queue", "Pending Review"
    elif classification.predicted_class is DocClass.OFF_LABEL:
        # Off-Label POLICY: off-label inquiries are compliance-sensitive -> NEVER auto-approve,
        # always human review (off-label information may only be shared via approved channels).
        failed = failed + ["Off-Label policy: off-label inquiries are never auto-approved (compliance) -> MIS"]
        target, status = "MIS Queue", "Pending Review"
    elif failed:
        target, status = "MIS Queue", "Pending Review"
    else:
        target, status = "Auto-Approve", "Approved"

    return {
        "routing_target": target,
        "record_status": status,
        "catalog_match_result": catalog_id,
        "hcp_system_id": hcp_id,
        "failed_gates": failed,
        "gates": gates,
        "day_zero": day_zero.isoformat() if day_zero else None,
        "regulatory_due_date": due_date.isoformat() if due_date else None,
        "reporting_regime": regime,
    }
