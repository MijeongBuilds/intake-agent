"""
prompt_registry.py — the SINGLE place that inventories every prompt the agent uses.

The prompts themselves live next to their code (prompts.py, graders.py); this module
imports the LIVE text (never duplicates it) and attaches version + purpose + change
history, so the UI can show & track ALL prompt changes in one view — regardless of which
file the prompt lives in.

Two version stamps are written into every record's 21 CFR Part 11 audit trail:
  • classify_v3  — the classifier
  • extract_v7   — all extractors (shared stamp)
Graders are eval-only (not stamped on records). History below is sourced from PROMPT_CHANGELOG.md.
"""
from prompts import (
    CLASSIFY_SYSTEM_PROMPT, EXTRACT_METADATA_SYSTEM_PROMPT, ONLABEL_EXTRACT_SYSTEM_PROMPT,
    OFFLABEL_EXTRACT_SYSTEM_PROMPT, AE_EXTRACT_SYSTEM_PROMPT, AE_SERIOUSNESS_SYSTEM_PROMPT,
    AE_EXTRACT_PROMPTS, AE_EXTRACT_VERSION_INFO, PROMPT_VERSION, EXTRACT_PROMPT_VERSION,
    METADATA_PROMPTS, METADATA_VERSION_INFO, ONLABEL_PROMPTS, ONLABEL_VERSION_INFO,
    CLASSIFY_PROMPTS, CLASSIFY_VERSION_INFO,
)
from graders import FAITHFULNESS_SYSTEM, SYMPTOM_COVERAGE_SYSTEM, SUMMARY_ACCURACY_SYSTEM

# Each entry: live text + the audit-stamp version it belongs to + plain-English purpose +
# dated change history (what changed, why) + optional runnable variants (for the sweep).
PROMPT_REGISTRY = [
    {
        "id": "classify", "name": "Classifier", "stage": "1 · Classify", "version": PROMPT_VERSION,
        "text": CLASSIFY_SYSTEM_PROMPT,
        "purpose": "Score the document against all classes; detect adverse events; assess scope + the 4 ICSR pillars.",
        "history": [
            {"v": "v1", "date": "2026-06-02", "change": "Baseline: per-class scoring + AE-detection emphasis", "why": "catch the hidden-AE trap instead of mis-routing"},
            {"v": "v2", "date": "2026-06-06", "change": "Added classification_rationale (one-line why)", "why": "a reviewer couldn't tell WHY confidence was low"},
            {"v": "v3", "date": "2026-06-06", "change": "Per-class scoring; code derives confidence & margin", "why": "self-reported confidence was noisy / un-calibratable"},
            {"v": "v4", "date": "2026-06-10", "change": "Given the company's product portfolio; flags inquiries solely about another company's drug as out-of-scope", "why": "a competitor-only inquiry was read as a normal On-Label inquiry (case_051) — the classifier didn't know which products are ours"},
        ],
        "variants": {"current": "classify_v4", "options": CLASSIFY_PROMPTS, "info": CLASSIFY_VERSION_INFO},
    },
    {
        "id": "extract_metadata", "name": "Common-metadata extractor", "stage": "2 · Extract", "version": EXTRACT_PROMPT_VERSION,
        "text": EXTRACT_METADATA_SYSTEM_PROMPT,
        "purpose": "Pull reporter / contact / source metadata; strip honorifics so the HCP-registry match works.",
        "history": [
            {"v": "v1", "date": "2026-06-02", "change": "Baseline: reporter / contact / source-channel extraction", "why": "— first version —"},
            {"v": "v2", "date": "2026-06-06", "change": "reporter_name strips honorifics/credentials", "why": "'Dr. Samuel Brennan' broke the Gate 6 HCP match (registry holds 'Samuel Brennan')"},
            {"v": "v3", "date": "2026-06-10", "change": "GROUNDING contract: country only from an explicit place; no inference from weak signals", "why": "country was inferred from a credential + email TLD with no address stated (case_027) — now aligned with the faithfulness grader"},
        ],
        "variants": {"current": "metadata_v3", "options": METADATA_PROMPTS, "info": METADATA_VERSION_INFO},
    },
    {
        "id": "onlabel", "name": "On-Label record extractor", "stage": "2 · Extract", "version": EXTRACT_PROMPT_VERSION,
        "text": ONLABEL_EXTRACT_SYSTEM_PROMPT,
        "purpose": "Build the On-Label inquiry record; suggest an SRD only if it matches the product.",
        "history": [
            {"v": "v1", "date": "2026-06-02", "change": "Baseline: On-Label inquiry record + SRD suggestion", "why": "— first version —"},
            {"v": "v2", "date": "2026-06-06", "change": "SRD suggestion must match the PRODUCT, not just the topic", "why": "suggested our CholoClear-X SRD for a competitor product (case_015)"},
            {"v": "v3", "date": "2026-06-10", "change": "GROUNDING contract: summary may use only entities named in the doc — no inserting an unnamed product", "why": "the model injected 'CholoClear-X' into the summary of a competitor-product inquiry (case_015) — now aligned with the faithfulness grader"},
        ],
        "variants": {"current": "onlabel_v3", "options": ONLABEL_PROMPTS, "info": ONLABEL_VERSION_INFO},
    },
    {
        "id": "offlabel", "name": "Off-Label record extractor", "stage": "2 · Extract", "version": EXTRACT_PROMPT_VERSION,
        "text": OFFLABEL_EXTRACT_SYSTEM_PROMPT,
        "purpose": "Build the Off-Label inquiry record (unapproved dose / age / indication / route).",
        "history": [
            {"v": "v1", "date": "2026-06-04", "change": "Baseline: Off-Label inquiry record extractor", "why": "— first version —"},
        ],
        "variants": None,
    },
    {
        "id": "ae_extract", "name": "Adverse-event (ICSR) extractor", "stage": "2 · Extract", "version": EXTRACT_PROMPT_VERSION,
        "text": AE_EXTRACT_SYSTEM_PROMPT,
        "purpose": "Draft the ICSR: patient initials, verbatim symptoms, onset date, seriousness.",
        "history": [
            {"v": "v1", "date": "2026-06-02", "change": "Baseline: draft ICSR (initials, symptoms, onset, seriousness)", "why": "— first version —"},
            {"v": "v2", "date": "2026-06-03", "change": "Seriousness: vague catch-all → canonical ICH E2A bar + rationale", "why": "verdict was an unstable Serious/Non-Serious coin-flip"},
            {"v": "v3", "date": "2026-06-04", "change": "Symptoms: 'plain clinical phrasing' → VERBATIM (patient's words)", "why": "the faithfulness grader caught the model inventing clinical terms — faithfulness 50% → 100%"},
            {"v": "v4", "date": "2026-06-06", "change": "onset_date firmed to null on relative refs", "why": "model invented a calendar date from 'this morning'"},
            {"v": "v5", "date": "2026-06-10", "change": "Symptoms: added 'capture EVERY symptom — a missed one is the worst error'", "why": "make the prompt's contract match the symptom-coverage grader (recall is the AE floor)"},
        ],
        # runnable A/B for the Compare/Sweep — selecting the old version re-runs the golden set
        "variants": {"current": EXTRACT_PROMPT_VERSION, "options": AE_EXTRACT_PROMPTS, "info": AE_EXTRACT_VERSION_INFO},
    },
    {
        "id": "seriousness", "name": "Seriousness self-consistency judge", "stage": "2 · Extract", "version": EXTRACT_PROMPT_VERSION,
        "text": AE_SERIOUSNESS_SYSTEM_PROMPT,
        "purpose": "Judge Serious vs Non-Serious under ICH E2A — best-of-3 vote (drives the 15- vs 90-day clock).",
        "history": [
            {"v": "v1", "date": "2026-06-03", "change": "Baseline: single-shot ICH E2A seriousness verdict", "why": "— first version —"},
            {"v": "v2", "date": "2026-06-06", "change": "Single-shot → best-of-3 self-consistency vote", "why": "a borderline cardiac case drifted between runs; vote stabilizes it"},
        ],
        "variants": None,
    },
    {
        "id": "grader_faithfulness", "name": "Grader · Faithfulness", "stage": "3 · Eval (LLM-judge)", "version": "grader_v1",
        "text": FAITHFULNESS_SYSTEM,
        "purpose": "Verify every extracted value is grounded in the document — no hallucination, no contradiction.",
        "history": [], "variants": None,
    },
    {
        "id": "grader_symptom", "name": "Grader · Symptom coverage", "stage": "3 · Eval (LLM-judge)", "version": "grader_v1",
        "text": SYMPTOM_COVERAGE_SYSTEM,
        "purpose": "Verify the symptom list captures every symptom (recall) and invents none (AE safety).",
        "history": [], "variants": None,
    },
    {
        "id": "grader_summary", "name": "Grader · Summary accuracy", "stage": "3 · Eval (LLM-judge)", "version": "grader_v1",
        "text": SUMMARY_ACCURACY_SYSTEM,
        "purpose": "Rate 1–5 how faithfully the inquiry summary captures the actual question.",
        "history": [], "variants": None,
    },
]
