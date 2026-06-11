"""
build_reviewer_data.py — the Python → Lovable-UI bridge.

ONE command rebuilds the reviewer UI's data from the real agent:

    python build_reviewer_data.py

Flow:
  1. Incremental cache: for every golden_dataset/case_*.json, reuse the cached
     agent result if the case input is unchanged; otherwise run the real agent
     (classify → extract → 7-gate) and update the cache. Add a new golden case
     and only THAT case runs live — everything else is served from cache.
  2. Transform the agent output into the exact shape the Lovable screens read,
     and write it into the Lovable project as src/data/reviewer_data.json.
"""
from __future__ import annotations

import hashlib
import json
import re
import typing
from datetime import date
from enum import Enum
from pathlib import Path

import evaluate as ev
import grounding
from schemas import (DocClass, OnLabelRecord, OffLabelRecord, StandardResponseRecord,
                     FormulationRecord, ExpandedAccessRecord, IITRecord, AEReportRecord, PQCRecord)

HERE = Path(__file__).parent
GOLDEN = HERE / "golden_dataset"
CACHE = GOLDEN / "_worklist_cache.json"
GROUND_CACHE = GOLDEN / "_grounding_cache.json"  # LLM evidence per case (cached; only new cases re-run)
UI_OUT = HERE.parent / "lovable-project" / "src" / "data" / "reviewer_data.json"

# Fields we never try to ground (system stamps / internal IDs have no source quote).
GROUND_SKIP = {"Received Date", "Suggested SRD Match", "Requested SRD ID", "Investigational Compound ID",
               "Treating Physician DEA Number"}

CLASS_OPTIONS = [c.value for c in DocClass]
# Every class → its transactional record model (None = no record, e.g. Legal/Reg, Public FAQ).
CLASS_RECORD = {
    DocClass.ON_LABEL: OnLabelRecord, DocClass.OFF_LABEL: OffLabelRecord,
    DocClass.STANDARD_RESPONSE: StandardResponseRecord, DocClass.FORMULATION: FormulationRecord,
    DocClass.EXPANDED_ACCESS: ExpandedAccessRecord, DocClass.IIT: IITRecord,
    DocClass.ADVERSE_EVENT: AEReportRecord, DocClass.PQC: PQCRecord,
    DocClass.LEGAL_REG: None, DocClass.PUBLIC_FAQ: None,
}


# ------------------------------------------------------------------ cache layer
def _same_input(a, b):
    return a.get("input_text") == b.get("input_text") and a.get("input_metadata") == b.get("input_metadata")


def build_cache(model: str = ev.MODEL):
    prior = {}
    if CACHE.exists():
        for item in json.loads(CACHE.read_text()):
            prior[item["case"].get("case_id")] = item
    out, ran = [], []
    for p in sorted(GOLDEN.glob("case_*.json")):
        case = json.loads(p.read_text())
        cid = case.get("case_id")
        hit = prior.get(cid)
        if hit and _same_input(hit["case"], case):
            out.append({"case": case, "res": hit["res"]})
        else:
            res = ev.run_case(case["input_text"], case["input_metadata"], model)
            out.append({"case": case, "res": res})
            ran.append(cid)
    CACHE.write_text(json.dumps(out, indent=2))
    return out, ran


# ------------------------------------------------------------------ small utils
def _humanize(key: str) -> str:
    fix = {"srd": "SRD", "icsr": "ICSR", "hcp": "HCP", "dea": "DEA", "iit": "IIT",
           "ae": "AE", "id": "ID", "pqc": "PQC"}
    return " ".join(fix.get(w, w.capitalize()) for w in key.split("_"))


def _fmt(v) -> str:
    if v is None or v == "":
        return "—"
    if isinstance(v, list):
        return "; ".join(str(x) for x in v) if v else "—"
    return str(v)


def _unwrap(ann):
    args = typing.get_args(ann)
    if args:
        for a in args:
            if a is not type(None):
                return a
    return ann


def _field_meta(fld):
    """(kind, options) for a Pydantic model field — enum/list/text."""
    core = _unwrap(fld.annotation)
    if isinstance(core, type) and issubclass(core, Enum):
        return "enum", [e.value for e in core]
    if typing.get_origin(core) is list:
        return "list", []
    return "text", []


def _due_bits(due):
    if not due:
        return None, None, None
    try:
        d = date.fromisoformat(str(due))
        days = (d - date.today()).days
        return ("OVERDUE" if days < 0 else f"{days}d left"), d.strftime("%d-%b"), \
               ("critical" if days <= 1 else "warning" if days <= 3 else "normal")
    except Exception:
        return str(due), str(due), "normal"


def _age(received):
    if not received:
        return 0, "—"
    try:
        d = date.fromisoformat(str(received))
        return (date.today() - d).days, d.strftime("%d-%b")
    except Exception:
        return 0, str(received)


def _evidence(value, source: str) -> str:
    """Source span that VERBATIM grounds this value (for click-to-locate). '' if none.

    Only matches the full value (≥4 chars) or a phone's digit run. We deliberately do
    NOT match single tokens — an interpreted/enum field like "Dosage and Administration"
    isn't a quote from the source, so highlighting a coincidental word would mislead.
    """
    if value in (None, "", "—"):
        return ""
    v = _fmt(value).strip()
    if len(v) >= 4:  # whole value appears verbatim (product, name, email, language…)
        i = source.lower().find(v.lower())
        if i >= 0:
            return source[i:i + len(v)]
    digits = re.sub(r"\D", "", v)  # phone: match the digit run however it's formatted in source
    if len(digits) >= 7:
        for m in re.finditer(r"[\d().+\-\s]{7,}", source):
            if digits[-7:] in re.sub(r"\D", "", m.group()):
                return m.group().strip()
    return ""


# ------------------------------------------------------------------ record templates
def record_templates():
    """class value → {type, fields:[{key,label,kind,options}]} (None where no record)."""
    out = {}
    for cls, model in CLASS_RECORD.items():
        if model is None:
            out[cls.value] = None
            continue
        fields = []
        for k, fld in model.model_fields.items():
            if k in ("record_type", "is_valid_icsr"):
                continue
            kind, opts = _field_meta(fld)
            fields.append({"key": k, "label": _humanize(k), "kind": kind, "options": opts})
        out[cls.value] = {"type": model.model_fields["record_type"].default, "fields": fields}
    return out


TEMPLATES = record_templates()


# ------------------------------------------------------------------ flags → plain language
PLAIN = {
    "scope": ("Looks Like Spam", "This doesn't look like a medical inquiry — confirm it's in scope or dismiss it."),
    "confidence": ("Category Unclear", "The AI wasn't sure which category this belongs to — please confirm the classification."),
    "catalog": ("Product Not Recognized", "The product wasn't found in the catalog — please verify the product name."),
    "hcp": ("Reporter Not Verified", "The reporter couldn't be matched to the HCP registry — please verify who reported this."),
    "ocr": ("Scan Quality Low", "The scan/text quality is low — please verify the extracted values against the source."),
    "offlabel": ("Off-Label Use", "This asks about use outside the approved label — it always needs a human (never auto-approved)."),
    "pillar": ("Incomplete Safety Report", "A required detail for a valid safety report is missing — outreach may be needed."),
    "adverse_event": ("Possible Adverse Event", "A possible adverse event was detected — routed to Pharmacovigilance."),
    "unbuilt": ("New Document Type", "This category doesn't have an automated record yet — please handle manually."),
    "schema": ("Could Not Structure Data", "The AI couldn't produce a valid record — please review manually."),
    "unreadable": ("Unreadable Scan", "The scan is too low-quality to read — send for rescan / manual transcription."),
}
PRIORITY = ["unreadable", "scope", "catalog", "hcp", "offlabel", "ocr", "pillar",
            "confidence", "unbuilt", "schema", "adverse_event"]


def _flags_of(res) -> set:
    if res.get("halted"):
        return {"unreadable"}
    if res.get("unbuilt_class"):
        return {"unbuilt"}
    if res.get("schema_failed"):
        return {"schema"}
    f, g = set(), res["gates"]
    gid = {gg["id"]: gg["passed"] for gg in g["gates"]}
    if gid.get("Gate 1") is False: f.add("ocr")
    if gid.get("Gate 2") is False: f.add("confidence")
    if gid.get("Gate 5") is False: f.add("catalog")
    if gid.get("Gate 6") is False: f.add("hcp")
    if gid.get("Gate 8") is False: f.add("scope")
    for s in g.get("failed_gates", []):
        if "Off-Label" in s: f.add("offlabel")
        if "Adverse-event" in s or "AE flag" in s: f.add("adverse_event")
    rec = res.get("record") or {}
    if "is_valid_icsr" in rec and not rec["is_valid_icsr"]:
        f.add("pillar")
    return f


def _top_flag(flags):
    return next((k for k in PRIORITY if k in flags), None)


# ------------------------------------------------------------------ field builder
def _field(label, value, source, needs_review=False, note="", kind="text", options=None):
    empty = value in (None, "", [])
    return {"label": label, "value": _fmt(value),
            "dot": "a" if (needs_review or empty) else "g",
            "note": note if needs_review else ("Needs a value" if empty else ""),
            "evidence": "" if kind == "enum" else _evidence(value, source),
            "reason": "",  # filled by the LLM grounding step for interpreted fields
            "kind": kind, "options": options or []}


def _common_section(common, md, flags, source):
    if not common:
        return []
    return [
        _field("Reporter Name", common.get("reporter_name"), source, "hcp" in flags, "Not in HCP registry"),
        _field("Reporter Type", common.get("reporter_type"), source),
        _field("Organization", common.get("customer_org"), source),
        _field("Contact Email", common.get("contact_email"), source),
        _field("Contact Phone", common.get("contact_phone"), source),
        _field("Country", common.get("country_code"), source),
        _field("Language", common.get("language"), source),
        _field("Received Date", md.get("received_date"), source),
    ]


def _narrative(record, cls):
    if not record:
        return ("Summary", (cls or {}).get("classification_rationale", "—"), None)
    if record.get("adverse_symptoms_list"):
        return ("Event Narrative", "; ".join(record["adverse_symptoms_list"]) + ".", "adverse_symptoms_list")
    if record.get("inquiry_summary_text"):
        return ("Inquiry Summary", record["inquiry_summary_text"], "inquiry_summary_text")
    if record.get("off_label_indication"):
        return ("Off-Label Indication", record["off_label_indication"], "off_label_indication")
    return ("Summary", (cls or {}).get("classification_rationale", "—"), None)


def _record_section(cls_value, record, narrative_key, flags, source):
    """Fill the predicted class's template with extracted values (empty if no record)."""
    tmpl = TEMPLATES.get(cls_value)
    if tmpl is None:
        return None
    ocr = "ocr" in flags
    fields = []
    for f in tmpl["fields"]:
        if f["key"] == narrative_key:
            continue
        v = (record or {}).get(f["key"])
        review = ocr or (f["key"] == "product_mentioned" and "catalog" in flags)
        cell = _field(f["label"], v, source, review, "Verify against source" if review else "",
                      kind=f["kind"], options=f["options"])
        cell["key"] = f["key"]
        fields.append(cell)
    return {"type": tmpl["type"], "fields": fields}


def _pillars(cls):
    if cls is None:
        return None
    return [{"k": "REPORTER", "ok": bool(cls.get("has_reporter_pillar"))},
            {"k": "PATIENT", "ok": bool(cls.get("has_patient_pillar"))},
            {"k": "EVENT", "ok": bool(cls.get("has_event_pillar"))},
            {"k": "DRUG", "ok": bool(cls.get("has_product_pillar"))}]


# ------------------------------------------------------------------ transform
def to_ui_case(item: dict) -> dict:
    case, res = item["case"], item["res"]
    md = case["input_metadata"]
    src = case["input_text"]
    cid = case.get("case_id", md["document_id"])
    halted = bool(res.get("halted"))
    route = res.get("routing_target") if halted else res["gates"]["routing_target"]

    flags = _flags_of(res)
    top = _top_flag(flags)
    age_days, received = _age(md.get("received_date"))

    if route == "Auto-Approve":
        queue, status = "MIS", "READY TO APPROVE"
        flag_title, flag_detail = "Ready to Approve", "AI confident — final sign-off"
        banner = "No exceptions — the AI is confident. A quick confirm and create the record."
        due = (None, None, None)
    elif halted:  # Unreadable folded into the MIS desk (intake handles rescans)
        queue, status = "MIS", "NEEDS REVIEW"
        flag_title, flag_detail, banner = PLAIN["unreadable"][0], "Needs rescan", PLAIN["unreadable"][1]
        due = (None, None, None)
    else:
        queue = "PV" if route == "PV Queue" else "MIS"
        status = "NEEDS REVIEW"
        flag_title, banner = PLAIN.get(top, ("Needs Review", "Flagged for manual authorization."))
        flag_detail = ""
        due = _due_bits(res["gates"].get("regulatory_due_date"))

    if halted:
        cls = common = record = None
        cls_value = cls_conf = cls_rationale = None
        common_rows, rec_section, pillars = [], None, None
        nlabel, ntext = "Status", "No AI draft — the document was halted before extraction."
    else:
        cls = res["classification"]
        common, record = res.get("common"), res.get("record")
        cls_value = cls["predicted_class"]
        cls_conf = round(cls["class_confidence"] * 100)
        cls_rationale = cls.get("classification_rationale") or ""
        common_rows = _common_section(common, md, flags, src)
        nlabel, ntext, nkey = _narrative(record, cls)
        rec_section = _record_section(cls_value, record, nkey, flags, src)
        pillars = _pillars(cls) if (record and "is_valid_icsr" in record) else None

    return {
        "id": cid, "source": md["source_channel"],
        "product": (common or {}).get("product_mentioned") or "—",
        "queue": queue,
        "severity": "CRITICAL" if (queue == "PV" and (record or {}).get("ae_seriousness") == "Serious")
                    else ("HIGH" if queue == "PV" else "STANDARD"),
        "status": status,
        "hasAdverseEvent": bool((cls or {}).get("adverse_event_flag")),
        "flagTitle": flag_title, "flagDetail": flag_detail, "banner": banner,
        "receivedDate": received, "ageDays": age_days,
        "dueLabel": due[0], "dueDate": due[1], "dueUrgency": due[2] or "normal",
        "predictedClass": cls_value, "classOptions": CLASS_OPTIONS,
        "classConfidence": cls_conf, "classRationale": cls_rationale,
        "classNeedsReview": bool(flags & {"confidence", "scope", "offlabel"}),
        "productField": _field("Product", (common or {}).get("product_mentioned"), src,
                               bool(flags & {"catalog", "ocr"}),
                               "Not in catalog" if "catalog" in flags else "Verify against source"),
        "common": common_rows, "record": rec_section,
        "narrativeLabel": nlabel, "narrative": ntext,
        "pillars": pillars, "isValidIcsr": (record or {}).get("is_valid_icsr"),
        "sourceText": src,
    }


def _empty_template(cls_value):
    """Blank version of a class's record (for when the reviewer switches the class)."""
    tmpl = TEMPLATES.get(cls_value)
    if tmpl is None:
        return None
    return {"type": tmpl["type"], "fields": [
        {"label": f["label"], "value": "—", "dot": "a", "note": "Enter for this class",
         "evidence": "", "reason": "", "kind": f["kind"], "options": f["options"], "key": f["key"]}
        for f in tmpl["fields"]]}


# ------------------------------------------------------------------ LLM grounding (cited evidence)
def _all_fields(uic):
    return [uic["productField"]] + uic["common"] + (uic["record"]["fields"] if uic["record"] else [])


def _pairs_to_ground(uic):
    """Interpreted fields that substring-matching couldn't ground → send to the LLM."""
    pairs = [(f["label"], f["value"]) for f in _all_fields(uic)
             if f["value"] != "—" and not f["evidence"] and f["label"] not in GROUND_SKIP]
    if uic.get("predictedClass"):
        pairs.append(("Classification", uic["predictedClass"]))
    return pairs


def _attach(uic, gmap):
    for f in _all_fields(uic):
        g = gmap.get(f["label"])
        if not g:
            continue
        if g.get("evidence"):
            f["evidence"] = g["evidence"]
        if g.get("reason"):
            f["reason"] = g["reason"]
    cg = gmap.get("Classification", {})
    uic["classEvidence"] = cg.get("evidence", "")
    uic["classReason"] = cg.get("reason", "")


def ground_all(cases, cache_items):
    """Cached LLM grounding — only new/changed (case, field-set) combos run live."""
    gcache = json.loads(GROUND_CACHE.read_text()) if GROUND_CACHE.exists() else {}
    ran = []
    for uic, item in zip(cases, cache_items):
        cid, source = uic["id"], item["case"]["input_text"]
        pairs = _pairs_to_ground(uic)
        key = hashlib.md5((source + json.dumps(pairs, sort_keys=True)).encode()).hexdigest()
        hit = gcache.get(cid)
        if hit and hit.get("key") == key:
            gmap = hit["grounding"]
        else:
            gmap = grounding.ground_fields(source, pairs) if pairs else {}
            gcache[cid] = {"key": key, "grounding": gmap}
            ran.append(cid)
        _attach(uic, gmap)
    GROUND_CACHE.write_text(json.dumps(gcache, indent=2))
    return ran


def main():
    cache, ran = build_cache()
    cases = [to_ui_case(it) for it in cache]
    gran = ground_all(cases, cache)
    ready = sum(1 for c in cases if c["status"] == "READY TO APPROVE")
    templates = {k: _empty_template(k) for k in CLASS_OPTIONS}

    payload = {"readyCount": ready, "totalCases": len(cache),
               "cases": cases, "recordTemplates": templates}
    UI_OUT.parent.mkdir(parents=True, exist_ok=True)
    UI_OUT.write_text(json.dumps(payload, indent=2))

    print(f"cache:     {len(cache)} cases ({'ran live: ' + ', '.join(ran) if ran else 'all from cache'})")
    print(f"grounding: {'ran live: ' + ', '.join(gran) if gran else 'all from cache'}")
    print(f"UI data:   {len(cases)} cases ({ready} ready-to-approve) → {UI_OUT.relative_to(HERE.parent)}")
    for c in cases:
        tag = "READY" if c["status"] == "READY TO APPROVE" else c["flagTitle"]
        print(f"  {c['queue']:5} {c['id']:16} {c['severity']:9} {tag}")


if __name__ == "__main__":
    main()
