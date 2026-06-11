"""
grounding.py — cite the source for each extracted/classified value.

Substring-matching only finds VERBATIM values (product, name, phone). Interpreted
fields — Reporter Type "HCP - Pharmacist", a class, an enum like "Dosage and
Administration" — are the model's *judgment*, never a quote, so they can't be
located that way. This step asks the LLM to return, per field:

  - evidence : the exact verbatim source span the value is based on (or "" if none)
  - reason   : a one-line why

We VALIDATE the quote is a real substring of the source and drop it otherwise, so a
hallucinated highlight can never reach the reviewer. Decoupled from the eval pipeline.
"""
from __future__ import annotations

from langchain_anthropic import ChatAnthropic
from pydantic import BaseModel, Field

from agent_engine import MODEL  # importing also loads .env (agent_engine does it at import)


class FieldEvidence(BaseModel):
    field: str = Field(description="the field label, echoed back exactly")
    evidence: str = Field(description="SHORTEST exact verbatim substring from the source that "
                                      "justifies the value, copied character-for-character; "
                                      "empty string if the source contains no support")
    reason: str = Field(description="one short clause explaining why the value follows from the source")


class GroundingResult(BaseModel):
    items: list[FieldEvidence]


SYSTEM = (
    "You ground a system's extracted/classified field values against the SOURCE document a "
    "reviewer is verifying. For EACH field you are given its value. Return:\n"
    "  • evidence: the SHORTEST exact, verbatim substring from the SOURCE that supports the value. "
    "Copy it character-for-character (same casing, punctuation) so it can be found by a string search. "
    "Do NOT paraphrase. If the source genuinely contains nothing that supports the value, return an empty string.\n"
    "  • reason: one short clause on why the value follows from that evidence.\n"
    "Prefer a short phrase (3-12 words) over a whole sentence. Never invent text that isn't in the source."
)


def _user(source: str, pairs: list[tuple[str, str]]) -> str:
    lines = "\n".join(f'- {label}: "{value}"' for label, value in pairs)
    return f"SOURCE DOCUMENT:\n\"\"\"\n{source}\n\"\"\"\n\nFIELDS TO GROUND (label: value):\n{lines}"


def ground_fields(source: str, pairs: list[tuple[str, str]], model: str = MODEL) -> dict:
    """{field_label -> {"evidence": <verbatim span or "">, "reason": <clause>}}.

    Fail-safe: any LLM/parse error returns {} so the build never breaks — those fields
    just won't carry grounding. max_tokens is generous (evidence quotes can be long).
    """
    if not pairs:
        return {}
    llm = ChatAnthropic(model=model, temperature=0, max_tokens=4096).with_structured_output(GroundingResult)
    try:
        result = llm.invoke([("system", SYSTEM), ("user", _user(source, pairs))])
    except Exception as e:
        print(f"  [grounding] skipped ({type(e).__name__}) — fields keep substring evidence only")
        return {}
    sl = source.lower()
    labels = [lbl for lbl, _ in pairs]
    out = {}
    for it in result.items:
        raw = (it.field or "").strip()
        # the model sometimes echoes `Label: "value"` — map it back to the input label
        label = next((L for L in labels if raw == L or raw.startswith(L)), raw)
        ev = (it.evidence or "").strip()
        if ev and ev.lower() not in sl:   # hard guard: never surface a non-verbatim "quote"
            ev = ""
        reason = (it.reason or "").strip()
        if len(reason) > 160:             # keep captions tidy
            reason = reason[:157].rstrip() + "…"
        out[label] = {"evidence": ev, "reason": reason}
    return out
