"""
prompt_store.py — persistent, versioned prompt snapshots (the FULL TEXT of each version).

Principle: a "version" only exists if we kept its actual text — otherwise you can't
compare it. This store freezes a full snapshot every time a prompt changes, so the
Prompt Library only ever shows versions we can truly diff. A prompt with one stored
snapshot is simply "v1 (current)"; we never show a version number we can't reproduce.

When you change a prompt, call this so the change is tracked + comparable forever:

    from prompt_store import save_version
    save_version("ae_extract", NEW_TEXT,
                 change="Symptoms → verbatim", why="faithfulness grader caught invented terms")

It appends a new auto-incremented version ONLY if the text actually changed.
"""
import json
from datetime import date
from pathlib import Path

STORE = Path(__file__).parent / "prompt_store.json"


def load() -> dict:
    return json.loads(STORE.read_text()) if STORE.exists() else {}


def versions(prompt_id: str) -> list:
    """All saved versions of a prompt, oldest → newest. [] if none stored."""
    return load().get(prompt_id, [])


def save_version(prompt_id: str, text: str, change: str, why: str, when: str = None) -> int:
    """Freeze a full snapshot of `text` as the next version — only if it changed."""
    store = load()
    vs = store.setdefault(prompt_id, [])
    if vs and vs[-1]["text"] == text:
        return vs[-1]["version"]                      # unchanged → no new version
    v = (vs[-1]["version"] + 1) if vs else 1
    vs.append({"version": v, "date": when or date.today().isoformat(),
               "change": change, "why": why, "text": text})
    STORE.write_text(json.dumps(store, indent=2))
    return v


def _live_prompts() -> dict:
    """The current text of every tracked prompt, pulled live from where it's defined."""
    import prompts as P
    from graders import FAITHFULNESS_SYSTEM, SYMPTOM_COVERAGE_SYSTEM, SUMMARY_ACCURACY_SYSTEM
    return {
        "classify": P.CLASSIFY_SYSTEM_PROMPT,
        "extract_metadata": P.EXTRACT_METADATA_SYSTEM_PROMPT,
        "onlabel": P.ONLABEL_EXTRACT_SYSTEM_PROMPT,
        "offlabel": P.OFFLABEL_EXTRACT_SYSTEM_PROMPT,
        "ae_extract": P.AE_EXTRACT_SYSTEM_PROMPT,
        "seriousness": P.AE_SERIOUSNESS_SYSTEM_PROMPT,
        "grader_faithfulness": FAITHFULNESS_SYSTEM,
        "grader_symptom": SYMPTOM_COVERAGE_SYSTEM,
        "grader_summary": SUMMARY_ACCURACY_SYSTEM,
    }


def sync() -> list:
    """SAFETY NET — auto-capture any live prompt whose text changed since its last version.

    Called on app startup so a prompt edit is NEVER lost, even if you edit prompts.py
    directly and forget to log it. Idempotent: no change → no write. The auto-captured
    note is a placeholder you (or Claude) can replace with the real 'what/why' later.
    """
    captured = []
    for pid, text in _live_prompts().items():
        vs = versions(pid)
        if not vs or vs[-1]["text"] != text:
            v = save_version(pid, text, change="(auto-captured on change — annotate the reason)", why="—")
            captured.append((pid, v))
    return captured


def seed():
    """Populate the store from the live prompts (run once). Idempotent on text.
    Only the AE extractor has TWO real versions kept (the clinical→verbatim A/B);
    every other prompt has just its current text, so it shows as a single version."""
    import prompts as P
    from graders import FAITHFULNESS_SYSTEM, SYMPTOM_COVERAGE_SYSTEM, SUMMARY_ACCURACY_SYSTEM

    single = [
        ("classify", P.CLASSIFY_SYSTEM_PROMPT, "2026-06-06"),
        ("extract_metadata", P.EXTRACT_METADATA_SYSTEM_PROMPT, "2026-06-06"),
        ("onlabel", P.ONLABEL_EXTRACT_SYSTEM_PROMPT, "2026-06-06"),
        ("offlabel", P.OFFLABEL_EXTRACT_SYSTEM_PROMPT, "2026-06-04"),
        ("seriousness", P.AE_SERIOUSNESS_SYSTEM_PROMPT, "2026-06-06"),
        ("grader_faithfulness", FAITHFULNESS_SYSTEM, "2026-06-04"),
        ("grader_symptom", SYMPTOM_COVERAGE_SYSTEM, "2026-06-04"),
        ("grader_summary", SUMMARY_ACCURACY_SYSTEM, "2026-06-04"),
    ]
    for pid, text, when in single:
        save_version(pid, text, "Current version", "—", when)

    # The one prompt we kept TWO real versions of → a genuine, diff-able history.
    save_version("ae_extract", P.AE_EXTRACT_PROMPTS["extract_v2_clinical"],
                 "Baseline — symptoms in 'plain clinical phrasing'", "first version", "2026-06-04")
    save_version("ae_extract", P.AE_EXTRACT_PROMPTS["extract_v7"],
                 "Symptoms → VERBATIM (patient's own words)",
                 "faithfulness grader caught the model inventing clinical terms (50% → 100%)", "2026-06-09")


if __name__ == "__main__":
    if not load():
        seed()                       # first run → seed from current prompts (+ the AE A/B)
    else:
        got = sync()                 # later runs → capture any edits
        print("captured:", got or "nothing new")
    for pid, vs in load().items():
        print(f"{pid:22} {[ 'v'+str(v['version']) for v in vs ]}")
