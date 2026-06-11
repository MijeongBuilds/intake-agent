# Document Intake Agent — Regulated Life Sciences

An end-to-end AI agent that automates document intake for a regulated document-management
workflow: it **classifies** each inbound document, **extracts** its metadata and a
transactional record, and **routes** it to the right human queue — auto-approving only
clean cases and **always escalating adverse events to safety review**.

Built as a self-directed product + engineering case study around a **Medical Information /
Pharmacovigilance** intake workflow — the highest-regulated bar, which generalizes back to
any document class (orders, contracts, complaints).

> **▶ Live demo:** https://review-whisper-fast.lovable.app
> *(the human-in-the-loop reviewer UI, running on the real agent's output)*

---

## The problem

Specialists hand-triage every inbound document — reading dense text to find the core
request, keying metadata, and deciding whether an adverse event is hidden inside —
**15–40 min per case, ~6.5% manual error, under a 15-day legal reporting clock.** A missed
adverse event isn't a UX bug; it's a regulatory and patient-safety failure.

## What's here

| Deliverable | Where |
|---|---|
| **A representative dataset + eval method** — a 39-case golden set (input + ground truth + criteria) across all classes, edge cases, and adversarial inputs | [`agent/golden_dataset/`](agent/golden_dataset) |
| **The agent** — classify → extract → 7-gate compliance unit → route, with grounded extraction | [`agent/`](agent) |
| **An eval console** — scorecard, model bake-off, prompt library, per-case pipeline trace | [`agent/app.py`](agent/app.py) |
| **A human-in-the-loop reviewer UI** — exception queue + split-screen verifier | [`lovable-project/`](lovable-project) |

## Architecture — *“AI proposes, code disposes”*

```
Intake → Readability / OCR → Classify (LLM) → Extract draft (LLM)
       → 7-gate compliance unit (deterministic) → Route
                                                    ├─ Auto-Approve
                                                    ├─ Medical-Info queue
                                                    ├─ Pharmacovigilance queue
                                                    └─ Unreadable
```

- **The LLM does the judgment** (classify, extract). **Deterministic code does all routing,
  validation, catalog/registry lookups, and the compliance gates** — never the model.
- **Grounded extraction:** every value is cited to a verbatim source span, with a code-level
  guard that *drops any quote not actually in the source* — so a hallucinated highlight can
  never reach the reviewer.
- **Multi-tenant by config:** taxonomy, metadata schema, and catalogs are read **per tenant at
  runtime**, so one agent serves any customer or document class without code changes.

## Evaluation (the rigor)

Real results from the 39-case golden set, each config run **×3** for mean ± variance:

| Metric | Result |
|---|---|
| **Adverse-event recall** | **100%** — held across all 12 runs (every model, every run); the safety floor that never moves |
| **Auto-approve precision** | **100%** — the agent never auto-commits a record it shouldn't |
| **Extraction faithfulness** | **100%** (LLM-judge grader) |
| **Classification accuracy** | 91% (Claude Sonnet) |

**4-model bake-off** (Claude Sonnet/Haiku, GPT-4o/4o-mini, same eval): the OpenAI models had
strong raw classification but **failed auto-approve precision** — confidently auto-committing
*wrong* records (accurate-but-unsafe). Accuracy converged; the safety gates separated them.

**Honest limits:** this is correctness on a deliberately adversarial coverage set, **not a
production rate**. Production calibrates confidence (reliability curve), sets the auto-approve
threshold from a risk-coverage curve, and scales to hundreds of cases per class on a held-out split.

## Key decisions & trade-offs

- **No RAG for reading the document** — top-K retrieval can rank a buried adverse event below
  the cutoff and miss it; for compliance we read **100% of the text**. *(RAG's right home here
  is matching to a Standard-Response library — not reading the intake doc.)*
- **Reject option** — when no class clears the confidence bar, the agent doesn't guess; it
  surfaces the top class + confidence + *why* → human.
- **Progressive autonomy** — launch human-supervised; earn per-class auto-commit only after the
  eval proves ≥99% precision.
- **21 CFR Part 11** — every commit is gated by a re-authenticated e-signature into an immutable
  audit trail; the human is the authorizer of record.

## Run it

**The agent + eval console** (Python):
```bash
cd agent
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
cp .env.example .env          # add your ANTHROPIC_API_KEY
.venv/bin/streamlit run app.py
```

**The reviewer UI** (React / TanStack):
```bash
cd lovable-project
npm install && npm run dev
```

**Rebuild the UI data from the agent** (incremental — only new/changed cases call the model):
```bash
cd agent && .venv/bin/python build_reviewer_data.py
```

## Repository

```
agent/
  agent_engine.py     classify · extract · 7-gate compliance unit
  evaluate.py         golden-set scorecard + per-case eval
  grounding.py        cite the source, code-guard the evidence
  schemas.py          Pydantic contracts (structured output)
  prompts.py          versioned prompts
  app.py              the eval console (Streamlit)
  build_reviewer_data.py   Python → reviewer-UI bridge (cached)
  golden_dataset/     the 39-case golden set (input + ground truth)
  tenants/            per-tenant config (taxonomy, catalog, registries)
lovable-project/      the human-in-the-loop reviewer UI
```

## Tech stack

Python · LangChain (`ChatAnthropic` / `ChatOpenAI`) · Pydantic structured output ·
LLM-judge evals & golden datasets · Streamlit · React / TanStack Start

## Roadmap

Production eval scale-up (300+ AE cases for recall confidence) · MedDRA symptom coding ·
fuzzy entity resolution (alias + vector matching) · multilingual extraction · hardening
non-prose extraction (forms, call-center transcripts).

---

*Self-directed product + engineering case study. Synthetic data throughout; no real patient
or proprietary information.*
