"""Run the ×3 variance for the 4 bake-off models (current prompt).
    .venv/bin/python build_variance.py
Seeds run 1 from each model's existing single-run cache, then runs 2 more -> 8 live evals.
After this, the Model Bake-off's 'Median of 3' / 'Mean ± range' views light up.
"""
import evaluate as ev

MODELS = ["claude-sonnet-4-6", "claude-haiku-4-5-20251001", "gpt-4o", "gpt-4o-mini"]
K = 3

for model in MODELS:
    payload = ev.evaluate_variance(model, None, k=K, force=False)
    stats = ev.variance_stats(payload)

    def rng(key):
        s = stats.get(key)
        return "—" if not s else f"{s['median']*100:.0f}% (median) · {s['min']*100:.0f}–{s['max']*100:.0f}"

    print(f"DONE {model:28} runs={len(payload['runs'])}  "
          f"class={rng('classification_accuracy')}  ae_recall={rng('ae_recall')}  "
          f"faith={rng('faithfulness')}  auto_prec={rng('auto_approve_precision')}", flush=True)

print("VARIANCE DONE — open Model Bake-off and pick 'Median of 3 runs' or 'Mean ± range'.")
