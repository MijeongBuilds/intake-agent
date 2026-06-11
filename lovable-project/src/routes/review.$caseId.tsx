import { createFileRoute, Link, useNavigate } from "@tanstack/react-router";
import { useEffect, useRef, useState } from "react";
import { AppHeader } from "@/components/AppHeader";
import { CASES, RECORD_TEMPLATES, type Case, type Field, type RecordSection } from "@/lib/cases";

export const Route = createFileRoute("/review/$caseId")({
  head: ({ params }) => ({
    meta: [
      { title: `Reviewing ${params.caseId} — Vigilant.AI` },
      { name: "description", content: "Verify the AI's classification and extracted record against the source." },
    ],
  }),
  component: ReviewCase,
});

function ReviewCase() {
  const { caseId } = Route.useParams();
  const navigate = useNavigate();
  const c: Case = CASES.find((x) => x.id === caseId) ?? CASES[0];

  const allFields = [c.productField, ...c.common, ...(c.record?.fields ?? [])];
  const firstEvidence = allFields.find((f) => f.dot === "a" && f.evidence)?.evidence
    ?? allFields.find((f) => f.evidence)?.evidence ?? null;
  const [activeEvidence, setActiveEvidence] = useState<string | null>(firstEvidence);
  const [selectedClass, setSelectedClass] = useState<string>(c.predictedClass ?? "");
  const [showSignature, setShowSignature] = useState(false);
  const [committed, setCommitted] = useState(false);
  const [toast, setToast] = useState<string | null>(null);
  const flash = (msg: string) => { setToast(msg); window.setTimeout(() => setToast(null), 2600); };
  // Live field values — so editing/clearing a field updates its dot and the sign-off guard.
  const [vals, setVals] = useState<Record<string, string>>({});
  const setVal = (label: string, v: string) => setVals((p) => ({ ...p, [label]: v }));
  const liveVal = (f: Field) => (vals[f.label] ?? f.value);

  const ready = c.status === "READY TO APPROVE";
  // The record follows the (possibly overridden) classification.
  const recordView: RecordSection | null =
    selectedClass === c.predictedClass ? c.record : RECORD_TEMPLATES[selectedClass] ?? null;
  const classChanged = selectedClass !== c.predictedClass;

  // PV and MIS are separate, role-gated teams — the signer reflects the queue.
  const reviewer = c.queue === "PV"
    ? { id: "PV_REVIEWER_02", role: "Pharmacovigilance", email: "s.reyes@pharma-corp.com" }
    : { id: "MIS_REVIEWER_04", role: "Medical Information", email: "m.welby@pharma-corp.com" };

  // Soft guard: fields still unresolved when signing — empty (live), or flagged by a
  // check the reviewer can't fix by typing (catalog/registry/scan-quality).
  const isUnresolved = (f: Field) => {
    const v = liveVal(f).trim();
    const emptyNow = v === "" || v === "—";
    const checkFlag = f.dot === "a" && f.note !== "Needs a value";
    return emptyNow || checkFlag;
  };
  const unresolved = [c.productField, ...c.common, ...(recordView?.fields ?? [])]
    .filter(isUnresolved).map((f) => f.label);

  return (
    <div className="min-h-screen bg-reg-slate text-reg-blue">
      <AppHeader user={reviewer} />
      <main className="p-6 max-w-[1600px] mx-auto">
        <div className="flex items-center justify-between mb-6">
          <div className="flex items-center gap-4">
            <Link to="/" className="size-8 flex items-center justify-center rounded-full border border-reg-border hover:bg-white text-slate-500" aria-label="Back to queue">←</Link>
            <div>
              <div className="flex items-center gap-2">
                <h1 className="text-xl font-bold tracking-tight">Reviewing #{c.id}</h1>
                {c.severity === "CRITICAL" && <span className="px-2 py-0.5 bg-red-100 text-reg-danger text-[10px] font-black rounded">CRITICAL</span>}
                <span className="font-mono text-[10px] text-slate-400">Queue: {c.queue}</span>
              </div>
              <p className="text-slate-500 text-xs">Source: {c.id} · {c.source} · received {c.receivedDate} · waiting {c.ageDays}d</p>
            </div>
          </div>
          <div className="flex gap-2">
            <button onClick={() => flash("Information request sent to the reporter.")} className="px-4 py-2 border border-reg-border rounded font-semibold text-xs text-slate-600 hover:bg-white">Request Info</button>
            <button onClick={() => flash("Escalated to a senior reviewer.")} className="px-4 py-2 border border-reg-border rounded font-semibold text-xs text-slate-600 hover:bg-white">Escalate</button>
            <button onClick={() => flash("Case rejected — logged and reversible.")} className="px-4 py-2 border border-reg-border rounded font-semibold text-xs text-reg-danger hover:bg-white">Reject</button>
            {c.queue === "MIS" && (
              <button onClick={() => flash("Transferred to the PV (safety) queue.")} className="px-4 py-2 border border-reg-accent/30 text-reg-accent rounded font-semibold text-xs hover:bg-blue-50" title="Safety backstop: re-route to Pharmacovigilance if an adverse event is detected">
                Transfer to PV ↗
              </button>
            )}
            <button onClick={() => setShowSignature(true)} className="px-6 py-2 bg-reg-accent text-white rounded font-bold text-xs shadow-md shadow-blue-200 hover:bg-blue-700">
              Approve &amp; Create Record
            </button>
          </div>
        </div>

        <div className="grid grid-cols-2 gap-6 h-[760px]">
          {/* source */}
          <div className="bg-slate-200 rounded-lg overflow-hidden border border-reg-border relative">
            <div className="absolute top-4 left-4 right-4 z-10 flex justify-between">
              <div className="bg-white/90 backdrop-blur px-3 py-1.5 rounded-md text-[10px] font-bold border border-black/5">SOURCE TEXT</div>
              <div className="bg-white/90 backdrop-blur px-3 py-1.5 rounded-md text-[10px] font-bold border border-black/5">{c.source}</div>
            </div>
            <div className="w-full h-full overflow-auto p-12 pt-16">
              <div className="bg-white shadow-lg rounded-sm p-8 mx-auto max-w-xl min-h-full">
                <SourceText text={c.sourceText} highlight={activeEvidence} />
              </div>
            </div>
          </div>

          {/* draft */}
          <div className="bg-white rounded-lg border border-reg-border flex flex-col shadow-sm">
            <div className={`p-4 border-b flex gap-4 ${ready ? "bg-green-50 border-green-100" : "bg-amber-50 border-amber-100"}`}>
              <div className={`size-10 rounded-full flex items-center justify-center shrink-0 ${ready ? "bg-green-200" : "bg-amber-200"}`}>
                <span className={`font-bold italic ${ready ? "text-reg-success" : "text-amber-700"}`}>{ready ? "✓" : "!"}</span>
              </div>
              <div>
                <h4 className={`text-xs font-bold uppercase ${ready ? "text-reg-success" : "text-amber-900"}`}>{ready ? "Ready to Approve" : "Needs Your Review"}</h4>
                <p className={`text-xs leading-relaxed ${ready ? "text-green-800" : "text-amber-800"}`}>{c.banner}</p>
              </div>
            </div>

            <div className="flex-1 overflow-y-auto p-6 space-y-6">
              {/* CLASSIFICATION — the core output, editable */}
              <div className="space-y-1.5">
                <div className="flex items-center justify-between">
                  <label className="text-[10px] font-bold text-slate-400 uppercase">Classification {c.classNeedsReview && <span className="text-reg-warning">· confirm</span>}</label>
                  {c.classConfidence != null && <span className="text-[10px] font-mono text-slate-400">{c.classConfidence}% confident</span>}
                </div>
                <select value={selectedClass} onChange={(e) => setSelectedClass(e.target.value)}
                  onFocus={() => c.classEvidence && setActiveEvidence(c.classEvidence)}
                  className={`w-full px-3 py-2 border rounded text-sm font-semibold focus:outline-none focus:ring-2 focus:ring-reg-accent/20 ${c.classNeedsReview ? "bg-amber-50 border-reg-warning/40" : "bg-slate-50 border-reg-border"}`}>
                  {c.predictedClass == null && <option value="">— (no classification — document halted) —</option>}
                  {c.classOptions.map((o) => <option key={o} value={o}>{o}</option>)}
                </select>
                {c.predictedClass && (
                  <p className="text-[11px] text-slate-500 leading-relaxed">
                    AI recommends <b>{c.predictedClass}</b>{c.classConfidence != null ? ` (${c.classConfidence}% confident)` : ""}
                    {c.classRationale ? <> because {c.classRationale.charAt(0).toLowerCase() + c.classRationale.slice(1).replace(/\.$/, "")}.</> : "."}
                    {" "}You can change it — your choice is recorded.
                  </p>
                )}
                {c.classEvidence && (
                  <p className="text-[10px] text-reg-accent cursor-pointer" onClick={() => setActiveEvidence(c.classEvidence)}>📍 locate the basis in source</p>
                )}
                {classChanged && <p className="text-[11px] text-reg-warning font-medium">Class changed — the record below now shows the fields for "{selectedClass}".</p>}
              </div>

              {/* legend + how-it's-decided */}
              <div className="rounded-md bg-slate-50 border border-reg-border px-3 py-2">
                <div className="flex items-center gap-4 text-[10px] text-slate-500 mb-1">
                  <span className="flex items-center gap-1.5"><span className="size-2 rounded-full bg-reg-success" /> looks good</span>
                  <span className="flex items-center gap-1.5"><span className="size-2 rounded-full bg-reg-warning" /> needs attention</span>
                </div>
                <p className="text-[10px] text-slate-400 leading-relaxed">
                  A field is flagged <span className="text-reg-warning font-medium">amber</span> when it's empty or an automated check caught
                  something (product not in catalog · reporter not in registry · low scan quality) — it's not an AI score. Click a field with a value to find it in the source.
                </p>
              </div>

              {/* PRODUCT + COMMON METADATA */}
              <Section title="Common Metadata">
                <FieldGrid fields={[c.productField, ...c.common]} active={activeEvidence} onPick={setActiveEvidence} vals={vals} setVal={setVal} />
              </Section>

              {/* TRANSACTIONAL RECORD — follows the classification */}
              {recordView ? (
                <Section title={`Transactional Record · ${recordView.type}`}>
                  <FieldGrid fields={recordView.fields} active={activeEvidence} onPick={setActiveEvidence} vals={vals} setVal={setVal} />
                </Section>
              ) : (
                <Section title="Transactional Record">
                  <p className="text-xs text-slate-500">
                    {selectedClass && RECORD_TEMPLATES[selectedClass] === null
                      ? "This category produces no transactional record (e.g. Legal/Regulatory, Public FAQ)."
                      : "No record — the document was halted before extraction."}
                  </p>
                </Section>
              )}

              {/* narrative */}
              <div className="space-y-1.5">
                <label className="text-[10px] font-bold text-slate-400 uppercase">{c.narrativeLabel}</label>
                <textarea rows={4} defaultValue={c.narrative} className="w-full px-3 py-2 bg-slate-50 border border-reg-border rounded text-sm leading-relaxed focus:outline-none focus:ring-2 focus:ring-reg-accent/20" />
              </div>

              {/* ICSR pillars (PV only) */}
              {c.pillars && selectedClass === c.predictedClass && (
                <Section title="ICSR Regulatory Validity — 4 Pillars">
                  <div className="grid grid-cols-4 gap-2">
                    {c.pillars.map((p) => (
                      <div key={p.k} className={`p-2 rounded text-center border ${p.ok ? "bg-reg-success/5 border-reg-success/20 text-reg-success" : "bg-red-50 border-red-200 text-reg-danger"}`}>
                        <div className="text-[10px] font-bold">{p.k}</div>
                        <div className="text-xs font-medium">{p.ok ? "VALID" : "VERIFY"}</div>
                      </div>
                    ))}
                  </div>
                  <p className="mt-2 text-[11px] text-slate-500">
                    {c.isValidIcsr ? "All four pillars valid — case is submission-ready. Field-level flags don't block ICSR validity." : "Submission-ready when all four pillars are valid — outreach needed for the missing pillar."}
                  </p>
                </Section>
              )}
            </div>
          </div>
        </div>
      </main>

      {showSignature && (
        <div className="fixed inset-0 z-[60] bg-reg-blue/40 backdrop-blur-sm flex items-center justify-center p-4">
          <div className="w-full max-w-md bg-white rounded-lg shadow-2xl border border-reg-border overflow-hidden">
            <div className="p-6">
              <div className="flex items-center justify-between mb-1">
                <h3 className="text-sm font-bold">Electronic Signature Required</h3>
                <span className="text-[10px] font-mono text-slate-400">21 CFR 11.10(a)</span>
              </div>
              <p className="text-xs text-slate-500 mb-4">Re-enter your password to sign and create this record as <b>{reviewer.id}</b>. An immutable audit entry will be generated.</p>
              {unresolved.length > 0 && (
                <div className="mb-4 rounded-md bg-amber-50 border border-amber-200 px-3 py-2">
                  <p className="text-[11px] font-bold text-amber-800">⚠️ {unresolved.length} field{unresolved.length === 1 ? "" : "s"} still need attention</p>
                  <p className="text-[11px] text-amber-700 leading-snug">{unresolved.join(" · ")} — confirm you've reviewed {unresolved.length === 1 ? "it" : "them"} before signing.</p>
                </div>
              )}
              <label className="text-[10px] font-bold text-slate-400 uppercase block mb-1.5">User Identity</label>
              <div className="text-sm font-medium p-2 bg-slate-50 rounded border border-reg-border mb-3">{reviewer.email} <span className="text-slate-400">· {reviewer.role}</span></div>
              <label className="text-[10px] font-bold text-slate-400 uppercase block mb-1.5">Password</label>
              <input type="password" autoFocus placeholder="••••••••" className="w-full px-3 py-2 border border-reg-border rounded text-sm focus:outline-none focus:ring-2 focus:ring-reg-accent/20 mb-5" />
              <div className="flex gap-2">
                <button onClick={() => setShowSignature(false)} className="flex-1 px-4 py-2 text-xs font-semibold text-slate-600 border border-reg-border rounded hover:bg-slate-50">Cancel</button>
                <button onClick={() => { setShowSignature(false); setCommitted(true); setTimeout(() => navigate({ to: "/" }), 1200); }} className="flex-1 px-4 py-2 bg-reg-blue text-white font-bold text-xs rounded shadow-sm hover:bg-slate-800">SIGN &amp; CREATE RECORD</button>
              </div>
            </div>
            <div className="bg-slate-50 px-6 py-3 border-t border-reg-border"><p className="text-[10px] font-mono text-slate-400 text-center">AUDIT ID: SIG-{c.id}</p></div>
          </div>
        </div>
      )}

      {committed && (
        <div className="fixed bottom-6 right-6 z-[70] bg-reg-success text-white px-4 py-3 rounded-lg shadow-2xl text-sm font-semibold flex items-center gap-2">
          <span className="size-2 rounded-full bg-white" /> Record created for #{c.id}.
        </div>
      )}

      {toast && !committed && (
        <div className="fixed bottom-6 right-6 z-[70] bg-reg-blue text-white px-4 py-3 rounded-lg shadow-2xl text-sm font-semibold flex items-center gap-2">
          <span className="size-2 rounded-full bg-white" /> {toast}
        </div>
      )}
    </div>
  );
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="pt-4 border-t border-slate-100 first:border-t-0 first:pt-0">
      <h4 className="text-[10px] font-bold text-slate-400 uppercase mb-3">{title}</h4>
      {children}
    </div>
  );
}

function FieldGrid({ fields, active, onPick, vals, setVal }:
  { fields: Field[]; active: string | null; onPick: (v: string) => void; vals: Record<string, string>; setVal: (l: string, v: string) => void }) {
  return (
    <div className="grid grid-cols-2 gap-4">
      {fields.map((f, i) => (
        <FieldCell key={f.label + i} field={f} value={vals[f.label] ?? f.value}
          isActive={!!f.evidence && active === f.evidence} onPick={() => onPick(f.evidence)}
          onChange={(v) => setVal(f.label, v)} />
      ))}
    </div>
  );
}

function FieldCell({ field, value, isActive, onPick, onChange }:
  { field: Field; value: string; isActive: boolean; onPick: () => void; onChange: (v: string) => void }) {
  const live = (value ?? "").trim();
  const emptyNow = live === "" || live === "—";
  // amber if empty now, or flagged by a check the reviewer can't fix by typing
  const amber = emptyNow || (field.dot === "a" && field.note !== "Needs a value");
  const note = emptyNow ? "Needs a value" : field.note;
  const ring = isActive ? "ring-2 ring-reg-accent/30" : "";
  const base = amber ? "bg-amber-50 border-reg-warning/40" : "bg-slate-50 border-reg-border";
  return (
    <div className="space-y-1">
      <label className="text-[10px] font-bold text-slate-400 uppercase">{field.label}</label>
      <div className="relative">
        {field.kind === "enum" ? (
          <select value={field.options.includes(value) ? value : ""} onFocus={onPick} onClick={onPick} onChange={(e) => onChange(e.target.value)}
            className={`w-full px-3 py-2 border rounded text-sm cursor-pointer focus:outline-none focus:ring-2 focus:ring-reg-accent/20 ${base} ${ring}`}>
            <option value="">— select —</option>
            {field.options.map((o) => <option key={o} value={o}>{o}</option>)}
          </select>
        ) : (
          <input type="text" value={value} onFocus={onPick} onClick={onPick} onChange={(e) => onChange(e.target.value)}
            className={`w-full px-3 py-2 border rounded text-sm cursor-pointer focus:outline-none focus:ring-2 focus:ring-reg-accent/20 ${base} ${ring}`} />
        )}
        <div className={`absolute right-2 top-2.5 size-2 rounded-full ${amber ? "bg-reg-warning" : "bg-reg-success"}`} />
      </div>
      {amber && note && <p className="text-[10px] text-reg-warning font-medium">{note}</p>}
      {field.evidence && <p className="text-[10px] text-reg-accent cursor-pointer" onClick={onPick}>📍 click to locate in source</p>}
      {field.reason && <p className="text-[10px] text-slate-400 italic leading-snug">why: {field.reason}</p>}
    </div>
  );
}

/** Renders the real source text; highlights the active field's evidence span (real click-to-locate). */
function SourceText({ text, highlight }: { text: string; highlight: string | null }) {
  const markRef = useRef<HTMLElement>(null);
  const h = highlight && highlight !== "—" ? highlight : null;
  const idx = h ? text.toLowerCase().indexOf(h.toLowerCase()) : -1;
  useEffect(() => {
    if (idx !== -1) markRef.current?.scrollIntoView({ behavior: "smooth", block: "center" });
  }, [highlight, idx]);
  const base = "whitespace-pre-wrap font-mono text-[11px] leading-relaxed text-slate-700";
  if (!h || idx === -1) return <pre className={base}>{text}</pre>;
  return (
    <pre className={base}>
      {text.slice(0, idx)}
      <mark ref={markRef} className="bg-amber-300/60 rounded px-0.5 ring-1 ring-amber-400">{text.slice(idx, idx + h.length)}</mark>
      {text.slice(idx + h.length)}
    </pre>
  );
}
