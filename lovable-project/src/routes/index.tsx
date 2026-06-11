import { createFileRoute, Link } from "@tanstack/react-router";
import { useState } from "react";
import { AppHeader } from "@/components/AppHeader";
import { CASES, type Case, type Queue, type Status } from "@/lib/cases";

export const Route = createFileRoute("/")({
  head: () => ({
    meta: [
      { title: "Intake Queue — Vigilant.AI" },
      { name: "description", content: "Triage AI-classified medical information and pharmacovigilance cases." },
    ],
  }),
  component: Worklist,
});

const QUEUE_TABS: { key: Queue; label: string }[] = [
  { key: "MIS", label: "MIS Queue" },
  { key: "PV", label: "PV Queue" },
];

function urgencyClasses(u: "critical" | "warning" | "normal") {
  if (u === "critical") return "bg-red-50 text-reg-danger";
  if (u === "warning") return "bg-amber-50 text-reg-warning";
  return "bg-slate-100 text-slate-600";
}

function statusBadge(status: Status) {
  const base = "inline-flex items-center gap-1.5 px-2 py-0.5 rounded text-[10px] font-bold border";
  if (status === "READY TO APPROVE") return `${base} bg-green-50 text-reg-success border-green-100`;
  if (status === "AWAITING INFO") return `${base} bg-blue-50 text-reg-accent border-blue-100`;
  return `${base} bg-amber-50 text-reg-warning border-amber-100`;
}

function statusHint(status: Status) {
  if (status === "READY TO APPROVE") return "AI is highly confident in the classification — just needs a final human sign-off.";
  if (status === "AWAITING INFO") return "Blocked on missing input (e.g. an unreadable scan needs a rescan).";
  return "The AI flagged something to verify before this can be approved.";
}

function Worklist() {
  const [queue, setQueue] = useState<Queue>("MIS");
  const [status, setStatus] = useState<Status | "ALL">("ALL");
  const [oldestFirst, setOldestFirst] = useState(true);

  const counts = { MIS: 0, PV: 0 } as Record<Queue, number>;
  CASES.forEach((c) => (counts[c.queue] += 1));

  const filters: { key: Status | "ALL"; label: string }[] =
    queue === "PV"
      ? [{ key: "ALL", label: "All" }, { key: "NEEDS REVIEW", label: "Needs Review" }]
      : [
          { key: "ALL", label: "All" },
          { key: "NEEDS REVIEW", label: "Needs Review" },
          { key: "READY TO APPROVE", label: "Ready to Approve" },
        ];

  const cases = CASES.filter((c) => c.queue === queue)
    .filter((c) => status === "ALL" || c.status === status)
    .sort((a, b) => {
      if (queue === "PV") {
        const order = { critical: 0, warning: 1, normal: 2 } as const;
        const d = order[a.dueUrgency] - order[b.dueUrgency];
        return oldestFirst ? d : -d;
      }
      return oldestFirst ? b.ageDays - a.ageDays : a.ageDays - b.ageDays;
    });

  const readyInQueue = CASES.filter((c) => c.queue === queue && c.status === "READY TO APPROVE").length;
  const timingHeader = queue === "PV" ? "Deadline" : "Received";

  return (
    <div className="min-h-screen bg-reg-slate text-reg-blue">
      <AppHeader user={queue === "PV"
        ? { id: "PV_REVIEWER_02", role: "Pharmacovigilance" }
        : { id: "MIS_REVIEWER_04", role: "Medical Information" }} />
      <main className="p-6 max-w-[1600px] mx-auto">
        <section className="mb-10">
          <div className="flex items-end justify-between mb-5">
            <div>
              <h1 className="text-2xl font-bold tracking-tight">Intake Queue</h1>
              <p className="text-slate-500 text-sm">
                {cases.length} case{cases.length === 1 ? "" : "s"}
                {queue === "MIS" && readyInQueue > 0 && (
                  <> · <span className="text-reg-success font-semibold">{readyInQueue} greenlit by the AI</span></>
                )}
              </p>
            </div>
            <div className="flex bg-white border border-reg-border rounded-md p-0.5">
              {QUEUE_TABS.map((t) => (
                <button
                  key={t.key}
                  onClick={() => { setQueue(t.key); setStatus("ALL"); }}
                  className={`px-4 py-1.5 text-xs font-semibold rounded transition-colors ${
                    queue === t.key ? "bg-reg-blue text-white" : "text-slate-500 hover:text-reg-blue"
                  }`}
                >
                  {t.label} <span className="opacity-60">{counts[t.key]}</span>
                </button>
              ))}
            </div>
          </div>

          <div className="flex gap-2 mb-4">
            {filters.map((s) => (
              <button
                key={s.key}
                onClick={() => setStatus(s.key)}
                className={`px-3 py-1 text-[11px] font-semibold rounded-full border transition-colors ${
                  status === s.key ? "bg-reg-blue text-white border-reg-blue" : "bg-white text-slate-500 border-reg-border hover:text-reg-blue"
                }`}
              >
                {s.label}
              </button>
            ))}
          </div>

          <div className="bg-white border border-reg-border rounded-lg shadow-sm overflow-hidden">
            <table className="w-full text-left border-collapse">
              <thead>
                <tr className="bg-slate-50 border-b border-reg-border text-[11px] font-bold text-slate-400 uppercase tracking-wider">
                  <th className="px-6 py-4 w-40">Case ID</th>
                  <th className="px-4 py-4">Source</th>
                  <th className="px-4 py-4">Product</th>
                  <th className="px-4 py-4">
                    <button onClick={() => setOldestFirst((v) => !v)} className="flex items-center gap-1 uppercase tracking-wider hover:text-reg-blue">
                      {timingHeader} <span className="text-[9px]">{oldestFirst ? "▲" : "▼"}</span>
                    </button>
                  </th>
                  <th className="px-4 py-4">Flag Reason</th>
                  <th className="px-4 py-4">Status</th>
                  <th className="px-6 py-4 text-right">Action</th>
                </tr>
              </thead>
              <tbody className="text-sm">
                {cases.map((c) => (
                  <tr key={c.id} className="border-b border-reg-border last:border-b-0 hover:bg-slate-50/50 transition-colors">
                    <td className="px-6 py-4 font-mono font-medium text-reg-accent">#{c.id}</td>
                    <td className="px-4 py-4 text-slate-600">{c.source}</td>
                    <td className="px-4 py-4 font-semibold">{c.product}</td>
                    <td className="px-4 py-4">
                      {queue === "PV" && c.dueLabel ? (
                        <div className="flex flex-col gap-0.5">
                          <span className={`px-2 py-0.5 font-bold text-[10px] rounded font-mono w-fit ${urgencyClasses(c.dueUrgency)}`}>{c.dueLabel}</span>
                          <span className="text-slate-400 text-[11px]">due {c.dueDate}</span>
                        </div>
                      ) : (
                        <span className="text-slate-600 text-[13px]">{c.receivedDate}</span>
                      )}
                    </td>
                    <td className="px-4 py-4">
                      <div className="flex flex-col">
                        <span className={`font-semibold ${c.status === "READY TO APPROVE" ? "text-reg-success" : c.dueUrgency === "critical" ? "text-reg-danger" : "text-reg-blue"}`}>{c.flagTitle}</span>
                        {c.flagDetail && <span className="text-[11px] text-slate-400">{c.flagDetail}</span>}
                      </div>
                    </td>
                    <td className="px-4 py-4">
                      <span className={statusBadge(c.status)} title={statusHint(c.status)}>{c.status}</span>
                    </td>
                    <td className="px-6 py-4 text-right">
                      <Link to="/review/$caseId" params={{ caseId: c.id }} className="text-reg-accent font-bold text-xs hover:underline">
                        {c.status === "READY TO APPROVE" ? "Confirm →" : "Review →"}
                      </Link>
                    </td>
                  </tr>
                ))}
                {cases.length === 0 && (
                  <tr><td colSpan={7} className="px-6 py-10 text-center text-slate-400 text-sm">No cases in this view.</td></tr>
                )}
              </tbody>
            </table>
          </div>
          <p className="text-[11px] text-slate-400 mt-2">
            Hover a status to see what it means · <span className="text-reg-success font-semibold">Ready to Approve</span> = AI is confident, just needs your sign-off.
          </p>
        </section>
      </main>
    </div>
  );
}
