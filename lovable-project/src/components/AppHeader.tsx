import { Link, useRouterState } from "@tanstack/react-router";

export function AppHeader({ user }: { user?: { id: string; role: string } }) {
  const u = user ?? { id: "MIS_REVIEWER_04", role: "Medical Information" };
  const pathname = useRouterState({ select: (s) => s.location.pathname });
  const navItem = (label: string, active: boolean) =>
    `h-14 flex items-center transition-colors ${
      active
        ? "text-reg-accent border-b-2 border-reg-accent"
        : "text-slate-500 hover:text-reg-blue"
    }`;

  return (
    <header className="h-14 border-b border-reg-border bg-white flex items-center justify-between px-6 sticky top-0 z-50">
      <div className="flex items-center gap-8">
        <Link to="/" className="flex items-center gap-2">
          <div className="size-6 bg-reg-accent rounded flex items-center justify-center text-white font-bold text-xs">
            V
          </div>
          <span className="font-bold tracking-tight text-sm">VIGILANT.AI</span>
        </Link>
        <nav className="flex gap-6 text-xs font-medium">
          <Link to="/" className={navItem("INTAKE QUEUE", pathname === "/" || pathname.startsWith("/review"))}>
            INTAKE QUEUE
          </Link>
        </nav>
      </div>
      <div className="flex items-center gap-4 text-xs">
        <div className="flex items-center gap-2 px-3 py-1 bg-slate-100 rounded-full font-medium text-slate-600">
          <span className="size-2 bg-reg-success rounded-full" />
          {u.id} <span className="text-slate-400">· {u.role}</span>
        </div>
      </div>
    </header>
  );
}
