import { ReactNode } from "react";
import { Link, useLocation } from "react-router-dom";
import { Logo } from "../atoms/Logo";

const NAV = [
  { path: "/workbench",      label: "workbench" },
  { path: "/catalog",        label: "catalog" },
  { path: "/llm-trainer",    label: "llm" },
  { path: "/best-alphas",    label: "best" },
  { path: "/backtest-studio",label: "validate" },
  { path: "/results-report", label: "report" },
];

/**
 * CChrome — universal frame (1280×820 default).
 * 4-row grid: 40px top + 22px status + 1fr body + 22px bottom.
 *
 * Mock: minerva-chrome.jsx CChrome komponenti.
 */
function NavLinks({ title }: { title: string }) {
  const loc = useLocation();
  return (
    <nav style={{ display: "flex", alignItems: "center", gap: 2 }}>
      {NAV.map((n) => {
        const active = loc.pathname === n.path;
        return (
          <Link
            key={n.path}
            to={n.path}
            style={{
              fontFamily: "var(--mono)",
              fontSize: 10,
              color: active ? "var(--fg-0)" : "var(--fg-3)",
              background: active ? "var(--bg-2)" : "transparent",
              padding: "3px 8px",
              borderRadius: 2,
              textDecoration: "none",
              letterSpacing: "0.02em",
              border: active ? "1px solid var(--line)" : "1px solid transparent",
              transition: "color 0.15s",
            }}
          >
            {n.label}
          </Link>
        );
      })}
    </nav>
  );
}

export function CChrome({
  title,
  sub,
  top,
  statusExtra,
  bottom,
  meta,
  width = 1280,
  height = 820,
  children,
}: {
  title: string;
  sub?: string;
  top?: ReactNode;        // top bar sağ aksiyonlar
  statusExtra?: ReactNode; // status bar ekstra
  bottom?: ReactNode;     // bottom bar override
  meta?: {
    train?: string;
    test?: string;
    split?: string;
    benchmark?: string;
    benchmarkInfo?: string;
  };
  width?: number | string;
  height?: number | string;
  children: ReactNode;
}) {
  const m = {
    train:         meta?.train         ?? "—",
    test:          meta?.test          ?? "—",
    split:         meta?.split         ?? "—",
    benchmark:     meta?.benchmark     ?? "—",
    benchmarkInfo: meta?.benchmarkInfo ?? "",
  };
  return (
    <div
      style={{
        width,
        height,
        background: "var(--bg-0)",
        color: "var(--fg-0)",
        fontFamily: "var(--sans)",
        display: "grid",
        gridTemplateRows: "40px 22px 1fr 22px",
        borderRadius: 6,
        overflow: "hidden",
        border: "1px solid var(--line-soft)",
      }}
    >
      {/* Top bar */}
      <div
        className="flex items-center gap-3.5 px-3.5"
        style={{
          borderBottom: "1px solid var(--line-soft)",
          background: "var(--bg-1)",
        }}
      >
        <Logo size={13} />
        <span style={{ height: 14, width: 1, background: "var(--line)" }} />
        <span
          style={{
            fontFamily: "var(--mono)",
            fontSize: 10.5,
            color: "var(--fg-1)",
            letterSpacing: "0.03em",
          }}
        >
          {title}
        </span>
        {sub && (
          <span
            style={{
              fontFamily: "var(--mono)",
              fontSize: 10,
              color: "var(--fg-3)",
            }}
          >
            · {sub}
          </span>
        )}
        <span style={{ height: 14, width: 1, background: "var(--line)", margin: "0 4px" }} />
        <NavLinks title={title} />
        <span style={{ flex: 1 }} />
        {top}
      </div>

      {/* Status sub-bar */}
      <div
        className="flex items-center gap-4.5 px-3.5"
        style={{
          borderBottom: "1px solid var(--line-soft)",
          background: "var(--bg-0)",
          fontFamily: "var(--mono)",
          fontSize: 10,
          color: "var(--fg-2)",
        }}
      >
        <span>
          <span style={{ color: "var(--fg-3)" }}>train</span> {m.train}
        </span>
        <span>
          <span style={{ color: "var(--fg-3)" }}>test</span>{" "}
          <span style={{ color: "var(--accent)" }}>{m.test}</span>
        </span>
        <span>
          <span style={{ color: "var(--fg-3)" }}>split</span> {m.split}
        </span>
        <span>
          <span style={{ color: "var(--fg-3)" }}>benchmark</span> {m.benchmark}
          {m.benchmarkInfo && (
            <span style={{ color: "var(--fg-3)" }}> · {m.benchmarkInfo}</span>
          )}
        </span>
        {statusExtra}
        <span style={{ flex: 1 }} />
        <span style={{ color: "var(--pos)" }}>● ready</span>
      </div>

      {/* Body */}
      <div className="flex flex-col" style={{ overflow: "hidden" }}>
        {children}
      </div>

      {/* Bottom bar */}
      <div
        className="flex items-center gap-3.5 px-3.5"
        style={{
          borderTop: "1px solid var(--line-soft)",
          background: "var(--bg-1)",
          fontFamily: "var(--mono)",
          fontSize: 10,
          color: "var(--fg-3)",
        }}
      >
        {bottom ?? (
          <>
            <span>workers 4/4</span>
            <span>last run · —</span>
            <span style={{ flex: 1 }} />
            <span>
              cmd: <span style={{ color: "var(--fg-1)" }}>?</span> help ·{" "}
              <span style={{ color: "var(--fg-1)" }}>k</span> palette ·{" "}
              <span style={{ color: "var(--fg-1)" }}>r</span> run
            </span>
          </>
        )}
      </div>
    </div>
  );
}
