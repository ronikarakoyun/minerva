import { ReactNode } from "react";

export type PillTone = "neutral" | "accent" | "pos" | "neg" | "ghost";

const TONE: Record<PillTone, { bg: string; fg: string; bd: string }> = {
  neutral: { bg: "var(--bg-3)",        fg: "var(--fg-1)", bd: "var(--line)" },
  accent:  { bg: "var(--accent-soft)", fg: "var(--accent)", bd: "transparent" },
  pos:     { bg: "transparent",        fg: "var(--pos)", bd: "var(--pos)" },
  neg:     { bg: "transparent",        fg: "var(--neg)", bd: "var(--neg)" },
  ghost:   { bg: "transparent",        fg: "var(--fg-2)", bd: "var(--line)" },
};

export function Pill({
  children,
  tone = "neutral",
  mono = false,
}: {
  children: ReactNode;
  tone?: PillTone;
  mono?: boolean;
}) {
  const t = TONE[tone];
  return (
    <span
      className="inline-flex items-center gap-1 px-1.5 leading-[1.4]"
      style={{
        padding: "2px 7px",
        borderRadius: 3,
        background: t.bg,
        color: t.fg,
        border: `1px solid ${t.bd}`,
        fontFamily: mono ? "var(--mono)" : "var(--sans)",
        fontSize: 10,
        letterSpacing: mono ? 0 : "0.02em",
        textTransform: mono ? "none" : "uppercase",
        fontWeight: 500,
      }}
    >
      {children}
    </span>
  );
}
