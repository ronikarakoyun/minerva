import { ReactNode } from "react";

export type StatTone = "default" | "pos" | "neg";

export function Stat({
  label,
  value,
  hint,
  mono = true,
  tone = "default",
}: {
  label: ReactNode;
  value: ReactNode;
  hint?: ReactNode;
  mono?: boolean;
  tone?: StatTone;
}) {
  const color =
    tone === "pos" ? "var(--pos)"
    : tone === "neg" ? "var(--neg)"
    : "var(--fg-0)";
  return (
    <div className="flex flex-col gap-0.5">
      <div
        className="font-sans text-fg-2 uppercase"
        style={{ fontSize: 10, letterSpacing: "0.06em" }}
      >
        {label}
      </div>
      <div
        style={{
          fontSize: 22,
          color,
          fontFamily: mono ? "var(--mono)" : "var(--serif)",
          fontWeight: 500,
          lineHeight: 1.1,
          letterSpacing: "-0.015em",
        }}
      >
        {value}
      </div>
      {hint != null && (
        <div
          className="text-fg-3"
          style={{ fontSize: 10, fontFamily: "var(--mono)" }}
        >
          {hint}
        </div>
      )}
    </div>
  );
}
