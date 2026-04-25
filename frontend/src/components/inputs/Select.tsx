import { CSSProperties, ReactNode } from "react";

type SelectOption = string | { value: string; label: string };

function optVal(o: SelectOption) { return typeof o === "string" ? o : o.value; }
function optLabel(o: SelectOption) { return typeof o === "string" ? o : o.label; }

export function Select({
  label,
  hint,
  value,
  onChange,
  options,
  width,
  style,
}: {
  label?: ReactNode;
  hint?: ReactNode;
  value: string;
  onChange?: (next: string) => void;
  options?: SelectOption[];
  width?: string | number;
  style?: CSSProperties;
}) {
  const interactive = !!onChange && !!options;
  return (
    <div className="flex flex-col gap-1" style={{ width, ...style }}>
      {label != null && (
        <div className="flex items-baseline gap-1.5">
          <span style={{ fontSize: 11, color: "var(--fg-1)" }}>{label}</span>
          {hint != null && (
            <span
              style={{
                fontFamily: "var(--mono)",
                fontSize: 9.5,
                color: "var(--fg-3)",
              }}
            >
              {hint}
            </span>
          )}
        </div>
      )}
      <div
        className="flex items-center gap-1.5"
        style={{
          background: "var(--bg-2)",
          border: "1px solid var(--line)",
          borderRadius: 2,
          padding: "5px 9px",
          cursor: interactive ? "pointer" : "default",
        }}
      >
        {interactive ? (
          <select
            value={value}
            onChange={(e) => onChange!(e.target.value)}
            style={{
              flex: 1,
              background: "transparent",
              color: "var(--fg-0)",
              border: "none",
              outline: "none",
              fontSize: 11.5,
              fontFamily: "var(--sans)",
              appearance: "none",
              WebkitAppearance: "none",
            }}
          >
            {options!.map((o) => (
              <option key={optVal(o)} value={optVal(o)} style={{ background: "var(--bg-2)" }}>
                {optLabel(o)}
              </option>
            ))}
          </select>
        ) : (
          <span style={{ flex: 1, fontSize: 11.5, color: "var(--fg-0)" }}>
            {value}
          </span>
        )}
        <span style={{ color: "var(--fg-3)", fontSize: 10 }}>▾</span>
      </div>
    </div>
  );
}
