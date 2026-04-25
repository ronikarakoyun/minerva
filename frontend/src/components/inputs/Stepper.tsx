import { ReactNode } from "react";

/**
 * Stepper — numeric value + (− / +) butonları. Read-only ya da interaktif.
 */
export function Stepper({
  label,
  hint,
  value,
  onChange,
  step = 1,
  min,
  max,
  width,
}: {
  label?: ReactNode;
  hint?: ReactNode;
  value: number | string;
  onChange?: (next: number) => void;
  step?: number;
  min?: number;
  max?: number;
  width?: string | number;
}) {
  const numValue =
    typeof value === "number" ? value : parseFloat(String(value)) || 0;

  const apply = (delta: number) => {
    if (!onChange) return;
    let next = numValue + delta;
    if (typeof min === "number") next = Math.max(min, next);
    if (typeof max === "number") next = Math.min(max, next);
    onChange(next);
  };

  return (
    <div className="flex flex-col gap-1" style={{ width }}>
      {label != null && (
        <span style={{ fontSize: 11, color: "var(--fg-1)" }}>{label}</span>
      )}
      <div
        className="flex items-stretch"
        style={{
          background: "var(--bg-2)",
          border: "1px solid var(--line)",
          borderRadius: 2,
        }}
      >
        <span
          style={{
            flex: 1,
            fontFamily: "var(--mono)",
            fontSize: 12,
            color: "var(--fg-0)",
            padding: "5px 9px",
          }}
        >
          {value}
        </span>
        <button
          onClick={() => apply(-step)}
          disabled={!onChange}
          style={{
            width: 22,
            borderLeft: "1px solid var(--line)",
            color: "var(--fg-2)",
            fontSize: 10,
            background: "transparent",
            cursor: onChange ? "pointer" : "default",
          }}
        >
          −
        </button>
        <button
          onClick={() => apply(step)}
          disabled={!onChange}
          style={{
            width: 22,
            borderLeft: "1px solid var(--line)",
            color: "var(--fg-2)",
            fontSize: 10,
            background: "transparent",
            cursor: onChange ? "pointer" : "default",
          }}
        >
          +
        </button>
      </div>
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
  );
}
