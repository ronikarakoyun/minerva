import { ReactNode } from "react";

export function Field({
  label,
  value,
  hint,
  narrow = false,
  onChange,
}: {
  label: ReactNode;
  value: ReactNode;
  hint?: ReactNode;
  narrow?: boolean;
  onChange?: (v: string) => void;
}) {
  return (
    <div
      className="flex items-center justify-between gap-3"
      style={{ padding: "6px 0", borderBottom: "1px dotted var(--line-soft)" }}
    >
      <div className="flex flex-col">
        <span style={{ fontSize: 11.5, color: "var(--fg-1)" }}>{label}</span>
        {hint != null && (
          <span style={{ fontSize: 9, color: "var(--fg-3)", fontFamily: "var(--mono)" }}>
            {hint}
          </span>
        )}
      </div>
      {onChange ? (
        <input
          value={String(value)}
          onChange={(e) => onChange(e.target.value)}
          style={{
            fontFamily: "var(--mono)",
            fontSize: 12,
            color: "var(--fg-0)",
            background: "var(--bg-2)",
            padding: "2px 8px",
            borderRadius: 2,
            border: "1px solid var(--line)",
            minWidth: narrow ? 0 : 56,
            width: narrow ? 44 : 64,
            textAlign: "right",
            outline: "none",
          }}
        />
      ) : (
        <span
          style={{
            fontFamily: "var(--mono)",
            fontSize: 12,
            color: "var(--fg-0)",
            background: "var(--bg-2)",
            padding: "2px 8px",
            borderRadius: 2,
            border: "1px solid var(--line-soft)",
            minWidth: narrow ? 0 : 56,
            textAlign: "right",
          }}
        >
          {value}
        </span>
      )}
    </div>
  );
}
