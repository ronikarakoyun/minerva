import { ReactNode, useId } from "react";

export function Field({
  label,
  value,
  hint,
  narrow = false,
  type = "text",
  onChange,
}: {
  label: ReactNode;
  value: ReactNode;
  hint?: ReactNode;
  narrow?: boolean;
  /** HTML input type. Pass "number" for numeric fields to enable browser validation. */
  type?: "text" | "number";
  onChange?: (v: string) => void;
}) {
  const inputId = useId();

  return (
    <div
      className="flex items-center justify-between gap-3"
      style={{ padding: "6px 0", borderBottom: "1px dotted var(--line-soft)" }}
    >
      <div className="flex flex-col">
        <label
          htmlFor={onChange ? inputId : undefined}
          style={{ fontSize: 11.5, color: "var(--fg-1)", cursor: onChange ? "default" : undefined }}
        >
          {label}
        </label>
        {hint != null && (
          <span style={{ fontSize: 9, color: "var(--fg-3)", fontFamily: "var(--mono)" }}>
            {hint}
          </span>
        )}
      </div>
      {onChange ? (
        <input
          id={inputId}
          type={type}
          value={String(value)}
          onChange={(e) => onChange(e.target.value)}
          aria-label={typeof label === "string" ? label : undefined}
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
          role="status"
          aria-label={typeof label === "string" ? String(label) : undefined}
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
