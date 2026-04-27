import { ReactNode } from "react";

export function Check({
  label,
  hint,
  checked,
  on,
  onChange,
}: {
  label: ReactNode;
  hint?: ReactNode;
  checked?: boolean;
  on?: boolean;
  onChange?: (next: boolean) => void;
}) {
  const active = checked ?? on ?? false;
  const interactive = !!onChange;
  return (
    <div
      className="flex items-start gap-2 py-1"
      style={{ cursor: interactive ? "pointer" : "default" }}
      onClick={() => onChange?.(!active)}
    >
      <span
        style={{
          width: 12,
          height: 12,
          borderRadius: 2,
          marginTop: 2,
          border: `1px solid ${active ? "var(--accent)" : "var(--line)"}`,
          background: active ? "var(--accent)" : "transparent",
          position: "relative",
          flex: "0 0 auto",
        }}
      >
        {active && (
          <span
            style={{
              position: "absolute",
              inset: 0,
              color: "var(--bg-0)",
              fontSize: 9,
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              fontWeight: 700,
              lineHeight: 1,
            }}
          >
            ✓
          </span>
        )}
      </span>
      <div className="flex flex-col flex-1">
        <span
          style={{
            fontSize: 11.5,
            color: active ? "var(--fg-0)" : "var(--fg-1)",
          }}
        >
          {label}
        </span>
        {hint != null && (
          <span
            style={{
              fontSize: 9.5,
              color: "var(--fg-3)",
              fontFamily: "var(--mono)",
            }}
          >
            {hint}
          </span>
        )}
      </div>
    </div>
  );
}
