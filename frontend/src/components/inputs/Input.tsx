import { CSSProperties, ReactNode, ChangeEvent } from "react";

export function Input({
  label,
  hint,
  value,
  onChange,
  textarea = false,
  rows = 4,
  placeholder,
  suffix,
  mono = true,
  width,
  style,
}: {
  label?: ReactNode;
  hint?: ReactNode;
  value?: string;
  onChange?: (next: string) => void;
  textarea?: boolean;
  rows?: number;
  placeholder?: string;
  suffix?: ReactNode;
  mono?: boolean;
  width?: string | number;
  style?: CSSProperties;
}) {
  const interactive = !!onChange;
  const handleChange = (
    e: ChangeEvent<HTMLInputElement | HTMLTextAreaElement>,
  ) => onChange?.(e.target.value);

  const fieldStyle: React.CSSProperties = {
    background: "var(--bg-2)",
    border: "1px solid var(--line)",
    borderRadius: 2,
    padding: "8px 10px",
    fontFamily: "var(--mono)",
    fontSize: 11.5,
    color: value ? "var(--fg-0)" : "var(--fg-3)",
    lineHeight: 1.6,
    width: "100%",
    outline: "none",
    resize: "vertical",
    minHeight: textarea ? 22 * rows : undefined,
  };

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
      {textarea ? (
        interactive ? (
          <textarea
            value={value ?? ""}
            onChange={handleChange}
            placeholder={placeholder}
            rows={rows}
            style={fieldStyle}
          />
        ) : (
          <div style={{ ...fieldStyle, whiteSpace: "pre-wrap" }}>
            {value || (
              <span style={{ color: "var(--fg-3)" }}>{placeholder}</span>
            )}
          </div>
        )
      ) : (
        <div
          className="flex items-center gap-1.5"
          style={{
            background: "var(--bg-2)",
            border: "1px solid var(--line)",
            borderRadius: 2,
            padding: "5px 9px",
          }}
        >
          {interactive ? (
            <input
              type="text"
              value={value ?? ""}
              onChange={handleChange}
              placeholder={placeholder}
              style={{
                flex: 1,
                background: "transparent",
                border: "none",
                outline: "none",
                color: "var(--fg-0)",
                fontFamily: mono ? "var(--mono)" : "var(--sans)",
                fontSize: 11.5,
              }}
            />
          ) : (
            <span
              style={{
                flex: 1,
                fontFamily: mono ? "var(--mono)" : "var(--sans)",
                fontSize: 11.5,
                color: value ? "var(--fg-0)" : "var(--fg-3)",
              }}
            >
              {value || placeholder}
            </span>
          )}
          {suffix != null && (
            <span
              style={{
                fontFamily: "var(--mono)",
                fontSize: 10,
                color: "var(--fg-3)",
              }}
            >
              {suffix}
            </span>
          )}
        </div>
      )}
    </div>
  );
}
