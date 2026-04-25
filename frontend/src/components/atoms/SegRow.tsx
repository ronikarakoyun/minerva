export function SegRow({
  items,
  options,
  active,
  value,
  onChange,
}: {
  items?: string[];
  options?: string[];
  active?: string;
  value?: string;
  onChange?: (next: string) => void;
}) {
  const opts = items ?? options ?? [];
  const current = active ?? value ?? "";
  return (
    <div
      className="inline-flex"
      style={{
        border: "1px solid var(--line)",
        borderRadius: 3,
        background: "var(--bg-1)",
        padding: 2,
      }}
    >
      {opts.map((it) => {
        const isActive = it === current;
        return (
          <span
            key={it}
            onClick={onChange ? () => onChange(it) : undefined}
            style={{
              padding: "4px 10px",
              fontSize: 10.5,
              fontFamily: "var(--mono)",
              color: isActive ? "var(--bg-0)" : "var(--fg-1)",
              background: isActive ? "var(--fg-0)" : "transparent",
              borderRadius: 2,
              letterSpacing: "0.02em",
              cursor: onChange ? "pointer" : "default",
              userSelect: "none",
            }}
          >
            {it}
          </span>
        );
      })}
    </div>
  );
}
