import { ReactNode } from "react";

/**
 * Panel — Section başlığı (§ X · italic title) + içerik.
 * Body içinde kullanılır; flex item gibi davranır.
 */
export function Panel({
  num,
  title,
  sub,
  right,
  children,
  pad = true,
  flex,
}: {
  num: string;
  title: ReactNode;
  sub?: ReactNode;
  right?: ReactNode;
  children: ReactNode;
  pad?: boolean;
  flex?: number | string;
}) {
  return (
    <section
      className="flex flex-col"
      style={{ minHeight: 0, flex }}
    >
      <div
        className="flex items-baseline gap-2"
        style={{
          padding: "10px 16px",
          borderBottom: "1px solid var(--line-soft)",
          background: "var(--bg-1)",
        }}
      >
        <span
          style={{
            fontFamily: "var(--mono)",
            fontSize: 10,
            color: "var(--fg-3)",
          }}
        >
          § {num}
        </span>
        <span
          style={{
            fontFamily: "var(--serif)",
            fontStyle: "italic",
            fontSize: 13,
            color: "var(--fg-0)",
          }}
        >
          {title}
        </span>
        {sub != null && (
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
        <span style={{ flex: 1 }} />
        {right}
      </div>
      <div
        style={{
          flex: 1,
          minHeight: 0,
          overflow: "auto",
          padding: pad ? "14px 18px" : 0,
          display: "flex",
          flexDirection: "column",
        }}
      >
        {children}
      </div>
    </section>
  );
}
