import { CSSProperties, ReactNode } from "react";

export function Box({
  children,
  p = 12,
  style,
}: {
  children: ReactNode;
  p?: number;
  style?: CSSProperties;
}) {
  return (
    <div
      style={{
        background: "var(--bg-1)",
        border: "1px solid var(--line-soft)",
        borderRadius: 3,
        padding: p,
        ...style,
      }}
    >
      {children}
    </div>
  );
}
