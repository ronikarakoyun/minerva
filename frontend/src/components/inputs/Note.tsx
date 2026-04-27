import { ReactNode } from "react";

/**
 * Note — citation/açıklama bloğu. 2px sol border `--accent-soft`.
 * cite varsa serif italic accent renkli prefix.
 */
export function Note({
  children,
  cite,
}: {
  children: ReactNode;
  cite?: ReactNode;
}) {
  return (
    <div
      style={{
        borderLeft: "2px solid var(--accent-soft)",
        padding: "4px 0 4px 12px",
        margin: "4px 0 10px",
        fontSize: 11.5,
        color: "var(--fg-1)",
        lineHeight: 1.6,
      }}
    >
      {cite != null && (
        <span
          style={{
            fontFamily: "var(--serif)",
            fontStyle: "italic",
            color: "var(--accent)",
            marginRight: 6,
          }}
        >
          {cite}
        </span>
      )}
      {children}
    </div>
  );
}
