import { ReactNode } from "react";

/**
 * SectionLabel — `§ 0X  Title` editorial başlık. Source Serif 4 italic.
 */
export function SectionLabel({
  children,
  num,
}: {
  children: ReactNode;
  num?: string;
}) {
  return (
    <div
      className="flex items-baseline gap-2"
      style={{
        borderBottom: "1px solid var(--line-soft)",
        paddingBottom: 6,
        marginBottom: 12,
      }}
    >
      {num && (
        <span
          style={{
            fontFamily: "var(--mono)",
            fontSize: 10,
            color: "var(--fg-3)",
          }}
        >
          § {num}
        </span>
      )}
      <span
        style={{
          fontFamily: "var(--serif)",
          fontSize: 13,
          fontStyle: "italic",
          color: "var(--fg-1)",
          letterSpacing: "0.015em",
        }}
      >
        {children}
      </span>
    </div>
  );
}
