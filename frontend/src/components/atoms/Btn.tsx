import { ButtonHTMLAttributes, ReactNode } from "react";

export type BtnVariant = "default" | "primary" | "ghost" | "danger";

export type BtnProps = ButtonHTMLAttributes<HTMLButtonElement> & {
  variant?: BtnVariant;
  primary?: boolean;
  mono?: boolean;
  full?: boolean;
  small?: boolean;
  icon?: ReactNode;
};

export function Btn({
  children,
  variant = "default",
  primary,
  mono,
  full,
  small,
  icon,
  className = "",
  ...rest
}: BtnProps) {
  const isPrimary = primary || variant === "primary";
  const isGhost = variant === "ghost";
  const isDanger = variant === "danger";

  const bg = isPrimary ? "var(--fg-0)" : "transparent";
  const fg = isPrimary ? "var(--bg-0)" : isDanger ? "var(--neg)" : "var(--fg-0)";
  const border = isPrimary
    ? "var(--fg-0)"
    : isDanger
    ? "var(--neg)"
    : isGhost
    ? "var(--line-soft)"
    : "var(--line)";

  return (
    <button
      {...rest}
      className={`inline-flex items-center justify-center gap-1.5 transition-colors ${className}`}
      style={{
        padding: small ? "5px 10px" : "8px 14px",
        width: full ? "100%" : "auto",
        fontFamily: mono ? "var(--mono)" : "var(--sans)",
        fontSize: small ? 11 : 12,
        fontWeight: 500,
        background: bg,
        color: fg,
        border: `1px solid ${border}`,
        borderRadius: 3,
        cursor: rest.disabled ? "not-allowed" : "pointer",
        letterSpacing: "0.02em",
        opacity: rest.disabled ? 0.5 : 1,
      }}
    >
      {icon != null && (
        <span style={{ fontFamily: "var(--mono)", opacity: 0.7 }}>{icon}</span>
      )}
      {children}
    </button>
  );
}
