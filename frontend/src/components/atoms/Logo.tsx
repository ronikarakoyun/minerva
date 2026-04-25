/**
 * Minerva logosu — şapka çatısı SVG + serif "Minerva" + mono "v3".
 * Mock'taki minerva-shared.jsx Logo komponenti birebir.
 */
export function Logo({ size = 14 }: { size?: number }) {
  return (
    <span className="inline-flex items-center gap-2">
      <svg width={size + 2} height={size + 2} viewBox="0 0 16 16">
        <circle
          cx="8"
          cy="8"
          r="6.5"
          fill="none"
          stroke="var(--fg-0)"
          strokeWidth="1"
        />
        <path
          d="M3 11 L6 5 L8 9 L10 5 L13 11"
          fill="none"
          stroke="var(--fg-0)"
          strokeWidth="1"
          strokeLinecap="square"
          strokeLinejoin="miter"
        />
      </svg>
      <span
        style={{
          fontFamily: "var(--serif)",
          fontSize: size,
          color: "var(--fg-0)",
          letterSpacing: "0.04em",
          fontWeight: 500,
        }}
      >
        Minerva
      </span>
      <span
        style={{
          fontFamily: "var(--mono)",
          fontSize: size - 3,
          color: "var(--fg-3)",
        }}
      >
        v3
      </span>
    </span>
  );
}
