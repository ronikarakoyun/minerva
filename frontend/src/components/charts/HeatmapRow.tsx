export function HeatmapRow({
  values,
  label,
  width = 720,
  h = 18,
}: {
  values: number[];
  label: string;
  width?: number;
  h?: number;
}) {
  // N35: NaN guard — sonsuz veya NaN değerleri filtrele
  const safeValues = values.map((v) => (Number.isFinite(v) ? v : 0));
  const absMax = Math.max(...safeValues.map(Math.abs)) || 1;
  const cellW = width / values.length;
  return (
    <svg width="100%" viewBox={`0 0 ${width + 90} ${h + 2}`} style={{ display: "block" }}>
      <text x="0" y={h - 4} fill="var(--fg-2)" fontSize="10" fontFamily="var(--mono)">{label}</text>
      {safeValues.map((v, i) => {
        const positive = v >= 0;
        const opacity = 0.15 + (Math.abs(v) / absMax) * 0.7;
        return (
          <rect
            key={i}
            x={80 + i * cellW}
            y="0"
            width={cellW - 0.5}
            height={h}
            fill={positive ? "var(--pos)" : "var(--neg)"}
            opacity={opacity}
          />
        );
      })}
    </svg>
  );
}
