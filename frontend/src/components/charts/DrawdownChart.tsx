export function DrawdownChart({
  width = 720,
  height = 80,
  data,
  maxLabel,
}: {
  width?: number;
  height?: number;
  data: number[];
  maxLabel?: string;
}) {
  if (!data || data.length < 2) return <svg width="100%" viewBox={`0 0 ${width} ${height}`} />;
  const min = Math.min(...data);
  const x = (i: number) => (i / (data.length - 1)) * width;
  const y = (v: number) => -(v / (min || -1)) * (height - 8) + 4;
  const d = data.map((v, i) => (i === 0 ? "M" : "L") + x(i).toFixed(1) + "," + y(v).toFixed(1)).join(" ");
  const area = d + " L" + width + ",4 L0,4 Z";
  const maxPct = min !== 0 ? (min * 100).toFixed(1) : "0.0";
  return (
    <svg width="100%" viewBox={`0 0 ${width} ${height}`} style={{ display: "block" }}>
      <path d={area} fill="var(--neg)" opacity="0.18" />
      <path d={d} stroke="var(--neg)" strokeWidth="1" fill="none" opacity="0.9" />
      <text x="6" y={height - 6} fill="var(--fg-2)" fontSize="9" fontFamily="var(--mono)">
        {maxLabel ?? `drawdown · max ${maxPct}%`}
      </text>
    </svg>
  );
}
