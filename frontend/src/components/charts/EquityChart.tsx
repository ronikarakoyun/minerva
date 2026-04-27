export function EquityChart({
  width = 720,
  height = 220,
  train = 160,
  alpha,
  bench,
  label,
}: {
  width?: number;
  height?: number;
  train?: number;
  alpha: number[];
  bench: number[];
  label?: string;
}) {
  const all = [...alpha, ...bench];
  const min = Math.min(...all) * 0.98;
  const max = Math.max(...all) * 1.02;
  const span = max - min || 1;
  const x = (i: number) => (i / (alpha.length - 1)) * width;
  const y = (v: number) => height - ((v - min) / span) * (height - 24) - 12;
  const path = (arr: number[]) =>
    arr.map((v, i) => (i === 0 ? "M" : "L") + x(i).toFixed(1) + "," + y(v).toFixed(1)).join(" ");
  const trainX = x(train);
  return (
    <svg width="100%" viewBox={`0 0 ${width} ${height}`} style={{ display: "block" }}>
      {[0.25, 0.5, 0.75].map((t) => (
        <line
          key={t}
          x1="0" x2={width}
          y1={t * (height - 24) + 12} y2={t * (height - 24) + 12}
          stroke="var(--line-soft)" strokeWidth="0.5" strokeDasharray="2 4"
        />
      ))}
      <line
        x1={trainX} x2={trainX} y1="6" y2={height - 6}
        stroke="var(--accent)" strokeWidth="0.75" strokeDasharray="3 3" opacity="0.7"
      />
      <text x={trainX - 4} y="14" fill="var(--fg-2)" fontSize="9" textAnchor="end" fontFamily="var(--mono)">TRAIN</text>
      <text x={trainX + 4} y="14" fill="var(--accent)" fontSize="9" fontFamily="var(--mono)">TEST</text>
      <path d={path(bench)} stroke="var(--fg-3)" strokeWidth="1" fill="none" />
      <path d={path(alpha)} stroke="var(--fg-0)" strokeWidth="1.5" fill="none" />
      {label && (
        <text x={width - 6} y={height - 6} fill="var(--fg-2)" fontSize="10" textAnchor="end" fontFamily="var(--mono)">
          {label}
        </text>
      )}
    </svg>
  );
}
