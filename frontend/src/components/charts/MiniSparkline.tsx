export function MiniSparkline({
  data,
  width = 120,
  height = 28,
  color = "currentColor",
  fill = false,
}: {
  data: number[];
  width?: number;
  height?: number;
  color?: string;
  fill?: boolean;
}) {
  // N35: NaN guard — sonsuz veya NaN değerleri filtrele
  const safeData = (data ?? []).filter((v) => Number.isFinite(v));
  if (safeData.length < 2) return <svg width={width} height={height} />;
  const min = Math.min(...safeData);
  const max = Math.max(...safeData);
  const span = max - min || 1;
  const pts = safeData.map((v, i) => [
    (i / (safeData.length - 1)) * width,
    height - ((v - min) / span) * height,
  ]);
  const d = pts.map((p, i) => (i === 0 ? "M" : "L") + p[0].toFixed(1) + "," + p[1].toFixed(1)).join(" ");
  const area = d + " L" + width + "," + height + " L0," + height + " Z";
  return (
    <svg width={width} height={height} style={{ display: "block", overflow: "visible" }}>
      {fill && <path d={area} fill={color} opacity="0.12" />}
      <path d={d} stroke={color} strokeWidth="1.25" fill="none" />
    </svg>
  );
}
