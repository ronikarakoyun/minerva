import { useQuery } from "@tanstack/react-query";
import { apiFetch } from "../lib/api";

export interface MetaResponse {
  train_rows: number;
  test_rows: number;
  split_date: string;
  benchmark_days: number;
  date_min: string;
  date_max: string;
}

export function useMeta() {
  return useQuery<MetaResponse>({
    queryKey: ["meta"],
    queryFn: () => apiFetch<MetaResponse>("/api/meta"),
    staleTime: 60_000,
  });
}

export function formatMetaForChrome(m?: MetaResponse) {
  if (!m) return { train: "—", test: "—", split: "—", benchmark: "—", benchmarkInfo: "" };
  const total = m.train_rows + m.test_rows;
  const trainPct = total > 0 ? Math.round((m.train_rows / total) * 100) : 0;
  const testPct = 100 - trainPct;
  return {
    train: `${m.train_rows.toLocaleString("tr-TR")} (${trainPct}%)`,
    test: `${m.test_rows.toLocaleString("tr-TR")} (${testPct}%)`,
    split: m.split_date,
    benchmark: "BIST 100",
    benchmarkInfo: m.benchmark_days > 0
      ? `loaded · ${m.benchmark_days.toLocaleString("tr-TR")}d · ${m.date_min} → ${m.date_max}`
      : "—",
  };
}
