import { useState, useEffect } from "react";
import { useSearchParams, useNavigate } from "react-router-dom";
import { CChrome } from "../components/chrome/CChrome";
import { Panel } from "../components/chrome/Panel";
import { Box } from "../components/chrome/Box";
import { Btn } from "../components/atoms/Btn";
import { Stat } from "../components/atoms/Stat";
import { Pill } from "../components/atoms/Pill";
import { Select } from "../components/inputs/Select";
import { EquityChart } from "../components/charts/EquityChart";
import { DrawdownChart } from "../components/charts/DrawdownChart";
import { HeatmapRow } from "../components/charts/HeatmapRow";
import { useCatalog } from "../hooks/useCatalog";
import { useMeta, formatMetaForChrome } from "../hooks/useMeta";
import { useJob } from "../hooks/useJob";
import { apiFetch } from "../lib/api";

export default function ResultsReport() {
  const [searchParams] = useSearchParams();
  const navigate = useNavigate();
  const { data: records = [] } = useCatalog();

  const urlFormula = searchParams.get("id") ?? "";
  const [selectedFormula, setSelectedFormula] = useState(urlFormula);

  // records yüklenince ilk formülü seç (URL param yoksa)
  useEffect(() => {
    if (!selectedFormula && records.length > 0) {
      setSelectedFormula(records[0].formula);
    }
  }, [records, selectedFormula]);
  const [window_, setWindow_] = useState<"test" | "train" | "all">("test");

  const [jobId, setJobId] = useState<string | null>(null);
  const [isLaunching, setIsLaunching] = useState(false);
  const jobState = useJob(jobId);

  const isRunning = isLaunching || (!!jobId && !jobState.done);
  const result = jobState.result as Record<string, any> | null;
  const { data: metaData } = useMeta();
  const chromeMeta = formatMetaForChrome(metaData);

  const [launchError, setLaunchError] = useState<string | null>(null);

  const handleRun = async () => {
    if (!selectedFormula || isLaunching) return;
    setIsLaunching(true);
    setJobId(null);
    setLaunchError(null);
    try {
      const { job_id } = await apiFetch<{ job_id: string }>("/api/backtest/run", {
        method: "POST",
        body: JSON.stringify({ formula: selectedFormula, window: window_ }),
      });
      setJobId(job_id);
    } catch (e: any) {
      setLaunchError(e?.message ?? "Backtest başlatılamadı");
    } finally {
      setIsLaunching(false);
    }
  };

  const handlePrint = () => window.print();

  const formulaOptions = records.map((r) => ({
    value: r.formula,
    label: r.formula.length > 55 ? r.formula.slice(0, 55) + "…" : r.formula,
  }));

  const equityData = result?.equity_curve
    ? (result.equity_curve as { equity: number }[]).map((p) => p.equity)
    : [];
  const benchData = result?.equity_curve
    ? (result.equity_curve as { benchmark?: number }[]).map((p) => p.benchmark ?? 0)
    : [];
  const trainSplit = Math.floor(equityData.length * 0.63);

  const foldSharpes: number[] = result?.fold_sharpes ?? [];

  const windowStr = window_ === "test" ? "OOS (Test)" : window_ === "train" ? "IS (Train)" : "Tam veri";
  const now = new Date().toLocaleDateString("tr-TR");

  return (
    <>
      {/* Print styles */}
      <style>{`
        @media print {
          .no-print { display: none !important; }
          body { background: white !important; color: black !important; }
          .print-page { page-break-after: always; }
        }
      `}</style>

      <CChrome
        title="results report"
        sub="alpha performans raporu · PDF export"
        meta={chromeMeta}
        top={
          <div className="no-print" style={{ display: "flex", gap: 8, alignItems: "center" }}>
            <Btn variant="ghost" onClick={() => navigate("/workbench")}>← Workbench</Btn>
            <Btn variant="ghost" onClick={handleRun} disabled={isRunning || !selectedFormula}>
              {isRunning ? `Hesaplanıyor… ${Math.round(jobState.progress * 100)}%` : "↺ Yenile"}
            </Btn>
            <Btn variant="primary" onClick={handlePrint} disabled={!result}>
              ↓ PDF İndir
            </Btn>
          </div>
        }
        width="100%"
        height="100vh"
      >
        <div style={{ display: "flex", flexDirection: "column", flex: 1, overflow: "auto", padding: "18px 24px", gap: 18 }}>

          {/* Formula picker — no-print */}
          <div className="no-print" style={{ display: "flex", gap: 12, alignItems: "flex-end" }}>
            <div style={{ flex: 1 }}>
              <div style={{ fontFamily: "var(--mono)", fontSize: 9.5, color: "var(--fg-3)", letterSpacing: 0.5, marginBottom: 5 }}>
                FORMÜL SEÇ
              </div>
              <Select
                options={formulaOptions}
                value={selectedFormula}
                onChange={(v) => setSelectedFormula(v)}
                style={{ width: "100%" }}
              />
            </div>
            <div>
              <div style={{ fontFamily: "var(--mono)", fontSize: 9.5, color: "var(--fg-3)", letterSpacing: 0.5, marginBottom: 5 }}>
                PENCERE
              </div>
              <Select
                options={[
                  { value: "test", label: "TEST · oos" },
                  { value: "train", label: "TRAIN · is" },
                  { value: "all", label: "TAM" },
                ]}
                value={window_}
                onChange={(v) => setWindow_(v as "test" | "train" | "all")}
              />
            </div>
            <Btn variant="primary" onClick={handleRun} disabled={isRunning || !selectedFormula}>
              {isRunning ? `${Math.round(jobState.progress * 100)}%` : "▸ Çalıştır"}
            </Btn>
          </div>

          {/* Progress bar — no-print */}
          {isRunning && (
            <div className="no-print" style={{ display: "flex", alignItems: "center", gap: 8 }}>
              <div style={{ flex: 1, height: 3, background: "var(--bg-2)", borderRadius: 2, overflow: "hidden" }}>
                <div style={{ width: `${Math.round(jobState.progress * 100)}%`, height: "100%", background: "var(--accent)", transition: "width 0.3s" }} />
              </div>
              <span style={{ fontFamily: "var(--mono)", fontSize: 9.5, color: "var(--fg-3)" }}>
                {jobState.logs[jobState.logs.length - 1] ?? "başlatılıyor…"}
              </span>
            </div>
          )}

          {/* Error state */}
          {(jobState.error || launchError) && (
            <Box p={12}>
              <span style={{ fontFamily: "var(--mono)", fontSize: 11, color: "var(--neg)" }}>
                ⚠ {jobState.error ?? launchError}
              </span>
            </Box>
          )}

          {/* Report body — only shown when result available */}
          {result && (
            <>
              {/* § 00 — Başlık */}
              <div
                style={{
                  borderBottom: "2px solid var(--line-soft)",
                  paddingBottom: 14,
                  display: "flex",
                  justifyContent: "space-between",
                  alignItems: "flex-end",
                }}
              >
                <div>
                  <div style={{ fontFamily: "var(--serif)", fontStyle: "italic", fontSize: 20, color: "var(--fg-0)", marginBottom: 4 }}>
                    Alpha Performans Raporu
                  </div>
                  <div style={{ fontFamily: "var(--mono)", fontSize: 10, color: "var(--fg-3)" }}>
                    {windowStr} · {now}
                  </div>
                </div>
                <Pill mono tone="ghost">{window_} · window</Pill>
              </div>

              {/* Formül */}
              <Panel num="01" title="Formül" sub="alpha ifadesi · AST köklü" flex={0}>
                <Box p={12}>
                  <code style={{ fontFamily: "var(--mono)", fontSize: 12, color: "var(--fg-0)", wordBreak: "break-all", lineHeight: 1.7 }}>
                    {selectedFormula}
                  </code>
                </Box>
              </Panel>

              {/* KPI grid */}
              <Panel num="02" title="Performans Özeti" sub="temel metrikler" flex={0}>
                <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 12 }}>
                  <Stat
                    label="Sharpe / IR"
                    value={result.sharpe != null ? result.sharpe.toFixed(3) : "—"}
                    hint="yıllık"
                    tone={result.sharpe > 1 ? "pos" : result.sharpe < 0 ? "neg" : undefined}
                  />
                  <Stat
                    label="IC"
                    value={result.ic != null ? result.ic.toFixed(4) : "—"}
                    tone={result.ic > 0.005 ? "pos" : result.ic < 0 ? "neg" : undefined}
                  />
                  <Stat
                    label="Rank IC"
                    value={result.rank_ic != null ? result.rank_ic.toFixed(4) : "—"}
                    tone={result.rank_ic > 0.005 ? "pos" : undefined}
                  />
                  <Stat
                    label="Yıllık Getiri"
                    value={result.annual != null ? `${result.annual.toFixed(1)}%` : "—"}
                    tone={result.annual > 0 ? "pos" : "neg"}
                  />
                  <Stat
                    label="Max Drawdown"
                    value={result.mdd != null ? `${result.mdd.toFixed(1)}%` : "—"}
                    tone="neg"
                  />
                  <Stat
                    label="Net Getiri"
                    value={result.net_return != null ? `${result.net_return.toFixed(1)}%` : "—"}
                    tone={result.net_return > 0 ? "pos" : "neg"}
                  />
                  {result.alpha_ir != null && (
                    <Stat label="Alpha IR" value={result.alpha_ir.toFixed(3)} hint="benchmark'a karşı" />
                  )}
                  {result.beta != null && (
                    <Stat label="Beta" value={result.beta.toFixed(3)} hint="benchmark β" />
                  )}
                  <Stat
                    label="Gözlem sayısı"
                    value={result.n_observations != null ? result.n_observations.toLocaleString("tr-TR") : "—"}
                    hint="(Ticker, Date) çiftleri"
                  />
                </div>
              </Panel>

              {/* Equity chart */}
              <Panel num="03" title="Equity Eğrisi" sub="alpha vs benchmark" flex={0}>
                <div
                  style={{
                    background: "var(--bg-1)",
                    border: "1px solid var(--line-soft)",
                    borderRadius: 2,
                    overflow: "hidden",
                  }}
                >
                  <EquityChart
                    alpha={equityData}
                    bench={benchData.some((v) => v !== 0) ? benchData : []}
                    train={trainSplit}
                    width={760}
                    height={160}
                  />
                </div>
              </Panel>

              {/* Drawdown */}
              <Panel num="04" title="Drawdown" sub="pik'ten çekilme" flex={0}>
                <div
                  style={{
                    background: "var(--bg-1)",
                    border: "1px solid var(--line-soft)",
                    borderRadius: 2,
                    overflow: "hidden",
                  }}
                >
                  <DrawdownChart
                    data={equityData.length > 0
                      ? (() => {
                          let peak = equityData[0];
                          return equityData.map((v) => {
                            if (v > peak) peak = v;
                            return peak > 0 ? (v - peak) / peak : 0;
                          });
                        })()
                      : []
                    }
                    width={760}
                    height={70}
                  />
                </div>
              </Panel>

              {/* Walk-forward fold heatmap */}
              {foldSharpes.length > 0 && (
                <Panel num="05" title="Walk-Forward Foldlar" sub={`${foldSharpes.length} fold · Sharpe dağılımı`} flex={0}>
                  <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
                    {foldSharpes.map((_, fi) => (
                      <div key={fi} style={{ display: "flex", alignItems: "center", gap: 10 }}>
                        <span style={{ fontFamily: "var(--mono)", fontSize: 9.5, color: "var(--fg-3)", minWidth: 32 }}>
                          fold-{fi + 1}
                        </span>
                        <HeatmapRow values={[foldSharpes[fi]]} label={`fold-${fi + 1}`} width={700} h={20} />
                        <span style={{ fontFamily: "var(--mono)", fontSize: 9.5, color: foldSharpes[fi] > 0 ? "var(--pos)" : "var(--neg)", minWidth: 40 }}>
                          {foldSharpes[fi].toFixed(2)}
                        </span>
                      </div>
                    ))}
                  </div>
                </Panel>
              )}

              {/* Metodoloji notu */}
              <Panel num="06" title="Metodoloji" sub="referans" flex={0}>
                <Box p={12}>
                  <div style={{ display: "flex", flexDirection: "column", gap: 8, fontFamily: "var(--mono)", fontSize: 10, color: "var(--fg-2)", lineHeight: 1.7 }}>
                    <div>
                      <span style={{ color: "var(--fg-3)" }}>Backtest Motoru: </span>
                      Long-only, günlük rebalance. Ücretler: alış {0.05}% + satış {0.15}%. Top-50 formül ağırlığı eşit.
                    </div>
                    <div>
                      <span style={{ color: "var(--fg-3)" }}>IC: </span>
                      Spearman rank korelasyon (sinyal ↔ sonraki günlük getiri). Sektör-nötr ve günlük kesit.
                    </div>
                    <div>
                      <span style={{ color: "var(--fg-3)" }}>Walk-Forward: </span>
                      Purged K-Fold (embargo=5 gün, purge=10 gün). Bailey & López de Prado (2016).
                    </div>
                    <div>
                      <span style={{ color: "var(--fg-3)" }}>Veri: </span>
                      BIST-100 hisse senetleri · günlük OHLCV · temizlenmiş {result.n_observations?.toLocaleString("tr-TR") ?? "—"} gözlem.
                    </div>
                  </div>
                </Box>
              </Panel>

              {/* Footer */}
              <div style={{ borderTop: "1px solid var(--line-soft)", paddingTop: 12, display: "flex", justifyContent: "space-between" }}>
                <span style={{ fontFamily: "var(--mono)", fontSize: 9, color: "var(--fg-3)" }}>
                  Minerva v3 Studio — Ünal Roni Karakoyun
                </span>
                <span style={{ fontFamily: "var(--mono)", fontSize: 9, color: "var(--fg-3)" }}>
                  {now} · window={window_}
                </span>
              </div>
            </>
          )}

          {/* Empty state */}
          {!result && !isRunning && !jobState.error && (
            <div
              style={{
                flex: 1,
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                fontFamily: "var(--mono)",
                fontSize: 11,
                color: "var(--fg-3)",
              }}
            >
              Formül seç → ▸ Çalıştır → rapor görünür → ↓ PDF İndir
            </div>
          )}
        </div>
      </CChrome>
    </>
  );
}
