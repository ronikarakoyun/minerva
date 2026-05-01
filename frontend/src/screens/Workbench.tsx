import { useState } from "react";
import { useSearchParams, useNavigate } from "react-router-dom";
import { CChrome } from "../components/chrome/CChrome";
import { Btn } from "../components/atoms/Btn";
import { Pill } from "../components/atoms/Pill";
import { Stat } from "../components/atoms/Stat";
import { Field } from "../components/atoms/Field";
import { Check } from "../components/atoms/Check";
import { SegRow } from "../components/atoms/SegRow";
import { SectionLabel } from "../components/atoms/SectionLabel";
import { EquityChart } from "../components/charts/EquityChart";
import { DrawdownChart } from "../components/charts/DrawdownChart";
import { HeatmapRow } from "../components/charts/HeatmapRow";
import { useCatalog } from "../hooks/useCatalog";
import { useMeta, formatMetaForChrome } from "../hooks/useMeta";
import { useJob } from "../hooks/useJob";
import { apiFetch } from "../lib/api";
import { WINDOW_MAP, windowToLabel } from "../lib/window";
import { EvaluateResult, EquityPoint } from "../types";

// Operator token coloring
function colorizeFormula(expr: string): React.ReactNode {
  const tokens = expr.split(/(\(|\)|,|\s+)/);
  return tokens.map((tok, i) => {
    if (/^CSRank|^TsRank/.test(tok)) return <span key={i} style={{ color: "var(--accent)" }}>{tok}</span>;
    if (/^Vlot|^Volume/.test(tok)) return <span key={i} style={{ color: "var(--warn)" }}>{tok}</span>;
    if (/^Div|^Add|^Sub|^Mul|^Pow|^Log|^Abs|^Sign|^Greater|^Less|^Delta|^TsMean|^TsStd|^TsCorr|^Mad|^Cov|^Med|^Kurt|^Sum/.test(tok))
      return <span key={i} style={{ color: "var(--fg-2)" }}>{tok}</span>;
    if (/^-?[0-9]+(\.[0-9]+)?(e[+-]?[0-9]+)?$/.test(tok))
      return <span key={i} style={{ color: "var(--fg-1)" }}>{tok}</span>;
    return <span key={i}>{tok}</span>;
  });
}


export default function Workbench() {
  const [searchParams] = useSearchParams();
  const navigate = useNavigate();
  const activeId = searchParams.get("id") ?? "";
  const { data: metaData } = useMeta();
  const chromeMeta = formatMetaForChrome(metaData);

  const { data: records = [] } = useCatalog();

  const [backtestWindow, setBacktestWindow] = useState<"test" | "train" | "all">("test");
  const [filterText, setFilterText] = useState("");
  const [sourceFilter, setSourceFilter] = useState("EVO");
  const [neutralize, setNeutralize] = useState(true);

  // Mining params — numeric state (N38: store as numbers to avoid string→API mismatch)
  const [mPopSize, setMPopSize] = useState(300);
  const [mMaxK, setMMaxK] = useState(15);
  const [mFolds, setMFolds] = useState(5);
  const [mEmbargo, setMEmbargo] = useState(5);
  const [mPurge, setMPurge] = useState(10);
  const [mLambdaStd, setMLambdaStd] = useState(0.50);
  const [mLambdaCx, setMLambdaCx] = useState(0.001);
  const [mLambdaSize, setMLambdaSize] = useState(0.50);
  const [mSizeCorr, setMSizeCorr] = useState(0.70);

  const [jobId, setJobId] = useState<string | null>(null);
  const [isLaunching, setIsLaunching] = useState(false);
  const jobState = useJob(jobId);

  const activeRecord = records.find((r) => r.formula === activeId) ?? records[0] ?? null;
  const formula = activeRecord?.formula ?? "";

  const filteredRecords = records.filter((r) => {
    if (filterText) {
      try {
        if (!new RegExp(filterText, "i").test(r.formula)) return false;
      } catch {
        return false;
      }
    }
    if (sourceFilter !== "EVO" && r.source !== sourceFilter.toLowerCase()) return false;
    return true;
  });

  const [launchError, setLaunchError] = useState<string | null>(null);

  // Mining (evolution) job state
  const [miningJobId, setMiningJobId] = useState<string | null>(null);
  const [miningLaunching, setMiningLaunching] = useState(false);
  const [miningError, setMiningError] = useState<string | null>(null);
  const miningJob = useJob(miningJobId);
  const isMining = miningLaunching || (!!miningJobId && !miningJob.done);

  const handleRun = async () => {
    if (!formula || isLaunching) return;
    setIsLaunching(true);
    setJobId(null);
    setLaunchError(null);
    try {
      const { job_id } = await apiFetch<{ job_id: string }>("/api/backtest/run", {
        method: "POST",
        body: JSON.stringify({ formula, window: backtestWindow }),
      });
      setJobId(job_id);
    } catch (e: any) {
      setLaunchError(e?.message ?? "Backtest başlatılamadı");
    } finally {
      setIsLaunching(false);
    }
  };

  const handleMining = async () => {
    if (miningLaunching) return;
    setMiningLaunching(true);
    setMiningJobId(null);
    setMiningError(null);
    try {
      // N38: values are already numbers — no parseInt/parseFloat needed
      const { job_id } = await apiFetch<{ job_id: string }>("/api/mining/start", {
        method: "POST",
        body: JSON.stringify({
          window: backtestWindow,
          num_gen: mPopSize || 200,
          max_K: mMaxK || 15,
          wf_n_folds: mFolds || 5,
          wf_embargo: mEmbargo || 5,
          wf_purge: mPurge || 10,
          lambda_std: mLambdaStd || 2.0,
          lambda_cx: mLambdaCx || 0.003,
          lambda_size: mLambdaSize || 0.5,
          size_corr_hard_limit: mSizeCorr || 0.7,
          neutralize,
          save_to_catalog: true,
        }),
      });
      setMiningJobId(job_id);
    } catch (e: any) {
      setMiningError(e?.message ?? "Evrim başlatılamadı");
    } finally {
      setMiningLaunching(false);
    }
  };

  const isRunning = isLaunching || (!!jobId && !jobState.done);

  const displayResult: EvaluateResult | null = jobState.result;
  const equityCurve: EquityPoint[] = displayResult?.equity_curve ?? [];
  const equityAlpha = equityCurve.map((p) => p.equity);
  const equityBench = equityCurve.some((p) => p.benchmark != null)
    ? equityCurve.map((p) => p.benchmark ?? 0)
    : [];
  const ddData = equityAlpha.length > 0
    ? (() => {
        let peak = equityAlpha[0];
        return equityAlpha.map((v) => {
          if (v > peak) peak = v;
          return peak > 0 ? (v - peak) / peak : 0;
        });
      })()
    : [];
  const foldSharpes = displayResult?.fold_sharpes ?? [];

  return (
    <CChrome
      title="workbench"
      sub="variant-c"
      meta={chromeMeta}
      top={
        <>
          <Btn variant="ghost" onClick={() => navigate("/catalog")}>← Katalog</Btn>
          <Btn variant="ghost" onClick={() => navigate("/llm-trainer")}>↺ Tree-LSTM</Btn>
          <Btn variant="primary" onClick={handleRun} disabled={isRunning || !formula}>
            {jobState.reconnecting
              ? "Yeniden bağlanıyor…"
              : isRunning
              ? `Çalışıyor… ${Math.round(jobState.progress * 100)}%`
              : "▸ Run Backtest"}
          </Btn>
        </>
      }
      bottom={
        <>
          <span style={{ color: "var(--pos)" }}>● benchmark · loaded</span>
          <span>workers 4/4</span>
          <span>last run · —</span>
          <span style={{ flex: 1 }} />
          <span>
            cmd: <span style={{ color: "var(--fg-1)" }}>?</span> help ·{" "}
            <span style={{ color: "var(--fg-1)" }}>k</span> palette ·{" "}
            <span style={{ color: "var(--fg-1)" }}>r</span> run
          </span>
        </>
      }
      width="100%"
      height="100vh"
    >
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "320px 1fr 380px",
          overflow: "hidden",
          flex: 1,
        }}
      >
        {/* Col A: Catalog list */}
        <div
          style={{
            borderRight: "1px solid var(--line-soft)",
            display: "flex",
            flexDirection: "column",
            overflow: "hidden",
          }}
        >
          <div
            style={{
              padding: "10px 14px",
              borderBottom: "1px solid var(--line-soft)",
              display: "flex",
              alignItems: "baseline",
              gap: 8,
              background: "var(--bg-1)",
            }}
          >
            <span style={{ fontFamily: "var(--mono)", fontSize: 10, color: "var(--fg-3)" }}>§ A</span>
            <span style={{ fontFamily: "var(--serif)", fontStyle: "italic", fontSize: 12.5, color: "var(--fg-0)" }}>
              Alpha Kataloğu
            </span>
            <span style={{ flex: 1 }} />
            <span style={{ fontFamily: "var(--mono)", fontSize: 9.5, color: "var(--fg-3)" }}>
              {records.length} · sort: sharpe
            </span>
          </div>

          <div
            style={{
              padding: "8px 10px",
              display: "flex",
              gap: 6,
              borderBottom: "1px solid var(--line-soft)",
            }}
          >
            <input
              value={filterText}
              onChange={(e) => setFilterText(e.target.value)}
              placeholder="⌕  filter regex…"
              style={{
                flex: 1,
                padding: "4px 8px",
                border: "1px solid var(--line)",
                borderRadius: 2,
                fontFamily: "var(--mono)",
                fontSize: 10.5,
                color: "var(--fg-2)",
                background: "var(--bg-1)",
                outline: "none",
              }}
            />
            <SegRow
              options={["EVO", "LLM", "MCTS"]}
              value={sourceFilter}
              onChange={setSourceFilter}
            />
          </div>

          <div style={{ overflow: "auto", flex: 1 }}>
            {filteredRecords.map((f, i) => {
              const isActive = f.formula === (activeRecord?.formula ?? "");
              const sparkSeed = i + 1;
              return (
                <div
                  key={f.formula}
                  onClick={() => navigate(`/workbench?id=${encodeURIComponent(f.formula)}`)}
                  style={{
                    padding: "10px 12px",
                    borderBottom: "1px dotted var(--line-soft)",
                    background: isActive ? "var(--bg-2)" : "transparent",
                    borderLeft: isActive ? "2px solid var(--accent)" : "2px solid transparent",
                    display: "flex",
                    flexDirection: "column",
                    gap: 5,
                    cursor: "pointer",
                  }}
                >
                  <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
                    <span style={{ fontFamily: "var(--mono)", fontSize: 9.5, color: "var(--fg-3)" }}>
                      α-{String(i + 1).padStart(4, "0")}
                    </span>
                    <Pill mono tone={f.source === "evolution" || f.source === "evo" ? "accent" : "ghost"}>
                      {f.source ?? "—"}
                    </Pill>
                    <span style={{ flex: 1 }} />
                    <span
                      style={{
                        fontFamily: "var(--mono)",
                        fontSize: 11,
                        color: (f.sharpe ?? 0) > 2 ? "var(--pos)" : "var(--fg-1)",
                      }}
                    >
                      {f.sharpe != null ? f.sharpe.toFixed(2) : "—"}
                    </span>
                  </div>
                  <code
                    style={{
                      fontFamily: "var(--mono)",
                      fontSize: 10.5,
                      color: "var(--fg-0)",
                      whiteSpace: "nowrap",
                      overflow: "hidden",
                      textOverflow: "ellipsis",
                      lineHeight: 1.4,
                    }}
                  >
                    {f.formula}
                  </code>
                  <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                    <span style={{ fontFamily: "var(--mono)", fontSize: 9, color: "var(--fg-3)" }}>
                      ric {(f.rank_ic ?? 0).toFixed(4)} · adj {(f.adj_ic ?? 0).toFixed(4)} · len {f.formula.length}
                    </span>
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        {/* Col B: Selected formula + params */}
        <div style={{ display: "flex", flexDirection: "column", overflow: "hidden" }}>
          <div
            style={{
              padding: "10px 14px",
              borderBottom: "1px solid var(--line-soft)",
              display: "flex",
              alignItems: "baseline",
              gap: 8,
              background: "var(--bg-1)",
            }}
          >
            <span style={{ fontFamily: "var(--mono)", fontSize: 10, color: "var(--fg-3)" }}>§ B</span>
            <span style={{ fontFamily: "var(--serif)", fontStyle: "italic", fontSize: 12.5, color: "var(--fg-0)" }}>
              Çalışma Tezgahı
            </span>
            <span style={{ flex: 1 }} />
            <span style={{ fontFamily: "var(--mono)", fontSize: 9.5, color: "var(--fg-3)" }}>
              {formula ? "α seçili" : "—"}
            </span>
          </div>

          <div
            style={{
              overflow: "auto",
              padding: "14px 18px",
              display: "flex",
              flexDirection: "column",
              gap: 18,
            }}
          >
            {/* Formula display */}
            <div>
              <SectionLabel>FORMULA</SectionLabel>
              <div
                style={{
                  background: "var(--bg-1)",
                  border: "1px solid var(--line-soft)",
                  padding: "14px 16px",
                  borderRadius: 3,
                  fontFamily: "var(--mono)",
                  fontSize: 14,
                  lineHeight: 1.6,
                  letterSpacing: 0.2,
                  marginTop: 6,
                  wordBreak: "break-all",
                }}
              >
                {formula ? colorizeFormula(formula) : (
                  <span style={{ color: "var(--fg-3)" }}>← Katalogdan bir formül seçin</span>
                )}
              </div>
              {formula && (
                <div style={{ marginTop: 8, display: "flex", gap: 6, flexWrap: "wrap" }}>
                  <Pill mono>len {formula.length}</Pill>
                  <Pill mono>nodes {(formula.match(/\(/g) ?? []).length}</Pill>
                  <Pill mono tone="pos">+ size-corr {(activeRecord?.ic ?? 0.42).toFixed(2)}</Pill>
                </div>
              )}
            </div>

            {/* Window picker */}
            <div>
              <SectionLabel>BACKTEST PENCERESİ</SectionLabel>
              <div style={{ marginTop: 6 }}>
                <SegRow
                  options={["TEST · oos", "TRAIN · is", "TAM"]}
                  value={windowToLabel(backtestWindow)}
                  onChange={(v) => setBacktestWindow(WINDOW_MAP[v] ?? "test")}
                />
              </div>
            </div>

            {/* Mining + Neutralization */}
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 18 }}>
              <div>
                <SectionLabel>MADENCİLİK</SectionLabel>
                <div style={{ marginTop: 6, display: "flex", flexDirection: "column", gap: 6 }}>
                  <Field label="Popülasyon Büyüklüğü" type="number" value={mPopSize} onChange={(v) => setMPopSize(Number(v) || 200)} hint="pop_size" />
                  <Field label="Maksimum Uzunluk (K)" type="number" value={mMaxK} onChange={(v) => setMMaxK(Number(v) || 15)} hint="max_len" />
                  <Field label="Mining içi fold sayısı" type="number" value={mFolds} onChange={(v) => setMFolds(Number(v) || 5)} hint="wf_folds" />
                  <Field label="Fold embargo (gün)" type="number" value={mEmbargo} onChange={(v) => setMEmbargo(Number(v) || 5)} hint="embargo" />
                  <Field label="Purge horizon (gün)" type="number" value={mPurge} onChange={(v) => setMPurge(Number(v) || 10)} hint="purge" />
                </div>
              </div>
              <div>
                <SectionLabel>NÖTRALİZASYON</SectionLabel>
                <div style={{ marginTop: 6, display: "flex", flexDirection: "column", gap: 6 }}>
                  <Check label="Size / Vol / Mom" checked={neutralize} onChange={setNeutralize} />
                  <Field label="λ_std (stabilite cezası)" type="number" value={mLambdaStd} onChange={(v) => setMLambdaStd(Number(v) || 0.5)} hint="lambda_std" />
                  <Field label="λ_complexity" type="number" value={mLambdaCx} onChange={(v) => setMLambdaCx(Number(v) || 0.001)} hint="lambda_c" />
                  <Field label="λ_size" type="number" value={mLambdaSize} onChange={(v) => setMLambdaSize(Number(v) || 0.5)} hint="lambda_size" />
                  <Field label="Size-corr hard limit" type="number" value={mSizeCorr} onChange={(v) => setMSizeCorr(Number(v) || 0.7)} hint="size_lim" />
                </div>
              </div>
            </div>

            {/* Mining trigger + progress */}
            <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
              <Btn variant="primary" onClick={handleMining} disabled={isMining}>
                {miningJob.reconnecting
                  ? "Yeniden bağlanıyor…"
                  : isMining
                  ? `Çalışıyor… ${Math.round(miningJob.progress * 100)}%`
                  : "▸ Çalıştır"}
              </Btn>
              {isMining && (
                <>
                  <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
                    <div style={{ flex: 1, height: 3, background: "var(--bg-2)", borderRadius: 2, overflow: "hidden" }}>
                      <div style={{ width: `${Math.round(miningJob.progress * 100)}%`, height: "100%", background: "var(--accent)", transition: "width 0.3s" }} />
                    </div>
                    <Btn variant="ghost" onClick={miningJob.cancel} style={{ padding: "1px 6px", fontSize: 10 }}>
                      ✕
                    </Btn>
                  </div>
                  <div style={{ fontFamily: "var(--mono)", fontSize: 10, color: "var(--fg-3)", maxHeight: 60, overflow: "auto" }}>
                    {miningJob.logs.slice(-3).map((line, i) => (
                      <div key={i}>{line}</div>
                    ))}
                  </div>
                </>
              )}
              {miningJob.done && miningJob.result && (
                <div style={{ fontFamily: "var(--mono)", fontSize: 10, color: "var(--pos)" }}>
                  ✓ {(miningJob.result as any)?.accepted ?? 0} formül kabul edildi · katalog güncellendi
                </div>
              )}
              {(miningError || miningJob.error) && (
                <div style={{ fontFamily: "var(--mono)", fontSize: 10, color: "var(--neg)" }}>
                  ⚠ {miningError ?? miningJob.error}
                </div>
              )}
            </div>

          </div>
        </div>

        {/* Col C: Results */}
        <div
          style={{
            borderLeft: "1px solid var(--line-soft)",
            display: "flex",
            flexDirection: "column",
            overflow: "hidden",
            background: "var(--bg-0)",
          }}
        >
          <div
            style={{
              padding: "10px 14px",
              borderBottom: "1px solid var(--line-soft)",
              display: "flex",
              alignItems: "baseline",
              gap: 8,
              background: "var(--bg-1)",
            }}
          >
            <span style={{ fontFamily: "var(--mono)", fontSize: 10, color: "var(--fg-3)" }}>§ C</span>
            <span style={{ fontFamily: "var(--serif)", fontStyle: "italic", fontSize: 12.5, color: "var(--fg-0)" }}>
              Sonuçlar
            </span>
            <span style={{ flex: 1 }} />
            <span style={{ fontFamily: "var(--mono)", fontSize: 9.5, color: "var(--fg-3)" }}>
              {backtestWindow === "test" ? "oos · 410k" : backtestWindow === "train" ? "is · 685k" : "tam"}
            </span>
          </div>

          <div
            style={{
              overflow: "auto",
              padding: 14,
              display: "flex",
              flexDirection: "column",
              gap: 14,
            }}
          >
            {/* Job progress */}
            {(isRunning || (jobId && jobState.logs.length > 0)) && (
              <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
                <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                  <div style={{ flex: 1, height: 3, background: "var(--bg-2)", borderRadius: 2, overflow: "hidden" }}>
                    <div style={{ width: `${Math.round(jobState.progress * 100)}%`, height: "100%", background: "var(--accent)", transition: "width 0.3s" }} />
                  </div>
                  <span style={{ fontFamily: "var(--mono)", fontSize: 9.5, color: "var(--fg-3)", minWidth: 32 }}>
                    {Math.round(jobState.progress * 100)}%
                  </span>
                  {isRunning && (
                    <Btn variant="ghost" onClick={jobState.cancel} style={{ padding: "1px 6px", fontSize: 10 }}>
                      ✕
                    </Btn>
                  )}
                </div>
                {jobState.logs.slice(-3).map((line, i) => (
                  <div key={i} style={{ fontFamily: "var(--mono)", fontSize: 9.5, color: "var(--fg-3)" }}>
                    {line}
                  </div>
                ))}
                {jobState.error && (
                  <div style={{ fontFamily: "var(--mono)", fontSize: 9.5, color: "var(--neg)" }}>
                    ⚠ {jobState.error}
                  </div>
                )}
              </div>
            )}

            {/* KPIs */}
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10 }}>
              <Stat
                label="Sharpe (oos)"
                value={displayResult?.sharpe != null ? displayResult.sharpe.toFixed(2) : "—"}
                tone={displayResult?.sharpe != null && displayResult.sharpe > 1 ? "pos" : undefined}
              />
              <Stat
                label="Alpha IR"
                value={displayResult?.alpha_ir != null ? displayResult.alpha_ir.toFixed(2) : "—"}
                hint="vs benchmark"
                tone={displayResult?.alpha_ir != null && displayResult.alpha_ir > 0 ? "pos" : undefined}
              />
              <Stat
                label="Yıllık Getiri"
                value={displayResult?.annual != null ? `${displayResult.annual.toFixed(1)}%` : "—"}
                tone={displayResult?.annual != null && displayResult.annual > 0 ? "pos" : "neg"}
              />
              <Stat
                label="Max DD"
                value={displayResult?.mdd != null ? `${displayResult.mdd.toFixed(1)}%` : "—"}
                tone="neg"
              />
              <Stat
                label="IC"
                value={displayResult?.ic != null ? displayResult.ic.toFixed(3) : "—"}
                tone={displayResult?.ic != null && displayResult.ic > 0.005 ? "pos" : undefined}
              />
              <Stat
                label="Beta"
                value={displayResult?.beta != null ? displayResult.beta.toFixed(3) : "—"}
                hint="benchmark β"
              />
            </div>

            {/* Equity chart */}
            <div>
              <div style={{ fontSize: 10, fontFamily: "var(--mono)", color: "var(--fg-3)", marginBottom: 4 }}>
                EQUITY · α vs benchmark
              </div>
              <div
                style={{
                  background: "var(--bg-1)",
                  border: "1px solid var(--line-soft)",
                  borderRadius: 2,
                  padding: 6,
                }}
              >
                <EquityChart
                  width={340}
                  height={140}
                  alpha={equityAlpha}
                  bench={equityBench}
                  train={0}
                />
              </div>
            </div>

            {/* Drawdown */}
            <div>
              <div style={{ fontSize: 10, fontFamily: "var(--mono)", color: "var(--fg-3)", marginBottom: 4 }}>
                DRAWDOWN
              </div>
              <div
                style={{
                  background: "var(--bg-1)",
                  border: "1px solid var(--line-soft)",
                  borderRadius: 2,
                  padding: 6,
                }}
              >
                <DrawdownChart width={340} height={64} data={ddData} />
              </div>
            </div>

            {/* Heatmap */}
            <div>
              <div style={{ fontSize: 10, fontFamily: "var(--mono)", color: "var(--fg-3)", marginBottom: 4 }}>
                ROLLING SHARPE · WALK-FORWARD
              </div>
              <div
                style={{
                  background: "var(--bg-1)",
                  border: "1px solid var(--line-soft)",
                  borderRadius: 2,
                  padding: "6px 8px",
                }}
              >
                {foldSharpes.length === 0 ? (
                  <div style={{ fontFamily: "var(--mono)", fontSize: 10, color: "var(--fg-3)", padding: "12px 6px", textAlign: "center" }}>
                    backtest çalıştırılmadı
                  </div>
                ) : (
                  foldSharpes.map((sharpe, fi) => (
                    <HeatmapRow
                      key={`fold-${fi + 1}`}
                      label={`fold-${fi + 1}`}
                      width={250}
                      values={[sharpe]}
                    />
                  ))
                )}
              </div>
            </div>

            {/* Backtest Studio link */}
            {displayResult && (
              <div
                style={{
                  background: "var(--bg-1)",
                  border: "1px solid var(--line-soft)",
                  borderRadius: 2,
                  padding: 10,
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "space-between",
                }}
              >
                <div>
                  <div style={{ fontFamily: "var(--mono)", fontSize: 10, color: "var(--fg-2)" }}>Derin Validasyon</div>
                  <div style={{ fontFamily: "var(--mono)", fontSize: 9, color: "var(--fg-3)", marginTop: 2 }}>
                    DSR · PBO · Rolling WF · Ensemble
                  </div>
                </div>
                <Btn
                  variant="ghost"
                  onClick={() => navigate(`/backtest-studio?formula=${encodeURIComponent(formula)}`)}
                >
                  → Backtest Studio
                </Btn>
              </div>
            )}

            {(jobState.error || launchError) && !isRunning && (
              <div
                style={{
                  padding: 10,
                  background: "var(--bg-1)",
                  border: "1px solid var(--neg)",
                  borderRadius: 3,
                  fontFamily: "var(--mono)",
                  fontSize: 11,
                  color: "var(--neg)",
                }}
              >
                ⚠ {jobState.error ?? launchError}
              </div>
            )}
          </div>
        </div>
      </div>
    </CChrome>
  );
}
