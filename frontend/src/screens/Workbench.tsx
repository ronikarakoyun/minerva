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
import { MiniSparkline } from "../components/charts/MiniSparkline";
import { EquityChart } from "../components/charts/EquityChart";
import { DrawdownChart } from "../components/charts/DrawdownChart";
import { HeatmapRow } from "../components/charts/HeatmapRow";
import { useCatalog } from "../hooks/useCatalog";
import { useJob } from "../hooks/useJob";
import { apiFetch } from "../lib/api";
import { EvaluateResult } from "../types";

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

function equityCurve(seed = 1, n = 200, drift = 0.0008, vol = 0.012): number[] {
  let x = seed * 9301 + 49297;
  const rnd = () => {
    x = (x * 9301 + 49297) % 233280;
    return x / 233280 - 0.5;
  };
  const out = [1];
  for (let i = 1; i < n; i++) out.push(out[i - 1] * (1 + drift + rnd() * vol));
  return out;
}

const MINING_PARAMS = [
  { k: "Popülasyon Büyüklüğü", v: "300", hint: "pop_size" },
  { k: "Maksimum Uzunluk (K)", v: "15", hint: "max_len" },
  { k: "Mining içi fold sayısı", v: "5", hint: "wf_folds" },
  { k: "Fold embargo (gün)", v: "5", hint: "embargo" },
  { k: "Purge horizon (gün)", v: "10", hint: "purge" },
  { k: "λ_std (stabilite cezası)", v: "0.50", hint: "lambda_std" },
  { k: "λ_complexity", v: "0.001", hint: "lambda_c" },
  { k: "λ_size", v: "0.50", hint: "lambda_size" },
  { k: "Size-corr hard limit", v: "0.70", hint: "size_lim" },
];

export default function Workbench() {
  const [searchParams] = useSearchParams();
  const navigate = useNavigate();
  const activeId = searchParams.get("id") ?? "";

  const { data: records = [] } = useCatalog();

  const [backtestWindow, setBacktestWindow] = useState<"test" | "train" | "all">("test");
  const [filterText, setFilterText] = useState("");
  const [sourceFilter, setSourceFilter] = useState("EVO");
  const [neutralize, setNeutralize] = useState(true);
  const [validations, setValidations] = useState({
    meta_label: true,
    dsr: true,
    pbo: true,
    rolling_wf: false,
    ensemble: false,
    time_overfit: false,
  });

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

  const handleRun = async () => {
    if (!formula || isLaunching) return;
    setIsLaunching(true);
    setJobId(null);
    try {
      const { job_id } = await apiFetch<{ job_id: string }>("/api/backtest/run", {
        method: "POST",
        body: JSON.stringify({ formula, window: backtestWindow, validations }),
      });
      setJobId(job_id);
    } catch (e) {
      console.error(e);
    } finally {
      setIsLaunching(false);
    }
  };

  const windowLabel: Record<string, "test" | "train" | "all"> = {
    "TEST · oos": "test",
    "TRAIN · is": "train",
    "TAM": "all",
  };

  const isRunning = isLaunching || (!!jobId && !jobState.done);

  // Demo data for when no real result is available
  const demoAlpha = equityCurve(1, 200, 0.0014, 0.011);
  const demoBench = equityCurve(99, 200, 0.0004, 0.009);
  const demoDD = Array.from({ length: 200 }, (_, i) =>
    -Math.abs(Math.sin(i / 12)) * Math.min(0.12, i * 0.0007)
  );

  const displayResult: EvaluateResult | null = jobState.result;
  const equityAlpha = displayResult?.equity_curve ?? demoAlpha;
  const equityBench = demoBench;
  const ddData = displayResult?.drawdown ?? demoDD;
  const trainSplit = displayResult?.train_split ?? 126;
  const foldSharpes = displayResult?.fold_sharpes ?? [1.8, 2.1, 2.4, 1.9, 2.3];

  return (
    <CChrome
      title="workbench"
      sub="variant-c"
      meta={{
        train: "685,367 (63%)",
        test: "410,452 (37%)",
        split: "2023-04-18",
        benchmark: "CSI 500",
      }}
      top={
        <>
          <Btn variant="ghost" onClick={() => navigate("/catalog")}>← Katalog</Btn>
          <Btn variant="primary" onClick={handleRun} disabled={isRunning || !formula}>
            {isRunning ? `Çalışıyor… ${Math.round(jobState.progress * 100)}%` : "▸ Run Backtest"}
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
                      α-{String(2741 - i * 7).padStart(4, "0")}
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
                    <MiniSparkline
                      data={equityCurve(sparkSeed, 50, 0.001 + i * 0.0002, 0.012)}
                      width={140}
                      height={14}
                      color={isActive ? "var(--accent)" : "var(--fg-3)"}
                    />
                    <span style={{ flex: 1 }} />
                    <span style={{ fontFamily: "var(--mono)", fontSize: 9, color: "var(--fg-3)" }}>
                      ic {(f.ic ?? 0).toFixed(3)} · len {String(f.formula).split(/\b/).length}
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
                  value={
                    backtestWindow === "test" ? "TEST · oos" : backtestWindow === "train" ? "TRAIN · is" : "TAM"
                  }
                  onChange={(v) => setBacktestWindow(windowLabel[v] ?? "test")}
                />
              </div>
            </div>

            {/* Mining + Neutralization */}
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 18 }}>
              <div>
                <SectionLabel>MADENCİLİK</SectionLabel>
                <div style={{ marginTop: 6, display: "flex", flexDirection: "column", gap: 6 }}>
                  {MINING_PARAMS.slice(0, 5).map((p) => (
                    <Field key={p.k} label={p.k} value={p.v} />
                  ))}
                </div>
              </div>
              <div>
                <SectionLabel>NÖTRALİZASYON</SectionLabel>
                <div style={{ marginTop: 6, display: "flex", flexDirection: "column", gap: 6 }}>
                  <Check label="Size / Vol / Mom" checked={neutralize} onChange={setNeutralize} />
                  {MINING_PARAMS.slice(5, 9).map((p) => (
                    <Field key={p.k} label={p.k} value={p.v} />
                  ))}
                </div>
              </div>
            </div>

            {/* Validation */}
            <div>
              <SectionLabel>DOĞRULAMA</SectionLabel>
              <div style={{ marginTop: 6, display: "grid", gridTemplateColumns: "1fr 1fr", gap: 4 }}>
                <Check
                  label="Meta-Label"
                  checked={validations.meta_label}
                  onChange={(v) => setValidations((p) => ({ ...p, meta_label: v }))}
                  hint="ikincil güven filtresi"
                />
                <Check
                  label="DSR"
                  checked={validations.dsr}
                  onChange={(v) => setValidations((p) => ({ ...p, dsr: v }))}
                  hint="deflated sharpe"
                />
                <Check
                  label="PBO (CSCV)"
                  checked={validations.pbo}
                  onChange={(v) => setValidations((p) => ({ ...p, pbo: v }))}
                  hint="overfit olasılığı"
                />
                <Check
                  label="Rolling WF"
                  checked={validations.rolling_wf}
                  onChange={(v) => setValidations((p) => ({ ...p, rolling_wf: v }))}
                  hint="window=252 step=21"
                />
                <Check
                  label="Ensemble"
                  checked={validations.ensemble}
                  onChange={(v) => setValidations((p) => ({ ...p, ensemble: v }))}
                  hint="çoklu alpha"
                />
                <Check
                  label="Time-based overfit"
                  checked={validations.time_overfit}
                  onChange={(v) => setValidations((p) => ({ ...p, time_overfit: v }))}
                  hint="zaman validasyonu"
                />
              </div>
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
                value={displayResult ? displayResult.sharpe.toFixed(2) : "—"}
                tone={displayResult && displayResult.sharpe > 1 ? "pos" : undefined}
              />
              <Stat
                label="DSR"
                value={displayResult ? displayResult.dsr.toFixed(2) : "—"}
                hint="p < 0.01"
                tone={displayResult && displayResult.dsr > 0.5 ? "pos" : undefined}
              />
              <Stat
                label="Ann. ret"
                value={displayResult ? `${(displayResult.ann_ret * 100).toFixed(1)}%` : "—"}
                tone={displayResult && displayResult.ann_ret > 0 ? "pos" : undefined}
              />
              <Stat
                label="Max DD"
                value={displayResult ? `${(displayResult.max_dd * 100).toFixed(1)}%` : "—"}
                tone="neg"
              />
              <Stat label="IC" value={displayResult ? displayResult.ic.toFixed(3) : "—"} />
              <Stat
                label="Turnover"
                value={displayResult ? displayResult.turnover.toFixed(2) : "—"}
                hint="daily"
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
                  train={trainSplit}
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
                {["fold-1", "fold-2", "fold-3", "fold-4", "fold-5"].map((lbl, fi) => (
                  <HeatmapRow
                    key={lbl}
                    label={lbl}
                    width={250}
                    values={
                      foldSharpes.length > fi
                        ? Array.from({ length: 24 }, (_, j) =>
                            Math.sin((j + fi * 3) / 4) * (foldSharpes[fi] ?? 1.5) +
                            (Math.random() - 0.5) * 0.3
                          )
                        : Array.from({ length: 24 }, (_, j) =>
                            Math.sin((j + fi * 3) / 4) * 1.5
                          )
                    }
                  />
                ))}
              </div>
            </div>

            {/* PBO */}
            <div
              style={{
                background: "var(--bg-1)",
                border: "1px solid var(--line-soft)",
                borderRadius: 2,
                padding: 10,
              }}
            >
              <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 6 }}>
                <span style={{ fontSize: 11, color: "var(--fg-1)" }}>PBO · Backtest Overfit</span>
                <span style={{ fontFamily: "var(--mono)", fontSize: 11, color: "var(--pos)" }}>0.18</span>
              </div>
              <div style={{ height: 4, background: "var(--bg-2)", borderRadius: 1, overflow: "hidden" }}>
                <div style={{ height: "100%", width: "18%", background: "var(--pos)" }} />
              </div>
              <div style={{ marginTop: 4, fontFamily: "var(--mono)", fontSize: 9, color: "var(--fg-3)" }}>
                düşük = sağlıklı · threshold 0.50
              </div>
            </div>

            {jobState.error && !isRunning && (
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
                ⚠ {jobState.error}
              </div>
            )}
          </div>
        </div>
      </div>
    </CChrome>
  );
}
