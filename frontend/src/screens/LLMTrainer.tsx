import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { CChrome } from "../components/chrome/CChrome";
import { Panel } from "../components/chrome/Panel";
import { Box } from "../components/chrome/Box";
import { Btn } from "../components/atoms/Btn";
import { Pill } from "../components/atoms/Pill";
import { Stat } from "../components/atoms/Stat";
import { Check } from "../components/atoms/Check";
import { Note } from "../components/inputs/Note";
import { Stepper } from "../components/inputs/Stepper";
import { MiniSparkline } from "../components/charts/MiniSparkline";
import { apiFetch } from "../lib/api";
import { useJob } from "../hooks/useJob";
import { useMeta, formatMetaForChrome } from "../hooks/useMeta";

const SAMPLE = `Rank(Mul(Sub(Pclose, Popen), Vlot), 20)
CSRank(Delta(Pvwap, 30))
Corr(Pclose, Vlot, 40)
Div(0.05, Std(Pclose, 20))`;

const OPERATORS = [
  { k: "arithmetic", v: "Abs · Sign · Log · Add · Mul · Greater · Less · Div · Pow · Sub" },
  { k: "rolling", v: "Rank · WMA · EMA · Ref · Mean · Sum · Std · Var · Skew · Kurt · Max · Min · Med · Mad · Delta" },
  { k: "paired", v: "Corr · Cov" },
  { k: "cross-section", v: "CSRank" },
  { k: "features", v: "Popen · Phigh · Plow · Pclose · Vlot · Pvwap" },
];

interface ParseResult {
  formula: string;
  ok: boolean;
  ic?: number;
  rank_ic?: number;
  error?: string;
}

interface BufferStatus {
  size: number;
  capacity: number;
}

export default function LLMTrainer() {
  const [text, setText] = useState(SAMPLE);
  const [wfFitness, setWfFitness] = useState(false);
  const [epochs, setEpochs] = useState(5);
  const [results, setResults] = useState<ParseResult[]>([]);
  const [parseLoading, setParseLoading] = useState(false);
  const [lastRun, setLastRun] = useState<string | null>(null);

  // Training job state
  const [trainJobId, setTrainJobId] = useState<string | null>(null);
  const [isLaunching, setIsLaunching] = useState(false);
  const trainJob = useJob(trainJobId);
  const isTraining = isLaunching || (!!trainJobId && !trainJob.done);

  // Buffer size (auto-refresh every 5s, more frequent after parse/train)
  const { data: bufStatus, refetch: refetchBuf } = useQuery<BufferStatus>({
    queryKey: ["training-buffer"],
    queryFn: () => apiFetch("/api/training/buffer"),
    refetchInterval: 5000,
  });

  const handleParse = async () => {
    const formulas = text.split("\n").map((l) => l.trim()).filter(Boolean);
    if (!formulas.length) return;
    setParseLoading(true);
    try {
      const res = await apiFetch<ParseResult[]>("/api/backtest/parse-multi", {
        method: "POST",
        body: JSON.stringify({ formulas, window: "test", wf_fitness: wfFitness }),
      });
      setResults(res);
      setLastRun(`${formulas.length} formül`);
      refetchBuf();
    } catch (e: any) {
      setResults([{ formula: "?", ok: false, error: e.message }]);
    }
    setParseLoading(false);
  };

  const [trainLaunchError, setTrainLaunchError] = useState<string | null>(null);

  const handleTrain = async () => {
    if (isLaunching) return;
    setIsLaunching(true);
    setTrainJobId(null);
    setTrainLaunchError(null);
    try {
      const { job_id } = await apiFetch<{ job_id: string }>("/api/training/run", {
        method: "POST",
        body: JSON.stringify({ epochs, batch_size: 32, use_policy: false }),
      });
      setTrainJobId(job_id);
    } catch (e: any) {
      setTrainLaunchError(e?.message ?? "Eğitim başlatılamadı");
    } finally {
      setIsLaunching(false);
    }
  };

  // When training job finishes, refresh buffer count
  if (trainJob.done && trainJobId) {
    refetchBuf();
  }

  const trainResult = trainJob.result as Record<string, any> | null;
  const lossCurve: number[] = trainResult?.loss_curve ?? [];

  const bufferSize = bufStatus?.size ?? 0;
  const lastLoss = trainResult?.last_loss;
  const epochsDone = trainResult?.epochs_done;
  const { data: metaData } = useMeta();
  const chromeMeta = formatMetaForChrome(metaData);

  return (
    <CChrome
      title="llm → tree-lstm"
      sub="harici LLM formüllerini öğret"
      meta={chromeMeta}
      top={
        <>
          <Pill mono tone={bufferSize > 0 ? "pos" : "ghost"}>
            replay buffer · {bufferSize}
          </Pill>
          <Btn variant="ghost" onClick={handleTrain} disabled={isTraining || bufferSize < 2}>
            {isTraining
              ? `Eğitiliyor… ${Math.round(trainJob.progress * 100)}%`
              : "↺ Tree-LSTM Eğit"}
          </Btn>
          <Btn variant="primary" onClick={handleParse} disabled={parseLoading}>
            {parseLoading ? "Hesaplanıyor…" : "▸ Parse & Değerlendir"}
          </Btn>
        </>
      }
      width="100%"
      height="100vh"
    >
      <div style={{ display: "grid", gridTemplateColumns: "1.4fr 1fr", flex: 1, minHeight: 0, overflow: "hidden" }}>

        {/* Sol: paste + operatörler */}
        <div style={{ borderRight: "1px solid var(--line-soft)", display: "flex", flexDirection: "column", minHeight: 0, overflow: "hidden" }}>
          <Panel num="A" title="Formüller" sub="her satır bir tane" flex={1}>
            <Note cite="Workflow">
              Harici LLM'den (ChatGPT, Claude, Gemini vb.) aldığın formülleri her satıra bir tane olacak şekilde
              yapıştır. Sistem parse edip IC hesaplayacak ve replay buffer'a ekleyecek. Sonra{" "}
              <em style={{ color: "var(--fg-0)" }}>Tree-LSTM Eğit</em> butonuna bas.
            </Note>

            <textarea
              value={text}
              onChange={(e) => setText(e.target.value)}
              rows={10}
              style={{
                width: "100%",
                background: "var(--bg-2)",
                border: "1px solid var(--line)",
                borderRadius: 2,
                padding: "8px 10px",
                fontFamily: "var(--mono)",
                fontSize: 11.5,
                color: "var(--fg-0)",
                lineHeight: 1.6,
                resize: "vertical",
                outline: "none",
              }}
            />

            <div style={{ display: "flex", alignItems: "center", gap: 14, marginTop: 12 }}>
              <Check label="LLM formüller için WF-Fitness hesapla" hint="daha yavaş · fold=5" checked={wfFitness} onChange={setWfFitness} />
              <span style={{ flex: 1 }} />
              <Btn variant="ghost" onClick={() => setText(SAMPLE)}>Örnekleri yükle</Btn>
              <Btn variant="primary" onClick={handleParse} disabled={parseLoading}>
                {parseLoading ? "…" : "▸ Parse & Değerlendir"}
              </Btn>
            </div>

            <div style={{ marginTop: 18 }}>
              <div style={{ fontSize: 10, fontFamily: "var(--mono)", color: "var(--fg-3)", letterSpacing: 0.5, marginBottom: 6 }}>
                GEÇERLİ OPERATÖRLER
              </div>
              <Box>
                {OPERATORS.map((r) => (
                  <div
                    key={r.k}
                    style={{
                      display: "grid",
                      gridTemplateColumns: "110px 1fr",
                      gap: 12,
                      padding: "4px 0",
                      borderBottom: "1px dotted var(--line-soft)",
                    }}
                  >
                    <span style={{ fontFamily: "var(--mono)", fontSize: 10, color: "var(--fg-3)" }}>{r.k}</span>
                    <span style={{ fontFamily: "var(--mono)", fontSize: 11, color: "var(--fg-0)" }}>{r.v}</span>
                  </div>
                ))}
              </Box>
            </div>
          </Panel>
        </div>

        {/* Sağ: parse sonuçları + training stats */}
        <div style={{ display: "flex", flexDirection: "column", minHeight: 0, overflow: "hidden" }}>
          <Panel num="B" title="Parse Sonuçları" sub={lastRun ? `${results.length} formül · son çalıştırma` : "henüz çalıştırılmadı"} flex={1}>
            {results.length === 0 ? (
              <div style={{ fontFamily: "var(--mono)", fontSize: 11, color: "var(--fg-3)" }}>
                Formülleri girin ve ▸ Parse & Değerlendir butonuna basın.
              </div>
            ) : (
              <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
                {results.map((r, i) => (
                  <div
                    key={i}
                    style={{
                      display: "grid",
                      gridTemplateColumns: "1fr 70px 70px 70px",
                      gap: 10,
                      alignItems: "center",
                      padding: "8px 10px",
                      background: "var(--bg-1)",
                      border: `1px solid ${r.ok ? "var(--line-soft)" : "var(--neg)"}`,
                      borderRadius: 2,
                    }}
                  >
                    <code style={{ fontFamily: "var(--mono)", fontSize: 11, color: "var(--fg-0)", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                      {r.formula}
                    </code>
                    {r.ok ? (
                      <>
                        <span style={{ fontFamily: "var(--mono)", fontSize: 11, textAlign: "right", color: (r.ic ?? 0) > 0 ? "var(--pos)" : "var(--neg)" }}>
                          {r.ic != null ? `${r.ic >= 0 ? "+" : ""}${r.ic.toFixed(4)}` : "—"}
                        </span>
                        <span style={{ fontFamily: "var(--mono)", fontSize: 11, textAlign: "right", color: "var(--fg-2)" }}>
                          ric {r.rank_ic != null ? r.rank_ic.toFixed(4) : "—"}
                        </span>
                        <Pill mono tone={(r.ic ?? 0) > 0.005 ? "pos" : "ghost"}>
                          {(r.ic ?? 0) > 0.005 ? "buffer+" : "low ic"}
                        </Pill>
                      </>
                    ) : (
                      <span style={{ gridColumn: "2 / 5", fontFamily: "var(--mono)", fontSize: 10, color: "var(--neg)" }}>
                        ⚠ {r.error}
                      </span>
                    )}
                  </div>
                ))}
              </div>
            )}

            {/* Training progress */}
            {isTraining && (
              <div style={{ marginTop: 14 }}>
                <div style={{ height: 3, background: "var(--bg-2)", borderRadius: 2, overflow: "hidden", marginBottom: 6 }}>
                  <div style={{ width: `${Math.round(trainJob.progress * 100)}%`, height: "100%", background: "var(--accent)", transition: "width 0.3s" }} />
                </div>
                <div style={{ fontFamily: "var(--mono)", fontSize: 10, color: "var(--fg-3)" }}>
                  {trainJob.logs[trainJob.logs.length - 1] ?? "başlatılıyor…"}
                </div>
              </div>
            )}

            {(trainJob.error || trainLaunchError) && (
              <div style={{ marginTop: 10, fontFamily: "var(--mono)", fontSize: 10, color: "var(--neg)" }}>
                ⚠ {trainJob.error ?? trainLaunchError}
              </div>
            )}

            <div style={{ marginTop: 18 }}>
              <div style={{ fontSize: 10, fontFamily: "var(--mono)", color: "var(--fg-3)", letterSpacing: 0.5, marginBottom: 6 }}>
                TREE-LSTM · TRAINING STATS
              </div>
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 14 }}>
                <Stat label="Buffer" value={bufferSize > 0 ? bufferSize.toLocaleString("tr-TR") : "—"} hint="formül" />
                <Stat
                  label="Last loss"
                  value={lastLoss != null ? lastLoss.toFixed(4) : "—"}
                  hint={epochsDone != null ? `epoch ${epochsDone}` : "—"}
                />
                <Stat label="Embedding dim" value="64" />
                <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
                  <Stepper label="Epoch" value={epochs} onChange={setEpochs} min={1} max={100} />
                  <Btn mono small full onClick={handleTrain} disabled={isTraining || bufferSize < 2}>
                    {isTraining ? `${Math.round(trainJob.progress * 100)}%` : "↺ Eğit"}
                  </Btn>
                </div>
              </div>
              <div style={{ marginTop: 10 }}>
                <Box p={8}>
                  <div style={{ fontFamily: "var(--mono)", fontSize: 10, color: "var(--fg-3)", marginBottom: 4 }}>
                    LOSS · {lossCurve.length > 0 ? `son ${lossCurve.length} epoch` : "henüz eğitilmedi"}
                  </div>
                  {lossCurve.length > 0 ? (
                    <MiniSparkline data={lossCurve} width={300} height={36} color="var(--accent)" fill />
                  ) : (
                    <div style={{ height: 36, display: "flex", alignItems: "center", justifyContent: "center", fontFamily: "var(--mono)", fontSize: 10, color: "var(--fg-3)" }}>
                      eğitim çalıştırılınca dolar
                    </div>
                  )}
                </Box>
              </div>
            </div>
          </Panel>
        </div>
      </div>
    </CChrome>
  );
}
