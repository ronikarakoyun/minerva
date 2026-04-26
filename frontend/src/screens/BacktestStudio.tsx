import { useState } from "react";
import { CChrome } from "../components/chrome/CChrome";
import { Box } from "../components/chrome/Box";
import { Btn } from "../components/atoms/Btn";
import { Pill } from "../components/atoms/Pill";
import { Stat } from "../components/atoms/Stat";
import { SegRow } from "../components/atoms/SegRow";
import { SectionLabel } from "../components/atoms/SectionLabel";
import { Check } from "../components/atoms/Check";
import { Note } from "../components/inputs/Note";
import { Select } from "../components/inputs/Select";
import { Stepper } from "../components/inputs/Stepper";
import { Input } from "../components/inputs/Input";
import { apiFetch } from "../lib/api";
import { useCatalog } from "../hooks/useCatalog";
import { useMeta, formatMetaForChrome } from "../hooks/useMeta";
import { WINDOW_MAP, windowToLabel } from "../lib/window";

const ROLLING_MODES = [
  { t: "Mod 1 — Anchored", d: "formül + sinyal sabit, test kaydır", mode: 1 },
  { t: "Mod 2 — Rolling Re-fit", d: "her pencerede yeniden nötralize + size_corr", mode: 2 },
  { t: "Mod 3 — Full Discovery", d: "her pencerede yeniden keşif = Hall of Fame", mode: 3 },
];

export default function BacktestStudio() {
  const { data: records = [] } = useCatalog();
  const { data: metaData } = useMeta();
  const chromeMeta = formatMetaForChrome(metaData);
  const top = [...records].sort((a, b) => (b.rank_ic ?? 0) - (a.rank_ic ?? 0));
  const [formula, setFormula] = useState("");
  const [window, setWindow] = useState<"test" | "train" | "all">("test");

  // DSR state
  const [dsrN, setDsrN] = useState(20);
  const [dsrNTrials, setDsrNTrials] = useState(0);
  const [dsrResult, setDsrResult] = useState<any>(null);
  const [dsrLoading, setDsrLoading] = useState(false);

  // PBO state
  const [pboSplits, setPboSplits] = useState(8);
  const [pboMaxCombo, setPboMaxCombo] = useState(500);
  const [pboResult, setPboResult] = useState<any>(null);
  const [pboLoading, setPboLoading] = useState(false);

  // Rolling WF state
  const [wfMode, setWfMode] = useState(1);
  const [wfTestMonths, setWfTestMonths] = useState(6);
  const [wfMinTrain, setWfMinTrain] = useState(18);
  const [wfResult, setWfResult] = useState<any>(null);
  const [wfLoading, setWfLoading] = useState(false);

  // Ensemble state
  const [ensembleText, setEnsembleText] = useState(
    `Corr(Sub(-0.01, Popen), Mul(Vlot, 0.01), 20)\nCSRank(Sub(-0.01, Skew(Pclose, 20)))\nSub(-0.1, Greater(Delta(Abs(Vlot), 20), Mul(Sign(Plow), -0.1)))`
  );
  const [ensWeighting, setEnsWeighting] = useState("Eşit ağırlık");
  const [ensWindow, setEnsWindow] = useState("TEST");
  const [ensMaxCorr, setEnsMaxCorr] = useState(0.70);
  const [ensResult, setEnsResult] = useState<any>(null);
  const [ensLoading, setEnsLoading] = useState(false);

  // Overfit state
  const [ovfMode, setOvfMode] = useState("Walk-Forward (güvenilir)");
  const [ovfNTop, setOvfNTop] = useState(10);
  const [ovfFolds, setOvfFolds] = useState(5);
  const [ovfResult, setOvfResult] = useState<any>(null);
  const [ovfLoading, setOvfLoading] = useState(false);

  const activeFormula = formula || (top[0]?.formula ?? "");

  const handleRunAll = () => {
    if (!activeFormula) return;
    handleDSR();
    handlePBO();
    handleWF();
    handleOverfit();
  };

  const handleDSR = async () => {
    if (!activeFormula) return;
    setDsrLoading(true);
    try { setDsrResult(await apiFetch("/api/backtest/dsr", { method: "POST", body: JSON.stringify({ formula: activeFormula, window, n_trials: dsrNTrials }) })); }
    catch (e: any) { setDsrResult({ error: e.message }); }
    setDsrLoading(false);
  };

  const handlePBO = async () => {
    if (!activeFormula) return;
    setPboLoading(true);
    try { setPboResult(await apiFetch("/api/backtest/pbo", { method: "POST", body: JSON.stringify({ formula: activeFormula, window, n_splits: pboSplits, max_combinations: pboMaxCombo }) })); }
    catch (e: any) { setPboResult({ error: e.message }); }
    setPboLoading(false);
  };

  const handleWF = async () => {
    if (!activeFormula) return;
    setWfLoading(true);
    try { setWfResult(await apiFetch("/api/backtest/rolling-wf", { method: "POST", body: JSON.stringify({ formula: activeFormula, test_window_months: wfTestMonths, min_train_months: wfMinTrain, window, mode: wfMode }) })); }
    catch (e: any) { setWfResult({ error: e.message }); }
    setWfLoading(false);
  };

  const handleEnsemble = async () => {
    const formulas = ensembleText.split("\n").map((l) => l.trim()).filter(Boolean);
    if (!formulas.length) return;
    setEnsLoading(true);
    try { setEnsResult(await apiFetch("/api/backtest/ensemble", { method: "POST", body: JSON.stringify({ formulas, window: WINDOW_MAP[ensWindow] ?? "test", weighting: "equal", max_corr: ensMaxCorr }) })); }
    catch (e: any) { setEnsResult({ error: e.message }); }
    setEnsLoading(false);
  };

  const handleOverfit = async () => {
    setOvfLoading(true);
    try { setOvfResult(await apiFetch("/api/backtest/overfit", { method: "POST", body: JSON.stringify({ n_top: ovfNTop, n_folds: ovfFolds, mode: ovfMode.includes("Walk") ? "walk_forward" : "single_split" }) })); }
    catch (e: any) { setOvfResult({ error: e.message }); }
    setOvfLoading(false);
  };

  return (
    <CChrome
      title="backtest studio"
      sub={activeFormula ? "α seçili · doğrulama modülleri" : "formül seçin"}
      meta={chromeMeta}
      top={
        <>
          <Pill mono tone="accent">{activeFormula ? "α seçili" : "—"}</Pill>
          <Btn variant="primary" onClick={handleRunAll} disabled={!activeFormula}>▸ Tüm Modülleri Koştur</Btn>
        </>
      }
      width="100%"
      height="100vh"
    >
      <div style={{ flex: 1, overflow: "auto", padding: "18px 22px", display: "flex", flexDirection: "column", gap: 22 }}>

        {/* Formula picker */}
        <Box>
          <div style={{ display: "grid", gridTemplateColumns: "1fr 280px", gap: 18, alignItems: "center" }}>
            <div>
              <SectionLabel>SİMÜLASYON İÇİN SEÇ</SectionLabel>
              <div style={{ marginTop: 6 }}>
                <Select
                  value={formula || (top[0]?.formula ?? "—")}
                  options={top.map((r) => r.formula)}
                  onChange={setFormula}
                />
              </div>
              {activeFormula && (
                <code style={{ fontFamily: "var(--mono)", fontSize: 12, color: "var(--fg-0)", display: "block", marginTop: 6, whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>
                  {activeFormula}
                </code>
              )}
            </div>
            <div>
              <SectionLabel>BACKTEST PENCERESİ</SectionLabel>
              <div style={{ marginTop: 6 }}>
                <SegRow
                  options={["TEST · oos", "TRAIN · is", "TAM"]}
                  value={window === "test" ? "TEST · oos" : window === "train" ? "TRAIN · is" : "TAM"}
                  onChange={(v) => setWindow(WINDOW_MAP[v] ?? "test")}
                />
              </div>
            </div>
          </div>
        </Box>

        {/* 2-col grid for modules */}
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 18 }}>

          {/* § 01 Meta-Label */}
          <Box>
            <ModuleHeader n="01" title="Meta-Label — İkincil Güven Filtresi" />
            <Note cite="López de Prado AFML §3.6">
              Primary formül sinyaline ek olarak logistic regression meta-model eğitir. P(TB_Label=1) &lt; threshold
              olan günlerde pozisyon açılmaz → daha temiz sinyal.
            </Note>
            <Check label="TB_Label hesaplanmış" checked hint="sol panel → Hedef Değişkeni" />
            <Check label="Meta-Label aktif" checked hint="sidebar'dan" />
            <div style={{ marginTop: 10, display: "flex", gap: 14 }}>
              <Stat label="Threshold" value="0.55" />
              <Stat label="Kept trades" value="— / —" />
              <Stat label="F1" value="—" />
            </div>
          </Box>

          {/* § 02 DSR */}
          <Box>
            <ModuleHeader n="02" title="Deflated Sharpe Ratio (DSR)" />
            <Note cite="Bailey & López de Prado (2014)">
              Mining havuzunda N formül denendiğinde en iyinin Sharpe Ratio'su kısmen şans eseridir. DSR
              havuz büyüklüğünü, skewness ve kurtosis'i hesaba katarak SR'ı deflate eder.
              <span style={{ display: "block", marginTop: 4, color: "var(--fg-2)" }}>
                p_value ≥ 0.95 → istatistiksel olarak anlamlı (tek taraflı α=5%).
              </span>
            </Note>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr auto", gap: 12, alignItems: "flex-end" }}>
              <Stepper label="Kaç formül için DSR hesaplansın?" value={dsrN} onChange={setDsrN} min={1} max={362} step={5} />
              <Stepper label="Toplam havuz (n_trials)" value={dsrNTrials} onChange={setDsrNTrials} min={0} max={1000} step={50} hint="0 = otomatik" />
              <Btn mono small onClick={handleDSR} disabled={dsrLoading}>▸ {dsrLoading ? "…" : "DSR Hesapla"}</Btn>
            </div>
            {dsrResult && (
              <div style={{ marginTop: 10, display: "flex", gap: 14, flexWrap: "wrap" }}>
                {dsrResult.error ? (
                  <span style={{ fontFamily: "var(--mono)", fontSize: 10, color: "var(--neg)" }}>⚠ {dsrResult.error}</span>
                ) : (
                  <>
                    <Stat label="Obs Sharpe" value={dsrResult.sharpe_obs?.toFixed(2) ?? "—"} />
                    <Stat label="DSR z-score" value={dsrResult.dsr?.toFixed(3) ?? "—"} tone={dsrResult.dsr > 1.65 ? "pos" : "neg"} />
                    <Stat label="p-value" value={dsrResult.p_value?.toFixed(3) ?? "—"} tone={dsrResult.p_value > 0.95 ? "pos" : undefined} hint="≥ 0.95" />
                  </>
                )}
              </div>
            )}
          </Box>

          {/* § 03 PBO */}
          <Box>
            <ModuleHeader n="03" title="PBO — Probability of Backtest Overfitting (CSCV)" />
            <Note cite="Bailey, LdP & Zhu (2014)">
              Mining havuzunu M zaman dilimine böler. Her C(M, M/2) kombinasyonunda IS'te en iyi formülün OOS rank'i ölçülür.
              <span style={{ display: "block", marginTop: 4, color: "var(--fg-2)" }}>
                PBO &lt; 0.5 → düşük overfit · PBO ≥ 0.5 → IS seçim sürecinde overfit.
              </span>
            </Note>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr auto", gap: 12, alignItems: "flex-end" }}>
              <Stepper label="Zaman dilimi sayısı (M)" value={pboSplits} onChange={setPboSplits} min={4} max={20} step={2} />
              <Stepper label="Maks kombinasyon" value={pboMaxCombo} onChange={setPboMaxCombo} min={100} max={2000} step={100} />
              <Btn mono small onClick={handlePBO} disabled={pboLoading}>▸ {pboLoading ? "…" : "PBO Hesapla"}</Btn>
            </div>
            {pboResult && (
              <div style={{ marginTop: 10 }}>
                {pboResult.error ? (
                  <span style={{ fontFamily: "var(--mono)", fontSize: 10, color: "var(--neg)" }}>⚠ {pboResult.error}</span>
                ) : (
                  <div style={{ display: "flex", alignItems: "baseline", gap: 12 }}>
                    <span style={{ fontFamily: "var(--mono)", fontSize: 30, color: pboResult.pbo < 0.5 ? "var(--pos)" : "var(--neg)", letterSpacing: -1 }}>
                      {pboResult.pbo?.toFixed(2) ?? "—"}
                    </span>
                    <div style={{ flex: 1 }}>
                      <div style={{ height: 4, background: "var(--bg-2)", borderRadius: 1, overflow: "hidden", position: "relative" }}>
                        <div style={{ height: "100%", width: `${(pboResult.pbo ?? 0) * 100}%`, background: pboResult.pbo < 0.5 ? "var(--pos)" : "var(--neg)" }} />
                        <div style={{ position: "absolute", left: "50%", top: -3, width: 1, height: 10, background: "var(--fg-3)" }} />
                      </div>
                      <div style={{ display: "flex", justifyContent: "space-between", marginTop: 3, fontFamily: "var(--mono)", fontSize: 9, color: "var(--fg-3)" }}>
                        <span>0 healthy</span><span>0.5</span><span>1.0 overfit</span>
                      </div>
                      <div style={{ fontFamily: "var(--mono)", fontSize: 9.5, color: "var(--fg-3)", marginTop: 4 }}>
                        {pboResult.n_combinations} kombinasyon
                      </div>
                    </div>
                  </div>
                )}
              </div>
            )}
          </Box>

          {/* § 04 Rolling WF */}
          <Box>
            <ModuleHeader n="04" title="Rolling Walk-Forward Backtest" />
            <Note>
              Kaydıran pencerelerle her dönemi bağımsız bir gerçek hayat simülasyonu olarak test eder.
              Tek dönemde şans eseri parlayan formülleri rejim değişikliklerine dayanıklı olanlardan ayırır.
            </Note>
            <div style={{ marginBottom: 10 }}>
              <SectionLabel>ROLLING MODU</SectionLabel>
              <div style={{ display: "flex", flexDirection: "column", gap: 4, marginTop: 6 }}>
                {ROLLING_MODES.map((m) => (
                  <div
                    key={m.mode}
                    onClick={() => setWfMode(m.mode)}
                    style={{
                      display: "flex",
                      alignItems: "center",
                      gap: 10,
                      padding: "7px 10px",
                      background: wfMode === m.mode ? "var(--bg-2)" : "transparent",
                      border: `1px solid ${wfMode === m.mode ? "var(--accent)" : "var(--line-soft)"}`,
                      borderRadius: 2,
                      cursor: "pointer",
                    }}
                  >
                    <span style={{ width: 8, height: 8, borderRadius: "50%", background: wfMode === m.mode ? "var(--accent)" : "transparent", border: `1px solid ${wfMode === m.mode ? "var(--accent)" : "var(--line)"}` }} />
                    <span style={{ fontSize: 11.5, color: wfMode === m.mode ? "var(--fg-0)" : "var(--fg-1)" }}>{m.t}</span>
                    <span style={{ flex: 1 }} />
                    <span style={{ fontFamily: "var(--mono)", fontSize: 9.5, color: "var(--fg-3)" }}>{m.d}</span>
                  </div>
                ))}
              </div>
            </div>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr auto", gap: 12, alignItems: "flex-end" }}>
              <Stepper label="Test pencere uzunluğu" value={wfTestMonths} onChange={setWfTestMonths} min={1} max={24} hint="ay" />
              <Stepper label="Min train" value={wfMinTrain} onChange={setWfMinTrain} min={6} max={60} hint="ay" />
              <Select label="Veri penceresi" value={window === "test" ? "TEST" : window === "train" ? "TRAIN" : "TAM"} options={["TEST", "TRAIN", "TAM"]} onChange={(v) => setWindow(WINDOW_MAP[v] ?? "test")} />
              <Btn mono small onClick={handleWF} disabled={wfLoading}>▸ {wfLoading ? "…" : "Rolling WF Koştur"}</Btn>
            </div>
            {wfResult && (
              <div style={{ marginTop: 10 }}>
                {wfResult.error ? (
                  <span style={{ fontFamily: "var(--mono)", fontSize: 10, color: "var(--neg)" }}>⚠ {wfResult.error}</span>
                ) : (
                  <div style={{ display: "flex", gap: 14 }}>
                    <Stat label="Avg Rank IC" value={wfResult.avg_rank_ic?.toFixed(4) ?? "—"} tone={wfResult.avg_rank_ic > 0 ? "pos" : "neg"} />
                    <Stat label="Pozitif dönem" value={`${wfResult.n_positive ?? 0} / ${wfResult.periods?.length ?? 0}`} tone="pos" />
                  </div>
                )}
              </div>
            )}
          </Box>

          {/* § 05 Ensemble */}
          <Box style={{ gridColumn: "span 2" }}>
            <ModuleHeader n="05" title="Ensemble Backtest — çoklu alpha birleşimi" />
            <Note>
              Her satıra bir formül yaz; sinyaller cross-sectional rank ile normalize edilip ağırlıklı ortalaması alınır
              ve tek strateji olarak backtest edilir. Korelasyonsuz sinyaller birleşince IR diversifikasyon kazancı sağlar.
            </Note>
            <div style={{ display: "grid", gridTemplateColumns: "1.3fr 1fr", gap: 18 }}>
              <Input textarea rows={5} label="Ensemble formülleri" hint="her satıra bir tane" value={ensembleText} onChange={setEnsembleText} />
              <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
                <Select label="Ağırlıklandırma" value={ensWeighting} options={["Eşit ağırlık", "IR weighted", "Inverse-vol"]} onChange={setEnsWeighting} hint="IR weighted · inverse-vol" />
                <div>
                  <div style={{ fontSize: 11, color: "var(--fg-1)", marginBottom: 4 }}>Backtest penceresi</div>
                  <SegRow options={["TEST", "TRAIN", "TAM"]} value={ensWindow} onChange={setEnsWindow} />
                </div>
                <Stepper label="Çeşitlilik eşiği (max korelasyon)" value={ensMaxCorr} onChange={setEnsMaxCorr} min={0.1} max={1.0} step={0.05} />
                <Btn mono small full onClick={handleEnsemble} disabled={ensLoading}>
                  ▸ {ensLoading ? "Hesaplanıyor…" : "Ensemble backtest çalıştır"}
                </Btn>
                {ensResult && (
                  <div style={{ display: "flex", gap: 12, flexWrap: "wrap" }}>
                    {ensResult.error ? (
                      <span style={{ fontFamily: "var(--mono)", fontSize: 10, color: "var(--neg)" }}>⚠ {ensResult.error}</span>
                    ) : (
                      <>
                        <Stat label="Ensemble IC" value={ensResult.ic?.toFixed(4) ?? "—"} tone={ensResult.ic > 0 ? "pos" : "neg"} />
                        <Stat label="Rank IC" value={ensResult.rank_ic?.toFixed(4) ?? "—"} tone={ensResult.rank_ic > 0 ? "pos" : "neg"} />
                        <Stat label="Kullanılan" value={`${ensResult.n_formulas_used ?? 0} formül`} />
                      </>
                    )}
                  </div>
                )}
              </div>
            </div>
          </Box>

          {/* § 06 Overfit */}
          <Box style={{ gridColumn: "span 2" }}>
            <ModuleHeader n="06" title="Overfit Testi — Zaman-bazlı validasyon" />
            <Note>
              Sidebar'daki split tarihi kullanılır. Top formüller hem train'de hem test'te değerlendirilir, degradation hesaplanır.
            </Note>
            <div style={{ display: "grid", gridTemplateColumns: "1.5fr 1fr 1fr auto", gap: 14, alignItems: "flex-end" }}>
              <div>
                <div style={{ fontSize: 11, color: "var(--fg-1)", marginBottom: 4 }}>Validasyon modu</div>
                <SegRow options={["Tek Split (hızlı)", "Walk-Forward (güvenilir)"]} value={ovfMode} onChange={setOvfMode} />
              </div>
              <Stepper label="Kaç top formül test edilsin?" value={ovfNTop} onChange={setOvfNTop} min={1} max={50} step={5} />
              <Stepper label="Fold sayısı (WF)" value={ovfFolds} onChange={setOvfFolds} min={2} max={10} step={1} />
              <Btn mono small onClick={handleOverfit} disabled={ovfLoading}>▸ {ovfLoading ? "…" : "Validasyon çalıştır"}</Btn>
            </div>
            {ovfResult && (
              <div style={{ marginTop: 14 }}>
                {ovfResult.error ? (
                  <span style={{ fontFamily: "var(--mono)", fontSize: 10, color: "var(--neg)" }}>⚠ {ovfResult.error}</span>
                ) : (
                  <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 14 }}>
                    <Stat label="Train avg rank ic" value={ovfResult.avg_train_ic?.toFixed(4) ?? "—"} tone="pos" />
                    <Stat label="Test avg rank ic" value={ovfResult.avg_test_ic?.toFixed(4) ?? "—"} tone={ovfResult.avg_test_ic > 0 ? "pos" : "neg"} />
                    <Stat label="Degradation" value={ovfResult.avg_degradation != null ? `${(ovfResult.avg_degradation * 100).toFixed(0)}%` : "—"} tone={ovfResult.avg_degradation != null && ovfResult.avg_degradation > -0.5 ? "pos" : "neg"} hint="acceptable if > -50%" />
                    <Stat label="Passing" value={`${ovfResult.passing ?? 0} / ${ovfResult.results?.length ?? ovfNTop}`} tone="pos" />
                  </div>
                )}
              </div>
            )}
          </Box>
        </div>
      </div>
    </CChrome>
  );
}

function ModuleHeader({ n, title }: { n: string; title: string }) {
  return (
    <div style={{ display: "flex", alignItems: "baseline", gap: 8, marginBottom: 6 }}>
      <span style={{ fontFamily: "var(--mono)", fontSize: 10, color: "var(--fg-3)" }}>§ {n}</span>
      <span style={{ fontFamily: "var(--serif)", fontStyle: "italic", fontSize: 13.5, color: "var(--fg-0)" }}>{title}</span>
    </div>
  );
}
