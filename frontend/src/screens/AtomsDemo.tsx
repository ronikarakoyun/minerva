import { useState } from "react";
import { Pill, Stat, Btn, Field, Check, SegRow, SectionLabel, Logo } from "../components/atoms";
import { Input, Select, Stepper, Note } from "../components/inputs";
import { CChrome } from "../components/chrome/CChrome";
import { Panel } from "../components/chrome/Panel";
import { Box } from "../components/chrome/Box";
import { MiniSparkline, EquityChart, DrawdownChart, HeatmapRow } from "../components/charts";

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

const sparkData = equityCurve(1, 50, 0.001, 0.012);
const alphaData = equityCurve(1, 200, 0.0014, 0.011);
const benchData = equityCurve(99, 200, 0.0004, 0.009);
const ddData = Array.from({ length: 200 }, (_, i) =>
  -Math.abs(Math.sin(i / 12)) * Math.min(0.12, i * 0.0007)
);
const heatData = Array.from({ length: 24 }, (_, i) => Math.sin(i * 0.4) * 0.8 + (Math.random() - 0.5) * 0.3);

export default function AtomsDemo() {
  const [seg, setSeg] = useState("TEST · oos");
  const [checked, setChecked] = useState(true);
  const [stepVal, setStepVal] = useState(5);

  return (
    <div style={{ padding: 32, background: "var(--bg-0)", minHeight: "100vh", display: "flex", flexDirection: "column", gap: 32 }}>
      <h1 style={{ fontFamily: "var(--serif)", fontStyle: "italic", fontSize: 22, color: "var(--fg-0)" }}>
        Minerva v3 — Design System Showcase
      </h1>

      {/* Atoms */}
      <section>
        <SectionLabel>Atoms · Pill</SectionLabel>
        <div className="flex gap-2 flex-wrap mt-2">
          <Pill>neutral</Pill>
          <Pill tone="accent">accent</Pill>
          <Pill tone="pos">pos +2.41</Pill>
          <Pill tone="neg">neg −0.12</Pill>
          <Pill tone="ghost">ghost</Pill>
          <Pill mono>mono pill</Pill>
        </div>
      </section>

      <section>
        <SectionLabel>Atoms · Stat</SectionLabel>
        <div className="flex gap-4 flex-wrap mt-2">
          <Stat label="Sharpe" value="2.41" tone="pos" />
          <Stat label="DSR" value="0.62" />
          <Stat label="Ann Ret" value="18.4%" tone="pos" />
          <Stat label="Max DD" value="−12.4%" tone="neg" />
          <Stat label="IC" value="0.084" />
          <Stat label="Turnover" value="34%" />
        </div>
      </section>

      <section>
        <SectionLabel>Atoms · Btn</SectionLabel>
        <div className="flex gap-2 flex-wrap mt-2">
          <Btn>Default</Btn>
          <Btn primary>Primary</Btn>
          <Btn variant="ghost">Ghost</Btn>
          <Btn variant="danger">Danger</Btn>
          <Btn disabled>Disabled</Btn>
        </div>
      </section>

      <section>
        <SectionLabel>Atoms · Logo</SectionLabel>
        <div className="flex items-center gap-4 mt-2">
          <Logo size={12} />
          <Logo size={16} />
          <Logo size={20} />
        </div>
      </section>

      <section>
        <SectionLabel>Atoms · SegRow</SectionLabel>
        <div className="mt-2">
          <SegRow
            options={["TEST · oos", "TRAIN · is", "TAM"]}
            value={seg}
            onChange={setSeg}
          />
        </div>
      </section>

      <section>
        <SectionLabel>Atoms · Check + Field</SectionLabel>
        <div className="flex gap-6 mt-2">
          <Check label="Meta-Label" checked={checked} onChange={setChecked} />
          <Check label="DSR" checked={false} onChange={() => {}} />
          <Field label="IC" value="0.084" />
          <Field label="Sharpe" value="2.41" />
        </div>
      </section>

      {/* Inputs */}
      <section>
        <SectionLabel>Inputs</SectionLabel>
        <div className="flex gap-4 flex-wrap mt-2 items-start">
          <Input label="Formül" hint="expression" placeholder="CSRank(Delta(Pclose, 20))" style={{ width: 260 }} />
          <Select label="Kaynak" value="Tümü" options={["Tümü", "EVO", "LLM", "MCTS"]} style={{ width: 160 }} />
          <Stepper label="Popülasyon" value={stepVal} onChange={setStepVal} min={1} max={1000} step={50} width={140} />
        </div>
        <div className="mt-3">
          <Note cite="Bilgi:">
            Formül tokenları AST olarak parse edilir; size-corr limiti aşılırsa kırmızı hata gösterilir.
          </Note>
        </div>
      </section>

      {/* Chrome */}
      <section>
        <SectionLabel>Chrome · CChrome + Panel + Box</SectionLabel>
        <div className="mt-2" style={{ width: 900 }}>
          <CChrome
            title="Minerva v3"
            sub="workbench · variant-c"
            meta={{ train: "2018–2022", test: "2023–2024", split: "80/20", benchmark: "CSI 500" }}
            width={900}
            height={380}
          >
            <div className="flex" style={{ flex: 1, overflow: "hidden" }}>
              <Panel num="1" title="Katalog" flex="0 0 260px">
                <Box p={10}>
                  <span style={{ fontFamily: "var(--mono)", fontSize: 11, color: "var(--fg-2)" }}>
                    362 alfa · 24 backtest yapılmış
                  </span>
                </Box>
              </Panel>
              <Panel num="2" title="Seçili Formül" flex={1}>
                <div style={{ fontFamily: "var(--mono)", fontSize: 12, color: "var(--fg-0)" }}>
                  CSRank(Div(0.01, Add(CSRank(Add(Vlot, -0.01)), -0.1)))
                </div>
              </Panel>
            </div>
          </CChrome>
        </div>
      </section>

      {/* Charts */}
      <section>
        <SectionLabel>Charts · MiniSparkline</SectionLabel>
        <div className="flex gap-4 mt-2 items-end">
          <MiniSparkline data={sparkData} width={120} height={28} color="var(--accent)" fill />
          <MiniSparkline data={sparkData} width={140} height={14} color="var(--fg-3)" />
          <MiniSparkline data={sparkData} width={80} height={20} color="var(--pos)" fill />
        </div>
      </section>

      <section>
        <SectionLabel>Charts · EquityChart</SectionLabel>
        <div className="mt-2" style={{ width: 600 }}>
          <EquityChart width={600} height={160} train={140} alpha={alphaData} bench={benchData} label="CSRank·Div" />
        </div>
      </section>

      <section>
        <SectionLabel>Charts · DrawdownChart</SectionLabel>
        <div className="mt-2" style={{ width: 600 }}>
          <DrawdownChart width={600} height={64} data={ddData} />
        </div>
      </section>

      <section>
        <SectionLabel>Charts · HeatmapRow</SectionLabel>
        <div className="mt-2" style={{ width: 600 }}>
          {["Fold 1", "Fold 2", "Fold 3", "Fold 4", "Fold 5"].map((lbl, i) => (
            <HeatmapRow
              key={lbl}
              label={lbl}
              values={heatData.slice(i * 4, i * 4 + 24 / 5 * (i + 1))}
              width={500}
            />
          ))}
        </div>
      </section>
    </div>
  );
}
