import { lazy, Suspense } from "react";
import { Routes, Route, Link } from "react-router-dom";

const AtomsDemo = lazy(() => import("./screens/AtomsDemo"));
const Catalog = lazy(() => import("./screens/Catalog"));
const Workbench = lazy(() => import("./screens/Workbench"));
const LLMTrainer = lazy(() => import("./screens/LLMTrainer"));
const BestAlphas = lazy(() => import("./screens/BestAlphas"));
const BacktestStudio = lazy(() => import("./screens/BacktestStudio"));
const ResultsReport = lazy(() => import("./screens/ResultsReport"));

function Loading() {
  return (
    <div style={{ padding: 32, fontFamily: "var(--mono)", fontSize: 11, color: "var(--fg-3)" }}>
      yükleniyor…
    </div>
  );
}

export default function App() {
  return (
    <Suspense fallback={<Loading />}>
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/_atoms" element={<AtomsDemo />} />
        <Route path="/catalog" element={<Catalog />} />
        <Route path="/workbench" element={<Workbench />} />
        <Route path="/llm-trainer" element={<LLMTrainer />} />
        <Route path="/best-alphas" element={<BestAlphas />} />
        <Route path="/backtest-studio" element={<BacktestStudio />} />
        <Route path="/results-report" element={<ResultsReport />} />
        <Route path="*" element={<Home />} />
      </Routes>
    </Suspense>
  );
}

function Home() {
  return (
    <div className="min-h-screen p-7 font-sans" style={{ background: "var(--bg-0)" }}>
      <h1 style={{ fontFamily: "var(--serif)", fontStyle: "italic", fontSize: 22, color: "var(--fg-0)" }}>
        Minerva v3
      </h1>
      <p style={{ color: "var(--fg-2)", marginTop: 8, fontSize: 12 }}>
        Variant C — workbench
      </p>
      <ul style={{ marginTop: 16, display: "flex", flexDirection: "column", gap: 6 }}>
        {[
          ["/_atoms", "Design system showcase"],
          ["/catalog", "Alpha Kataloğu"],
          ["/workbench", "Workbench — Variant C"],
          ["/llm-trainer", "LLM → Tree-LSTM Eğitici"],
          ["/best-alphas", "En İyi Alphalar"],
          ["/backtest-studio", "Backtest Studio — Doğrulama"],
          ["/results-report", "Results Report — PDF Export"],
        ].map(([path, label]) => (
          <li key={path} style={{ fontFamily: "var(--mono)", fontSize: 11 }}>
            →{" "}
            <Link to={path} style={{ color: "var(--accent)", textDecoration: "underline" }}>
              {path}
            </Link>{" "}
            <span style={{ color: "var(--fg-3)" }}>— {label}</span>
          </li>
        ))}
      </ul>
    </div>
  );
}
