import { lazy, Suspense } from "react";
import { Routes, Route, Navigate } from "react-router-dom";

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
        <Route path="/" element={<Navigate to="/workbench" replace />} />
        <Route path="/_atoms" element={<AtomsDemo />} />
        <Route path="/catalog" element={<Catalog />} />
        <Route path="/workbench" element={<Workbench />} />
        <Route path="/llm-trainer" element={<LLMTrainer />} />
        <Route path="/best-alphas" element={<BestAlphas />} />
        <Route path="/backtest-studio" element={<BacktestStudio />} />
        <Route path="/results-report" element={<ResultsReport />} />
        <Route path="*" element={<Navigate to="/workbench" replace />} />
      </Routes>
    </Suspense>
  );
}
