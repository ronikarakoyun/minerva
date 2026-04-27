export interface CatalogRecord {
  formula: string;
  ic: number | null;
  rank_ic: number | null;
  adj_ic: number | null;
  wf_score: number | null;
  sharpe: number | null;
  overfit_score: number | null;
  best_window: string | null;
  source: string | null;
  has_backtest: boolean;
  created_at?: string;
}

export interface EquityPoint {
  date: string;
  equity: number;
  benchmark?: number;
}

export interface EvaluateResult {
  ic: number | null;
  rank_ic: number | null;
  sharpe: number | null;
  annual: number | null;
  mdd: number | null;
  net_return: number | null;
  alpha_ir?: number | null;
  beta?: number | null;
  equity_curve: EquityPoint[];
  n_observations: number | null;
  fold_sharpes?: number[];
}

export interface JobEvent {
  type: "progress" | "log" | "result" | "error";
  value?: number;
  line?: string;
  data?: EvaluateResult;
  message?: string;
}

export interface BacktestRequest {
  formula: string;
  window: "test" | "train" | "all";
}
