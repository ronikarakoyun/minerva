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

export interface EvaluateResult {
  ic: number;
  rank_ic: number;
  sharpe: number;
  ann_ret: number;
  max_dd: number;
  turnover: number;
  dsr: number;
  equity_curve: number[];
  drawdown: number[];
  fold_sharpes: number[];
  train_split: number;
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
  validations?: {
    meta_label?: boolean;
    dsr?: boolean;
    pbo?: boolean;
    rolling_wf?: boolean;
    ensemble?: boolean;
    time_overfit?: boolean;
  };
}
