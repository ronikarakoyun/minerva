"""
engine/ — Minerva v3 alpha engine

Alt-paket yapısı:
  engine.core       — alpha_cfg, formula_parser, alpha_catalog, backtest_engine, api_helpers
  engine.data       — db_builder, factor_neutralize, regime, triple_barrier, meta_label
  engine.validation — wf_fitness, deflated_sharpe, pbo_cscv, ensemble
  engine.ml         — tree_lstm, trainer, replay_buffer
  engine.strategies — mcts, mining_runner

Geriye-dönük uyumluluk için sık kullanılan semboller buradan da import edilebilir.
"""

# Public API re-exports (from engine.xxx import yyy still works via engine.subpkg.xxx)
from engine.core.alpha_cfg import AlphaCFG, Node
from engine.core.formula_parser import parse_formula, ParseError
from engine.core.alpha_catalog import load_catalog, save_alpha
from engine.core.backtest_engine import run_pro_backtest
from engine.core.api_helpers import run_full_evaluate, evaluate_ic, prepare_eval_idx, slice_db_by_window

__all__ = [
    "AlphaCFG", "Node",
    "parse_formula", "ParseError",
    "load_catalog", "save_alpha",
    "run_pro_backtest",
    "run_full_evaluate", "evaluate_ic", "prepare_eval_idx", "slice_db_by_window",
]
