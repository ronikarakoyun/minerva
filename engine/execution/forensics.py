"""
engine/execution/forensics.py — Faz 6.2: Adli (post-mortem) karar logu.

paper_trades.parquet sadece PnL satırı saklar (entry/exit/slip). Karar
gerekçesini (HMM rejim vektörü, hangi şampiyon baskındı, ADV oranı, slipaj
beklentisi) saklamaz. Bir ay sonra zarar ettiğimizde "neden bu hisseye %4.5
verdik?" sorusunun cevabı kaybolur.

Bu modül her gün her aktif pozisyon için bir "karar fotoğrafı" çekip
data/decisions_log.parquet'e append eder. Şema:
    execution_id, timestamp, date, ticker, action, target_weight, prev_weight,
    hmm_state (JSON), hmm_top_regime, hmm_top_p,
    champion_id, champion_formula,
    adv_TL, adv_limit_ratio, expected_slip_bps, asset_vol,
    notes
"""
from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from engine.execution.slippage import SlippageConfig, compute_slippage_bps
from engine.risk.capacity import CapacityConfig, compute_adv
from engine.risk.position_sizer import compute_asset_vol


FORENSIC_COLUMNS = [
    "execution_id", "timestamp", "date", "ticker", "action",
    "target_weight", "prev_weight", "weight_delta", "turnover",
    "hmm_state", "hmm_top_regime", "hmm_top_p",
    "champion_id", "champion_formula",
    "adv_TL", "adv_limit_ratio", "expected_slip_bps", "asset_vol",
    "notes",
]


@dataclass
class ForensicConfig:
    output_path: Path = field(default_factory=lambda: Path("data/decisions_log.parquet"))
    capacity_cfg: CapacityConfig = field(default_factory=CapacityConfig)
    slippage_cfg: SlippageConfig = field(default_factory=lambda: SlippageConfig(use_dynamic_slippage=True))
    portfolio_capital_TL: float = 1_000_000


def _load_existing(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_parquet(path)
    return pd.DataFrame(columns=FORENSIC_COLUMNS)


def _save(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def _classify_action(target: float, prev: float, threshold: float = 1e-4) -> str:
    delta = target - prev
    if abs(delta) < threshold:
        return "HOLD"
    return "BUY" if delta > 0 else "SELL"


def _dominant_champion(
    ticker: str,
    prob_row: pd.Series,
    champions,
    db_today: pd.DataFrame,
    alpha_cfg=None,
) -> tuple[Optional[int], str]:
    """
    Bu satır için en baskın rejim şampiyonunu bul.

    N23: argmax(prob) yerine ağırlıklı sinyal × olasılık skoru kullanılır.
    Her rejim için: score = |signal_value| × prob[regime]. En yüksek skoru
    alan rejim baskın kabul edilir. Sinyal hesaplanamıyorsa prob tek başına
    kullanılır (graceful fallback).

    `champions` dict[int, Node] (regime_id → AST). alpha_cfg sağlandığında
    tam sinyal değerlendirmesi yapılır; yoksa prob ağırlığına göre seçim.
    """
    if not champions:
        return None, ""

    try:
        # N23: Weighted signal × probability dominant selection
        idx_today = db_today.set_index(["Ticker", "Date"]) if (
            "Ticker" in db_today.columns and "Date" in db_today.columns
        ) else None

        best_regime: Optional[int] = None
        best_score: float = -1.0

        for regime_id, tree in champions.items():
            prob_val = float(prob_row.get(regime_id, 0.0)) if hasattr(prob_row, "get") \
                else (float(prob_row.iloc[regime_id]) if regime_id < len(prob_row) else 0.0)

            signal_mag = 1.0  # fallback: sadece prob kullan
            if alpha_cfg is not None and idx_today is not None and len(idx_today) > 0:
                try:
                    sig = alpha_cfg.evaluate(tree, idx_today)
                    if sig is not None and ticker in sig.index.get_level_values(0):
                        ticker_sig = sig.xs(ticker, level=0)
                        if len(ticker_sig) > 0:
                            signal_mag = abs(float(ticker_sig.iloc[-1]))
                            if not np.isfinite(signal_mag):
                                signal_mag = 1.0
                except Exception:
                    pass  # sinyal hesaplanamadı → signal_mag=1.0 ile devam

            score = signal_mag * prob_val
            if score > best_score:
                best_score = score
                best_regime = regime_id

        if best_regime is not None and best_regime in champions:
            return int(best_regime), str(champions[best_regime])

        # Son çare: ilk mevcut şampiyon
        first_k = next(iter(champions))
        return int(first_k), str(champions[first_k])
    except Exception:
        return None, ""


def log_decision_forensics(
    date: pd.Timestamp,
    weights: pd.Series,
    prob_row: pd.Series,
    champions: dict,
    db: pd.DataFrame,
    formula_id: str,
    prev_weights: Optional[pd.Series] = None,
    cfg: Optional[ForensicConfig] = None,
    notes: str = "",
) -> int:
    """
    Her aktif/değişen pozisyon için bir adli kayıt yaz.

    Parameters
    ----------
    date : pd.Timestamp
        Karar tarihi.
    weights : pd.Series
        Index=Ticker, value=target weight. Sadece pozitifler işlenir.
    prob_row : pd.Series
        O günün rejim olasılık vektörü (Σ=1).
    champions : dict[int, Node]
        Rejim şampiyonları (blender'dan).
    db : pd.DataFrame
        Flat market_db.
    formula_id : str
        Üst-seviye blend formula id (paper_trader log eşleşmesi için).
    prev_weights : pd.Series | None
        Önceki günün weights'i — turnover tespit için.
    cfg : ForensicConfig
    notes : str
        Serbest metin (decay alarm, fallback, vs.).

    Returns
    -------
    int
        Yazılan satır sayısı.
    """
    cfg = cfg or ForensicConfig()
    prev_weights = prev_weights if prev_weights is not None else pd.Series(dtype=float)

    active = weights[weights > 0].dropna()
    if len(active) == 0:
        return 0

    # ADV — sadece o güne kadar
    db_window = db[db["Date"] <= date].copy()
    db_window["Date"] = pd.to_datetime(db_window["Date"])
    try:
        adv_df = compute_adv(db_window, cfg.capacity_cfg)
    except Exception:
        adv_df = pd.DataFrame()

    # Vol için wide return matrisi
    try:
        ret_wide = (
            db_window.pivot_table(index="Date", columns="Ticker", values="Pclose")
            .pct_change()
        )
    except Exception:
        ret_wide = pd.DataFrame()

    hmm_state_str = json.dumps(
        {str(k): float(v) for k, v in prob_row.items()}, ensure_ascii=False,
    )
    top_regime = int(prob_row.values.argmax()) if len(prob_row) else -1
    top_p = float(prob_row.values.max()) if len(prob_row) else 0.0

    db_today = db[db["Date"] == date]
    ts_iso = datetime.now(timezone.utc).isoformat()

    rows = []
    for ticker, weight in active.items():
        prev = float(prev_weights.get(ticker, 0.0))
        action = _classify_action(float(weight), prev)

        # ADV TL
        try:
            adv_TL = float(adv_df.loc[(ticker, date), "ADV_TL"])
        except (KeyError, ValueError, TypeError):
            adv_TL = float("nan")

        # ADV limit oranı: v_traded / (adv_pct × ADV)
        v_traded = cfg.portfolio_capital_TL * float(weight)
        if np.isfinite(adv_TL) and adv_TL > 0:
            adv_limit_ratio = float(v_traded / (cfg.capacity_cfg.adv_pct_limit * adv_TL))
        else:
            adv_limit_ratio = float("nan")

        # Asset vol
        if isinstance(ret_wide, pd.DataFrame) and ticker in ret_wide.columns:
            vol_series = compute_asset_vol(ret_wide[ticker], window=cfg.slippage_cfg.sigma_window)
            vs = vol_series.loc[:date].dropna()
            asset_vol = float(vs.iloc[-1]) if len(vs) else float("nan")
        else:
            asset_vol = float("nan")

        slip_bps = compute_slippage_bps(v_traded, asset_vol, adv_TL, cfg.slippage_cfg)

        champ_id, champ_formula = _dominant_champion(
            ticker, prob_row, champions, db_today,
        )

        weight_delta = float(weight) - prev
        rows.append({
            "execution_id":      uuid.uuid4().hex,
            "timestamp":         ts_iso,
            "date":              pd.Timestamp(date),
            "ticker":            str(ticker),
            "action":            action,
            "target_weight":     float(weight),
            "prev_weight":       prev,
            "weight_delta":      weight_delta,
            "turnover":          abs(weight_delta),
            "hmm_state":         hmm_state_str,
            "hmm_top_regime":    top_regime,
            "hmm_top_p":         top_p,
            "champion_id":       champ_id if champ_id is not None else -1,
            "champion_formula":  champ_formula,
            "adv_TL":            adv_TL,
            "adv_limit_ratio":   adv_limit_ratio,
            "expected_slip_bps": float(slip_bps),
            "asset_vol":         asset_vol,
            "notes":             notes,
        })

    if not rows:
        return 0

    new_df = pd.DataFrame(rows, columns=FORENSIC_COLUMNS)
    existing = _load_existing(cfg.output_path)
    combined = pd.concat([existing, new_df], ignore_index=True)
    _save(combined, cfg.output_path)
    return len(rows)


def load_decisions(
    start: Optional[pd.Timestamp] = None,
    end: Optional[pd.Timestamp] = None,
    ticker: Optional[str] = None,
    cfg: Optional[ForensicConfig] = None,
) -> pd.DataFrame:
    """Tarih/hisse filtresiyle adli kayıtları döndür — post-mortem analiz."""
    cfg = cfg or ForensicConfig()
    df = _load_existing(cfg.output_path)
    if len(df) == 0:
        return df

    df["date"] = pd.to_datetime(df["date"])
    if start is not None:
        df = df[df["date"] >= pd.Timestamp(start)]
    if end is not None:
        df = df[df["date"] <= pd.Timestamp(end)]
    if ticker is not None:
        df = df[df["ticker"] == ticker]
    return df.reset_index(drop=True)
