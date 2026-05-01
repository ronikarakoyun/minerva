"""
engine/data/regime_detector.py — Faz 1: Kısıtlı HMM tabanlı piyasa rejim tespiti.

BIST100 EOD verisinden 4 durağan özellik üretir, Gaussian HMM ile K∈[2,6] arasında
arama yapar, "minimum 200 örnek/rejim" kuralıyla aday K'ları filtreler ve en düşük
BIC'li modeli seçer. Çıktı, Faz 2'nin (regime-conditional weighting) doğrudan
girdisi olan bir **olasılık vektörü DataFrame**'idir.

Kullanım (CLI):
    python -m engine.data.regime_detector

Kullanım (programmatic):
    from engine.data.regime_detector import run_pipeline, RegimeConfig
    prob_df = run_pipeline()                       # default cfg
    prob_df = run_pipeline(RegimeConfig(period="5y"))

Çıktılar (data/ klasörüne):
    regime_hmm.pkl         — joblib ile serileştirilmiş GaussianHMM
    regime_metadata.json   — eğitim tarihi, K, BIC/AIC, per-regime stats, son gün
    regime_plot.png        — log fiyat (rejim renkli) + son gün olasılık bar chart
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import joblib
import matplotlib

matplotlib.use("Agg")  # headless / Airflow-safe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import RobustScaler

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────
@dataclass
class RegimeConfig:
    """Tüm Faz 1 parametreleri tek noktada."""
    ticker: str = "XU100.IS"
    period: str = "10y"
    min_K: int = 2
    max_K: int = 6
    min_samples_per_regime: int = 200
    n_iter: int = 1000
    # N7: "diag" → 4 param/regime (vs "full"=16), az veri durumunda daha kararlı
    covariance_type: str = "diag"
    random_state: int = 42
    atr_window: int = 14
    volume_sma_window: int = 20
    model_path: Path = field(default_factory=lambda: Path("data/regime_hmm.pkl"))
    metadata_path: Path = field(default_factory=lambda: Path("data/regime_metadata.json"))
    plot_path: Path = field(default_factory=lambda: Path("data/regime_plot.png"))
    # True (default) → sadece Forward algoritması — backtest'te look-ahead yok.
    # False → Forward-Backward (Smoothed) — daha pürüzsüz ama geçmiş noktaları
    #         gelecek veriyle hesaplar; yalnızca gerçek zamanlı son gün tahmini için uygun.
    use_filtered_probs: bool = True


FEATURES = ["Log_Return", "Norm_ATR_14", "Volume_Accel", "Choppiness"]


# ──────────────────────────────────────────────────────────────────────
# 1.1 — Veri çekimi & feature engineering
# ──────────────────────────────────────────────────────────────────────
def fetch_bist_data(cfg: RegimeConfig) -> pd.DataFrame:
    """yfinance → OHLCV DataFrame (Date index)."""
    logger.info("yfinance fetch: %s, period=%s", cfg.ticker, cfg.period)
    df = yf.download(
        cfg.ticker, period=cfg.period, auto_adjust=False, progress=False
    )
    if df.empty:
        raise RuntimeError(f"yfinance boş döndü: {cfg.ticker}")
    # MultiIndex sütun varsa (yfinance >=0.2.40), düzleştir
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.index = pd.to_datetime(df.index)
    df.index.name = "Date"
    df = df.dropna(subset=["Open", "High", "Low", "Close", "Volume"])
    logger.info("Veri: %d gün, %s → %s",
                len(df), df.index[0].date(), df.index[-1].date())
    return df


def compute_features(
    df: pd.DataFrame,
    cfg: RegimeConfig,
    train_end_date: Optional[pd.Timestamp] = None,
) -> tuple[pd.DataFrame, RobustScaler]:
    """
    4 durağan özellik üretir ve RobustScaler ile ölçeklendirir.

    N6: Scaler yalnızca `train_end_date`'e kadar olan veriye fit edilir;
    tüm veri bu scaler ile transform edilir.
    `train_end_date=None` → mevcut davranış (tüm veriye fit).

    Returns
    -------
    tuple[pd.DataFrame, RobustScaler]
        (scaled_features DataFrame, fit edilmiş scaler)
    """
    out = pd.DataFrame(index=df.index)
    out["Log_Return"] = np.log(df["Close"] / df["Close"].shift(1))

    # True range → ATR_14 → Close ile normalize
    high_low = df["High"] - df["Low"]
    high_pc = (df["High"] - df["Close"].shift(1)).abs()
    low_pc = (df["Low"] - df["Close"].shift(1)).abs()
    tr = pd.concat([high_low, high_pc, low_pc], axis=1).max(axis=1)
    out["Norm_ATR_14"] = tr.rolling(cfg.atr_window).mean() / df["Close"]

    out["Volume_Accel"] = df["Volume"] / df["Volume"].rolling(cfg.volume_sma_window).mean()
    out["Choppiness"] = (df["High"] - df["Low"]) / df["Close"]

    out = out.dropna()
    if len(out) == 0:
        raise RuntimeError("Feature engineering: tüm satırlar NaN — veri yetersiz")

    scaler = RobustScaler()
    # N6: Scaler sadece train penceresi üzerine fit edilir
    if train_end_date is not None:
        train_mask = out.index <= pd.Timestamp(train_end_date)
        train_data = out[train_mask].values
        if len(train_data) < 10:
            logger.warning("N6: train_end_date öncesi veri az (%d satır) — tüm veriye fit ediliyor", len(train_data))
            train_data = out.values
    else:
        train_data = out.values

    scaler.fit(train_data)
    scaled = scaler.transform(out.values)
    logger.info("Features hazır: %d satır × %d kolon (scaled, train_end=%s)",
                *scaled.shape, train_end_date)
    return pd.DataFrame(scaled, index=out.index, columns=FEATURES), scaler


# ──────────────────────────────────────────────────────────────────────
# 1.2 — Constrained HMM with K-selection
# ──────────────────────────────────────────────────────────────────────
def _bic_aic(model: GaussianHMM, X: np.ndarray) -> tuple[float, float, float]:
    """GaussianHMM (full covariance) için BIC ve AIC."""
    n, d = X.shape
    K = model.n_components
    # Param sayısı: transmat (K·(K-1)) + startprob (K-1) + means (K·d) + full covars (K·d·(d+1)/2)
    n_params = K * (K - 1) + (K - 1) + K * d + K * d * (d + 1) // 2
    log_lik = float(model.score(X))
    bic = -2 * log_lik + n_params * np.log(n)
    aic = -2 * log_lik + 2 * n_params
    return bic, aic, log_lik


def fit_constrained_hmm(
    features: pd.DataFrame, cfg: RegimeConfig
) -> tuple[GaussianHMM, int, dict[int, dict[str, Any]]]:
    """
    K∈[min_K, max_K] her biri için GaussianHMM fit + BIC + min-samples kontrolü.

    Returns
    -------
    best_model : GaussianHMM
    best_K : int
    candidates : dict
        {K: {"model", "bic", "aic", "log_lik", "counts"}} — sadece geçerli K'lar
    """
    X = features.values
    candidates: dict[int, dict[str, Any]] = {}

    for K in range(cfg.min_K, cfg.max_K + 1):
        try:
            model = GaussianHMM(
                n_components=K,
                covariance_type=cfg.covariance_type,
                n_iter=cfg.n_iter,
                random_state=cfg.random_state,
            )
            model.fit(X)
            states = model.predict(X)
            counts = np.bincount(states, minlength=K)

            if counts.min() < cfg.min_samples_per_regime:
                logger.info(
                    "K=%d DISQUALIFIED — min count=%d < %d (counts=%s)",
                    K, int(counts.min()), cfg.min_samples_per_regime, counts.tolist(),
                )
                continue

            bic, aic, log_lik = _bic_aic(model, X)
            candidates[K] = {
                "model": model,
                "bic": float(bic),
                "aic": float(aic),
                "log_lik": float(log_lik),
                "counts": counts.tolist(),
            }
            logger.info(
                "K=%d  BIC=%.1f  AIC=%.1f  logL=%.1f  counts=%s",
                K, bic, aic, log_lik, counts.tolist(),
            )
        except Exception as e:
            logger.warning("K=%d fit failed: %s", K, e)

    if not candidates:
        raise RuntimeError(
            f"Hiçbir K∈[{cfg.min_K},{cfg.max_K}] aralığı min_samples_per_regime="
            f"{cfg.min_samples_per_regime} kuralını geçemedi."
        )

    best_K = min(candidates, key=lambda k: candidates[k]["bic"])
    logger.info("OPTIMAL K=%d (BIC=%.1f)", best_K, candidates[best_K]["bic"])
    return candidates[best_K]["model"], best_K, candidates


# ──────────────────────────────────────────────────────────────────────
# 1.3 — Soft probability vector
# ──────────────────────────────────────────────────────────────────────
def compute_probability_vector(
    model: GaussianHMM, features: pd.DataFrame
) -> pd.DataFrame:
    """
    Forward-Backward (Smoothed) olasılıklar. Tüm diziyi kullanır.

    ⚠️  Backtest'te look-ahead bias yaratır: t anındaki olasılık, t'den sonraki
    veriyle hesaplanmış olur. Yalnızca canlı son-gün tahmini için kullanılmalıdır.
    Backtest için compute_filtered_probability_vector() tercih edilmeli.
    """
    probs = model.predict_proba(features.values)
    cols = [f"regime_{i}" for i in range(probs.shape[1])]
    return pd.DataFrame(probs, index=features.index, columns=cols)


def compute_filtered_probability_vector(
    model: GaussianHMM, features: pd.DataFrame
) -> pd.DataFrame:
    """
    Sadece Forward (Filtered) olasılıklar — backtest güvenli, look-ahead yok.

    predict_proba() Forward-Backward algoritması kullanır: t noktasındaki
    olasılık hesaplanırken t+1..T gözlemleri de kullanılır (Smoothed).
    Bu, backtest'te rejim geçişlerinin "sihirli" önceden bilinmesine yol açar.

    Bu fonksiyon yalnızca Forward pass çalıştırır:
        α₀ = π ⊙ emit(x₀)
        αₜ = normalize( A' @ αₜ₋₁ ⊙ emit(xₜ) )
    Böylece αₜ yalnızca x₀..xₜ gözlemlerini kullanır — causal.

    Returns
    -------
    pd.DataFrame
        index = features.index (Date), columns = ["regime_0", ..., "regime_K-1"].
    """
    X = features.values
    T = len(X)
    K = model.n_components

    # Emission log-olasılıkları: (T, K) — hmmlearn internal, tüm modeller destekler
    log_emit = model._compute_log_likelihood(X)

    # Numerik taşmayı önlemek: her satırdan max çıkart, sonra exp al
    log_emit_stable = log_emit - log_emit.max(axis=1, keepdims=True)
    emit = np.exp(log_emit_stable)  # (T, K) — relative doğru, normalize değil

    A = model.transmat_   # (K, K): A[i, j] = P(j | i)
    pi = model.startprob_  # (K,)

    alpha = np.empty((T, K), dtype=float)
    alpha[0] = pi * emit[0]
    norm = alpha[0].sum()
    alpha[0] /= norm if norm > 1e-300 else 1.0

    for t in range(1, T):
        alpha[t] = (A.T @ alpha[t - 1]) * emit[t]
        norm = alpha[t].sum()
        alpha[t] /= norm if norm > 1e-300 else 1.0

    cols = [f"regime_{i}" for i in range(K)]
    return pd.DataFrame(alpha, index=features.index, columns=cols)


# ──────────────────────────────────────────────────────────────────────
# 1.4 — Görselleştirme
# ──────────────────────────────────────────────────────────────────────
def plot_regimes(
    price: pd.Series, prob_df: pd.DataFrame, cfg: RegimeConfig
) -> None:
    """
    2 alt grafik: log fiyat (rejim renkli arka plan) + son gün olasılık bar chart.
    Headless backend (Agg) ile PNG'ye kaydeder.
    """
    states = prob_df.values.argmax(axis=1)
    K = prob_df.shape[1]
    cmap = plt.get_cmap("tab10", max(K, 3))

    fig, axes = plt.subplots(
        2, 1, figsize=(14, 8), gridspec_kw={"height_ratios": [3, 1]}
    )

    # ── Üst panel: log fiyat + rejim shading ──
    log_price = np.log(price.reindex(prob_df.index).astype(float))
    axes[0].plot(log_price.index, log_price.values, color="black", lw=1.0)

    # Bitişik rejim segmentlerini boyamak için runs hesapla
    cur_state = int(states[0])
    seg_start = prob_df.index[0]
    for i in range(1, len(states)):
        if int(states[i]) != cur_state:
            axes[0].axvspan(seg_start, prob_df.index[i],
                            color=cmap(cur_state), alpha=0.18)
            cur_state = int(states[i])
            seg_start = prob_df.index[i]
    axes[0].axvspan(seg_start, prob_df.index[-1], color=cmap(cur_state), alpha=0.18)
    axes[0].set_title(f"BIST100 Log Fiyat + Rejim Tespiti  (K={K})")
    axes[0].set_ylabel("log(Close)")
    axes[0].grid(alpha=0.2)

    # ── Alt panel: son gün olasılık vektörü ──
    last = prob_df.iloc[-1]
    bars = axes[1].bar(
        last.index, last.values * 100, color=[cmap(i) for i in range(K)]
    )
    axes[1].set_title(f"Güncel Olasılık Vektörü ({prob_df.index[-1].date()})")
    axes[1].set_ylabel("%")
    axes[1].set_ylim(0, 100)
    for b, v in zip(bars, last.values * 100):
        axes[1].text(b.get_x() + b.get_width() / 2, v + 1.5,
                     f"{v:.1f}%", ha="center", fontsize=9)

    plt.tight_layout()
    cfg.plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(cfg.plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Plot saved: %s", cfg.plot_path)


# ──────────────────────────────────────────────────────────────────────
# 1.5 — Persistence
# ──────────────────────────────────────────────────────────────────────
def save_model(
    model: GaussianHMM,
    best_K: int,
    candidates: dict[int, dict[str, Any]],
    prob_df: pd.DataFrame,
    raw_returns: pd.Series,
    cfg: RegimeConfig,
) -> None:
    """Modeli pkl'ye, metadata'yı JSON'a yaz."""
    cfg.model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, cfg.model_path)
    logger.info("Model saved: %s", cfg.model_path)

    states = prob_df.values.argmax(axis=1)
    raw_aligned = raw_returns.reindex(prob_df.index)

    regime_stats = {}
    for k in range(best_K):
        mask = states == k
        rets = raw_aligned[mask].dropna()
        regime_stats[f"regime_{k}"] = {
            "n_days": int(mask.sum()),
            "mean_daily_return": float(rets.mean()) if len(rets) else None,
            "annualized_vol": float(rets.std() * np.sqrt(252)) if len(rets) else None,
        }

    metadata = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "ticker": cfg.ticker,
        "period": cfg.period,
        "optimal_K": int(best_K),
        "scores": {
            str(k): {"bic": v["bic"], "aic": v["aic"], "log_lik": v["log_lik"],
                     "counts": v["counts"]}
            for k, v in candidates.items()
        },
        "regime_stats": regime_stats,
        "last_day": str(prob_df.index[-1].date()),
        "last_day_probs": {k: float(v) for k, v in prob_df.iloc[-1].items()},
        # Label switching koruması: bir sonraki refit'te eski means ile align edilir
        "regime_means": model.means_.tolist(),
    }
    cfg.metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False)
    )
    logger.info("Metadata saved: %s", cfg.metadata_path)


def load_model(cfg: Optional[RegimeConfig] = None) -> GaussianHMM:
    """data/regime_hmm.pkl'den modeli geri yükle."""
    cfg = cfg or RegimeConfig()
    return joblib.load(cfg.model_path)


# ──────────────────────────────────────────────────────────────────────
# 1.6 — Label switching koruması (Hungarian permutation match)
# ──────────────────────────────────────────────────────────────────────
def align_regime_labels(
    new_model: GaussianHMM,
    old_means: Optional[np.ndarray] = None,
    metadata_path: Optional[Path] = None,
) -> "np.ndarray[int]":
    """
    Yeni HMM state ID'lerini eski state ID'lerine en yakın biçimde eşle.

    HMM her refit'te state numaralandırmasını değiştirebilir (label switching).
    Bu durum `regime_champion_for` eşleşmesini bozar.

    Algoritma:
      - Eski model'in per-regime mean return vektörünü referans al.
      - Yeni modelin per-regime means'i ile Öklid uzaklığı maliyet matrisi kur.
      - scipy.optimize.linear_sum_assignment (Hungarian) ile minimum maliyetli
        atamayı bul.
      - Döndürülen `perm` dizisinde perm[new_id] = old_id.

    Parameters
    ----------
    new_model : GaussianHMM
        Yeni fit edilmiş model.
    old_means : np.ndarray (K_old, n_features), opsiyonel
        Eski modelin öğrenilmiş means_. Sağlanmazsa metadata_path'ten yüklenir.
    metadata_path : Path, opsiyonel
        `data/regime_metadata.json` — old_means yoksa buradan okunur.

    Returns
    -------
    perm : np.ndarray[int]
        Boyut (K_new,). perm[new_state] = best_matching_old_state.
        K_new ≠ K_old ise en iyi eşleme (min dist) döner; eşleşmeyen new_state'ler
        kendine atanır (perm[i] = i fallback).
    """
    try:
        from scipy.optimize import linear_sum_assignment
    except ImportError:
        logger.warning("scipy yok — label alignment atlanıyor, kimlik permütasyonu kullanılıyor")
        K_new = new_model.n_components
        return np.arange(K_new, dtype=int)

    new_means = new_model.means_  # (K_new, n_features)
    K_new = new_means.shape[0]

    # Eski means'i al
    if old_means is None and metadata_path is not None and metadata_path.exists():
        try:
            meta = json.loads(metadata_path.read_text())
            if "regime_means" in meta:
                old_means = np.array(meta["regime_means"], dtype=float)
        except Exception:
            old_means = None

    if old_means is None:
        logger.info("label_align: eski means yok — kimlik permütasyonu döndürülüyor")
        return np.arange(K_new, dtype=int)

    K_old = old_means.shape[0]

    # Maliyet matrisi: (K_new, K_old) — Öklid uzaklığı
    cost = np.zeros((K_new, K_old), dtype=float)
    for i in range(K_new):
        for j in range(K_old):
            cost[i, j] = float(np.linalg.norm(new_means[i] - old_means[j]))

    row_ind, col_ind = linear_sum_assignment(cost)
    perm = np.arange(K_new, dtype=int)  # fallback: kimlik
    for new_id, old_id in zip(row_ind, col_ind):
        perm[new_id] = old_id

    logger.info(
        "HMM label alignment: perm=%s (cost=%.4f)",
        perm.tolist(),
        cost[row_ind, col_ind].sum(),
    )
    return perm


def reorder_prob_df(prob_df: pd.DataFrame, perm: "np.ndarray[int]") -> pd.DataFrame:
    """
    prob_df kolonlarını `perm` permütasyonuna göre yeniden sırala.

    prob_df.columns = ['regime_0', 'regime_1', ...] varsayımı.
    perm[new_col_idx] = old_col_idx → eski indekslere göre yeniden adlandır.
    """
    K = prob_df.shape[1]
    if len(perm) != K:
        return prob_df  # boyut uyuşmazsa değiştirme

    new_cols = [f"regime_{perm[i]}" for i in range(K)]
    result = prob_df.copy()
    result.columns = new_cols
    # Sütunları eski sıraya göre düzenle (regime_0, regime_1, ...)
    sorted_cols = sorted(result.columns, key=lambda c: int(c.split("_")[1]))
    return result[sorted_cols]


# ──────────────────────────────────────────────────────────────────────
# Pipeline
# ──────────────────────────────────────────────────────────────────────
def run_pipeline(cfg: Optional[RegimeConfig] = None) -> pd.DataFrame:
    """End-to-end Faz 1 pipeline. Returns prob_df (Faz 2 girdisi)."""
    cfg = cfg or RegimeConfig()
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(name)s — %(message)s",
            datefmt="%H:%M:%S",
        )

    logger.info("=== Minerva Faz 1: Regime Detection ===")
    df = fetch_bist_data(cfg)
    # N6: 5 yıllık verinin ilk %80'ini train penceresi kabul et (scaler fit için)
    train_end = df.index[int(len(df) * 0.80)] if len(df) > 50 else None
    features, _scaler = compute_features(df, cfg, train_end_date=train_end)
    model, best_K, candidates = fit_constrained_hmm(features, cfg)
    # Backtest güvenli: yalnızca Forward algoritması (look-ahead yok).
    # Smoothed versiyon (predict_proba) canlı son gün için daha pürüzsüz olsa da
    # geçmiş noktalar için gelecek veriyi kullanır → backtest eğimi.
    if cfg.use_filtered_probs:
        prob_df = compute_filtered_probability_vector(model, features)
    else:
        prob_df = compute_probability_vector(model, features)
    raw_returns = np.log(df["Close"] / df["Close"].shift(1)).dropna()

    # Label switching koruması: eski metadata'daki means ile Hungarian align
    perm = align_regime_labels(
        model, old_means=None, metadata_path=cfg.metadata_path
    )
    if not np.array_equal(perm, np.arange(best_K)):
        logger.info("Label switching tespiti — prob_df sütunları yeniden sıralanıyor")
        prob_df = reorder_prob_df(prob_df, perm)

    plot_regimes(df["Close"], prob_df, cfg)
    save_model(model, best_K, candidates, prob_df, raw_returns, cfg)

    last_probs = {k: round(float(v), 3) for k, v in prob_df.iloc[-1].items()}
    logger.info("=== DONE — K=%d, last_day_probs=%s ===", best_K, last_probs)
    return prob_df


if __name__ == "__main__":
    run_pipeline()
