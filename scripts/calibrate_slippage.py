#!/usr/bin/env python3
"""
scripts/calibrate_slippage.py — N24: Almgren-Chriss γ kalibrasyonu.

Kullanım:
    python scripts/calibrate_slippage.py

Gereksinimler:
    - data/paper_trades.parquet  (entry_px, exit_px, gross_pnl_pct, net_pnl_pct,
                                   trade_value, adv, sigma kolonları beklenir)
    - numpy, pandas, scipy

Çıktı:
    - Kalibre edilmiş γ terminale yazdırılır
    - data/slippage_calibration.json dosyasına kaydedilir

SlippageConfig kullanımı (örnek):
    # engine/execution/paper_trader.py içinde:
    #   from engine.core.alpha_cfg import SlippageConfig
    #   cfg = SlippageConfig(gamma=<kalibre_gamma>)
    #   slip = cfg.gamma * (trade_value / adv) ** 0.5 * sigma
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

DATA_DIR = Path("data")
TRADES_PATH = DATA_DIR / "paper_trades.parquet"
OUTPUT_PATH = DATA_DIR / "slippage_calibration.json"


def load_trades() -> pd.DataFrame:
    """paper_trades.parquet yükle, gerekli kolonları kontrol et."""
    if not TRADES_PATH.exists():
        raise FileNotFoundError(
            f"{TRADES_PATH} bulunamadı. Paper trader'ın çalışmış olması gerekir."
        )
    df = pd.read_parquet(TRADES_PATH)
    print(f"Yüklendi: {len(df):,} satır, kolonlar: {list(df.columns)}")
    return df


def filter_valid_trades(df: pd.DataFrame) -> pd.DataFrame:
    """entry_px ve exit_px olan geçerli işlemleri filtrele."""
    required = {"entry_px", "exit_px", "gross_pnl_pct", "net_pnl_pct"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Eksik kolonlar: {missing}")

    mask = df["entry_px"].notna() & df["exit_px"].notna()
    mask &= df["gross_pnl_pct"].notna() & df["net_pnl_pct"].notna()
    filtered = df[mask].copy()
    print(f"Geçerli trade: {len(filtered):,} / {len(df):,}")
    return filtered


def compute_realized_slip(df: pd.DataFrame) -> pd.Series:
    """realized_slip = gross_pnl_pct - net_pnl_pct (işlem maliyeti etkisi)."""
    return df["gross_pnl_pct"] - df["net_pnl_pct"]


def build_regressor(df: pd.DataFrame) -> pd.Series:
    """
    X = (trade_value / ADV)^0.5 * σ

    ADV ve sigma kolonları yoksa basit yaklaşımlar kullanılır:
    - adv eksikse trade_value'nun medyanı kullanılır
    - sigma eksikse entry_px'in %2'si kullanılır (kaba tahmin)
    """
    if "adv" in df.columns:
        adv = df["adv"].clip(lower=1.0)
    elif "trade_value" in df.columns:
        adv_median = df["trade_value"].median()
        adv = pd.Series(adv_median, index=df.index)
        print(f"ADV kolonu yok; trade_value medyanı kullanılıyor: {adv_median:,.0f}")
    else:
        adv = pd.Series(1e6, index=df.index)
        print("ADV ve trade_value yok; sabit 1M kullanılıyor.")

    if "sigma" in df.columns:
        sigma = df["sigma"].clip(lower=1e-6)
    elif "entry_px" in df.columns:
        sigma = df["entry_px"] * 0.02
        print("Sigma kolonu yok; entry_px * 0.02 kullanılıyor.")
    else:
        sigma = pd.Series(0.02, index=df.index)
        print("Sigma tahmini: sabit 0.02")

    if "trade_value" in df.columns:
        tv = df["trade_value"].clip(lower=0.0)
    else:
        tv = pd.Series(1e5, index=df.index)
        print("trade_value kolonu yok; sabit 100K kullanılıyor.")

    X = (tv / adv).clip(lower=0.0) ** 0.5 * sigma
    return X


def fit_gamma(y: pd.Series, X: pd.Series) -> tuple[float, float, float]:
    """
    OLS: realized_slip = γ × X + ε  (intercept=False)

    Döner: (gamma, r_squared, std_err)
    """
    mask = X.notna() & y.notna() & np.isfinite(X) & np.isfinite(y) & (X > 0)
    X_clean = X[mask].values
    y_clean = y[mask].values

    if len(X_clean) < 5:
        raise ValueError(
            f"OLS için yeterli gözlem yok ({len(X_clean)} < 5). "
            "Daha fazla paper trade biriktirin."
        )

    # scipy OLS (intercept yok)
    slope, intercept, r_value, p_value, std_err = scipy_stats.linregress(X_clean, y_clean)
    print(f"\nOLS sonuçları (intercept dahil referans için):")
    print(f"  slope (γ)  : {slope:.6f}")
    print(f"  intercept  : {intercept:.6f}")
    print(f"  R²         : {r_value**2:.4f}")
    print(f"  p-value    : {p_value:.4e}")
    print(f"  std_err    : {std_err:.6f}")

    # Intercept=False versiyon (origin üzerinden OLS)
    gamma_no_intercept = float(np.dot(X_clean, y_clean) / np.dot(X_clean, X_clean))
    ss_res = np.sum((y_clean - gamma_no_intercept * X_clean) ** 2)
    ss_tot = np.sum((y_clean - y_clean.mean()) ** 2)
    r2_no_int = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    return gamma_no_intercept, r2_no_int, std_err


def save_calibration(gamma: float, r2: float, n_obs: int) -> None:
    """Kalibre γ'yı JSON olarak kaydet."""
    payload = {
        "gamma": gamma,
        "r_squared": r2,
        "n_observations": n_obs,
        "model": "Almgren-Chriss: slip = gamma * (trade_value / ADV)^0.5 * sigma",
        "usage_example": (
            "SlippageConfig(gamma=<gamma>) → "
            "slip = gamma * (trade_value / adv)**0.5 * sigma"
        ),
    }
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(payload, indent=2))
    print(f"\nKayıt: {OUTPUT_PATH}")


def main() -> None:
    print("=== Almgren-Chriss γ Kalibrasyonu ===\n")
    df = load_trades()
    df = filter_valid_trades(df)

    y = compute_realized_slip(df)
    X = build_regressor(df)

    gamma, r2, std_err = fit_gamma(y, X)

    print(f"\n--- Kalibre Edilmiş Sonuç ---")
    print(f"  γ (gamma)  : {gamma:.6f}")
    print(f"  R² (origin): {r2:.4f}")
    print(f"  N gözlem   : {len(df):,}")
    print(
        f"\nKullanım: SlippageConfig(gamma={gamma:.4f})\n"
        f"  → slip = {gamma:.4f} * (trade_value / adv)**0.5 * sigma"
    )

    save_calibration(gamma, r2, n_obs=len(df))
    print("Tamamlandı.")


if __name__ == "__main__":
    main()
