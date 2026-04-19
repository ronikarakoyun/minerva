"""
engine/deflated_sharpe.py — Deflated Sharpe Ratio (DSR) hesaplayıcı.

Bailey & López de Prado (2014): "The Deflated Sharpe Ratio: Correcting for
Selection Bias, Skewness, and Kurtosis".

Sorun: N formül deneriz, en iyisini seçeriz. En iyinin SR'ı kısmen şans.
DSR: Havuz büyüklüğü, skew, kurt, T göz önüne alınarak SR deflate edilir.
Sonuç: p-value — "bu SR sadece şans eseri değil" olasılığı.

Formüller (Bailey-LdP 2014, Eq. 4-7):
  SE[SR]     = sqrt((1 - skew·SR + (kurt/4)·SR²) / T)
  E[max SR*] = SE_null · {(1-γ)·Φ⁻¹(1-1/N) + γ·Φ⁻¹(1-1/(N·e))}
  DSR_z      = (SR_hat - E[max SR*]) / SE[SR_hat]
  p_value    = Φ(DSR_z)

Burada γ = Euler-Mascheroni sabiti ≈ 0.5772.
SE_null = 1/sqrt(T) (gürültü havuzu için null SE, skew=0, kurt=0, SR=0).
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats


# Euler-Mascheroni sabiti
_GAMMA_EM: float = 0.5772156649015328


def compute_sharpe_series(
    equity: pd.Series,
    freq: int = 252,
) -> tuple[float, float, float]:
    """
    Equity eğrisinden annualize edilmiş Sharpe Ratio, skewness ve kurtosis hesapla.

    Parametreler
    ------------
    equity : pd.Series
        Kümülatif portföy değeri (başlangıç=1 veya serbestçe).
    freq : int
        Yıllık iş günü sayısı (252 BIST için uygun).

    Döner
    ------
    (SR, skew, kurt) : tuple[float, float, float]
        - SR: annualized Sharpe (aritmetik ortalama / std * sqrt(freq))
        - skew: günlük getiri serisi skewness'ı
        - kurt: excess kurtosis (normal dağılım için 0)
    """
    if equity is None or len(equity) < 5:
        return (np.nan, 0.0, 0.0)

    rets = equity.pct_change().dropna()
    if len(rets) < 5 or rets.std() == 0:
        return (np.nan, 0.0, 0.0)

    sr   = float(rets.mean() / rets.std() * math.sqrt(freq))
    skw  = float(stats.skew(rets))
    kurt = float(stats.kurtosis(rets))   # excess kurtosis
    return (sr, skw, kurt)


def expected_max_sr_null(n_trials: int, T: int) -> float:
    """
    N bağımsız gürültü formülünün beklenen maksimum SR'ı (null dağılım).

    Bailey-LdP Proposition 2: Gürültü SE = 1/sqrt(T), SR_null = 0.

    E[max SR_null] = (1/sqrt(T)) · {(1-γ)·Φ⁻¹(1-1/N) + γ·Φ⁻¹(1-1/(N·e))}

    Parametreler
    ------------
    n_trials : int
        Toplam denenen formül sayısı (havuz büyüklüğü).
    T : int
        Gözlem sayısı (iş günü).

    Döner
    ------
    float : Beklenen maksimum null SR değeri.
    """
    if n_trials <= 1 or T < 2:
        return 0.0

    se_null = 1.0 / math.sqrt(T)

    # Φ⁻¹(1 - 1/N) ve Φ⁻¹(1 - 1/(N·e))
    p1 = max(0.0, min(1.0 - 1.0 / n_trials, 1.0 - 1e-12))
    p2 = max(0.0, min(1.0 - 1.0 / (n_trials * math.e), 1.0 - 1e-12))
    z1 = stats.norm.ppf(p1)
    z2 = stats.norm.ppf(p2)

    return float(se_null * ((1.0 - _GAMMA_EM) * z1 + _GAMMA_EM * z2))


def deflated_sharpe_ratio(
    sr: float,
    T: int,
    skew: float,
    kurt: float,
    n_trials: int,
) -> tuple[float, float]:
    """
    Deflated Sharpe Ratio (Bailey & LdP 2014, Eq. 5-7).

    DSR_z   = (SR_hat − E[max SR_null]) / SE[SR_hat]
    p_value = Φ(DSR_z)

    Parametreler
    ------------
    sr : float
        Seçilen formülün annualized Sharpe Ratio'su.
    T : int
        Gözlem sayısı (iş günü).
    skew : float
        Getiri serisi çarpıklığı (skewness).
    kurt : float
        Getiri serisi fazla kurtosis'i (excess kurtosis).
    n_trials : int
        Havuzdaki toplam formül sayısı.

    Döner
    ------
    (DSR_z, p_value) : tuple[float, float]
        - DSR_z  : z-skoru; >0 → SR anlamlı (gürültü barını aştı)
        - p_value: Φ(DSR_z) ∈ [0,1]; ≥0.95 → anlamlı (tek taraflı α=5%)

    Not: SR annualized (×√252) fakat T, freq'e normalize EDİLMEMİŞ ham gün sayısı.
    SE[SR_hat] formülünde SR²/4 baskın olduğunda büyük SR'da SE artar; bu kasıtlı.
    """
    if not np.isfinite(sr) or T < 5 or n_trials < 1:
        return (np.nan, np.nan)

    # SE[SR_hat] — Bailey Proposition 1 (skew ve kurt düzeltmesi)
    # var_sr = (1 - skew·SR + (kurt/4)·SR²) / T
    var_sr = (1.0 - skew * sr + (kurt / 4.0) * sr ** 2) / max(T, 1)
    # Negatif olmaması için kenet: minimum 1/T
    var_sr = max(var_sr, 1.0 / max(T, 1))
    se_sr  = math.sqrt(var_sr)

    # Beklenen maksimum null SR (N bağımsız gürültü formülünden)
    e_max = expected_max_sr_null(n_trials, T)

    # DSR_z: SR, gürültü barını kaç SE üstünde geçiyor?
    dsr_z   = float((sr - e_max) / se_sr)
    p_value = float(stats.norm.cdf(dsr_z))
    return (dsr_z, p_value)


def compute_pool_dsr(
    equity_curves: "dict[str, pd.Series]",
    n_trials: Optional[int] = None,
    freq: int = 252,
) -> pd.DataFrame:
    """
    Havuzdaki tüm formüller için DSR hesapla.

    Parametreler
    ------------
    equity_curves : dict[str, pd.Series]
        Anahtar = formül metni, değer = kümülatif equity eğrisi.
    n_trials : int, opsiyonel
        Havuz büyüklüğü; None → len(equity_curves).
    freq : int
        Yıllık iş günü sayısı.

    Döner
    ------
    pd.DataFrame sütunları:
        formula, SR, skew, kurt, T, DSR_z, p_value, significant
    """
    if n_trials is None:
        n_trials = len(equity_curves)

    rows = []
    for formula, eq in equity_curves.items():
        if eq is None or len(eq) < 10:
            rows.append({
                "formula": formula, "SR": np.nan, "skew": np.nan,
                "kurt": np.nan, "T": 0, "DSR_z": np.nan,
                "p_value": np.nan, "significant": False,
            })
            continue

        sr, skw, kurt = compute_sharpe_series(eq, freq=freq)
        rets = eq.pct_change().dropna()
        T    = len(rets)
        dsr_z, pv = deflated_sharpe_ratio(sr, T, skw, kurt, n_trials)
        rows.append({
            "formula":     formula,
            "SR":          round(sr, 4) if np.isfinite(sr) else np.nan,
            "skew":        round(skw, 3),
            "kurt":        round(kurt, 3),
            "T":           T,
            "DSR_z":       round(dsr_z, 3) if np.isfinite(dsr_z) else np.nan,
            "p_value":     round(pv, 4) if np.isfinite(pv) else np.nan,
            "significant": bool(np.isfinite(pv) and pv >= 0.95),  # tek taraflı α=5%
        })

    df = pd.DataFrame(rows)
    if not df.empty and "SR" in df.columns:
        df = df.sort_values("SR", ascending=False).reset_index(drop=True)
    return df
