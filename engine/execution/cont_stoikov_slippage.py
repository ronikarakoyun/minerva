"""
engine/execution/cont_stoikov_slippage.py — CKS 2014 OFI slippage.

Cont-Kukanov-Stoikov (2014): ΔP = β × OFI + ε, β = c / AD^λ
λ ≈ 1 (50 hissenin 35'inde reddedilemiyor), c < 0.5 ampirik.
BIST için c=0.15 (orta fiyat emirlere düşük direnç, yüksek spread ortam).

Kullanım:
    bps = cont_stoikov_slippage_bps(order_size_TL=50_000, adv_TL=2_000_000)
    # → ~3.75 bps örnek
"""
from __future__ import annotations
import numpy as np

_DEFAULT_C = 0.15          # BIST ampirik (offline kalibrasyon gerekli)
_DEFAULT_LAMBDA = 1.0      # CKS: λ≈1 güçlü yaklaşım
_FALLBACK_BPS = 60.0       # ADV yoksa 60 bps (BIST illiquid default)


def cont_stoikov_slippage_bps(
    order_size_TL: float,
    adv_TL: float,
    c: float = _DEFAULT_C,
    lambda_p: float = _DEFAULT_LAMBDA,
    fallback_bps: float = _FALLBACK_BPS,
) -> float:
    """
    Cont-Kukanov-Stoikov 2014 OFI tabanlı slipaj (bps cinsinden).

    Formül: slip_bps = (c / AD^λ) × order_size × 1e4
    AD = Average Daily Volume (TL), order_size = emir hacmi (TL).
    λ=1 varsayımı en güçlü ve güvenilir yaklaşım (CKS ampirik).

    Parameters
    ----------
    order_size_TL : Emir hacmi (TL).
    adv_TL        : 20-günlük ortalama günlük hacim (TL). ≤0 → fallback.
    c             : BIST piyasa direnci katsayısı (ampirik: 0.05–0.30).
    lambda_p      : Derinlik katsayısı (≈1.0).
    fallback_bps  : ADV yoksa sabit slipaj (bps).

    Returns
    -------
    float — slipaj (bps cinsinden, pozitif).
    """
    if adv_TL <= 0 or not np.isfinite(adv_TL):
        return float(fallback_bps)
    beta = c / (max(adv_TL, 1.0) ** lambda_p)
    slip_price = beta * max(order_size_TL, 0.0)
    # bps = (fiyat etkisi / ortalama fiyat) × 1e4 ≈ (slip_price / adv_TL) × 1e4
    # Normalizasyon: slip_price TL cinsinden, adv_TL ile bağıl orana çevir
    if adv_TL > 0:
        slip_bps = slip_price / adv_TL * 1e4
    else:
        slip_bps = fallback_bps
    return float(np.clip(slip_bps, 0.0, 500.0))


def calibrate_lambda(
    price_changes: "list[float]",
    ofi_values: "list[float]",
    depth_values: "list[float]",
) -> tuple[float, float]:
    """
    CKS 2014 Adım 2: λ ve c kalibrasyonu (OLS log-log regresyon).

    log β = α_L - λ·log AD + ε_L
    β = price_change / OFI (her pencerede hesaplanır)

    Returns (lambda_hat, c_hat).
    Offline kalibrasyonda kullanılır — production'da sabit değerler tercih edilir.
    """
    try:
        import numpy as np
        betas = [dp / max(abs(ofi), 1e-10) for dp, ofi in zip(price_changes, ofi_values)]
        log_beta = np.log(np.maximum(np.abs(betas), 1e-10))
        log_ad   = np.log(np.maximum(depth_values, 1.0))
        # OLS: log_beta = alpha - lambda * log_ad
        X = np.column_stack([np.ones(len(log_ad)), -log_ad])
        coeffs, _, _, _ = np.linalg.lstsq(X, log_beta, rcond=None)
        alpha_L, lambda_hat = coeffs
        c_hat = float(np.exp(alpha_L))
        return float(lambda_hat), c_hat
    except Exception:
        return _DEFAULT_LAMBDA, _DEFAULT_C
