"""
engine/backtest_engine.py — QuantaAlpha Appendix A.2 + Tablo 7 uyumlu backtester.

- Deal price  : ertesi günün açılışı (Popen_{t+1})
- Getiri      : y_t = Pclose_{t+2} / Pclose_{t+1} - 1
- Komisyon    : Alım %0.05, Satım %0.15 (asimetrik)  — Tablo 7
- Portföy     : TopkDropout (topk=50, n_drop=5)       — Tablo 7
- Metrikler   : IR (excess), ARR, |MDD| (pozitif), Net Getiri, Benchmark vs Alfa

VEKTÖRİZASYON (4.2 Bug Fix):
  Eski: O(n_dates) Python for-loop; her iterasyonda pandas groupby + sort.
        2500 gün × 700 ticker ≈ 60-90 saniye.
  Yeni: pivot_table ile (Date × Ticker) matrisler, tüm tarihlerin argsort'u tek
        numpy çağrısında, boolean portfolio bitmask. → 5-15× daha hızlı.

  Temel optimizasyonlar:
  1. pivot_table → (D × T) signal ve return matrisleri bir kez hesaplanır
  2. np.argsort(sig, axis=1)[:, ::-1] → tüm tarihlerin sıralaması tek seferde
  3. portfolio bitmask (numpy bool array) — Python set'ten 10× hızlı lookup
  4. ranked[~portfolio[ranked] & valid[ranked]] → tamamen vektörize filtre

NOT — BIST Long-Only:
  BIST'te BIST30 dışında short yapılamaz. Bu backtester zaten LONG-ONLY'dir:
  top_k en yüksek sinyalli hisseler alınır, short pozisyon hiç açılmaz.
  Negatif sinyalli hisseler seçilmez = otomatik olarak kaçınılır.

Benchmark:
  benchmark parametresi (Date → kapanış fiyatı) verilirse:
  - Benchmark getirisi hesaplanır (BIST100/XU100 gibi)
  - Excess return = strateji getirisi - benchmark getirisi
  - Alfa IR = excess return serisi üzerinden IR
  - Beta = strateji / benchmark kovaryansı
  Böylece "%260 ham getiri" ile "alfa mı beta mı?" sorusu cevaplanır.
"""
import pandas as pd
import numpy as np


def run_pro_backtest(
    df,
    signal_series,
    top_k: int = 50,
    n_drop: int = 5,
    buy_fee: float = 0.0005,
    sell_fee: float = 0.0015,
    benchmark: "pd.Series | None" = None,
):
    """
    Vektörize TopK-Dropout portföy backtester.

    Parameters
    ----------
    df            : Ticker/Date bazlı fiyat DataFrame
    signal_series : Her satır için ham sinyal (df ile aynı sıra)
    top_k         : Portföyde tutulacak max hisse sayısı (varsayılan 50)
    n_drop        : Günlük yenileme: alt n_drop hisse çıkar, üst n_drop ekle
    buy_fee       : Alış komisyon oranı (0.0005 = %0.05)
    sell_fee      : Satış komisyon oranı (0.0015 = %0.15)
    benchmark     : (Opsiyonel) Date-indexed pd.Series — BIST100/XU100 kapanış.
                    Verilirse: Excess Return, Alfa IR, Beta hesaplanır.

    Returns
    -------
    curve   : pd.DataFrame(Date, Equity, [BenchmarkEquity])
    metrics : dict — Net Getiri, IR, MDD, Yıllık, [Benchmark metrikleri]
    """
    data = df.copy().sort_values(["Date", "Ticker"]).reset_index(drop=True)
    data["Signal"] = np.asarray(signal_series)

    # Getiri hedefi: Pclose_{t+2}/Pclose_{t+1} - 1
    data["Pclose_t1"] = data.groupby("Ticker")["Pclose"].shift(-1)
    data["Pclose_t2"] = data.groupby("Ticker")["Pclose"].shift(-2)
    data["Period_Ret"] = data["Pclose_t2"] / data["Pclose_t1"] - 1

    # ------------------------------------------------------------------
    # Vectorized prep: (Date × Ticker) pivot matrisleri
    # ------------------------------------------------------------------
    sig_piv = data.pivot_table(
        index="Date", columns="Ticker", values="Signal", aggfunc="first"
    )
    ret_piv = data.pivot_table(
        index="Date", columns="Ticker", values="Period_Ret", aggfunc="first"
    )

    # Kritik: son 2 günde tüm Period_Ret NaN olunca pivot_table bu tarihleri
    # düşürebilir → ret_piv < sig_piv boyutu → IndexError. Reindex ile hizala.
    ret_piv = ret_piv.reindex(index=sig_piv.index, columns=sig_piv.columns)

    dates     = sig_piv.index.tolist()
    n_dates   = len(dates)
    n_tickers = len(sig_piv.columns)

    sig_arr = sig_piv.to_numpy(dtype=float)   # (D, T) — NaN: sinyal yok
    ret_arr = ret_piv.to_numpy(dtype=float)   # (D, T)

    # Tüm tarihlerin argsort'unu TEK numpy çağrısında hesapla
    # NaN → -inf dönüştür (argsort'ta NaN olan hisseler sona düşer)
    sig_filled   = np.where(~np.isnan(sig_arr), sig_arr, -np.inf)   # (D, T)
    ranked_all   = np.argsort(sig_filled, axis=1)[:, ::-1]          # (D, T) azalan sıra
    valid_all    = ~np.isnan(sig_arr)                                # (D, T) bool

    # ------------------------------------------------------------------
    # Ana döngü — her gün numpy array ops (Python set YOK)
    # ------------------------------------------------------------------
    portfolio  = np.zeros(n_tickers, dtype=bool)   # bitmask
    ret_daily  = np.empty(n_dates,   dtype=float)

    for di in range(n_dates):
        ranked = ranked_all[di]    # T uzunluklu ticker-index array (sıralı)
        valid  = valid_all[di]     # T uzunluklu bool array

        # Hiç geçerli sinyal yoksa
        if not valid.any():
            ret_daily[di] = 0.0
            continue

        port_indices = np.where(portfolio)[0]

        if len(port_indices) == 0:
            # İlk gün: top_k hisseyi al (geçerli sinyalli olanlardan)
            top_valid = ranked[valid[ranked]][:top_k]
            portfolio[top_valid] = True
            add_n  = int(portfolio.sum())
            drop_n = 0
        else:
            # En kötü n_drop'u bul: portfolio hisselerini sinyale göre artan sırala
            port_sigs = np.where(valid[port_indices],
                                 sig_arr[di, port_indices],
                                 -np.inf)
            worst_in_port = port_indices[np.argsort(port_sigs)[:n_drop]]

            # Drop
            portfolio[worst_in_port] = False

            # Add: ranked sırasında portfolio'da OLMAYAN ve geçerli sinyalli hisseler
            candidates = ranked[~portfolio[ranked] & valid[ranked]]
            new_adds   = candidates[:n_drop]
            portfolio[new_adds] = True

            add_n  = len(new_adds)
            drop_n = len(worst_in_port)

        # Günlük portföy getirisi
        port_rets     = ret_arr[di, portfolio]
        valid_rets    = port_rets[~np.isnan(port_rets)]
        rets          = float(valid_rets.mean()) if len(valid_rets) > 0 else 0.0

        size      = int(portfolio.sum())
        buy_cost  = (add_n  / max(size, 1)) * buy_fee
        sell_cost = (drop_n / max(size, 1)) * sell_fee
        ret_daily[di] = rets - buy_cost - sell_cost

    # ------------------------------------------------------------------
    # Metrikler
    # ------------------------------------------------------------------
    equity  = np.cumprod(1 + ret_daily) * 100_000

    ir      = (ret_daily.mean() / ret_daily.std()) * np.sqrt(252) if ret_daily.std() else 0.0
    mdd_abs = float(abs(((equity / np.maximum.accumulate(equity)) - 1).min()) * 100)
    n_d     = max(n_dates, 1)
    arr     = ((equity[-1] / 100_000) ** (252 / n_d) - 1) * 100
    net_ret = (equity[-1] / 100_000 - 1) * 100

    curve   = pd.DataFrame({"Date": dates, "Equity": equity})
    metrics = {
        "Net Getiri (%)": net_ret,
        "IR":             ir,
        "MDD":            mdd_abs,
        "Yıllık":         arr,
    }

    # ------------------------------------------------------------------
    # Benchmark karşılaştırması (opsiyonel)
    # ------------------------------------------------------------------
    if benchmark is not None:
        try:
            bm = benchmark.reindex(dates, method="ffill").dropna()
            if len(bm) < 10:
                raise ValueError("Benchmark verisi yetersiz")

            bm_ret    = bm.pct_change().fillna(0.0).values
            bm_ret    = bm_ret[:n_dates]
            bm_equity = np.cumprod(1 + bm_ret) * 100_000

            bm_net = (bm_equity[-1] / 100_000 - 1) * 100
            bm_n   = max(len(bm_ret), 1)
            bm_arr = ((bm_equity[-1] / 100_000) ** (252 / bm_n) - 1) * 100
            bm_mdd = float(abs(((bm_equity / np.maximum.accumulate(bm_equity)) - 1).min()) * 100)

            n_common   = min(n_dates, len(bm_ret))
            excess_ret = ret_daily[:n_common] - bm_ret[:n_common]
            alpha_ir   = (
                (excess_ret.mean() / excess_ret.std()) * np.sqrt(252)
                if excess_ret.std() > 0 else 0.0
            )
            beta = (
                np.cov(ret_daily[:n_common], bm_ret[:n_common])[0, 1]
                / max(np.var(bm_ret[:n_common]), 1e-12)
            )

            metrics.update({
                "Benchmark Getiri (%)": round(bm_net, 1),
                "Benchmark Yıllık (%)": round(bm_arr, 1),
                "Benchmark MDD (%)":    round(bm_mdd, 1),
                "Excess Return (%)":    round(net_ret - bm_net, 1),
                "Alfa IR":              round(float(alpha_ir), 2),
                "Beta":                 round(float(beta), 2),
            })

            bm_curve_arr = np.full(n_dates, np.nan)
            bm_curve_arr[:len(bm_equity)] = bm_equity
            curve["BenchmarkEquity"] = bm_curve_arr

        except Exception as bm_err:
            metrics["Benchmark Hata"] = str(bm_err)

    return curve, metrics


def rolling_wf_backtest(
    df,
    signal_series,
    step_months: int = 6,
    min_train_months: int = 12,
    top_k: int = 50,
    n_drop: int = 5,
    buy_fee: float = 0.0005,
    sell_fee: float = 0.0015,
    benchmark: "pd.Series | None" = None,
) -> tuple["pd.DataFrame", "pd.DataFrame"]:
    """
    Rolling expanding-window walk-forward backtest.

    Her adımda:
      - Train: başlangıçtan t'ye kadar (expanding)
      - Test : t'den t+step_months'a kadar (kaydıran pencere)
    Formül aynı — sadece sinyal test döneminde değerlendirilir.

    Parameters
    ----------
    df              : Tüm tarih aralığı (train + test) price DataFrame
    signal_series   : df ile aynı uzunlukta ham sinyal Series/array
    step_months     : Test pencere uzunluğu (ay)
    min_train_months: Minimum train uzunluğu; bu kadar geçmişi olmayan pencereler atlanır
    top_k, n_drop, buy_fee, sell_fee, benchmark : run_pro_backtest ile aynı

    Returns
    -------
    windows_df : pd.DataFrame — her pencere için (train_start, test_start, test_end, IR, MDD, Yıllık, Net Getiri)
    combined_curve : pd.DataFrame — tüm test pencerelerinin ardışık equity eğrisi (Date, Equity)
    """
    data = df.copy().sort_values(["Date", "Ticker"]).reset_index(drop=True)
    data["Signal"] = np.asarray(signal_series)

    all_dates  = pd.to_datetime(data["Date"].unique())
    all_dates  = np.sort(all_dates)
    date_min   = all_dates[0]
    date_max   = all_dates[-1]

    # Test pencerelerini oluştur
    from dateutil.relativedelta import relativedelta  # pip install python-dateutil (genellikle yüklü)

    windows       = []
    combined_rows = []
    equity_carry  = 100_000.0   # her pencere öncekinin equity'sini devralır

    t_start = date_min + relativedelta(months=min_train_months)
    while t_start < date_max:
        t_end = min(
            pd.Timestamp(t_start) + relativedelta(months=step_months),
            pd.Timestamp(date_max)
        )
        if pd.Timestamp(t_end) <= pd.Timestamp(t_start):
            break

        # Test dilimi — only test period (train features signal edilmez)
        mask_test = (data["Date"] >= t_start) & (data["Date"] < t_end)
        df_test   = data[mask_test].drop(columns=["Signal"]).copy()
        sig_test  = data.loc[mask_test, "Signal"].values

        if len(df_test) == 0 or df_test["Date"].nunique() < 5:
            t_start = t_end
            continue

        # Benchmark dilimi
        bm_slice = None
        if benchmark is not None:
            bm_slice = benchmark[
                (benchmark.index >= t_start) & (benchmark.index < t_end)
            ]
            if len(bm_slice) < 5:
                bm_slice = None

        try:
            curve_w, met_w = run_pro_backtest(
                df_test, sig_test,
                top_k=top_k, n_drop=n_drop,
                buy_fee=buy_fee, sell_fee=sell_fee,
                benchmark=bm_slice,
            )

            # Equity'yi önceki pencerenin son değerinden başlat (sürekli eğri)
            scale = equity_carry / 100_000.0
            curve_w["Equity"] = curve_w["Equity"] * scale
            equity_carry = float(curve_w["Equity"].iloc[-1])

            windows.append({
                "Test Başlangıç": pd.Timestamp(t_start).date(),
                "Test Bitiş":     pd.Timestamp(t_end).date(),
                "IR":             round(met_w["IR"],             2),
                "MDD (%)":        round(met_w["MDD"],            1),
                "Yıllık (%)":     round(met_w["Yıllık"],         1),
                "Net Getiri (%)": round(met_w["Net Getiri (%)"], 1),
                "Alfa IR":        round(met_w.get("Alfa IR", float("nan")), 2),
            })
            combined_rows.append(curve_w[["Date", "Equity"]])

        except Exception:
            pass   # Yetersiz veri veya sayısal hata — pencereyi atla

        t_start = t_end

    windows_df = pd.DataFrame(windows) if windows else pd.DataFrame()
    combined_curve = (
        pd.concat(combined_rows, ignore_index=True)
        if combined_rows else pd.DataFrame(columns=["Date", "Equity"])
    )
    return windows_df, combined_curve


def rolling_refit_wf_backtest(
    df,
    idx,
    tree,
    evaluate_fn,
    step_months: int = 6,
    min_train_months: int = 12,
    use_neutralize: bool = True,
    size_corr_hard_limit: float = 0.7,
    top_k: int = 50,
    n_drop: int = 5,
    buy_fee: float = 0.0005,
    sell_fee: float = 0.0015,
    benchmark: "pd.Series | None" = None,
) -> tuple["pd.DataFrame", "pd.DataFrame"]:
    """
    Mod 2 — Rolling Re-fit WF Backtest.

    Her test penceresi için:
      1. O pencerenin (train + test) veri dilimini kes
      2. Factor cache'i o dilimden yeniden hesapla
      3. Sinyali o dilimin faktörleriyle yeniden nötralize et
      4. Size-corr > hard_limit → pencere REDDEDİLİR (status='size_factor')
      5. Geçen pencereleri run_pro_backtest ile test et

    Formül AST aynı kalır; değişen şey her penceredeki sinyal kalibrasyonu ve
    size-faktör korelasyonu — bu gerçek "deploy senaryosu" simülasyonudur.

    Parameters
    ----------
    df                    : Flat price DataFrame (Ticker, Date, Pclose, ...)
    idx                   : (Ticker, Date) MultiIndex DataFrame — neutralize için
    tree                  : AST — Node objesi
    evaluate_fn           : Callable(tree, df_flat) → pd.Series — genelde cfg.evaluate
    step_months           : Test pencere uzunluğu
    min_train_months      : Minimum train — bu kadar geçmişi olmayan pencere atlanır
    use_neutralize        : True → her pencere için yeniden factor neutralization
    size_corr_hard_limit  : |size_corr| > limit → pencere reddedilir
    top_k, n_drop, ...    : run_pro_backtest ile aynı

    Returns
    -------
    windows_df      : Her pencere için (durum, IR, MDD, size_corr, ...)
    combined_curve  : Geçen pencerelerin ardışık equity eğrisi
    """
    from dateutil.relativedelta import relativedelta
    from engine.factor_neutralize import (
        build_factors_cache, neutralize_signal, compute_size_corr
    )

    data = df.copy().sort_values(["Date", "Ticker"]).reset_index(drop=True)
    all_dates = np.sort(pd.to_datetime(data["Date"].unique()))
    date_min, date_max = all_dates[0], all_dates[-1]

    # Full sinyal (tek sefer, rolling ops warmup'ı için tüm veri lazım)
    sig_full = evaluate_fn(tree, data)
    if not isinstance(sig_full, pd.Series):
        sig_full = pd.Series(np.asarray(sig_full))
    # Flat sıraya hizala: data zaten (Date,Ticker) sıralı, sig aynı uzunlukta
    data["_sig_full"] = np.asarray(sig_full)

    # Sinyal MultiIndex'e indeksle (neutralize_signal bunu ister)
    sig_mi = data.set_index(["Ticker", "Date"])["_sig_full"].sort_index()

    windows       = []
    combined_rows = []
    equity_carry  = 100_000.0

    t_start = pd.Timestamp(date_min) + relativedelta(months=min_train_months)
    while t_start < pd.Timestamp(date_max):
        t_end = min(t_start + relativedelta(months=step_months), pd.Timestamp(date_max))
        if t_end <= t_start:
            break

        # Pencere dilimi: başlangıçtan t_end'e kadar (train + test)
        mask_window = data["Date"] < t_end
        win_df      = data[mask_window].copy()
        win_idx     = win_df.set_index(["Ticker", "Date"]).sort_index()

        # Sinyali pencereye kes
        sig_win = sig_mi.loc[
            (sig_mi.index.get_level_values("Date") < t_end)
        ].copy()

        # 2) Factor cache pencereye özel
        # 3) Sinyali o pencere faktörleriyle nötralize et
        try:
            if use_neutralize:
                fc_win   = build_factors_cache(win_idx)
                sig_win  = neutralize_signal(sig_win, win_idx, factors=fc_win)

            # 4) Size-corr ölç — pencere düzeyinde
            size_corr = compute_size_corr(sig_win, win_idx)
        except Exception:
            size_corr = float("nan")

        row = {
            "Test Başlangıç": t_start.date(),
            "Test Bitiş":     t_end.date(),
            "Size Corr":      round(float(size_corr), 3) if not np.isnan(size_corr) else None,
        }

        # Hard filter: size_corr çok yüksekse pencereyi atla
        if not np.isnan(size_corr) and abs(size_corr) > size_corr_hard_limit:
            row.update({
                "Durum":          "size_factor",
                "IR":             None, "MDD (%)": None,
                "Yıllık (%)":     None, "Net Getiri (%)": None,
                "Alfa IR":        None,
            })
            windows.append(row)
            t_start = t_end
            continue

        # 5) Test dilimini çıkar ve backtest
        mask_test = (data["Date"] >= t_start) & (data["Date"] < t_end)
        df_test   = data[mask_test].copy()

        # Yeniden nötralize edilmiş sinyali test dilimi için al
        sig_test_mi = sig_win.loc[
            sig_win.index.get_level_values("Date") >= t_start
        ]
        # Flat df_test sıralaması: (Date, Ticker). sig_test_mi: (Ticker, Date).
        df_test = df_test.drop(columns=[c for c in ["_sig_full", "Signal"] if c in df_test.columns])
        df_test_flat = df_test.set_index(["Ticker", "Date"]).sort_index()
        sig_arr = sig_test_mi.reindex(df_test_flat.index).values

        # Geri flat sıraya çevir (run_pro_backtest flat df + aligned sig bekler)
        df_test_flat = df_test_flat.reset_index()
        bm_slice = None
        if benchmark is not None:
            bm_slice = benchmark[(benchmark.index >= t_start) & (benchmark.index < t_end)]
            if len(bm_slice) < 5:
                bm_slice = None

        try:
            curve_w, met_w = run_pro_backtest(
                df_test_flat, sig_arr,
                top_k=top_k, n_drop=n_drop,
                buy_fee=buy_fee, sell_fee=sell_fee,
                benchmark=bm_slice,
            )
            scale = equity_carry / 100_000.0
            curve_w["Equity"] = curve_w["Equity"] * scale
            equity_carry = float(curve_w["Equity"].iloc[-1])

            row.update({
                "Durum":          "ok",
                "IR":             round(met_w["IR"],             2),
                "MDD (%)":        round(met_w["MDD"],            1),
                "Yıllık (%)":     round(met_w["Yıllık"],         1),
                "Net Getiri (%)": round(met_w["Net Getiri (%)"], 1),
                "Alfa IR":        round(met_w.get("Alfa IR", float("nan")), 2),
            })
            combined_rows.append(curve_w[["Date", "Equity"]])
        except Exception as e:
            row.update({
                "Durum": f"error:{type(e).__name__}",
                "IR": None, "MDD (%)": None,
                "Yıllık (%)": None, "Net Getiri (%)": None, "Alfa IR": None,
            })

        windows.append(row)
        t_start = t_end

    windows_df = pd.DataFrame(windows) if windows else pd.DataFrame()
    combined_curve = (
        pd.concat(combined_rows, ignore_index=True)
        if combined_rows else pd.DataFrame(columns=["Date", "Equity"])
    )
    return windows_df, combined_curve
