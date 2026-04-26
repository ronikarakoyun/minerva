import random
import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import torch

from engine.alpha_cfg import AlphaCFG, Node
from engine.alpha_catalog import get_tree, load_catalog, save_alpha
from engine.backtest_engine import run_pro_backtest
from engine.formula_parser import parse_many
from engine.mcts import GrammarMCTS
from engine.replay_buffer import ReplayBuffer
from engine.tree_lstm import PolicyValueNet, build_action_vocab, build_token_vocab
from engine.trainer import TreeLSTMTrainer
from engine.wf_fitness import compute_wf_fitness, make_date_folds, make_purged_date_folds, wf_verdict
from engine.triple_barrier import add_triple_barrier_to_idx, label_stats
from engine.factor_neutralize import build_factors_cache

st.set_page_config(page_title="Minerva v3 Studio Pro", layout="wide")

# ------------------------------------------------------------------
# Session state
# ------------------------------------------------------------------
if "alphas" not in st.session_state:
    # Katalogdan önceki bulunanları yükle
    _cat = load_catalog()
    if _cat:
        st.session_state.alphas = pd.DataFrame([{
            "Formül": r["formula"],
            "IC":     r["ic"],
            "RankIC": r["rank_ic"],
            "Adj IC": r["adj_ic"],
        } for r in _cat])
    else:
        st.session_state.alphas = pd.DataFrame()
if "trees" not in st.session_state:
    # AST'leri katalogdan geri yükle
    _cat = load_catalog()
    st.session_state.trees = {
        r["formula"]: get_tree(r["formula"])
        for r in _cat
        if get_tree(r["formula"]) is not None
    }
if "train_hist" not in st.session_state:
    st.session_state.train_hist = []  # [{total,value,policy,n,buffer},...]
# Split-date sentinel: önceki split ile üretilmiş ağaçlar yeni split'te
# veri sızıntısına yol açar → split değişince in-memory sonuçları temizle.
if "_split_date_sentinel" not in st.session_state:
    st.session_state["_split_date_sentinel"] = None


@st.cache_data
def load_data():
    from config import load_paths
    _paths = load_paths()
    df = pd.read_parquet(_paths["market_db"])
    df["Date"] = pd.to_datetime(df["Date"])
    # Savunma: yfinance artımsal fetch veya ticker listesi tekrarı duplicate üretebilir;
    # aynı (Ticker, Date) için ilkini tut — TB/WF indeks-uniqueness sağlar.
    _n0 = len(df)
    df = df.drop_duplicates(subset=["Ticker", "Date"], keep="first")
    if len(df) < _n0:
        import warnings
        warnings.warn(f"load_data: {_n0 - len(df)} duplicate (Ticker,Date) satırı atıldı.")
    if "Pvwap" not in df.columns:
        # NOT: Pvwap = (High + Low + Close) / 3 = typical price (HLC/3)
        # Gerçek VWAP = Σ(Price × Volume) / Σ(Volume) gün-içi barlardan hesaplanır.
        # End-of-day verisi kullandığımızdan gerçek VWAP mevcut değil.
        # Pvwap adı tarihsel; doğrusu Ptyp (typical price) olmalı.
        # İleride intraday data gelirse burası güncellenecek.
        df["Pvwap"] = (df["Phigh"] + df["Plow"] + df["Pclose"]) / 3
    return df


@st.cache_data
def load_benchmark(path: str) -> "pd.Series | None":
    """
    BIST100 / XU100 benchmark yükle (Date → kapanış fiyatı).

    Beklenen format: CSV veya Parquet, 'Date' ve 'Close' (veya 'Pclose') sütunları.
    Dosya yoksa None döner — backtest benchmark olmadan çalışır.
    """
    import os
    if not os.path.exists(path):
        return None
    try:
        if path.endswith(".parquet"):
            bm = pd.read_parquet(path)
        else:
            bm = pd.read_csv(path)
        bm["Date"] = pd.to_datetime(bm["Date"])
        close_col = next((c for c in ["Close", "Pclose", "close", "CLOSE"] if c in bm.columns), None)
        if close_col is None:
            return None
        return bm.set_index("Date")[close_col].sort_index()
    except Exception:
        return None


@st.cache_resource
def get_brain(device: str):
    """Tree-LSTM, trainer, replay buffer, vocab — oturumlar arası cache'li."""
    cfg          = AlphaCFG()
    token_vocab  = build_token_vocab(cfg)
    action_vocab = build_action_vocab(cfg)
    net          = PolicyValueNet(
        token_vocab_size=len(token_vocab),
        action_size=len(action_vocab),
    )
    trainer = TreeLSTMTrainer(net, token_vocab, device=device).load()
    buffer  = ReplayBuffer().load()
    return {
        "cfg": cfg, "vocab": token_vocab, "actions": action_vocab,
        "net": net, "trainer": trainer, "buffer": buffer,
    }


# ------------------------------------------------------------------
# Data & brain
# ------------------------------------------------------------------
db = load_data()

# Benchmark (opsiyonel — data/bist100.csv veya .parquet yoksa None)
_benchmark_series = load_benchmark("data/bist100.parquet")
if _benchmark_series is None:
    _benchmark_series = load_benchmark("data/bist100.csv")

st.title("🔬 Minerva v3: Evolutionary Alpha Factory")
st.markdown("AlphaCFG (α-Sem-k + MCTS + Tree-LSTM) × QuantaAlpha (trajectory evolution)")

# ------------------------------------------------------------------
# Train/Test Split — KRİTİK: tüm mining işlemleri train'de yapılır.
# Test penceresine ne LLM paneli ne evrimsel döngü ne Tree-LSTM dokunur.
# Test sadece overfit validasyonu için ayrıldı.
# ------------------------------------------------------------------
_date_min = pd.to_datetime(db["Date"].min())
_date_max = pd.to_datetime(db["Date"].max())
_default_split = _date_min + (_date_max - _date_min) * 0.7

st.sidebar.header("📅 Train / Test Split")
split_date = st.sidebar.date_input(
    "Split tarihi (öncesi=TRAIN, sonrası=TEST):",
    value=_default_split.date(),
    min_value=_date_min.date(),
    max_value=_date_max.date(),
)
split_ts = pd.to_datetime(split_date)
db_train = db[db["Date"] < split_ts].copy()
db_test  = db[db["Date"] >= split_ts].copy()

# Veri sızıntısı koruması: split_date değişince önceki split'te üretilmiş
# tüm in-memory ağaçları ve alpha tablosunu temizle. Katalog dosyası
# dokunulmadan kalır; kullanıcı tekrar mining yaparak yeniden doldurabilir.
_current_sentinel = str(split_date)
if st.session_state["_split_date_sentinel"] not in (None, _current_sentinel):
    st.session_state.trees  = {}
    st.session_state.alphas = pd.DataFrame()
    st.warning(
        f"⚠️ Split tarihi değişti → önceki mining sonuçları temizlendi. "
        f"Yeni split: **{split_date}** — veri sızıntısı önlendi."
    )
st.session_state["_split_date_sentinel"] = _current_sentinel

_n_tr, _n_te = len(db_train), len(db_test)
_pct = 100 * _n_tr / max(len(db), 1)
st.sidebar.caption(
    f"🟢 Train: **{_n_tr:,}** satır ({_pct:.0f}%)  \n"
    f"🔴 Test: **{_n_te:,}** satır ({100-_pct:.0f}%)  \n"
    f"📊 {_date_min.date()} → **{split_date}** → {_date_max.date()}"
)

# Üstte büyük uyarı/bilgi bandı
st.info(
    f"🔒 **Train/Test ayrımı aktif** — mining sadece train'de ({_date_min.date()} → {split_date}). "
    f"Test penceresi ({split_date} → {_date_max.date()}) overfit validasyonu için korunuyor."
)

# ------------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------------
st.sidebar.header("⚙️ Madencilik")
num_gen = st.sidebar.slider("Popülasyon Büyüklüğü", 100, 1000, 300)
max_K   = st.sidebar.slider("Maksimum Uzunluk (K)", 10, 30, 15)

st.sidebar.header("🔬 WF-Fitness (Mining içi stabilite)")
use_wf_fitness = st.sidebar.checkbox(
    "Walk-Forward fitness kullan", value=True,
    help="Her formül için train'i K fold'a ayırır, fitness = mean(ric) - 2·std(ric) - λ·complexity. "
         "Tek-dönem şans eseri formülleri elenir. Mining süresi ~%30 artar."
)
wf_n_folds = st.sidebar.slider("Mining içi fold sayısı", 3, 8, 5,
                               disabled=not use_wf_fitness)
wf_embargo = st.sidebar.slider(
    "Fold embargo (gün)", 0, 15, 5,
    disabled=not use_wf_fitness,
    help=(
        "Her fold sınırının başından ve sonundan bu kadar gün çıkarılır. "
        "Triple-Barrier kullanıyorsan horizon ile eşit tut (varsayılan 5). "
        "Bu sayede fold sınırlarındaki label leakage önlenir."
    ),
)
wf_purge = st.sidebar.slider(
    "Purge horizon (gün)", 0, 20, 10,
    disabled=not use_wf_fitness,
    help=(
        "Purged K-Fold (López de Prado, AFML §7): Her test fold başlangıcından "
        "geriye bu kadar gün, train kümesinden çıkarılır. "
        "TB horizon kadar ileriye bakan label'ların fold sınırından sızmasını engeller. "
        "0 = purge yok (eski davranış); ≥1 = purge aktif."
    ),
)
# Purge ↔ return_window güvenlik kontrolü
if use_wf_fitness and int(wf_purge) > 0:
    _min_purge = int(tb_horizon) if target_mode.startswith("🎯") else 2
    if int(wf_purge) < _min_purge:
        st.sidebar.warning(
            f"⚠️ Purge horizon ({wf_purge} gün) < return window ({_min_purge} gün) — "
            "fold sınırlarında label leakage riski! "
            f"En az **{_min_purge}** olarak ayarla."
        )
wf_lambda_std = st.sidebar.slider("λ_std (stabilite cezası)", 0.0, 4.0, 0.5, 0.5,
                                   disabled=not use_wf_fitness,
                                   help="Yüksek → sadece çok tutarlı formüller geçer. "
                                        "BIST'te RankIC ~0.02 olduğundan 0.5 önerilir.")
wf_lambda_cx = st.sidebar.slider("λ_complexity (sadelik cezası)", 0.0, 0.01, 0.001, 0.001,
                                  disabled=not use_wf_fitness,
                                  format="%.3f",
                                  help="Yüksek → kısa formüller tercih edilir (overfit koruması)")

# Benchmark durumu
if _benchmark_series is not None:
    st.sidebar.success(
        f"📊 Benchmark yüklendi: {len(_benchmark_series):,} gün  \n"
        f"({_benchmark_series.index.min().date()} → {_benchmark_series.index.max().date()})"
    )
else:
    st.sidebar.info(
        "📊 Benchmark yok — `data/bist100.csv` veya `.parquet` ekle  \n"
        "Sütunlar: `Date`, `Close`  \n"
        "Olmadan: ham getiri (beta + alfa karışık) gösterilir."
    )

st.sidebar.header("⚗️ Faktör Nötralizasyonu")
use_neutralize = st.sidebar.checkbox(
    "Size/Vol/Mom faktörlerini nötralize et", value=True,
    help=(
        "Mining'in size factor (küçük hisse = iyi getiri) tuzağına düşmesini önler. "
        "Her formül sinyali cross-sectional OLS ile size, volatilite ve momentum'a "
        "regresse edilir; artık (residual) 'saf alfa' olarak kullanılır. "
        "Nötralizasyon olmadığında bile size_corr > 0.3 olan formüller fitness penaltısı alır."
    ),
)
wf_lambda_size = st.sidebar.slider(
    "λ_size (size-korelasyon cezası)", 0.0, 2.0, 0.5, 0.1,
    disabled=use_neutralize,
    help="Nötralizasyon kapalıyken aktif: |size_corr| > 0.3 olan formüller cezalandırılır. "
         "Nötralizasyon açıkken bu parametre gereksiz olur.",
)
size_corr_limit = st.sidebar.slider(
    "Size-corr hard limit", 0.3, 1.0, 0.7, 0.05,
    help=(
        "Ham sinyalde |size_corr| bu eşiği aşarsa formül direkt reddedilir. "
        "Pclose, Plow, Pvwap gibi fiyat-seviyesi formülleri (~1.0) eleniri. "
        "0.7 önerilir: gerçek alfa formüller genellikle 0.5 altında."
    ),
)
use_regime_breakdown = st.sidebar.checkbox(
    "Rejim ayrıştırması (bull/chop/bear)", value=False,
    help=(
        "Mining sonrası formüllerin IC'ini piyasa rejimine göre ayrıştırır.  \n"
        "Benchmark (BIST100) gerekli. Panel formül detayında 3-satır rejim tablosu görünür.  \n"
        "Kural: trend_60g > 0 + vol_20g < %2.2 → bull; aksi → chop/bear."
    ),
)

st.sidebar.subheader("🧠 Meta-Label (ikincil model)")
use_meta_label = st.sidebar.checkbox(
    "Meta-Label filtresi uygula", value=False,
    help=(
        "López de Prado AFML §3.6: Formül sinyalini logistic regression ile filtreler.  \n"
        "Düşük güvenli pozisyonlar (p < threshold) atlanır → daha temiz giriş sinyali.  \n"
        "TB_Label gereklidir. Eğit butonu Backtest Bölümünde görünür."
    ),
)
meta_threshold = st.sidebar.slider(
    "Meta eşiği (threshold)", 0.4, 0.8, 0.55, 0.05,
    disabled=not use_meta_label,
    help="P(TB_Label=1) < threshold olan pozisyonlar skip edilir.",
)

st.sidebar.header("🎯 Hedef Değişkeni")
target_mode = st.sidebar.radio(
    "Tahmin hedefi:",
    ["📈 Next_Ret (klasik)", "🎯 Triple-Barrier (risk-adjusted)"],
    index=0,
    help=(
        "Next_Ret: ertesi gün getirisi — hızlı ama gürültülü.\n"
        "Triple-Barrier: N gün içinde ±k×σ bariyerini geçerse ±1, geçmezse 0. "
        "Risk-adjusted sinyal — küçük gürültülü günler 0 alır, "
        "IC kalitesi yükselir ama hesaplama daha yavaş."
    ),
)
if target_mode.startswith("🎯"):
    tb_col1, tb_col2 = st.sidebar.columns(2)
    tb_horizon    = tb_col1.number_input("Horizon (gün)", 5, 30, 10)
    tb_multiplier = tb_col2.number_input("Çarpan (×σ)", 0.5, 3.0, 1.5, 0.5)
    tb_long_only  = st.sidebar.checkbox(
        "Long-only mod (BIST gerçeği)", value=True,
        help=(
            "BIST'te BIST30 dışında short işlem yapılamaz. "
            "Açık → alt bariyer etiketleri -1 yerine 0 olur. "
            "IC artık binary: 1=güçlü yükseliş bekleniyor, 0=kaçın/bekle. "
            "Mining, 'hangisi yükselir?' sorusunu optimize eder."
        ),
    )
else:
    tb_long_only = True   # Kullanılmaz ama tanımlı olsun

st.sidebar.header("💰 Backtest Parametreleri")
bt_top_k   = st.sidebar.slider("Top-K portföy büyüklüğü", 10, 200, 50,
                                help="Günlük olarak sinyal sıralamasında ilk K hisse alınır.")
bt_n_drop  = st.sidebar.slider("N-Drop günlük yenileme", 1, 30, 5,
                                help="Her gün en kötü N hisse çıkar, en iyi N yeni hisse girer.")
_buy_pct   = st.sidebar.slider("Alış komisyonu (%)", 0.0, 0.30, 0.05, 0.01, format="%.2f",
                                help="Broker alış komisyonu. BIST tipik: %0.04-0.08")
_sell_pct  = st.sidebar.slider("Satış komisyonu (%)", 0.0, 0.50, 0.15, 0.01, format="%.2f",
                                help="Satış = komisyon + BSMV + damga. BIST tipik: %0.12-0.18")
bt_buy_fee  = _buy_pct  / 100.0
bt_sell_fee = _sell_pct / 100.0
st.sidebar.caption(
    f"Alış: %{_buy_pct:.2f} · Satış: %{_sell_pct:.2f} · "
    f"Toplam gidiş-dönüş: %{_buy_pct + _sell_pct:.2f}"
)

st.sidebar.header("⚡ Performans")
use_parallel_eval = st.sidebar.checkbox(
    "Paralel Faz 3 (joblib)", value=True,
    help=(
        "Faz 3 değerlendirmeyi birden fazla CPU çekirdeğinde paralel çalıştır. "
        "300+ formülde ~3-5× hızlanma. joblib kurulu olmalı: pip install joblib. "
        "Hata olursa otomatik seri moda geçer."
    ),
)

st.sidebar.header("🧠 Tree-LSTM")
device     = st.sidebar.selectbox("Cihaz", ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"])
brain      = get_brain(device)
cfg        = brain["cfg"]
token_vocab = brain["vocab"]
net        = brain["net"]
trainer    = brain["trainer"]
buffer     = brain["buffer"]

use_value_fn = st.sidebar.checkbox("Tree-LSTM value_fn (MCTS'de)", value=True)
use_mcts     = st.sidebar.checkbox("Grammar-aware MCTS", value=False)
mcts_iters   = st.sidebar.slider("MCTS iter", 50, 1000, 200, disabled=not use_mcts)

st.sidebar.metric("Replay buffer", f"{len(buffer)} örnek")

col_tr1, col_tr2 = st.sidebar.columns(2)
train_epochs = col_tr1.number_input("Epoch", 1, 100, 5)
batch_size   = col_tr2.number_input("Batch", 4, 256, 32)

if st.sidebar.button("🏋️ Tree-LSTM eğit"):
    if len(buffer) < 4:
        st.sidebar.warning("Buffer çok küçük. Önce bir evrim döngüsü çalıştırın.")
    else:
        bar = st.sidebar.progress(0.0)
        txt = st.sidebar.empty()
        def cb(e, total, entry):
            bar.progress(e / total)
            txt.text(f"ep {e}/{total}  val_loss={entry['value']:.4f}")
        hist = trainer.train_epochs(buffer, epochs=int(train_epochs),
                                    batch_size=int(batch_size),
                                    progress_cb=cb)
        st.session_state.train_hist.extend(hist)
        trainer.save()
        st.sidebar.success(f"✅ {len(hist)} epoch eğitildi, kaydedildi")

if st.sidebar.button("🗑️ Buffer sıfırla"):
    buffer.clear()
    buffer.save()
    st.sidebar.info("Buffer temizlendi")

st.sidebar.divider()
st.sidebar.header("🗄️ Veri Tabanı")
if st.sidebar.button("🔄 Veritabanını Yenile (yfinance)"):
    try:
        from engine.db_builder import build_database
        with st.sidebar.status("📥 Veri indiriliyor (bu işlem dakikalar sürebilir)...",
                               expanded=True) as _db_status:
            build_database()
            _db_status.update(label="✅ Veritabanı güncellendi!", state="complete")
        # Cache'i temizle — yeni veri yüklensin
        load_data.clear()
        st.sidebar.success("Veriler güncellendi. Sayfayı yenileyin.")
    except ImportError:
        st.sidebar.error("yfinance kurulu değil: pip install yfinance")
    except Exception as _dbe:
        st.sidebar.error(f"Veritabanı güncellenemedi: {_dbe}")


# ------------------------------------------------------------------
# LLM çıktılarını buffer'a besle
# ------------------------------------------------------------------
with st.expander("🤖 LLM formüllerini Tree-LSTM'e öğret", expanded=False):
    st.caption(
        "Harici LLM'den (ChatGPT, Claude, Gemini vb.) aldığın formülleri "
        "her satıra bir tane olacak şekilde yapıştır. Sistem parse edip "
        "IC hesaplayacak ve replay buffer'a ekleyecek. Sonra yan panelden "
        "**Tree-LSTM eğit** butonuna bas."
    )
    st.code(
        "Rank(Mul(Sub(Pclose, Popen), Vlot), 20)\n"
        "CSRank(Delta(Pvwap, 30))\n"
        "Corr(Pclose, Vlot, 40)\n"
        "Div(0.05, Std(Pclose, 20))",
        language="text",
    )
    llm_text = st.text_area("Formüller (her satır bir tane):", height=180,
                            key="llm_formulas")
    llm_use_wf = st.checkbox(
        "LLM formüller için WF-Fitness hesapla (daha yavaş)",
        value=False,
        help=(
            "Her LLM formülü için mining ayarlarındaki WF-Fitness parametrelerini kullanarak "
            f"{wf_n_folds}-fold walk-forward test yapılır. Kataloga fitness bilgisi kaydedilir. "
            "WF-Fitness kapalıysa bu seçenek devre dışı."
        ),
        disabled=not use_wf_fitness,
    )

    col_lm1, col_lm2 = st.columns([1, 3])
    if col_lm1.button("📥 Parse & Değerlendir"):
        if not llm_text.strip():
            st.warning("Önce formül yapıştır.")
        else:
            # Veriyi hazırla — SADECE TRAIN (test'e dokunulmaz)
            with st.spinner("Veri hazırlanıyor (train-only)..."):
                _db = db_train.sort_values(["Ticker", "Date"])
                _db["Pclose_t1"] = _db.groupby("Ticker")["Pclose"].shift(-1)
                _db["Pclose_t2"] = _db.groupby("Ticker")["Pclose"].shift(-2)
                _db["Next_Ret"]  = _db["Pclose_t2"] / _db["Pclose_t1"] - 1
                _eval = _db[["Date", "Ticker", "Popen", "Phigh", "Plow",
                             "Pclose", "Vlot", "Pvwap", "Next_Ret"]].copy()
                _idx = _eval.set_index(["Ticker", "Date"]).sort_index()

            parsed = parse_many(llm_text, cfg)
            # Atlanmış (boş/yorum) satır sayısını hesapla
            _total_input_lines = sum(
                1 for ln in llm_text.splitlines()
                if ln.strip() and not ln.strip().startswith("#")
            )
            _skipped = _total_input_lines - len(parsed)
            if _skipped > 0:
                st.caption(f"ℹ️ {_skipped} satır yorum veya boş — atlandı.")
            if not parsed:
                st.warning("Parse edilecek formül bulunamadı. Yorum (#) ve boş satırları kaldır.")
            rows = []
            alpha_rows = []   # session_state.alphas'a eklenecekler
            bar_llm = st.progress(0.0)
            for i, (raw, tree, err) in enumerate(parsed):
                if tree is None:
                    rows.append({"Formül": raw, "Durum": f"❌ parse: {err}",
                                 "IC": None, "RankIC": None})
                    bar_llm.progress((i + 1) / len(parsed))
                    continue
                try:
                    sig = cfg.evaluate(tree, _idx)
                    tmp = pd.DataFrame({
                        "Date":     _idx.index.get_level_values("Date"),
                        "Signal":   sig.values,
                        "Next_Ret": _idx["Next_Ret"].values,
                    }).dropna()
                    if len(tmp) == 0:
                        rows.append({"Formül": raw, "Durum": "⚠️ boş sinyal",
                                     "IC": None, "RankIC": None})
                    else:
                        def _ic(g, method):
                            if g["Signal"].std() == 0:
                                return 0
                            return g["Signal"].corr(g["Next_Ret"], method=method)
                        ic      = tmp.groupby("Date").apply(lambda g: _ic(g, "pearson")).mean()
                        rank_ic = tmp.groupby("Date").apply(lambda g: _ic(g, "spearman")).mean()
                        if not np.isnan(rank_ic):
                            buffer.add(tree, float(rank_ic))
                            s = str(tree)
                            st.session_state.trees[s] = tree

                            # WF-Fitness (LLM formüller için opsiyonel)
                            _llm_wf_kwargs = {}
                            if llm_use_wf and use_wf_fitness:
                                try:
                                    _llm_dates = _idx.index.get_level_values("Date").values
                                    if int(wf_purge) > 0:
                                        _llm_folds = make_purged_date_folds(
                                            _llm_dates, n_folds=int(wf_n_folds),
                                            min_fold_days=20, embargo_days=int(wf_embargo),
                                            purge_horizon=int(wf_purge),
                                            return_window=2,  # LLM paneli her zaman Next_Ret (2-gün pencere)
                                        )
                                    else:
                                        _llm_folds = make_date_folds(
                                            _llm_dates, n_folds=int(wf_n_folds),
                                            min_fold_days=20, embargo_days=int(wf_embargo),
                                        )
                                    if len(_llm_folds) >= 3:
                                        _llm_fc = None
                                        if use_neutralize or wf_lambda_size > 0:
                                            from engine.factor_neutralize import build_factors_cache as _bfc
                                            try:
                                                _llm_fc = _bfc(_idx)
                                            except Exception:
                                                pass
                                        _llm_stats = compute_wf_fitness(
                                            tree, cfg.evaluate, _idx, _llm_folds,
                                            lambda_std=float(wf_lambda_std),
                                            lambda_cx=float(wf_lambda_cx),
                                            min_valid_folds=3,
                                            target_col="Next_Ret",  # LLM paneli her zaman Next_Ret
                                            neutralize=use_neutralize,
                                            factor_cache=_llm_fc,
                                            lambda_size=float(wf_lambda_size),
                                            size_corr_hard_limit=float(size_corr_limit),
                                        )
                                        _llm_wf_kwargs = dict(
                                            wf_mean_ric=_llm_stats["mean_ric"],
                                            wf_std_ric=_llm_stats["std_ric"],
                                            wf_pos_folds=_llm_stats["pos_folds"],
                                            wf_n_folds=len(_llm_stats["fold_rics"]),
                                            wf_fitness=_llm_stats["fitness"],
                                            wf_fold_rics=_llm_stats["fold_rics"],
                                            complexity=_llm_stats["complexity"],
                                        )
                                        _wf_badge = (
                                            f" | WF={_llm_stats['fitness']:.4f} "
                                            f"({_llm_stats['pos_folds']}/{len(_llm_stats['fold_rics'])} pos)"
                                        )
                                    else:
                                        _wf_badge = " | WF: yeterli fold yok"
                                except Exception as _wfe:
                                    _wf_badge = f" | WF: hata ({_wfe})"
                            else:
                                _wf_badge = ""

                            rows.append({
                                "Formül": raw,
                                "Durum":  f"✅ buffer +1{_wf_badge}",
                                "IC":     round(float(ic), 4),
                                "RankIC": round(float(rank_ic), 4),
                            })
                            adj = abs(float(rank_ic))
                            alpha_rows.append({
                                "Formül": s,
                                "IC":     float(ic),
                                "RankIC": float(rank_ic),
                                "Adj IC": adj,
                            })
                            # Kataloğa kalıcı kaydet (WF-metadata dahil)
                            save_alpha(
                                formula=s, tree=tree,
                                ic=float(ic), rank_ic=float(rank_ic), adj_ic=adj,
                                split_date=str(split_date), source="llm",
                                **_llm_wf_kwargs,
                            )
                        else:
                            rows.append({"Formül": raw, "Durum": "⚠️ NaN IC",
                                         "IC": None, "RankIC": None})
                except Exception as e:
                    rows.append({"Formül": raw, "Durum": f"❌ eval: {e}",
                                 "IC": None, "RankIC": None})
                bar_llm.progress((i + 1) / len(parsed))

            buffer.save()

            # Alphas tablosunu birleştir (mevcut + yeni)
            if alpha_rows:
                new_df = pd.DataFrame(alpha_rows)
                if st.session_state.alphas.empty:
                    merged = new_df
                else:
                    merged = pd.concat([st.session_state.alphas, new_df],
                                       ignore_index=True)
                    merged = merged.drop_duplicates(subset=["Formül"], keep="last")
                merged = merged.sort_values("Adj IC", ascending=False).reset_index(drop=True)
                st.session_state.alphas = merged

            df_llm = pd.DataFrame(rows)
            st.dataframe(df_llm, use_container_width=True)
            ok = sum(1 for r in rows if r["Durum"].startswith("✅"))
            st.success(f"🎯 {ok}/{len(rows)} formül buffer'a eklendi, "
                       f"{len(alpha_rows)} tanesi backtest listesine geçti "
                       f"(toplam buffer = {len(buffer)})")

    col_lm2.caption(
        f"Geçerli operatörler: "
        f"{', '.join(list(cfg.UNARY_OPS) + list(cfg.BINARY_OPS) + list(cfg.BINARY_ASYM_OPS))} · "
        f"rolling: {', '.join(list(cfg.ROLLING_OPS))} · "
        f"paired: {', '.join(list(cfg.PAIRED_OPS))} · CSRank · "
        f"özellik: {', '.join(cfg.FEATURES)}"
    )


# ------------------------------------------------------------------
# Ana buton: evrimsel döngü
# ------------------------------------------------------------------
if st.sidebar.button("⛏️ Evrimsel Döngüyü Başlat"):
    t0 = time.time()
    st.info(f"🚀 Döngü başladı — popülasyon={num_gen}, K={max_K}, "
            f"MCTS={'ON' if use_mcts else 'OFF'}, "
            f"value_fn={'Tree-LSTM' if use_value_fn else 'heuristik'}")

    _factor_cache = None   # Faz-0'da doldurulacak, Faz-3'te kullanılacak

    # ---------- Faz 0: Veri hazırlığı (TRAIN-ONLY) ----------
    with st.status("📦 Veri hazırlanıyor (train-only)...", expanded=False) as s0:
        db_sorted = db_train.sort_values(["Ticker", "Date"])
        db_sorted["Pclose_t1"] = db_sorted.groupby("Ticker")["Pclose"].shift(-1)
        db_sorted["Pclose_t2"] = db_sorted.groupby("Ticker")["Pclose"].shift(-2)
        db_sorted["Next_Ret"]  = db_sorted["Pclose_t2"] / db_sorted["Pclose_t1"] - 1
        eval_db = db_sorted[["Date", "Ticker", "Popen", "Phigh", "Plow",
                             "Pclose", "Vlot", "Pvwap", "Next_Ret"]].copy()
        idx = eval_db.set_index(["Ticker", "Date"]).sort_index()

        # Triple-Barrier hedefi seçildiyse etiketleri hesapla
        _target_col = "Next_Ret"
        if target_mode.startswith("🎯"):
            with st.spinner(f"🎯 Triple-Barrier etiketleri hesaplanıyor "
                            f"(horizon={tb_horizon}, ×{tb_multiplier}σ)..."):
                idx = add_triple_barrier_to_idx(
                    idx,
                    horizon=int(tb_horizon),
                    multiplier=float(tb_multiplier),
                    long_only=tb_long_only,
                )
                _target_col = "TB_Label"
                stats_tb = label_stats(idx["TB_Label"].dropna())
                _mode_label = "Long-only (0/1)" if tb_long_only else "Long/Short (-1/0/1)"
                s0.update(
                    label=(
                        f"✅ Train + TB_Label hazır ({len(idx):,} satır) — "
                        f"{_mode_label}  |  "
                        f"buy {stats_tb['buy    (+1)']:.0%} / "
                        f"flat {stats_tb['flat    (0)']:.0%}"
                        + (f" / short {stats_tb.get('short  (-1)', 0):.0%}" if not tb_long_only else "")
                    ),
                    state="complete"
                )
        else:
            s0.update(label=f"✅ Train verisi hazır ({len(idx):,} satır)", state="complete")

        # --- Faktör cache'i (size/vol/mom) — bir kez hesaplanır ---
        _factor_cache = None
        _regime_series = None  # Rejim serisi (opsiyonel)
        if use_wf_fitness and (use_neutralize or wf_lambda_size > 0):
            with st.spinner("⚗️ Faktör matrisi hesaplanıyor (size/vol/mom)..."):
                try:
                    # Rejim serisi (benchmark varsa ve kullanıcı istiyorsa)
                    if use_regime_breakdown and _benchmark_series is not None:
                        try:
                            from engine.regime import compute_regime
                            _regime_series = compute_regime(_benchmark_series)
                        except Exception as _re:
                            st.caption(f"ℹ️ Rejim hesaplanamadı: {_re}")
                    _factor_cache = build_factors_cache(idx, attach_regime=_regime_series)
                except Exception as _fe:
                    st.warning(f"⚠️ Faktör cache hatası: {_fe} — nötralizasyon atlanacak")

    # Tree-LSTM value_fn bind
    value_fn = None
    if use_value_fn:
        value_fn = lambda n: net.predict_value(n, token_vocab)

    # ---------- Faz 1: Diversified Planning ----------
    st.write(f"### 🌱 Faz 1 — Başlangıç havuzu ({num_gen // 2})")
    p1_bar = st.progress(0.0); p1_txt = st.empty()
    pool: list[Node] = []
    if use_mcts:
        # Subtree prior (6.3): önceki başarılı formüllerin alt-ağaçlarına bonus
        _subtree_prior = {
            str(t): 0.1 for t in st.session_state.trees.values()
        } if st.session_state.trees else {}
        mcts = GrammarMCTS(
            cfg, max_K=max_K, value_fn=value_fn,
            subtree_prior=_subtree_prior,
        )
        for i in range(num_gen // 2):
            pool.append(mcts.search(iterations=mcts_iters))
            p1_bar.progress((i + 1) / (num_gen // 2))
            p1_txt.text(f"MCTS {i+1}/{num_gen // 2}")
    else:
        for i in range(num_gen // 2):
            pool.append(cfg.generate(max_K))
            if (i + 1) % max(1, num_gen // 20) == 0:
                p1_bar.progress((i + 1) / (num_gen // 2))
                p1_txt.text(f"üretildi {i+1}/{num_gen // 2}")
        p1_bar.progress(1.0)
    p1_txt.text(f"✅ Faz 1 — {len(pool)} formül ({time.time()-t0:.1f}s)")

    # ---------- Faz 2: Self-Evolution ----------
    # BUG FIX #1.1: pool[:20] Sorting Bug
    # Faz 1 formülleri henüz evaluate edilmedi — pool[:20] sadece oluşturma
    # sırasına göre ilk 20 (= rastgele, en iyi değil).
    #
    # Çözüm — "warm-start" stratejisi:
    #   · Önceki koşulardan bilinen iyi formüller varsa (session_state.trees)
    #     onları tohum havuzu olarak kullan.
    #   · İlk koşu (geçmiş yok) → Faz 1'in TÜM havuzu kullanılır.
    #     Rastgele seçim kaçınılmaz ama artık [:20] yerine pool_all var.
    _prev_trees = list(st.session_state.trees.values())
    if len(_prev_trees) >= 5:
        seed_pool = _prev_trees
        _seed_label = f"🔥 warm-start: {len(_prev_trees)} önceki alfa tohum olarak kullanılıyor"
    else:
        seed_pool = pool   # cold-start: Faz 1'in tüm havuzu ([:20] değil)
        _seed_label = "🌱 cold-start: Faz 1 havuzundan evrim (ilk koşu)"

    st.write(f"### 🧬 Faz 2 — Mutation & Crossover ({num_gen // 2})")
    st.caption(_seed_label)
    p2_bar = st.progress(0.0); p2_txt = st.empty()
    n2 = num_gen // 2
    for i in range(n2):
        if random.random() < 0.7:
            p1, p2 = random.sample(seed_pool, 2) if len(seed_pool) > 1 else (seed_pool[0], seed_pool[0])
            pool.append(cfg.crossover(p1, p2))
        else:
            pool.append(cfg.mutate(random.choice(seed_pool)))
        if (i + 1) % max(1, n2 // 20) == 0:
            p2_bar.progress((i + 1) / n2)
            p2_txt.text(f"evrim {i+1}/{n2}")
    p2_bar.progress(1.0)
    p2_txt.text(f"✅ Faz 2 — toplam {len(pool)} ({time.time()-t0:.1f}s)")

    # ---------- Faz 3: Evaluation + replay buffer ----------
    st.write(f"### 📊 Faz 3 — Değerlendirme ({len(pool)})")
    if use_wf_fitness:
        _neut_label = "🔵 nötralize aktif" if use_neutralize else f"⚠️ λ_size={wf_lambda_size}"
        st.caption(
            f"🔬 **WF-Fitness aktif** — her formül {wf_n_folds} fold'da test ediliyor  |  "
            f"fitness = mean(ric) − {wf_lambda_std}·std − {wf_lambda_cx}·|AST|  |  "
            f"⚗️ Faktör nötralizasyonu: {_neut_label}"
        )
    bar    = st.progress(0.0)
    status = st.empty()
    results      = []
    validated    = []

    # Fold'ları bir kere hesapla (embargo + purge ile fold sınırlarında label leakage koruması)
    mining_folds = None
    if use_wf_fitness:
        dates_arr = idx.index.get_level_values("Date").values
        if int(wf_purge) > 0:
            # Purged K-Fold: her fold için ayrı test/train split (LdP §7)
            _rw = int(tb_horizon) if target_mode.startswith("🎯") else 2
            mining_folds = make_purged_date_folds(
                dates_arr,
                n_folds=int(wf_n_folds),
                min_fold_days=20,
                embargo_days=int(wf_embargo),
                purge_horizon=int(wf_purge),
                return_window=_rw,
            )
        else:
            # Klasik non-overlapping fold (embargo var, purge yok)
            mining_folds = make_date_folds(
                dates_arr,
                n_folds=int(wf_n_folds),
                min_fold_days=20,
                embargo_days=int(wf_embargo),
            )

    # ─── Faz 3 ana döngü (WF veya klasik) ────────────────────────────────
    # WF modu: Faz A (hesaplama, opsiyonel paralel) + Faz B (seri sonuç işleme)
    # Klasik mod: tek seri döngü

    _wf_eval_kwargs = dict(
        lambda_std=float(wf_lambda_std),
        lambda_cx=float(wf_lambda_cx),
        min_valid_folds=3,
        target_col=_target_col,
        neutralize=use_neutralize,
        factor_cache=_factor_cache,
        lambda_size=float(wf_lambda_size),
        size_corr_hard_limit=float(size_corr_limit),
        regime=_regime_series if use_regime_breakdown else None,
    )

    def _eval_one_wf(tree):
        """Tek ağaç için WF-fitness hesapla (joblib worker veya seri)."""
        try:
            return compute_wf_fitness(
                tree, cfg.evaluate, idx, mining_folds, **_wf_eval_kwargs
            )
        except Exception as _e:
            return {
                "status": "error", "ic": float("nan"), "rank_ic": float("nan"),
                "fitness": -999.0, "mean_ric": 0.0, "std_ric": 0.0,
                "pos_folds": 0, "fold_rics": [], "complexity": 0,
                "size_corr": float("nan"),
            }

    if use_wf_fitness and mining_folds and len(mining_folds) >= 3:
        # ── Faz A: WF-fitness hesaplama ───────────────────────────────────
        all_wf_stats: list = []

        if use_parallel_eval:
            status.text("⚡ Paralel WF-fitness hesaplanıyor…")
            try:
                from concurrent.futures import ThreadPoolExecutor, as_completed
                _n_workers = min(4, max(1, len(pool) // 10))
                _results_map: dict = {}
                with ThreadPoolExecutor(max_workers=_n_workers) as _exe:
                    _futures = {_exe.submit(_eval_one_wf, tree): i for i, tree in enumerate(pool)}
                    _done = 0
                    for _fut in as_completed(_futures):
                        _idx = _futures[_fut]
                        try:
                            _results_map[_idx] = _fut.result()
                        except Exception:
                            _results_map[_idx] = {
                                "status": "error", "ic": float("nan"), "rank_ic": float("nan"),
                                "fitness": -999.0, "mean_ric": 0.0, "std_ric": 0.0,
                                "pos_folds": 0, "fold_rics": [], "complexity": 0,
                                "size_corr": float("nan"),
                            }
                        _done += 1
                        bar.progress(_done / len(pool) * 0.85)
                all_wf_stats = [_results_map[i] for i in range(len(pool))]
                status.text(f"⚡ Paralel hesaplama tamam — {len(pool)} formül  ({time.time()-t0:.1f}s)")
            except Exception as _pe:
                st.warning(f"⚠️ Paralel mod başarısız ({_pe}) — seri moda geçiliyor…")
                all_wf_stats = []   # fallback seri aşağıda

        if not all_wf_stats:   # seri fallback (veya paralel kapalı)
            all_wf_stats = []
            for i, tree in enumerate(pool):
                all_wf_stats.append(_eval_one_wf(tree))
                bar.progress((i + 1) / len(pool))
                if (i + 1) % max(1, len(pool) // 50) == 0:
                    status.text(f"WF hesaplanıyor {i+1}/{len(pool)}  •  {time.time()-t0:.1f}s")

        bar.progress(0.9)   # sonuç işleme başlıyor

        # ── Faz B: Sonuç işleme (seri — UI / session_state / catalog) ────
        for i, (tree, stats) in enumerate(zip(pool, all_wf_stats)):
            ic      = stats["ic"]
            rank_ic = stats["rank_ic"]

            # Replay buffer'a ekle
            if not np.isnan(rank_ic):
                buffer.add(tree, float(rank_ic))

            # Seçim kriteri: mean_ric pozitif VE en az %40 fold pozitif IC
            fit       = stats["fitness"]
            n_folds_v = len(stats["fold_rics"])
            pos_ratio = stats["pos_folds"] / max(n_folds_v, 1)
            passes    = (stats["status"] == "ok" and
                         stats["mean_ric"] > 0.003 and
                         pos_ratio >= 0.4)

            if passes:
                max_sim = max([cfg.similarity(tree, f) for f in validated]) if validated else 0.0
                adjusted = fit * (1 - max_sim)
                if adjusted > 0:
                    s = str(tree)
                    _sc = stats.get("size_corr", float("nan"))

                    # IC Drop % = (RankIC_train - WF_mean_ric) / |RankIC_train| × 100
                    _ic_drop = None
                    if not np.isnan(rank_ic) and abs(rank_ic) > 1e-6:
                        _ic_drop = round(
                            (float(rank_ic) - stats["mean_ric"]) / abs(float(rank_ic)) * 100, 1
                        )

                    results.append({
                        "Formül":    s,
                        "IC":        ic,
                        "RankIC":    rank_ic,
                        "WF mean":   round(stats["mean_ric"], 4),
                        "WF std":    round(stats["std_ric"], 4),
                        "+ fold":    f"{stats['pos_folds']}/{n_folds_v}",
                        "IC Drop %": _ic_drop,
                        "Fitness":   round(fit, 4),
                        "Adj IC":    float(adjusted),
                        "|AST|":     stats["complexity"],
                        "SizeCorr":  round(float(_sc), 3) if not np.isnan(_sc) else None,
                    })
                    validated.append(tree)
                    st.session_state.trees[s] = tree
                    save_alpha(
                        formula=s, tree=tree,
                        ic=float(ic) if not np.isnan(ic) else 0.0,
                        rank_ic=float(rank_ic) if not np.isnan(rank_ic) else 0.0,
                        adj_ic=float(adjusted),
                        split_date=str(split_date), max_k=max_K,
                        population=num_gen, mcts_iters=mcts_iters if use_mcts else None,
                        source="evolution",
                        wf_mean_ric=stats["mean_ric"],
                        wf_std_ric=stats["std_ric"],
                        wf_pos_folds=stats["pos_folds"],
                        wf_n_folds=n_folds_v,
                        wf_fitness=fit,
                        wf_fold_rics=stats["fold_rics"],
                        complexity=stats["complexity"],
                    )

            bar.progress(0.9 + 0.1 * (i + 1) / len(pool))
            if (i + 1) % max(1, len(pool) // 50) == 0:
                status.text(f"sonuçlar işleniyor {i+1}/{len(pool)}  •  "
                            f"geçerli alpha {len(results)}  •  "
                            f"buffer {len(buffer)}  •  {time.time()-t0:.1f}s")

    else:
        # --- Klasik mod (WF kapalı veya fold sayısı yetersiz) ---
        for i, tree in enumerate(pool):
            sig = cfg.evaluate(tree, idx)
            tmp = pd.DataFrame({
                "Date":     idx.index.get_level_values("Date"),
                "Signal":   sig.values,
                "Next_Ret": idx["Next_Ret"].values,
            }).dropna()

            if len(tmp) > 0:
                def _ic(g, method):
                    if g["Signal"].std() == 0:
                        return 0
                    return g["Signal"].corr(g["Next_Ret"], method=method)

                ic      = tmp.groupby("Date").apply(lambda g: _ic(g, "pearson")).mean()
                rank_ic = tmp.groupby("Date").apply(lambda g: _ic(g, "spearman")).mean()

                max_sim = max([cfg.similarity(tree, f) for f in validated]) if validated else 0.0
                adjusted = rank_ic * (1 - max_sim)

                if not np.isnan(rank_ic):
                    buffer.add(tree, float(rank_ic))

                if not np.isnan(adjusted) and adjusted > 0:
                    s = str(tree)
                    results.append({
                        "Formül": s, "IC": ic, "RankIC": rank_ic, "Adj IC": adjusted,
                    })
                    validated.append(tree)
                    st.session_state.trees[s] = tree
                    save_alpha(
                        formula=s, tree=tree,
                        ic=float(ic), rank_ic=float(rank_ic), adj_ic=float(adjusted),
                        split_date=str(split_date), max_k=max_K,
                        population=num_gen, mcts_iters=mcts_iters if use_mcts else None,
                        source="evolution",
                    )

            bar.progress((i + 1) / len(pool))
            if (i + 1) % max(1, len(pool) // 50) == 0:
                status.text(f"değerlendirilen {i+1}/{len(pool)}  •  "
                            f"geçerli alpha {len(results)}  •  "
                            f"buffer {len(buffer)}  •  {time.time()-t0:.1f}s")

    buffer.save()
    status.text(f"✅ Döngü tamam — {len(results)} alpha, buffer={len(buffer)} "
                f"({time.time()-t0:.1f}s)")
    st.success(f"🎉 Bitti: {len(results)} alpha, "
               f"replay buffer → {len(buffer)} örnek "
               f"({time.time()-t0:.1f} s)")

    df_res = pd.DataFrame(results)
    if not df_res.empty:
        sort_col = "Fitness" if use_wf_fitness and "Fitness" in df_res.columns else "Adj IC"
        df_res = df_res.sort_values(sort_col, ascending=False).reset_index(drop=True)
    st.session_state.alphas = df_res


# ------------------------------------------------------------------
# Training history plot
# ------------------------------------------------------------------
if st.session_state.train_hist:
    with st.expander("📈 Tree-LSTM eğitim geçmişi", expanded=False):
        h = pd.DataFrame(st.session_state.train_hist)
        st.line_chart(h[["value", "policy"]])
        st.caption(f"Son epoch: val_loss={h.iloc[-1]['value']:.4f}  •  "
                   f"buffer={h.iloc[-1]['buffer']}")


# ------------------------------------------------------------------
# Alpha Kataloğu — kalıcı kayıt / düzenleme
# ------------------------------------------------------------------
_catalog = load_catalog()
if _catalog:
    with st.expander(f"📚 Alpha Kataloğu ({len(_catalog)} kayıt)", expanded=False):

        # --- Filtreler ---
        fc1, fc2, fc3 = st.columns(3)
        filt_verdict = fc1.selectbox(
            "Overfit filtresi:",
            ["Tümü", "✅ Stabil", "⚠️ Biraz bozuldu", "❌ Overfit", "💀 İşaret döndü", "— (test yok)"],
        )
        filt_source = fc2.selectbox("Kaynak:", ["Tümü", "llm", "evolution"])
        filt_bt     = fc3.checkbox("Sadece backtest yapılmışlar", value=False)

        # Filtreyi uygula
        _filtered = _catalog
        if filt_verdict != "Tümü":
            _v = "—" if filt_verdict == "— (test yok)" else filt_verdict
            _filtered = [r for r in _filtered
                         if r.get("overfit", {}).get("verdict", "—") == _v]
        if filt_source != "Tümü":
            _filtered = [r for r in _filtered if r.get("source") == filt_source]
        if filt_bt:
            _filtered = [r for r in _filtered if r.get("backtests")]

        st.caption(f"{len(_filtered)} / {len(_catalog)} kayıt gösteriliyor")

        # --- Tablo ---
        rows_cat = []
        for i, r in enumerate(_filtered):
            bt_best = ""
            if r.get("backtests"):
                best_bt = max(r["backtests"].values(),
                              key=lambda b: b.get("ir") or 0)
                bt_best = (f"IR={best_bt['ir']:.2f} | "
                           f"Yıllık={best_bt['annual']:.1f}% | "
                           f"MDD={best_bt['mdd']:.1f}%")
            ov  = r.get("overfit", {})
            wf  = r.get("wf", {}) or {}
            wf_info = ""
            if wf.get("fitness") is not None:
                wf_info = (f"fit={wf['fitness']:.4f} · "
                           f"{wf.get('pos_folds', '?')}/{wf.get('n_folds', '?')} pos")
            rows_cat.append({
                "#":             i,
                "Formül":        r["formula"][:60] + ("…" if len(r["formula"]) > 60 else ""),
                "RankIC":        r["rank_ic"],
                "WF":            wf_info,
                "Test RankIC":   ov.get("test_ric", "—"),
                "Overfit":       ov.get("verdict", "—"),
                "Best Backtest": bt_best,
                "Kaynak":        r.get("source", "?"),
            })
        st.dataframe(pd.DataFrame(rows_cat), use_container_width=True, height=300)

        # --- Silme işlemi ---
        st.divider()
        del_col1, del_col2 = st.columns([3, 1])
        del_formulas = del_col1.multiselect(
            "Silinecek formülleri seç:",
            options=[r["formula"] for r in _filtered],
            format_func=lambda f: f[:80] + ("…" if len(f) > 80 else ""),
        )
        if del_col2.button("🗑️ Seçilenleri sil", disabled=not del_formulas):
            from engine.alpha_catalog import _load_raw, _save_raw
            all_records = _load_raw()
            all_records = [r for r in all_records if r["formula"] not in del_formulas]
            _save_raw(all_records)
            # session_state'ten de kaldır
            keep = st.session_state.alphas[
                ~st.session_state.alphas["Formül"].isin(del_formulas)
            ]
            st.session_state.alphas = keep
            for f in del_formulas:
                st.session_state.trees.pop(f, None)
            st.success(f"🗑️ {len(del_formulas)} formül silindi.")
            st.rerun()

        # --- Tümünü temizle ---
        st.divider()
        c_clear1, c_clear2 = st.columns([3, 1])
        c_clear1.caption("⚠️ Katalogdaki TÜM kayıtları sil (sadece JSON — buffer ve ağırlıklar korunur)")
        if c_clear2.button("🧹 Kataloğu temizle"):
            from engine.alpha_catalog import _save_raw
            _save_raw([])
            st.session_state.alphas = pd.DataFrame()
            st.session_state.trees  = {}
            st.success("Katalog temizlendi.")
            st.rerun()

        # --- CSV indirme ---
        st.divider()
        csv_data = pd.DataFrame([{
            "formula":         r["formula"],
            "rank_ic":         r["rank_ic"],
            "ic":              r["ic"],
            "source":          r.get("source", ""),
            "split_date":      r.get("split_date", ""),
            "overfit_verdict": r.get("overfit", {}).get("verdict", ""),
            "test_ric":        r.get("overfit", {}).get("test_ric", ""),
            "degradation":     r.get("overfit", {}).get("degradation_pct", ""),
            "best_ir":         max((b.get("ir", 0) or 0
                                    for b in r.get("backtests", {}).values()), default=""),
            "best_annual":     max((b.get("annual", 0) or 0
                                    for b in r.get("backtests", {}).values()), default=""),
        } for r in _catalog]).to_csv(index=False)
        st.download_button("⬇️ Katalog CSV indir", csv_data,
                           "alpha_catalog.csv", "text/csv")


# ------------------------------------------------------------------
# Results & backtest
# ------------------------------------------------------------------
if not st.session_state.alphas.empty:
    st.write("### 💎 En İyi Evrimleşmiş Alphalar")
    st.dataframe(st.session_state.alphas.head(15), use_container_width=True)
    sel = st.selectbox("Simülasyon için seç:", st.session_state.alphas["Formül"])

    bt_mode = st.radio(
        "Backtest penceresi:",
        ["🔴 TEST (out-of-sample)", "🟢 TRAIN (in-sample)", "⚪ TAM veri"],
        horizontal=True, index=0,
        help="Test önerilir — out-of-sample performans gerçek alpha göstergesidir."
    )
    if st.button("🚀 Backtesti Koştur"):
        tree = st.session_state.trees.get(sel)
        if tree is None:
            st.error("AST bulunamadı — döngüyü tekrar çalıştırın.")
            st.stop()
        if bt_mode.startswith("🔴"):
            bt_db, bt_label, bt_color = db_test, "TEST", "#FF4444"
        elif bt_mode.startswith("🟢"):
            bt_db, bt_label, bt_color = db_train, "TRAIN", "#44FF44"
        else:
            bt_db, bt_label, bt_color = db, "TAM", "#00FFCC"

        sig = cfg.evaluate(tree, bt_db).values
        # Benchmark'ı backtest penceresine göre filtrele
        _bm_for_bt = None
        if _benchmark_series is not None:
            _bt_dates = pd.to_datetime(bt_db["Date"].unique())
            _bm_for_bt = _benchmark_series[
                (_benchmark_series.index >= _bt_dates.min()) &
                (_benchmark_series.index <= _bt_dates.max())
            ]
            if len(_bm_for_bt) < 5:
                _bm_for_bt = None

        curve, met = run_pro_backtest(
            bt_db, sig,
            top_k=bt_top_k, n_drop=bt_n_drop,
            buy_fee=bt_buy_fee, sell_fee=bt_sell_fee,
            benchmark=_bm_for_bt,
        )
        st.caption(f"📊 Backtest penceresi: **{bt_label}** — {len(bt_db):,} satır")

        # Temel metrikler
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Net Getiri", f"%{met['Net Getiri (%)']:.1f}")
        c2.metric("IR",         f"{met['IR']:.2f}")
        c3.metric("|MDD|",      f"%{met['MDD']:.1f}")
        c4.metric("Yıllık",     f"%{met['Yıllık']:.1f}")

        # Benchmark karşılaştırması (varsa)
        if "Benchmark Getiri (%)" in met:
            st.caption("📊 **Benchmark Karşılaştırması** (alfa = excess return)")
            cb1, cb2, cb3, cb4 = st.columns(4)
            cb1.metric(
                "Benchmark Getiri", f"%{met['Benchmark Getiri (%)']:.1f}",
                help="BIST100 buy-and-hold getirisi (aynı dönem)"
            )
            cb2.metric(
                "Excess Return (Alfa)", f"%{met['Excess Return (%)']:.1f}",
                delta=f"%{met['Excess Return (%)']:.1f}",
                delta_color="normal",
                help="Strateji − Benchmark. Pozitif = gerçek alfa üretiliyor."
            )
            cb3.metric(
                "Alfa IR", f"{met['Alfa IR']:.2f}",
                help="Excess return serisi üzerinden Information Ratio"
            )
            cb4.metric(
                "Beta", f"{met['Beta']:.2f}",
                help="1.0 = piyasa ile birebir hareket, <0.5 = düşük piyasa duyarlılığı"
            )
            if met["Beta"] > 0.8:
                st.warning(
                    f"⚠️ Beta = {met['Beta']:.2f} — Strateji büyük ölçüde piyasa betası taşıyor. "
                    f"Excess Return = %{met['Excess Return (%)']:.1f} — gerçek alfa bu kadar."
                )
            elif met["Excess Return (%)"] > 0:
                st.success(
                    f"✅ Beta = {met['Beta']:.2f}, Excess Return = %{met['Excess Return (%)']:.1f} "
                    f"— Strateji benchmarkın üzerinde alfa üretiyor."
                )

        # Kataloğa backtest sonucunu kaydet
        _cat_rec = next((r for r in load_catalog() if r["formula"] == sel), None)
        if _cat_rec:
            save_alpha(
                formula=sel, tree=tree,
                ic=_cat_rec["ic"], rank_ic=_cat_rec["rank_ic"],
                adj_ic=_cat_rec["adj_ic"],
                bt_mode=bt_label,
                bt_net_return=met["Net Getiri (%)"],
                bt_ir=met["IR"],
                bt_mdd=met["MDD"],
                bt_annual=met["Yıllık"],
            )

        # Equity eğrisi
        fig_bt = go.Figure()
        fig_bt.add_trace(go.Scatter(
            x=curve["Date"], y=curve["Equity"],
            name=f"Alpha ({bt_label})", line=dict(color=bt_color, width=2)
        ))
        if "BenchmarkEquity" in curve.columns:
            fig_bt.add_trace(go.Scatter(
                x=curve["Date"], y=curve["BenchmarkEquity"],
                name="BIST100 (B&H)", line=dict(color="#AAAAAA", width=1, dash="dot")
            ))
        st.plotly_chart(fig_bt, use_container_width=True)

    # ------------------------------------------------------------------
    # META-LABEL — İkincil güven filtresi
    # ------------------------------------------------------------------
    with st.expander("🧠 Meta-Label — İkincil Güven Filtresi", expanded=False):
        st.caption(
            "**López de Prado AFML §3.6**: Primary formül sinyaline ek olarak "
            "logistic regression meta-model eğitir.  \n"
            "P(TB_Label=1) < threshold olan günlerde pozisyon açılmaz → daha temiz sinyal.  \n"
            "⚠️ TB_Label hesaplanmış olmalı (sol panel → Hedef Değişkeni)."
        )
        if not use_meta_label:
            st.info("Meta-Label sidebar'dan aktif edilmeli.")
        else:
            _sel_formula = sel if "sel" in dir() else (
                st.session_state.alphas["Formül"].iloc[0]
                if not st.session_state.alphas.empty else None
            )
            if _sel_formula is None:
                st.warning("Önce bir formül seç / mining çalıştır.")
            else:
                _meta_train_ratio = st.slider(
                    "Train oranı", 0.5, 0.85, 0.7, 0.05,
                    help="Verinin bu kadarı meta modeli eğitmek için kullanılır."
                )
                if st.button("🧠 Meta Modeli Eğit"):
                    _meta_tree = st.session_state.trees.get(_sel_formula)
                    if _meta_tree is None:
                        st.error("AST bulunamadı.")
                    elif "TB_Label" not in idx.columns:
                        st.error("TB_Label sütunu yok — Hedef Değişkeni → Triple-Barrier seç.")
                    else:
                        with st.spinner("Meta dataset hazırlanıyor…"):
                            try:
                                from engine.meta_label import build_meta_dataset, train_meta_model
                                _meta_sig = cfg.evaluate(_meta_tree, idx)
                                _meta_ds = build_meta_dataset(
                                    _meta_sig, idx,
                                    factors=_factor_cache,
                                    regime=_regime_series if use_regime_breakdown else None,
                                    target_col="TB_Label",
                                )
                                # Train/test split tarihi
                                all_meta_dates = sorted(db["Date"].unique())
                                _cut_idx = int(len(all_meta_dates) * float(_meta_train_ratio))
                                _meta_train_end = pd.Timestamp(all_meta_dates[_cut_idx])
                                _meta_model = train_meta_model(_meta_ds, _meta_train_end)
                                st.session_state["_meta_model"] = _meta_model
                                st.session_state["_meta_formula"] = _sel_formula
                                if _meta_model.fit_failed:
                                    st.error(f"Eğitim başarısız: {_meta_model.fail_reason}")
                                else:
                                    _auc_str = f"{_meta_model.auc:.3f}" if np.isfinite(_meta_model.auc) else "—"
                                    st.success(
                                        f"✅ Meta model eğitildi  ·  AUC = {_auc_str}  ·  "
                                        f"threshold = {meta_threshold}  ·  "
                                        f"Features: {', '.join(_meta_model.feature_cols)}"
                                    )
                            except Exception as _me:
                                st.error(f"Meta-label hatası: {_me}")

                # Kayıtlı meta model varsa filtre uygulayarak backtest koştur
                if ("_meta_model" in st.session_state and
                        st.session_state.get("_meta_formula") == _sel_formula):
                    _mm = st.session_state["_meta_model"]
                    if not _mm.fit_failed and st.button("📊 Meta-Filtreli Backtesti Koştur"):
                        try:
                            from engine.meta_label import (
                                build_meta_dataset, apply_meta_filter
                            )
                            from engine.backtest_engine import run_pro_backtest
                            _meta_tree2 = st.session_state.trees.get(_sel_formula)
                            _raw_sig = cfg.evaluate(_meta_tree2, db_test)
                            # Feature dataset (test seti)
                            _meta_ds_test = build_meta_dataset(
                                _raw_sig,
                                db_test.set_index(["Ticker", "Date"]).sort_index(),
                                factors=None,
                                regime=_regime_series if use_regime_breakdown else None,
                                target_col="TB_Label" if "TB_Label" in db_test.columns else "Next_Ret",
                            )
                            _proba = _mm.predict_proba(_meta_ds_test)
                            _filtered_sig = apply_meta_filter(_raw_sig, _proba, threshold=float(meta_threshold))
                            # Karşılaştırmalı backtest
                            _c_raw, _  = run_pro_backtest(db_test, _raw_sig, top_k=bt_top_k, n_drop=bt_n_drop, buy_fee=bt_buy_fee, sell_fee=bt_sell_fee)
                            _c_meta, _ = run_pro_backtest(db_test, _filtered_sig, top_k=bt_top_k, n_drop=bt_n_drop, buy_fee=bt_buy_fee, sell_fee=bt_sell_fee)
                            if _c_raw is not None and _c_meta is not None:
                                import plotly.graph_objects as go
                                _fig_meta = go.Figure()
                                _fig_meta.add_trace(go.Scatter(x=_c_raw["Date"], y=_c_raw["Equity"], name="Ham Sinyal", line=dict(color="#5588ff")))
                                _fig_meta.add_trace(go.Scatter(x=_c_meta["Date"], y=_c_meta["Equity"], name=f"Meta-Filtrelenmiş (t>{meta_threshold})", line=dict(color="#ff8855")))
                                _fig_meta.update_layout(title="Meta-Label Karşılaştırma", height=350)
                                st.plotly_chart(_fig_meta, use_container_width=True)
                                _kept = float((_filtered_sig.notna().sum()) / max(_raw_sig.notna().sum(), 1))
                                st.caption(f"Korunan pozisyon oranı: {_kept:.1%}  ·  AUC: {_mm.auc:.3f}")
                        except Exception as _me2:
                            st.error(f"Meta backtest hatası: {_me2}")

    # ------------------------------------------------------------------
    # DEFLATED SHARPE RATIO — Havuz geneli çoklu-test düzeltmesi
    # ------------------------------------------------------------------
    with st.expander("🏛️ Deflated Sharpe Ratio (DSR) — Çoklu-Test Düzeltmesi", expanded=False):
        st.caption(
            "**Bailey & López de Prado (2014)**: Mining havuzunda N formül denendiğinde "
            "en iyinin Sharpe Ratio'su kısmen şans eseridir. DSR, havuz büyüklüğünü, "
            "skewness ve kurtosis'i hesaba katarak SR'ı deflate eder.  \n"
            "p_value ≥ 0.95 → formül istatistiksel olarak anlamlı (tek taraflı α=5%)."
        )
        _dsr_n_top = st.slider(
            "Kaç formül için DSR hesaplansın?",
            min_value=5, max_value=50, value=20,
            help="Backtest sonuçlarından en yüksek Sharpe'lı formüller seçilir."
        )
        _dsr_n_trials_override = st.number_input(
            "Toplam havuz büyüklüğü (n_trials) — 0 = otomatik",
            min_value=0, max_value=10000, value=0, step=50,
            help="0 seçilirse session_state'teki mining havuzu boyutu kullanılır."
        )
        if st.button("📐 DSR Hesapla"):
            from engine.deflated_sharpe import compute_sharpe_series, deflated_sharpe_ratio
            _alphas_df = st.session_state.alphas
            if _alphas_df.empty:
                st.warning("Henüz mining çalıştırılmadı — önce bir döngü başlatın.")
            else:
                # n_trials: havuz boyutu
                _dsr_n_all = int(_dsr_n_trials_override) if _dsr_n_trials_override > 0 else len(_alphas_df)
                # Top-N formülü backtest et ve equity eğrilerini hesapla
                _dsr_rows = []
                _dsr_subset = _alphas_df.head(int(_dsr_n_top))
                _dsr_bar = st.progress(0.0)
                for _di, (_dsr_idx, _dsr_row) in enumerate(_dsr_subset.iterrows()):
                    _dsr_bar.progress((_di + 1) / len(_dsr_subset))
                    _formula = _dsr_row.get("Formül", "")
                    try:
                        _dsr_tree = st.session_state.trees.get(_formula)
                        if _dsr_tree is None:
                            from engine.formula_parser import parse_formula
                            _dsr_tree = parse_formula(_formula, cfg)
                        if _dsr_tree is None:
                            raise ValueError("AST yok")
                        from engine.backtest_engine import run_pro_backtest
                        _dsr_sig  = cfg.evaluate(_dsr_tree, db_test)
                        _dsr_curve, _ = run_pro_backtest(
                            db_test, _dsr_sig,
                            top_k=bt_top_k, n_drop=bt_n_drop,
                            buy_fee=bt_buy_fee, sell_fee=bt_sell_fee,
                        )
                        if _dsr_curve is None or len(_dsr_curve) < 10:
                            raise ValueError("Equity boş")
                        _eq_series = _dsr_curve.set_index("Date")["Equity"]
                        sr, skw, kurt = compute_sharpe_series(_eq_series, freq=252)
                        T = len(_eq_series.pct_change().dropna())
                        dsr_z, pv = deflated_sharpe_ratio(sr, T, skw, kurt, _dsr_n_all)
                        _dsr_rows.append({
                            "Formül":    _formula,
                            "SR":        round(sr, 3) if np.isfinite(sr) else np.nan,
                            "Skew":      round(skw, 2),
                            "Kurt":      round(kurt, 2),
                            "T":         T,
                            "DSR_z":     round(dsr_z, 2) if np.isfinite(dsr_z) else np.nan,
                            "p_value":   round(pv, 4) if np.isfinite(pv) else np.nan,
                            "✅ Anlamlı": "✅" if (np.isfinite(pv) and pv >= 0.95) else "—",
                        })
                    except Exception as _dsr_exc:
                        _dsr_rows.append({
                            "Formül": _formula, "SR": np.nan, "Skew": np.nan,
                            "Kurt": np.nan, "T": 0, "DSR_z": np.nan,
                            "p_value": np.nan, "✅ Anlamlı": "⚠️",
                        })
                _dsr_bar.empty()
                if _dsr_rows:
                    _dsr_result = pd.DataFrame(_dsr_rows).sort_values("SR", ascending=False)
                    _n_sig = (_dsr_result["✅ Anlamlı"] == "✅").sum()
                    st.metric(
                        "Anlamlı formül (p ≥ 0.95)",
                        f"{_n_sig} / {len(_dsr_result)}",
                        help=f"n_trials={_dsr_n_all} ile deflate edildi"
                    )
                    st.dataframe(_dsr_result, use_container_width=True)
                    st.caption(
                        f"💡 n_trials={_dsr_n_all}: Havuzda bu kadar formül denendi. "
                        "Daha büyük havuz → daha yüksek gürültü barı → DSR_z düşer."
                    )

    # ------------------------------------------------------------------
    # PBO — Probability of Backtest Overfitting (CSCV)
    # ------------------------------------------------------------------
    with st.expander("🎲 PBO — Probability of Backtest Overfitting (CSCV)", expanded=False):
        st.caption(
            "**Bailey, LdP & Zhu (2014)**: Mining havuzunu M zaman dilimine böler.  \n"
            "Her C(M, M/2) kombinasyonunda IS'te en iyi formülün OOS rank'i ölçülür.  \n"
            "**PBO < 0.5** → düşük overfit riski · **PBO ≥ 0.5** → IS seçim sürecinde overfit var."
        )
        _pbo_col1, _pbo_col2 = st.columns(2)
        _pbo_n_slices = _pbo_col1.slider(
            "Zaman dilimi sayısı (M)", min_value=4, max_value=20, value=8,
            help="C(M, M/2) kombinasyonlar üretilir. M=8 → 70, M=16 → 12870 (max_comb ile kısıtlı)."
        )
        _pbo_max_comb = _pbo_col2.slider(
            "Maks kombinasyon sayısı", min_value=50, max_value=2000, value=500,
            help="Büyük M için rastgele alt-örnekleme yapar (seed=42, deterministik)."
        )
        if st.button("🎲 PBO Hesapla"):
            _alphas_df = st.session_state.alphas
            if _alphas_df.empty:
                st.warning("Henüz mining çalıştırılmadı — önce bir döngü başlatın.")
            else:
                with st.spinner("PnL matrisi hesaplanıyor…"):
                    try:
                        from engine.pbo_cscv import compute_pool_pnl, cscv_pbo, pbo_verdict
                        _pbo_mat, _pbo_names = compute_pool_pnl(
                            _alphas_df, st.session_state.trees, db_test,
                            cfg.evaluate,
                            n_slices=int(_pbo_n_slices),
                            top_k=bt_top_k, n_drop=bt_n_drop,
                            buy_fee=bt_buy_fee, sell_fee=bt_sell_fee,
                        )
                        if _pbo_mat.shape[1] < 2:
                            st.warning("PBO için en az 2 geçerli formül gerekli.")
                        else:
                            _pbo_result = cscv_pbo(
                                _pbo_mat, max_combinations=int(_pbo_max_comb)
                            )
                            # Metrik kartlar
                            _m1, _m2, _m3 = st.columns(3)
                            _m1.metric("PBO", f"{_pbo_result['pbo']:.3f}")
                            _m2.metric(
                                "Kombinasyon sayısı",
                                f"{_pbo_result['n_combinations']:,}"
                            )
                            _m3.metric(
                                "Performans düşüşü",
                                f"{_pbo_result['perf_degradation']:.2f}x"
                                if np.isfinite(_pbo_result.get("perf_degradation", np.nan))
                                else "—"
                            )
                            st.info(_pbo_result["verdict"])
                            # Logit histogramı
                            _logit = _pbo_result["logit_lambda"]
                            if len(_logit) > 0:
                                import plotly.graph_objects as go
                                _fig_pbo = go.Figure()
                                _fig_pbo.add_trace(go.Histogram(
                                    x=_logit, nbinsx=30,
                                    name="logit(ω*)",
                                    marker_color="#5c9fff",
                                    opacity=0.75,
                                ))
                                _fig_pbo.add_vline(x=0, line_dash="dash", line_color="red",
                                                   annotation_text="Eşik (0)", annotation_position="top right")
                                _fig_pbo.update_layout(
                                    title="Logit(ω*) Dağılımı — sağ taraf (>0) overfit bölgesi",
                                    xaxis_title="logit(ω*)", yaxis_title="Frekans",
                                    height=300,
                                )
                                st.plotly_chart(_fig_pbo, use_container_width=True)
                            st.caption(
                                f"IS SR ortalama: {_pbo_result['is_sr']:.4f}  ·  "
                                f"OOS SR ortalama: {_pbo_result['oos_sr']:.4f}  ·  "
                                f"{len(_pbo_names)} formül, {int(_pbo_n_slices)} dilim"
                            )
                    except Exception as _pbo_exc:
                        st.error(f"PBO hesaplaması başarısız: {_pbo_exc}")

    # ------------------------------------------------------------------
    # ROLLING WALK-FORWARD BACKTEST
    # ------------------------------------------------------------------
    with st.expander("🔄 Rolling Walk-Forward Backtest", expanded=False):
        st.caption(
            "Tüm veriyi kaydıran pencerelerle test eder — her pencere bağımsız bir "
            "gerçek hayat simülasyonu. Tek bir dönemde şans eseri parlayan formülleri "
            "rejim değişikliklerine dayanıklı olanlardan ayırır."
        )

        _rwf_mode = st.radio(
            "Rolling Modu:",
            [
                "Mod 1 — Anchored (formül + sinyal sabit, test kaydır)",
                "Mod 2 — Rolling Re-fit (her pencerede yeniden nötralize + size_corr)",
                "Mod 3 — Full Discovery (her pencerede yeniden keşif = Hall of Fame)",
            ],
            index=0,
            help=(
                "**Mod 1**: Hızlı. Sinyal bir kere hesaplanır, sadece test penceresi kaydırılır. "
                "Formülün zaman yönünde kararlılığını ölçer.\n\n"
                "**Mod 2**: Yavaş. Her pencerede factor cache yeniden hesaplanır, sinyal yeniden "
                "nötralize edilir, size_corr yeniden ölçülür. Rejim değişikliğinde formülün "
                "size-faktöre kayıp kaymadığını yakalar — gerçek deploy senaryosu.\n\n"
                "**Mod 3** ⚠️ ÇOK YAVAŞ: Her rolling pencerede sıfırdan mining yapar. "
                "Pencere-başına top-K formül Hall of Fame'e eklenir. "
                "Süre ≈ num_gen × n_windows × wf_fitness_süresi."
            ),
        )
        _is_mod3 = _rwf_mode.startswith("Mod 3")

        _rwf_sel = st.selectbox(
            "Formül seç (Rolling WF):",
            st.session_state.alphas["Formül"],
            key="rwf_formula_sel",
        )
        _rwf_c1, _rwf_c2, _rwf_c3 = st.columns(3)
        _rwf_step     = _rwf_c1.selectbox("Test pencere uzunluğu", [3, 6, 12], index=1,
                            help="Her test penceresi bu kadar aylık olur")
        _rwf_min_train = _rwf_c2.selectbox("Min train (ay)", [12, 18, 24], index=1,
                            help="Bu kadar geçmişi olmayan dönemler atlanır")
        _rwf_window   = _rwf_c3.radio("Veri penceresi", ["🔴 TEST", "⚪ TAM"], horizontal=True,
                            help="TEST: split tarihinden sonrası. TAM: tüm veri.")

        # Mod 3 ek parametreleri
        if _is_mod3:
            st.warning(
                "⚠️ **Mod 3 — Full Discovery**: Her rolling pencerede sıfırdan mining yapar. "
                "Süre dakikalar alabilir. Küçük num_gen (50-100) ile dene."
            )
            _m3_c1, _m3_c2 = st.columns(2)
            _m3_num_gen = _m3_c1.slider(
                "Mod 3 pencere başı nesil sayısı", 50, 400, 100, 50,
                help="Her pencerede üretilecek formül sayısı (Faz1+Faz2)."
            )
            _m3_k_keep = _m3_c2.slider(
                "Pencere başı top-K formül", 1, 10, 3,
                help="Her pencereden en iyi K formülü HoF'a ekle."
            )

        if st.button("🔄 Rolling WF Koştur", key="btn_rolling_wf"):
            _rwf_db = db_test if _rwf_window.startswith("🔴") else db

            _rwf_bm = None
            if _benchmark_series is not None:
                _rwf_dates_all = pd.to_datetime(_rwf_db["Date"].unique())
                _rwf_bm = _benchmark_series[
                    (_benchmark_series.index >= _rwf_dates_all.min()) &
                    (_benchmark_series.index <= _rwf_dates_all.max())
                ]
                if len(_rwf_bm) < 5:
                    _rwf_bm = None

            if _is_mod3:
                # ── MOD 3: Full Rolling Discovery ──────────────────────
                from engine.mining_runner import MiningConfig, run_mining_window
                from engine.ensemble import HallOfFame, WindowResult, run_ensemble_backtest

                # Rejim serisi (opsiyonel)
                _m3_regime = None
                if use_regime_breakdown and _benchmark_series is not None:
                    try:
                        from engine.regime import compute_regime
                        _m3_regime = compute_regime(_benchmark_series)
                    except Exception:
                        pass

                _m3_mining_cfg = MiningConfig(
                    num_gen=_m3_num_gen,
                    max_K=int(max_K),
                    use_wf_fitness=bool(use_wf_fitness),
                    wf_n_folds=int(wf_n_folds),
                    wf_embargo=int(wf_embargo),
                    wf_purge=int(wf_purge),
                    lambda_std=float(wf_lambda_std),
                    lambda_cx=float(wf_lambda_cx),
                    lambda_size=float(wf_lambda_size),
                    size_corr_hard_limit=float(size_corr_limit),
                    neutralize=bool(use_neutralize),
                    target_col="Next_Ret",
                    seed=42,
                )

                # Rolling pencere sınırlarını oluştur
                _m3_all_dates = pd.to_datetime(sorted(_rwf_db["Date"].unique()))
                _m3_global_start = _m3_all_dates[0]
                _m3_step_off = pd.DateOffset(months=int(_rwf_step))
                _m3_cursor = _m3_global_start + pd.DateOffset(months=int(_rwf_min_train))
                _m3_window_ends = []
                while _m3_cursor + _m3_step_off <= _m3_all_dates[-1] + pd.Timedelta(days=1):
                    _m3_window_ends.append(pd.Timestamp(_m3_cursor))
                    _m3_cursor = _m3_cursor + _m3_step_off

                if not _m3_window_ends:
                    st.warning("Yeterli veri yok — min train + step aralığını daralt.")
                else:
                    _m3_hof = HallOfFame()
                    _m3_n_wins = len(_m3_window_ends)
                    _m3_prog = st.progress(0, text="Mod 3: başlıyor…")
                    _m3_seed_trees = None

                    for _wi, _train_end_ts in enumerate(_m3_window_ends):
                        _test_end_ts = _train_end_ts + _m3_step_off
                        _m3_prog.progress(
                            _wi / _m3_n_wins,
                            text=(f"Mod 3 Pencere {_wi+1}/{_m3_n_wins} — "
                                  f"train→{str(_train_end_ts)[:10]}")
                        )

                        # Train verisi
                        _m3_train_db = _rwf_db[
                            pd.to_datetime(_rwf_db["Date"]) < _train_end_ts
                        ].copy()
                        if len(_m3_train_db) < 100:
                            continue

                        # Mining döngüsü
                        try:
                            _m3_results = run_mining_window(
                                _m3_train_db, cfg, _m3_mining_cfg,
                                seed_trees=_m3_seed_trees,
                                regime=_m3_regime,
                            )
                        except Exception as _m3_exc:
                            st.warning(f"Pencere {_wi+1} mining hatası: {_m3_exc}")
                            _m3_results = []

                        # Sonraki pencere için warm-start tohumları
                        if _m3_results:
                            _m3_seed_trees = [
                                r.tree for r in _m3_results[:_m3_k_keep * 2]
                            ]

                        # Test dönemi backtest
                        _m3_test_db = _rwf_db[
                            (pd.to_datetime(_rwf_db["Date"]) >= _train_end_ts) &
                            (pd.to_datetime(_rwf_db["Date"]) < _test_end_ts)
                        ].copy()
                        _m3_top_trees = (
                            [r.tree for r in _m3_results[:_m3_k_keep]]
                            if _m3_results else []
                        )
                        if _m3_top_trees and len(_m3_test_db) >= 20:
                            _m3_eq, _ = run_ensemble_backtest(
                                _m3_test_db, _m3_top_trees, cfg.evaluate,
                                top_k=bt_top_k, n_drop=bt_n_drop,
                                buy_fee=bt_buy_fee, sell_fee=bt_sell_fee,
                                benchmark=_rwf_bm,
                            )
                        else:
                            _m3_eq = None

                        _m3_hof.add(WindowResult(
                            window_id=_wi + 1,
                            train_start=pd.Timestamp(_m3_global_start),
                            train_end=pd.Timestamp(_train_end_ts),
                            test_start=pd.Timestamp(_train_end_ts),
                            test_end=pd.Timestamp(_test_end_ts),
                            n_formulas=len(_m3_results),
                            top_formula=(
                                _m3_results[0].formula if _m3_results else "(boş)"
                            ),
                            top_fitness=(
                                _m3_results[0].fitness if _m3_results
                                else float("-inf")
                            ),
                            equity=_m3_eq,
                            formula_names=[
                                r.formula[:30] for r in _m3_results[:_m3_k_keep]
                            ],
                        ))

                    _m3_prog.progress(1.0, text="✅ Mod 3 tamamlandı.")

                    # Sonuçları göster
                    _m3_hof_df = _m3_hof.to_dataframe()
                    if _m3_hof_df.empty:
                        st.warning("Hiçbir pencerede kabul edilen formül yok.")
                    else:
                        _m3_n_total = len(_m3_hof.windows)
                        _m3_n_active = sum(
                            1 for w in _m3_hof.windows if w.n_formulas > 0
                        )
                        _m3_avg_f = float(
                            np.mean([w.n_formulas for w in _m3_hof.windows])
                        )
                        _mh1, _mh2, _mh3 = st.columns(3)
                        _mh1.metric("Toplam Pencere", str(_m3_n_total))
                        _mh2.metric(
                            "Formüllü Pencere", f"{_m3_n_active}/{_m3_n_total}"
                        )
                        _mh3.metric("Ort. Formül/Pencere", f"{_m3_avg_f:.1f}")

                        st.success(
                            f"✅ Hall of Fame — {_m3_n_total} pencere, "
                            f"{_m3_n_active} aktif, top-{_m3_k_keep} per pencere"
                        )
                        st.dataframe(_m3_hof_df, use_container_width=True)

                        _m3_ceq = _m3_hof.combined_equity()
                        if _m3_ceq is not None and len(_m3_ceq) > 0:
                            _fig_m3 = go.Figure()
                            _fig_m3.add_trace(go.Scatter(
                                x=_m3_ceq["Date"], y=_m3_ceq["Equity"],
                                name="Mod 3 HoF Equity",
                                line=dict(color="#00CC96", width=2),
                                fill="tozeroy",
                                fillcolor="rgba(0,204,150,0.07)",
                            ))
                            for _m3w in _m3_hof.windows:
                                _ts_m3 = pd.Timestamp(_m3w.test_start)
                                _fig_m3.add_vline(
                                    x=_ts_m3.value // 10**6,
                                    line_width=1, line_dash="dot",
                                    line_color="#888888",
                                    annotation_text=f"W{_m3w.window_id}",
                                    annotation_position="top",
                                )
                            _fig_m3.update_layout(
                                title=(
                                    f"Mod 3 Full Discovery — {_m3_n_total} pencere "
                                    f"× top-{_m3_k_keep}"
                                ),
                                height=400, showlegend=True,
                            )
                            st.plotly_chart(_fig_m3, use_container_width=True)

            else:
                # ── MOD 1 / MOD 2 ──────────────────────────────────────
                _rwf_tree = st.session_state.trees.get(_rwf_sel)
                if _rwf_tree is None:
                    st.error("AST bulunamadı — döngüyü tekrar çalıştırın.")
                else:
                    _is_mod2 = _rwf_mode.startswith("Mod 2")

                    with st.spinner(
                        f"Rolling WF hesaplanıyor "
                        f"({'Mod 2' if _is_mod2 else 'Mod 1'})…"
                    ):
                        if _is_mod2:
                            from engine.backtest_engine import rolling_refit_wf_backtest
                            _rwf_idx = _rwf_db.set_index(["Ticker", "Date"]).sort_index()
                            _rwf_windows, _rwf_curve = rolling_refit_wf_backtest(
                                _rwf_db, _rwf_idx, _rwf_tree, cfg.evaluate,
                                step_months=_rwf_step,
                                min_train_months=_rwf_min_train,
                                use_neutralize=use_neutralize,
                                size_corr_hard_limit=size_corr_limit,
                                top_k=bt_top_k, n_drop=bt_n_drop,
                                buy_fee=bt_buy_fee, sell_fee=bt_sell_fee,
                                benchmark=_rwf_bm,
                            )
                        else:
                            from engine.backtest_engine import rolling_wf_backtest
                            _rwf_sig = cfg.evaluate(_rwf_tree, _rwf_db).values
                            _rwf_windows, _rwf_curve = rolling_wf_backtest(
                                _rwf_db, _rwf_sig,
                                step_months=_rwf_step,
                                min_train_months=_rwf_min_train,
                                top_k=bt_top_k, n_drop=bt_n_drop,
                                buy_fee=bt_buy_fee, sell_fee=bt_sell_fee,
                                benchmark=_rwf_bm,
                            )

                    if _rwf_windows.empty:
                        st.warning("Yeterli veri yok — test penceresi aralığını daralt.")
                    else:
                        n_win = len(_rwf_windows)
                        # Mod 2'de bazı pencereler size_factor reddiyle IR=None olabilir
                        _valid = (
                            _rwf_windows[_rwf_windows["IR"].notna()]
                            if "IR" in _rwf_windows.columns
                            else _rwf_windows
                        )
                        n_valid = len(_valid)
                        pos_ir    = int((_valid["IR"] > 0).sum()) if n_valid > 0 else 0
                        mean_ir   = float(_valid["IR"].mean()) if n_valid > 0 else float("nan")
                        worst_mdd = float(_valid["MDD (%)"].max()) if n_valid > 0 else float("nan")

                        # Özet metrikler
                        _rm1, _rm2, _rm3, _rm4 = st.columns(4)
                        _rm1.metric("Pencere Sayısı", f"{n_win}")
                        if _is_mod2 and n_valid < n_win:
                            _rm2.metric(
                                "Geçerli Pencere", f"{n_valid}/{n_win}",
                                delta=f"{n_win-n_valid} reddedildi",
                                delta_color="inverse",
                            )
                        else:
                            _rm2.metric(
                                "Pozitif IR", f"{pos_ir}/{n_valid or n_win}",
                                delta=(
                                    "kararlı"
                                    if (n_valid and pos_ir / n_valid >= 0.8)
                                    else "dikkat"
                                ),
                            )
                        _rm3.metric(
                            "Ort. IR",
                            f"{mean_ir:.2f}" if not np.isnan(mean_ir) else "—",
                        )
                        _rm4.metric(
                            "En Kötü MDD",
                            f"%{worst_mdd:.1f}" if not np.isnan(worst_mdd) else "—",
                        )

                        # Karar
                        if n_valid > 0:
                            sign_stability = pos_ir / n_valid
                            if sign_stability >= 0.8 and mean_ir > 0.3:
                                st.success(
                                    f"✅ WF-Kararlı — {pos_ir}/{n_valid} pencerede "
                                    f"pozitif IR, ort. IR={mean_ir:.2f}"
                                )
                            elif sign_stability >= 0.6 and mean_ir > 0:
                                st.warning(
                                    f"⚠️ WF-Kararsız — {pos_ir}/{n_valid} pencerede "
                                    f"pozitif, bazı rejimlerde zayıf"
                                )
                            else:
                                st.error(
                                    f"❌ WF-Geçersiz — yalnızca {pos_ir}/{n_valid} "
                                    f"pencerede pozitif IR"
                                )
                        else:
                            st.error(
                                "❌ Hiç geçerli pencere yok — Mod 2'de tüm "
                                "pencerelerde size_corr çok yüksek"
                            )

                        # Pencere tablosu
                        st.dataframe(_rwf_windows, use_container_width=True)

                        # Ardışık equity eğrisi
                        if not _rwf_curve.empty:
                            _fig_rwf = go.Figure()
                            _fig_rwf.add_trace(go.Scatter(
                                x=_rwf_curve["Date"], y=_rwf_curve["Equity"],
                                name="Rolling WF Equity",
                                line=dict(color="#FF9900", width=2),
                                fill="tozeroy",
                                fillcolor="rgba(255,153,0,0.07)",
                            ))
                            # Pencere sınırlarını dikey çizgiyle işaretle
                            for _, _wr in _rwf_windows.iterrows():
                                _ts_vl = pd.Timestamp(_wr["Test Başlangıç"])
                                _fig_rwf.add_vline(
                                    x=_ts_vl.value // 10**6,
                                    line_width=1, line_dash="dot",
                                    line_color="#666666",
                                    annotation_text=_ts_vl.strftime("%Y-%m"),
                                    annotation_position="top",
                                )
                            _mode_lbl = "Mod 2 (refit)" if _is_mod2 else "Mod 1 (anchored)"
                            _fig_rwf.update_layout(
                                title=(
                                    f"Rolling {_rwf_step}A WF Equity — "
                                    f"{n_win} pencere • {_mode_lbl}"
                                ),
                                height=380, showlegend=True,
                            )
                            st.plotly_chart(_fig_rwf, use_container_width=True)

    # ------------------------------------------------------------------
    # ENSEMBLE BACKTEST — çoklu alpha birleşimi
    # ------------------------------------------------------------------
    with st.expander("🎼 Ensemble Backtest — çoklu alpha birleşimi", expanded=False):
        st.caption(
            "Her satıra bir formül yaz, sinyaller cross-sectional rank ile normalize edilip "
            "ağırlıklı ortalaması alınır ve tek strateji olarak backtest edilir. "
            "Korelasyonsuz sinyaller birleşince IR **diversifikasyon kazancı** sağlar."
        )

        ens_formula_text = st.text_area(
            "Ensemble formülleri (her satıra bir tane):",
            height=160,
            key="ens_formulas",
            placeholder=(
                "Corr(Sub(-0.01, Popen), Mul(Vlot, 0.01), 20)\n"
                "CSRank(Sub(-0.01, Skew(Pclose, 20)))\n"
                "Sub(-0.1, Greater(Delta(Abs(Vlot), 20), Mul(Sign(Plow), -0.1)))"
            ),
        )

        ens_c1, ens_c2, ens_c3 = st.columns(3)
        weight_mode = ens_c1.selectbox(
            "Ağırlıklandırma:",
            ["Eşit ağırlık", "IR-ağırlıklı (TEST)", "Manuel"],
        )
        ens_bt_mode = ens_c2.radio(
            "Backtest penceresi:",
            ["🔴 TEST", "🟢 TRAIN", "⚪ TAM"],
            horizontal=True, index=0,
        )
        ens_max_corr = ens_c3.slider(
            "Çeşitlilik eşiği (max korelasyon)",
            min_value=0.3, max_value=1.0, value=0.7, step=0.05,
            help=(
                "Formüller arasındaki Spearman korelasyonu bu eşiği aşarsa "
                "yüksek korelasyonlu olan ensemble'a alınmaz. "
                "Greedy seçim: ilk formül kesinlikle alınır, sonrakiler "
                "seçilmişlerle max korelasyonu bu değerin altındaysa eklenir. "
                "0.7 önerilir — redundant sinyalleri temizler, çeşitliliği artırır."
            ),
        )

        # Parse formülleri önizle
        _ens_lines = [l.strip() for l in ens_formula_text.splitlines() if l.strip()]

        # Manuel ağırlık UI — parse sonrası formül sayısına göre
        manual_weights = {}
        if weight_mode == "Manuel" and _ens_lines:
            st.caption("Her formül için ağırlık (otomatik normalize edilir):")
            mw_cols = st.columns(min(5, len(_ens_lines)))
            for i, f in enumerate(_ens_lines):
                col = mw_cols[i % len(mw_cols)]
                short = f[:30] + "…" if len(f) > 30 else f
                manual_weights[i] = col.number_input(
                    short, min_value=0.0, max_value=10.0, value=1.0,
                    step=0.1, key=f"mw_{i}"
                )

        _ens_ready = len(_ens_lines) >= 2
        if st.button("🎼 Ensemble backtest çalıştır", disabled=not _ens_ready):
            # Window seç
            if ens_bt_mode.startswith("🔴"):
                bt_db, bt_label, bt_color = db_test, "TEST", "#FF4444"
            elif ens_bt_mode.startswith("🟢"):
                bt_db, bt_label, bt_color = db_train, "TRAIN", "#44FF44"
            else:
                bt_db, bt_label, bt_color = db, "TAM", "#00FFCC"

            bt_idx = bt_db.set_index(["Ticker", "Date"]).sort_index() \
                if not isinstance(bt_db.index, pd.MultiIndex) else bt_db

            signals_df = pd.DataFrame(index=bt_idx.index)
            valid_formulas = []   # (display_name, canonical_formula)
            weights = {}
            parse_log = []

            with st.spinner("Formüller parse ediliyor & sinyaller hesaplanıyor..."):
                parsed = parse_many(ens_formula_text, cfg)

                for i, (raw, tree, err) in enumerate(parsed):
                    if tree is None:
                        parse_log.append({"#": i+1, "Girdi": raw[:60], "Durum": f"❌ parse: {err}"})
                        continue
                    canonical = str(tree)
                    # Session state'te veya katalogda AST var mı? Yoksa parse ettikten sonra kullan
                    cached_tree = st.session_state.trees.get(canonical) or get_tree(canonical) or tree
                    try:
                        sig = cfg.evaluate(cached_tree, bt_db)
                        sig_ranked = sig.groupby(level="Date").rank(pct=True)
                        col_key = f"#{i+1}"
                        signals_df[col_key] = sig_ranked.values
                        valid_formulas.append((col_key, canonical))
                        parse_log.append({"#": i+1, "Girdi": raw[:60], "Durum": "✅ OK"})
                    except Exception as e:
                        parse_log.append({"#": i+1, "Girdi": raw[:60], "Durum": f"❌ eval: {e}"})

            # Parse log göster
            st.dataframe(pd.DataFrame(parse_log), use_container_width=True,
                         height=min(150, 35 + 35 * len(parse_log)))

            if len(valid_formulas) < 2:
                st.error("En az 2 geçerli formül gerekli.")
                st.stop()

            # ── 4.4 Diverse ensemble: yüksek korelasyonlu formülleri ele ──
            # Greedy: ilk formülü al; sonraki formüllerden seçilmişlerle
            # max Spearman korelasyonu ens_max_corr altında olanları ekle.
            _diverse_keys     = []
            _diverse_formulas = []
            _dropped_by_corr  = []
            for _dkey, _dcanon in valid_formulas:
                if not _diverse_keys:
                    _diverse_keys.append(_dkey)
                    _diverse_formulas.append((_dkey, _dcanon))
                    continue
                # Seçilmişlerle maksimum |Spearman korelasyon| hesapla
                _sig_new = signals_df[_dkey].dropna()
                _max_c   = 0.0
                for _sel_k in _diverse_keys:
                    _sig_sel = signals_df[_sel_k].dropna()
                    _common  = _sig_new.index.intersection(_sig_sel.index)
                    if len(_common) > 10:
                        _c = abs(float(_sig_new[_common].corr(_sig_sel[_common], method="spearman")))
                        _max_c = max(_max_c, _c)
                if _max_c <= ens_max_corr:
                    _diverse_keys.append(_dkey)
                    _diverse_formulas.append((_dkey, _dcanon))
                else:
                    _dropped_by_corr.append((_dkey, _dcanon, round(_max_c, 2)))

            if _dropped_by_corr:
                _drop_info = ", ".join(
                    f"{k} (max_corr={c})" for k, _, c in _dropped_by_corr
                )
                st.info(
                    f"🔀 **Çeşitlilik filtresi**: {len(_dropped_by_corr)} formül çıkarıldı "
                    f"(max_corr > {ens_max_corr})  \n"
                    f"Çıkarılanlar: {_drop_info}  \n"
                    f"Kalan: {len(_diverse_formulas)} formül"
                )
            valid_formulas = _diverse_formulas

            if len(valid_formulas) < 2:
                st.warning(
                    "⚠️ Çeşitlilik filtresi sonrası yeterli formül kalmadı. "
                    f"Eşiği artırın (şu an {ens_max_corr:.2f}) veya daha farklı formüller ekleyin."
                )
                st.stop()

            # Ağırlık hesapla
            valid_keys = [k for k, _ in valid_formulas]
            if weight_mode == "Eşit ağırlık":
                weights = {k: 1.0 / len(valid_keys) for k in valid_keys}

            elif weight_mode == "IR-ağırlıklı (TEST)":
                _cat_map = {r["formula"]: r for r in load_catalog()}
                ir_vals = {}
                for k, canon in valid_formulas:
                    rec = _cat_map.get(canon, {})
                    bts = rec.get("backtests", {})
                    test_bt = bts.get("TEST") or bts.get("TAM") or bts.get("TRAIN")
                    ir_vals[k] = abs(test_bt.get("ir", 0)) if test_bt else 1.0
                total = sum(ir_vals.values()) or 1.0
                weights = {k: v / total for k, v in ir_vals.items()}

            else:  # Manuel — i indexine göre
                raw_w = {f"#{i+1}": manual_weights.get(i, 1.0) for i in range(len(_ens_lines))}
                total = sum(raw_w.get(k, 0) for k in valid_keys) or 1.0
                weights = {k: raw_w.get(k, 0) / total for k in valid_keys}

            # Ağırlıklı ortalama sinyal
            combined = pd.Series(0.0, index=signals_df.index)
            for k, w in weights.items():
                combined += w * signals_df[k].fillna(0.5)

            # Benchmark'ı ensemble penceresine göre filtrele
            _bm_ens = None
            if _benchmark_series is not None:
                _ens_dates = pd.to_datetime(bt_db["Date"].unique())
                _bm_ens = _benchmark_series[
                    (_benchmark_series.index >= _ens_dates.min()) &
                    (_benchmark_series.index <= _ens_dates.max())
                ]
                if len(_bm_ens) < 5:
                    _bm_ens = None

            # Backtest
            curve, met = run_pro_backtest(
                bt_db, combined.values,
                top_k=bt_top_k, n_drop=bt_n_drop,
                buy_fee=bt_buy_fee, sell_fee=bt_sell_fee,
                benchmark=_bm_ens,
            )

            # --- Sonuçlar ---
            st.caption(f"📊 Pencere: **{bt_label}** · {len(bt_db):,} satır · "
                       f"{len(valid_formulas)} formül")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Net Getiri", f"%{met['Net Getiri (%)']:.1f}")
            c2.metric("IR",         f"{met['IR']:.2f}")
            c3.metric("|MDD|",      f"%{met['MDD']:.1f}")
            c4.metric("Yıllık",     f"%{met['Yıllık']:.1f}")

            calmar = met["Yıllık"] / max(met["MDD"], 0.1)
            st.caption(
                f"📐 **Calmar = Yıllık/MDD = {calmar:.2f}** "
                f"(>2 iyi, >3 mükemmel, >4 dünya çapında)"
            )

            # Benchmark (varsa)
            if "Benchmark Getiri (%)" in met:
                st.caption("📊 **Benchmark Karşılaştırması**")
                cb1, cb2, cb3, cb4 = st.columns(4)
                cb1.metric("Benchmark Getiri", f"%{met['Benchmark Getiri (%)']:.1f}")
                cb2.metric("Excess Return", f"%{met['Excess Return (%)']:.1f}",
                           delta=f"%{met['Excess Return (%)']:.1f}", delta_color="normal")
                cb3.metric("Alfa IR", f"{met['Alfa IR']:.2f}")
                cb4.metric("Beta", f"{met['Beta']:.2f}")

            # Ağırlık + formül tablosu
            wt_rows = [{
                "#":       k,
                "Formül":  canon[:65] + ("…" if len(canon) > 65 else ""),
                "Ağırlık": f"{weights.get(k, 0) * 100:.1f}%",
            } for k, canon in valid_formulas]
            st.dataframe(pd.DataFrame(wt_rows), use_container_width=True,
                         height=min(300, 35 + 35 * len(wt_rows)))

            # Equity eğrisi
            fig_ens = go.Figure()
            fig_ens.add_trace(go.Scatter(
                x=curve["Date"], y=curve["Equity"],
                name=f"Ensemble ({bt_label}, {len(valid_formulas)} alpha)",
                line=dict(color=bt_color, width=2)
            ))
            if "BenchmarkEquity" in curve.columns:
                fig_ens.add_trace(go.Scatter(
                    x=curve["Date"], y=curve["BenchmarkEquity"],
                    name="BIST100 (B&H)", line=dict(color="#AAAAAA", width=1, dash="dot")
                ))
            st.plotly_chart(fig_ens, use_container_width=True)

            # Korelasyon matrisi
            if len(valid_keys) >= 2:
                st.caption("🔗 Sinyal korelasyonları (düşük = iyi diversifikasyon)")
                corr = signals_df[valid_keys].corr().round(2)
                st.dataframe(corr, use_container_width=True)
                legend_lines = " · ".join(
                    f"**{k}**: {canon[:55]}{'…' if len(canon) > 55 else ''}"
                    for k, canon in valid_formulas
                )
                st.caption("Legend: " + legend_lines)

                n = len(valid_keys)
                avg_corr = (corr.sum().sum() - n) / max(n * (n - 1), 1)
                st.caption(f"Ortalama korelasyon: **{avg_corr:.2f}** "
                           f"({'düşük → iyi diversifikasyon' if avg_corr < 0.3 else 'yüksek → redundant sinyaller'})")

    # ------------------------------------------------------------------
    # Zaman-bazlı validasyon (train/test split)
    # ------------------------------------------------------------------
    with st.expander("🔬 Overfit Testi — Zaman-bazlı validasyon", expanded=False):
        st.caption(
            f"Sidebar'daki split tarihini kullanır: **{split_date}**. "
            f"Top formüller hem train'de hem test'te değerlendirilir, degradation hesaplanır."
        )

        val_mode = st.radio(
            "Validasyon modu:",
            ["📍 Tek Split (hızlı)", "🔄 Walk-Forward (güvenilir)"],
            horizontal=True,
            help=(
                "Tek Split: sidebar'daki %70/%30 kesimiyle tek test penceresi.\n"
                "Walk-Forward: genişleyen 5 pencere — her dönemde test RankIC hesaplanır, "
                "tutarlılık ölçülür. Multiple testing bias'ına karşı çok daha sağlam."
            ),
        )

        wf_col1, wf_col2 = st.columns(2)
        n_top = wf_col1.number_input("Kaç tane top formül test edilsin?",
                                     min_value=1, max_value=50, value=10)
        n_folds = wf_col2.number_input(
            "Fold sayısı (Walk-Forward):", min_value=3, max_value=10, value=5,
            disabled=(val_mode != "🔄 Walk-Forward (güvenilir)"),
        )

        if st.button("🔬 Validasyon çalıştır"):

            # ── Ortak yardımcı ────────────────────────────────────────────
            def _prep_idx(df_slice):
                d = df_slice.sort_values(["Ticker", "Date"]).copy()
                d["Pclose_t1"] = d.groupby("Ticker")["Pclose"].shift(-1)
                d["Pclose_t2"] = d.groupby("Ticker")["Pclose"].shift(-2)
                d["Next_Ret"]  = d["Pclose_t2"] / d["Pclose_t1"] - 1
                return d.set_index(["Ticker", "Date"]).sort_index()

            def _compute_ric(tree_ast, idx):
                try:
                    sig = cfg.evaluate(tree_ast, idx)
                    tmp = pd.DataFrame({
                        "Date":     idx.index.get_level_values("Date"),
                        "Signal":   sig.values,
                        "Next_Ret": idx["Next_Ret"].values,
                    }).dropna()
                    if len(tmp) == 0:
                        return np.nan
                    def _ic(g, method):
                        if g["Signal"].std() == 0:
                            return 0.0
                        return g["Signal"].corr(g["Next_Ret"], method=method)
                    return float(tmp.groupby("Date").apply(lambda g: _ic(g, "spearman")).mean())
                except Exception:
                    return np.nan

            def _verdict(tr_ric, te_ric):
                if np.isnan(tr_ric) or np.isnan(te_ric) or abs(tr_ric) < 1e-6:
                    return "—", np.nan
                same_sign = (tr_ric * te_ric) > 0
                ratio = abs(te_ric) / abs(tr_ric)
                if not same_sign:
                    return "💀 İşaret döndü", (1 - ratio) * 100
                elif ratio >= 0.8:
                    return "✅ Stabil",        (1 - ratio) * 100
                elif ratio >= 0.5:
                    return "⚠️ Biraz bozuldu", (1 - ratio) * 100
                else:
                    return "❌ Overfit",       (1 - ratio) * 100

            def _summary_block(rows):
                n_stable = sum(1 for r in rows if r.get("Karar") == "✅ Stabil")
                n_mild   = sum(1 for r in rows if r.get("Karar") == "⚠️ Biraz bozuldu")
                n_over   = sum(1 for r in rows if r.get("Karar") == "❌ Overfit")
                n_flip   = sum(1 for r in rows if r.get("Karar") == "💀 İşaret döndü")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("✅ Stabil",        n_stable)
                c2.metric("⚠️ Biraz bozuldu", n_mild)
                c3.metric("❌ Overfit",        n_over)
                c4.metric("💀 Döndü",          n_flip)
                total = len(rows)
                if total > 0:
                    stability = (n_stable + 0.5 * n_mild) / total
                    if stability >= 0.6:
                        st.success(
                            f"🎯 Sistem sağlıklı — {n_stable}/{total} formül stabil. "
                            f"Bu gerçek alpha sinyali."
                        )
                    elif stability >= 0.3:
                        st.warning(
                            f"⚠️ Karışık — {n_stable}/{total} stabil, "
                            f"{n_over + n_flip}/{total} overfit."
                        )
                    else:
                        st.error(
                            f"🚨 Ciddi overfit — {total} formülden sadece {n_stable} stabil. "
                            f"Sonuçların çoğu şans eseri."
                        )

            top = st.session_state.alphas.head(int(n_top))

            # ══════════════════════════════════════════════════════════════
            # MOD 1 — Tek Split
            # ══════════════════════════════════════════════════════════════
            if val_mode == "📍 Tek Split (hızlı)":
                if len(db_train) == 0 or len(db_test) == 0:
                    st.error("Split tarihi uçtaki bir dönem seçmiş — sidebar'dan yeniden seç.")
                    st.stop()

                train_idx = _prep_idx(db_train)
                test_idx  = _prep_idx(db_test)

                rows = []
                bar_v = st.progress(0.0)
                for i, r in enumerate(top.itertuples()):
                    s = r.Formül
                    tree_ast = st.session_state.trees.get(s) or get_tree(s)
                    if tree_ast is None:
                        rows.append({"Formül": s[:80], "Karar": "❌ AST yok"})
                        bar_v.progress((i + 1) / len(top))
                        continue
                    tr_ric = _compute_ric(tree_ast, train_idx)
                    te_ric = _compute_ric(tree_ast, test_idx)
                    verd, degr = _verdict(tr_ric, te_ric)
                    rows.append({
                        "Formül":        s[:80] + ("…" if len(s) > 80 else ""),
                        "Train RankIC":  round(tr_ric, 4) if not np.isnan(tr_ric) else None,
                        "Test RankIC":   round(te_ric, 4) if not np.isnan(te_ric) else None,
                        "Degradation %": round(degr, 1)   if not np.isnan(degr)   else None,
                        "Karar":         verd,
                    })
                    if tree_ast is not None and verd != "—":
                        _cat_rec = next((rc for rc in load_catalog() if rc["formula"] == s), None)
                        if _cat_rec:
                            save_alpha(
                                formula=s, tree=tree_ast,
                                ic=_cat_rec["ic"], rank_ic=_cat_rec["rank_ic"],
                                adj_ic=_cat_rec["adj_ic"],
                                train_ric=tr_ric if not np.isnan(tr_ric) else None,
                                test_ric=te_ric  if not np.isnan(te_ric) else None,
                                degradation_pct=degr if not np.isnan(degr) else None,
                                overfit_verdict=verd,
                            )
                    bar_v.progress((i + 1) / len(top))

                st.dataframe(pd.DataFrame(rows), use_container_width=True)
                _summary_block(rows)

            # ══════════════════════════════════════════════════════════════
            # MOD 2 — Walk-Forward (genişleyen pencere)
            # ══════════════════════════════════════════════════════════════
            else:
                all_dates = sorted(db["Date"].unique())
                n_dates   = len(all_dates)
                k         = int(n_folds)

                # Minimum train = ilk %40, her fold test penceresi = toplam / (k+1)
                # Genişleyen pencere: train [0→split_i], test [split_i→split_i+1]
                # split noktaları: eşit aralıklarla %40 ile %95 arasında k+1 nokta
                start_pct = 0.40
                end_pct   = 0.95
                split_indices = [
                    int(n_dates * (start_pct + (end_pct - start_pct) * j / k))
                    for j in range(1, k + 1)
                ]

                st.caption(
                    f"🔄 **{k} fold** — genişleyen train, kayan test penceresi  \n"
                    + "  \n".join(
                        f"Fold {j+1}: train → **{all_dates[split_indices[j]-1].strftime('%Y-%m-%d') if j > 0 else all_dates[split_indices[0]-1].strftime('%Y-%m-%d')}**"
                        f"  |  test **{all_dates[split_indices[j-1]].strftime('%Y-%m-%d') if j > 0 else all_dates[0].strftime('%Y-%m-%d')}**"
                        f" → **{all_dates[min(split_indices[j], n_dates-1)].strftime('%Y-%m-%d') if j < k else all_dates[-1].strftime('%Y-%m-%d')}**"
                        for j in range(k)
                    )
                )

                # Fold sınırlarını hesapla
                folds = []
                for j in range(k):
                    tr_end  = split_indices[j]          # train [0 : tr_end]
                    te_start = split_indices[j]          # test  [tr_end : te_end]
                    te_end   = split_indices[j + 1] if j + 1 < k else n_dates
                    tr_dates = set(all_dates[:tr_end])
                    te_dates = set(all_dates[te_start:te_end])
                    folds.append((j + 1, tr_dates, te_dates))

                rows    = []
                bar_v   = st.progress(0.0)
                total_steps = len(top) * k

                for fi, (fold_n, tr_dates, te_dates) in enumerate(folds):
                    pass  # pre-heat

                step = 0
                for i, r in enumerate(top.itertuples()):
                    s = r.Formül
                    tree_ast = st.session_state.trees.get(s) or get_tree(s)
                    if tree_ast is None:
                        rows.append({"Formül": s[:70], "Karar": "❌ AST yok",
                                     "Ort Test RIC": None})
                        step += k
                        bar_v.progress(min(step / total_steps, 1.0))
                        continue

                    fold_rics  = []
                    fold_signs = []
                    row = {"Formül": s[:70] + ("…" if len(s) > 70 else "")}

                    for fold_n, tr_dates, te_dates in folds:
                        tr_slice = db[db["Date"].isin(tr_dates)]
                        te_slice = db[db["Date"].isin(te_dates)]
                        tr_idx = _prep_idx(tr_slice)
                        te_idx = _prep_idx(te_slice)

                        tr_ric = _compute_ric(tree_ast, tr_idx)
                        te_ric = _compute_ric(tree_ast, te_idx)

                        row[f"F{fold_n} Train"] = round(tr_ric, 4) if not np.isnan(tr_ric) else None
                        row[f"F{fold_n} Test"]  = round(te_ric, 4) if not np.isnan(te_ric) else None

                        if not np.isnan(te_ric):
                            fold_rics.append(te_ric)
                        if not np.isnan(tr_ric) and not np.isnan(te_ric):
                            fold_signs.append((tr_ric * te_ric) > 0)  # True = aynı işaret

                        step += 1
                        bar_v.progress(min(step / total_steps, 1.0))

                    # Özet istatistik
                    if fold_rics:
                        avg_te = float(np.mean(fold_rics))
                        std_te = float(np.std(fold_rics))
                        pos_folds = sum(1 for v in fold_rics if v > 0)
                        sign_stable = sum(fold_signs) / max(len(fold_signs), 1)

                        row["Ort Test RIC"] = round(avg_te, 4)
                        row["Std Test RIC"] = round(std_te, 4)
                        row["+ Fold / N"]   = f"{pos_folds}/{len(fold_rics)}"

                        # Walk-Forward verdict
                        if sign_stable >= 0.8 and avg_te > 0.005:
                            wf_verd = "✅ WF-Stabil"
                        elif sign_stable >= 0.6 and avg_te > 0:
                            wf_verd = "⚠️ WF-Kararsız"
                        elif avg_te <= 0 or sign_stable < 0.4:
                            wf_verd = "💀 WF-Geçersiz"
                        else:
                            wf_verd = "❌ WF-Overfit"

                        row["Karar"] = wf_verd
                    else:
                        row["Ort Test RIC"] = None
                        row["Karar"]        = "—"

                    rows.append(row)

                df_wf = pd.DataFrame(rows)
                st.dataframe(df_wf, use_container_width=True)

                # Walk-Forward özet grafiği — her formülün fold RIC trendi
                st.caption("📈 **Test RankIC — fold trendi** (düz çizgi = tutarlı sinyal)")
                fold_cols = [c for c in df_wf.columns if "Test" in c and c.startswith("F")]
                if fold_cols and len(df_wf) > 0:
                    chart_data = df_wf[["Formül"] + fold_cols].set_index("Formül").T
                    chart_data.index = [f"Fold {j+1}" for j in range(len(chart_data))]
                    st.line_chart(chart_data)

                # Verdict özeti — WF formatına uyarla
                wf_rows_for_summary = [{"Karar": r.get("Karar", "—")} for r in rows]
                n_stable = sum(1 for r in wf_rows_for_summary if "Stabil"  in r["Karar"])
                n_mild   = sum(1 for r in wf_rows_for_summary if "Kararsız" in r["Karar"])
                n_over   = sum(1 for r in wf_rows_for_summary if "Overfit"  in r["Karar"])
                n_flip   = sum(1 for r in wf_rows_for_summary if "Geçersiz" in r["Karar"])
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("✅ WF-Stabil",    n_stable)
                c2.metric("⚠️ WF-Kararsız",  n_mild)
                c3.metric("❌ WF-Overfit",    n_over)
                c4.metric("💀 WF-Geçersiz",  n_flip)

                total_wf = len(rows)
                if total_wf > 0:
                    stability = (n_stable + 0.5 * n_mild) / total_wf
                    if stability >= 0.5:
                        st.success(
                            f"🎯 Walk-Forward sağlıklı — {n_stable}/{total_wf} formül "
                            f"birden fazla zaman diliminde stabil. Üretim kalitesi alpha."
                        )
                    elif stability >= 0.25:
                        st.warning(
                            f"⚠️ Karma sonuç — {n_stable} stabil, {n_over + n_flip} başarısız. "
                            f"Sadece WF-Stabil formülleri ensemble'a al."
                        )
                    else:
                        st.error(
                            f"🚨 Formüller farklı dönemlerde tutarsız. "
                            f"Multiple testing bias yüksek — mining parametrelerini gözden geçir."
                        )
