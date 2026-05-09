import os, logging, json, pandas as pd, numpy as np
from pathlib import Path
from datetime import datetime
from unittest.mock import patch

# Minerva Engine
import engine.core.alpha_catalog as catalog
from engine.core.alpha_cfg import AlphaCFG
from engine.data.regime_detector import run_pipeline, RegimeConfig
from engine.strategies.mining_runner import run_mining_window, MiningConfig
from engine.execution.blender import blend_regime_signals, BlenderConfig, load_champions_from_catalog
from engine.execution.paper_trader import PaperTraderConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s — %(message)s")
logger = logging.getLogger("marathon_v3")

# --- PATH SETUP ---
MARATHON_DIR = Path("data/marathon")
MARATHON_DIR.mkdir(parents=True, exist_ok=True)
catalog.CATALOG_PATH = str(MARATHON_DIR / "marathon_catalog.json")
MARATHON_TRADES_PATH = MARATHON_DIR / "marathon_trades.parquet"

def get_local_xu100(*args, **kwargs):
    df = pd.read_parquet("data/market_db.parquet")
    xu100 = df[df["Ticker"].isin(["XU100.IS", "XU100"])].copy()
    xu100 = xu100.rename(columns={"Popen": "Open", "Phigh": "High", "Plow": "Low", "Pclose": "Close", "Vlot": "Volume"})
    xu100["Date"] = pd.to_datetime(xu100["Date"])
    return xu100.set_index("Date").sort_index()

class MarathonV3:
    """
    Bellek Tabanlı ve Vektörize edilmiş dürüst simülasyon.
    Optimasyon 1: İşlemleri RAM'de tut, her ay diske yaz.
    Optimasyon 3: Stop ve Kar/Zarar kontrollerini vektörize et.
    """
    def __init__(self):
        self.cfg_alpha = AlphaCFG()
        self.pt_cfg = PaperTraderConfig(output_path=MARATHON_TRADES_PATH, top_k=13, n_drop=2)
        self.db = pd.read_parquet("data/market_db.parquet")
        self.db['Date'] = pd.to_datetime(self.db['Date'])
        # Tüm tarihleri önceden sırala ve hazırla
        self.all_dates = sorted(self.db['Date'].unique())
        self.all_dates = [d for d in self.all_dates if d >= pd.Timestamp("2018-01-01")]
        self.prob_df = None
        # OPTİMASYON 1: Bellek tamponu
        self.trade_buffer = [] 

    def _initialize_catalog(self):
        if not Path(catalog.CATALOG_PATH).exists():
            with open(catalog.CATALOG_PATH, "w") as f: json.dump([], f)

    def _save_buffer(self):
        """RAM'deki işlemleri diske güvenli bir şekilde yazar."""
        if not self.trade_buffer: return
        df = pd.DataFrame(self.trade_buffer)
        df.to_parquet(MARATHON_TRADES_PATH, index=False)
        logger.info(f"💾 Bellek tamponu diske yazıldı ({len(df)} işlem).")

    def run(self):
        self._initialize_catalog()
        logger.info("🌀 Rejim olasılıkları hesaplanıyor (Causal)...")
        with patch("yfinance.download", side_effect=get_local_xu100):
            self.prob_df = run_pipeline(RegimeConfig(period="max", use_filtered_probs=True))
        
        last_mining_date = None
        save_counter = 0

        for current_date in self.all_dates:
            if current_date not in self.prob_df.index: continue

            # MADENCİLİK (Her 90 günde bir)
            if last_mining_date is None or (current_date - last_mining_date).days >= 90:
                self._save_buffer() # Mining öncesi kaydet
                self._mining_cycle(current_date)
                last_mining_date = current_date

            # İCRA (Her Gün)
            self._execution_cycle(current_date)
            
            # Ayda bir yedek al (Her 21 iş günü yaklaşık)
            save_counter += 1
            if save_counter % 21 == 0:
                self._save_buffer()

        self._save_buffer() # Final kaydı
        logger.info("🏁 Simülasyon başarıyla tamamlandı.")

    def _mining_cycle(self, current_date):
        logger.info(f"--- 🚀 MADENCİLİK GÜNÜ: {current_date.date()} ---")
        db_train = self.db[self.db['Date'] < current_date].copy()
        if db_train['Date'].nunique() < 500: return

        mcfg = MiningConfig(num_gen=500, max_K=15, min_mean_ric=0.002, min_pos_ratio=0.38, use_wf_fitness=True)
        try:
            accepted = run_mining_window(db_train, self.cfg_alpha, mcfg)
            if accepted:
                recs = [r for r in catalog._load_raw() if "regime_champion_for" not in r]
                catalog._save_raw(recs)
                for i in range(min(6, len(accepted))):
                    catalog.save_regime_champion(i, accepted[i].formula, accepted[i].tree, accepted[i].mean_ric)
                logger.info(f"✅ Rejim şampiyonları güncellendi.")
        except Exception as e:
            logger.error(f"Mining Hatası: {e}")

    def _execution_cycle(self, current_date):
        """PaperTrade mantığının vektörize ve bellek-dostu sürümü."""
        champions = load_champions_from_catalog(Path(catalog.CATALOG_PATH), self.cfg_alpha)
        if not champions: return

        # 1. Mevcut Açık Pozisyonları Tespit Et
        open_pos = [r for r in self.trade_buffer if pd.isna(r["exit_px"])]
        added = 0 # HATANIN ÇÖZÜMÜ: added değişkenini burada tanımla
        
        all_dates = sorted(self.db["Date"].unique())
        try:
            next_date = all_dates[all_dates.index(current_date) + 1]
        except (ValueError, IndexError): return

        db_curr = self.db[self.db["Date"] == current_date].set_index("Ticker")
        db_next = self.db[self.db["Date"] == next_date].set_index("Ticker")

        # 2. Sinyalleri Hesapla (Blender)
        weights_df = blend_regime_signals(champions, self.prob_df, self.db, BlenderConfig(use_blending=True, top_k=13), self.cfg_alpha)
        today_weights = weights_df.loc[current_date].dropna().nlargest(13) if current_date in weights_df.index else pd.Series()

        # 3. SLACK EXIT (Sadece 20. sıranın altına düşerse sat)
        closed_count = 0
        # Puanlara göre rank hesapla (büyük puan = küçük rank)
        rank_series = today_weights.sort_values(ascending=False).rank(ascending=False, method="first")
        
        for pos in open_pos:
            ticker = pos["ticker"]
            if ticker not in db_curr.index or ticker not in db_next.index: continue
            
            # Stop veya Slack Exit (Rank > 20) kontrolü
            current_px = db_curr.at[ticker, "Pclose"]
            t_plus_1_open = db_next.at[ticker, "Popen"]
            
            is_stop = current_px < pos["entry_px"] * (1 - self.pt_cfg.trailing_stop_pct)
            
            current_rank = rank_series.get(ticker, 999)
            is_slack_exit = current_rank > 20
            
            if is_stop or is_slack_exit:
                # Taban Kilidi Kontrolü
                if t_plus_1_open > db_curr.at[ticker, "Pclose"] * 0.901:
                    pos["exit_px"] = t_plus_1_open
                    pos["gross_pnl_pct"] = pos["exit_px"] / pos["entry_px"] - 1
                    pos["net_pnl_pct"] = pos["gross_pnl_pct"] - (pos["slippage_bps"] / 1e4)
                    closed_count += 1

        # 4. YENİ ALIMLAR
        open_tickers = {r["ticker"] for r in self.trade_buffer if pd.isna(r["exit_px"])}
        needed = self.pt_cfg.top_k - len(open_tickers)
        
        if needed > 0:
            candidates = today_weights[~today_weights.index.isin(open_tickers)].nlargest(needed * 2)
            added = 0
            for ticker, weight in candidates.items():
                if added >= needed: break
                if ticker not in db_next.index or ticker not in db_curr.index: continue
                
                t_plus_1_open = db_next.at[ticker, "Popen"]
                if t_plus_1_open < db_curr.at[ticker, "Pclose"] * 1.099: # Tavan değilse al
                    self.trade_buffer.append({
                        "date": pd.Timestamp(next_date),
                        "formula_id": "marathon_v3",
                        "ticker": ticker,
                        "weight": float(weight),
                        "signal_value": float(weight),
                        "entry_px": t_plus_1_open,
                        "exit_px": np.nan,
                        "gross_pnl_pct": np.nan,
                        "slippage_bps": 10.0,
                        "net_pnl_pct": np.nan,
                    })
                    added += 1
        
        if closed_count > 0 or added > 0:
            logger.info(f"[{current_date.date()}] Portföy: {len(open_tickers) - closed_count + added} Aktif | -{closed_count} Kapandı | +{added} Açıldı")

if __name__ == "__main__":
    MarathonV3().run()
