"""
config.py — Minerva v3 Studio konfigürasyon sistemi.

Kullanım:
  1. config.yaml varsa otomatik yüklenir (opsiyonel).
  2. Yoksa tüm varsayılanlar geçerli — geriye uyumlu.
  3. MiningConfig dataclass → deney tekrar edilebilirliği için JSON serialize.

config.yaml örneği:
  paths:
    market_db:  data/market_db.parquet
    catalog:    data/alpha_catalog.json
    benchmark:  data/bist100.parquet
  mining:
    n_iter: 500
    use_neutralize: true
    wf_n_folds: 5
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field

# ------------------------------------------------------------------
# Varsayılan dosya yolları
# ------------------------------------------------------------------
_DEFAULTS = {
    "market_db":  "data/market_db.parquet",
    "catalog":    "data/alpha_catalog.json",
    "benchmark":  "data/bist100.parquet",
    "bm_csv":     "data/bist100.csv",
    "config":     "config.yaml",
}


def load_paths() -> dict:
    """
    config.yaml'dan dosya yollarını yükle.
    Dosya yoksa veya hatalıysa varsayılanları döndür.
    """
    paths = dict(_DEFAULTS)
    cfg_file = _DEFAULTS["config"]
    if not os.path.exists(cfg_file):
        return paths
    try:
        import yaml  # pyyaml
        with open(cfg_file, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if data and "paths" in data:
            paths.update({k: v for k, v in data["paths"].items() if v})
    except ImportError:
        pass   # pyyaml kurulu değil — sessizce varsayılanları kullan
    except Exception:
        pass
    return paths


# ------------------------------------------------------------------
# MiningConfig dataclass (10.2)
# ------------------------------------------------------------------
@dataclass
class MiningConfig:
    """
    Tüm mining parametrelerini tek objede tutan konfigürasyon.

    Kullanım örnekleri:
      cfg = MiningConfig()                     # varsayılanlar
      cfg = MiningConfig(n_iter=1000, max_K=20)  # özelleştirilmiş
      cfg.to_json()                            # JSON → deney logu
      MiningConfig.from_json(json_str)         # JSON'dan geri yükle
      MiningConfig.from_yaml()                 # config.yaml'dan yükle
    """
    # Popülasyon
    n_iter: int = 300
    max_K:  int = 15

    # WF-Fitness
    use_wf_fitness:  bool  = True
    wf_n_folds:      int   = 5
    wf_embargo:      int   = 5      # fold sınırlarında label leakage koruması (gün)
    wf_lambda_std:   float = 0.5
    wf_lambda_cx:    float = 0.001

    # Faktör Nötralizasyonu
    use_neutralize:    bool  = True
    wf_lambda_size:    float = 0.5
    size_corr_limit:   float = 0.7

    # Hedef Değişkeni
    target_mode:  str   = "Next_Ret"   # "Next_Ret" | "TB_Label"
    tb_horizon:   int   = 10
    tb_multiplier: float = 1.5
    tb_long_only: bool  = True

    # Backtest
    buy_fee:  float = 0.0005   # %0.05 alış
    sell_fee: float = 0.0015   # %0.15 satış
    top_k:    int   = 50
    n_drop:   int   = 5

    # Tree-LSTM / MCTS
    use_value_fn: bool = True
    use_mcts:     bool = False
    mcts_iters:   int  = 200

    # Performans
    use_parallel: bool = True   # Faz 3 joblib paralel

    def to_json(self) -> str:
        """Konfigürasyonu JSON string'e dönüştür (deney logu için)."""
        return json.dumps(asdict(self), ensure_ascii=False, indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> "MiningConfig":
        """JSON string'den MiningConfig oluştur."""
        data = json.loads(json_str)
        valid = {k for k in data if k in cls.__dataclass_fields__}
        return cls(**{k: data[k] for k in valid})

    @classmethod
    def from_yaml(cls, path: str = "config.yaml") -> "MiningConfig":
        """config.yaml'daki [mining] bölümünden MiningConfig oluştur."""
        if not os.path.exists(path):
            return cls()
        try:
            import yaml
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            mining = data.get("mining", {}) if data else {}
            valid = {k for k in mining if k in cls.__dataclass_fields__}
            return cls(**{k: mining[k] for k in valid})
        except Exception:
            return cls()


# ------------------------------------------------------------------
# Örnek config.yaml üretici
# ------------------------------------------------------------------
def write_example_config(path: str = "config.yaml") -> None:
    """
    Kullanıcı için örnek config.yaml oluştur.
    Dosya zaten varsa üzerine yazmaz.
    """
    if os.path.exists(path):
        return
    content = """\
# Minerva v3 Studio — konfigürasyon dosyası
# Bu dosyayı düzenleyerek varsayılanları değiştirebilirsiniz.

paths:
  market_db:  data/market_db.parquet   # BIST fiyat verisi
  catalog:    data/alpha_catalog.json  # Keşfedilen alpha'lar
  benchmark:  data/bist100.parquet     # BIST100 index (opsiyonel)

mining:
  n_iter:          300      # Popülasyon büyüklüğü
  max_K:           15       # Maksimum AST uzunluğu
  use_wf_fitness:  true     # Walk-Forward fitness
  wf_n_folds:      5        # Fold sayısı
  wf_embargo:      5        # Fold sınırı embargo (gün)
  use_neutralize:  true     # Faktör nötralizasyonu
  tb_long_only:    true     # BIST long-only kısıtı
  buy_fee:         0.0005   # Alış komisyonu (%0.05)
  sell_fee:        0.0015   # Satış komisyonu (%0.15)
  use_parallel:    true     # Paralel Faz 3 (joblib)
"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
