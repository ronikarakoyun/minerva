"""
engine/alpha_catalog.py — Bulunan alpha'ları kalıcı olarak saklar.

Her kayıt:
- Formül (string + AST dict)
- IC, RankIC, Adj IC
- Keşif zamanı + mining parametreleri
- Overfit test sonucu
- Backtest metrikleri (hangi modda çalıştırıldı)

Şema Migrasyonu (9.2):
  Eski kayıtlar eksik alan içerebilir (örn: size_corr, fold_rics).
  Her yüklemede mevcut şemaya otomatik taşınır.
  CATALOG_SCHEMA_VERSION artırıldığında _migrate_record güncellenmelidir.
"""
import json
import os
from datetime import datetime
from typing import Optional

from .alpha_cfg import Node
from .replay_buffer import _tree_to_dict, _tree_from_dict

CATALOG_PATH = "data/alpha_catalog.json"

# Şema sürümü — yeni alan eklendiğinde artır + _migrate_record güncelle
CATALOG_SCHEMA_VERSION = 5

# Node dict formatı sürümü — _tree_to_dict / _tree_from_dict değiştiğinde artır.
# Kayıtlardaki "ast_schema" bu değerle eşleşmiyorsa yeniden parse edilmeli.
AST_SCHEMA_VERSION = 1


def _migrate_record(r: dict) -> dict:
    """
    Eski katalog kaydını mevcut şemaya taşı.
    Eksik alanları varsayılan değerlerle doldurur; mevcut verilere dokunmaz.
    """
    schema = r.get("_schema", 1)

    # v1 → v2: adj_ic eklendi
    if schema < 2:
        r.setdefault("adj_ic", abs(r.get("rank_ic", 0.0)))

    # v2 → v3: wf bloğu eklendi
    if schema < 3:
        r.setdefault("wf", None)

    # v3 → v4: size_corr, fold_rics, ic_drop_pct, overfit.ic_drop_pct eklendi
    if schema < 4:
        if r.get("wf") and isinstance(r["wf"], dict):
            r["wf"].setdefault("fold_rics", None)
            r["wf"].setdefault("size_corr", None)
        if r.get("overfit") and isinstance(r["overfit"], dict):
            r["overfit"].setdefault("ic_drop_pct", None)

    # v4 → v5: ast_schema eklendi — Node dict formatı versiyonlama
    if schema < 5:
        r.setdefault("ast_schema", AST_SCHEMA_VERSION)

    r["_schema"] = CATALOG_SCHEMA_VERSION
    return r


def _load_raw() -> list:
    if not os.path.exists(CATALOG_PATH):
        return []
    try:
        with open(CATALOG_PATH, "r", encoding="utf-8") as f:
            records = json.load(f)
        # Eski kayıtları mevcut şemaya taşı (sessizce, veri kaybı yok)
        return [_migrate_record(r) for r in records]
    except Exception:
        return []


def _save_raw(records: list) -> None:
    os.makedirs(os.path.dirname(CATALOG_PATH), exist_ok=True)
    with open(CATALOG_PATH, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)


def save_alpha(
    formula: str,
    tree: Node,
    ic: float,
    rank_ic: float,
    adj_ic: float,
    # Mining parametreleri
    split_date: Optional[str] = None,
    max_k: Optional[int] = None,
    population: Optional[int] = None,
    mcts_iters: Optional[int] = None,
    source: str = "llm",          # "llm" | "evolution"
    # WF-fitness (opsiyonel) — mining sırasında hesaplanır
    wf_mean_ric:  Optional[float] = None,
    wf_std_ric:   Optional[float] = None,
    wf_pos_folds: Optional[int]   = None,
    wf_n_folds:   Optional[int]   = None,
    wf_fitness:   Optional[float] = None,
    wf_fold_rics: Optional[list]  = None,
    complexity:   Optional[int]   = None,
    # Overfit testi (opsiyonel)
    train_ric: Optional[float] = None,
    test_ric: Optional[float] = None,
    degradation_pct: Optional[float] = None,
    overfit_verdict: Optional[str] = None,
    # Backtest (opsiyonel)
    bt_mode: Optional[str] = None,
    bt_net_return: Optional[float] = None,
    bt_ir: Optional[float] = None,
    bt_mdd: Optional[float] = None,
    bt_annual: Optional[float] = None,
) -> dict:
    records = _load_raw()

    # Aynı formül varsa güncelle, yoksa ekle
    existing = next((r for r in records if r["formula"] == formula), None)

    record = existing or {}
    record.update({
        "formula":       formula,
        "ast":           _tree_to_dict(tree),
        "ast_schema":    AST_SCHEMA_VERSION,       # Node dict format sürümü
        "ic":            round(float(ic), 6),
        "rank_ic":       round(float(rank_ic), 6),
        "adj_ic":        round(float(adj_ic), 6),
        "source":        source,
        "discovered_at": record.get("discovered_at", datetime.now().isoformat()),
        "updated_at":    datetime.now().isoformat(),
        "_schema":       CATALOG_SCHEMA_VERSION,   # şema sürüm etiketi
    })

    if split_date   is not None: record["split_date"]   = str(split_date)
    if max_k        is not None: record["max_k"]        = max_k
    if population   is not None: record["population"]   = population
    if mcts_iters   is not None: record["mcts_iters"]   = mcts_iters
    if complexity   is not None: record["complexity"]   = int(complexity)

    # WF-fitness blok
    if wf_fitness is not None or wf_mean_ric is not None:
        record["wf"] = {
            "mean_ric":  round(float(wf_mean_ric), 6)  if wf_mean_ric  is not None else None,
            "std_ric":   round(float(wf_std_ric), 6)   if wf_std_ric   is not None else None,
            "pos_folds": int(wf_pos_folds)             if wf_pos_folds is not None else None,
            "n_folds":   int(wf_n_folds)               if wf_n_folds   is not None else None,
            "fitness":   round(float(wf_fitness), 6)   if wf_fitness   is not None else None,
            "fold_rics": [round(float(x), 6) for x in wf_fold_rics]
                         if wf_fold_rics is not None else None,
        }

    if overfit_verdict is not None:
        record["overfit"] = {
            "train_ric":       round(float(train_ric), 6) if train_ric is not None else None,
            "test_ric":        round(float(test_ric), 6) if test_ric is not None else None,
            "degradation_pct": round(float(degradation_pct), 1) if degradation_pct is not None else None,
            "verdict":         overfit_verdict,
        }

    if bt_mode is not None:
        record.setdefault("backtests", {})
        record["backtests"][bt_mode] = {
            "net_return": round(float(bt_net_return), 2) if bt_net_return is not None else None,
            "ir":         round(float(bt_ir), 4) if bt_ir is not None else None,
            "mdd":        round(float(bt_mdd), 2) if bt_mdd is not None else None,
            "annual":     round(float(bt_annual), 2) if bt_annual is not None else None,
            "at":         datetime.now().isoformat(),
        }

    if existing is None:
        records.append(record)

    _save_raw(records)
    return record


def load_catalog() -> list:
    """Tüm kayıtları yükle; WF-fitness varsa ona göre, yoksa |rank_ic|'e göre sırala."""
    records = _load_raw()

    def _sort_key(r):
        wf = r.get("wf") or {}
        fit = wf.get("fitness")
        if fit is not None:
            # WF-fitness pozitifse yüksek → iyi; yoksa |rank_ic| yedeği
            return (1, fit)
        return (0, abs(r.get("rank_ic", 0)))

    return sorted(records, key=_sort_key, reverse=True)


def get_tree(formula: str) -> Optional[Node]:
    """Katalogdan formülün AST'sini geri yükle."""
    import warnings
    for r in _load_raw():
        if r["formula"] == formula:
            saved_ast_schema = r.get("ast_schema", 0)
            if saved_ast_schema != AST_SCHEMA_VERSION:
                warnings.warn(
                    f"'{formula}' için kaydedilmiş AST formatı "
                    f"(ast_schema={saved_ast_schema}) mevcut sürümle "
                    f"({AST_SCHEMA_VERSION}) uyuşmuyor — yeniden parse gerekebilir.",
                    stacklevel=2,
                )
            try:
                return _tree_from_dict(r["ast"])
            except Exception:
                return None
    return None
