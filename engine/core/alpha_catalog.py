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
import glob
import json
import logging
import os
import shutil
import tempfile
from datetime import datetime
from typing import Optional

_cat_log = logging.getLogger(__name__)

# Rotating snapshot — kaç yedek saklanacak (en eskisi silinir)
CATALOG_SNAPSHOT_DIR = "data/catalog_snapshots"
CATALOG_SNAPSHOT_KEEP = 7  # son 7 günlük yedek


def _take_snapshot() -> None:
    """
    Mevcut alpha_catalog.json'u snapshot dizinine kopyalar (tarihli).

    N46: Günde en fazla bir snapshot — aynı gün zaten snapshot varsa atla.
    """
    if not os.path.exists(CATALOG_PATH):
        return
    try:
        os.makedirs(CATALOG_SNAPSHOT_DIR, exist_ok=True)
        today_prefix = datetime.utcnow().strftime("%Y%m%d")

        # N46: Bugün için zaten snapshot varsa atla
        pattern = os.path.join(CATALOG_SNAPSHOT_DIR, "alpha_catalog_*.json")
        all_snaps = sorted(glob.glob(pattern))
        if all_snaps and os.path.basename(all_snaps[-1]).startswith(
            f"alpha_catalog_{today_prefix}"
        ):
            return  # Bugün zaten alındı

        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        dest = os.path.join(CATALOG_SNAPSHOT_DIR, f"alpha_catalog_{ts}.json")
        shutil.copy2(CATALOG_PATH, dest)
        # Eski yedekleri temizle: en yeni KEEP adet dışındakileri sil
        all_snaps = sorted(glob.glob(pattern))
        for old in all_snaps[:-CATALOG_SNAPSHOT_KEEP]:
            try:
                os.unlink(old)
            except OSError:
                pass
    except Exception as exc:
        _cat_log.warning("Catalog snapshot alınamadı: %s", exc)

from filelock import FileLock

from .alpha_cfg import Node
from ..ml.replay_buffer import _tree_to_dict, _tree_from_dict

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
    """JSON'u atomik olarak yazar: önce geçici dosya, sonra rename.

    FileLock (cross-platform) eş zamanlı yazmaları engeller.
    Rename atomik olduğundan yarım yazma riski yoktur.
    """
    os.makedirs(os.path.dirname(CATALOG_PATH), exist_ok=True)
    lock_path = CATALOG_PATH + ".lock"
    catalog_dir = os.path.dirname(os.path.abspath(CATALOG_PATH))

    with FileLock(lock_path, timeout=15):
        # Overwrite öncesi snapshot al (7-günlük döngüsel yedek)
        _take_snapshot()
        # Geçici dosyaya yaz, ardından atomik rename
        fd, tmp_path = tempfile.mkstemp(dir=catalog_dir, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(records, f, ensure_ascii=False, indent=2)
            os.replace(tmp_path, CATALOG_PATH)
        except Exception:
            # Geçici dosyayı temizle, hatayı yukarı ilet
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise


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


def save_regime_champion(
    regime_id: int,
    formula: str,
    tree: Node,
    ic: float = 0.0,
    rank_ic: float = 0.0,
    adj_ic: float = 0.0,
) -> dict:
    """
    Bir formülü belirtilen rejimin "şampiyonu" olarak işaretle (Faz 5.2).

    Aynı formül zaten kayıtlıysa `regime_champion_for` alanı eklenir; yoksa
    yeni kayıt açılır. Her rejim için aynı anda tek şampiyon olur — eski
    kayıtların `regime_champion_for == regime_id` alanı temizlenir.
    """
    records = _load_raw()

    # Eski şampiyonu bu rejim için temizle
    for r in records:
        if r.get("regime_champion_for") == regime_id and r.get("formula") != formula:
            r.pop("regime_champion_for", None)

    existing = next((r for r in records if r.get("formula") == formula), None)
    if existing is None:
        save_alpha(formula, tree, ic, rank_ic, adj_ic)
        records = _load_raw()
        existing = next((r for r in records if r.get("formula") == formula), None)

    if existing is None:
        return {}

    existing["regime_champion_for"] = int(regime_id)
    existing["updated_at"] = datetime.now().isoformat()
    _save_raw(records)
    return existing


def set_inactive(formula: str) -> bool:
    """
    Decay tetiklendiğinde formülü pasife al.

    `live=False` ve `deactivated_at` timestamp eklenir; `regime_champion_for`
    alanı temizlenir.  `get_active_champions()` live=False kayıtları döndürmez.

    Returns
    -------
    bool — formül katalogda bulunursa True, bulunamazsa False.
    """
    records = _load_raw()
    rec = next((r for r in records if r.get("formula") == formula), None)
    if rec is None:
        return False
    rec["live"] = False
    rec["deactivated_at"] = datetime.now().isoformat()
    rec["updated_at"] = datetime.now().isoformat()
    rec.pop("regime_champion_for", None)
    _save_raw(records)
    return True


def get_active_champions() -> list[tuple[str, dict]]:
    """
    Faz 6: Decay-monitor için aktif şampiyon listesi.

    `regime_champion_for` alanı set olan formülleri döndürür. Mean/std fallback
    sırası:
        1) backtest_mean / backtest_std   (paper trade istatistikleri)
        2) wf_mean_ric / wf_std_ric        (walk-forward fitness)
        3) ic / 0.005                       (basit ic + tipik BIST std default)

    Returns
    -------
    list[tuple[formula_id, meta_dict]]
        meta_dict: {"backtest_mean", "backtest_std", "regime_id", "formula"}
    """
    out: list[tuple[str, dict]] = []
    for r in _load_raw():
        if r.get("regime_champion_for") is None:
            continue
        if r.get("live") is False:  # set_inactive() tarafından pasife alındı
            continue
        formula = r.get("formula")
        if not formula:
            continue
        mean = r.get("backtest_mean")
        std = r.get("backtest_std")
        if mean is None:
            mean = r.get("wf_mean_ric")
        if std is None:
            std = r.get("wf_std_ric")
        # Son fallback: ic varsa onu kullan, std için BIST tipik 0.005
        if mean is None:
            mean = r.get("ic")
        if std is None and mean is not None:
            std = 0.005   # BIST günlük IC tipik dağılımı
        if mean is None or std is None:
            continue
        out.append((formula, {
            "backtest_mean": float(mean),
            "backtest_std": float(std),
            "regime_id": int(r["regime_champion_for"]),
            "formula": formula,
        }))
    return out


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
