"""
tests/test_regression.py — Regresyon testleri (§11.3).

İki katmanlı altın dosya sistemi:
  1. Sentetik veri  → tests/data/golden.json       (her ortamda deterministik)
  2. Gerçek BIST    → tests/data/golden_real.json  (gerçek market davranışı)

Çalışma prensibi:
  - İlk çalıştırma (--update-golden): her iki altın dosya da üretilir
  - Sonraki çalıştırmalar: altın dosyayla karşılaştırır, %5+ sapma → test başarısız
  - Gerçek veri testi: tests/data/bist_snapshot.parquet yoksa otomatik atlanır

Kullanım:
  python -m pytest tests/test_regression.py -v              # karşılaştır
  python -m pytest tests/test_regression.py --update-golden # yenile
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import math
import numpy as np
import pandas as pd
import pytest

from engine.core.alpha_cfg import AlphaCFG
from engine.validation.wf_fitness import compute_wf_fitness, make_date_folds
from engine.core.formula_parser import parse_formula
from tests.conftest import make_synthetic_idx


# ─── Konfigürasyon ────────────────────────────────────────────────────────────

GOLDEN_PATH      = os.path.join(os.path.dirname(__file__), "data", "golden.json")
GOLDEN_REAL_PATH = os.path.join(os.path.dirname(__file__), "data", "golden_real.json")
SNAPSHOT_PATH    = os.path.join(os.path.dirname(__file__), "data", "bist_snapshot.parquet")

# Sabit regresyon formülleri — gerçek alpha sinyalleri değil; davranış testleri
REGRESSION_FORMULAS = [
    "Delta(Pclose, 5)",
    "Std(Pclose, 20)",
    "Mean(Pclose, 20)",
    "Rank(Pclose, 20)",
    "Corr(Pclose, Vlot, 20)",
]

# Deterministik test parametreleri — ASLA değiştirme!
_SEED    = 314159
_TICKERS = 25
_DAYS    = 400
_FOLDS   = 4
_EMBARGO = 3

# Tolerans: fitness/mean_ric ±%5 sapma kabul edilir
_TOL = 0.05


# ─── Yardımcılar ──────────────────────────────────────────────────────────────

def _compute_reference(update: bool = False) -> dict:
    """
    Referans değerleri hesapla (veya yükle).

    update=True → yeniden hesapla ve kaydet.
    update=False → varsa dosyadan yükle, yoksa hesapla.
    """
    if not update and os.path.exists(GOLDEN_PATH):
        with open(GOLDEN_PATH, "r") as f:
            return json.load(f)

    # Hesapla
    cfg   = AlphaCFG()
    idx   = make_synthetic_idx(n_tickers=_TICKERS, n_days=_DAYS, seed=_SEED)
    dates = idx.index.get_level_values("Date").values
    folds = make_date_folds(
        dates, n_folds=_FOLDS, min_fold_days=30, embargo_days=_EMBARGO
    )

    golden = {}
    for formula in REGRESSION_FORMULAS:
        try:
            tree   = parse_formula(formula, cfg)
            result = compute_wf_fitness(
                tree, cfg.evaluate, idx, folds,
                lambda_std=0.5, lambda_cx=0.001,
                min_valid_folds=2,
                target_col="Next_Ret",
                neutralize=False,
                size_corr_hard_limit=0.95,
            )
            golden[formula] = {
                "status":    result["status"],
                "mean_ric":  result["mean_ric"] if not math.isnan(result.get("mean_ric", float("nan"))) else None,
                "fitness":   result["fitness"]  if not math.isnan(result.get("fitness",  float("nan"))) else None,
                "complexity": result["complexity"],
                "pos_folds": result["pos_folds"],
                "n_folds":   len(result["fold_rics"]),
            }
        except Exception as e:
            golden[formula] = {"status": f"exception:{e}"}

    # Kaydet
    if update:
        os.makedirs(os.path.dirname(GOLDEN_PATH), exist_ok=True)
        with open(GOLDEN_PATH, "w") as f:
            json.dump(golden, f, indent=2, ensure_ascii=False)
        print(f"\n✅ Altın dosya güncellendi: {GOLDEN_PATH}")

    return golden


def _relative_diff(a, b) -> float:
    """Göreceli fark: |a - b| / max(|a|, |b|, 1e-9)."""
    if a is None or b is None:
        return 0.0   # None → karşılaştırma atla
    if math.isnan(a) or math.isnan(b):
        return 0.0
    denom = max(abs(a), abs(b), 1e-9)
    return abs(a - b) / denom


# ─── Gerçek veri yardımcıları ─────────────────────────────────────────────────

def _load_bist_snapshot() -> pd.DataFrame:
    """
    tests/data/bist_snapshot.parquet dosyasını (Ticker, Date) MultiIndex'e yükler.
    Dönen değer: compute_wf_fitness'in beklediği idx formatı.
    """
    df = pd.read_parquet(SNAPSHOT_PATH)
    df["Date"] = pd.to_datetime(df["Date"])
    return df.set_index(["Ticker", "Date"]).sort_index()


def _compute_reference_real(update: bool = False) -> dict:
    """
    Gerçek BIST snapshot üzerinde referans değerleri hesapla / yükle.
    Snapshot yoksa boş dict döner (testler otomatik atlanır).
    """
    if not os.path.exists(SNAPSHOT_PATH):
        return {}

    if not update and os.path.exists(GOLDEN_REAL_PATH):
        with open(GOLDEN_REAL_PATH, "r") as f:
            return json.load(f)

    cfg   = AlphaCFG()
    idx   = _load_bist_snapshot()
    dates = idx.index.get_level_values("Date").values
    folds = make_date_folds(
        dates, n_folds=_FOLDS, min_fold_days=60, embargo_days=_EMBARGO
    )

    golden = {}
    for formula in REGRESSION_FORMULAS:
        try:
            tree   = parse_formula(formula, cfg)
            result = compute_wf_fitness(
                tree, cfg.evaluate, idx, folds,
                lambda_std=0.5, lambda_cx=0.001,
                min_valid_folds=2,
                target_col="Next_Ret",
                neutralize=False,
                size_corr_hard_limit=0.95,
            )
            golden[formula] = {
                "status":     result["status"],
                "mean_ric":   result["mean_ric"]  if not math.isnan(result.get("mean_ric",  float("nan"))) else None,
                "fitness":    result["fitness"]   if not math.isnan(result.get("fitness",   float("nan"))) else None,
                "complexity": result["complexity"],
                "pos_folds":  result["pos_folds"],
                "n_folds":    len(result["fold_rics"]),
            }
        except Exception as e:
            golden[formula] = {"status": f"exception:{e}"}

    if update:
        os.makedirs(os.path.dirname(GOLDEN_REAL_PATH), exist_ok=True)
        with open(GOLDEN_REAL_PATH, "w") as f:
            json.dump(golden, f, indent=2, ensure_ascii=False)
        print(f"\n✅ Gerçek veri altın dosyası güncellendi: {GOLDEN_REAL_PATH}")

    return golden


# ─── Ana regresyon testi ──────────────────────────────────────────────────────

class TestRegressionGoldenFile:
    """
    §11.3: Referans formüllerin fitness değerleri ±%5'ten fazla sapmamalı.
    """

    @pytest.fixture(scope="class", autouse=True)
    def golden(self, request):
        """Altın dosyayı yükle (veya ilk kez oluştur)."""
        update = request.config.getoption("--update-golden", default=False)
        return _compute_reference(update=update)

    def test_status_unchanged(self, golden):
        """Her formülün status değeri değişmemeli."""
        cfg   = AlphaCFG()
        idx   = make_synthetic_idx(n_tickers=_TICKERS, n_days=_DAYS, seed=_SEED)
        dates = idx.index.get_level_values("Date").values
        folds = make_date_folds(
            dates, n_folds=_FOLDS, min_fold_days=30, embargo_days=_EMBARGO
        )

        if len(folds) < 2:
            pytest.skip("Yeterli fold yok")

        mismatches = []
        for formula in REGRESSION_FORMULAS:
            ref = golden.get(formula, {})
            if "exception" in str(ref.get("status", "")):
                continue  # önceki hata → atla

            tree   = parse_formula(formula, cfg)
            result = compute_wf_fitness(
                tree, cfg.evaluate, idx, folds,
                lambda_std=0.5, lambda_cx=0.001, min_valid_folds=2,
                target_col="Next_Ret", neutralize=False,
                size_corr_hard_limit=0.95,
            )
            if result["status"] != ref.get("status"):
                mismatches.append(
                    f"{formula}: beklenen='{ref.get('status')}', "
                    f"gerçek='{result['status']}'"
                )

        assert not mismatches, (
            f"Status değişiklikleri:\n" + "\n".join(mismatches)
        )

    def test_fitness_within_tolerance(self, golden):
        """fitness değerleri referanstan ±%5'ten az sapmalı."""
        cfg   = AlphaCFG()
        idx   = make_synthetic_idx(n_tickers=_TICKERS, n_days=_DAYS, seed=_SEED)
        dates = idx.index.get_level_values("Date").values
        folds = make_date_folds(
            dates, n_folds=_FOLDS, min_fold_days=30, embargo_days=_EMBARGO
        )

        if len(folds) < 2:
            pytest.skip("Yeterli fold yok")

        violations = []
        for formula in REGRESSION_FORMULAS:
            ref = golden.get(formula, {})
            if ref.get("fitness") is None:
                continue

            tree   = parse_formula(formula, cfg)
            result = compute_wf_fitness(
                tree, cfg.evaluate, idx, folds,
                lambda_std=0.5, lambda_cx=0.001, min_valid_folds=2,
                target_col="Next_Ret", neutralize=False,
                size_corr_hard_limit=0.95,
            )
            if result["status"] not in ("ok",):
                continue

            diff = _relative_diff(result["fitness"], ref["fitness"])
            if diff > _TOL:
                violations.append(
                    f"{formula}: ref={ref['fitness']:.5f}, "
                    f"gerçek={result['fitness']:.5f}, "
                    f"sapma={diff*100:.1f}%"
                )

        assert not violations, (
            f"Fitness tolerans aşımı (>{_TOL*100:.0f}%):\n"
            + "\n".join(violations)
        )

    def test_mean_ric_within_tolerance(self, golden):
        """mean_ric değerleri referanstan ±%5'ten az sapmalı."""
        cfg   = AlphaCFG()
        idx   = make_synthetic_idx(n_tickers=_TICKERS, n_days=_DAYS, seed=_SEED)
        dates = idx.index.get_level_values("Date").values
        folds = make_date_folds(
            dates, n_folds=_FOLDS, min_fold_days=30, embargo_days=_EMBARGO
        )

        if len(folds) < 2:
            pytest.skip("Yeterli fold yok")

        violations = []
        for formula in REGRESSION_FORMULAS:
            ref = golden.get(formula, {})
            if ref.get("mean_ric") is None:
                continue

            tree   = parse_formula(formula, cfg)
            result = compute_wf_fitness(
                tree, cfg.evaluate, idx, folds,
                lambda_std=0.5, lambda_cx=0.001, min_valid_folds=2,
                target_col="Next_Ret", neutralize=False,
                size_corr_hard_limit=0.95,
            )
            if result["status"] not in ("ok",):
                continue

            diff = _relative_diff(result["mean_ric"], ref["mean_ric"])
            if diff > _TOL:
                violations.append(
                    f"{formula}: ref={ref['mean_ric']:.5f}, "
                    f"gerçek={result['mean_ric']:.5f}, "
                    f"sapma={diff*100:.1f}%"
                )

        assert not violations, (
            f"mean_ric tolerans aşımı (>{_TOL*100:.0f}%):\n"
            + "\n".join(violations)
        )

    def test_complexity_unchanged(self, golden):
        """AST karmaşıklığı değişmemeli (formula → node sayısı deterministik)."""
        cfg = AlphaCFG()

        mismatches = []
        for formula in REGRESSION_FORMULAS:
            ref = golden.get(formula, {})
            ref_cx = ref.get("complexity")
            if ref_cx is None:
                continue

            tree = parse_formula(formula, cfg)
            actual_cx = tree.size()
            if actual_cx != ref_cx:
                mismatches.append(
                    f"{formula}: ref={ref_cx}, gerçek={actual_cx}"
                )

        assert not mismatches, (
            "Complexity değişti (AST yapısı değişmiş olabilir):\n"
            + "\n".join(mismatches)
        )


# ─── Altın dosya bootstrap testi ──────────────────────────────────────────────

class TestGoldenFileBootstrap:
    """
    Altın dosya yoksa otomatik oluşturulmalı; oluşturulunca testler geçmeli.
    """

    def test_golden_file_can_be_created(self, tmp_path, monkeypatch):
        """
        Geçici dizine altın dosya oluşturulabilmeli.
        """
        import tests.test_regression as tr_module
        golden_path_orig = tr_module.GOLDEN_PATH
        tmp_golden = str(tmp_path / "golden.json")
        monkeypatch.setattr(tr_module, "GOLDEN_PATH", tmp_golden)

        golden = _compute_reference(update=True)

        assert os.path.exists(tmp_golden), "Altın dosya oluşturulamadı"
        assert len(golden) == len(REGRESSION_FORMULAS)
        for formula in REGRESSION_FORMULAS:
            assert formula in golden, f"'{formula}' altın dosyada yok"

        # Geri yükle
        monkeypatch.setattr(tr_module, "GOLDEN_PATH", golden_path_orig)

    def test_golden_file_json_valid(self, tmp_path, monkeypatch):
        """Oluşturulan JSON geçerli ve yüklenebilir olmalı."""
        import tests.test_regression as tr_module
        tmp_golden = str(tmp_path / "golden2.json")
        monkeypatch.setattr(tr_module, "GOLDEN_PATH", tmp_golden)

        _compute_reference(update=True)

        with open(tmp_golden) as f:
            data = json.load(f)

        assert isinstance(data, dict)
        for formula, vals in data.items():
            assert "status" in vals, f"'{formula}' için 'status' yok"


# ─── Gerçek BIST verisi regresyon testi ───────────────────────────────────────

class TestRegressionGoldenFileReal:
    """
    §11.3 (Gerçek Veri): bist_snapshot.parquet üzerinde referans değerler.

    Snapshot yoksa tüm testler atlanır.
    --update-golden ile hem sentetik hem gerçek altın dosyalar yenilenir.
    """

    @pytest.fixture(scope="class", autouse=True)
    def golden_real(self, request):
        """Gerçek veri altın dosyasını yükle (veya ilk kez oluştur)."""
        if not os.path.exists(SNAPSHOT_PATH):
            pytest.skip("bist_snapshot.parquet yok — gerçek veri testleri atlandı")
        update = request.config.getoption("--update-golden", default=False)
        return _compute_reference_real(update=update)

    def _run_formula(self, formula: str, cfg, idx, folds) -> dict:
        tree = parse_formula(formula, cfg)
        return compute_wf_fitness(
            tree, cfg.evaluate, idx, folds,
            lambda_std=0.5, lambda_cx=0.001, min_valid_folds=2,
            target_col="Next_Ret", neutralize=False,
            size_corr_hard_limit=0.95,
        )

    @pytest.fixture(scope="class")
    def bist_setup(self):
        """Snapshot'tan idx ve fold'ları hazırla."""
        cfg   = AlphaCFG()
        idx   = _load_bist_snapshot()
        dates = idx.index.get_level_values("Date").values
        folds = make_date_folds(
            dates, n_folds=_FOLDS, min_fold_days=60, embargo_days=_EMBARGO
        )
        return cfg, idx, folds

    def test_real_status_unchanged(self, golden_real, bist_setup):
        """Gerçek veri: her formülün status değeri değişmemeli."""
        cfg, idx, folds = bist_setup
        if len(folds) < 2:
            pytest.skip("Yeterli fold yok")

        mismatches = []
        for formula in REGRESSION_FORMULAS:
            ref = golden_real.get(formula, {})
            if "exception" in str(ref.get("status", "")):
                continue
            result = self._run_formula(formula, cfg, idx, folds)
            if result["status"] != ref.get("status"):
                mismatches.append(
                    f"{formula}: beklenen='{ref.get('status')}', "
                    f"gerçek='{result['status']}'"
                )

        assert not mismatches, "Gerçek veri status değişiklikleri:\n" + "\n".join(mismatches)

    def test_real_fitness_within_tolerance(self, golden_real, bist_setup):
        """Gerçek veri: fitness ±%5 tolerans."""
        cfg, idx, folds = bist_setup
        if len(folds) < 2:
            pytest.skip("Yeterli fold yok")

        violations = []
        for formula in REGRESSION_FORMULAS:
            ref = golden_real.get(formula, {})
            if ref.get("fitness") is None:
                continue
            result = self._run_formula(formula, cfg, idx, folds)
            if result["status"] != "ok":
                continue
            diff = _relative_diff(result["fitness"], ref["fitness"])
            if diff > _TOL:
                violations.append(
                    f"{formula}: ref={ref['fitness']:.5f}, "
                    f"gerçek={result['fitness']:.5f}, sapma={diff*100:.1f}%"
                )

        assert not violations, (
            f"Gerçek veri fitness tolerans aşımı (>{_TOL*100:.0f}%):\n"
            + "\n".join(violations)
        )

    def test_real_mean_ric_within_tolerance(self, golden_real, bist_setup):
        """Gerçek veri: mean_ric ±%5 tolerans."""
        cfg, idx, folds = bist_setup
        if len(folds) < 2:
            pytest.skip("Yeterli fold yok")

        violations = []
        for formula in REGRESSION_FORMULAS:
            ref = golden_real.get(formula, {})
            if ref.get("mean_ric") is None:
                continue
            result = self._run_formula(formula, cfg, idx, folds)
            if result["status"] != "ok":
                continue
            diff = _relative_diff(result["mean_ric"], ref["mean_ric"])
            if diff > _TOL:
                violations.append(
                    f"{formula}: ref={ref['mean_ric']:.5f}, "
                    f"gerçek={result['mean_ric']:.5f}, sapma={diff*100:.1f}%"
                )

        assert not violations, (
            f"Gerçek veri mean_ric tolerans aşımı (>{_TOL*100:.0f}%):\n"
            + "\n".join(violations)
        )

    def test_real_snapshot_has_expected_tickers(self):
        """Snapshot 30 ticker içermeli."""
        idx = _load_bist_snapshot()
        n = idx.index.get_level_values("Ticker").nunique()
        assert n == 30, f"Snapshot'ta {n} ticker var, 30 bekleniyor"

    def test_real_snapshot_has_next_ret(self):
        """Snapshot'ta Next_Ret kolonu mevcut olmalı."""
        idx = _load_bist_snapshot()
        assert "Next_Ret" in idx.columns, "Snapshot'ta Next_Ret kolonu yok"
