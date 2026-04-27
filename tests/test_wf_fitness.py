"""
tests/test_wf_fitness.py — engine/wf_fitness.py için birim testler.

Belgelenmiş gereksinimler (DOKUMAN_ONERILER.md §11.1):
  - test_size_factor_rejection : saf Pclose formülü → status="size_factor"
  - test_complexity_count      : _node_complexity(Delta(Pclose,20)) == 3
  - test_folds_non_overlapping : max(folds[i]) < min(folds[i+1])  (sıralı)

Çalıştırma:
    python -m pytest tests/ -v
    # veya
    python -m pytest tests/test_wf_fitness.py -v
"""
import sys
import os

# Proje kökünü Python yoluna ekle
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import pytest
import numpy as np

from engine.core.alpha_cfg import AlphaCFG, Node
from engine.validation.wf_fitness import _node_complexity, make_date_folds, make_purged_date_folds, compute_wf_fitness
from engine.core.formula_parser import parse_formula
from tests.conftest import make_synthetic_idx


# ─── _node_complexity testleri ────────────────────────────────────────────────

class TestNodeComplexity:
    """
    _node_complexity(node) → AST node sayısı.

    AlphaCFG.generate() ile üretilen ağaçlarda doğrulama:
    - Yaprak node (feature/constant) → 1
    - Unary op (1 çocuk) → 2  (kök + 1 çocuk)
    - Binary op (2 çocuk) → 3  (kök + 2 çocuk, yapraklar basit)
    - Derinlik arttıkça toplam node sayısı artar.
    - size() ile tutarlı: _node_complexity == node.size()
    """

    def setup_method(self):
        self.cfg = AlphaCFG()

    def _make_feature_node(self, name: str = "Pclose") -> Node:
        """Tek yaprak node (feature) oluştur."""
        n = Node("feature", name)
        return n

    def _make_constant_node(self, val: float = 0.5) -> Node:
        """Tek yaprak node (constant) oluştur."""
        n = Node("constant", val)
        return n

    # ── temel vakalar ─────────────────────────────────────────────────────

    def test_leaf_feature_node(self):
        """Tek feature node: complexity = 1."""
        node = self._make_feature_node("Pclose")
        assert _node_complexity(node) == 1

    def test_leaf_constant_node(self):
        """Tek constant node: complexity = 1."""
        node = self._make_constant_node(0.5)
        assert _node_complexity(node) == 1

    def test_unary_node(self):
        """Unary(Pclose): complexity = 2 (kök + 1 çocuk)."""
        child = self._make_feature_node("Pclose")
        root  = Node("unary", "Abs", children=[child])
        assert _node_complexity(root) == 2

    def test_binary_node_two_leaves(self):
        """Binary(Pclose, Vlot): complexity = 3 (kök + 2 yaprak)."""
        left  = self._make_feature_node("Pclose")
        right = self._make_feature_node("Vlot")
        root  = Node("binary", "Add", children=[left, right])
        assert _node_complexity(root) == 3

    def test_nested_unary(self):
        """Abs(Abs(Pclose)): complexity = 3."""
        leaf  = self._make_feature_node("Pclose")
        inner = Node("unary", "Abs", children=[leaf])
        outer = Node("unary", "Abs", children=[inner])
        assert _node_complexity(outer) == 3

    def test_deep_binary_tree(self):
        """
        Add(Sub(Pclose, Vlot), Mul(Popen, Phigh)):
        complexity = 7  (1 kök + 2 iç-ikili + 4 yaprak)
        """
        ll = self._make_feature_node("Pclose")
        lr = self._make_feature_node("Vlot")
        rl = self._make_feature_node("Popen")
        rr = self._make_feature_node("Phigh")
        left  = Node("binary", "Sub", children=[ll, lr])
        right = Node("binary", "Mul", children=[rl, rr])
        root  = Node("binary", "Add", children=[left, right])
        assert _node_complexity(root) == 7

    # ── node.size() ile tutarlılık ──────────────────────────────────────

    def test_consistency_with_node_size(self):
        """_node_complexity her zaman node.size() ile aynı değer döndürmeli."""
        for _ in range(20):
            tree = self.cfg.generate(max_K=10)
            assert _node_complexity(tree) == tree.size(), (
                f"Tutarsızlık: _node_complexity={_node_complexity(tree)}, "
                f"tree.size()={tree.size()}, tree={tree}"
            )

    # ── sınır vakaları ────────────────────────────────────────────────────

    def test_no_children_attribute(self):
        """children yoksa veya None ise → 1 (yaprak gibi davranır)."""
        n = Node("feature", "Pclose")
        # children'ı manuel sil
        object.__setattr__(n, "children", None)
        assert _node_complexity(n) >= 1   # crash olmamalı

    def test_generated_tree_positive(self):
        """Tüm üretilen ağaçlar için complexity > 0."""
        for _ in range(50):
            tree = self.cfg.generate(max_K=15)
            c = _node_complexity(tree)
            assert c > 0, f"complexity sıfır veya negatif: {c}"
            assert isinstance(c, int), f"complexity int olmalı: {type(c)}"

    def test_larger_tree_bigger_complexity(self):
        """Daha büyük max_k → ortalama complexity yüksek beklenir (stokastik)."""
        small_avg = np.mean([_node_complexity(self.cfg.generate(3))  for _ in range(30)])
        large_avg = np.mean([_node_complexity(self.cfg.generate(15)) for _ in range(30)])
        assert large_avg > small_avg, (
            f"Büyük ağaçlar daha karmaşık olmalı: "
            f"small_avg={small_avg:.1f}, large_avg={large_avg:.1f}"
        )


# ─── make_date_folds testleri ──────────────────────────────────────────────────

class TestMakeDateFolds:
    """
    make_date_folds(dates, n_folds, min_fold_days, embargo_days) → list[np.ndarray]
    """

    def _make_dates(self, n: int):
        """n iş günü üret (business days)."""
        return np.array(
            pd.date_range("2020-01-01", periods=n, freq="B"), dtype="datetime64[ns]"
        )

    def test_returns_list_of_arrays(self):
        """Dönen değer bir list; her eleman np.ndarray (datetime64)."""
        dates  = self._make_dates(200)
        folds  = make_date_folds(dates, n_folds=5)
        assert isinstance(folds, list)
        assert all(isinstance(f, np.ndarray) for f in folds)

    def test_fold_count(self):
        """Yeterli veri varsa n_folds kadar fold üretilmeli."""
        dates = self._make_dates(500)
        folds = make_date_folds(dates, n_folds=5, min_fold_days=20)
        assert len(folds) == 5

    def test_embargo_reduces_fold_size(self):
        """embargo_days=5 olduğunda fold'lar embargo=0'dan küçük olmalı."""
        dates  = self._make_dates(500)
        f_no   = make_date_folds(dates, n_folds=5, embargo_days=0)
        f_emb  = make_date_folds(dates, n_folds=5, embargo_days=5)
        # Toplam gün sayısı embargo ile azalmalı
        total_no  = sum(len(f) for f in f_no)
        total_emb = sum(len(f) for f in f_emb)
        assert total_emb < total_no, (
            f"Embargo fold boyutunu küçültmeli: no={total_no}, emb={total_emb}"
        )

    def test_no_overlap_between_folds(self):
        """Foldlar non-overlapping (kesişmemeli)."""
        dates = self._make_dates(500)
        folds = make_date_folds(dates, n_folds=5)
        all_dates = []
        for f in folds:
            all_dates.extend(f.tolist())
        assert len(all_dates) == len(set(all_dates)), "Foldlar çakışıyor!"

    def test_too_few_dates_reduces_folds(self):
        """Çok az veri varsa n_folds otomatik düşürülmeli (crash olmamalı)."""
        dates = self._make_dates(50)
        folds = make_date_folds(dates, n_folds=10, min_fold_days=20)
        # Crash olmamalı, sonuç liste olmalı
        assert isinstance(folds, list)
        assert len(folds) <= 10

    def test_folds_non_overlapping_ordered(self):
        """
        §11.1 Kritik Test:
        max(folds[i]) < min(folds[i+1]) — fold'lar kronolojik sırada ve sıkışık.
        Zaman sızıntısı yokluğunun temel garantisi.
        """
        dates = self._make_dates(500)
        folds = make_date_folds(dates, n_folds=5, embargo_days=0)
        assert len(folds) >= 2
        for i in range(len(folds) - 1):
            assert folds[i].max() < folds[i + 1].min(), (
                f"Fold {i} ve {i+1} sıralı değil veya çakışıyor: "
                f"max(fold[{i}])={folds[i].max()}, "
                f"min(fold[{i+1}])={folds[i+1].min()}"
            )

    def test_folds_ordered_with_embargo(self):
        """embargo_days>0 ile de fold sıralı olmalı."""
        dates = self._make_dates(500)
        folds = make_date_folds(dates, n_folds=5, embargo_days=5)
        if len(folds) >= 2:
            for i in range(len(folds) - 1):
                assert folds[i].max() < folds[i + 1].min()


# ─── compute_wf_fitness testleri ─────────────────────────────────────────────

class TestComputeWfFitness:
    """
    compute_wf_fitness → dict with status, fitness, fold_rics, etc.
    """

    @pytest.fixture(autouse=True)
    def setup(self):
        self.cfg = AlphaCFG()
        self.idx = make_synthetic_idx(n_tickers=25, n_days=350, seed=99)
        dates     = self.idx.index.get_level_values("Date").values
        self.folds = make_date_folds(
            dates, n_folds=4, min_fold_days=30, embargo_days=3
        )

    def test_size_factor_rejection(self):
        """
        §11.1 Kritik Test:
        Saf Pclose (= size faktörü) → status = "size_factor".

        Log(Pclose) sinyali cross-sectional size ile ≈ 1.0 korelasyona sahip.
        hard limit = 0.7 → direkt reddedilmeli.
        """
        if len(self.folds) < 3:
            pytest.skip("Yeterli fold yok")

        tree   = parse_formula("Log(Pclose)", self.cfg)
        result = compute_wf_fitness(
            tree, self.cfg.evaluate, self.idx, self.folds,
            size_corr_hard_limit=0.7,   # varsayılan limit
            neutralize=False,
        )
        assert result["status"] == "size_factor", (
            f"Log(Pclose) size_factor olarak reddedilmeli, "
            f"aldığı status='{result['status']}', "
            f"size_corr={result.get('size_corr', 'N/A'):.3f}"
        )

    def test_complexity_count(self):
        """
        §11.1 Kritik Test:
        Delta(Pclose, 20): kök(rolling) + Pclose(feature) + 20(num) = 3 node.
        """
        tree = parse_formula("Delta(Pclose, 20)", self.cfg)
        assert _node_complexity(tree) == 3, (
            f"Delta(Pclose, 20) complexity 3 olmalı, "
            f"bulundu: {_node_complexity(tree)}, tree={tree}"
        )

    def test_result_structure(self):
        """Dönüş dict'i beklenen tüm anahtarları içermeli."""
        if len(self.folds) < 3:
            pytest.skip("Yeterli fold yok")

        required_keys = {
            "status", "fitness", "fold_rics", "mean_ric", "std_ric",
            "pos_folds", "complexity", "rank_ic", "ic", "size_corr",
        }
        tree   = parse_formula("Delta(Pclose, 5)", self.cfg)
        result = compute_wf_fitness(
            tree, self.cfg.evaluate, self.idx, self.folds,
            size_corr_hard_limit=0.95, neutralize=False,
        )
        missing = required_keys - set(result.keys())
        assert not missing, f"Eksik anahtarlar: {missing}"

    def test_status_values_are_valid(self):
        """Status değerleri tanımlı kümeden biri olmalı."""
        if len(self.folds) < 3:
            pytest.skip("Yeterli fold yok")

        valid_statuses = {"ok", "invalid", "empty", "size_factor", "error"}
        formulas = ["Delta(Pclose, 5)", "Std(Pclose, 20)", "Log(Pclose)"]
        for f in formulas:
            tree   = parse_formula(f, self.cfg)
            result = compute_wf_fitness(
                tree, self.cfg.evaluate, self.idx, self.folds,
                size_corr_hard_limit=0.7, neutralize=False,
            )
            assert result["status"] in valid_statuses, (
                f"'{f}' geçersiz status: '{result['status']}'"
            )

    def test_complexity_matches_tree_size(self):
        """Sonuçtaki complexity, ağacın gerçek node sayısıyla eşleşmeli."""
        if len(self.folds) < 3:
            pytest.skip("Yeterli fold yok")

        tree   = parse_formula("Delta(Pclose, 20)", self.cfg)
        result = compute_wf_fitness(
            tree, self.cfg.evaluate, self.idx, self.folds,
            size_corr_hard_limit=0.95, neutralize=False,
        )
        assert result["complexity"] == tree.size()

    def test_fitness_decreases_with_lambda_std(self):
        """
        Daha yüksek lambda_std → fitness azalmalı
        (std penalty daha ağır basıyor).
        """
        if len(self.folds) < 3:
            pytest.skip("Yeterli fold yok")

        tree = parse_formula("Delta(Pclose, 5)", self.cfg)

        r_low  = compute_wf_fitness(
            tree, self.cfg.evaluate, self.idx, self.folds,
            lambda_std=0.0, lambda_cx=0.0, size_corr_hard_limit=0.95,
            neutralize=False,
        )
        r_high = compute_wf_fitness(
            tree, self.cfg.evaluate, self.idx, self.folds,
            lambda_std=5.0, lambda_cx=0.0, size_corr_hard_limit=0.95,
            neutralize=False,
        )
        if r_low["status"] == "ok" and r_high["status"] == "ok":
            assert r_high["fitness"] <= r_low["fitness"], (
                f"Yüksek lambda_std daha düşük fitness vermeli: "
                f"low={r_low['fitness']:.4f}, high={r_high['fitness']:.4f}"
            )

    def test_pclose_raw_rejected_size_hard_limit(self):
        """
        Ham Pclose formülü (Abs(Pclose) gibi price-level proxy)
        size_corr_hard_limit=0.7 ile reddedilmeli.
        """
        if len(self.folds) < 3:
            pytest.skip("Yeterli fold yok")

        tree   = parse_formula("Abs(Pclose)", self.cfg)
        result = compute_wf_fitness(
            tree, self.cfg.evaluate, self.idx, self.folds,
            size_corr_hard_limit=0.7, neutralize=False,
        )
        # Abs(Pclose) da price-level signal: size_corr yüksek
        assert result["status"] == "size_factor", (
            f"Abs(Pclose) size_factor reddedilmeli, "
            f"status='{result['status']}'"
        )


# ─── Purged K-Fold testleri ───────────────────────────────────────────────────

class TestPurgedFolds:
    """
    make_purged_date_folds(dates, n_folds, embargo_days, purge_horizon)
    → list[dict]  her eleman: {"test": np.ndarray, "train": np.ndarray}

    López de Prado, AFML §7: Test fold başlangıcından geriye purge_horizon gün
    train'den çıkarılır — TB label horizon'u kadar sızan bilgiyi engeller.
    """

    def _make_dates(self, n: int, freq: str = "B"):
        return np.array(
            pd.date_range("2020-01-01", periods=n, freq=freq), dtype="datetime64[ns]"
        )

    def test_returns_list_of_dicts(self):
        """Dönen değer list[dict]; her dict 'test' ve 'train' anahtarına sahip."""
        dates = self._make_dates(400)
        folds = make_purged_date_folds(dates, n_folds=4)
        assert isinstance(folds, list)
        assert len(folds) > 0
        for f in folds:
            assert isinstance(f, dict), "Her eleman dict olmalı"
            assert "test" in f and "train" in f

    def test_purge_zero_equivalent_to_classic(self):
        """
        purge_horizon=0 → 'test' tarihlerinin sırası ve kapsamı
        make_date_folds ile özdeş olmalı (train boşluk farkı olabilir).
        """
        dates = self._make_dates(500)
        classic = make_date_folds(dates, n_folds=5, embargo_days=5)
        purged  = make_purged_date_folds(dates, n_folds=5, embargo_days=5, purge_horizon=0)

        assert len(classic) == len(purged), (
            f"Fold sayısı eşit olmalı: classic={len(classic)}, purged={len(purged)}"
        )
        for i, (c, p) in enumerate(zip(classic, purged)):
            np.testing.assert_array_equal(
                c, p["test"],
                err_msg=f"Fold {i}: test tarihleri özdeş olmalı"
            )

    def test_purge_removes_dates_before_test_start(self):
        """
        purge_horizon=10 → her fold'un train kümesinde,
        test_start - 10 gün ile test_start arasında tarih bulunmamalı.
        """
        dates = self._make_dates(500)
        folds = make_purged_date_folds(dates, n_folds=5, embargo_days=3, purge_horizon=10)
        for i, f in enumerate(folds):
            test_start  = f["test"][0]
            purge_start = test_start - np.timedelta64(10, "D")
            # Bu aralıktaki tarihler train'de olmamalı
            leaked = f["train"][
                (f["train"] > purge_start) & (f["train"] < test_start)
            ]
            assert len(leaked) == 0, (
                f"Fold {i}: train'de purge bölgesinde {len(leaked)} tarih var!"
            )

    def test_embargo_and_purge_no_overlap(self):
        """
        Embargo + purge birlikte kullanıldığında test fold'ları
        birbiriyle çakışmamalı (non-overlapping).
        """
        dates = self._make_dates(600)
        folds = make_purged_date_folds(dates, n_folds=5, embargo_days=5, purge_horizon=10)
        all_test_dates = []
        for f in folds:
            all_test_dates.extend(f["test"].tolist())
        unique_test = set(all_test_dates)
        assert len(all_test_dates) == len(unique_test), (
            "Test fold'ları arasında çakışma var!"
        )

    def test_small_data_graceful_fallback(self):
        """Az veri durumunda crash olmadan çalışmalı."""
        dates = self._make_dates(60)
        folds = make_purged_date_folds(dates, n_folds=10, embargo_days=5, purge_horizon=10)
        assert isinstance(folds, list)   # crash yok
        assert len(folds) <= 10

    def test_train_size_decreases_with_larger_purge(self):
        """
        Purge horizon arttıkça train kümesi küçülmeli.
        Her fold'un train gün sayısı purge_horizon=0 durumundan az olmalı.
        """
        dates = self._make_dates(500)
        folds_no_purge = make_purged_date_folds(
            dates, n_folds=4, embargo_days=3, purge_horizon=0
        )
        folds_purge = make_purged_date_folds(
            dates, n_folds=4, embargo_days=3, purge_horizon=15
        )
        assert len(folds_no_purge) == len(folds_purge), "Fold sayısı eşit olmalı"
        for i, (f0, fp) in enumerate(zip(folds_no_purge, folds_purge)):
            assert len(fp["train"]) <= len(f0["train"]), (
                f"Fold {i}: purge ile train küçülmeli — "
                f"no_purge={len(f0['train'])}, purge={len(fp['train'])}"
            )

    def test_compute_wf_fitness_accepts_purged_folds(self):
        """
        compute_wf_fitness, list[dict] (purged) formatını kabul etmeli
        ve 'ok' veya geçerli bir status döndürmeli.
        """
        cfg = AlphaCFG()
        idx = make_synthetic_idx(n_tickers=20, n_days=300, seed=77)
        all_dates = idx.index.get_level_values("Date").values
        purged_folds = make_purged_date_folds(
            all_dates, n_folds=4, embargo_days=3, purge_horizon=10
        )
        if len(purged_folds) < 3:
            pytest.skip("Yeterli fold yok")

        tree   = parse_formula("Delta(Pclose, 5)", cfg)
        result = compute_wf_fitness(
            tree, cfg.evaluate, idx, purged_folds,
            size_corr_hard_limit=0.95, neutralize=False,
        )
        valid_statuses = {"ok", "invalid", "empty", "size_factor", "error"}
        assert result["status"] in valid_statuses, (
            f"Purged folds ile compute_wf_fitness geçersiz status: {result['status']}"
        )
