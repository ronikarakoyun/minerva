"""Birim testler: engine/strategies/mcts_pool.py + engine/util/shm_registry.py."""
from __future__ import annotations

import os
import time
import numpy as np
import pandas as pd
import pytest

from engine.strategies.mcts_pool import shared_array
from engine.util.shm_registry import (
    _pid_alive,
    cleanup_stale,
    register,
    unregister,
    _load,
)


# ─── shared_array ─────────────────────────────────────────────────────────────

def test_shared_array_data_integrity():
    """Shared memory'e yazılan veri doğru okunabilmeli."""
    arr = np.arange(100, dtype=np.float64).reshape(10, 10)
    with shared_array(arr) as (name, view, shape, dtype):
        assert name.startswith("minerva_mcts_")
        assert shape == (10, 10)
        assert dtype == "float64"
        np.testing.assert_array_equal(view, arr)


def test_shared_array_cleaned_after_exit(tmp_path):
    """Blok context manager'dan çıkınca shm unlink edilmeli."""
    from multiprocessing import shared_memory as shm_mod
    arr = np.ones((50,), dtype=np.float32)
    captured_name = None
    with shared_array(arr) as (name, _view, _shape, _dtype):
        captured_name = name
    # Context manager çıktıktan sonra aynı isimle attach edilememeli
    with pytest.raises(FileNotFoundError):
        s = shm_mod.SharedMemory(name=captured_name, create=False)
        s.close()


def test_shared_array_large_float32():
    """16 GB M2'de güvenli; float32 bellek tasarrufu test."""
    arr = np.random.rand(1000, 200).astype(np.float32)  # 800 KB
    with shared_array(arr) as (_name, view, shape, dtype):
        assert dtype == "float32"
        assert shape == (1000, 200)
        np.testing.assert_allclose(view, arr, rtol=1e-6)


# ─── shm_registry ─────────────────────────────────────────────────────────────

def test_register_unregister(tmp_path):
    """Register → kayıt defterinde var. Unregister → yok."""
    reg_path = tmp_path / "shm_registry.json"
    register("test_blk_01", path=reg_path)
    assert "test_blk_01" in _load(reg_path)
    unregister("test_blk_01", path=reg_path)
    assert "test_blk_01" not in _load(reg_path)


def test_cleanup_stale_removes_dead_pid(tmp_path):
    """Ölü PID'e ait kayıt (gerçek SHM olmadan) temizlenmeli (dry_run=True)."""
    reg_path = tmp_path / "shm_registry.json"
    fake_name = "minerva_mcts_dead_test"
    reg_path.write_text(f'{{"{fake_name}": {{"pid": 999999999}}}}')
    cleaned = cleanup_stale(path=reg_path, dry_run=True)
    assert fake_name in cleaned


def test_pid_alive_current():
    """Kendi PID'imiz alive olmalı."""
    assert _pid_alive(os.getpid()) is True


def test_pid_alive_dead():
    """Olmayan PID alive olmamalı."""
    assert _pid_alive(999999999) is False
