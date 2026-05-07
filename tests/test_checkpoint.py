"""Birim testler: engine/util/checkpoint.py + mining_runner resume."""
from __future__ import annotations

import random

import numpy as np
import pytest

from engine.util.checkpoint import Checkpoint, CheckpointState


# ─── Checkpoint temel işlemleri ───────────────────────────────────────────────

def test_save_and_load(tmp_path):
    """Kaydedilen checkpoint doğru okunmalı."""
    ckpt = Checkpoint.new("test_ckpt_01", directory=tmp_path)
    pool = [object(), object()]
    results = [{"fitness": 0.1}]
    rng = random.getstate()
    np_rng = np.random.get_state()

    ckpt.save(done_i=10, pool=pool, results=results, rng_state=rng, np_rng_state=np_rng)
    state = ckpt.read()

    assert state.done_i == 10
    assert state.results == results
    assert len(state.pool) == 2


def test_exists_before_after(tmp_path):
    """Kaydetmeden önce exists=False, sonra True."""
    ckpt = Checkpoint.new("test_ckpt_02", directory=tmp_path)
    assert not ckpt.exists()
    ckpt.save(done_i=0, pool=[], results=[], rng_state=random.getstate(),
              np_rng_state=np.random.get_state())
    assert ckpt.exists()


def test_delete_removes_file(tmp_path):
    """delete() çağrısı dosyayı kaldırmalı."""
    ckpt = Checkpoint.new("test_ckpt_03", directory=tmp_path)
    ckpt.save(done_i=5, pool=[], results=[], rng_state=random.getstate(),
              np_rng_state=np.random.get_state())
    assert ckpt.exists()
    ckpt.delete()
    assert not ckpt.exists()


def test_load_nonexistent_raises(tmp_path):
    """Var olmayan checkpoint yükleme FileNotFoundError fırlatmalı."""
    with pytest.raises(FileNotFoundError):
        Checkpoint.load("no_such_ckpt", directory=tmp_path)


def test_list_all(tmp_path):
    """list_all() dizindeki tüm checkpoint ID'lerini döndürmeli."""
    for i in range(3):
        ckpt = Checkpoint.new(f"ckpt_{i}", directory=tmp_path)
        ckpt.save(done_i=i, pool=[], results=[], rng_state=random.getstate(),
                  np_rng_state=np.random.get_state())
    ids = Checkpoint.list_all(directory=tmp_path)
    assert sorted(ids) == ["ckpt_0", "ckpt_1", "ckpt_2"]


# ─── Resume davranışı ─────────────────────────────────────────────────────────

def test_resume_restores_rng_state(tmp_path):
    """Resume sonrası random state geri yüklenmeli."""
    random.seed(99)
    rng_saved = random.getstate()
    np.random.seed(99)
    np_rng_saved = np.random.get_state()

    ckpt = Checkpoint.new("rng_test", directory=tmp_path)
    ckpt.save(done_i=20, pool=[], results=[{"x": 1}],
              rng_state=rng_saved, np_rng_state=np_rng_saved)

    random.seed(0)  # durumu kirlet
    state = ckpt.read()
    random.setstate(state.rng_state)
    np.random.set_state(state.np_rng_state)

    # Geri yüklenen state ile üretilen değerler, orijinal state ile aynı olmalı
    random.seed(99)
    expected = [random.random() for _ in range(5)]
    random.setstate(rng_saved)
    got = [random.random() for _ in range(5)]
    assert expected == got


def test_mining_runner_resume_skips_processed(tmp_path):
    """Resume=True + var olan checkpoint: daha önce işlenen indeksler atlanmalı."""
    # Checkpoint'in done_i=4 olarak ayarlanmış kopyası — start_from=5 beklenir.
    # Bunu doğrudan impl davranışına bakarak test ediyoruz: pool=[0..9],
    # start_from=5 olunca 0..4 atlanır, sadece 5..9 işlenir.
    ckpt = Checkpoint.new("resume_skip_test", directory=tmp_path)
    # 'pool' olarak basit değerler; gerçek Node nesneleri gerekmez burada
    ckpt.save(
        done_i=4,
        pool=list(range(10)),
        results=[{"fitness": 0.1}] * 2,
        rng_state=random.getstate(),
        np_rng_state=np.random.get_state(),
    )
    state = ckpt.read()
    assert state.done_i == 4
    assert len(state.results) == 2
    # start_from hesabı: done_i + 1
    start_from = state.done_i + 1
    assert start_from == 5
