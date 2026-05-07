"""Mining checkpoint — save/resume long-running formula searches.

Checkpoint dosyaları: data/mining_checkpoints/<checkpoint_id>.pkl
Her N formül değerlendirmesinden sonra otomatik kaydedilir.

Kullanım:
    # Kayıt
    ckpt = Checkpoint.new("window_20260101")
    ckpt.save(done_i=50, pool=pool, results=results, rng_state=random.getstate())

    # Resume
    ckpt = Checkpoint.load("window_20260101")
    start_from = ckpt.done_i + 1
"""
from __future__ import annotations

import logging
import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_DEFAULT_DIR = Path("data/mining_checkpoints")


@dataclass
class CheckpointState:
    checkpoint_id: str
    done_i: int
    pool: list
    results: list
    rng_state: Any
    np_rng_state: Any
    saved_at: float = field(default_factory=time.time)
    metadata: dict = field(default_factory=dict)


class Checkpoint:
    def __init__(self, checkpoint_id: str, directory: Path | None = None):
        self.checkpoint_id = checkpoint_id
        self.directory = Path(directory or _DEFAULT_DIR)
        self._path = self.directory / f"{checkpoint_id}.pkl"

    @classmethod
    def new(cls, checkpoint_id: str, directory: Path | None = None) -> "Checkpoint":
        return cls(checkpoint_id, directory)

    @classmethod
    def load(cls, checkpoint_id: str, directory: Path | None = None) -> "Checkpoint":
        ckpt = cls(checkpoint_id, directory)
        if not ckpt._path.exists():
            raise FileNotFoundError(f"Checkpoint bulunamadı: {ckpt._path}")
        return ckpt

    def save(
        self,
        done_i: int,
        pool: list,
        results: list,
        rng_state: Any,
        np_rng_state: Any,
        metadata: dict | None = None,
    ) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        state = CheckpointState(
            checkpoint_id=self.checkpoint_id,
            done_i=done_i,
            pool=pool,
            results=results,
            rng_state=rng_state,
            np_rng_state=np_rng_state,
            metadata=metadata or {},
        )
        tmp = self._path.with_suffix(".tmp")
        with open(tmp, "wb") as f:
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
        tmp.replace(self._path)
        logger.debug("Checkpoint kaydedildi: %s  done_i=%d", self._path, done_i)

    def read(self) -> CheckpointState:
        with open(self._path, "rb") as f:
            return pickle.load(f)

    def exists(self) -> bool:
        return self._path.exists()

    def delete(self) -> None:
        self._path.unlink(missing_ok=True)

    @staticmethod
    def list_all(directory: Path | None = None) -> list[str]:
        d = Path(directory or _DEFAULT_DIR)
        if not d.exists():
            return []
        return [p.stem for p in sorted(d.glob("*.pkl"))]
