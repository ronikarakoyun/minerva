"""
engine/replay_buffer.py — AlphaCFG §4.2 replay buffer.

Her örnek: (ASR tree, ölçülen IC, [opsiyonel] MCTS visit distribution).
Diske JSON ile serileştirilir (pickle yerine — arbitrary code execution riski yok).

Not: Streamlit hot-reload sırasında `Node` class-identity değiştiği için
Node'ları serileştirirken dict'e, okurken dict'ten Node'a çeviriyoruz.
"""
import json
import os
import pickle  # sadece eski .pkl dosyaları okumak için (migration)
import random
import tempfile
from collections import deque
from typing import Deque, List, Optional, Tuple

from filelock import FileLock

from ..core.alpha_cfg import Node

Sample = Tuple[Node, float, Optional[List[float]]]


def _tree_to_dict(n: Node) -> dict:
    return {"k": n.kind, "o": n.op,
            "c": [_tree_to_dict(ch) for ch in n.children]}


def _tree_from_dict(d: dict) -> Node:
    return Node(d["k"], d["o"], [_tree_from_dict(c) for c in d["c"]])


def _rebuild_legacy(t) -> Node:
    """Eski (farklı class-identity'li) Node instance'larını yeniden inşa et."""
    if isinstance(t, dict):
        return _tree_from_dict(t)
    # duck-typing: attribute bazlı
    kids = [_rebuild_legacy(c) for c in getattr(t, "children", [])]
    return Node(getattr(t, "kind"), getattr(t, "op"), kids)


class ReplayBuffer:
    def __init__(self, capacity: int = 10000,
                 path: str = "data/replay_buffer.json"):
        self.capacity = capacity
        self.path = path
        self.buf: Deque[Sample] = deque(maxlen=capacity)
        # N57: Lazy init — read-only sistemde __init__'te lock dosyası yaratmak fail eder.
        # Lock nesnesini oluştur ama dosyayı dokunma (FileLock acquire'da yaratır).
        try:
            self._lock = FileLock(path + ".lock", timeout=30)
        except Exception:
            # Son çare: işlevsiz lock (threading.Lock ile değiştir)
            import threading
            self._lock = threading.Lock()  # type: ignore[assignment]

    def add(self, tree: Node, ic: float,
            visit_dist: Optional[List[float]] = None) -> None:
        self.buf.append((tree, float(ic), visit_dist))

    def extend(self, samples: List[Sample]) -> None:
        for s in samples:
            self.buf.append(s)

    def __len__(self) -> int:
        return len(self.buf)

    def sample(self, batch_size: int) -> List[Sample]:
        n = min(batch_size, len(self.buf))
        return random.sample(list(self.buf), n)

    # ---- persistans ----
    def save(self) -> None:
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        serial = [(_tree_to_dict(t), ic, vd) for (t, ic, vd) in self.buf]
        payload = {"version": 3, "data": serial}
        # Atomic write: tmp → rename + cross-process filelock
        with self._lock:
            dir_ = os.path.dirname(self.path) or "."
            fd, tmp_path = tempfile.mkstemp(dir=dir_, suffix=".tmp")
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    json.dump(payload, f, ensure_ascii=False)
                os.replace(tmp_path, self.path)
            except Exception:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                raise

    def load(self) -> "ReplayBuffer":
        with self._lock:
            raw = self._read_raw()
        if raw is None:
            return self

        if isinstance(raw, dict) and raw.get("version") in (2, 3):
            items = raw["data"]
        else:
            # v1 legacy pickle: [(Node, ic, vd), ...]
            items = raw

        out: List[Sample] = []
        for rec in items:
            try:
                t, ic, vd = rec
                node = _rebuild_legacy(t)
                out.append((node, float(ic), vd))
            except Exception:
                continue
        self.buf = deque(out, maxlen=self.capacity)
        return self

    def _read_raw(self):
        """
        JSON'dan oku; yoksa eski .pkl'a bak (migration path).

        N45: Migration (pkl → json) da filelock altında — çağıran load()
        zaten lock tutuyor, bu method lock almaz (tekrar girme → deadlock).
        """
        # Önce JSON
        if os.path.exists(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return None
        # Geri dönüş: eski pkl adı (data/replay_buffer.pkl)
        legacy_pkl = self.path.replace(".json", ".pkl")
        if os.path.exists(legacy_pkl):
            try:
                with open(legacy_pkl, "rb") as f:
                    data = pickle.load(f)  # noqa: S301 — migration only
                # N45: Migration lock altında (çağıran load() lock'u tutuyor)
                self._write_json_from_raw(data)
                os.remove(legacy_pkl)
                return data
            except Exception:
                return None
        return None

    def _write_json_from_raw(self, raw) -> None:
        """
        Migration sırasında raw pickle içeriğini JSON'a yazar.
        N45: Bu metod filelock almaz — çağıran (load → _read_raw) zaten lock tutuyor.
        """
        if isinstance(raw, dict) and raw.get("version") == 2:
            items = raw["data"]
        else:
            return  # v1 legacy Node nesneleri JSON'a yazılamaz, atla
        payload = {"version": 3, "data": items}
        # N57: makedirs read-only sistemde fail edebilir → try/except
        try:
            os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        except OSError:
            return  # Dizin oluşturulamazsa migration atla
        dir_ = os.path.dirname(self.path) or "."
        fd, tmp_path = tempfile.mkstemp(dir=dir_, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False)
            os.replace(tmp_path, self.path)
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    def clear(self) -> None:
        self.buf.clear()
