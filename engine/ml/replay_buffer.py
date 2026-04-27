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
from collections import deque
from typing import Deque, List, Optional, Tuple

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
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        serial = [(_tree_to_dict(t), ic, vd) for (t, ic, vd) in self.buf]
        payload = {"version": 3, "data": serial}
        # JSON: arbitrary code execution riski yok (pickle'ın aksine)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)

    def load(self) -> "ReplayBuffer":
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
        """JSON'dan oku; yoksa eski .pkl'a bak (migration path)."""
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
                # Migration: hemen JSON'a taşı, pkl'ı sil
                self._write_json_from_raw(data)
                os.remove(legacy_pkl)
                return data
            except Exception:
                return None
        return None

    def _write_json_from_raw(self, raw) -> None:
        """Migration sırasında raw pickle içeriğini JSON'a yazar."""
        if isinstance(raw, dict) and raw.get("version") == 2:
            items = raw["data"]
        else:
            return  # v1 legacy Node nesneleri JSON'a yazılamaz, atla
        payload = {"version": 3, "data": items}
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)

    def clear(self) -> None:
        self.buf.clear()
