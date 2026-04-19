"""
engine/replay_buffer.py — AlphaCFG §4.2 replay buffer.

Her örnek: (ASR tree, ölçülen IC, [opsiyonel] MCTS visit distribution).
Diske pickle ile serileştirilir; oturumlar arası sürer.

Not: Streamlit hot-reload sırasında `Node` class-identity değiştiği için
pickle doğrudan Node objesi üzerinde çalışmaz. Bu yüzden Node'ları
serileştirirken dict'e, okurken dict'ten Node'a çeviriyoruz.
"""
import os
import pickle
import random
from collections import deque
from typing import Deque, List, Optional, Tuple

from .alpha_cfg import Node

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
                 path: str = "data/replay_buffer.pkl"):
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
        # Node'ları dict'e çevirerek serialize et → class-identity sorunu yok
        serial = [(_tree_to_dict(t), ic, vd) for (t, ic, vd) in self.buf]
        with open(self.path, "wb") as f:
            pickle.dump({"version": 2, "data": serial}, f)

    def load(self) -> "ReplayBuffer":
        if not os.path.exists(self.path):
            return self
        try:
            with open(self.path, "rb") as f:
                raw = pickle.load(f)
        except Exception:
            return self

        # v2 format: {"version": 2, "data": [(dict, ic, vd), ...]}
        if isinstance(raw, dict) and raw.get("version") == 2:
            items = raw["data"]
        else:
            # v1 legacy: [(Node, ic, vd), ...]
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

    def clear(self) -> None:
        self.buf.clear()
