"""
engine/alpha_cfg.py — AlphaCFG (arXiv 2601.22119) α-Sem-k grameri +
QuantaAlpha (arXiv 2602.07085) operatör kütüphanesi, AST (ASR) tabanlı.

Kaynaklar:
- Tablo 4  → 6 temel özellik (open, high, low, close, volume, vwap)
- Tablo 5  → Constant ∈ {-0.1,-0.05,-0.01,0.01,0.05,0.1}, Num ∈ {20,30,40}
- Tablo 6  → Tam operatör kümesi (Unary/Binary/BinaryAsym/Rolling/PairedRolling/CS)
- Tablo 7  → Üretim kuralı başına Δk (length increment)
- Eq. 5    → AST-tabanlı maksimum ortak izomorf altağaç benzerliği
- Alg.  2  → Uzunluk kısıtlı üretim  (k + Δk(l) ≤ K)
"""
import numpy as np
import pandas as pd
import random
from copy import deepcopy
from dataclasses import dataclass, field
from typing import List


# ---------- AST düğümü (Definition 1: Abstract Syntax Representation) ----------
@dataclass
class Node:
    kind: str                       # feature / constant / num / unary / binary /
                                    # binary_asym / rolling / paired_rolling / cs_op
    op: str
    children: List["Node"] = field(default_factory=list)

    def __str__(self):
        if self.kind in ("feature", "constant", "num"):
            return str(self.op)
        args = ", ".join(str(c) for c in self.children)
        return f"{self.op}({args})"

    def size(self) -> int:
        return 1 + sum(c.size() for c in self.children)


# ---------- Yardımcı: ticker bazlı groupby ----------
def _grp(x: pd.Series):
    return x.groupby(level="Ticker", group_keys=False)


def _paired_rolling(x: pd.Series, y: pd.Series, w: int, fn: str) -> pd.Series:
    d = pd.DataFrame({"x": x, "y": y})
    def _ap(g):
        if fn == "corr":
            return g["x"].rolling(int(w)).corr(g["y"])
        return g["x"].rolling(int(w)).cov(g["y"])
    return d.groupby(level="Ticker", group_keys=False).apply(_ap)


class AlphaCFG:
    # -------- Tablo 4: özellikler --------
    FEATURES = ["Popen", "Phigh", "Plow", "Pclose", "Vlot", "Ptyp"]

    # -------- Tablo 5: sabitler ve pencereler --------
    CONSTANTS = [-0.1, -0.05, -0.01, 0.01, 0.05, 0.1]
    NUMS      = [20, 30, 40]

    # -------- Tablo 7: Δk (length increments) --------
    DELTA_K = {
        "Expr->Feature": 0,
        "Num->value": 0,
        "Constant->value": 0,
        "Expr->UnaryOp(Expr)": 1,
        "Expr->BinaryOp(Expr,Expr)": 2,
        "Expr->BinaryOp(Expr,Constant)": 2,
        "Expr->BinaryOp_Asym(Constant,Expr)": 2,
        "Expr->RollingOp(Expr,Num)": 2,
        "Expr->CSOp(Expr)": 2,
        "Expr->PairedRollingOp(Expr,Expr,Num)": 3,
    }

    # -------- Tablo 6: operatör kütüphanesi --------
    UNARY_OPS = {
        "Abs":  lambda x: x.abs(),
        "Sign": lambda x: np.sign(x),
        "Log":  lambda x: np.log(x.where(x > 0, np.nan)),
    }
    # Simetrik binary: Add, Mul, Greater, Less
    BINARY_OPS = {
        "Add":     lambda x, y: x + y,
        "Mul":     lambda x, y: x * y,
        "Greater": lambda x, y: np.maximum(x, y),
        "Less":    lambda x, y: np.minimum(x, y),
    }
    # Asimetrik binary: Div, Pow, Sub  (operand sırası anlamlı)
    BINARY_ASYM_OPS = {
        "Div": lambda x, y: x / (np.sign(y) * np.maximum(np.abs(y), 1e-6)),
        "Pow": lambda x, y: np.sign(x) * (np.abs(x).clip(lower=1e-6) ** y),
        "Sub": lambda x, y: x - y,
    }
    # Rolling (time-series): Tablo 6'nın tamamı
    ROLLING_OPS = {
        "Rank":  lambda x, w: _grp(x).apply(
            lambda g: g.rolling(int(w)).apply(lambda a: pd.Series(a).rank(pct=True).iloc[-1], raw=False)),
        "WMA":   lambda x, w: _grp(x).apply(
            lambda g: g.rolling(int(w)).apply(lambda a: np.average(a, weights=np.arange(1, len(a)+1)), raw=True)),
        # N3: adjust=True + min_periods=window → warm-up bias önlenir
        "EMA":   lambda x, w: _grp(x).apply(
            lambda g: g.ewm(span=int(w), adjust=True, min_periods=int(w)).mean()),
        "Ref":   lambda x, w: _grp(x).shift(int(w)),
        "Mean":  lambda x, w: _grp(x).rolling(int(w)).mean().reset_index(0, drop=True),
        "Sum":   lambda x, w: _grp(x).rolling(int(w)).sum().reset_index(0, drop=True),
        "Std":   lambda x, w: _grp(x).rolling(int(w)).std().reset_index(0, drop=True),
        "Var":   lambda x, w: _grp(x).rolling(int(w)).var().reset_index(0, drop=True),
        "Skew":  lambda x, w: _grp(x).rolling(int(w)).skew().reset_index(0, drop=True),
        "Kurt":  lambda x, w: _grp(x).rolling(int(w)).kurt().reset_index(0, drop=True),
        "Max":   lambda x, w: _grp(x).rolling(int(w)).max().reset_index(0, drop=True),
        "Min":   lambda x, w: _grp(x).rolling(int(w)).min().reset_index(0, drop=True),
        "Med":   lambda x, w: _grp(x).rolling(int(w)).median().reset_index(0, drop=True),
        "Mad":   lambda x, w: _grp(x).rolling(int(w)).apply(
            lambda a: np.mean(np.abs(a - np.mean(a))), raw=True),
        "Delta": lambda x, w: x - _grp(x).shift(int(w)),
    }
    # Paired Rolling
    PAIRED_OPS = {
        "Corr": lambda x, y, w: _paired_rolling(x, y, int(w), "corr"),
        "Cov":  lambda x, y, w: _paired_rolling(x, y, int(w), "cov"),
    }
    # Cross-Section (pencere almaz)
    CS_OPS = {
        "CSRank": lambda x: x.groupby(level="Date").rank(pct=True),
    }

    # ================================================================
    # Algoritma 2 (α-Sem-k): uzunluk-kısıtlı gramer üretimi
    # ================================================================
    def generate(self, max_K: int = 15) -> Node:
        return self._expand(k_used=0, max_K=max_K)

    def _applicable_rules(self, k_used: int, max_K: int):
        rem = max_K - k_used
        rules = [("feature", self.DELTA_K["Expr->Feature"])]
        if rem >= 1:
            rules.append(("unary", self.DELTA_K["Expr->UnaryOp(Expr)"]))
        if rem >= 2:
            rules += [
                ("binary_ee",      self.DELTA_K["Expr->BinaryOp(Expr,Expr)"]),
                ("binary_ec",      self.DELTA_K["Expr->BinaryOp(Expr,Constant)"]),
                ("binary_asym_ce", self.DELTA_K["Expr->BinaryOp_Asym(Constant,Expr)"]),
                ("rolling",        self.DELTA_K["Expr->RollingOp(Expr,Num)"]),
                ("cs_op",          self.DELTA_K["Expr->CSOp(Expr)"]),
            ]
        if rem >= 3:
            rules.append(("paired_rolling",
                          self.DELTA_K["Expr->PairedRollingOp(Expr,Expr,Num)"]))
        return rules

    def _expand(self, k_used: int, max_K: int) -> Node:
        rules = self._applicable_rules(k_used, max_K)
        # Terminal'e zorla yaklaş
        if k_used >= max_K - 1 or len(rules) == 1:
            return Node("feature", random.choice(self.FEATURES))
        rule, dk = random.choice(rules)

        if rule == "feature":
            return Node("feature", random.choice(self.FEATURES))
        if rule == "unary":
            return Node("unary", random.choice(list(self.UNARY_OPS)),
                        [self._expand(k_used + dk, max_K)])
        if rule == "binary_ee":
            return Node("binary", random.choice(list(self.BINARY_OPS)),
                        [self._expand(k_used + dk, max_K),
                         self._expand(k_used + dk, max_K)])
        if rule == "binary_ec":
            return Node("binary", random.choice(["Add", "Mul"]),
                        [self._expand(k_used + dk, max_K),
                         Node("constant", random.choice(self.CONSTANTS))])
        if rule == "binary_asym_ce":
            return Node("binary_asym", random.choice(list(self.BINARY_ASYM_OPS)),
                        [Node("constant", random.choice(self.CONSTANTS)),
                         self._expand(k_used + dk, max_K)])
        if rule == "rolling":
            return Node("rolling", random.choice(list(self.ROLLING_OPS)),
                        [self._expand(k_used + dk, max_K),
                         Node("num", random.choice(self.NUMS))])
        if rule == "cs_op":
            return Node("cs_op", "CSRank", [self._expand(k_used + dk, max_K)])
        if rule == "paired_rolling":
            return Node("paired_rolling", random.choice(list(self.PAIRED_OPS)),
                        [self._expand(k_used + dk, max_K),
                         self._expand(k_used + dk, max_K),
                         Node("num", random.choice(self.NUMS))])
        return Node("feature", random.choice(self.FEATURES))

    # ================================================================
    # Evaluation: AST → pandas Series (Ticker,Date indexed)
    # ================================================================
    def evaluate(self, node: Node, df: pd.DataFrame) -> pd.Series:
        if isinstance(df.index, pd.MultiIndex):
            temp = df
        else:
            temp = df.set_index(["Ticker", "Date"]).sort_index()
        try:
            res = self._eval(node, temp)
            if np.isscalar(res):
                res = pd.Series(float(res), index=temp.index)
            res = res.replace([np.inf, -np.inf], np.nan).fillna(0)
            # N2: [1%,99%] → [0.5%,99.5%] — daha az agresif tail kesimi
            lo, hi = res.quantile(0.005), res.quantile(0.995)
            return res.clip(lo, hi)
        except Exception:
            return pd.Series(np.zeros(len(temp)), index=temp.index)

    def _eval(self, n: Node, df: pd.DataFrame):
        if n.kind == "feature":
            return df[n.op]
        if n.kind in ("constant", "num"):
            return float(n.op)
        ch = [self._eval(c, df) for c in n.children]
        if n.kind == "unary":
            return self.UNARY_OPS[n.op](ch[0])
        if n.kind == "binary":
            x, y = ch
            if np.isscalar(x): x = pd.Series(x, index=y.index)
            if np.isscalar(y): y = pd.Series(y, index=x.index)
            return self.BINARY_OPS[n.op](x, y)
        if n.kind == "binary_asym":
            x, y = ch
            if np.isscalar(x): x = pd.Series(x, index=y.index)
            if np.isscalar(y): y = pd.Series(y, index=x.index)
            return self.BINARY_ASYM_OPS[n.op](x, y)
        if n.kind == "rolling":
            return self.ROLLING_OPS[n.op](ch[0], ch[1])
        if n.kind == "paired_rolling":
            return self.PAIRED_OPS[n.op](ch[0], ch[1], ch[2])
        if n.kind == "cs_op":
            return self.CS_OPS[n.op](ch[0])
        raise ValueError(f"bilinmeyen kind: {n.kind}")

    # ================================================================
    # AST-tabanlı benzerlik (Eq. 5):
    #   s(f_i, f_j) = max |V(S)| for S ⊆ T(f_i), S ≅ S' ⊆ T(f_j)
    # ================================================================
    def _common_subtree(self, a: Node, b: Node) -> int:
        """Kök-hizalı en büyük ortak altağaç boyutu."""
        if a.kind != b.kind or a.op != b.op or len(a.children) != len(b.children):
            return 0
        score = 1
        for ca, cb in zip(a.children, b.children):
            score += self._common_subtree(ca, cb)
        return score

    def _max_common_subtree(self, a: Node, b: Node) -> int:
        best = self._common_subtree(a, b)
        for ca in a.children:
            best = max(best, self._max_common_subtree(ca, b))
        for cb in b.children:
            best = max(best, self._max_common_subtree(a, cb))
        return best

    def similarity(self, a: Node, b: Node) -> float:
        raw = self._max_common_subtree(a, b)
        return raw / max(a.size(), b.size(), 1)

    # ================================================================
    # Crossover & Mutation (AST seviyesinde)
    # ================================================================
    def _subtree_refs(self, n: Node):
        out = [n]
        for c in n.children:
            out.extend(self._subtree_refs(c))
        return out

    def crossover(self, p1: Node, p2: Node) -> Node:
        c1, c2 = deepcopy(p1), deepcopy(p2)
        subs1 = self._subtree_refs(c1)
        subs2 = self._subtree_refs(c2)
        tgt = random.choice(subs1[1:] if len(subs1) > 1 else subs1)
        rep = deepcopy(random.choice(subs2))
        tgt.kind, tgt.op, tgt.children = rep.kind, rep.op, rep.children
        return c1

    def mutate(self, p: Node) -> Node:
        c = deepcopy(p)
        tgt = random.choice(self._subtree_refs(c))
        ops_by_kind = {
            "feature":        (self.FEATURES, None),
            "num":            (self.NUMS, None),
            "constant":       (self.CONSTANTS, None),
            "unary":          (list(self.UNARY_OPS), None),
            "binary":         (list(self.BINARY_OPS), None),
            "binary_asym":    (list(self.BINARY_ASYM_OPS), None),
            "rolling":        (list(self.ROLLING_OPS), None),
            "paired_rolling": (list(self.PAIRED_OPS), None),
            "cs_op":          (list(self.CS_OPS), None),
        }
        if tgt.kind in ops_by_kind:
            opts, _ = ops_by_kind[tgt.kind]
            tgt.op = random.choice(opts)
        return c
