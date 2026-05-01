"""
engine/mcts.py — AlphaCFG §4.2 uyumlu grammar-aware MCTS + PUCT.

Makalenin Tree-LSTM+policy/value networkleri harici bir LLM/RL altyapısı
gerektiriyor; bu modül, aynı TSL-MDP formülasyonunda klasik MCTS (rollout
tabanlı value estimate, uniform prior) sağlar — neural bileşenler için
`policy_fn` / `value_fn` enjekte edilebilir.
"""
import math
import random
from dataclasses import dataclass, field
from typing import Callable, List, Optional

from ..core.alpha_cfg import AlphaCFG, Node


@dataclass
class _MCTSNode:
    state: Node                 # kısmi ASR (partial expression)
    k_used: int                 # mevcut uzunluk maliyeti
    parent: Optional["_MCTSNode"] = None
    action: Optional[str] = None
    children: List["_MCTSNode"] = field(default_factory=list)
    N: int = 0                  # visit count
    Q: float = 0.0              # mean value
    P: float = 1.0              # prior


class GrammarMCTS:
    def __init__(
        self,
        cfg: AlphaCFG,
        max_K: int = 15,
        c_puct: float = 1.4,
        rollouts: int = 16,
        value_fn: Optional[Callable[[Node], float]] = None,
        policy_fn: Optional[Callable[[Node, List[str]], List[float]]] = None,
        subtree_prior: Optional[dict] = None,
    ):
        """
        Parameters
        ----------
        value_fn      : Tree-LSTM value tahmini. Sağlanırsa AlphaZero-stili
                        rollout (rastgele tamamlama yok, direkt value estimate).
        subtree_prior : {str(Node) → frequency} dict. Katalogdaki iyi formüllerin
                        alt-ağaçlarının sıklığı → PUCT bonus olarak eklenir.
                        Warm-start prior (6.3) için kullanılır.
                        app.py'de: {str(t): 0.1 for t in session_state.trees.values()}
        """
        self.cfg = cfg
        self.max_K = max_K
        self.c_puct = c_puct
        self.rollouts = rollouts
        self.value_fn = value_fn
        self.policy_fn = policy_fn
        # Subtree prior: bilinen iyi alt-ağaçlara bonus — sıfırdan öğrenmeyi azaltır
        self.subtree_prior: dict = subtree_prior or {}

    # N11: Softmax temperature normalize — P + prior_bonus toplamı [0,2] aşar
    _PUCT_TEMPERATURE: float = 1.0  # softmax sıcaklığı; düşürdükçe keskin seçim

    def _compute_softmax_priors(self, children: list) -> dict:
        """
        N11: Tüm çocukların P + prior_bonus değerlerini softmax ile normalize et.
        Sonuç: her çocuk için [0,1] aralığında olasılık → toplam = 1.0.
        """
        import math as _math
        raw: list[float] = []
        for c in children:
            rb = self.subtree_prior.get(str(c.state), 0.0)
            max_p = max(self.subtree_prior.values(), default=1.0) or 1.0
            raw.append(c.P + rb / max_p)
        # Softmax ile [0,1] aralığına normalize
        max_raw = max(raw) if raw else 0.0
        exps = [_math.exp((r - max_raw) / max(self._PUCT_TEMPERATURE, 1e-6)) for r in raw]
        total = sum(exps) or 1.0
        return {id(c): e / total for c, e in zip(children, exps)}

    # PUCT (Alg. 3):  a* = argmax Q + c · √(b/b_ref) · P_norm · √ΣN / (1+N(s,a))
    def _puct(
        self,
        parent: _MCTSNode,
        child: _MCTSNode,
        b_ref: int,
        softmax_priors: dict | None = None,
    ) -> float:
        b = max(len(parent.children), 1)
        bal = math.sqrt(b / max(b_ref, 1))
        total_N = sum(c.N for c in parent.children)
        # N11: softmax-normalized prior — toplam kesinlikle 1.0
        p_norm = (softmax_priors or {}).get(id(child), child.P)
        u = self.c_puct * bal * p_norm * math.sqrt(total_N + 1) / (1 + child.N)
        return child.Q + u

    def _is_terminal(self, node: Node) -> bool:
        # Tüm yapraklar feature/constant/num (nonterminal kalmadı)
        if node.kind in ("feature", "constant", "num"):
            return True
        return all(self._is_terminal(c) for c in node.children)

    def _random_completion(self, partial: Node, k_used: int) -> Node:
        """Kısmi ifadeyi random rollout ile tamamlar."""
        # Bu basit tasarımda `partial` zaten tamamlanmış kabul ediliyor;
        # kısmi genişletme için cfg.generate tekrar çağrılır.
        return self.cfg.generate(self.max_K - k_used)

    def _simulate_value(self, node: Node) -> float:
        if self.value_fn is not None:
            return float(self.value_fn(node))
        # N13: `unique_ops/10` heuristic kaldırıldı — fonksiyonel ve çeşitli
        # formülleri cezalandırıyordu. value_fn yoksa sabit 0.5 (tarafsız prior).
        return 0.5

    def search(self, iterations: int = 200) -> Node:
        """Alg. 3: I iterasyon MCTS; kök durumdan en sık ziyaret edilen yolu döndürür."""
        root_state = self.cfg.generate(self.max_K)
        root = _MCTSNode(state=root_state, k_used=root_state.size())

        for _ in range(iterations):
            leaf = self._select(root)
            value = self._rollout(leaf)
            self._backprop(leaf, value)

        # Görüş sayısı en yüksek kök-çocuğun durumunu döndür
        if not root.children:
            return root.state
        best = max(root.children, key=lambda c: c.N)
        return best.state

    def _select(self, node: _MCTSNode) -> _MCTSNode:
        b_ref = 8
        cur = node
        while cur.children:
            # N11: Softmax priorları bir kez hesapla, tüm çocuklara geç
            sp = self._compute_softmax_priors(cur.children)
            cur = max(cur.children, key=lambda c: self._puct(cur, c, b_ref, sp))
        # Expansion: mutasyon/crossover benzeri üretim
        for _ in range(4):
            candidate = self.cfg.mutate(cur.state) if random.random() < 0.5 \
                        else self.cfg.generate(self.max_K)
            cur.children.append(
                _MCTSNode(state=candidate, k_used=candidate.size(),
                          parent=cur, action="expand"))
        return random.choice(cur.children)

    def _rollout(self, leaf: _MCTSNode) -> float:
        """
        AlphaZero-stili rollout — 6.2 fix.

        value_fn (Tree-LSTM) sağlanmışsa: doğrudan value estimate kullan,
        rastgele tamamlama yok. Bu, eğitim sinyalini temizler ve arama
        kalitesini artırır.

        value_fn yoksa: heuristik (operatör çeşitliliği).
        """
        if self.value_fn is not None:
            # AlphaZero: terminal olmayan node'da da value_fn kullan (rollout yok)
            return float(self.value_fn(leaf.state))
        # Fallback: naive structural heuristic (operatör çeşitliliği)
        vals = []
        for _ in range(max(1, self.rollouts // 4)):
            vals.append(self._simulate_value(leaf.state))
        return float(sum(vals) / len(vals))

    def _backprop(self, leaf: _MCTSNode, value: float):
        cur = leaf
        while cur is not None:
            cur.N += 1
            cur.Q += (value - cur.Q) / cur.N
            cur = cur.parent
