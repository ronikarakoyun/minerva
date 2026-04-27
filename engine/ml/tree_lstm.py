"""
engine/tree_lstm.py — AlphaCFG §4.3 Tree-LSTM policy/value network.

- Child-Sum Tree-LSTM (Tai et al. 2015) ASR encoder.
- Policy head   : üretim kuralları üzerinde softmax (gelecek: gramer-maskeli).
- Value head    : beklenen IC ∈ [-1, 1].
"""
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..core.alpha_cfg import AlphaCFG, Node


# -----------------------------------------------------------------
# Sözcük dağarcıkları
# -----------------------------------------------------------------
def build_token_vocab(cfg: AlphaCFG) -> Dict[Tuple[str, str], int]:
    """Her Node'un (kind, op) eşlemesine sıralı tam sayı id atar."""
    v: Dict[Tuple[str, str], int] = {}

    def add(k: str, o) -> None:
        key = (k, str(o))
        if key not in v:
            v[key] = len(v)

    # Nonterminaller (ileride incremental MCTS için)
    add("nt", "Expr")
    add("nt", "Constant")
    add("nt", "Num")

    for f in cfg.FEATURES:        add("feature", f)
    for c in cfg.CONSTANTS:       add("constant", c)
    for n in cfg.NUMS:            add("num", n)
    for op in cfg.UNARY_OPS:      add("unary", op)
    for op in cfg.BINARY_OPS:     add("binary", op)
    for op in cfg.BINARY_ASYM_OPS:add("binary_asym", op)
    for op in cfg.ROLLING_OPS:    add("rolling", op)
    for op in cfg.PAIRED_OPS:     add("paired_rolling", op)
    for op in cfg.CS_OPS:         add("cs_op", op)
    return v


def build_action_vocab(cfg: AlphaCFG) -> List[Tuple[str, str]]:
    """Her üretim kuralı × somut operatör kombinasyonu için eylem listesi."""
    A: List[Tuple[str, str]] = []
    for f in cfg.FEATURES:         A.append(("R_feat", f))
    for op in cfg.UNARY_OPS:       A.append(("R_unary", op))
    for op in cfg.BINARY_OPS:      A.append(("R_binary_ee", op))
    for op in ["Add", "Mul"]:      A.append(("R_binary_ec", op))
    for op in cfg.BINARY_ASYM_OPS: A.append(("R_binary_asym_ce", op))
    for op in cfg.ROLLING_OPS:     A.append(("R_rolling", op))
    A.append(("R_cs", "CSRank"))
    for op in cfg.PAIRED_OPS:      A.append(("R_paired", op))
    for c in cfg.CONSTANTS:        A.append(("R_const_val", str(c)))
    for n in cfg.NUMS:             A.append(("R_num_val", str(n)))
    return A


# -----------------------------------------------------------------
# Child-Sum Tree-LSTM hücresi (Tai et al. 2015)
# -----------------------------------------------------------------
class ChildSumTreeLSTMCell(nn.Module):
    def __init__(self, in_dim: int, hid_dim: int):
        super().__init__()
        self.hid_dim = hid_dim
        self.W_iou = nn.Linear(in_dim, 3 * hid_dim)
        self.U_iou = nn.Linear(hid_dim, 3 * hid_dim, bias=False)
        self.W_f   = nn.Linear(in_dim, hid_dim)
        self.U_f   = nn.Linear(hid_dim, hid_dim, bias=False)

    def forward(self, x: torch.Tensor,
                ch_h: List[torch.Tensor], ch_c: List[torch.Tensor]):
        # child-sum
        if ch_h:
            h_sum = torch.stack(ch_h, 0).sum(0)
        else:
            h_sum = torch.zeros(self.hid_dim, device=x.device)
        iou = self.W_iou(x) + self.U_iou(h_sum)
        i, o, u = torch.chunk(iou, 3, dim=-1)
        i, o, u = torch.sigmoid(i), torch.sigmoid(o), torch.tanh(u)
        if ch_c:
            # her çocuk için ayrı forget gate
            f_gates = [torch.sigmoid(self.W_f(x) + self.U_f(h_k)) for h_k in ch_h]
            c = i * u + sum(fk * ck for fk, ck in zip(f_gates, ch_c))
        else:
            c = i * u
        h = o * torch.tanh(c)
        return h, c


class TreeLSTMEncoder(nn.Module):
    """Post-order dolaşımla ASR'yi sabit boyutlu vektöre gömen encoder."""
    def __init__(self, vocab_size: int, emb_dim: int = 32, hid_dim: int = 64):
        super().__init__()
        self.emb  = nn.Embedding(vocab_size, emb_dim)
        self.cell = ChildSumTreeLSTMCell(emb_dim, hid_dim)
        self.hid_dim = hid_dim

    def _encode(self, node: Node, vocab: Dict[Tuple[str, str], int]):
        ch_outs = [self._encode(c, vocab) for c in node.children]
        ch_h = [o[0] for o in ch_outs]
        ch_c = [o[1] for o in ch_outs]
        key  = (node.kind, str(node.op))
        idx  = vocab.get(key, vocab.get(("nt", "Expr"), 0))
        x    = self.emb(torch.tensor(idx, device=self.emb.weight.device))
        return self.cell(x, ch_h, ch_c)

    def forward(self, node: Node, vocab: Dict[Tuple[str, str], int]) -> torch.Tensor:
        h, _ = self._encode(node, vocab)
        return h   # (hid_dim,)


# -----------------------------------------------------------------
# Policy + Value Network
# -----------------------------------------------------------------
class PolicyValueNet(nn.Module):
    def __init__(self, token_vocab_size: int, action_size: int,
                 emb_dim: int = 32, hid_dim: int = 64):
        super().__init__()
        self.encoder     = TreeLSTMEncoder(token_vocab_size, emb_dim, hid_dim)
        self.policy_head = nn.Linear(hid_dim, action_size)
        self.value_head  = nn.Sequential(
            nn.Linear(hid_dim, hid_dim // 2),
            nn.Tanh(),
            nn.Linear(hid_dim // 2, 1),
            nn.Tanh(),   # [-1, 1]
        )

    def forward(self, node: Node, vocab):
        h = self.encoder(node, vocab)
        return self.policy_head(h), self.value_head(h).squeeze(-1)

    # ------- Inference helper'ları (MCTS bağlamak için) -------
    @torch.no_grad()
    def predict_value(self, node: Node, vocab) -> float:
        self.eval()
        _, v = self.forward(node, vocab)
        return float(v.item())

    @torch.no_grad()
    def predict_policy(self, node: Node, vocab) -> torch.Tensor:
        """Tüm eylemler geçerli — masking yok."""
        self.eval()
        logits, _ = self.forward(node, vocab)
        return F.softmax(logits, dim=-1)

    @torch.no_grad()
    def predict_policy_masked(
        self,
        node: Node,
        vocab,
        legal_mask: "torch.Tensor | None" = None,
    ) -> torch.Tensor:
        """
        Grammar-maskeli policy — 6.1 fix.

        Geçersiz eylemler (grammar kısıtına uymayan) -1e9 logit alır →
        softmax sonrası ≈ 0. Bu sayede MCTS geçersiz eylemleri seçmez ve
        eğitim sinyali gürültüsü azalır.

        Parameters
        ----------
        legal_mask : bool tensor (action_size,) — True = geçerli eylem.
                     None → masking yok (predict_policy ile eşdeğer).
                     CFG'den üretilir: cfg.legal_children(parent_symbol, position).

        Örnek kullanım:
            actions = brain["actions"]
            mask    = torch.zeros(len(actions), dtype=torch.bool)
            # parent_symbol için geçerli eylemleri işaretle
            for i, (rule, op) in enumerate(actions):
                if rule == allowed_rule:
                    mask[i] = True
            probs = net.predict_policy_masked(partial_tree, vocab, legal_mask=mask)
        """
        self.eval()
        logits, _ = self.forward(node, vocab)
        if legal_mask is not None:
            logits = logits.masked_fill(
                ~legal_mask.to(logits.device), -1e9
            )
        return F.softmax(logits, dim=-1)
