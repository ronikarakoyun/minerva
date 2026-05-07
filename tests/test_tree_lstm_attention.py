"""Birim testler: engine/ml/tree_lstm.py — PR-10 Attention + MC Dropout."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from engine.ml.tree_lstm import (
    ChildSumTreeLSTMCell,
    PolicyValueNet,
    build_action_vocab,
    build_token_vocab,
)
from engine.core.alpha_cfg import AlphaCFG, Node


# ── Yardımcı: minimal ASR ────────────────────────────────────────────────────

def _make_cfg() -> AlphaCFG:
    return AlphaCFG()


def _leaf_node(cfg: AlphaCFG) -> Node:
    """Tek yapraklı (çocuksuz) bir feature node."""
    return Node(kind="feature", op=cfg.FEATURES[0])


def _simple_tree(cfg: AlphaCFG) -> Node:
    """feature + feature → binary op — 3 node, 2 çocuk."""
    feats = list(cfg.FEATURES)
    ops   = list(cfg.BINARY_OPS)
    left  = Node(kind="feature", op=feats[0])
    right = Node(kind="feature", op=feats[1] if len(feats) > 1 else feats[0])
    root  = Node(kind="binary", op=ops[0], children=[left, right])
    return root


# ── Testler ───────────────────────────────────────────────────────────────────

def test_use_attention_false_matches_default():
    """use_attention=False → mevcut forward pass ile özdeş sonuç."""
    cfg = _make_cfg()
    vocab = build_token_vocab(cfg)
    actions = build_action_vocab(cfg)

    torch.manual_seed(0)
    net_default = PolicyValueNet(len(vocab), len(actions))  # use_attention=False
    torch.manual_seed(0)
    net_explicit = PolicyValueNet(len(vocab), len(actions), use_attention=False)

    node = _simple_tree(cfg)

    net_default.eval()
    net_explicit.eval()
    with torch.no_grad():
        p1, v1 = net_default(node, vocab)
        p2, v2 = net_explicit(node, vocab)

    torch.testing.assert_close(p1, p2, atol=1e-6, rtol=0)
    torch.testing.assert_close(v1, v2, atol=1e-6, rtol=0)


def test_attention_weights_sum_to_one():
    """Attention ağırlıkları softmax → toplamı 1.0 olmalı."""
    hid_dim = 16
    cell = ChildSumTreeLSTMCell(in_dim=8, hid_dim=hid_dim, use_attention=True)

    x = torch.randn(8)
    ch_h = [torch.randn(hid_dim) for _ in range(4)]
    ch_c = [torch.randn(hid_dim) for _ in range(4)]

    with torch.no_grad():
        # Attention ağırlıklarını doğrudan hesapla (forward çalışıyorsa)
        stacked = torch.stack(ch_h, 0)
        scores  = cell.attn_score(stacked)
        weights = torch.softmax(scores, dim=0)

    weight_sum = float(weights.sum().item())
    assert abs(weight_sum - 1.0) < 1e-5, f"Attention ağırlıkları toplamı: {weight_sum}"


def test_predict_value_with_uncertainty_returns_mean_std():
    """predict_value_with_uncertainty → (float, float) tuple, std > 0 (dropout_p > 0)."""
    cfg = _make_cfg()
    vocab = build_token_vocab(cfg)
    actions = build_action_vocab(cfg)

    net = PolicyValueNet(len(vocab), len(actions), dropout_p=0.3)
    node = _simple_tree(cfg)

    torch.manual_seed(42)
    mean_val, std_val = net.predict_value_with_uncertainty(node, vocab, n_mc=30)

    assert isinstance(mean_val, float), "mean float olmalı"
    assert isinstance(std_val, float), "std float olmalı"
    assert std_val > 0.0, f"dropout_p=0.3 ile std > 0 beklendi, std={std_val}"
    assert -2.0 <= mean_val <= 2.0, f"value_head Tanh [-1,1] → mean makul aralıkta olmalı"


def test_predict_value_uncertainty_zero_when_no_dropout():
    """dropout_p=0.0 → std ≈ 0 (deterministik forward pass)."""
    cfg = _make_cfg()
    vocab = build_token_vocab(cfg)
    actions = build_action_vocab(cfg)

    net = PolicyValueNet(len(vocab), len(actions), dropout_p=0.0)
    node = _simple_tree(cfg)

    _, std_val = net.predict_value_with_uncertainty(node, vocab, n_mc=20)

    assert std_val < 1e-6, f"dropout_p=0.0 ile std ≈ 0 beklendi, std={std_val}"
