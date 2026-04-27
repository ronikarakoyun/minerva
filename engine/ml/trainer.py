"""
engine/trainer.py — Tree-LSTM eğitim döngüsü.

- Value regression (MSE)  : replay buffer'daki ölçülen IC'ler hedef.
- Policy imitation (CE)   : opsiyonel — MCTS visit distribution hedef.
"""
import os
from typing import Optional

import torch
import torch.nn.functional as F

from .replay_buffer import ReplayBuffer
from .tree_lstm import PolicyValueNet


class TreeLSTMTrainer:
    def __init__(self, net: PolicyValueNet, vocab: dict,
                 lr: float = 1e-3, device: str = "cpu",
                 ic_scale: float = 10.0):
        """
        ic_scale: IC tipik olarak 0.01-0.15 aralığında; tanh çıkışıyla uyumlu
        olması için hedef değer ic * ic_scale ile çarpılıp [-1,1]'e clamp'lenir.
        """
        self.net = net.to(device)
        self.vocab = vocab
        self.device = device
        self.ic_scale = ic_scale
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.history: list[dict] = []

    # -----------------------------------------------------------
    def train_step(self, buffer: ReplayBuffer, batch_size: int = 32,
                   use_policy: bool = False) -> Optional[dict]:
        if len(buffer) < max(2, batch_size // 4):
            return None
        batch = buffer.sample(batch_size)
        self.net.train()

        total, val_sum, pol_sum = 0.0, 0.0, 0.0
        n = 0

        for tree, ic, visit in batch:
            logits, v_pred = self.net(tree, self.vocab)

            v_target = torch.tensor(
                max(-1.0, min(1.0, ic * self.ic_scale)),
                device=self.device, dtype=torch.float32,
            )
            val_loss = F.mse_loss(v_pred, v_target)
            loss = val_loss

            if use_policy and visit is not None and sum(visit) > 0:
                p = torch.tensor(visit, device=self.device, dtype=torch.float32)
                p = p / p.sum()
                log_q = F.log_softmax(logits, dim=-1)
                pol_loss = -(p * log_q).sum()
                loss = loss + pol_loss
                pol_sum += float(pol_loss.item())

            self.opt.zero_grad()
            loss.backward()
            self.opt.step()

            total += float(loss.item())
            val_sum += float(val_loss.item())
            n += 1

        entry = {
            "total":  total / n,
            "value":  val_sum / n,
            "policy": pol_sum / n if use_policy else 0.0,
            "n":      n,
            "buffer": len(buffer),
        }
        self.history.append(entry)
        return entry

    def train_epochs(self, buffer: ReplayBuffer, epochs: int = 5,
                     batch_size: int = 32, use_policy: bool = False,
                     progress_cb=None) -> list[dict]:
        hist = []
        for e in range(epochs):
            entry = self.train_step(buffer, batch_size, use_policy)
            if entry is None:
                break
            hist.append(entry)
            if progress_cb:
                progress_cb(e + 1, epochs, entry)
        return hist

    # -----------------------------------------------------------
    def save(self, path: str = "data/tree_lstm.pt") -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "state_dict": self.net.state_dict(),
            "vocab_size": self.net.encoder.emb.num_embeddings,
        }, path)

    def load(self, path: str = "data/tree_lstm.pt") -> "TreeLSTMTrainer":
        if os.path.exists(path):
            try:
                ck = torch.load(path, map_location=self.device, weights_only=False)
                if ck.get("vocab_size") == self.net.encoder.emb.num_embeddings:
                    self.net.load_state_dict(ck["state_dict"])
            except Exception:
                pass
        return self
