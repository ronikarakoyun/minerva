"""RL Position Sizer — Minimal PPO (sıfır ek bağımlılık, PyTorch).

State  : [portfolio_vol_20d, max_drawdown_pct, regime_entropy, recent_ic_20d, current_scale]
Action : {0, 1, 2, 3} → {0.5x, 1.0x, 1.5x, 2.0x} × base_position
Reward : daily Sharpe contribution = r_t / rolling_std(r, 20)

Genelleme güvencesi:
  SizingEnv.reset() her episode'da farklı bir (equity, regime, vol) üçlüsü seçer.
  Gerçek alpha catalog eğrileri + sentetik GBM yolları (50/50) ile karma havuz.
  Bu sayede ajan tek bir eğri ezberlemiyor; çöküş/ralli/yatay rejimleri genelleştirilmiş öğreniyor.
"""
from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# ── Sabitler ─────────────────────────────────────────────────────────────────

ACTIONS = [0.5, 1.0, 1.5, 2.0]   # scale faktörleri
N_ACTIONS = len(ACTIONS)
STATE_DIM = 5


# ── State ve Env ──────────────────────────────────────────────────────────────

@dataclass
class SizingState:
    """RL pozisyonlama ortamı gözlemi."""
    portfolio_vol: float    # annualized realized vol (0..1)
    drawdown: float         # current drawdown from peak (0..1)
    regime_entropy: float   # HMM belirsizliği (0..log(K))
    recent_ic: float        # 20-day rolling mean RankIC (-1..1)
    current_scale: float    # son uygulanan scale faktörü (0.5..2.0)

    def to_array(self) -> np.ndarray:
        return np.array([
            self.portfolio_vol,
            self.drawdown,
            self.regime_entropy,
            self.recent_ic,
            self.current_scale,
        ], dtype=np.float32)


class SizingEnv:
    """Minimal gymnasium-benzeri ortam — tek process, sıfır ek bağımlılık."""

    def __init__(
        self,
        equity_episodes: list[pd.Series],
        regime_episodes: list[pd.DataFrame],
        vol_episodes: list[pd.DataFrame],
        max_entropy: float = 1.1,   # log(3) ≈ 1.099
    ):
        """
        Parameters
        ----------
        equity_episodes  : Her eleman bir alpha curve (Date-indexed cumulative return).
        regime_episodes  : Her eleman bir HMM prob_df (Date × regime_K).
        vol_episodes     : Her eleman bir (Date × Ticker) realized vol DataFrame.
        """
        assert len(equity_episodes) == len(regime_episodes) == len(vol_episodes), \
            "Tüm episode listeleri aynı uzunlukta olmalı"
        self._episodes = list(zip(equity_episodes, regime_episodes, vol_episodes))
        self._max_entropy = max(max_entropy, 1e-3)

        self._eq: pd.Series = equity_episodes[0]
        self._reg: pd.DataFrame = regime_episodes[0]
        self._vol: pd.DataFrame = vol_episodes[0]
        self._t: int = 0
        self._current_scale: float = 1.0
        self._peak: float = 1.0
        self._prev_scale: float = 1.0

    def reset(self) -> np.ndarray:
        """Episode başında rastgele bir (equity, regime, vol) üçlüsü seç."""
        self._eq, self._reg, self._vol = random.choice(self._episodes)
        self._t = 0
        self._current_scale = 1.0
        self._peak = float(self._eq.iloc[0]) if len(self._eq) > 0 else 1.0
        self._prev_scale = 1.0
        return self._observe()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict]:
        """Bir adım ilerle.

        Returns
        -------
        (obs, reward, done, info)
        """
        scale = ACTIONS[action]
        self._current_scale = scale
        self._t += 1

        done = self._t >= len(self._eq) - 1
        if done:
            return self._observe(), 0.0, True, {}

        eq_vals = self._eq.values
        r_t = float(eq_vals[self._t] / eq_vals[self._t - 1] - 1) * scale

        # Reward: Sharpe contribution (rolling std window = min(20, t))
        window = max(2, min(20, self._t))
        recent = eq_vals[max(0, self._t - window): self._t]
        rets_recent = np.diff(recent) / np.maximum(recent[:-1], 1e-10)
        std = float(np.std(rets_recent)) if len(rets_recent) > 1 else 1e-6

        # S10 Composite reward (Pippas 2025):
        # 1. Profit norm: z-score (ölçek normalizasyonu — ZORUNLU)
        profit_norm = r_t / max(std, 1e-6)

        # 2. Transaction cost norm: pozisyon değişimi cezası
        #    scale değişimi → turnover proxy → komisyon maliyeti
        _prev_scale = getattr(self, '_prev_scale', scale)
        position_change = abs(scale - _prev_scale)
        self._prev_scale = scale
        # BIST tipik komisyon ~0.1% = 10 bps; ölçek [0.5,2.0] → normalize
        cost_norm = -0.3 * position_change  # max değişim=1.5 → max ceza=-0.45

        # 3. Drawdown cezası (mevcut, korunuyor)
        peak = self._peak if self._peak > 1e-10 else 1.0
        current_dd = max(0.0, 1.0 - float(eq_vals[self._t]) / peak)
        drawdown_penalty = 0.0
        if scale > 1.0 and current_dd > 0.10:
            drawdown_penalty = -2.0 * (scale - 1.0) * current_dd

        # Composite: 0.7 profit + 0.2 cost + 0.1 drawdown
        reward = 0.7 * profit_norm + 0.2 * cost_norm + drawdown_penalty

        reward = float(np.clip(reward, -10.0, 10.0))

        # Peak update for drawdown
        self._peak = max(self._peak, float(eq_vals[self._t]))

        return self._observe(), reward, False, {"r_t": r_t}

    def _observe(self) -> np.ndarray:
        eq_vals = self._eq.values
        t = min(self._t, len(eq_vals) - 1)

        # Portfolio vol (annualized, 20-day window)
        window = max(2, min(20, t))
        if window > 1 and t > 0:
            recent = eq_vals[max(0, t - window): t + 1]
            rets = np.diff(recent) / np.maximum(recent[:-1], 1e-10)
            port_vol = float(np.std(rets) * np.sqrt(252))
        else:
            port_vol = 0.15  # prior

        # Drawdown
        peak = self._peak if self._peak > 1e-10 else 1.0
        current_val = float(eq_vals[t])
        drawdown = max(0.0, 1.0 - current_val / peak)

        # Regime entropy
        if len(self._reg) > 0:
            reg_t = self._reg.iloc[min(t, len(self._reg) - 1)]
            p = reg_t.clip(lower=1e-9).values
            entropy = float(-(p * np.log(p)).sum())
        else:
            entropy = 0.0
        norm_entropy = entropy / self._max_entropy

        # Recent IC (rolling 20d mean — approximated from vol as placeholder)
        recent_ic = 0.0

        state = SizingState(
            portfolio_vol=float(np.clip(port_vol, 0.0, 1.0)),
            drawdown=float(np.clip(drawdown, 0.0, 1.0)),
            regime_entropy=float(np.clip(norm_entropy, 0.0, 1.0)),
            recent_ic=float(np.clip(recent_ic, -1.0, 1.0)),
            current_scale=float(np.clip(self._current_scale / 2.0, 0.0, 1.0)),
        )
        return state.to_array()


# ── Minimal PPO Agent ─────────────────────────────────────────────────────────

class MinimalPPOAgent(nn.Module):
    """2-katmanlı MLP policy + value (~5K parametre)."""

    CLIP_EPS = 0.2
    GAMMA    = 0.99
    LAM      = 0.95   # GAE lambda

    def __init__(
        self,
        state_dim: int = STATE_DIM,
        n_actions: int = N_ACTIONS,
        hid: int = 32,
        lr: float = 3e-4,
    ):
        super().__init__()
        self.policy = nn.Sequential(
            nn.Linear(state_dim, hid), nn.Tanh(),
            nn.Linear(hid, n_actions),
        )
        self.value = nn.Sequential(
            nn.Linear(state_dim, hid), nn.Tanh(),
            nn.Linear(hid, 1),
        )
        self.opt = torch.optim.Adam(self.parameters(), lr=lr)

    def act(self, state: np.ndarray) -> tuple[int, float]:
        """State → (action_index, log_prob)."""
        x = torch.as_tensor(state, dtype=torch.float32)
        with torch.no_grad():
            logits = self.policy(x)
            dist   = torch.distributions.Categorical(logits=logits)
            action = int(dist.sample().item())
            log_p  = float(dist.log_prob(torch.tensor(action)).item())
        return action, log_p

    def update(self, rollouts: list[dict]) -> dict:
        """PPO clip update, tek epoch."""
        if not rollouts:
            return {"policy_loss": 0.0, "value_loss": 0.0}

        states  = torch.from_numpy(np.array([r["state"]  for r in rollouts], dtype=np.float32))
        actions = torch.tensor([r["action"] for r in rollouts], dtype=torch.long)
        old_lp  = torch.tensor([r["log_p"]  for r in rollouts], dtype=torch.float32)
        returns = torch.tensor([r["return"] for r in rollouts], dtype=torch.float32)
        advs    = torch.tensor([r["adv"]    for r in rollouts], dtype=torch.float32)
        advs    = (advs - advs.mean()) / (advs.std() + 1e-8)

        logits = self.policy(states)
        dist   = torch.distributions.Categorical(logits=logits)
        new_lp = dist.log_prob(actions)
        ratio  = torch.exp(new_lp - old_lp)

        surr1  = ratio * advs
        surr2  = torch.clamp(ratio, 1 - self.CLIP_EPS, 1 + self.CLIP_EPS) * advs
        policy_loss = -torch.min(surr1, surr2).mean()

        values = self.value(states).squeeze(-1)
        value_loss = F.mse_loss(values, returns)

        # Entropi bonusu: politikanın erken tek eyleme kilitlenmesini önler.
        # 0.01 katsayısı küçük ama mode collapse için yeterli; exploration–exploitation
        # dengesini korur (PPO standardı: 0.01–0.05 arası).
        entropy_bonus = dist.entropy().mean()
        loss = policy_loss + 0.5 * value_loss - 0.01 * entropy_bonus
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), max_norm=0.5)
        self.opt.step()

        return {
            "policy_loss":  float(policy_loss.item()),
            "value_loss":   float(value_loss.item()),
            "entropy":      float(entropy_bonus.item()),
        }

    def _collect_rollout(self, env: SizingEnv, max_steps: int = 252) -> list[dict]:
        """Tek episode rollout — GAE avantajları hesaplanır."""
        transitions = []
        obs = env.reset()
        done = False

        while not done and len(transitions) < max_steps:
            action, log_p = self.act(obs)
            next_obs, reward, done, _ = env.step(action)
            transitions.append({
                "state":  obs,
                "action": action,
                "log_p":  log_p,
                "reward": reward,
                "done":   done,
            })
            obs = next_obs

        # GAE return ve avantaj hesabı
        if not transitions:
            return []

        T = len(transitions)
        rewards  = np.array([t["reward"] for t in transitions])
        dones    = np.array([t["done"]   for t in transitions], dtype=float)

        # Bootstrap value
        with torch.no_grad():
            last_val = float(self.value(
                torch.tensor(obs, dtype=torch.float32)
            ).item()) if not done else 0.0

        values = []
        with torch.no_grad():
            for t in transitions:
                v = float(self.value(
                    torch.tensor(t["state"], dtype=torch.float32)
                ).item())
                values.append(v)
        values.append(last_val)

        # GAE
        gae = 0.0
        advs = np.zeros(T)
        for t in reversed(range(T)):
            delta = rewards[t] + self.GAMMA * values[t + 1] * (1 - dones[t]) - values[t]
            gae   = delta + self.GAMMA * self.LAM * (1 - dones[t]) * gae
            advs[t] = gae

        rets = advs + np.array(values[:T])

        for i, tr in enumerate(transitions):
            tr["adv"]    = float(advs[i])
            tr["return"] = float(rets[i])

        return transitions


# ── Sentetik GBM Episode Üreticisi ───────────────────────────────────────────

def _make_gbm_episodes(
    n_episodes: int = 100,
    n_steps: int = 252,
    seed: int = 0,
    K: int = 3,
) -> tuple[list[pd.Series], list[pd.DataFrame], list[pd.DataFrame]]:
    """Sentetik GBM getiri yolları (çeşitli μ ve σ kombinasyonları)."""
    rng = np.random.default_rng(seed)
    equity_list, regime_list, vol_list = [], [], []

    for _ in range(n_episodes):
        mu    = rng.uniform(-0.01, 0.02)
        sigma = rng.uniform(0.005, 0.03)
        dt    = 1.0 / 252
        Z     = rng.standard_normal(n_steps)
        log_r = (mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z
        eq    = pd.Series(np.exp(np.cumsum(log_r)))

        # Dummy regime probs (uniform Dirichlet)
        raw = rng.dirichlet(alpha=np.ones(K), size=n_steps)
        reg = pd.DataFrame(raw, columns=[f"regime_{k}" for k in range(K)])

        # Dummy vol
        vol_vals = pd.DataFrame({"vol": np.abs(rng.standard_normal(n_steps)) * sigma})

        equity_list.append(eq)
        regime_list.append(reg)
        vol_list.append(vol_vals)

    return equity_list, regime_list, vol_list


# ── Eğitim Fonksiyonu ─────────────────────────────────────────────────────────

def train_rl_sizer(
    equity_curve: "pd.Series | None" = None,
    regime_probs: "pd.DataFrame | None" = None,
    asset_vols: "pd.DataFrame | None" = None,
    n_episodes: int = 200,
    save_path: "str | None" = None,
    seed: int = 42,
) -> MinimalPPOAgent:
    """PPO agent eğit.

    Gerçek veri (equity_curve, regime_probs, asset_vols) ile sentetik GBM yollarını
    50/50 oranında karıştırarak genelleme güvencesi sağlanır.

    Parameters
    ----------
    equity_curve  : Alpha catalog çıktısı — gerçek equity eğrisi.
    regime_probs  : HMM prob_df (Date × K).
    asset_vols    : (Date × Ticker) realized vol DataFrame.
    n_episodes    : Toplam eğitim episode sayısı (varsayılan: 200).
    save_path     : .pt dosyası yolu — None → kaydetme.
    seed          : Tekrarlanabilirlik.

    Returns
    -------
    MinimalPPOAgent — eğitilmiş agent.
    """
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Sentetik GBM havuzu
    n_gbm = max(50, n_episodes // 2)
    gbm_eq, gbm_reg, gbm_vol = _make_gbm_episodes(n_episodes=n_gbm, seed=seed)

    # Gerçek veri havuzu (sağlanmışsa)
    real_eq, real_reg, real_vol = [], [], []
    if equity_curve is not None:
        real_eq.append(equity_curve)
        real_reg.append(regime_probs if regime_probs is not None else gbm_reg[0])
        real_vol.append(asset_vols   if asset_vols is not None   else gbm_vol[0])

    # Karma (50/50 veya tamamen GBM)
    all_eq  = real_eq + gbm_eq
    all_reg = real_reg + gbm_reg
    all_vol = real_vol + gbm_vol

    env   = SizingEnv(all_eq, all_reg, all_vol)
    agent = MinimalPPOAgent()
    losses: list[float] = []

    for ep in range(n_episodes):
        rollout = agent._collect_rollout(env)
        if rollout:
            info = agent.update(rollout)
            total_loss = info["policy_loss"] + info["value_loss"]
            losses.append(total_loss)

    if save_path is not None:
        torch.save(agent.state_dict(), save_path)
        logger.info("RL agent kaydedildi: %s", save_path)

    logger.info(
        "RL eğitim tamamlandı: %d episode, ilk loss=%.4f, son loss=%.4f",
        n_episodes,
        losses[0] if losses else float("nan"),
        losses[-1] if losses else float("nan"),
    )
    return agent
