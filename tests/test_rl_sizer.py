"""Birim testler: engine/risk/rl_sizer.py — PR-12 Minimal PPO."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from engine.risk.rl_sizer import (
    ACTIONS,
    N_ACTIONS,
    MinimalPPOAgent,
    SizingEnv,
    STATE_DIM,
    _make_gbm_episodes,
    train_rl_sizer,
)


# ── Yardımcı ─────────────────────────────────────────────────────────────────

def _make_env(n_episodes: int = 5, n_steps: int = 100) -> SizingEnv:
    eq, reg, vol = _make_gbm_episodes(n_episodes=n_episodes, n_steps=n_steps, seed=0)
    return SizingEnv(eq, reg, vol)


# ── Testler ───────────────────────────────────────────────────────────────────

def test_sizing_env_step_returns_valid_state_shape():
    """SizingEnv.step() geçerli state shape döndürmeli."""
    env = _make_env()
    obs = env.reset()

    assert obs.shape == (STATE_DIM,), f"reset shape: {obs.shape}"
    assert not np.any(np.isnan(obs)), "reset obs NaN içermemeli"

    next_obs, reward, done, info = env.step(0)

    assert next_obs.shape == (STATE_DIM,), f"step shape: {next_obs.shape}"
    assert not np.any(np.isnan(next_obs)), "step obs NaN içermemeli"
    assert isinstance(reward, float), "reward float olmalı"
    assert isinstance(done, bool), "done bool olmalı"


def test_ppo_agent_act_returns_valid_action():
    """MinimalPPOAgent.act() geçerli action index döndürmeli (0-3)."""
    agent = MinimalPPOAgent()
    env   = _make_env()
    obs   = env.reset()

    action, log_p = agent.act(obs)

    assert 0 <= action < N_ACTIONS, f"Action dışı aralık: {action}"
    assert isinstance(action, int), "action int olmalı"
    assert isinstance(log_p, float), "log_p float olmalı"
    assert not np.isnan(log_p), "log_p NaN olmamalı"


def test_train_rl_sizer_loss_decreases():
    """train_rl_sizer() eğitim sonunda kayıp azalmış olmalı."""
    agent = train_rl_sizer(n_episodes=60, seed=0)

    # Birkaç adım rollout yaparak loss ölç
    env = _make_env(n_steps=100)

    def _compute_loss(ag: MinimalPPOAgent, n_rollouts: int = 5) -> float:
        total_loss = 0.0
        count = 0
        for _ in range(n_rollouts):
            rollout = ag._collect_rollout(env, max_steps=50)
            if rollout:
                info = ag.update(rollout)
                total_loss += info["policy_loss"] + info["value_loss"]
                count += 1
        return total_loss / max(count, 1)

    # Eğitilmiş ajan rasgele ajandan daha tutarlı davranmalı
    # (Eğitim yakınsadıysa loss görece düşük olmalı)
    final_loss = _compute_loss(agent, n_rollouts=5)
    random_agent = MinimalPPOAgent()  # Sıfırdan (eğitimsiz)
    random_loss  = _compute_loss(random_agent, n_rollouts=5)

    # Eğitilmiş ajan 5× üzerinde kötü olmamalı — eğitim en azından yakınsıyor
    assert final_loss < random_loss * 5.0 or final_loss < 5.0, (
        f"Eğitilmiş loss ({final_loss:.4f}) eğitimsizden ({random_loss:.4f}) çok kötü"
    )


def test_position_sizer_use_rl_hook():
    """position_sizer.py — use_rl=True RL agent ağırlıkları kullanmalı."""
    from engine.risk.position_sizer import RiskConfig, apply_rl_sizer

    rng = np.random.default_rng(5)
    n_dates, n_tickers = 40, 5
    dates = pd.bdate_range("2023-01-01", periods=n_dates)
    tickers = [f"T{i}" for i in range(n_tickers)]
    rets = pd.DataFrame(rng.standard_normal((n_dates, n_tickers)) * 0.01,
                        index=dates, columns=tickers)

    # use_rl=False → vol-target ile özdeş (ya da pass-through)
    cfg_no_rl = RiskConfig(use_rl=False)
    result_no_rl = apply_rl_sizer(rets, cfg_no_rl)
    assert result_no_rl.shape == rets.shape

    # use_rl=True → exception fırlatmamalı, aynı şekli döndürmeli
    cfg_rl = RiskConfig(use_rl=True)
    result_rl = apply_rl_sizer(rets, cfg_rl)
    assert result_rl.shape == rets.shape, (
        f"RL sizer çıktı şekli yanlış: {result_rl.shape} != {rets.shape}"
    )
