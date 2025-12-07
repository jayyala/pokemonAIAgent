"""Proximal Policy Optimisation (PPO) agent.

Implements the clipped surrogate objective from Schulman et al. (2017)
with the following standard additions:

* Generalised Advantage Estimation (GAE)
* Value function clipping
* Entropy bonus for exploration
* Gradient norm clipping
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from ..environment.actions import N_ACTIONS
from ..models.ppo_network import ActorCritic
from .base import Agent


@dataclass
class PPOConfig:
    """Hyper-parameters for the PPO agent."""

    obs_dim:       int   = 10
    n_actions:     int   = N_ACTIONS
    hidden_dim:    int   = 128

    # Rollout
    rollout_steps: int   = 2_048   # steps per policy update
    n_epochs:      int   = 10      # gradient epochs per update
    batch_size:    int   = 64

    # PPO
    clip_eps:      float = 0.2
    vf_coef:       float = 0.5
    ent_coef:      float = 0.01
    max_grad_norm: float = 0.5

    # GAE
    gamma:         float = 0.99
    gae_lambda:    float = 0.95

    lr:            float = 2.5e-4
    device:        str   = "cpu"


class PPOAgent(Agent):
    """On-policy PPO agent using an actor-critic network."""

    def __init__(self, config: Optional[PPOConfig] = None) -> None:
        self.cfg    = config or PPOConfig()
        self.device = torch.device(self.cfg.device)

        self.ac = ActorCritic(
            obs_dim    = self.cfg.obs_dim,
            n_actions  = self.cfg.n_actions,
            hidden_dim = self.cfg.hidden_dim,
        ).to(self.device)

        self.optimizer = optim.Adam(self.ac.parameters(), lr=self.cfg.lr, eps=1e-5)

        # Rollout storage (filled by the training loop via push()).
        self._reset_rollout()

    # ------------------------------------------------------------------
    # Agent interface
    # ------------------------------------------------------------------

    def select_action(self, obs: np.ndarray) -> int:
        t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            action, log_prob, _, value = self.ac.get_action_and_value(t)
        # Cache for rollout storage.
        self._last_log_prob = log_prob.item()
        self._last_value    = value.item()
        return int(action.item())

    def push_transition(
        self,
        obs:      np.ndarray,
        action:   int,
        reward:   float,
        done:     bool,
        log_prob: float,
        value:    float,
    ) -> None:
        """Store a rollout transition (called by the training loop)."""
        self._obs.append(obs)
        self._actions.append(action)
        self._rewards.append(reward)
        self._dones.append(done)
        self._log_probs.append(log_prob)
        self._values.append(value)

    def update(  # type: ignore[override]
        self,
        last_obs:  np.ndarray,
        last_done: bool,
    ) -> dict[str, float]:
        """Run PPO update on the collected rollout.

        Args:
            last_obs:  Observation *after* the last step in the rollout
                       (needed to bootstrap the GAE computation).
            last_done: Whether the episode ended on the last step.

        Returns:
            Dictionary with training metrics.
        """
        with torch.no_grad():
            t          = torch.tensor(last_obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            _, last_v  = self.ac(t)
            last_value = last_v.item()

        advantages = self._compute_gae(last_value, last_done)
        returns    = advantages + np.array(self._values, np.float32)

        metrics = self._ppo_update(advantages, returns)
        self._reset_rollout()
        return metrics

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"ac": self.ac.state_dict(), "optimizer": self.optimizer.state_dict()}, path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.ac.load_state_dict(ckpt["ac"])
        self.optimizer.load_state_dict(ckpt["optimizer"])

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _reset_rollout(self) -> None:
        self._obs:       list[np.ndarray] = []
        self._actions:   list[int]        = []
        self._rewards:   list[float]      = []
        self._dones:     list[bool]       = []
        self._log_probs: list[float]      = []
        self._values:    list[float]      = []

    def _compute_gae(self, last_value: float, last_done: bool) -> np.ndarray:
        n          = len(self._rewards)
        advantages = np.zeros(n, dtype=np.float32)
        gae        = 0.0
        next_val   = last_value
        next_done  = float(last_done)

        for t in reversed(range(n)):
            delta = (
                self._rewards[t]
                + self.cfg.gamma * next_val * (1.0 - next_done)
                - self._values[t]
            )
            gae         = delta + self.cfg.gamma * self.cfg.gae_lambda * (1.0 - next_done) * gae
            advantages[t] = gae
            next_val    = self._values[t]
            next_done   = float(self._dones[t])

        return advantages

    def _ppo_update(
        self, advantages: np.ndarray, returns: np.ndarray
    ) -> dict[str, float]:
        obs_t      = torch.tensor(np.stack(self._obs),    dtype=torch.float32, device=self.device)
        actions_t  = torch.tensor(self._actions,          dtype=torch.int64,   device=self.device)
        old_lp_t   = torch.tensor(self._log_probs,        dtype=torch.float32, device=self.device)
        adv_t      = torch.tensor(advantages,             dtype=torch.float32, device=self.device)
        returns_t  = torch.tensor(returns,                dtype=torch.float32, device=self.device)
        old_vals_t = torch.tensor(self._values,           dtype=torch.float32, device=self.device)

        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

        n          = len(self._rewards)
        total_loss = 0.0
        n_updates  = 0

        for _ in range(self.cfg.n_epochs):
            perm = torch.randperm(n)
            for start in range(0, n, self.cfg.batch_size):
                idx = perm[start : start + self.cfg.batch_size]

                _, log_probs, entropy, values = self.ac.get_action_and_value(
                    obs_t[idx], actions_t[idx]
                )

                ratio = (log_probs - old_lp_t[idx]).exp()
                pg1   = ratio * adv_t[idx]
                pg2   = ratio.clamp(1.0 - self.cfg.clip_eps, 1.0 + self.cfg.clip_eps) * adv_t[idx]
                pg    = -torch.min(pg1, pg2).mean()

                # Clipped value loss
                v_clipped = old_vals_t[idx] + (values - old_vals_t[idx]).clamp(
                    -self.cfg.clip_eps, self.cfg.clip_eps
                )
                vf = torch.max(
                    (values - returns_t[idx]).pow(2),
                    (v_clipped - returns_t[idx]).pow(2),
                ).mean()

                loss = pg + self.cfg.vf_coef * vf - self.cfg.ent_coef * entropy.mean()

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.ac.parameters(), self.cfg.max_grad_norm)
                self.optimizer.step()

                total_loss += loss.item()
                n_updates  += 1

        return {"loss": total_loss / max(n_updates, 1)}
