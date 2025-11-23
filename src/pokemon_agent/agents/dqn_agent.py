"""Deep Q-Network agent.

Implements DQN (Mnih et al., 2015) with the following extensions:

* **Double DQN** – decouples action selection from value estimation to
  reduce maximisation bias (van Hasselt et al., 2016).
* **Dueling networks** – optional; toggle via ``use_dueling``.
* **Epsilon-greedy exploration** – linear annealing schedule.
* **Target network** – synced every ``target_update`` gradient steps.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from ..environment.actions import N_ACTIONS
from ..models.dqn_network import DuelingQNetwork, QNetwork
from ..training.replay_buffer import ReplayBuffer
from .base import Agent


@dataclass
class DQNConfig:
    """Hyper-parameters for the DQN agent."""

    obs_dim:       int   = 10
    n_actions:     int   = N_ACTIONS
    hidden_dim:    int   = 128
    n_layers:      int   = 3
    use_dueling:   bool  = True

    # Exploration
    eps_start:     float = 1.0
    eps_end:       float = 0.05
    eps_decay:     int   = 200_000   # steps until eps reaches eps_end

    # Learning
    lr:            float = 1e-4
    gamma:         float = 0.99
    batch_size:    int   = 64
    warmup_steps:  int   = 5_000
    target_update: int   = 1_000     # gradient steps between target sync

    # Replay
    buffer_size:   int   = 100_000

    device:        str   = "cpu"


class DQNAgent(Agent):
    """Double DQN agent with optional dueling architecture."""

    def __init__(self, config: Optional[DQNConfig] = None) -> None:
        self.cfg     = config or DQNConfig()
        self.device  = torch.device(self.cfg.device)

        Net = DuelingQNetwork if self.cfg.use_dueling else QNetwork

        self.policy_net: nn.Module = Net(
            obs_dim    = self.cfg.obs_dim,
            n_actions  = self.cfg.n_actions,
            hidden_dim = self.cfg.hidden_dim,
        ).to(self.device)

        self.target_net: nn.Module = Net(
            obs_dim    = self.cfg.obs_dim,
            n_actions  = self.cfg.n_actions,
            hidden_dim = self.cfg.hidden_dim,
        ).to(self.device)

        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.cfg.lr)
        self.buffer    = ReplayBuffer(capacity=self.cfg.buffer_size)

        self._steps_done: int = 0
        self._updates:    int = 0

    # ------------------------------------------------------------------
    # Agent interface
    # ------------------------------------------------------------------

    def select_action(self, obs: np.ndarray) -> int:
        """Epsilon-greedy action selection."""
        eps = self._current_epsilon()
        if random.random() < eps:
            return random.randint(0, self.cfg.n_actions - 1)

        with torch.no_grad():
            t   = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            q   = self.policy_net(t)
            return int(q.argmax(dim=1).item())

    def update(  # type: ignore[override]
        self,
        state:      np.ndarray,
        action:     int,
        reward:     float,
        next_state: np.ndarray,
        done:       bool,
    ) -> dict[str, float]:
        """Store transition and (optionally) run a gradient step.

        Returns:
            Dictionary with ``"loss"`` key when a gradient step was taken,
            otherwise empty.
        """
        self._steps_done += 1
        self.buffer.push(state, action, reward, next_state, done)

        if self._steps_done < self.cfg.warmup_steps:
            return {}
        if len(self.buffer) < self.cfg.batch_size:
            return {}

        return self._gradient_step()

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "policy_net": self.policy_net.state_dict(),
                "target_net": self.target_net.state_dict(),
                "optimizer":  self.optimizer.state_dict(),
                "steps_done": self._steps_done,
                "updates":    self._updates,
                "config":     self.cfg,
            },
            path,
        )

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(ckpt["policy_net"])
        self.target_net.load_state_dict(ckpt["target_net"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self._steps_done = ckpt.get("steps_done", 0)
        self._updates    = ckpt.get("updates",    0)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _current_epsilon(self) -> float:
        """Linearly decaying epsilon schedule."""
        progress = min(self._steps_done / self.cfg.eps_decay, 1.0)
        return self.cfg.eps_end + (self.cfg.eps_start - self.cfg.eps_end) * (1.0 - progress)

    def _gradient_step(self) -> dict[str, float]:
        states, actions, rewards, next_states, dones = self.buffer.sample(
            self.cfg.batch_size
        )

        s  = torch.tensor(states,      dtype=torch.float32, device=self.device)
        a  = torch.tensor(actions,     dtype=torch.int64,   device=self.device).unsqueeze(1)
        r  = torch.tensor(rewards,     dtype=torch.float32, device=self.device)
        ns = torch.tensor(next_states, dtype=torch.float32, device=self.device)
        d  = torch.tensor(dones,       dtype=torch.float32, device=self.device)

        # Double DQN: use policy net to SELECT action, target net to EVALUATE.
        with torch.no_grad():
            next_actions = self.policy_net(ns).argmax(dim=1, keepdim=True)
            next_q       = self.target_net(ns).gather(1, next_actions).squeeze(1)
            target_q     = r + (1.0 - d) * self.cfg.gamma * next_q

        current_q = self.policy_net(s).gather(1, a).squeeze(1)
        loss      = nn.SmoothL1Loss()(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        self._updates += 1
        if self._updates % self.cfg.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return {"loss": loss.item(), "epsilon": self._current_epsilon()}
