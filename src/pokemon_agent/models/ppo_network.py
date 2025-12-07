"""Actor-Critic network for PPO.

A shared-trunk MLP feeds into separate policy (actor) and value (critic)
heads.  This architecture allows feature reuse and reduces the total
parameter count compared to two independent networks.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Categorical


class ActorCritic(nn.Module):
    """Shared-trunk actor-critic network for Proximal Policy Optimisation.

    Args:
        obs_dim:    Observation vector dimensionality.
        n_actions:  Number of discrete actions.
        hidden_dim: Width of shared hidden layers.
    """

    def __init__(
        self,
        obs_dim: int = 10,
        n_actions: int = 9,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()

        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )

        self.actor  = nn.Linear(hidden_dim, n_actions)
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (logits, value) for the given observation batch."""
        h      = self.shared(x)
        logits = self.actor(h)
        value  = self.critic(h).squeeze(-1)
        return logits, value

    def get_action_and_value(
        self, x: torch.Tensor, action: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample (or evaluate) an action and return auxiliary quantities.

        Args:
            x:      Observation batch of shape (B, obs_dim).
            action: If provided, evaluate log-probs of these specific actions
                    instead of sampling new ones.

        Returns:
            (action, log_prob, entropy, value)
        """
        logits, value = self(x)
        dist          = Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy  = dist.entropy()
        return action, log_prob, entropy, value
