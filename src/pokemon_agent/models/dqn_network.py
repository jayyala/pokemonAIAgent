"""Q-network architectures for DQN and its variants.

Two architectures are provided:

* **QNetwork** – vanilla MLP Q-function.
* **DuelingQNetwork** – dueling streams (advantage + value) with the
  mean-subtraction stabilisation from Wang et al. (2016).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    """Simple multi-layer perceptron Q-function.

    Args:
        obs_dim:    Dimensionality of the observation vector.
        n_actions:  Number of discrete actions.
        hidden_dim: Width of each hidden layer.
        n_layers:   Number of hidden layers.
    """

    def __init__(
        self,
        obs_dim: int = 10,
        n_actions: int = 9,
        hidden_dim: int = 128,
        n_layers: int = 3,
    ) -> None:
        super().__init__()

        layers: list[nn.Module] = [nn.Linear(obs_dim, hidden_dim), nn.ReLU()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        layers.append(nn.Linear(hidden_dim, n_actions))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DuelingQNetwork(nn.Module):
    """Dueling DQN architecture (Wang et al., 2016).

    Separates Q(s, a) into a state value V(s) and advantage A(s, a):

        Q(s, a) = V(s) + A(s, a) - mean_a'[A(s, a')]

    This decomposition helps the network learn which states are valuable
    independent of the specific action taken.
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
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Value stream: V(s) – scalar
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        # Advantage stream: A(s, a) – per-action
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.shared(x)
        v = self.value_stream(h)           # (B, 1)
        a = self.advantage_stream(h)       # (B, n_actions)
        # Mean-subtraction for identifiability.
        q = v + (a - a.mean(dim=1, keepdim=True))
        return q
