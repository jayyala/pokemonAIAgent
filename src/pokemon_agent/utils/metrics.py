"""Episode metrics tracking.

Lightweight containers for recording and aggregating training statistics
without introducing heavy dependencies.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any


@dataclass
class EpisodeMetrics:
    """Accumulates metrics over a single episode."""

    total_return:  float = 0.0
    length:        int   = 0
    badge_count:   int   = 0
    pokedex_seen:  int   = 0
    max_map_id:    int   = 0

    def add(self, reward: float, info: dict[str, Any]) -> None:
        self.total_return += reward
        self.length       += 1
        self.badge_count   = max(self.badge_count,  info.get("badge_count",  0))
        self.pokedex_seen  = max(self.pokedex_seen,  info.get("pokedex_seen", 0))
        self.max_map_id    = max(self.max_map_id,    info.get("map_id",       0))

    def summary(self) -> dict[str, float]:
        return {
            "return":       self.total_return,
            "length":       float(self.length),
            "badge_count":  float(self.badge_count),
            "pokedex_seen": float(self.pokedex_seen),
        }


class MetricsTracker:
    """Stores completed episode metrics and computes rolling statistics."""

    def __init__(self, maxlen: int = 200) -> None:
        self._episodes: deque[EpisodeMetrics] = deque(maxlen=maxlen)

    def record(self, ep: EpisodeMetrics) -> None:
        self._episodes.append(ep)

    def recent(self, n: int = 10) -> dict[str, float]:
        """Return mean statistics over the last *n* episodes."""
        episodes = list(self._episodes)[-n:]
        if not episodes:
            return {}
        mean_return  = sum(e.total_return for e in episodes) / len(episodes)
        mean_len     = sum(e.length       for e in episodes) / len(episodes)
        mean_badges  = sum(e.badge_count  for e in episodes) / len(episodes)
        mean_pokedex = sum(e.pokedex_seen for e in episodes) / len(episodes)
        return {
            "mean_return":  mean_return,
            "mean_length":  mean_len,
            "mean_badges":  mean_badges,
            "mean_pokedex": mean_pokedex,
        }

    def __len__(self) -> int:
        return len(self._episodes)
