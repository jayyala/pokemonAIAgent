"""DQN training loop.

Orchestrates interaction between the DQNAgent and a PokemonRedEnv,
logging metrics to stdout and optionally to TensorBoard.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from ..agents.dqn_agent import DQNAgent, DQNConfig
from ..environment.env import PokemonRedEnv
from ..utils.metrics import EpisodeMetrics, MetricsTracker


@dataclass
class DQNConfig(DQNConfig):  # type: ignore[no-redef]
    """Extended config that adds trainer-level settings."""

    total_steps:    int  = 500_000
    log_interval:   int  = 1_000
    eval_interval:  int  = 10_000
    save_interval:  int  = 50_000
    checkpoint_dir: str  = "checkpoints/dqn"


def train_dqn(
    env:        PokemonRedEnv,
    agent:      DQNAgent,
    total_steps: int            = 500_000,
    log_interval: int           = 1_000,
    eval_interval: int          = 10_000,
    save_interval: int          = 50_000,
    checkpoint_dir: str         = "checkpoints/dqn",
    writer: Optional[object]    = None,
) -> MetricsTracker:
    """Run the DQN training loop.

    Args:
        env:            Gymnasium-compatible Pokémon Red environment.
        agent:          DQNAgent instance to train.
        total_steps:    Total environment steps to run.
        log_interval:   Print metrics every N steps.
        eval_interval:  Unused placeholder for eval hook.
        save_interval:  Save checkpoint every N steps.
        checkpoint_dir: Directory for model checkpoints.
        writer:         Optional TensorBoard SummaryWriter.

    Returns:
        MetricsTracker containing all recorded episode metrics.
    """
    tracker = MetricsTracker()
    obs, _  = env.reset()

    ep_metrics = EpisodeMetrics()
    step       = 0
    t0         = time.time()

    print(f"Starting DQN training for {total_steps:,} steps …")

    while step < total_steps:
        action  = agent.select_action(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)
        done    = terminated or truncated

        update_info = agent.update(obs, action, reward, next_obs, done)

        ep_metrics.add(reward, info)
        obs  = next_obs
        step += 1

        if done:
            tracker.record(ep_metrics)
            if writer is not None:
                for k, v in ep_metrics.summary().items():
                    writer.add_scalar(f"train/{k}", v, step)
            ep_metrics = EpisodeMetrics()
            obs, _ = env.reset()

        if step % log_interval == 0:
            elapsed = time.time() - t0
            fps     = step / elapsed
            recent  = tracker.recent(10)
            loss    = update_info.get("loss", float("nan"))
            eps     = update_info.get("epsilon", float("nan"))
            print(
                f"step={step:>8,}  "
                f"fps={fps:.0f}  "
                f"loss={loss:.4f}  "
                f"ε={eps:.3f}  "
                f"ep_ret={recent.get('mean_return', float('nan')):.2f}  "
                f"badges={recent.get('mean_badges', float('nan')):.1f}"
            )

        if step % save_interval == 0 and step > 0:
            ckpt_path = Path(checkpoint_dir) / f"dqn_step{step:08d}.pt"
            agent.save(str(ckpt_path))
            print(f"  → checkpoint saved: {ckpt_path}")

    print(f"Training complete in {time.time() - t0:.1f}s")
    return tracker
