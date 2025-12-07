"""PPO training loop.

Collects on-policy rollouts and runs periodic PPO updates.  The loop
follows the standard vectorised workflow but uses a single environment
for simplicity.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

from ..agents.ppo_agent import PPOAgent, PPOConfig
from ..environment.env import PokemonRedEnv
from ..utils.metrics import EpisodeMetrics, MetricsTracker


def train_ppo(
    env:            PokemonRedEnv,
    agent:          PPOAgent,
    total_steps:    int          = 1_000_000,
    log_interval:   int          = 2_048,
    save_interval:  int          = 100_000,
    checkpoint_dir: str          = "checkpoints/ppo",
    writer: Optional[object]     = None,
) -> MetricsTracker:
    """Run the PPO training loop.

    Args:
        env:            Gymnasium-compatible Pokémon Red environment.
        agent:          PPOAgent instance to train.
        total_steps:    Total environment steps to run.
        log_interval:   Print metrics every N steps.
        save_interval:  Save checkpoint every N steps.
        checkpoint_dir: Directory for model checkpoints.
        writer:         Optional TensorBoard SummaryWriter.

    Returns:
        MetricsTracker containing all recorded episode metrics.
    """
    tracker    = MetricsTracker()
    obs, _     = env.reset()
    ep_metrics = EpisodeMetrics()
    step       = 0
    t0         = time.time()
    rollout_step = 0

    rollout_len = agent.cfg.rollout_steps
    print(f"Starting PPO training for {total_steps:,} steps "
          f"(rollout={rollout_len}, epochs={agent.cfg.n_epochs}) …")

    while step < total_steps:
        action   = agent.select_action(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)
        done     = terminated or truncated

        agent.push_transition(
            obs      = obs,
            action   = action,
            reward   = reward,
            done     = done,
            log_prob = agent._last_log_prob,  # cached by select_action
            value    = agent._last_value,
        )

        ep_metrics.add(reward, info)
        obs   = next_obs
        step += 1
        rollout_step += 1

        if done:
            tracker.record(ep_metrics)
            ep_metrics = EpisodeMetrics()
            obs, _     = env.reset()

        if rollout_step >= rollout_len:
            update_info  = agent.update(last_obs=obs, last_done=done)
            rollout_step = 0

            if step % log_interval == 0:
                elapsed = time.time() - t0
                fps     = step / elapsed
                recent  = tracker.recent(10)
                loss    = update_info.get("loss", float("nan"))
                print(
                    f"step={step:>8,}  "
                    f"fps={fps:.0f}  "
                    f"loss={loss:.4f}  "
                    f"ep_ret={recent.get('mean_return', float('nan')):.2f}  "
                    f"badges={recent.get('mean_badges', float('nan')):.1f}"
                )

            if step % save_interval == 0 and step > 0:
                ckpt = Path(checkpoint_dir) / f"ppo_step{step:08d}.pt"
                agent.save(str(ckpt))
                print(f"  → checkpoint saved: {ckpt}")

    print(f"Training complete in {time.time() - t0:.1f}s")
    return tracker
