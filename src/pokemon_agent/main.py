"""CLI entry point for the Pokémon Red agent.

Commands
--------
rollout   Run a single episode with a heuristic or random agent.
train     Train a DQN or PPO agent from scratch.
eval      Evaluate a saved checkpoint.

Examples
--------
# Quick sanity check (no ROM needed):
pokemon-agent rollout --agent heuristic --max-steps 500

# Train DQN (requires ROM + PyBoy):
pokemon-agent train dqn --rom roms/PokemonRed.gb --steps 500000

# Evaluate a checkpoint:
pokemon-agent eval --rom roms/PokemonRed.gb --checkpoint checkpoints/dqn/dqn_step00500000.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_emulator(rom: Optional[str], window: bool):
    """Return a real or mock emulator depending on availability."""
    if rom is not None:
        try:
            from pokemon_agent.emulator.pyboy_emulator import PyBoyConfig, PyBoyEmulator

            cfg = PyBoyConfig(
                rom_path    = rom,
                window_type = "SDL2" if window else "headless",
            )
            return PyBoyEmulator(cfg)
        except ImportError:
            print("[warn] PyBoy not found – falling back to MockEmulator.", file=sys.stderr)

    from pokemon_agent.emulator.mock_emulator import MockEmulator

    return MockEmulator()


def _build_env(emulator, max_steps: int):
    from pokemon_agent.environment.env import PokemonRedEnv

    return PokemonRedEnv(emulator, max_steps=max_steps)


# ---------------------------------------------------------------------------
# Subcommand handlers
# ---------------------------------------------------------------------------

def cmd_rollout(args: argparse.Namespace) -> None:
    """Execute one episode and print summary statistics."""
    from pokemon_agent.agents.heuristic_agent import HeuristicAgent
    from pokemon_agent.agents.random_agent import RandomAgent
    from pokemon_agent.utils.logging import setup_logging

    setup_logging(level="INFO")

    emulator = _build_emulator(args.rom, args.window)
    env      = _build_env(emulator, args.max_steps)

    if args.agent == "random":
        agent = RandomAgent(seed=args.seed)
    else:
        agent = HeuristicAgent(seed=args.seed)

    obs, _ = env.reset()
    total_reward = 0.0
    step = 0

    while True:
        action = agent.select_action(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        step += 1
        if terminated or truncated:
            break

    env.close()
    print(
        f"\nEpisode finished in {step} steps  |  "
        f"return={total_reward:.2f}  |  "
        f"badges={info.get('badge_count', 0)}  |  "
        f"pokédex_seen={info.get('pokedex_seen', 0)}"
    )


def cmd_train(args: argparse.Namespace) -> None:
    """Train a DQN or PPO agent."""
    from pokemon_agent.utils.logging import setup_logging

    setup_logging(level="INFO", log_file=f"logs/{args.algo}_train.log")

    emulator = _build_emulator(args.rom, args.window)
    env      = _build_env(emulator, max_steps=20_480)

    writer = None
    if args.tensorboard:
        try:
            from torch.utils.tensorboard import SummaryWriter

            writer = SummaryWriter(log_dir=f"runs/{args.algo}")
        except ImportError:
            print("[warn] TensorBoard not available; skipping.", file=sys.stderr)

    if args.algo == "dqn":
        from pokemon_agent.agents.dqn_agent import DQNAgent, DQNConfig
        from pokemon_agent.training.dqn_trainer import train_dqn

        agent = DQNAgent(DQNConfig(device=args.device))
        train_dqn(
            env            = env,
            agent          = agent,
            total_steps    = args.steps,
            checkpoint_dir = args.checkpoint_dir,
            writer         = writer,
        )
    elif args.algo == "ppo":
        from pokemon_agent.agents.ppo_agent import PPOAgent, PPOConfig
        from pokemon_agent.training.ppo_trainer import train_ppo

        agent = PPOAgent(PPOConfig(device=args.device))
        train_ppo(
            env            = env,
            agent          = agent,
            total_steps    = args.steps,
            checkpoint_dir = args.checkpoint_dir,
            writer         = writer,
        )
    else:
        print(f"Unknown algorithm: {args.algo}", file=sys.stderr)
        sys.exit(1)

    env.close()


def cmd_eval(args: argparse.Namespace) -> None:
    """Evaluate a saved checkpoint for one episode."""
    from pokemon_agent.agents.dqn_agent import DQNAgent, DQNConfig
    from pokemon_agent.utils.logging import setup_logging

    setup_logging(level="INFO")

    emulator = _build_emulator(args.rom, args.window)
    env      = _build_env(emulator, args.max_steps)

    agent = DQNAgent(DQNConfig(eps_start=0.0, eps_end=0.0))
    agent.load(args.checkpoint)

    obs, _ = env.reset()
    total_reward = 0.0
    step = 0

    while True:
        action = agent.select_action(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        step += 1
        if terminated or truncated:
            break

    env.close()
    print(
        f"\nEval finished in {step} steps  |  "
        f"return={total_reward:.2f}  |  "
        f"badges={info.get('badge_count', 0)}  |  "
        f"pokédex_seen={info.get('pokedex_seen', 0)}"
    )


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog        = "pokemon-agent",
        description = "Reinforcement learning agent for Pokémon Red",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # --- rollout ---
    p_roll = sub.add_parser("rollout", help="Run a single episode")
    p_roll.add_argument("--rom",       default=None,        help="Path to ROM file")
    p_roll.add_argument("--agent",     default="heuristic", choices=["heuristic", "random"])
    p_roll.add_argument("--max-steps", default=2_000,       type=int)
    p_roll.add_argument("--window",    action="store_true",  help="Open emulator window")
    p_roll.add_argument("--seed",      default=42,           type=int)

    # --- train ---
    p_train = sub.add_parser("train", help="Train an RL agent")
    p_train.add_argument("algo",              choices=["dqn", "ppo"])
    p_train.add_argument("--rom",             default=None,              help="Path to ROM file")
    p_train.add_argument("--steps",           default=500_000,           type=int)
    p_train.add_argument("--device",          default="cpu")
    p_train.add_argument("--window",          action="store_true")
    p_train.add_argument("--tensorboard",     action="store_true")
    p_train.add_argument("--checkpoint-dir",  default="checkpoints")

    # --- eval ---
    p_eval = sub.add_parser("eval", help="Evaluate a saved checkpoint")
    p_eval.add_argument("--checkpoint", required=True,    help="Path to .pt checkpoint")
    p_eval.add_argument("--rom",        default=None)
    p_eval.add_argument("--max-steps",  default=20_480,  type=int)
    p_eval.add_argument("--window",     action="store_true")

    return parser


def main() -> None:
    parser  = build_parser()
    args    = parser.parse_args()

    dispatch = {
        "rollout": cmd_rollout,
        "train":   cmd_train,
        "eval":    cmd_eval,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
