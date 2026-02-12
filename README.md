# Pokémon Red AI Agent

A reinforcement learning agent that learns to play **Pokémon Red** on the original Game Boy hardware, powered by the [PyBoy](https://github.com/Baekalfen/PyBoy) emulator.

The project implements two complete RL algorithms — **Double DQN** and **PPO** — side-by-side with a hand-crafted heuristic baseline, all sharing the same environment wrapper and reward-shaping pipeline.

## Current Status

This repository is an end-to-end MVP with:

- A working environment wrapper over real (`PyBoyEmulator`) and test (`MockEmulator`) backends.
- Two trainable agents (DQN, PPO) plus heuristic/random baselines.
- Checkpoint save/load support.
- Unit tests for core state parsing, actions, and replay-buffer behavior.
- Shell scripts for quickstart, training, and evaluation.

Known limitations:

- The CLI currently wires algorithm/runtime options directly and does **not** auto-load YAML files from `configs/`.
- `eval` currently loads a **DQN** checkpoint path.
- Long-horizon game mastery still requires substantial reward tuning and training time.

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Project Structure](#project-structure)
4. [Installation](#installation)
5. [Quick Start](#quick-start)
6. [Usage](#usage)
   - [Rollout (no training)](#rollout-no-training)
   - [Training](#training)
   - [Evaluation](#evaluation)
7. [Agents](#agents)
   - [Random Agent](#random-agent)
   - [Heuristic Agent](#heuristic-agent)
   - [DQN Agent](#dqn-agent)
   - [PPO Agent](#ppo-agent)
8. [Environment](#environment)
   - [Observation Space](#observation-space)
   - [Action Space](#action-space)
   - [Reward Shaping](#reward-shaping)
9. [Configuration](#configuration)
10. [Memory Map](#memory-map)
11. [Training Tips](#training-tips)
12. [Results](#results)
13. [Development](#development)
14. [References](#references)

---

## Overview

Pokémon Red presents a challenging RL environment:

| Property | Value |
|---|---|
| Game Boy screen | 160 × 144 pixels, 4-colour palette |
| Agent step = | 24 Game Boy frames (~0.4 s of game time) |
| Action space | 9 discrete buttons |
| Episode horizon | 20,480 steps (~2.3 hours of game time) |
| Sparse rewards | Badges awarded every ~2–5 hours of real gameplay |
| Long credit assignment | 100s of steps between cause and effect |

Rather than feeding raw pixel frames to a CNN (which would require days of GPU training), this project extracts a compact **10-dimensional feature vector** directly from the Game Boy's memory bus.  This lets training converge on consumer hardware in a few hours while keeping the code readable and easy to modify.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         CLI / main.py                           │
│          rollout │ train dqn │ train ppo │ eval                 │
└──────────────────┬──────────────────────────────────────────────┘
                   │
          ┌────────▼────────┐
          │  PokemonRedEnv  │  ← Gymnasium-compatible wrapper
          │  (env.py)       │
          └────┬────────────┘
               │ step(action) → obs, reward, done, info
    ┌──────────▼───────────┐        ┌─────────────────────────────┐
    │      Emulator        │        │        GameState             │
    │  PyBoyEmulator  or   │──────▶│  state.py  (memory parser)  │
    │  MockEmulator        │        │  rewards.py (reward shaper)  │
    └──────────────────────┘        └─────────────────────────────┘
               │
    ┌──────────▼──────────────────────────────────────────────────┐
    │                        Agents                               │
    │  RandomAgent │ HeuristicAgent │ DQNAgent │ PPOAgent         │
    └─────────────────────────────────────────────────────────────┘
               │
    ┌──────────▼──────────────────────────────────────────────────┐
    │                    Neural Networks                           │
    │  QNetwork │ DuelingQNetwork │ ActorCritic                   │
    └─────────────────────────────────────────────────────────────┘
               │
    ┌──────────▼──────────────────────────────────────────────────┐
    │                    Training Loops                            │
    │  dqn_trainer.py (off-policy) │ ppo_trainer.py (on-policy)  │
    └─────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
pokemonaiagent/
├── configs/
│   ├── default.yaml          # Shared defaults
│   ├── dqn.yaml              # DQN hyper-parameters
│   └── ppo.yaml              # PPO hyper-parameters
├── scripts/
│   ├── quickstart.sh         # One-command setup & demo
│   ├── train.sh              # Training convenience wrapper
│   └── eval.sh               # Evaluation wrapper
├── src/pokemon_agent/
│   ├── main.py               # CLI entry point
│   ├── emulator/
│   │   ├── base.py           # Abstract Emulator interface
│   │   ├── pyboy_emulator.py # Real PyBoy backend
│   │   └── mock_emulator.py  # Scriptable mock for tests & CI
│   ├── environment/
│   │   ├── actions.py        # Action space (IntEnum, 9 buttons)
│   │   ├── state.py          # Memory-mapped GameState parser
│   │   ├── rewards.py        # Shaped reward decomposition
│   │   └── env.py            # Gymnasium Env wrapper
│   ├── agents/
│   │   ├── base.py           # Abstract Agent
│   │   ├── random_agent.py   # Uniform random baseline
│   │   ├── heuristic_agent.py# Rule-based baseline
│   │   ├── dqn_agent.py      # Double DQN + Dueling
│   │   └── ppo_agent.py      # Proximal Policy Optimisation
│   ├── models/
│   │   ├── dqn_network.py    # QNetwork + DuelingQNetwork
│   │   └── ppo_network.py    # ActorCritic (shared trunk)
│   ├── training/
│   │   ├── replay_buffer.py  # Uniform experience replay
│   │   ├── dqn_trainer.py    # Off-policy training loop
│   │   └── ppo_trainer.py    # On-policy training loop
│   └── utils/
│       ├── logging.py        # Logger factory + setup
│       └── metrics.py        # Episode metric tracking
└── tests/
    ├── test_actions.py
    ├── test_state.py
    ├── test_replay_buffer.py
    └── test_agents.py
```

---

## Installation

### Prerequisites

| Requirement | Version |
|---|---|
| Python | ≥ 3.10 |
| PyBoy | ≥ 1.5.2 (optional) |
| PyTorch | ≥ 2.2 (optional) |
| Gymnasium | ≥ 0.29 (optional) |

> **ROM required for real training.**  You need a legal copy of the Pokémon Red ROM (`PokemonRed.gb`).  Place it in the `roms/` directory (excluded from version control by `.gitignore`).

### Steps

```bash
# 1. Clone
git clone https://github.com/jayyala/pokemonaiagent.git
cd pokemonaiagent

# 2. Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install (choose your extras)
pip install -e ".[all]"          # everything
pip install -e ".[emulator,ml]"  # emulator + PyTorch, no dev tools
pip install -e "."               # core only (mock emulator, no training)
```

---

## Quick Start

The quickstart script sets up a virtual environment and runs a 500-step heuristic rollout using the **mock emulator** — no ROM required.

```bash
./scripts/quickstart.sh
```

Expected output:

```
=== Pokémon Red Agent – Quickstart ===
→ Creating virtual environment …
→ Installing dependencies …
→ Running 500-step heuristic rollout (mock emulator) …

Episode finished in 500 steps  |  return=3.47  |  badges=0  |  pokédex_seen=0
```

---

## Usage

All commands are available via the `pokemon-agent` CLI or directly as `python -m pokemon_agent.main`.

### Rollout (no training)

Run a single episode with a pre-built agent:

```bash
# Heuristic agent, mock emulator (no ROM needed)
pokemon-agent rollout --agent heuristic --max-steps 2000

# Random agent with a specific ROM
pokemon-agent rollout --agent random --rom roms/PokemonRed.gb --window
```

### Training

```bash
# Train DQN for 500k steps (mock emulator, no ROM)
pokemon-agent train dqn --steps 500000

# Train DQN with a real ROM on GPU, with TensorBoard logging
pokemon-agent train dqn \
  --rom roms/PokemonRed.gb \
  --steps 500000 \
  --device cuda \
  --tensorboard

# Train PPO
pokemon-agent train ppo \
  --rom roms/PokemonRed.gb \
  --steps 1000000 \
  --device cuda \
  --tensorboard
```

Monitor training with TensorBoard:

```bash
tensorboard --logdir runs/
```

### Evaluation

```bash
pokemon-agent eval \
  --checkpoint checkpoints/dqn/dqn_step00500000.pt \
  --rom roms/PokemonRed.gb \
  --window
```

---

## Agents

### Random Agent

Selects uniformly at random from all 9 buttons each step.  Used as the absolute lower bound in benchmark comparisons.

```python
from pokemon_agent.agents import RandomAgent
agent = RandomAgent(seed=42)
action = agent.select_action(obs)
```

### Heuristic Agent

A rule-based policy encoding basic game knowledge:

- **Battle mode** – press A with 80% probability (attack), occasionally press B or navigate the move menu.
- **Overworld** – choose a random cardinal direction; 25% chance to press A (interact / advance dialog).

The heuristic is surprisingly competitive on short horizons because battle victory requires mostly pressing A, and dialog progression requires the same.

```python
from pokemon_agent.agents import HeuristicAgent
agent = HeuristicAgent(battle_a_prob=0.80, confirm_prob=0.25, seed=42)
```

### DQN Agent

Implements **Double DQN** (van Hasselt et al., 2016) with a **Dueling Network** architecture (Wang et al., 2016):

- **Double DQN** decouples action selection (policy network) from value estimation (target network), reducing Q-value overestimation.
- **Dueling streams** separately learn a state-value V(s) and per-action advantage A(s, a), helping the agent identify good states independent of actions.
- **Epsilon-greedy exploration** with linear annealing from ε=1.0 → 0.05 over 200k steps.
- **Target network** synchronised every 1,000 gradient steps.
- **Smooth L1 loss** (Huber loss) for gradient stability.
- **Gradient clipping** at max norm 10.

```python
from pokemon_agent.agents import DQNAgent
from pokemon_agent.agents.dqn_agent import DQNConfig

agent = DQNAgent(DQNConfig(
    hidden_dim    = 128,
    use_dueling   = True,
    eps_decay     = 200_000,
    target_update = 1_000,
    device        = "cuda",
))
```

### PPO Agent

Implements **Proximal Policy Optimisation** (Schulman et al., 2017) with:

- **Shared actor-critic trunk** (Tanh activations) for feature reuse.
- **Generalised Advantage Estimation** (GAE, λ=0.95) for lower-variance returns.
- **Clipped surrogate objective** (ε=0.2) to constrain policy updates.
- **Value function clipping** for training stability.
- **Entropy bonus** (coef=0.01) to prevent premature convergence.
- **10 epochs** of mini-batch gradient descent per rollout.

```python
from pokemon_agent.agents import PPOAgent
from pokemon_agent.agents.ppo_agent import PPOConfig

agent = PPOAgent(PPOConfig(
    rollout_steps = 2048,
    n_epochs      = 10,
    clip_eps      = 0.2,
    device        = "cuda",
))
```

---

## Environment

### Observation Space

The observation is a **10-dimensional float32 vector** extracted from Game Boy memory and normalised to [0, 1]:

| Index | Feature | Source address | Normalisation |
|---|---|---|---|
| 0 | Map ID | `0xD35E` | / 255 |
| 1 | Player X tile | `0xD362` | / 255 |
| 2 | Player Y tile | `0xD361` | / 255 |
| 3 | Battle type | `0xD057` | / 2 |
| 4 | HP fraction | `0xD16B-C / 0xD16D-E` | — |
| 5 | Level | `0xD18C` | / 100 |
| 6 | Badge count | `0xD356` (popcount) | / 8 |
| 7 | Pokédex seen | `0xD30A` | / 151 |
| 8 | Pokédex owned | `0xD2F7` | / 151 |
| 9 | In battle flag | derived | {0, 1} |

### Action Space

`Discrete(9)` — maps to physical Game Boy buttons:

| Index | Action |
|---|---|
| 0 | NOOP |
| 1 | UP |
| 2 | DOWN |
| 3 | LEFT |
| 4 | RIGHT |
| 5 | A |
| 6 | B |
| 7 | START |
| 8 | SELECT |

Each action advances the Game Boy by **24 frames** (~0.4 s), giving the agent time to observe the result of button presses before the next decision.

### Reward Shaping

The reward is decomposed into six components to provide dense feedback throughout training:

| Component | Trigger | Coefficient |
|---|---|---|
| HP survival | Every step | +0.01 × HP fraction |
| Exploration | New map entered | +0.02 |
| Badge | New badge obtained | +5.0 per badge |
| Pokédex seen | New species encountered | +0.5 |
| Pokédex owned | New species caught | +1.0 |
| Level-up | Lead Pokémon gains a level | +2.0 |
| Faint penalty | Lead Pokémon HP → 0 | −5.0 |

All coefficients are configurable via `configs/default.yaml`.

---

## Configuration

`configs/` contains ready-to-use configuration templates for environment, reward shaping, and algorithm hyper-parameters.  
The current CLI does not auto-parse these files yet; values are passed through dataclass defaults / CLI flags in code.

```yaml
# configs/dqn.yaml (excerpt)
agent:
  hidden_dim:    128
  use_dueling:   true
  eps_start:     1.0
  eps_end:       0.05
  eps_decay:     200000
  lr:            0.0001
  gamma:         0.99
  target_update: 1000
  buffer_size:   100000
```

---

## Memory Map

Key Pokémon Red RAM addresses used for state parsing:

| Symbol | Address | Description |
|---|---|---|
| `BATTLE_TYPE` | `0xD057` | 0=none, 1=wild, 2=trainer |
| `MAP_ID` | `0xD35E` | Current map index |
| `X_COORD` | `0xD362` | Player X tile position |
| `Y_COORD` | `0xD361` | Player Y tile position |
| `PARTY_COUNT` | `0xD163` | Number of Pokémon in party |
| `HP_HIGH/LOW` | `0xD16B-C` | Lead Pokémon current HP (16-bit BE) |
| `MAX_HP_HIGH/LOW` | `0xD16D-E` | Lead Pokémon max HP (16-bit BE) |
| `LEVEL_1` | `0xD18C` | Lead Pokémon level |
| `STATUS_1` | `0xD16F` | Status condition bitmask |
| `BADGES` | `0xD356` | 8-badge bitmask |
| `POKEDEX_OWNED` | `0xD2F7` | Species owned count |
| `POKEDEX_SEEN` | `0xD30A` | Species seen count |

Source: [pret/pokered](https://github.com/pret/pokered) disassembly.

---

## Training Tips

**No ROM?**  Every command falls back to the `MockEmulator` automatically.  You can develop and test the full pipeline without a ROM.

**Short feedback loops:**  Run `rollout` with the heuristic agent first to verify the environment is working correctly before investing in long training runs.

**Device selection:**
```bash
# Apple Silicon
pokemon-agent train dqn --device mps --rom roms/PokemonRed.gb

# NVIDIA GPU
pokemon-agent train dqn --device cuda --rom roms/PokemonRed.gb
```

**Reward tuning:**  The badge scale (default 5.0) is the dominant sparse reward.  If the agent is not progressing past the first gym, try increasing it to 10.0–20.0.

**DQN vs PPO:**  DQN tends to converge faster on this task due to sample efficiency from experience replay.  PPO is more stable but requires ~2× the environment interaction.

**Save-state speedups:**  PyBoy supports loading a `.state` file at episode reset, letting you start training from a specific point in the game (e.g., after the opening sequence).

---

## Results

The codebase includes logging hooks for collecting metrics (`return`, `badge_count`, `pokedex_seen`) during rollout and training.

To generate your own benchmark table:

1. Train DQN/PPO with fixed seeds and identical reward scales.
2. Evaluate saved checkpoints across multiple episodes.
3. Aggregate metrics from stdout logs or TensorBoard runs in `runs/`.

If you want deterministic comparisons, keep emulator settings, seed, frame skip, and max steps constant across runs.

---

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Lint + type-check
ruff check src/ tests/
mypy src/
```

Tests use the `MockEmulator` exclusively, so no ROM is required to run the full test suite.

---

## References

1. Mnih, V., et al. (2015). **Human-level control through deep reinforcement learning.** *Nature*, 518, 529–533.
2. van Hasselt, H., Guez, A., & Silver, D. (2016). **Deep Reinforcement Learning with Double Q-learning.** *AAAI*.
3. Wang, Z., et al. (2016). **Dueling Network Architectures for Deep Reinforcement Learning.** *ICML*.
4. Schulman, J., et al. (2017). **Proximal Policy Optimization Algorithms.** *arXiv:1707.06347*.
5. Schulman, J., et al. (2015). **High-Dimensional Continuous Control Using Generalized Advantage Estimation.** *ICLR*.
6. [pret/pokered](https://github.com/pret/pokered) – Pokémon Red disassembly (memory map reference).
7. [Baekalfen/PyBoy](https://github.com/Baekalfen/PyBoy) – Python Game Boy emulator.

---

*MIT License © 2025 Jay Yala*
