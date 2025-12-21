#!/usr/bin/env bash
# train.sh – train a DQN or PPO agent.
#
# Usage:
#   ./scripts/train.sh dqn --rom roms/PokemonRed.gb
#   ./scripts/train.sh ppo --rom roms/PokemonRed.gb --steps 1000000
set -euo pipefail

ALGO="${1:-dqn}"
shift || true

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

source .venv/bin/activate 2>/dev/null || true

echo "Training ${ALGO^^} agent …"
pokemon-agent train "$ALGO" \
  --tensorboard \
  "$@"
