#!/usr/bin/env bash
# eval.sh – evaluate a saved checkpoint.
#
# Usage:
#   ./scripts/eval.sh checkpoints/dqn/dqn_step00500000.pt --rom roms/PokemonRed.gb
set -euo pipefail

CHECKPOINT="${1:?Usage: eval.sh <checkpoint.pt> [--rom ROM] [--window]}"
shift

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

source .venv/bin/activate 2>/dev/null || true

echo "Evaluating checkpoint: $CHECKPOINT"
pokemon-agent eval --checkpoint "$CHECKPOINT" "$@"
