#!/usr/bin/env bash
# quickstart.sh – set up the environment and run a short heuristic rollout.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "=== Pokémon Red Agent – Quickstart ==="

# 1. Create virtual environment
if [ ! -d ".venv" ]; then
  echo "→ Creating virtual environment …"
  python3 -m venv .venv
fi

source .venv/bin/activate

# 2. Install dependencies
echo "→ Installing dependencies …"
pip install -q -e ".[emulator,ml,logging]"

# 3. Run a short heuristic rollout (mock emulator – no ROM required)
echo ""
echo "→ Running 500-step heuristic rollout (mock emulator) …"
python -m pokemon_agent.main rollout \
  --agent heuristic \
  --max-steps 500 \
  --seed 42

echo ""
echo "Done!  To train with a real ROM:"
echo "  pokemon-agent train dqn --rom roms/PokemonRed.gb --steps 500000"
