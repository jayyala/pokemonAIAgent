"""Mock emulator for testing and CI.

Returns deterministic, scripted memory values so that game-state parsing,
reward shaping, and agent logic can all be unit-tested without a real ROM.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .base import Emulator, EmulatorConfig


@dataclass
class MockEmulatorConfig(EmulatorConfig):
    """Configuration knobs for the mock backend."""

    # Memory overrides: address → value returned by read_memory().
    # Values not in this dict fall back to a deterministic counter.
    memory_map: dict[int, int] = field(default_factory=dict)

    # When True, HP drains by 1 each step to trigger episode termination.
    drain_hp: bool = False
    initial_hp: int = 100
    max_hp: int = 100


class MockEmulator(Emulator):
    """Scriptable in-memory emulator used in tests and CI pipelines.

    The mock emulator uses a simple counter-based scheme for unknown
    addresses so that ``read_memory`` always returns a stable byte value.
    """

    # Pokémon Red memory addresses (same as real game)
    _ADDR_BATTLE   = 0xD057
    _ADDR_MAP_ID   = 0xD35E
    _ADDR_HP_HIGH  = 0xD16B
    _ADDR_HP_LOW   = 0xD16C
    _ADDR_MHP_HIGH = 0xD16D
    _ADDR_MHP_LOW  = 0xD16E
    _ADDR_BADGES   = 0xD356
    _ADDR_PARTY_SZ = 0xD163

    def __init__(self, config: Optional[MockEmulatorConfig] = None) -> None:
        cfg = config or MockEmulatorConfig()
        super().__init__(cfg)
        self._cfg: MockEmulatorConfig = cfg
        self._hp: int = cfg.initial_hp
        self._action_history: list[int] = []

    # ------------------------------------------------------------------
    # Emulator interface
    # ------------------------------------------------------------------

    def reset(self) -> None:
        self._hp = self._cfg.initial_hp
        self._step_count = 0
        self._action_history.clear()

    def step(self, action: int) -> None:
        self._action_history.append(action)
        if self._cfg.drain_hp:
            self._hp = max(0, self._hp - 1)
        self._step_count += 1

    def read_memory(self, address: int) -> int:
        # Allow explicit overrides first.
        if address in self._cfg.memory_map:
            return self._cfg.memory_map[address]

        hp = max(0, self._hp)
        if address == self._ADDR_BATTLE:
            return 1  # always in battle for simplicity
        if address == self._ADDR_MAP_ID:
            return (self._step_count // 100) & 0xFF
        if address == self._ADDR_HP_HIGH:
            return (hp >> 8) & 0xFF
        if address == self._ADDR_HP_LOW:
            return hp & 0xFF
        if address == self._ADDR_MHP_HIGH:
            return (self._cfg.max_hp >> 8) & 0xFF
        if address == self._ADDR_MHP_LOW:
            return self._cfg.max_hp & 0xFF
        if address == self._ADDR_BADGES:
            return min(self._step_count // 500, 8)
        if address == self._ADDR_PARTY_SZ:
            return 1

        # Fallback: deterministic hash of address + step count
        return (address ^ self._step_count) & 0xFF

    def screen_rgb(self) -> np.ndarray:
        """Return a blank 144×160 screen filled with a grey gradient."""
        frame = np.full((144, 160, 3), fill_value=128, dtype=np.uint8)
        # Overlay step count as a faint brightness pattern.
        frame[:, :, 0] = (self._step_count % 256)
        return frame

    def close(self) -> None:
        pass  # nothing to release

    # ------------------------------------------------------------------
    # Test helpers
    # ------------------------------------------------------------------

    @property
    def action_history(self) -> list[int]:
        return list(self._action_history)

    def set_hp(self, hp: int) -> None:
        self._hp = hp
