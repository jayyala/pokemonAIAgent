"""Game-state parsing from emulator memory.

All Game Boy memory addresses are taken from the Pokémon Red disassembly
project (pret/pokered).  Values are read byte-by-byte from the emulator's
memory bus to reconstruct structured game state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..emulator.base import Emulator


class MemoryAddresses:
    """Known Pokémon Red RAM addresses (wram bank 0).

    Source: https://github.com/pret/pokered/blob/master/ram/wram.asm
    """

    # Battle / overworld mode
    BATTLE_TYPE      = 0xD057  # 0 = not in battle, 1 = wild, 2 = trainer
    BATTLE_TURN      = 0xD059  # whose turn it is

    # Map / location
    MAP_ID           = 0xD35E  # current map index
    X_COORD          = 0xD362  # player X tile
    Y_COORD          = 0xD361  # player Y tile

    # Party
    PARTY_COUNT      = 0xD163  # number of Pokémon in party (0-6)
    PARTY_SPECIES_1  = 0xD164  # species ID of first party member

    # First party member HP (big-endian 16-bit)
    HP_HIGH          = 0xD16B
    HP_LOW           = 0xD16C
    MAX_HP_HIGH      = 0xD16D
    MAX_HP_LOW       = 0xD16E

    # First party member level
    LEVEL_1          = 0xD18C

    # First party member status condition bitmask
    STATUS_1         = 0xD16F  # bit 2=PSN, 3=BRN, 4=FRZ, 5=PAR

    # Badges obtained bitmask (bit N = badge N+1)
    BADGES           = 0xD356

    # Pokédex seen / owned counts
    POKEDEX_OWNED    = 0xD2F7  # number of species owned (0-151)
    POKEDEX_SEEN     = 0xD30A  # number of species seen (0-151)

    # Money (BCD, 3 bytes)
    MONEY_HIGH       = 0xD347
    MONEY_MID        = 0xD348
    MONEY_LOW        = 0xD349


@dataclass
class GameState:
    """Structured snapshot of the game at a single timestep."""

    # Location
    map_id:  int = 0
    x_coord: int = 0
    y_coord: int = 0

    # Battle
    battle_type: int = 0  # 0=none, 1=wild, 2=trainer

    # Party (lead Pokémon)
    party_count: int = 0
    species:     int = 0
    level:       int = 1
    current_hp:  int = 0
    max_hp:      int = 1
    status:      int = 0   # status condition bitmask

    # Progression
    badges:        int = 0  # bitmask of 8 badges
    pokedex_seen:  int = 0
    pokedex_owned: int = 0
    money:         int = 0

    # Derived
    @property
    def hp_fraction(self) -> float:
        return self.current_hp / max(self.max_hp, 1)

    @property
    def badge_count(self) -> int:
        return bin(self.badges).count("1")

    @property
    def in_battle(self) -> bool:
        return self.battle_type > 0

    @property
    def is_fainted(self) -> bool:
        return self.current_hp <= 0

    def as_vector(self) -> "list[float]":
        """Return a fixed-length float feature vector for the neural network."""
        return [
            float(self.map_id) / 255.0,
            float(self.x_coord) / 255.0,
            float(self.y_coord) / 255.0,
            float(self.battle_type) / 2.0,
            self.hp_fraction,
            float(self.level) / 100.0,
            float(self.badge_count) / 8.0,
            float(self.pokedex_seen) / 151.0,
            float(self.pokedex_owned) / 151.0,
            float(self.in_battle),
        ]


def _bcd_to_int(high: int, mid: int, low: int) -> int:
    """Convert three BCD-encoded bytes to an integer."""
    def nibbles(b: int) -> tuple[int, int]:
        return (b >> 4) & 0xF, b & 0xF

    h1, h2 = nibbles(high)
    m1, m2 = nibbles(mid)
    l1, l2 = nibbles(low)
    return (
        h1 * 100_000 + h2 * 10_000
        + m1 * 1_000  + m2 * 100
        + l1 * 10     + l2
    )


def read_state(emulator: "Emulator") -> GameState:
    """Parse the current game state from emulator memory."""
    A = MemoryAddresses

    hp      = emulator.read_u16_be(A.HP_HIGH, A.HP_LOW)
    max_hp  = max(emulator.read_u16_be(A.MAX_HP_HIGH, A.MAX_HP_LOW), 1)
    money   = _bcd_to_int(
        emulator.read_memory(A.MONEY_HIGH),
        emulator.read_memory(A.MONEY_MID),
        emulator.read_memory(A.MONEY_LOW),
    )

    return GameState(
        map_id       = emulator.read_memory(A.MAP_ID) & 0xFF,
        x_coord      = emulator.read_memory(A.X_COORD),
        y_coord      = emulator.read_memory(A.Y_COORD),
        battle_type  = emulator.read_memory(A.BATTLE_TYPE),
        party_count  = emulator.read_memory(A.PARTY_COUNT),
        species      = emulator.read_memory(A.PARTY_SPECIES_1),
        level        = max(emulator.read_memory(A.LEVEL_1), 1),
        current_hp   = hp,
        max_hp       = max_hp,
        status       = emulator.read_memory(A.STATUS_1),
        badges       = emulator.read_memory(A.BADGES),
        pokedex_seen  = emulator.read_memory(A.POKEDEX_SEEN),
        pokedex_owned = emulator.read_memory(A.POKEDEX_OWNED),
        money        = money,
    )
