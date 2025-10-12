"""Abstract emulator interface.

All concrete emulator backends must implement this interface so that agents
and training loops remain backend-agnostic.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class EmulatorConfig:
    """Base configuration for any emulator backend."""

    rom_path: Optional[str] = None
    headless: bool = True
    speed: int = 0  # 0 = unlimited
    frame_skip: int = 24  # advance N frames per agent step


class Emulator(ABC):
    """Abstract base class for Game Boy emulator backends.

    Concrete subclasses wrap either a real emulator (PyBoy) or a mock
    implementation used during testing and CI runs.
    """

    def __init__(self, config: EmulatorConfig) -> None:
        self.config = config
        self._step_count: int = 0

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    @abstractmethod
    def reset(self) -> None:
        """Reset the emulator to the beginning of a new episode."""

    @abstractmethod
    def step(self, action: int) -> None:
        """Advance the emulator by one agent step.

        Args:
            action: Integer action index (see ``actions.py``).
        """

    @abstractmethod
    def read_memory(self, address: int) -> int:
        """Read a single byte from the Game Boy memory bus.

        Args:
            address: 16-bit memory address.

        Returns:
            Byte value 0-255.
        """

    @abstractmethod
    def screen_rgb(self) -> "np.ndarray":  # noqa: F821
        """Return the current screen as an (H, W, 3) uint8 array."""

    @abstractmethod
    def close(self) -> None:
        """Cleanly shut down the emulator and release resources."""

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def read_memory_range(self, start: int, length: int) -> list[int]:
        """Read *length* consecutive bytes starting at *start*."""
        return [self.read_memory(start + i) for i in range(length)]

    def read_u16_be(self, high_addr: int, low_addr: int) -> int:
        """Read a big-endian 16-bit value split across two addresses."""
        return (self.read_memory(high_addr) << 8) | self.read_memory(low_addr)

    @property
    def step_count(self) -> int:
        return self._step_count

    def __enter__(self) -> "Emulator":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
