"""PyBoy-backed Game Boy emulator.

Requires ``pip install pyboy``.  Will raise ``ImportError`` at import time
when PyBoy is not installed; the rest of the codebase falls back to
``MockEmulator`` in that case.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

try:
    from pyboy import PyBoy
    from pyboy.utils import WindowEvent
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "PyBoy is required for the real emulator backend.  "
        "Install it with: pip install pyboy"
    ) from exc

from .base import Emulator, EmulatorConfig


# Map integer action indices to PyBoy WindowEvent pairs (press / release).
_ACTION_MAP: dict[int, tuple[WindowEvent, WindowEvent]] = {
    0: (WindowEvent.PASS, WindowEvent.PASS),                          # NOOP
    1: (WindowEvent.PRESS_ARROW_UP, WindowEvent.RELEASE_ARROW_UP),
    2: (WindowEvent.PRESS_ARROW_DOWN, WindowEvent.RELEASE_ARROW_DOWN),
    3: (WindowEvent.PRESS_ARROW_LEFT, WindowEvent.RELEASE_ARROW_LEFT),
    4: (WindowEvent.PRESS_ARROW_RIGHT, WindowEvent.RELEASE_ARROW_RIGHT),
    5: (WindowEvent.PRESS_BUTTON_A, WindowEvent.RELEASE_BUTTON_A),
    6: (WindowEvent.PRESS_BUTTON_B, WindowEvent.RELEASE_BUTTON_B),
    7: (WindowEvent.PRESS_BUTTON_START, WindowEvent.RELEASE_BUTTON_START),
    8: (WindowEvent.PRESS_BUTTON_SELECT, WindowEvent.RELEASE_BUTTON_SELECT),
}


@dataclass
class PyBoyConfig(EmulatorConfig):
    """Configuration specific to the PyBoy backend."""

    window_type: str = "headless"   # "headless" | "SDL2" | "OpenGL"
    debug: bool = False
    sound: bool = False


class PyBoyEmulator(Emulator):
    """Wraps a PyBoy instance to satisfy the :class:`Emulator` interface."""

    def __init__(self, config: PyBoyConfig) -> None:
        super().__init__(config)
        if config.rom_path is None:
            raise ValueError("PyBoyConfig.rom_path must be set to a valid ROM file.")

        self._pyboy = PyBoy(
            config.rom_path,
            window_type=config.window_type,
            sound=config.sound,
            debug=config.debug,
        )
        self._pyboy.set_emulation_speed(config.speed)
        self._screen = self._pyboy.botsupport_manager().screen()

    # ------------------------------------------------------------------
    # Emulator interface
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reload the ROM from the beginning.

        PyBoy does not expose a true soft-reset API, so we close and
        re-open the emulator.  For training speed, consider loading a
        save-state instead.
        """
        rom_path = self.config.rom_path
        window_type = self.config.window_type  # type: ignore[attr-defined]
        self._pyboy.stop(save=False)
        cfg: PyBoyConfig = self.config  # type: ignore[assignment]
        self._pyboy = PyBoy(
            rom_path,
            window_type=window_type,
            sound=cfg.sound,
            debug=cfg.debug,
        )
        self._pyboy.set_emulation_speed(cfg.speed)
        self._screen = self._pyboy.botsupport_manager().screen()
        self._step_count = 0

    def step(self, action: int) -> None:
        press, release = _ACTION_MAP[action]
        self._pyboy.send_input(press)
        # Advance game engine for the first half of the frame skip.
        half = self.config.frame_skip // 2
        for _ in range(half):
            self._pyboy.tick()
        self._pyboy.send_input(release)
        for _ in range(self.config.frame_skip - half):
            self._pyboy.tick()
        self._step_count += 1

    def read_memory(self, address: int) -> int:
        return self._pyboy.get_memory_value(address)

    def screen_rgb(self) -> np.ndarray:
        return self._screen.screen_ndarray()  # (144, 160, 3) uint8

    def close(self) -> None:
        self._pyboy.stop(save=False)
