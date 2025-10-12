from .base import Emulator, EmulatorConfig
from .mock_emulator import MockEmulator

__all__ = ["Emulator", "EmulatorConfig", "MockEmulator"]

try:
    from .pyboy_emulator import PyBoyEmulator, PyBoyConfig

    __all__ += ["PyBoyEmulator", "PyBoyConfig"]
except ImportError:
    pass
