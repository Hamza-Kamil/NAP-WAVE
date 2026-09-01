"""
NAP-WAVE: A Neural Adaptive Plane-Wave Architecture for High-Frequency Helmholtz Problems.
Developer: Hamza Kamil | Version: 0.0.1 | Date of release: 09-01-2026
"""

from importlib.metadata import version as _pkg_version, PackageNotFoundError

from .trainer import Trainer
from .archs import build_architecture
from .loss import build_optimizer, build_lr_schedule
from .utils import section, kv_block

__developer__ = "Hamza Kamil"
__release_date__ = "2026-09-01"

try:
    __version__ = _pkg_version("nap_wave")
except PackageNotFoundError:
    __version__ = "0.0.1"

print(section("NAP-WAVE"))
print(kv_block([
    ("version", __version__),
    ("release date", __release_date__),
    ("developer", __developer__),
]))

__all__ = [
    "build_architecture",
    "build_optimizer",
    "build_lr_schedule",
    "Trainer",
    "__version__",
    "__developer__",
    "__release_date__",
]