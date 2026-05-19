"""
Apple MLX inference backend for TRELLIS.2.

Status: experimental. Inference-only (image -> 3D mesh + per-vertex PBR voxel attrs).
The CUDA path under `trellis2/` is the source of truth for training. This package is
a faithful translation focused on running on Apple Silicon, accepting performance
trade-offs on sparse operations.

See docs/MLX_MIGRATION.md for the full design.
"""

from . import ops  # noqa: F401
from . import modules  # noqa: F401
from . import models  # noqa: F401
from . import pipelines  # noqa: F401
from . import io  # noqa: F401
from . import geometry  # noqa: F401

__all__ = ["ops", "modules", "models", "pipelines", "io", "geometry"]
__version__ = "0.1.0-dev"
