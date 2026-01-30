"""
FlashSVD Kernels

Thin wrapper to re-export Triton kernel implementations from src.kernels.
This provides stable import paths: from flashsvd.kernels import flashsvdattn
"""

import sys
from pathlib import Path

# Ensure src package is importable (needed when running from installed entry points)
_repo_root = Path(__file__).parent.parent.parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

# Import from src.kernels package using absolute imports
from src.kernels import flash_attn_triton
from src.kernels import flashsvdattn
from src.kernels import flashsvdffnv1
from src.kernels import flashsvdffnv2

# Re-export all public members
from src.kernels.flash_attn_triton import *  # noqa: F401, F403
from src.kernels.flashsvdattn import *  # noqa: F401, F403
from src.kernels.flashsvdffnv1 import *  # noqa: F401, F403
from src.kernels.flashsvdffnv2 import *  # noqa: F401, F403

__all__ = [
    "flash_attn_triton",
    "flashsvdattn",
    "flashsvdffnv1",
    "flashsvdffnv2",
]
