"""
FlashSVD Utils

Thin wrapper to re-export utility modules from src.utils.
This provides stable import paths: from flashsvd.utils import FlashSVDBlocks
"""

import sys
from pathlib import Path

# Ensure src package is importable (needed when running from installed entry points)
_repo_root = Path(__file__).parent.parent.parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

# Import from src.utils package using absolute imports
from src.utils import SVDBlocks
from src.utils import FlashSVDBlocks
from src.utils import fwsvd
from src.utils import svd_helpers
from src.utils import metrics

# Re-export all public members
from src.utils.SVDBlocks import *  # noqa: F401, F403
from src.utils.FlashSVDBlocks import *  # noqa: F401, F403
from src.utils.fwsvd import *  # noqa: F401, F403
from src.utils.svd_helpers import *  # noqa: F401, F403
from src.utils.metrics import *  # noqa: F401, F403

__all__ = [
    "SVDBlocks",
    "FlashSVDBlocks",
    "fwsvd",
    "svd_helpers",
    "metrics",
]
