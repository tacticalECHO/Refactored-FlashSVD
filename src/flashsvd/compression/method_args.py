"""
Method-Specific Arguments for Compression

M7 Phase 2.2: Unified structure to pass method-specific args to compression functions,
eliminating temporary files and TypeError issues.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Any


@dataclass
class CommonArgs:
    """Common arguments for all compression methods."""
    build_only: bool = False
    device: str = "cuda"


@dataclass
class AdaArgs:
    """
    Arguments for AdaSVD compression.

    M7 Phase 2.2: ranks_dict takes priority over ranks_json_path.
    Loader should always provide ranks_dict (from compression_info.json).
    """
    ranks_dict: Optional[Dict[str, int]] = None  # Per-operation ranks (PRIORITY)
    ranks_json_path: Optional[str] = None  # Path to ranks.json (fallback)
    ffn_kernel: str = "v1"  # FFN kernel variant: v1 or v2
    strict: bool = True  # Fail loudly if rank missing for any Linear


@dataclass
class FWArgs:
    """Arguments for Fisher-Weighted SVD compression."""
    calib_samples: Optional[int] = 128
    calib_task: Optional[str] = None
    fisher_weights: Optional[Any] = None  # Pre-computed Fisher weights (future)
    model_dir: Optional[str] = None  # Model directory for calibration


@dataclass
class WhitenArgs:
    """Arguments for Whitening/DRONE compression."""
    calib_samples: Optional[int] = 128
    calib_task: Optional[str] = None
    model_dir: Optional[str] = None  # Model directory for calibration


@dataclass
class StandardArgs:
    """Arguments for Standard SVD compression (minimal)."""
    pass


def build_method_args(method: str, **kwargs) -> Dict[str, Any]:
    """
    Build method-specific arguments from kwargs.

    M7 Phase 2.2: Filters kwargs to only include relevant args for each method,
    preventing TypeError from unexpected keyword arguments.

    Args:
        method: Compression method (normalized: standard, fw, ada, whiten)
        **kwargs: All kwargs from CLI or loader

    Returns:
        Filtered dict with only method-relevant arguments

    Example:
        >>> args = build_method_args("ada", ranks_dict={...}, build_only=True, calib_samples=128)
        >>> # Returns only: {"ranks_dict": {...}, "build_only": True, "ffn_kernel": "v1", "strict": True}
    """
    common_keys = {"build_only", "device"}

    if method == "ada":
        method_keys = {"ranks_dict", "ranks_json_path", "ffn_kernel", "strict", "model_dir"}
        # Set defaults
        filtered = {
            "ffn_kernel": kwargs.get("ffn_kernel", "v1"),
            "strict": kwargs.get("strict", True),
        }
    elif method == "fw":
        method_keys = {"calib_samples", "calib_task", "fisher_weights", "model_dir"}
        filtered = {
            "calib_samples": kwargs.get("calib_samples", 128),
        }
    elif method == "whiten":
        method_keys = {"calib_samples", "calib_task", "model_dir"}
        filtered = {
            "calib_samples": kwargs.get("calib_samples", 128),
        }
    elif method == "standard":
        method_keys = set()
        filtered = {}
    else:
        raise ValueError(f"Unknown method: {method}")

    # Add common args
    for key in common_keys:
        if key in kwargs:
            filtered[key] = kwargs[key]

    # Add method-specific args
    for key in method_keys:
        if key in kwargs and kwargs[key] is not None:
            filtered[key] = kwargs[key]

    return filtered
