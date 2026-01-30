"""
FlashSVD: Memory-Efficient Inference with Streaming for Low-Rank Models

This package provides SVD-based compression methods and rank-aware streaming
kernels for efficient LLM inference.
"""

__version__ = "0.1.0"

# Re-export submodules for convenient access
from . import kernels
from . import utils
from . import io

__all__ = [
    "__version__",
    "kernels",
    "utils",
    "io",
]
