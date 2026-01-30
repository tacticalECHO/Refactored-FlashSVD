"""
Metadata utilities for compression tracking.
M7 Phase 2: Schema v2 support for stable checkpoint loading.
"""

import subprocess
import datetime
from typing import Optional, Dict, Any


def get_git_commit() -> Optional[str]:
    """
    Get current git commit hash.

    Returns:
        Commit hash string, or None if not in a git repo
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
        return None


# M7 Phase 2: Block implementation mapping (arch, method) -> block_impl
# Note: This maps to the ACTUAL block classes used in compression functions
BLOCK_IMPL_MAP = {
    ("bert", "standard"): "BertSVDBlock",  # From SVDBlocks.py (not FlashSVDBlocks)
    ("bert", "fw"): "BertFlashFWSVDBlock",  # From FlashSVDBlocks.py
    ("bert", "ada"): "BertAdaSVDBlock",  # From FlashSVDBlocks.py
    ("bert", "whiten"): "BertWhitenSVDBlock",  # From FlashSVDBlocks.py

    ("roberta", "standard"): "RobertaSVDBlock",  # From SVDBlocks.py
    ("roberta", "fw"): "RobertaFlashFWSVDBlock",  # From FlashSVDBlocks.py
    ("roberta", "ada"): "RobertaAdaSVDBlock",  # From FlashSVDBlocks.py
    # Note: roberta + whiten not supported (raises NotImplementedError)
}


def get_block_impl(arch: str, method: str) -> Optional[str]:
    """
    Get block implementation class name for given architecture and method.

    Args:
        arch: Architecture (bert, roberta, modernbert)
        method: Method (already normalized: standard, fw, ada, whiten)

    Returns:
        Block class name (e.g., "BertAdaSVDBlock"), or None if not supported
    """
    return BLOCK_IMPL_MAP.get((arch, method))


def create_compression_info(
    method: str,
    ranks: dict,
    base_model: str,
    task: str,
    flashsvd_version: str,
    # M7 Phase 2: New v2 schema fields
    arch: Optional[str] = None,
    block_impl: Optional[str] = None,
    adasvd_ranks: Optional[Dict[str, int]] = None,
    ranks_json_path: Optional[str] = None,
    calib_samples: Optional[int] = None,
    calib_task: Optional[str] = None,
    ffn_kernel: Optional[str] = None,
    schema_version: int = 2,
) -> dict:
    """
    Create compression_info.json metadata with schema v2 support.

    M7 Phase 2: Adds schema_version, arch, block_impl, and method-specific metadata
    to enable stable checkpoint loading without re-compression.

    Args:
        method: Compression method name (normalized)
        ranks: Rank configuration dict (attn/ffn/wo keys for standard/fw/whiten)
        base_model: Base model name/path
        task: Task name (e.g., "sst2")
        flashsvd_version: FlashSVD package version

        # Schema v2 fields (M7 Phase 2)
        arch: Architecture (bert, roberta, modernbert)
        block_impl: Block class name (e.g., "BertAdaSVDBlock")
        adasvd_ranks: Full per-operation ranks dict (for ada method)
        ranks_json_path: Path to ranks.json (for ada method, optional)
        calib_samples: Calibration samples (for fw/whiten)
        calib_task: Calibration task (for fw/whiten)
        ffn_kernel: FFN kernel variant (for ada: v1 or v2)
        schema_version: Schema version (default: 2)

    Returns:
        Metadata dict ready for JSON serialization
    """
    # Base metadata (v1 compatible)
    metadata = {
        "method": method,
        "ranks": ranks,
        "base_model": base_model,
        "task": task,
        "timestamp": datetime.datetime.now().isoformat(),
        "git_commit": get_git_commit(),
        "flashsvd_version": flashsvd_version,
    }

    # Schema v2 fields (M7 Phase 2)
    if schema_version >= 2:
        metadata["schema_version"] = schema_version

        # Architecture and block implementation (required for v2 loader)
        if arch is not None:
            metadata["arch"] = arch
        if block_impl is not None:
            metadata["block_impl"] = block_impl
            metadata["block_impl_version"] = "1"  # For future upgrades

        # Method-specific metadata
        if method == "ada":
            # AdaSVD: Must save full per-op ranks dict
            if adasvd_ranks is not None:
                metadata["adasvd_ranks"] = adasvd_ranks
            if ranks_json_path is not None:
                metadata["ranks_json_path"] = ranks_json_path
            if ffn_kernel is not None:
                metadata["ffn_kernel"] = ffn_kernel

        elif method in ["fw", "whiten"]:
            # FWSVD/Whiten: Calibration settings
            if calib_samples is not None:
                metadata["calib_samples"] = calib_samples
            if calib_task is not None:
                metadata["calib_task"] = calib_task

    return metadata
