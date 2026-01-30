"""
Standard SVD compression for BERT models.

Extracts and reuses the proven SVD decomposition logic from existing experiments.
"""

import torch
import torch.nn as nn
from typing import Dict

# Import from flashsvd package (not src.*)
from flashsvd.utils import svd_helpers, SVDBlocks


def compress_bert_standard_svd(
    model: nn.Module,
    ranks: Dict[str, int],
    device: str = "cuda",
    build_only: bool = False,
    **kwargs
) -> nn.Module:
    """
    Apply standard SVD compression to BERT model.

    Args:
        model: BERT model (e.g., BertForSequenceClassification)
        ranks: Dict with keys "attn", "ffn", "wo" specifying truncation ranks
        device: Device to perform compression on

    Returns:
        Compressed model with SVDBlocks replacing original layers

    Example:
        >>> model = BertForSequenceClassification.from_pretrained("bert-base-uncased")
        >>> ranks = {"attn": 64, "ffn": 256, "wo": 256}
        >>> compressed = compress_bert_standard_svd(model, ranks, "cuda")
    """
    model = model.to(device).eval()

    # Extract rank values
    rank_attn = ranks.get("attn", 64)
    rank_ffn = ranks.get("ffn", 256)
    rank_wo = ranks.get("wo", 256)

    # M7 Phase 2.1: Handle build_only mode
    if build_only:
        print("  [StandardSVD] build_only=True: SKIP SVD decomposition")

    # Build SVD decomposition helpers (only if not build_only)
    if not build_only:
        svd_per_head, svd_low_rank = svd_helpers.build_plain_svd_helpers(model)
    else:
        # Dummy helpers (not used in build_only mode)
        svd_per_head, svd_low_rank = None, None

    # Replace each encoder layer with SVD block
    # This follows the exact pattern from BERT/profile_svd.py
    for i, layer in enumerate(model.bert.encoder.layer):
        # Create SVD block (performs decomposition unless build_only)
        svd_block = SVDBlocks.BertSVDBlock(
            layer,
            rank_attn=rank_attn,
            rank_ff=rank_ffn,
            svd_per_head=svd_per_head,
            svd_low_rank=svd_low_rank,
            rank_wo=rank_wo,
            build_only=build_only,
        )

        # Wrap with LayerShim for HuggingFace compatibility
        shimmed_block = svd_helpers.BertLayerShim(svd_block)

        # Replace original layer
        model.bert.encoder.layer[i] = shimmed_block.to(device).eval()

    # Clean up helper functions to free memory
    if not build_only:
        del svd_per_head, svd_low_rank

    # Clean up any cached helpers in layers
    for layer in model.bert.encoder.layer:
        if hasattr(layer, 'svd_per_head'):
            del layer.svd_per_head
        if hasattr(layer, 'svd_low_rank'):
            del layer.svd_low_rank

    torch.cuda.empty_cache()

    return model
