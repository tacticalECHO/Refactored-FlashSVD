"""
FlashSVD Compression Methods

Provides SVD-based compression implementations for model compression.
M6: Unified method selection with registry-based dispatch.
"""

from .standard_svd import compress_bert_standard_svd
from .roberta_svd import compress_roberta_standard_svd
from .fwsvd import compress_bert_fwsvd, compress_roberta_fwsvd
from .adasvd import compress_bert_adasvd, compress_roberta_adasvd
from .whiten import compress_bert_whiten, compress_roberta_whiten
from .registry import detect_architecture, normalize_method, get_compression_fn

__all__ = [
    "compress_model",
    "compress_bert_standard_svd",
    "compress_roberta_standard_svd",
    "compress_bert_fwsvd",
    "compress_roberta_fwsvd",
    "compress_bert_adasvd",
    "compress_roberta_adasvd",
    "compress_bert_whiten",
    "compress_roberta_whiten",
    "detect_architecture",
    "normalize_method",
]


def compress_model(model, method: str, ranks: dict, device: str = "cuda", **extra_args):
    """
    Unified compression interface (registry-based dispatcher).

    M6: Supports all methods (standard/fw/ada/whiten) for BERT/RoBERTa.
    ModernBERT routes to original scripts with clear error messages.

    M7 Phase 2.1: Added build_only mode - when True, creates model structure
    without performing SVD decomposition, Fisher calibration, or Whitening calibration.

    Args:
        model: HuggingFace model to compress
        method: Compression method (standard, fw, ada, whiten)
                Accepts aliases: FW/fwsvd, ada/AdaSVD, whiten/Whiten/DRONE
        ranks: Dict with keys {"attn": int, "ffn": int, "wo": int}
               For AdaSVD, this is overridden by ranks_json
        device: Device to use for compression
        **extra_args: Method-specific arguments:
            - build_only (bool): Only create structure, skip decomposition (M7 Phase 2.1)
            - calib_samples (int): Number of calibration samples (fw, whiten)
            - calib_task (str): GLUE task for calibration (fw, whiten)
            - ranks_json (str): Path to ranks.json (ada)
            - model_dir (str): Model directory path (fw, whiten, ada)
            - modernbert_variant (str): ModernBERT variant (mask/fwmask/long)

    Returns:
        Compressed model

    Raises:
        ValueError: For unknown architecture or method
        NotImplementedError: For ModernBERT (routing only in M6)
    """
    # Detect architecture
    arch = detect_architecture(model)

    # Normalize method name
    method_normalized = normalize_method(method)

    # Get compression function from registry
    compress_fn = get_compression_fn(arch, method_normalized)

    # Call compression function with extra_args (including build_only)
    return compress_fn(model, ranks, device, **extra_args)
