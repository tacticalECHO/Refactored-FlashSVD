"""
Compression Method Registry

Maps (architecture, method) to compression functions.
Provides architecture detection and method normalization.
"""

from typing import Callable, Tuple
import torch.nn as nn


# Supported architectures and methods
SUPPORTED_ARCHES = ["bert", "roberta", "modernbert"]
SUPPORTED_METHODS = ["standard", "fw", "ada", "whiten"]


def normalize_method(method: str) -> str:
    """
    Normalize method name to lowercase canonical form.

    Accepts aliases:
    - standard, Standard
    - fw, FW, fwsvd, FWSVD
    - ada, Ada, AdaSVD
    - whiten, Whiten, WHITEN

    Returns:
        Canonical method name (lowercase): standard, fw, ada, whiten
    """
    method_lower = method.lower()

    # Normalize aliases
    if method_lower in ["standard"]:
        return "standard"
    elif method_lower in ["fw", "fwsvd"]:
        return "fw"
    elif method_lower in ["ada", "adasvd"]:
        return "ada"
    elif method_lower in ["whiten", "drone"]:
        return "whiten"
    else:
        raise ValueError(
            f"Unknown method: {method}. "
            f"Supported: standard, fw (fwsvd), ada (adasvd), whiten (drone)"
        )


def detect_architecture(model: nn.Module) -> str:
    """
    Detect model architecture from attributes (not name heuristics).

    Args:
        model: HuggingFace model

    Returns:
        Architecture name: bert, roberta, or modernbert

    Raises:
        ValueError: If architecture cannot be determined
    """
    # Check for BERT
    if hasattr(model, 'bert') and hasattr(model.bert, 'encoder'):
        return "bert"

    # Check for RoBERTa
    if hasattr(model, 'roberta') and hasattr(model.roberta, 'encoder'):
        return "roberta"

    # Check for ModernBERT
    if hasattr(model, 'config'):
        model_type = getattr(model.config, 'model_type', '').lower()
        if 'modernbert' in model_type:
            return "modernbert"

    raise ValueError(
        f"Unknown architecture. Model must have .bert or .roberta attribute.\n"
        f"Detected attributes: {[k for k in dir(model) if not k.startswith('_')][:10]}"
    )


def get_compression_fn(arch: str, method: str) -> Callable:
    """
    Get compression function from registry.

    Args:
        arch: Architecture (bert, roberta, modernbert)
        method: Method (already normalized to lowercase)

    Returns:
        Compression function

    Raises:
        ValueError: If combination not supported
        NotImplementedError: If ModernBERT (routing to original scripts)
    """
    # ModernBERT always errors (not integrated in M6)
    if arch == "modernbert":
        return _compress_modernbert_not_supported

    # Import compression functions (lazy import to avoid circular deps)
    from .standard_svd import compress_bert_standard_svd
    from .roberta_svd import compress_roberta_standard_svd
    from .fwsvd import compress_bert_fwsvd, compress_roberta_fwsvd
    from .adasvd import compress_bert_adasvd, compress_roberta_adasvd
    from .whiten import compress_bert_whiten, compress_roberta_whiten

    # Registry mapping
    COMPRESSION_REGISTRY = {
        ("bert", "standard"): compress_bert_standard_svd,
        ("bert", "fw"): compress_bert_fwsvd,
        ("bert", "ada"): compress_bert_adasvd,
        ("bert", "whiten"): compress_bert_whiten,

        ("roberta", "standard"): compress_roberta_standard_svd,
        ("roberta", "fw"): compress_roberta_fwsvd,
        ("roberta", "ada"): compress_roberta_adasvd,
        ("roberta", "whiten"): compress_roberta_whiten,
    }

    key = (arch, method)
    compress_fn = COMPRESSION_REGISTRY.get(key)

    if compress_fn is None:
        raise ValueError(
            f"Unsupported combination: arch={arch}, method={method}\n"
            f"Supported: {list(COMPRESSION_REGISTRY.keys())}"
        )

    return compress_fn


def _compress_modernbert_not_supported(model, ranks, device, method="standard", variant="mask", **kwargs):
    """
    ModernBERT compression not integrated in M6.
    Provides clear error with paths to original scripts.
    """
    script_map = {
        "standard": {
            "mask": "cd ModernBERT/BERT_MASK && python run_modernbert_flashsvd.py",
            "fwmask": "cd ModernBERT/BERT_FWMASK && python run_modernbert_flashfwsvd.py",
            "long": "cd ModernBERT/BERT_LONG && python profile_imdb.py"
        },
        "fw": {
            "mask": "cd ModernBERT/BERT_MASK && python run_modernbert_flashsvd.py",
            "fwmask": "cd ModernBERT/BERT_FWMASK && python run_modernbert_flashfwsvd.py",
            "long": "cd ModernBERT/BERT_LONG && python profile_imdb.py"
        }
    }

    script = script_map.get(method, {}).get(variant, "cd ModernBERT/BERT_MASK && python run_modernbert_flashsvd.py")

    raise NotImplementedError(
        f"ModernBERT compression not yet integrated into CLI (M6 scope: routing only).\n"
        f"\n"
        f"To compress ModernBERT with method='{method}', variant='{variant}', use original script:\n"
        f"  {script}\n"
        f"\n"
        f"Available variants:\n"
        f"  - mask:   ModernBERT/BERT_MASK/\n"
        f"  - fwmask: ModernBERT/BERT_FWMASK/ (Fisher-Weighted)\n"
        f"  - long:   ModernBERT/BERT_LONG/ (long context)"
    )
