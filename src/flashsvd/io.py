"""
FlashSVD Model Loading

Functions to load compressed models with SVDBlocks structure preserved.
M7 Phase 2: Schema v2 support for stable checkpoint loading.
"""

import os
import json
import tempfile
import torch
from transformers import AutoModelForSequenceClassification, AutoConfig

# Import from flashsvd package
from flashsvd.compression import compress_model, normalize_method


def _load_v1_checkpoint(checkpoint_dir: str, device: str, compression_info: dict):
    """
    Load v1 checkpoint (backward compatibility).

    Uses compress_model() + load_state_dict() approach (original behavior).

    Args:
        checkpoint_dir: Path to checkpoint
        device: Device to load on
        compression_info: Metadata from compression_info.json

    Returns:
        Loaded model
    """
    method = compression_info["method"]
    ranks = compression_info["ranks"]
    base_model = compression_info["base_model"]
    task = compression_info.get("task", "sst2")

    print(f"Loading v1 checkpoint (backward compatibility):")
    print(f"  Method: {method}")
    print(f"  Ranks: {ranks}")
    print(f"  Base model: {base_model}")

    # Load base model config
    config_path = os.path.join(checkpoint_dir, "config.json")
    config = AutoConfig.from_pretrained(config_path)
    num_labels = config.num_labels
    problem_type = getattr(config, "problem_type", None)

    # Load base model architecture (without pretrained weights)
    print(f"\nLoading base model architecture from: {base_model}")
    base_config = AutoConfig.from_pretrained(base_model)
    base_config.num_labels = num_labels
    if problem_type:
        base_config.problem_type = problem_type

    model = AutoModelForSequenceClassification.from_config(base_config)

    # Apply compression (replaces layers with SVDBlocks)
    print(f"Applying {method} compression with ranks: {ranks}")
    model = compress_model(
        model=model,
        method=method,
        ranks=ranks,
        device=device
    )

    # Load compressed state dict
    state_dict_path = os.path.join(checkpoint_dir, "flashsvd_state_dict.pt")
    if not os.path.exists(state_dict_path):
        raise FileNotFoundError(
            f"flashsvd_state_dict.pt not found in {checkpoint_dir}. "
            f"Cannot restore compressed model weights."
        )

    print(f"Loading compressed weights from flashsvd_state_dict.pt")
    state_dict = torch.load(state_dict_path, map_location=device)
    model.load_state_dict(state_dict)

    return model


def _load_v2_checkpoint(checkpoint_dir: str, device: str, compression_info: dict):
    """
    Load v2 checkpoint with schema-based structure recovery.

    M7 Phase 2.1: Uses block_impl and method-specific metadata to construct
    the exact model structure, then loads state_dict.

    M7 Phase 2.2: CRITICAL - Forces build_only=True with hard assert.
    NO temporary files created (AdaSVD ranks_dict passed directly from compression_info).

    Args:
        checkpoint_dir: Path to checkpoint
        device: Device to load on
        compression_info: Metadata from compression_info.json (schema v2)

    Returns:
        Loaded model

    Raises:
        NotImplementedError: If architecture not supported
        ValueError: If required metadata missing
        RuntimeError: If build_only constraint violated
    """
    arch = compression_info.get("arch")
    method = compression_info["method"]
    method_normalized = normalize_method(method)
    block_impl = compression_info.get("block_impl")
    ranks = compression_info.get("ranks", {})
    base_model = compression_info["base_model"]

    print(f"Loading v2 checkpoint (schema-based):")
    print(f"  Architecture: {arch}")
    print(f"  Method: {method_normalized}")
    print(f"  Block: {block_impl}")
    print(f"  Ranks (summary): {ranks}")

    # Check for ModernBERT (not supported in CLI)
    if arch == "modernbert":
        raise NotImplementedError(
            f"ModernBERT checkpoint loading not supported in CLI.\n"
            f"This checkpoint was created with ModernBERT architecture.\n"
            f"Please use the original ModernBERT scripts to load:\n"
            f"  - ModernBERT/BERT_MASK/run_modernbert_flashsvd.py\n"
            f"  - ModernBERT/BERT_FWMASK/run_modernbert_flashfwsvd.py\n"
            f"  - ModernBERT/BERT_LONG/profile_imdb.py"
        )

    # Load base model config
    config_path = os.path.join(checkpoint_dir, "config.json")
    config = AutoConfig.from_pretrained(config_path)
    num_labels = config.num_labels
    problem_type = getattr(config, "problem_type", None)

    # Load base model architecture (without pretrained weights)
    print(f"\nLoading base model architecture from: {base_model}")
    base_config = AutoConfig.from_pretrained(base_model)
    base_config.num_labels = num_labels
    if problem_type:
        base_config.problem_type = problem_type

    model = AutoModelForSequenceClassification.from_config(base_config)

    # M7 Phase 2.2: Method-specific handling (NO TEMPORARY FILES)
    extra_args = {}

    if method_normalized == "ada":
        # AdaSVD: Pass ranks_dict directly (NO temporary ranks.json file)
        adasvd_ranks = compression_info.get("adasvd_ranks")
        if not adasvd_ranks:
            raise ValueError(
                f"AdaSVD checkpoint missing 'adasvd_ranks' in compression_info.json.\n"
                f"This checkpoint was created with an older version.\n"
                f"Please re-compress the model with the latest version to generate v2 checkpoint."
            )

        # M7 Phase 2.2: Direct dict passing (no temp file)
        extra_args["ranks_dict"] = adasvd_ranks  # Direct passing
        extra_args["ranks_json_path"] = None  # No file needed
        extra_args["ffn_kernel"] = compression_info.get("ffn_kernel", "v1")
        extra_args["strict"] = True

        print(f"  Per-op ranks: {len(adasvd_ranks)} operations (passed directly from compression_info)")

    elif method_normalized in ["fw", "whiten"]:
        # FWSVD/Whiten: Calibration settings (not needed for loading)
        # The checkpoint already contains calibrated weights
        pass

    # M7 Phase 2.2: HARD ASSERT - build_only MUST be True for v2 loader
    BUILD_ONLY = True
    if not BUILD_ONLY:
        raise RuntimeError(
            "CRITICAL BUG: v2 loader MUST use build_only=True.\n"
            "This is a hard constraint to prevent SVD decomposition during loading."
        )

    # Apply compression with build_only=True (no decomposition/calibration)
    print(f"Building model structure with {method_normalized} compression (build_only={BUILD_ONLY})...")
    result = compress_model(
        model=model,
        method=method_normalized,
        ranks=ranks,
        device=device,
        build_only=BUILD_ONLY,  # M7 Phase 2.2: HARD ASSERT True
        **extra_args
    )

    # M7 Phase 2.3: Handle tuple return for AdaSVD
    if method_normalized == "ada" and isinstance(result, tuple):
        model, _ = result  # Discard actual_ranks (already in compression_info)
    else:
        model = result

    # M7 Phase 2.2: Verify no temporary files created
    # (No cleanup needed - we never created temp files)

    # Load compressed state dict
    state_dict_path = os.path.join(checkpoint_dir, "flashsvd_state_dict.pt")
    if not os.path.exists(state_dict_path):
        raise FileNotFoundError(
            f"flashsvd_state_dict.pt not found in {checkpoint_dir}. "
            f"Cannot restore compressed model weights."
        )

    print(f"Loading compressed weights from flashsvd_state_dict.pt")
    state_dict = torch.load(state_dict_path, map_location=device)
    model.load_state_dict(state_dict)

    return model


def load_compressed(checkpoint_dir: str, device: str = "cuda"):
    """
    Load a compressed model with SVDBlocks structure preserved.

    M7 Phase 2: Supports both v1 (backward compatibility) and v2 (schema-based) checkpoints.

    This function:
    1. Reads compression_info.json to get method/ranks/metadata
    2. Detects schema version (v1 or v2)
    3. For v2: Uses block_impl and metadata to build exact structure
    4. For v1: Falls back to original compress_model() approach
    5. Loads the saved flashsvd_state_dict.pt

    Args:
        checkpoint_dir: Path to compressed model directory
        device: Device to load model on

    Returns:
        Compressed model with SVDBlocks structure

    Example:
        >>> model = load_compressed("./compressed_models/bert/ada/bert_ada_r64", device="cuda")
        >>> # Model has AdaSVDBlocks, ready for inference
    """
    checkpoint_dir = os.path.abspath(checkpoint_dir)

    # 1. Load compression metadata
    metadata_path = os.path.join(checkpoint_dir, "compression_info.json")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(
            f"compression_info.json not found in {checkpoint_dir}. "
            f"This may not be a FlashSVD compressed model."
        )

    with open(metadata_path, "r") as f:
        compression_info = json.load(f)

    # 2. Detect schema version
    schema_version = compression_info.get("schema_version", 1)
    print(f"Checkpoint schema version: {schema_version}")

    # 3. Load based on schema version
    if schema_version >= 2 and "block_impl" in compression_info:
        # v2: Schema-based loading
        model = _load_v2_checkpoint(checkpoint_dir, device, compression_info)
    else:
        # v1: Backward compatibility
        model = _load_v1_checkpoint(checkpoint_dir, device, compression_info)

    model = model.to(device).eval()

    print(f"✅ Compressed model loaded successfully!")
    print(f"   Model has {sum(p.numel() for p in model.parameters()):,} parameters")

    # Verify SVDBlock structure
    if hasattr(model, 'bert'):
        first_layer = model.bert.encoder.layer[0]
        arch_name = "BERT"
    elif hasattr(model, 'roberta'):
        first_layer = model.roberta.encoder.layer[0]
        arch_name = "RoBERTa"
    else:
        first_layer = None
        arch_name = "Unknown"

    if first_layer is not None:
        layer_type = type(first_layer).__name__
        print(f"   Architecture: {arch_name}")
        print(f"   First encoder layer type: {layer_type}")

        if hasattr(first_layer, "block"):
            block_type = type(first_layer.block).__name__
            print(f"   Inner block type: {block_type}")

    return model
