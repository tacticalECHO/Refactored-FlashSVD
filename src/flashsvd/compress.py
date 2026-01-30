"""
FlashSVD Compression Pipeline

Unified compression interface with CLI entry point.
M2.0: Only supports encoder models (BERT/RoBERTa) for sequence classification.
"""

import os
import json
import argparse
from dataclasses import dataclass
from typing import Optional, Dict

import torch
from transformers import AutoModelForSequenceClassification, AutoConfig
from tqdm import tqdm

# Import from flashsvd package (constraint 4: no 'from src.*')
import flashsvd
from flashsvd.compression import compress_model
from flashsvd.compression._metadata import create_compression_info


@dataclass
class CompressConfig:
    """Configuration for model compression."""

    # Model settings
    model_name: str = "bert-base-uncased"
    task: str = "sst2"
    num_labels: Optional[int] = None

    # Compression settings (M6: all methods + architectures)
    arch: Optional[str] = None  # auto, bert, roberta, modernbert
    method: str = "standard"
    rank_attn: int = 64
    rank_ffn: int = 256
    rank_wo: int = 256

    # Method-specific settings
    ranks_json: Optional[str] = None  # For AdaSVD
    calib_samples: int = 128  # For FWSVD and Whiten
    calib_task: Optional[str] = None  # For FWSVD and Whiten (defaults to task)
    modernbert_variant: str = "mask"  # For ModernBERT: mask, fwmask, long
    ffn_kernel: str = "v1"  # For AdaSVD: v1 or v2

    # I/O settings
    checkpoint_dir: Optional[str] = None  # If loading finetuned checkpoint
    output_dir: str = "./compressed_models"

    # Hardware settings
    device: str = "cuda"

    def __post_init__(self):
        # Auto-detect num_labels from task
        if self.num_labels is None:
            task_labels = {
                "cola": 2, "sst2": 2, "mrpc": 2, "qqp": 2,
                "mnli": 3, "qnli": 2, "rte": 2, "stsb": 1
            }
            self.num_labels = task_labels.get(self.task, 2)

        # Default calib_task to task
        if self.calib_task is None:
            self.calib_task = self.task

    @property
    def ranks_dict(self) -> Dict[str, int]:
        """Return ranks as dict for compression methods."""
        return {
            "attn": self.rank_attn,
            "ffn": self.rank_ffn,
            "wo": self.rank_wo,
        }

    def extra_args(self, method_normalized: str) -> Dict:
        """
        Return method-specific extra arguments.

        M7 Phase 2.1: Only include relevant args for each method to avoid
        TypeError from unexpected keyword arguments.

        M7 Phase 2.2: For AdaSVD, pass ranks_json_path (not ranks_json) to match new API.

        M7 Phase 2.3: For FWSVD/Whiten, fallback to model_name if checkpoint_dir not provided.
        """
        # For methods needing tokenizer: use checkpoint_dir if available, else model_name
        model_dir = self.checkpoint_dir if self.checkpoint_dir else self.model_name
        args = {}

        if method_normalized == "ada":
            # M7 Phase 2.2: AdaSVD-specific (use ranks_json_path)
            args["ranks_json_path"] = self.ranks_json  # Note: renamed param
            args["ffn_kernel"] = self.ffn_kernel
            args["model_dir"] = model_dir
            args["strict"] = True  # Fail loudly if rank missing
        elif method_normalized in ["fw", "whiten"]:
            # FWSVD/Whiten-specific
            args["calib_samples"] = self.calib_samples
            args["calib_task"] = self.calib_task
            args["model_dir"] = model_dir  # M7 Phase 2.3: Now includes model_name fallback
        elif method_normalized == "standard":
            # Standard SVD: no extra args
            pass

        return args


def load_model(config: CompressConfig):
    """
    Load encoder model for compression (BERT/RoBERTa only for M2.0).

    Args:
        config: Compression configuration

    Returns:
        Loaded model ready for compression
    """
    # Determine model source (finetuned checkpoint or pretrained)
    if config.checkpoint_dir and os.path.exists(config.checkpoint_dir):
        model_path = config.checkpoint_dir
        print(f"Loading finetuned model from: {model_path}")
    else:
        model_path = config.model_name
        print(f"Loading pretrained model: {model_path}")

    # Configure model
    model_config = AutoConfig.from_pretrained(model_path)
    model_config.num_labels = config.num_labels

    # Handle regression tasks (STS-B)
    if config.task == "stsb":
        model_config.problem_type = "regression"

    # Load model (constraint 3: only SequenceClassification for M2.0)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        config=model_config
    )

    return model


def save_compressed_model(model, config: CompressConfig, output_dir: str, adasvd_ranks: dict = None):
    """
    Save compressed model with metadata (constraint 2: use save_pretrained).

    Critical fix: Also saves flashsvd_state_dict.pt to preserve SVDBlocks structure.
    M7 Phase 2: Saves schema v2 metadata for stable checkpoint loading.

    Args:
        model: Compressed model
        config: Compression configuration
        output_dir: Output directory path
        adasvd_ranks: Full per-operation ranks dict (for AdaSVD only)
    """
    import torch
    os.makedirs(output_dir, exist_ok=True)

    print(f"Saving model to {output_dir}...")

    # Add compression metadata to model config before saving
    model.config._compression_info = "compression_info.json"
    model.config._compressed = True
    model.config._compression_method = config.method

    # 1. Save HuggingFace format (config.json + model weights)
    # This allows standard HF loading but loses SVDBlock structure
    model.save_pretrained(output_dir)
    print(f"✓ Saved HuggingFace format (config.json + model weights)")

    # 2. Save FlashSVD state dict (preserves SVDBlocks structure)
    # This is the KEY FIX: allows load_compressed() to restore exact structure
    flashsvd_state_path = os.path.join(output_dir, "flashsvd_state_dict.pt")
    torch.save(model.state_dict(), flashsvd_state_path)
    print(f"✓ Saved FlashSVD state dict: flashsvd_state_dict.pt")

    # 3. Save compression metadata with schema v2 (M7 Phase 2)
    from flashsvd.compression import detect_architecture, normalize_method
    from flashsvd.compression._metadata import get_block_impl

    arch = detect_architecture(model)
    method_normalized = normalize_method(config.method)
    block_impl = get_block_impl(arch, method_normalized)

    compression_info = create_compression_info(
        method=method_normalized,
        ranks=config.ranks_dict,  # Dict format with attn/ffn/wo keys
        base_model=config.model_name,
        task=config.task,
        flashsvd_version=flashsvd.__version__,
        # Schema v2 fields (M7 Phase 2)
        arch=arch,
        block_impl=block_impl,
        adasvd_ranks=adasvd_ranks,  # Full per-op ranks for AdaSVD
        ranks_json_path=config.ranks_json,  # For AdaSVD
        calib_samples=config.calib_samples if method_normalized in ["fw", "whiten"] else None,
        calib_task=config.calib_task if method_normalized in ["fw", "whiten"] else None,
        ffn_kernel=config.ffn_kernel if method_normalized == "ada" else None,
        schema_version=2,
    )

    metadata_path = os.path.join(output_dir, "compression_info.json")
    with open(metadata_path, "w") as f:
        json.dump(compression_info, f, indent=2)
    print(f"✓ Saved compression metadata: compression_info.json (schema v2)")

    print(f"\n✅ Compression complete! Model saved to: {output_dir}")
    print(f"   - Use load_compressed() to restore SVDBlocks structure")
    print(f"   - Or use HF AutoModel (restores to standard BERT)")


def run_compress(config: CompressConfig) -> str:
    """
    Run compression pipeline (M6: all methods + architectures).

    Args:
        config: Compression configuration

    Returns:
        Path to output directory
    """
    # Import detect_architecture and normalize_method
    from flashsvd.compression import detect_architecture, normalize_method

    # M4: Input validation
    # Check CUDA availability if requested
    if config.device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA requested but not available.\n"
                "Options:\n"
                "  1. Use CPU: add --device cpu flag\n"
                "  2. Install CUDA-enabled PyTorch: https://pytorch.org/get-started/locally/"
            )
        # Extract device index if specified (e.g., "cuda:0" -> 0)
        device_idx = 0
        if ":" in config.device:
            try:
                device_idx = int(config.device.split(":")[1])
            except (ValueError, IndexError):
                device_idx = 0
        print(f"✓ CUDA available: {torch.cuda.get_device_name(device_idx)}")

    # Check checkpoint if provided
    if config.checkpoint_dir and not os.path.exists(config.checkpoint_dir):
        raise FileNotFoundError(
            f"Checkpoint directory not found: {config.checkpoint_dir}\n"
            f"Please verify the path is correct."
        )

    print("=" * 60)
    print("FlashSVD Compression Pipeline (M6: Method Selection)")
    print("=" * 60)
    print(f"Model: {config.model_name}")
    print(f"Task: {config.task}")
    print(f"Method: {config.method}")
    print(f"Ranks: attn={config.rank_attn}, ffn={config.rank_ffn}, wo={config.rank_wo}")
    print(f"Device: {config.device}")
    print("=" * 60)

    # M4: Progress indicators with tqdm
    with tqdm(total=3, desc="Overall Progress", bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}') as pbar:
        # 1. Load model
        pbar.set_description("[1/3] Loading model")
        model = load_model(config)
        pbar.update(1)

        # Detect architecture if not specified
        if config.arch is None or config.arch == "auto":
            detected_arch = detect_architecture(model)
            print(f"Detected architecture: {detected_arch}")
        else:
            detected_arch = config.arch

        # Normalize method name to lowercase
        method_normalized = normalize_method(config.method)

        # 2. Compress model
        pbar.set_description(f"[2/3] Compressing ({config.method})")
        result = compress_model(
            model=model,
            method=config.method,
            ranks=config.ranks_dict,
            device=config.device,
            **config.extra_args(method_normalized)  # M7 Phase 2: Pass method-specific args only
        )

        # M7 Phase 2.3: Handle tuple return for AdaSVD
        if method_normalized == "ada" and isinstance(result, tuple):
            compressed_model, adasvd_ranks_actual = result
            print(f"  Received actual ranks from AdaSVD compression: {len(adasvd_ranks_actual)} operations")
        else:
            compressed_model = result
            adasvd_ranks_actual = None

        pbar.update(1)

        # 3. Save compressed model
        # M6: Output path structure: compressed_models/{arch}/{method}/{run_name}/
        pbar.set_description("[3/3] Saving")
        output_name = f"{config.model_name.split('/')[-1]}_{method_normalized}_r{config.rank_attn}"
        output_dir = os.path.join(
            config.output_dir,
            detected_arch,
            method_normalized,  # Lowercase
            output_name
        )

        # M7 Phase 2.3: Use actual_ranks from compression (not input ranks.json)
        # For AdaSVD: adasvd_ranks_actual contains post-clamping ranks
        # For other methods: adasvd_ranks_actual = None
        save_compressed_model(compressed_model, config, output_dir, adasvd_ranks=adasvd_ranks_actual)
        pbar.update(1)

    return output_dir


def main():
    """CLI entry point for flashsvd-compress."""
    parser = argparse.ArgumentParser(
        description="FlashSVD Model Compression (M6: Method Selection + Unified Output)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Standard SVD (BERT)
  flashsvd-compress --model bert-base-uncased --task sst2 --method standard --rank 64

  # Fisher-Weighted SVD (RoBERTa)
  flashsvd-compress --model roberta-base --task mnli --method fw --rank 64 --calib-samples 128

  # Adaptive Rank Selection SVD
  flashsvd-compress --checkpoint ./models/bert-sst2 --task sst2 --method ada --ranks-json ./BERTAda/ars_out/ranks.json

  # Whitening/DRONE (BERT only)
  flashsvd-compress --model bert-base-uncased --task sst2 --method whiten --rank 64 --calib-samples 128

Supported architectures: BERT, RoBERTa, ModernBERT (routing only)
Supported methods: standard, fw (fwsvd), ada (adasvd), whiten (drone)
        """
    )

    # Model settings
    parser.add_argument("--model", type=str, default="bert-base-uncased",
                        help="Model name or path (default: bert-base-uncased)")
    parser.add_argument("--task", type=str, default="sst2",
                        choices=["cola", "sst2", "mrpc", "qqp", "mnli", "qnli", "rte", "stsb"],
                        help="GLUE task name (default: sst2)")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to finetuned checkpoint (optional)")

    # Architecture (M6: auto-detect by default)
    parser.add_argument("--arch", type=str, default="auto",
                        choices=["auto", "bert", "roberta", "modernbert"],
                        help="Model architecture (default: auto-detect)")

    # Compression settings (M6: all methods)
    parser.add_argument("--method", type=str, default="standard",
                        choices=["standard", "fw", "fwsvd", "ada", "adasvd", "whiten", "drone"],
                        help="Compression method (accepts aliases: fw/fwsvd, ada/adasvd, whiten/drone)")

    # Rank settings
    parser.add_argument("--rank", type=int, default=None,
                        help="Unified rank for all components (overrides --rank-attn/ffn/wo)")
    parser.add_argument("--rank-attn", type=int, default=64,
                        help="Rank for attention Q/K/V (default: 64)")
    parser.add_argument("--rank-ffn", type=int, default=256,
                        help="Rank for FFN layers (default: 256)")
    parser.add_argument("--rank-wo", type=int, default=256,
                        help="Rank for attention output projection (default: 256)")

    # Method-specific settings
    parser.add_argument("--ranks-json", type=str, default=None,
                        help="Path to ranks.json for AdaSVD (required for ada method)")
    parser.add_argument("--calib-samples", type=int, default=128,
                        help="Number of calibration samples for FWSVD/Whiten (default: 128)")
    parser.add_argument("--calib-task", type=str, default=None,
                        help="Calibration task for FWSVD/Whiten (default: same as --task)")
    parser.add_argument("--modernbert-variant", type=str, default="mask",
                        choices=["mask", "fwmask", "long"],
                        help="ModernBERT variant: mask, fwmask, or long (default: mask)")
    parser.add_argument("--ffn-kernel", type=str, default="v1",
                        choices=["v1", "v2"],
                        help="FFN kernel variant for AdaSVD: v1 or v2 (default: v1)")

    # I/O settings
    parser.add_argument("--output-dir", type=str, default="./compressed_models",
                        help="Output directory (default: ./compressed_models)")

    # Hardware settings
    parser.add_argument("--device", type=str, default="cuda",
                        choices=["cuda", "cpu"],
                        help="Device to use: cuda or cpu (default: cuda)")

    args = parser.parse_args()

    # If --rank is specified, use it for attention and scale for FFN/output
    # Following experimental code convention: ffn_rank = 6x attn_rank, wo_rank = 6x attn_rank
    # (Original experiments use 40/240/240, where 240 = 6 * 40)
    if args.rank is not None:
        rank_attn = args.rank
        rank_ffn = args.rank * 6
        rank_wo = args.rank * 6
        print(f"Using unified rank: attn={rank_attn}, ffn={rank_ffn}, wo={rank_wo} (6x scaling)")
    else:
        rank_attn = args.rank_attn
        rank_ffn = args.rank_ffn
        rank_wo = args.rank_wo

    # Create config
    config = CompressConfig(
        model_name=args.model,
        task=args.task,
        arch=args.arch,
        method=args.method,
        rank_attn=rank_attn,
        rank_ffn=rank_ffn,
        rank_wo=rank_wo,
        ranks_json=args.ranks_json,
        calib_samples=args.calib_samples,
        calib_task=args.calib_task,
        modernbert_variant=args.modernbert_variant,
        ffn_kernel=args.ffn_kernel,
        checkpoint_dir=args.checkpoint,
        output_dir=args.output_dir,
        device=args.device,
    )

    # Run compression
    try:
        output_dir = run_compress(config)
        print(f"\n🎉 Success! Compressed model saved to: {output_dir}")
        print(f"\nVerify with: python -c \"from transformers import AutoModelForSequenceClassification; "
              f"AutoModelForSequenceClassification.from_pretrained('{output_dir}')\"")
        return 0
    except Exception as e:
        print(f"\n❌ Compression failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
