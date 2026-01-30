"""
FlashSVD Evaluation Pipeline

Evaluates compressed models on GLUE tasks with accuracy and performance metrics.
M3: Supports encoder models (BERT/RoBERTa) compressed with standard SVD.
"""

import os
import json
import argparse
import time
from dataclasses import dataclass
from typing import Optional, Dict, Any

# Requirement 1: Check dependencies with clear error messages
try:
    import torch
    from tqdm import tqdm
except ImportError:
    raise ImportError(
        "PyTorch is required. Install with: pip install torch tqdm\n"
        "For CUDA support, see: https://pytorch.org/get-started/locally/"
    )

try:
    from torch.utils.data import DataLoader
except ImportError:
    raise ImportError("torch.utils.data not found. Please reinstall PyTorch.")

try:
    from datasets import load_dataset
except ImportError:
    raise ImportError(
        "datasets library is required. Install with: pip install datasets"
    )

try:
    from transformers import AutoTokenizer
except ImportError:
    raise ImportError(
        "transformers library is required. Install with: pip install transformers"
    )

try:
    from evaluate import load as load_metric
except ImportError:
    raise ImportError(
        "evaluate library is required. Install with: pip install evaluate"
    )

# Import from flashsvd package
import flashsvd
from flashsvd.io import load_compressed
from flashsvd.utils import metrics
from flashsvd.compression._metadata import get_git_commit


@dataclass
class EvalConfig:
    """Configuration for model evaluation."""

    # Model settings
    checkpoint_dir: str = "./compressed_models/bert_r64"

    # Task settings
    task: str = "sst2"

    # Data settings
    batch_size: int = 32
    seq_len: int = 128
    max_eval_samples: Optional[int] = None  # None = use all validation data

    # Performance measurement settings
    warmup_steps: int = 5
    measure_steps: int = 10

    # Hardware settings
    device: str = "cuda"

    # Output settings
    output: str = "eval_results.json"


def load_dataset_and_tokenizer(
    task: str,
    base_model: str,
    checkpoint_dir: str,
    seq_len: int,
    batch_size: int,
    max_samples: Optional[int] = None
):
    """
    Load GLUE dataset and tokenizer.

    Reuses logic from BERT/profile_svd.py (lines 60-117).

    Args:
        task: GLUE task name
        base_model: Base model name for tokenizer
        checkpoint_dir: Checkpoint directory (may contain tokenizer files)
        seq_len: Max sequence length
        batch_size: Batch size
        max_samples: Max samples to evaluate (None = all)

    Returns:
        (dataloader, metric, tokenizer)
    """
    # Requirement 2: Task split rules
    # mnli uses validation_matched, others use validation
    if task == "mnli":
        val_split = "validation_matched"
    else:
        val_split = "validation"

    # Load dataset
    print(f"Loading GLUE/{task} (split: {val_split})...")
    raw = load_dataset("glue", task, split=val_split)

    # Requirement 4: max_eval_samples - subsample validation set
    original_size = len(raw)
    if max_samples is not None and original_size > max_samples:
        raw = raw.select(range(max_samples))
        print(f"Subsampled validation set: {max_samples} of {original_size} samples")

    # Load tokenizer (try checkpoint first, fallback to base_model)
    try:
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir, use_fast=True)
        print(f"Loaded tokenizer from checkpoint: {checkpoint_dir}")
    except:
        tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
        print(f"Loaded tokenizer from base model: {base_model}")

    # Task field mapping
    single_sent_tasks = {"cola", "sst2"}
    pair_sent_tasks = {"qqp", "mnli", "qnli", "stsb", "rte", "mrpc"}
    field_map = {
        "qqp": ("question1", "question2"),
        "mnli": ("premise", "hypothesis"),
        "qnli": ("question", "sentence"),
        "stsb": ("sentence1", "sentence2"),
        "rte": ("sentence1", "sentence2"),
        "mrpc": ("sentence1", "sentence2"),
    }

    # Tokenization function
    def tokenize_fn(batch):
        if task in single_sent_tasks:
            return tokenizer(
                batch["sentence"],
                padding="max_length",
                truncation=True,
                max_length=seq_len,
            )
        else:
            f1, f2 = field_map[task]
            return tokenizer(
                batch[f1],
                batch[f2],
                padding="max_length",
                truncation=True,
                max_length=seq_len,
            )

    # Tokenize dataset
    remove_cols = [c for c in raw.column_names if c != "label"]
    tokenized = raw.map(
        tokenize_fn,
        batched=True,
        remove_columns=remove_cols,
    )
    tokenized.set_format("torch")

    # Create dataloader
    loader = DataLoader(
        tokenized,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda b: {
            "input_ids": torch.stack([x["input_ids"] for x in b]),
            "attention_mask": torch.stack([x["attention_mask"] for x in b]),
            "labels": torch.tensor([x["label"] for x in b]),
        },
    )

    # Load appropriate metric
    if task == "stsb":
        metric = load_metric("pearsonr")
    else:
        metric = load_metric("accuracy")

    return loader, metric, tokenizer


def run_eval(config: EvalConfig) -> Dict[str, Any]:
    """
    Run evaluation pipeline.

    Args:
        config: Evaluation configuration

    Returns:
        Metrics dictionary with accuracy, memory, latency, etc.
    """
    # M4: Input validation
    # Check checkpoint exists
    if not os.path.exists(config.checkpoint_dir):
        raise FileNotFoundError(
            f"Checkpoint directory not found: {config.checkpoint_dir}\n"
            f"Please verify the path is correct.\n"
            f"Hint: Use 'flashsvd info <path>' to verify checkpoint."
        )

    # Check compression_info.json exists
    compression_info_path = os.path.join(config.checkpoint_dir, "compression_info.json")
    if not os.path.exists(compression_info_path):
        raise FileNotFoundError(
            f"compression_info.json not found in {config.checkpoint_dir}\n"
            f"This may not be a FlashSVD compressed model."
        )

    # Check CUDA availability if requested
    if config.device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA requested but not available.\n"
                "Options:\n"
                "  1. Use CPU: add --device cpu flag (Note: Triton kernels require CUDA)\n"
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
    else:
        # M4: Warn about CPU + Triton limitation
        print("⚠️  Warning: Using CPU mode.")
        print("    Triton kernels (used in SVDBlocks) require CUDA and will fail on CPU.")
        print("    To use CPU evaluation, you would need dense (non-Triton) kernels.")
        print("    Recommendation: Use --device cuda if GPU is available.")

    print("=" * 60)
    print("FlashSVD Evaluation Pipeline (M3: Encoder Models)")
    print("=" * 60)
    print(f"Checkpoint: {config.checkpoint_dir}")
    print(f"Task: {config.task}")
    print(f"Batch size: {config.batch_size}")
    print(f"Sequence length: {config.seq_len}")
    print(f"Device: {config.device}")
    print("=" * 60)

    # M4: Progress indicators
    with tqdm(total=3, desc="Evaluation Progress", bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}') as pbar:
        # 1. Load compressed model
        pbar.set_description("[1/3] Loading model")
        model = load_compressed(config.checkpoint_dir, device=config.device)
        pbar.update(1)

        # Read compression info for metadata
        compression_info_path = os.path.join(config.checkpoint_dir, "compression_info.json")
        with open(compression_info_path, "r") as f:
            compression_info = json.load(f)

        # 2. Load dataset and tokenizer
        pbar.set_description("[2/3] Loading dataset")
        loader, metric, tokenizer = load_dataset_and_tokenizer(
            task=config.task,
            base_model=compression_info["base_model"],
            checkpoint_dir=config.checkpoint_dir,
            seq_len=config.seq_len,
            batch_size=config.batch_size,
            max_samples=config.max_eval_samples
        )

        num_samples = len(loader.dataset)
        num_batches = len(loader)
        print(f"Loaded {num_samples} samples ({num_batches} batches)")
        pbar.update(1)

        # 3. Evaluate
        pbar.set_description(f"[3/3] Evaluating {config.task}")

        # Requirement 3: Use metrics.acc_peak_time from src/utils/metrics.py
        # Note: latency_ms is averaged over all batches in eval loop
        # This function already handles:
        # - CUDA synchronization
        # - Peak memory tracking
        # - Latency measurement (averaged over eval loop)
        # - stsb (pearson) vs other tasks (accuracy)
        accuracy, peak_memory_mib, latency_ms = metrics.acc_peak_time(
            mdl=model,
            loader=loader,
            metric=metric,
            task_name=config.task,
            device=config.device,
            use_mask=True
        )
        pbar.update(1)

    # Handle CPU case (peak_memory not meaningful on CPU)
    if config.device == "cpu":
        peak_memory_mib = None

    # Determine metric name
    metric_name = "pearson" if config.task == "stsb" else "accuracy"

    # Requirement 5: Build results dictionary with all required fields
    results = {
        # Task info
        "task": config.task,
        "metric_name": metric_name,
        "metric_value": float(accuracy),

        # Model info (from compression_info.json)
        "base_model": compression_info["base_model"],
        "method": compression_info["method"],
        "ranks": compression_info["ranks"],

        # Performance metrics
        "peak_memory_mib": peak_memory_mib,
        "latency_ms": float(latency_ms),  # Averaged over eval loop

        # Eval config
        "batch_size": config.batch_size,
        "seq_len": config.seq_len,
        "num_eval_samples": num_samples,
        "device": config.device,

        # Checkpoint info
        "checkpoint_dir": config.checkpoint_dir,

        # Metadata (Requirement 5: git_commit, flashsvd_version, timestamp)
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "git_commit": get_git_commit(),
        "flashsvd_version": flashsvd.__version__,

        # Requirement 3: Latency definition
        "_latency_definition": "Average milliseconds per batch over entire eval loop",
    }

    # Add note for CPU peak memory
    if config.device == "cpu":
        results["_note_peak_memory"] = "peak_memory_mib is null on CPU (only CUDA supports memory tracking)"

    return results


def main():
    """CLI entry point for flashsvd-eval."""
    parser = argparse.ArgumentParser(
        description="FlashSVD Model Evaluation (M3: Encoder models)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate compressed model on SST-2
  flashsvd-eval --checkpoint ./compressed_models/bert-base-uncased_standard_r64 --task sst2

  # Custom batch size and sequence length
  flashsvd-eval --checkpoint ./compressed_models/bert_r64 --task sst2 --batch-size 16 --seq-len 256

  # CPU evaluation
  flashsvd-eval --checkpoint ./compressed_models/bert_r64 --task sst2 --no-cuda

  # Save to custom output file
  flashsvd-eval --checkpoint ./compressed_models/bert_r64 --task sst2 --output my_results.json

Note: This evaluates models compressed with flashsvd-compress (M2).
      Use load_compressed() to restore SVDBlocks structure.
        """
    )

    # Model settings
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to compressed model checkpoint directory")

    # Task settings
    parser.add_argument("--task", type=str, default="sst2",
                        choices=["cola", "sst2", "mrpc", "qqp", "mnli", "qnli", "rte", "stsb"],
                        help="GLUE task name (default: sst2)")

    # Data settings
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size for evaluation (default: 32)")
    parser.add_argument("--seq-len", type=int, default=128,
                        help="Max sequence length (default: 128)")
    parser.add_argument("--max-eval-samples", type=int, default=None,
                        help="Max samples to evaluate (default: None = all)")

    # Performance settings
    parser.add_argument("--warmup-steps", type=int, default=5,
                        help="Warmup steps before measurement (default: 5)")
    parser.add_argument("--measure-steps", type=int, default=10,
                        help="Steps to measure for latency (default: 10)")

    # Hardware settings
    parser.add_argument("--device", type=str, default="cuda",
                        choices=["cuda", "cpu"],
                        help="Device to use: cuda or cpu (default: cuda)")

    # Output settings
    parser.add_argument("--output", type=str, default="eval_results.json",
                        help="Output JSON file (default: eval_results.json)")

    args = parser.parse_args()

    # Create config
    config = EvalConfig(
        checkpoint_dir=args.checkpoint,
        task=args.task,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        max_eval_samples=args.max_eval_samples,
        warmup_steps=args.warmup_steps,
        measure_steps=args.measure_steps,
        device=args.device,
        output=args.output,
    )

    # Run evaluation
    try:
        results = run_eval(config)

        # Save results to JSON
        print(f"\n{'=' * 60}")
        print("Results")
        print("=" * 60)

        # Print summary line
        metric_name = results["metric_name"]
        metric_value = results["metric_value"]
        peak_mem = results["peak_memory_mib"]
        latency = results["latency_ms"]

        if peak_mem is not None:
            print(f"{config.task} | {metric_name}={metric_value:.4f} | "
                  f"peak={peak_mem:.1f} MiB | latency={latency:.2f} ms/batch")
        else:
            print(f"{config.task} | {metric_name}={metric_value:.4f} | "
                  f"peak=N/A (CPU) | latency={latency:.2f} ms/batch")

        # Save to JSON
        with open(config.output, "w") as f:
            json.dump(results, f, indent=2)

        print(f"\n✅ Results saved to: {config.output}")
        return 0

    except Exception as e:
        print(f"\n❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
