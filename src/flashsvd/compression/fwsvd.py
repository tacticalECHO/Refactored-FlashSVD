"""
Fisher-Weighted SVD (FWSVD) compression for BERT and RoBERTa models.

Reuses existing src/utils/fwsvd.py utilities exactly as specified in M6 contract.
"""

import torch
import torch.nn as nn
from typing import Dict
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

# Import from flashsvd package
from flashsvd.utils import svd_helpers, FlashSVDBlocks


def _build_calibration_dataloader(
    model_dir: str,
    task: str = "sst2",
    num_samples: int = 128,
    batch_size: int = 32,
    seq_len: int = 128
) -> DataLoader:
    """
    Build calibration dataloader for Fisher weight estimation.

    Args:
        model_dir: Path to model directory (for tokenizer)
        task: GLUE task name
        num_samples: Number of calibration samples
        batch_size: Batch size
        seq_len: Sequence length

    Returns:
        DataLoader for calibration
    """
    # Load dataset
    if task == "mnli":
        val_split = "validation_matched"
    else:
        val_split = "validation"

    raw = load_dataset("glue", task, split=val_split)

    # Limit to num_samples
    if len(raw) > num_samples:
        raw = raw.select(range(num_samples))

    tokz = AutoTokenizer.from_pretrained(model_dir)

    # Task-specific tokenization
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

    def tokenize_fn(batch):
        if task in single_sent_tasks:
            return tokz(
                batch["sentence"],
                padding="max_length",
                truncation=True,
                max_length=seq_len,
            )
        else:
            f1, f2 = field_map[task]
            return tokz(
                batch[f1],
                batch[f2],
                padding="max_length",
                truncation=True,
                max_length=seq_len,
            )

    # Tokenize
    remove_cols = [c for c in raw.column_names if c != "label"]
    ds = raw.map(tokenize_fn, batched=True, remove_columns=remove_cols)
    ds.set_format("torch")

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda b: {
            "input_ids": torch.stack([x["input_ids"] for x in b]),
            "attention_mask": torch.stack([x["attention_mask"] for x in b]),
            "labels": torch.tensor([x["label"] for x in b]),
        },
    )

    return loader


def compress_bert_fwsvd(
    model: nn.Module,
    ranks: Dict[str, int],
    device: str = "cuda",
    calib_samples: int = 128,
    calib_task: str = "sst2",
    model_dir: str = None,
    build_only: bool = False,
    **kwargs
) -> nn.Module:
    """
    Apply Fisher-Weighted SVD compression to BERT model.

    This function reuses src/utils/fwsvd.py utilities exactly.

    M7 Phase 2.x: Added build_only mode for v2 loader (no Fisher calibration).

    Args:
        model: BERT model (e.g., BertForSequenceClassification)
        ranks: Dict with keys "attn", "ffn", "wo" specifying truncation ranks
        device: Device to perform compression on
        calib_samples: Number of calibration samples for Fisher weight estimation
        calib_task: GLUE task for calibration (default: sst2)
        model_dir: Path to model directory (for tokenizer)
        build_only: If True, only create parameter shapes (skip Fisher calibration)

    Returns:
        Compressed model with FlashFWSVDBlocks replacing original layers

    Example:
        >>> model = BertForSequenceClassification.from_pretrained("bert-base-uncased")
        >>> ranks = {"attn": 64, "ffn": 256, "wo": 256}
        >>> compressed = compress_bert_fwsvd(
        ...     model, ranks, "cuda",
        ...     calib_samples=128,
        ...     calib_task="sst2",
        ...     model_dir="./models/bert-base-uncased"
        ... )
    """
    model = model.to(device).eval()

    # Extract rank values
    rank_attn = ranks.get("attn", 64)
    rank_ffn = ranks.get("ffn", 256)
    rank_wo = ranks.get("wo", 256)

    # M7 Phase 2.x: Handle build_only mode
    if build_only:
        print("  [FWSVD] build_only=True: SKIP Fisher calibration")
        fwsvd_per_head, fwsvd_low_rank = None, None
    else:
        # Build calibration dataloader
        if model_dir is None:
            raise ValueError("model_dir is required for FWSVD (needed for tokenizer)")

        print(f"Building calibration dataloader: task={calib_task}, samples={calib_samples}")
        calib_loader = _build_calibration_dataloader(
            model_dir=model_dir,
            task=calib_task,
            num_samples=calib_samples,
            batch_size=32,
            seq_len=128
        )

        # Build FWSVD helpers (reuses src/utils/fwsvd.py exactly)
        # This calls estimate_fisher_weights_bert_with_attention() internally
        print("Computing Fisher weights...")
        fwsvd_per_head, fwsvd_low_rank = svd_helpers.build_fwsvd_helpers(
            model, calib_loader, device=device
        )

    # Replace each encoder layer with FWSVD block
    # This follows the exact pattern from BERTFW/profile_flashfwsvd.py:153-162
    print(f"Replacing layers with FlashFWSVD blocks (build_only={build_only})...")
    for i, layer in enumerate(model.bert.encoder.layer):
        # Create FWSVD block (performs Fisher-Weighted decomposition unless build_only)
        fwsvd_block = FlashSVDBlocks.BertFlashFWSVDBlock(
            layer,
            rank_attn=rank_attn,
            rank_ff=rank_ffn,
            fwsvd_per_head=fwsvd_per_head,
            fwsvd_low_rank=fwsvd_low_rank,
            rank_wo=rank_wo,
            build_only=build_only
        )

        # Wrap with FWLayerShim for HuggingFace compatibility
        shimmed_block = svd_helpers.BertFWLayerShim(fwsvd_block)

        # Replace original layer
        model.bert.encoder.layer[i] = shimmed_block.to(device).eval()

    # Clean up helper functions to free memory
    del fwsvd_per_head, fwsvd_low_rank

    # Clean up any cached helpers in layers
    for layer in model.bert.encoder.layer:
        if hasattr(layer, 'fwsvd_per_head'):
            del layer.fwsvd_per_head
        if hasattr(layer, 'fwsvd_low_rank'):
            del layer.fwsvd_low_rank

    torch.cuda.empty_cache()

    return model


def compress_roberta_fwsvd(
    model: nn.Module,
    ranks: Dict[str, int],
    device: str = "cuda",
    calib_samples: int = 128,
    calib_task: str = "mnli",
    model_dir: str = None,
    build_only: bool = False,
    **kwargs
) -> nn.Module:
    """
    Apply Fisher-Weighted SVD compression to RoBERTa model.

    This function reuses src/utils/fwsvd.py utilities exactly.

    M7 Phase 2.x: Added build_only mode for v2 loader (no Fisher calibration).

    Args:
        model: RoBERTa model (e.g., RobertaForSequenceClassification)
        ranks: Dict with keys "attn", "ffn", "wo" specifying truncation ranks
        device: Device to perform compression on
        calib_samples: Number of calibration samples for Fisher weight estimation
        calib_task: GLUE task for calibration (default: mnli)
        model_dir: Path to model directory (for tokenizer)
        build_only: If True, only create parameter shapes (skip Fisher calibration)

    Returns:
        Compressed model with FlashFWSVDBlocks replacing original layers

    Example:
        >>> model = RobertaForSequenceClassification.from_pretrained("roberta-base")
        >>> ranks = {"attn": 64, "ffn": 256, "wo": 256}
        >>> compressed = compress_roberta_fwsvd(
        ...     model, ranks, "cuda",
        ...     calib_samples=128,
        ...     calib_task="mnli",
        ...     model_dir="./models/roberta-base-mnli"
        ... )
    """
    model = model.to(device).eval()

    # Extract rank values
    rank_attn = ranks.get("attn", 64)
    rank_ffn = ranks.get("ffn", 256)
    rank_wo = ranks.get("wo", 256)

    # M7 Phase 2.x: Handle build_only mode
    if build_only:
        print("  [RobertaFWSVD] build_only=True: SKIP Fisher calibration")
        fwsvd_per_head, fwsvd_low_rank = None, None
    else:
        # Build calibration dataloader
        if model_dir is None:
            raise ValueError("model_dir is required for FWSVD (needed for tokenizer)")

        print(f"Building calibration dataloader: task={calib_task}, samples={calib_samples}")
        calib_loader = _build_calibration_dataloader(
            model_dir=model_dir,
            task=calib_task,
            num_samples=calib_samples,
            batch_size=32,
            seq_len=128
        )

        # IMPORTANT: build_fwsvd_helpers() expects model.bert.encoder.layer
        # Create temporary alias (following RoBERTaFW/profile_flashfwsvd_roberta.py:135)
        model.bert = model.roberta

        # Build FWSVD helpers (reuses src/utils/fwsvd.py exactly)
        print("Computing Fisher weights...")
        fwsvd_per_head, fwsvd_low_rank = svd_helpers.build_fwsvd_helpers(
            model, calib_loader, device=device
        )

    # Replace each encoder layer with FWSVD block
    # This follows the exact pattern from RoBERTaFW/profile_flashfwsvd_roberta.py:143-152
    print(f"Replacing layers with FlashFWSVD blocks (build_only={build_only})...")
    for i, layer in enumerate(model.roberta.encoder.layer):
        # Create FWSVD block (performs Fisher-Weighted decomposition unless build_only)
        fwsvd_block = FlashSVDBlocks.RobertaFlashFWSVDBlock(
            layer,
            rank_attn=rank_attn,
            rank_ff=rank_ffn,
            fwsvd_per_head=fwsvd_per_head,
            fwsvd_low_rank=fwsvd_low_rank,
            rank_wo=rank_wo,
            build_only=build_only
        )

        # Wrap with FWLayerShim for HuggingFace compatibility
        # RoBERTa uses the same BertFWLayerShim as BERT
        shimmed_block = svd_helpers.BertFWLayerShim(fwsvd_block)

        # Replace original layer
        model.roberta.encoder.layer[i] = shimmed_block.to(device).eval()

    # Clean up helper functions to free memory
    del fwsvd_per_head, fwsvd_low_rank

    # Clean up any cached helpers in layers
    for layer in model.roberta.encoder.layer:
        if hasattr(layer, 'fwsvd_per_head'):
            del layer.fwsvd_per_head
        if hasattr(layer, 'fwsvd_low_rank'):
            del layer.fwsvd_low_rank

    torch.cuda.empty_cache()

    return model
