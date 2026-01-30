"""
Whitening/DRONE-style data-aware SVD compression for BERT models.

Extracts and reuses the DRONE factorization logic from BERTWhiten experiments.
"""

import torch
import torch.nn as nn
from typing import Dict, List
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

# Import from flashsvd package (unified import path)
from flashsvd.utils import FlashSVDBlocks, svd_helpers


def _safe_cholesky(C: torch.Tensor, max_tries: int = 5, base_eps: float = 1e-6):
    """
    Robust Cholesky with adaptive jitter if C is near-singular.
    From BERTWhiten/profile_flashsvd.py:47-57
    """
    D = C.shape[-1]
    eps = base_eps * float(C.diag().mean().item() + 1.0)
    I = torch.eye(D, dtype=C.dtype, device=C.device)
    for _ in range(max_tries):
        try:
            return torch.linalg.cholesky(C + eps * I)
        except RuntimeError:
            eps *= 10.0
    return torch.linalg.cholesky(C + (1e-2 * float(C.diag().mean().item() + 1.0)) * I)


@torch.no_grad()
def _data_aware_low_rank(W_in_out: torch.Tensor, rank: int, cov_in: torch.Tensor):
    """
    DRONE-style low-rank factorization.

    Given Linear weight W [d_in, d_out] and input covariance C,
    compute U:[d_in,k], V:[k,d_out] minimizing ||X^T(W - U V)||_F.

    From BERTWhiten/profile_flashsvd.py:60-82
    """
    d_in, d_out = W_in_out.shape
    assert rank > 0, "rank must be positive"
    Wf, Cf = W_in_out.float(), cov_in.float()
    S = _safe_cholesky(Cf)  # [d_in, d_in], lower
    A = Wf.t().contiguous() @ S  # [d_out, d_in]
    U, s, Vh = torch.linalg.svd(A, full_matrices=False)  # U:[d_out,r], s:[r], Vh:[r,d_in]
    V = Vh.t()  # [d_in, r]
    k = min(rank, s.numel())
    U_k, s_k, V_k = U[:, :k], s[:k], V[:, :k]
    # Solve S^T X = V_k  ->  X = S^{-T} V_k
    X = torch.linalg.solve_triangular(S.t(), V_k, upper=True)  # [d_in, k]
    sqrt_s = torch.sqrt(torch.clamp(s_k, min=0))
    U_data = X * sqrt_s.unsqueeze(0)  # [d_in, k]
    V_data = (sqrt_s.unsqueeze(1) * U_k.t())  # [k, d_out]
    return U_data.to(W_in_out.dtype), V_data.to(W_in_out.dtype)


@torch.no_grad()
def _data_aware_per_head(Wt_dm_total: torch.Tensor, rank: int, cov_in: torch.Tensor, num_heads: int):
    """
    DRONE-style per-head factorization for attention Q/K/V.

    From BERTWhiten/profile_flashsvd.py:85-103
    """
    d_model = Wt_dm_total.shape[0]
    dh = Wt_dm_total.shape[1] // num_heads
    Wt3 = Wt_dm_total.view(d_model, num_heads, dh)
    Us, Vs = [], []
    for h in range(num_heads):
        Wh = Wt3[:, h, :]  # [d_model, dh]
        Uh, Vh = _data_aware_low_rank(Wh, rank, cov_in)
        Us.append(Uh)  # [dm, k]
        Vs.append(Vh)  # [k, dh]
    U = torch.stack(Us, dim=0)  # [H, dm, k]
    V = torch.stack(Vs, dim=0)  # [H, k, dh]
    return U, V


def _calibrate_covariances(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    max_batches: int = 4
) -> Dict[str, List[torch.Tensor]]:
    """
    Collect per-layer input covariances for DRONE factorization.

    From BERTWhiten/profile_flashsvd.py:215-297

    Returns:
        Dict with keys: cov_attn_in, cov_attn_out, cov_ffn_in, cov_ffn_out
        Each is a list of [num_layers] covariance matrices
    """
    model.eval()
    enc = model.bert.encoder
    num_layers = len(enc.layer)
    dm = model.config.hidden_size
    dff = model.config.intermediate_size

    # Initialize accumulators
    cov_attn_in = [torch.zeros(dm, dm, dtype=torch.float32, device=device) for _ in range(num_layers)]
    n_attn_in = [0 for _ in range(num_layers)]
    cov_attn_out = [torch.zeros(dm, dm, dtype=torch.float32, device=device) for _ in range(num_layers)]
    n_attn_out = [0 for _ in range(num_layers)]
    cov_ffn_in = [torch.zeros(dm, dm, dtype=torch.float32, device=device) for _ in range(num_layers)]
    n_ffn_in = [0 for _ in range(num_layers)]
    cov_ffn_out = [torch.zeros(dff, dff, dtype=torch.float32, device=device) for _ in range(num_layers)]
    n_ffn_out = [0 for _ in range(num_layers)]
    handles = []

    def _upd(cov_mat, n_store, idx, x):
        if x is None:
            return
        x = x.detach()
        BMD = x.shape[0] * x.shape[1]
        X2d = x.reshape(BMD, x.shape[-1]).to(device=device, dtype=torch.float32)
        cov_mat[idx] += X2d.t() @ X2d
        n_store[idx] += BMD

    # Register hooks
    for i, layer in enumerate(enc.layer):
        # Inputs to Q/K/V: hook query pre-forward (shared input)
        def q_pre(mod, inp, idx=i): _upd(cov_attn_in, n_attn_in, idx, inp[0])
        handles.append(layer.attention.self.query.register_forward_pre_hook(q_pre))

        # Inputs to attention.output.dense
        def attn_out_pre(mod, inp, idx=i): _upd(cov_attn_out, n_attn_out, idx, inp[0])
        handles.append(layer.attention.output.dense.register_forward_pre_hook(attn_out_pre))

        # Inputs to FFN intermediate.dense
        def ffn_in_pre(mod, inp, idx=i): _upd(cov_ffn_in, n_ffn_in, idx, inp[0])
        handles.append(layer.intermediate.dense.register_forward_pre_hook(ffn_in_pre))

        # Inputs to FFN output.dense (post-GELU)
        def ffn_out_pre(mod, inp, idx=i): _upd(cov_ffn_out, n_ffn_out, idx, inp[0])
        handles.append(layer.output.dense.register_forward_pre_hook(ffn_out_pre))

    # Run calibration
    seen = 0
    for batch in loader:
        if seen >= max_batches:
            break
        batch = {k: v.to(device) for k, v in batch.items()}
        _ = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        seen += 1

    # Remove hooks
    for h in handles:
        h.remove()

    # Finalize covariances
    def _finalize(cov_list, n_list):
        out = []
        for C, n in zip(cov_list, n_list):
            if n == 0:
                D = C.shape[0]
                Cn = torch.eye(D, dtype=torch.float32, device=C.device)
            else:
                Cn = C / float(n)
                ridge = 1e-6 * float(Cn.diag().mean().item() + 1.0)
                Cn = Cn + ridge * torch.eye(Cn.shape[0], dtype=Cn.dtype, device=Cn.device)
            out.append(Cn.cpu())
        return out

    return {
        "cov_attn_in": _finalize(cov_attn_in, n_attn_in),
        "cov_attn_out": _finalize(cov_attn_out, n_attn_out),
        "cov_ffn_in": _finalize(cov_ffn_in, n_ffn_in),
        "cov_ffn_out": _finalize(cov_ffn_out, n_ffn_out),
    }


def _build_calibration_dataloader(
    model_dir: str,
    task: str = "sst2",
    num_samples: int = 128,
    batch_size: int = 32,
    seq_len: int = 128
) -> DataLoader:
    """Build calibration dataloader for covariance estimation."""
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


def compress_bert_whiten(
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
    Apply Whitening/DRONE-style data-aware SVD compression to BERT model.

    This function extracts and reuses the DRONE factorization logic from
    BERTWhiten experiments.

    M7 Phase 2.x: Added build_only mode for v2 loader (no covariance calibration).

    Args:
        model: BERT model (e.g., BertForSequenceClassification)
        ranks: Dict with keys "attn", "ffn", "wo" specifying truncation ranks
        device: Device to perform compression on
        calib_samples: Number of calibration samples for covariance estimation
        calib_task: GLUE task for calibration (default: sst2)
        model_dir: Path to model directory (for tokenizer)
        build_only: If True, only create parameter shapes (skip covariance calibration)

    Returns:
        Compressed model with Whiten blocks replacing original layers

    Example:
        >>> model = BertForSequenceClassification.from_pretrained("bert-base-uncased")
        >>> ranks = {"attn": 64, "ffn": 256, "wo": 256}
        >>> compressed = compress_bert_whiten(
        ...     model, ranks, "cuda",
        ...     calib_samples=128,
        ...     calib_task="sst2",
        ...     model_dir="./models/bert-base-uncased-sst2"
        ... )
    """
    model = model.to(device).eval()

    # Extract rank values
    rank_attn = ranks.get("attn", 64)
    rank_ffn = ranks.get("ffn", 256)
    rank_wo = ranks.get("wo", 256)

    # M7 Phase 2.x: Handle build_only mode
    if build_only:
        print("  [Whiten] build_only=True: SKIP covariance calibration")
        # Create dummy covariances (won't be used, blocks will create empty params)
        num_layers = len(model.bert.encoder.layer)
        dm = model.config.hidden_size
        dff = model.config.intermediate_size
        covs = {
            "cov_attn_in": [None] * num_layers,
            "cov_attn_out": [None] * num_layers,
            "cov_ffn_in": [None] * num_layers,
            "cov_ffn_out": [None] * num_layers,
        }
        data_aware_per_head = None
        data_aware_low_rank = None
    else:
        # Build calibration dataloader
        if model_dir is None:
            raise ValueError("model_dir is required for Whiten compression (needed for tokenizer)")

        print(f"Building calibration dataloader: task={calib_task}, samples={calib_samples}")
        calib_loader = _build_calibration_dataloader(
            model_dir=model_dir,
            task=calib_task,
            num_samples=calib_samples,
            batch_size=32,
            seq_len=128
        )

        # Calibrate covariances
        print("Calibrating input covariances...")
        covs = _calibrate_covariances(model, calib_loader, device, max_batches=4)

        # Build data-aware decomposition helpers (use functions defined in this file)
        data_aware_per_head = _data_aware_per_head
        data_aware_low_rank = _data_aware_low_rank

    # Replace each encoder layer with Whiten block
    print(f"Replacing layers with Whiten (DRONE) blocks (build_only={build_only})...")
    for i, layer in enumerate(model.bert.encoder.layer):
        # Create Whiten block (performs DRONE-style factorization unless build_only)
        if build_only:
            # Simplified constructor call for build_only
            whiten_block = FlashSVDBlocks.BertWhitenSVDBlock(
                layer,
                rank_attn=rank_attn,
                rank_ff=rank_ffn,
                rank_wo=rank_wo,
                build_only=True
            )
        else:
            whiten_block = FlashSVDBlocks.BertWhitenSVDBlock(
                layer,
                rank_attn=rank_attn,
                rank_ff=rank_ffn,
                cov_attn_in=covs["cov_attn_in"][i].to(device),
                cov_attn_out=covs["cov_attn_out"][i].to(device),
                cov_ffn_in=covs["cov_ffn_in"][i].to(device),
                cov_ffn_out=covs["cov_ffn_out"][i].to(device),
                rank_wo=rank_wo,
                data_aware_per_head=data_aware_per_head,
                data_aware_low_rank=data_aware_low_rank,
            )

        # Wrap with LayerShim for HuggingFace compatibility
        shimmed_block = svd_helpers.BertLayerShim(whiten_block)

        # Replace original layer
        model.bert.encoder.layer[i] = shimmed_block.to(device).eval()

    # Clean up
    if not build_only:
        del covs
    torch.cuda.empty_cache()

    return model


def compress_roberta_whiten(
    model: nn.Module,
    ranks: Dict[str, int],
    device: str = "cuda",
    **kwargs
) -> nn.Module:
    """
    RoBERTa + Whiten compression not supported.

    The repository has no RoBERTaWhiten experiments, so this method is not
    integrated in M6 scope.

    Raises:
        NotImplementedError: Always (RoBERTa + Whiten not in repository)
    """
    raise NotImplementedError(
        "RoBERTa + Whiten compression not available.\n"
        "\n"
        "The repository has no RoBERTaWhiten experiments.\n"
        "Whiten (DRONE-style data-aware SVD) is only implemented for BERT.\n"
        "\n"
        "Supported RoBERTa methods:\n"
        "  - standard: Standard SVD compression\n"
        "  - fw: Fisher-Weighted SVD\n"
        "  - ada: Adaptive Rank Selection SVD"
    )
