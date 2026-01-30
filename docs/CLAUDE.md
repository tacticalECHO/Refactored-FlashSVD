# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

## 🚨 EXECUTION CONTRACT

**READ THIS FIRST - THIS IS NOT A RESEARCH PROJECT**

### Current Phase: PRODUCTIZATION (Not Research Extension)

This codebase contains **proven research implementations** (5 SVD methods, Triton kernels, block wrappers). The current goal is **product-level engineering**:

**✅ What We Are Building:**
- Unified compression pipeline: `run_compress` → checkpoint + metadata
- Unified evaluation pipeline: `run_eval` → metrics JSON
- User-friendly interfaces: CLI + Gradio UI (Gradio is a thin wrapper only)
- Proper Python packaging: `pip install flashsvd` with entry points

**❌ Strictly PROHIBITED Actions:**
- ❌ Do NOT add new SVD methods or compression algorithms
- ❌ Do NOT rewrite kernel logic (`src/kernels/*.py`)
- ❌ Do NOT rewrite block logic (`src/utils/*Blocks.py`)
- ❌ Do NOT merge/refactor experiment directories (`BERT*/`, `RoBERTa*/`, `decoders/`, `ModernBERT/`)
- ❌ Do NOT create new training pipelines (reuse existing `train_bert_unified_min.py`)
- ❌ Do NOT modify low-level Triton kernels unless fixing critical bugs

### Execution Rules (MANDATORY)

1. **One Milestone at a Time**
   - Complete M1 before starting M2
   - Each milestone has explicit acceptance criteria
   - No parallel milestone work

2. **Before Making ANY Changes:**
   - List all files you will create/modify
   - Explain the change scope (1-3 sentences per file)
   - Get approval before proceeding

3. **After Making Changes:**
   - Provide a complete file change manifest
   - Give runnable verification commands (copy-pasteable)
   - Test commands must succeed before marking milestone complete

4. **When Uncertain:**
   - STOP immediately
   - Present 2-3 specific options with pros/cons
   - Wait for user decision
   - DO NOT guess or "try something"

5. **Code Reuse Priority:**
   - Reuse existing code from experiment directories
   - Extract common patterns, don't rewrite
   - Preserve original implementations as reference

---

## 📋 PRODUCT MILESTONES

### M1: Packaging Skeleton ✅ COMPLETE

**Goal**: Create proper Python package structure with pip-installable setup.

**Allowed Changes:**
- Create `src/flashsvd/__init__.py` and submodules
- Move/symlink code from `src/kernels/` and `src/utils/` into package
- Create `pyproject.toml` (modern packaging, replaces `setup.py`)
- Update `.gitignore` for build artifacts

**Prohibited:**
- Do NOT change kernel/block logic
- Do NOT reorganize experiment directories

**Acceptance Criteria:**
```bash
# Must succeed:
pip install -e .                                                        # ✅ VERIFIED
python -c "import flashsvd; print(flashsvd.__version__)"               # ✅ VERIFIED (0.1.0)
python -c "from flashsvd.kernels import flashsvdattn; from flashsvd.utils import FlashSVDBlocks"  # ✅ VERIFIED
```

**Deliverables:** ✅ ALL COMPLETE
- ✅ `src/flashsvd/__init__.py` with version
- ✅ `src/flashsvd/kernels/` (kernel imports via thin wrapper)
- ✅ `src/flashsvd/utils/` (block imports via thin wrapper)
- ✅ `pyproject.toml` with dependencies and entry points
- ✅ Updated `README.md` installation section

**Files Created:**
- `src/flashsvd/__init__.py`: Version and module exports
- `src/flashsvd/kernels/__init__.py`: Thin wrapper re-exporting from src.kernels
- `src/flashsvd/utils/__init__.py`: Thin wrapper re-exporting from src.utils
- `src/__init__.py`: Marks src as package for sys.path manipulation
- `pyproject.toml`: Modern PEP 621 packaging format

---

### M2: Compression Pipeline ✅ COMPLETE (M2.0: Encoder Models Only)

**Goal**: Unified `run_compress` command that takes any checkpoint and produces compressed model + metadata.

**Scope Adjustment**: M2.0 only implements encoder models (BERT/RoBERTa) with standard SVD. Other methods (fwsvd, adasvd, drone, asvd) and decoder models deferred to future milestones.

**Allowed Changes:**
- Create `src/flashsvd/compress.py` (main pipeline)
- Create `src/flashsvd/compression/` submodule:
  - `standard_svd.py` (extract from `BERT/profile_svd.py`) ✅
  - `fwsvd.py` (deferred to future)
  - `adasvd.py` (deferred to future)
  - `drone.py` (deferred to future)
  - `asvd.py` (deferred to future)
- Add CLI entry point in `pyproject.toml`

**Prohibited:**
- Do NOT modify existing experiment scripts
- Do NOT change SVD algorithm implementations
- Do NOT add new compression methods

**Acceptance Criteria:**
```bash
# Must succeed:
flashsvd-compress \                           # ✅ VERIFIED (also flashsvd compress)
  --model bert-base-uncased \
  --method standard \
  --rank 64 \
  --task sst2

# Unified rank support:
flashsvd-compress --model bert-base-uncased --task sst2 --rank 64         # ✅ VERIFIED

# Separate rank support:
flashsvd-compress --model bert-base-uncased --task sst2 \                 # ✅ VERIFIED
  --rank-attn 40 --rank-ffn 240 --rank-wo 240

# Must produce (saved via HuggingFace save_pretrained):
ls compressed_models/bert-base-uncased_standard_r64/
# Expected files:
#   - config.json (HF model config + compression metadata)
#   - model.safetensors or pytorch_model.bin (HF weights)
#   - flashsvd_state_dict.pt (FlashSVD state for exact recovery)  # ✅ CRITICAL FIX
#   - compression_info.json (method, ranks dict, timestamp, git commit)
```

**Deliverables:** ✅ ALL COMPLETE (M2.0 scope)
- ✅ `src/flashsvd/compress.py` with `CompressConfig` dataclass and `run_compress()`
- ✅ `src/flashsvd/compression/standard_svd.py` (encoder implementation)
- ✅ `src/flashsvd/compression/__init__.py` with `compress_model()` dispatcher
- ✅ `src/flashsvd/compression/_metadata.py` (compression_info.json generation)
- ✅ `src/flashsvd/io.py` with `load_compressed()` function (CRITICAL FIX for structure recovery)
- ✅ CLI: `flashsvd-compress` command
- ✅ Ranks as dict format (attn/ffn/wo keys)
- ✅ HuggingFace save_pretrained() compatibility

**Files Created:**
- `src/flashsvd/compress.py`: Main compression pipeline
- `src/flashsvd/compression/__init__.py`: Method dispatcher
- `src/flashsvd/compression/standard_svd.py`: Standard SVD for BERT/RoBERTa
- `src/flashsvd/compression/_metadata.py`: Metadata generation
- `src/flashsvd/io.py`: Model loading with structure recovery

**Hard Constraints Met:**
1. ✅ CLI supports both `--rank 64` and `--rank-attn/--rank-ffn/--rank-wo`
2. ✅ Output uses HuggingFace `save_pretrained()` format
3. ✅ M2.0 only supports encoder models (BERT/RoBERTa)
4. ✅ New code only imports `flashsvd.*`, not `from src.*`

---

### M3: Evaluation Pipeline ✅ COMPLETE

**Goal**: Unified `run_eval` command that loads compressed checkpoint and outputs metrics.

**Allowed Changes:**
- Create `src/flashsvd/evaluate.py` (main pipeline)
- Extract evaluation logic from existing `profile_*.py` scripts
- Reuse `src/utils/metrics.py` (no changes needed)
- Add CLI entry point in `pyproject.toml`

**Prohibited:**
- Do NOT modify metrics calculation logic
- Do NOT change existing profiling scripts

**Acceptance Criteria:**
```bash
# Must succeed:
flashsvd-eval \                                # ✅ VERIFIED (also flashsvd eval)
  --checkpoint ./compressed_models/bert-base-uncased_standard_r64 \
  --task sst2 \
  --batch-size 32 \
  --output ./results/bert_r64_eval.json

# Must produce JSON with:
cat results/bert_r64_eval.json
# Expected fields: ✅ ALL PRESENT
# {
#   "task": "sst2",
#   "metric_name": "accuracy",
#   "metric_value": 0.xxxx,
#   "base_model": "bert-base-uncased",
#   "method": "standard",
#   "ranks": {"attn": 64, "ffn": 256, "wo": 256},
#   "peak_memory_mib": xxxx,
#   "latency_ms": xx.xx,
#   "batch_size": 32,
#   "seq_len": 128,
#   "num_eval_samples": xxxx,
#   "device": "cuda",
#   "checkpoint_dir": "...",
#   "timestamp": "2026-01-29T...",
#   "git_commit": "abc123...",
#   "flashsvd_version": "0.1.0",
#   "_latency_definition": "Average milliseconds per batch over entire eval loop"
# }
```

**Deliverables:** ✅ ALL COMPLETE
- ✅ `src/flashsvd/evaluate.py` with `EvalConfig` dataclass and `run_eval()`
- ✅ CLI: `flashsvd-eval` command
- ✅ JSON output format with all required fields
- ✅ Reuses `metrics.acc_peak_time()` from src/utils/metrics.py
- ✅ Reuses GLUE loading logic from BERT/profile_svd.py

**Hard Requirements Met:**
1. ✅ Clear error messages for missing dependencies (torch, transformers, datasets, evaluate)
2. ✅ Task split rules: mnli → validation_matched, others → validation
3. ✅ Latency measurement definition documented in JSON output
4. ✅ max_eval_samples correctly subsamples validation set
5. ✅ Output JSON includes all required fields (task, metric, model, ranks, memory, latency, metadata)

---

### M4: CLI Interface ✅ COMPLETE

**Goal**: Polished command-line interface with help, validation, and error messages.

**Allowed Changes:**
- Enhance `flashsvd-compress` and `flashsvd-eval` with:
  - Rich help messages (use `argparse` or `typer`)
  - Input validation (checkpoint exists, valid method, etc.)
  - Progress bars (use `tqdm`)
  - Error handling with actionable messages
- Add `flashsvd-info` command (show checkpoint metadata)
- Add unified `flashsvd` command with subcommands

**Prohibited:**
- Do NOT add new functionality beyond CLI polish
- Do NOT change core compress/eval logic

**Acceptance Criteria:**
```bash
# Must succeed with helpful output:
flashsvd --help                    # ✅ VERIFIED
flashsvd compress --help           # ✅ VERIFIED
flashsvd eval --help               # ✅ VERIFIED
flashsvd info --help               # ✅ VERIFIED

# Backward compatibility:
flashsvd-compress --help           # ✅ VERIFIED
flashsvd-eval --help               # ✅ VERIFIED
flashsvd-info --help               # ✅ VERIFIED

# Must show progress:
flashsvd compress --model bert-base-uncased --task sst2 --rank 64
# Expected: Progress bar for loading, compression, saving

# Must validate:
flashsvd compress --model invalid-model --task sst2 --rank 64
# Expected: Clear error message about model not found

# Must show info:
flashsvd info ./compressed_models/bert-base-uncased_standard_r64
# Expected: Pretty-printed metadata (method, ranks, timestamp, files, usage)
```

**Deliverables:** ✅ ALL COMPLETE
- ✅ Enhanced CLI with validation and progress (compress.py, evaluate.py)
- ✅ `flashsvd-info` command (info.py)
- ✅ Unified `flashsvd` command with subcommands (cli.py)
- ✅ Updated README with CLI examples and smoke tests
- ✅ Backward compatibility maintained (standalone commands still work)

**Files Modified:**
- `src/flashsvd/cli.py` (NEW): Unified entry point with compress/eval/info subcommands
- `src/flashsvd/info.py` (NEW): Checkpoint metadata display
- `src/flashsvd/compress.py` (ENHANCED): Added CUDA checks, checkpoint validation, tqdm progress
- `src/flashsvd/evaluate.py` (ENHANCED): Added input validation, CPU/Triton warnings, tqdm progress
- `pyproject.toml` (UPDATED): Added `flashsvd` entry point
- `README.md` (UPDATED): Added CLI usage section with examples and smoke tests

**Extra Constraints Met:**
1. ✅ All modifications are wrapper-only (no core logic changes)
2. ✅ CPU/Triton handling is clear and context-aware (warns but doesn't block)
3. ✅ Smoke tests documented in README for regression protection

---

### M5: Gradio UI ✅ COMPLETE

**Goal**: Thin web UI wrapper for compress + eval workflows (no business logic in UI).

**Allowed Changes:**
- Create `src/flashsvd/ui/app.py` (Gradio interface)
- UI components:
  - Compression tab: model selection, method, rank, task
  - Evaluation tab: checkpoint selection, task, batch size
  - Info tab: checkpoint metadata display
  - Results display: metrics JSON, file download
- Add `flashsvd-ui` entry point

**Prohibited:**
- Do NOT duplicate compression/evaluation logic in UI code
- Do NOT add features not in CLI
- Gradio code ONLY calls `flashsvd.compress`, `flashsvd.evaluate`, `flashsvd.info` modules

**Acceptance Criteria:**
```bash
# Must succeed:
flashsvd-ui                                    # ✅ VERIFIED
# Expected: Gradio interface opens at http://localhost:7860

# UI functionality:
# 1. Compression tab: select model, method, rank → shows progress → outputs checkpoint path  # ✅ IMPLEMENTED
# 2. Evaluation tab: select checkpoint, task → shows metrics JSON → download results        # ✅ IMPLEMENTED
# 3. Info tab: display checkpoint metadata → download compression_info.json                 # ✅ IMPLEMENTED
# 4. Results persist (can download JSON)                                                    # ✅ IMPLEMENTED
```

**Deliverables:** ✅ ALL COMPLETE
- ✅ `src/flashsvd/ui/__init__.py` (7 lines)
- ✅ `src/flashsvd/ui/app.py` (507 lines) with three tabs
- ✅ Entry point: `flashsvd-ui` command
- ✅ Thin wrapper (no business logic, only calls existing modules)
- ✅ Log capture with `redirect_stdout`/`redirect_stderr`
- ✅ JSON result display and download
- ✅ Device parameter: {cuda, cpu} (consistent with CLI)

**Files Created:**
- `src/flashsvd/ui/__init__.py`: Module exports
- `src/flashsvd/ui/app.py`: Gradio Blocks interface with 3 tabs
  - Tab 1: Compress (calls `run_compress(CompressConfig)`)
  - Tab 2: Evaluate (calls `run_eval(EvalConfig)`)
  - Tab 3: Info (calls `show_checkpoint_info()`)
- `M5_UI_GUIDE.md`: Complete usage guide with step-by-step instructions

**Files Modified:**
- `pyproject.toml`: Uncommented `flashsvd-ui` entry point
- `README.md`: Added Web UI section

**Architecture:**
- **Thin Wrapper Pattern**: UI code contains zero business logic
- **Output Capture**: Uses `io.StringIO` + `contextlib.redirect_stdout`
- **Error Handling**: All exceptions caught and displayed in UI
- **File Download**: Compression info and eval results downloadable as JSON
- Screenshot or demo video (optional)
- Updated README with UI section

---

## ✅ DEFINITION OF DONE

**Phase 1 is complete when ALL of the following are true:**

### Packaging
- [ ] `pip install -e .` succeeds
- [ ] `import flashsvd` works
- [ ] Package has proper `__version__`
- [ ] All dependencies specified in `pyproject.toml`

### Compression Pipeline
- [ ] `flashsvd-compress` command exists and runs
- [ ] Supports all 5 methods: standard, fwsvd, adasvd, drone, asvd
- [ ] Outputs valid compressed checkpoint + metadata JSON
- [ ] Can compress BERT and RoBERTa models
- [ ] Validation passes: compressing bert-base-uncased with standard SVD at rank 64 produces checkpoint that loads correctly

### Evaluation Pipeline
- [ ] `flashsvd-eval` command exists and runs
- [ ] Loads compressed checkpoint and evaluates on GLUE task
- [ ] Outputs metrics JSON with accuracy, memory, latency
- [ ] Validation passes: evaluating compressed bert-base-uncased on sst2 returns accuracy within 2% of reference

### CLI
- [ ] All commands have `--help` with clear descriptions
- [ ] Input validation with actionable error messages
- [ ] Progress indicators for long operations
- [ ] `flashsvd-info` displays checkpoint metadata

### Gradio UI
- [ ] `flashsvd-ui` launches web interface
- [ ] Compression tab works end-to-end
- [ ] Evaluation tab works end-to-end
- [ ] Results are downloadable

### Documentation
- [ ] README updated with installation, CLI examples, UI usage
- [ ] CLAUDE.md updated (this file)
- [ ] Docstrings in all new modules
- [ ] Example commands are copy-pasteable and tested

### Testing
- [ ] At least one end-to-end test:
  ```bash
  flashsvd-compress --model bert-base-uncased --method standard --rank 64 --output /tmp/test_compress --task sst2
  flashsvd-eval --checkpoint /tmp/test_compress --task sst2 --batch-size 32 --output /tmp/test_eval.json
  cat /tmp/test_eval.json  # Must show valid metrics
  ```
- [ ] No regressions in existing experiment scripts (spot check 2-3 scripts)

### Code Quality
- [ ] No duplicate compression/evaluation logic (DRY principle)
- [ ] UI code is thin (calls core modules only)
- [ ] Proper error handling in all user-facing commands
- [ ] Type hints in new code (`CompressConfig`, `EvalConfig`, etc.)

---

## Project Overview

FlashSVD is an end-to-end rank-aware streaming inference framework for SVD-compressed large language models. The key innovation: fused Triton kernels that stream computation through SRAM tiles rather than materializing full activation buffers, achieving up to 70.2% reduction in peak activation memory while preserving accuracy.

Paper: https://arxiv.org/abs/2508.01506

## Development Setup

### Installation

**Prerequisites**: You must manually install PyTorch with CUDA support first:

```bash
# CUDA 12.1
pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio

# or CPU only
pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision torchaudio
```

**Install project** (choose one):

```bash
# Option A: Using install script (creates .venv automatically)
./install_local.sh

# Option B: Manual venv setup
python3 -m venv .venv
source .venv/bin/activate
pip install -e .

# Option C: Conda
conda create -n flashsvd python=3.10
conda activate flashsvd
pip install -e .
```

### Environment

- Python 3.8+
- PyTorch 2.7.1+ with CUDA support (required for Triton kernels)
- Triton 3.3.1+ for GPU kernel compilation
- Linux only (WSL2 supported)

## Common Commands

**📌 DURING M1-M5**: These are existing experiment commands for reference. You will create NEW unified commands (`flashsvd-compress`, `flashsvd-eval`) that wrap this functionality.

### Training

**Interactive UI** (recommended for quick experimentation):
```bash
python app.py
```
This launches a Gradio interface for training BERT models on GLUE tasks with configurable parameters.

**Command-line training**:
```bash
# Classification (CLS) mode
python train_bert_unified_min.py \
  --mode cls --task sst2 --model bert-base-uncased \
  --epochs 3 --batch_size 32 --learning_rate 2e-5 \
  --logging_steps 100 --eval_steps 0 --seed 0

# Masked Language Modeling (MLM) mode
python train_bert_unified_min.py \
  --mode mlm --task mnli --model bert-base-uncased \
  --epochs 3 --batch_size 32 --learning_rate 2e-5
```

Supported tasks: `cola`, `sst2`, `qnli`, `qqp`, `mnli`, `stsb`, `mrpc`, `rte`

### Profiling & Inference (Experiment Scripts - Reference Only)

**📌 EXTRACTION SOURCE**: These scripts contain the logic you will extract into `src/flashsvd/compress.py` and `src/flashsvd/evaluate.py` during M2-M3. Do NOT modify these original scripts.

After training, profile different compression methods:

```bash
# BERT baseline comparisons
cd BERT/
python profile_dense.py      # Dense baseline (no compression)
python profile_svd.py         # Standard SVD + dense kernels
python profile_flashsvd.py    # Standard SVD + FlashSVD streaming kernels

# Fisher-Weighted SVD (FWSVD)
cd BERTFW/
python profile_fwsvd.py           # FWSVD + dense kernels
python profile_flashfwsvd.py      # FWSVD + FlashSVD kernels
python profile_flashfwsvd_offload.py  # With CPU offloading

# Adaptive Rank Selection (AdaSVD)
cd BERTAda/
python profile_flashsvd.py    # Reads ranks from ars_out/ranks.json

# DRONE (data-aware whitening)
cd BERTWhiten/
python profile_flashsvd.py

# RoBERTa variants
cd RoBERTa/
python profile_dense_roberta.py
python profile_svd_roberta.py
python profile_flashsvd_roberta.py

# Decoder models (GPT-2, LLaMA)
cd decoders/gpt2/
python profile_dense.py
python profile_asvd.py               # Activation-aware SVD
python profile_asvd_accum_flashsvd.py  # ASVD + FlashSVD kernels

cd decoders/llama/
python profile_dense.py
python profile_svd_kv_llama_decode.py     # KV-cache compression
python profile_asvd_flashsvd_llama_decode.py
```

### Benchmarking

Micro-benchmarks for kernel performance:

```bash
cd benchmark/
python benchmark_flashsvdattn_ranks.py     # Attention kernel sweep (batch, seq, rank)
python benchmark_flashsvdffn.py            # FFN kernel benchmarks
python benchmark_long_context_attn.py      # Long-context attention tests
python benchmark_long_context_decoder_attn.py  # Decoder attention benchmarks
python benchmark_long_context_decoder_ffn.py   # Decoder FFN benchmarks
```

Results are saved in `benchmark/benchmark/*.csv`.

## Architecture Overview

**📖 TECHNICAL REFERENCE**: This section documents the existing research codebase. During M1-M5, you will CREATE NEW structure (`src/flashsvd/`) alongside these directories, NOT modify them.

### Directory Structure

```
src/                       # Core library (main entry point)
├── kernels/              # Triton streaming kernels
│   ├── flash_attn_triton.py      # Baseline FlashAttention
│   ├── flashsvdattn.py           # Rank-aware fused attention
│   ├── flashsvdffnv1.py          # Two-stage fused FFN
│   └── flashsvdffnv2.py          # Full-batched fused FFN
└── utils/                # SVD decomposition & block wrappers
    ├── SVDBlocks.py              # Non-rank-aware SVD blocks (baseline)
    ├── FlashSVDBlocks.py         # Rank-aware FlashSVD blocks
    ├── fwsvd.py                  # Fisher-Weighted SVD
    ├── svd_helpers.py            # SVD factorization helpers
    └── metrics.py                # Accuracy, memory, timing metrics

BERT/              # Standard BERT with SVD
BERTFW/            # BERT + Fisher-Weighted SVD
BERTAda/           # BERT + Adaptive Rank Selection (NAACL 2024)
BERTWhiten/        # BERT + DRONE data-aware compression
RoBERTa/           # RoBERTa variants
RoBERTaFW/         # RoBERTa + FWSVD

ModernBERT/        # Modern BERT architectures
├── BERT_MASK/            # Standard masked attention
├── BERT_FWMASK/          # Forward-masked variant
└── BERT_LONG/            # Long-context variant

decoders/          # Autoregressive models
├── gpt2/                 # GPT-2 + SVD/ASVD
│   ├── kernels/          # Causal attention kernels
│   ├── profile_*.py      # Various profiling scripts
│   └── with_finetune/    # Fine-tuning examples
└── llama/                # LLaMA-2-7B + SVD/ASVD
    ├── asvd_rep/         # Activation-aware SVD reproduction
    ├── kernels/          # RoPE + causal kernels
    └── eval/             # Evaluation scripts

benchmark/         # Comprehensive benchmarking suite
train/             # Training utilities (older, prefer train_bert_unified_min.py)
why_finetuning/    # Ablation studies on fine-tuning impact
```

### Core Architectural Layers

FlashSVD implements a 4-layer architecture:

1. **Compression Layer**: SVD factorization methods (Standard, FWSVD, DRONE, AdaSVD, ASVD)
2. **Kernel Layer**: Triton-based streaming implementations for attention & FFN
3. **Block Layer**: PyTorch nn.Module wrappers with forward/backward compatibility
4. **Model Layer**: HuggingFace integration via LayerShim adapters

### SVD Compression Methods

The codebase supports 5 SVD-based compression approaches:

1. **Standard SVD** (`src/utils/svd_helpers.py::build_plain_svd_helpers`)
   - Baseline: `U, S, V_T = SVD(W)`, truncate to rank r
   - Factors: `U_r * sqrt(S_r)` and `sqrt(S_r) * V_r^T`

2. **Fisher-Weighted SVD (FWSVD)** (`src/utils/fwsvd.py`)
   - Paper: [Language model compression with weighted low-rank factorization](https://arxiv.org/abs/2207.00112)
   - Pre-conditions with Fisher weights before SVD
   - Better preserves important features

3. **DRONE** (`BERTWhiten/profile_flashsvd.py`)
   - Paper: [DRONE: Data-aware Low-rank Compression](https://proceedings.neurips.cc/paper/2021/)
   - Uses Cholesky decomposition of input covariance
   - Minimizes reconstruction error on actual input distribution

4. **Adaptive Rank Selection (AdaSVD)** (`BERTAda/adaptive_rank_selection.py`)
   - Paper: [Adaptive Rank Selections for Low-Rank Approximation](https://aclanthology.org/2024.naacl-long.13/)
   - Trains hypernetwork with Gumbel-Sigmoid masking
   - Automatic, data-driven rank selection under global budget
   - Outputs per-layer ranks to `ars_out/ranks.json`

5. **Activation-aware SVD (ASVD)** (`decoders/llama/asvd_rep/utils/asvd.py`)
   - Decoder-specific: compresses KV-cache in autoregressive inference
   - Calibrates activation statistics and uses binary search for optimal ranks
   - Supports quantization alongside SVD

### Streaming Kernel Architecture

**Core Innovation**: Rather than materializing full intermediate activations, FlashSVD loads small tiles of low-rank factors into SRAM, computes on-the-fly, and immediately evicts:

```python
# Traditional approach (high memory):
x [B,M,d] @ W [d,r] @ V [r,d] → temp [B,M,d] → next_op(temp)

# FlashSVD streaming approach:
for tile in x [BLOCK_M, d]:
    temp_tile = tile @ P [d,r]      # in registers
    result += temp_tile @ V [r,d]   # immediate reduction
    # temp_tile freed, SRAM reused
```

**Key Kernels**:

- `flashsvdattn.py`: Fuses Q/K/V low-rank projections into attention computation
  - Input: `Pq, Vq, bq` (Q factors), `Pk, Vk, bk` (K factors), `Pv, Vv, bv` (V factors)
  - Parameters: `block_m=64` (query tile), `block_r=64` (rank tile)

- `flashsvdffnv1.py`: Two-stage fused FFN
  - Stage 1: `S = GELU(P @ V1 + b1) @ U2` (Triton)
  - Stage 2: `output = S @ V2 + b2` (PyTorch)

- `flashsvdffnv2.py`: Full-batched FFN with more aggressive fusion

**Decoder-specific kernels**:
- `flash_attn_causal.py`: Causal masking for autoregressive models
- `flashsvdropeattn.py`: Rotary position embeddings (RoPE) + FlashSVD
- `flashsvdswiglu.py`: SwiGLU activation fusion

### Block Design Patterns

**SVDBlock classes** (`src/utils/SVDBlocks.py`):
- Non-rank-aware baseline using standard dense matrix multiplications
- Variants: `BertSVDBlock`, `RobertaSVDBlock`, `BertFWSVDBlock`, `RobertaFWSVDBlock`
- Decomposes: Q/K/V (per-head), FFN intermediate/output, attention output projection

**FlashSVDBlock classes** (`src/utils/FlashSVDBlocks.py`):
- Rank-aware with streaming kernels
- Stores factors as `P` (projections [H, d_model, r]) and `V` (lifts [H, r, dh])
- Memory benefit: intermediate tensors are [B, H, M, R] where R << d_model
- Variants: `BertFlashSVDBlock`, `BertFlashFWSVDBlock`, `RobertaFlashSVDBlock`, `RobertaFlashFWSVDBlock`

**Shim classes**: HuggingFace compatibility adapters
- `BertLayerShim`, `BertFWLayerShim`: Wrap custom blocks to match HF signatures
- Convert 4D attention masks → 2D padding masks
- Return tuple outputs for HF Trainer compatibility

## Training & Inference Pipeline

### Typical Workflow

1. **Train dense model** (or use pretrained checkpoint):
   ```bash
   python train_bert_unified_min.py --mode cls --task sst2 --model bert-base-uncased
   ```

2. **Compress model** (done in profile scripts):
   - Load trained dense checkpoint
   - Apply SVD decomposition (method-specific)
   - Replace layers with SVDBlocks or FlashSVDBlocks

3. **Evaluate**:
   - Measure accuracy, latency, peak memory
   - Compare dense vs SVD vs FlashSVD
   - Profiling scripts output metrics via `src/utils/metrics.py`

### Calibration (for DRONE/ASVD)

Some methods require calibration data to estimate statistics:
- DRONE: Collects input covariances
- ASVD: Collects activation statistics and Fisher information
- Typically done with a small sample of training data (e.g., 128 examples)

## Key Implementation Details

### Metrics & Evaluation

All profiling scripts use `src/utils/metrics.py`:

- `acc_peak_time()`: Returns (accuracy, peak_mem_MiB, latency_ms)
  - Runs inference on validation set
  - Tracks peak GPU memory allocation
  - Measures batch latency

- `compute_persistent_memory()`: Sums parameter + buffer sizes
- `summarize_dense_vs_lowrank()`: Memory breakdown by component type

### Kernel Tuning Parameters

All Triton kernels expose tile size parameters for optimization:
- `BLOCK_M`: Query/sequence tile size (typically 64)
- `BLOCK_R`: Rank tile size (typically 64)
- `BLOCK_D`: Hidden dimension tile (128 for FFN)
- `BL`, `BD`: Sequence and hidden blocks for FFN

These can be tuned for different GPU architectures or workload characteristics.

### Attention Mask Handling

- **4D masks** ([batch, heads, seq, seq]): Full attention masks from HuggingFace
- **2D masks** ([batch, seq]): Padding masks for FlashSVD kernels
- Shim classes handle conversion automatically

### ModernBERT Extensions

ModernBERT variants introduce additional components:
- **GEGLU** (`flashsvdgeglu.py`): Gated linear units with fusion
- **RoPE** (`flashsvdropeattn.py`): Rotary position embeddings
- **Fused LayerNorm** (`fused_ln.py`): Pre-norm residual connections
- Multiple variants: standard masked, forward-masked, long-context

## Development Notes

**⚠️ PRODUCTIZATION CONSTRAINT**: During Phase 1 (Milestones M1-M5), the following sections are **REFERENCE ONLY**. Do NOT perform these actions unless explicitly requested after Phase 1 completion.

### Adding New Models

**🚫 BLOCKED DURING M1-M5**: This is for future research extensions only.

To add support for a new model architecture:

1. Create SVD block class in `src/utils/SVDBlocks.py` or `src/utils/FlashSVDBlocks.py`:
   - Inherit appropriate base or copy existing pattern
   - Decompose Q/K/V and FFN layers
   - Implement forward pass with low-rank factors

2. Create shim class for HuggingFace compatibility:
   - Wrap block to match original model's layer signature
   - Handle mask conversions if needed

3. Add profiling scripts:
   - `profile_dense.py`: Baseline without compression
   - `profile_svd.py`: SVD with dense kernels
   - `profile_flashsvd.py`: SVD with streaming kernels

4. Test on representative tasks and measure accuracy/memory/latency

### Adding New SVD Methods

**🚫 BLOCKED DURING M1-M5**: This is for future research extensions only.

To integrate a new SVD-based compression technique:

1. Implement factorization in `src/utils/svd_helpers.py` or new file:
   - Follow pattern of `build_plain_svd_helpers()` or `build_fwsvd_helpers()`
   - Return tuple of (U_factor, V_factor, bias) per layer

2. Create Block class variant if needed (or reuse existing FlashSVDBlock)

3. Add profiling script in appropriate directory

4. Update CLAUDE.md with new method details

### Kernel Development

**🚫 BLOCKED DURING M1-M5**: Kernel modifications are prohibited unless fixing critical bugs with user approval.

When modifying Triton kernels:
- Test correctness against PyTorch reference implementation
- Profile memory usage with `torch.cuda.max_memory_allocated()`
- Benchmark latency across different batch/sequence/rank configurations
- Verify numerical stability (especially for attention softmax)

### Debugging Tips

- **Memory errors**: Check BLOCK_M/BLOCK_R parameters don't exceed SRAM limits
- **Numerical issues**: Verify mask handling and softmax numerics
- **Performance**: Use Triton's autotuning for optimal tile sizes
- **Compatibility**: Test with different HuggingFace Transformers versions

## Contact & Support

- Issues and bugs: Open a GitHub issue
- Feature requests (new models/methods): Open an issue with details
- Questions or collaboration: Email Zishan at zs89@duke.edu
