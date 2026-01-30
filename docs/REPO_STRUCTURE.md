# FlashSVD Repository Structure After Refactoring

## 📁 Complete Directory Tree

```
FlashSVD/
├── 📦 Core Package (Production Layer - M1-M5 Complete)
│   └── src/
│       ├── __init__.py                          # Mark src as package
│       │
│       ├── flashsvd/                            # ✨ Main package (pip install flashsvd)
│       │   ├── __init__.py                      # Version: 0.1.0, module exports
│       │   │
│       │   ├── cli.py                           # M4: Unified CLI entry (flashsvd)
│       │   ├── compress.py                      # M2: Compression pipeline main entry
│       │   ├── evaluate.py                      # M3: Evaluation pipeline main entry
│       │   ├── info.py                          # M4: Checkpoint info display
│       │   ├── io.py                            # M2: Model load/save + structure recovery
│       │   │
│       │   ├── compression/                     # M2: Compression method implementations
│       │   │   ├── __init__.py                  # compress_model() dispatcher
│       │   │   ├── _metadata.py                 # compression_info.json generation
│       │   │   ├── registry.py                  # Method registry
│       │   │   ├── method_args.py               # Method argument validation
│       │   │   ├── standard_svd.py              # Standard SVD (BERT)
│       │   │   ├── roberta_svd.py               # RoBERTa-specific SVD
│       │   │   ├── fwsvd.py                     # Fisher-Weighted SVD
│       │   │   ├── adasvd.py                    # Adaptive Rank Selection
│       │   │   └── whiten.py                    # DRONE (Data-Aware Whitening)
│       │   │
│       │   ├── finetune/                        # ✨ M6: Fine-tuning module (new)
│       │   │   ├── __init__.py
│       │   │   ├── config.py                    # FinetuneConfig dataclass
│       │   │   └── trainer.py                   # Fine-tuning trainer
│       │   │
│       │   ├── kernels/                         # M1: Kernel wrapper layer (thin)
│       │   │   └── __init__.py                  # Re-export src.kernels.*
│       │   │
│       │   ├── utils/                           # M1: Utils wrapper layer (thin)
│       │   │   └── __init__.py                  # Re-export src.utils.*
│       │   │
│       │   └── ui/                              # M5: Gradio Web UI
│       │       ├── __init__.py
│       │       └── app.py                       # Gradio interface (3 tabs: compress/eval/info)
│       │
│       ├── kernels/                             # Original Triton kernels (research impl, keep unchanged)
│       │   ├── flash_attn_triton.py             # FlashAttention baseline
│       │   ├── flashsvdattn.py                  # Rank-aware Fused Attention
│       │   ├── flashsvdffnv1.py                 # FFN v1 (two-stage fusion)
│       │   └── flashsvdffnv2.py                 # FFN v2 (full fusion, theoretically optimal)
│       │
│       └── utils/                               # Original SVD utilities (research impl, keep unchanged)
│           ├── SVDBlocks.py                     # Non-rank-aware blocks (baseline)
│           ├── FlashSVDBlocks.py                # Rank-aware blocks (core)
│           ├── fwsvd.py                         # FWSVD math implementation
│           ├── svd_helpers.py                   # SVD decomposition helpers
│           ├── metrics.py                       # Evaluation metrics (acc_peak_time)
│           └── kernel_api.py                    # Kernel API interface
│
├── 🧪 Experiment Directories (Research Code Archive)
│   ├── experiments/                             # ✨ Reorganized encoder experiments
│   │   ├── BERT/                                # Standard BERT + SVD
│   │   │   ├── profile_dense.py                 # Dense baseline performance
│   │   │   ├── profile_svd.py                   # SVD + dense kernels
│   │   │   └── profile_flashsvd.py              # SVD + FlashSVD kernels
│   │   │
│   │   ├── BERTFW/                              # BERT + Fisher-Weighted SVD
│   │   │   ├── profile_dense.py
│   │   │   ├── profile_fwsvd.py                 # FWSVD + dense kernels
│   │   │   ├── profile_flashfwsvd.py            # FWSVD + FlashSVD kernels
│   │   │   └── profile_flashfwsvd_offload.py    # With CPU offloading
│   │   │
│   │   ├── BERTAda/                             # BERT + Adaptive Rank Selection
│   │   │   ├── adaptive_rank_selection.py       # Rank selection training
│   │   │   ├── ars_out/ranks.json               # Output rank configuration
│   │   │   └── profile_flashsvd.py              # Using adaptive ranks
│   │   │
│   │   ├── BERTWhiten/                          # BERT + DRONE (Whitening)
│   │   │   ├── profile_dense.py
│   │   │   ├── profile_svd.py
│   │   │   └── profile_flashsvd.py
│   │   │
│   │   ├── RoBERTa/                             # RoBERTa variants
│   │   │   ├── profile_dense_roberta.py
│   │   │   ├── profile_svd_roberta.py
│   │   │   └── profile_flashsvd_roberta.py
│   │   │
│   │   ├── RoBERTaFW/                           # RoBERTa + FWSVD
│   │   │   ├── profile_dense_roberta.py
│   │   │   ├── profile_fwsvd_roberta.py
│   │   │   └── profile_flashfwsvd_roberta.py
│   │   │
│   │   └── ModernBERT/                          # ModernBERT architecture
│   │       ├── BERT_MASK/                       # Standard masked attention
│   │       │   ├── run_modernbert.py
│   │       │   ├── run_modernbert_flashsvd.py
│   │       │   └── run_modernbert_svd.py
│   │       ├── BERT_FWMASK/                     # Forward-masked variant
│   │       │   ├── run_modernbert_flashfwsvd.py
│   │       │   └── run_modernbert_fwsvd.py
│   │       ├── BERT_LONG/                       # Long-context variant
│   │       │   └── profile_imdb.py
│   │       ├── eval_modernbert.py
│   │       ├── train_modernbert.py
│   │       └── train_modernbert_long.py
│   │
│   └── legacy/                                  # ✨ Old file archive (M0 cleanup)
│       ├── BERT/                                # Original BERT experiments before move
│       ├── BERTAda/
│       ├── BERTFW/
│       ├── BERTWhiten/
│       ├── RoBERTa/
│       ├── RoBERTaFW/
│       ├── ModernBERT/
│       ├── app.py                               # Old Gradio training UI
│       ├── train_bert_unified_min.py            # Old unified training script
│       └── utils_nlp.py                         # Old NLP utilities
│
├── 🔬 Decoder Experiments (Keep in root, independently maintained)
│   ├── decoders/
│   │   ├── gpt2/                                # GPT-2 + SVD/ASVD
│   │   │   ├── kernels/                         # Causal attention kernels
│   │   │   │   ├── flash_attn_causal.py
│   │   │   │   ├── flashsvdattn.py
│   │   │   │   ├── flashsvdffn.py
│   │   │   │   └── utils_mask.py
│   │   │   ├── with_finetune/                   # Fine-tuning examples
│   │   │   │   ├── finetune_lowrank.py
│   │   │   │   ├── infer_lowrank.py
│   │   │   │   └── lowrank_gpt2.py
│   │   │   ├── profile_dense.py
│   │   │   ├── profile_asvd.py                  # Activation-aware SVD
│   │   │   ├── profile_asvd_accum_flash.py
│   │   │   ├── profile_asvd_accum_flashsvd.py
│   │   │   └── profile_svd_kv.py                # KV-cache compression
│   │   │
│   │   └── llama/                               # LLaMA-2-7B + SVD/ASVD
│   │       ├── asvd_rep/                        # ASVD method reproduction
│   │       │   ├── huggingface_repos/           # HF model integration
│   │       │   ├── modules/svd_linear.py
│   │       │   ├── utils/                       # ASVD utilities
│   │       │   └── profile_*.py
│   │       ├── kernels/                         # RoPE + causal kernels
│   │       │   ├── flash_attn_causal.py
│   │       │   ├── flashsvdropeattn.py          # RoPE + FlashSVD
│   │       │   └── flashsvdswiglu.py            # SwiGLU fusion
│   │       ├── eval/
│   │       │   ├── profile_asvd_flashsvd_llama.py
│   │       │   └── profile_asvd_llama.py
│   │       └── profile_*.py
│   │
│   ├── benchmark/                               # Kernel performance micro-benchmarks
│   │   ├── encoder_kernel/                      # Encoder kernel benchmarks
│   │   │   ├── flash_attn_triton.py
│   │   │   ├── flashsvdattn.py
│   │   │   ├── flashsvdffn.py
│   │   │   ├── flashsvdffnv1.py
│   │   │   └── utils_mask.py
│   │   ├── decoder_kernel/                      # Decoder kernel benchmarks
│   │   │   ├── flash_attn_causal.py
│   │   │   ├── flashsvdropeattn.py
│   │   │   └── flashsvdswiglu.py
│   │   ├── benchmark/                           # CSV result outputs
│   │   │   ├── decoder_attn_decode.csv
│   │   │   ├── decoder_attn_prefill.csv
│   │   │   ├── decoder_ffn_long_context.csv
│   │   │   └── long_context_ffn.csv
│   │   ├── benchmark_flashsvdattn_ranks.py      # Attention rank sweep
│   │   ├── benchmark_flashsvdffn.py             # FFN benchmarks
│   │   ├── benchmark_long_context_attn.py       # Long-context attention
│   │   ├── benchmark_long_context_decoder_attn.py
│   │   ├── benchmark_long_context_decoder_ffn.py
│   │   └── benchmark_long_context_ffn.py
│   │
│   ├── train/                                   # Training utilities (old, reference only)
│   │   ├── train_bert.py
│   │   ├── train_bert_mlm.py
│   │   ├── train_roberta.py
│   │   ├── train_roberta_large.py
│   │   └── train_roberta_mlm.py
│   │
│   └── why_finetuning/                          # Fine-tuning ablation studies
│       ├── kernel/                              # Fine-tuning-specific kernels
│       │   ├── flash_attn_triton.py
│       │   ├── flashsvdattn.py
│       │   └── flashsvdffn*.py
│       ├── finetune_svd.py                      # Fine-tuning experiment scripts
│       ├── finetune_svd_saveable.py
│       ├── inference_svd_model.py
│       └── profile_*.py
│
├── 🧰 Test Suite (M4-M6 New)
│   └── test/
│       ├── scripts/                             # Test scripts
│       │   ├── test_all_methods.py              # Python test suite
│       │   ├── test_all_methods.sh              # Full test workflow
│       │   ├── test_compression_only.sh         # Compression-only tests
│       │   ├── test_compression_r64_flat.sh     # Rank=64 flat tests
│       │   ├── test_finetuned_models.py         # Fine-tuned model tests
│       │   ├── test_finetuned_organization.py   # Directory structure tests
│       │   └── test_svd_reconstruction.py       # SVD reconstruction tests
│       │
│       ├── logs/                                # Test log outputs
│       │   ├── test_YYYYMMDD_HHMMSS.log
│       │   └── ...
│       │
│       ├── results/                             # Test results
│       │   ├── test_report_YYYYMMDD.json
│       │   ├── benchmark_summary.csv
│       │   └── ...
│       │
│       └── docs/                                # Test documentation
│           ├── TEST_PLAN.md
│           └── IMPLEMENTATION_NOTES.md
│
├── 📁 Models and Outputs
│   ├── models/                                  # Fine-tuned model storage
│   │   ├── README.md                            # Directory structure explanation
│   │   └── bert-sst2-finetuned/                 # Example: fine-tuned BERT
│   │       ├── config.json
│   │       ├── pytorch_model.bin
│   │       ├── tokenizer_config.json
│   │       └── ...
│   │
│   ├── compressed_models/                       # ✨ Compressed model outputs (auto-created)
│   │   └── bert-base-uncased_fwsvd_r64/         # Example: compressed checkpoint
│   │       ├── config.json                      # HF model config
│   │       ├── model.safetensors                # HF weights
│   │       ├── flashsvd_state_dict.pt           # FlashSVD state (structure recovery)
│   │       └── compression_info.json            # Compression metadata
│   │
│   ├── compression_test/                        # Compression test outputs
│   │   ├── standard/
│   │   ├── fw/
│   │   ├── ada/
│   │   └── whiten/
│   │
│   ├── compression_test_r64_flat/               # Rank=64 flat tests
│   │   ├── standard/
│   │   ├── fw/
│   │   ├── whiten/
│   │   └── results/
│   │
│   ├── test_output/                             # Test temporary outputs
│   │   ├── bert/
│   │   └── cli_test/
│   │
│   └── figs/                                    # Figure resources
│       └── ...
│
├── 📝 Configuration Files
│   ├── pyproject.toml                           # ✨ M1: Modern packaging config (PEP 621)
│   │   # [project]
│   │   #   name = "flashsvd"
│   │   #   version = "0.1.0"
│   │   #   dependencies = [torch, transformers, ...]
│   │   # [project.scripts]
│   │   #   flashsvd = "flashsvd.cli:main"
│   │   #   flashsvd-compress = "flashsvd.compress:main"
│   │   #   flashsvd-eval = "flashsvd.evaluate:main"
│   │   #   flashsvd-info = "flashsvd.info:main"
│   │   #   flashsvd-finetune = "flashsvd.finetune:main"
│   │   #   flashsvd-ui = "flashsvd.ui.app:main"
│   │
│   ├── requirements.txt                         # Dependency list
│   ├── environment.yml                          # Conda environment config
│   ├── .gitignore                               # Git ignore rules
│   └── install_local.sh                         # Local installation script
│
├── 📚 Documentation (M4-M6 Enhanced)
│   ├── README.md                                # ✨ Main README (updated: CLI+UI+benchmarks)
│   ├── CLAUDE.md                                # ✨ Project guidance doc (must read for dev!)
│   ├── CHANGELOG.md                             # ✨ Version changelog
│   ├── CONTRIBUTING.md                          # ✨ Contribution guide
│   ├── LICENSE                                  # MIT License
│   │
│   ├── REPO_STRUCTURE.md                        # ✨ This document (directory structure)
│   ├── README_OLD_BACKUP.md                     # Old README backup
│   │
│   ├── QUICK_START_UI.md                        # ✨ UI quick start guide
│   ├── M5_UI_GUIDE.md                           # ✨ M5 UI detailed usage guide
│   ├── FINETUNED_MODEL_ORGANIZATION.md          # ✨ Fine-tuned model organization
│   ├── DATASET_GUIDE.md                         # ✨ Dataset usage guide
│   └── DATASET_QUICK_REFERENCE.md               # ✨ Dataset quick reference
│
└── 📄 Other Files
    ├── 2508.01506v1.pdf                         # FlashSVD paper PDF
    └── ...

```

---

## 📊 Directory Responsibilities

### 🎯 Product Layer (User Interaction)

| Directory/File | Responsibility | User-Visible | Status |
|----------------|----------------|--------------|--------|
| `src/flashsvd/cli.py` | Unified CLI entry | ✅ `flashsvd` | ✅ M4 Complete |
| `src/flashsvd/compress.py` | Compression API | ✅ `flashsvd compress` | ✅ M2 Complete |
| `src/flashsvd/evaluate.py` | Evaluation API | ✅ `flashsvd eval` | ✅ M3 Complete |
| `src/flashsvd/info.py` | Info display | ✅ `flashsvd info` | ✅ M4 Complete |
| `src/flashsvd/finetune/` | Fine-tuning API | ✅ `flashsvd finetune` | ✅ M6 Complete |
| `src/flashsvd/ui/app.py` | Web interface | ✅ `flashsvd-ui` | ✅ M5 Complete |

**Backward Compatibility**: Standalone commands `flashsvd-compress`, `flashsvd-eval`, `flashsvd-info`, `flashsvd-finetune` still available.

### 🧩 Business Logic Layer (Method Implementation)

| Directory/File | Responsibility | Called By | Status |
|----------------|----------------|-----------|--------|
| `src/flashsvd/compression/*.py` | Compression method implementations | compress.py | ✅ M2 Complete |
| `src/flashsvd/compression/registry.py` | Method registration & dispatch | compress.py | ✅ M2 Complete |
| `src/flashsvd/io.py` | Model load/save | compress.py, evaluate.py | ✅ M2 Complete |
| `src/flashsvd/finetune/trainer.py` | Fine-tuning trainer | finetune module | ✅ M6 Complete |

**Supported Compression Methods** (M2.0 Scope):
- ✅ `standard`: Standard SVD (BERT/RoBERTa)
- ✅ `fwsvd` / `fw`: Fisher-Weighted SVD
- ✅ `whiten` / `drone`: Data-aware Whitening (DRONE)
- ✅ `adasvd` / `ada`: Adaptive Rank Selection
- 🔜 `asvd`: Activation-aware SVD (decoders, future)

### 🔧 Low-Level Implementation (Original Research Code)

| Directory | Responsibility | Type | Modification Constraints |
|-----------|----------------|------|-------------------------|
| `src/kernels/` | Triton GPU kernels | Research impl | ❌ Do NOT modify (unless bugfix) |
| `src/utils/` | SVD math/blocks | Research impl | ❌ Do NOT modify (unless bugfix) |

### 🧪 Experiment Code (Reference and Archive)

| Directory | Responsibility | Status | Purpose |
|-----------|----------------|--------|---------|
| `experiments/BERT*/` | Encoder experiments | ✅ Moved & preserved | Reference & logic extraction |
| `experiments/ModernBERT/` | ModernBERT experiments | ✅ Moved & preserved | Reference implementation |
| `legacy/` | Old file archive | ✅ Archived | Historical reference |
| `decoders/` | Decoder experiments | ✅ Keep in root | Independent maintenance |
| `benchmark/` | Performance benchmarks | ✅ Keep in root | Continuous benchmarking |

### 🧰 Testing and Validation

| Directory | Responsibility | Status |
|-----------|----------------|--------|
| `test/scripts/` | Test scripts | ✅ M4-M6 Complete |
| `test/logs/` | Test logs | ✅ Auto-generated |
| `test/results/` | Test results | ✅ Auto-generated |
| `compression_test/` | Compression test outputs | ✅ Auto-created |

---

## 🔄 Code Flow Relationships

### Compression Flow

```
User
  ↓
flashsvd compress --model bert-base-uncased --task sst2 --method fwsvd --rank 64
  ↓
src/flashsvd/cli.py (parse command)
  ↓
src/flashsvd/compress.py::run_compress(CompressConfig)
  ↓
src/flashsvd/compression/__init__.py::compress_model() (dispatcher)
  ↓
src/flashsvd/compression/fwsvd.py::compress_bert_fwsvd() (method implementation)
  ↓
src/flashsvd/utils (wrapper layer) → src/utils/fwsvd.py (FWSVD math)
  ↓
src/flashsvd/utils (wrapper layer) → src/utils/FlashSVDBlocks.py (block construction)
  ↓
src/flashsvd/kernels (wrapper layer) → src/kernels/flashsvdattn.py (Triton kernel)
  ↓
GPU execution
  ↓
src/flashsvd/io.py::save_compressed() (save checkpoint)
  ↓
compressed_models/bert-base-uncased_fwsvd_r64/
  ├── config.json (HF config + compression metadata)
  ├── model.safetensors (HF weights)
  ├── flashsvd_state_dict.pt (FlashSVD state)
  └── compression_info.json (compression metadata)
```

### Evaluation Flow

```
User
  ↓
flashsvd eval --checkpoint ./compressed_models/bert-base-uncased_fwsvd_r64 --task sst2
  ↓
src/flashsvd/cli.py
  ↓
src/flashsvd/evaluate.py::run_eval(EvalConfig)
  ↓
src/flashsvd/io.py::load_compressed() (load model + structure recovery)
  ↓
src/utils/metrics.py::acc_peak_time() (evaluation metrics)
  ↓
Output JSON:
{
  "task": "sst2",
  "metric_name": "accuracy",
  "metric_value": 0.8991,
  "peak_memory_mib": 542,
  "latency_ms": 139.6,
  ...
}
```

### Fine-tuning Flow

```
User
  ↓
flashsvd finetune --checkpoint <compressed_model> --task sst2 --epochs 3
  ↓
src/flashsvd/cli.py
  ↓
src/flashsvd/finetune/__init__.py::main()
  ↓
src/flashsvd/finetune/trainer.py::train()
  ↓
models/finetuned/bert/fwsvd/<model_name>/
  ├── best/ (best checkpoint - use this!)
  ├── checkpoint-<epoch>-<step>/
  └── tensorboard/
```

---

## 📦 Package Structure After Installation

### Installation Commands

```bash
# Install PyTorch (must install first)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install FlashSVD
cd FlashSVD
pip install -e .
```

### Python Import Paths

```python
# Version info
from flashsvd import __version__
print(__version__)  # "0.1.0"

# Compression API
from flashsvd.compress import CompressConfig, run_compress
config = CompressConfig(model="bert-base-uncased", task="sst2", method="fwsvd", rank=64)
run_compress(config)

# Evaluation API
from flashsvd.evaluate import EvalConfig, run_eval
eval_config = EvalConfig(checkpoint="./compressed_models/...", task="sst2")
results = run_eval(eval_config)

# Fine-tuning API
from flashsvd.finetune import FinetuneConfig
from flashsvd.finetune.trainer import train
ft_config = FinetuneConfig(checkpoint="./compressed_models/...", task="sst2")
train(ft_config)

# Compression method dispatcher
from flashsvd.compression import compress_model
compressed_model = compress_model(model, method="fwsvd", ranks={"attn": 64, "ffn": 384, "wo": 384})

# Low-level utilities (advanced users)
from flashsvd.utils import SVDBlocks, FlashSVDBlocks
from flashsvd.kernels import flashsvdattn
```

### Command-Line Tools

```bash
# Unified CLI (recommended)
flashsvd --help
flashsvd compress --help
flashsvd eval --help
flashsvd info --help
flashsvd finetune --help

# Standalone commands (backward compatible)
flashsvd-compress --help
flashsvd-eval --help
flashsvd-info --help
flashsvd-finetune --help
flashsvd-ui  # Launch web interface

# Example: Complete workflow
flashsvd compress --model textattack/bert-base-uncased-SST-2 --task sst2 --method fwsvd --rank 64
flashsvd eval --checkpoint ./compressed_models/bert-base-uncased-SST-2_fwsvd_r64 --task sst2
flashsvd finetune --checkpoint ./compressed_models/bert-base-uncased-SST-2_fwsvd_r64 --task sst2 --epochs 3
flashsvd info ./compressed_models/bert-base-uncased-SST-2_fwsvd_r64
```

---

## 🎯 Design Principles

### ✅ Principles Followed

1. **Clear Layering** (Product → Business → Implementation)
   - **Product Layer**: `flashsvd/` (user interaction: CLI, UI, API)
   - **Business Layer**: `compression/`, `finetune/` (method implementations)
   - **Implementation Layer**: `utils/`, `kernels/` (research code)

2. **Backward Compatibility** (don't break old code)
   - Original experiment code preserved in `experiments/` and `decoders/`
   - Standalone commands (`flashsvd-compress` etc.) still available
   - New code calls old code via wrapper layers, doesn't modify old implementations

3. **Thin Wrapper Principle** (avoid duplicate implementation)
   - `flashsvd.utils` → `src.utils` (re-export)
   - `flashsvd.kernels` → `src.kernels` (re-export)
   - Compression methods extracted from experiment scripts, not rewritten

4. **DRY Principle** (Don't Repeat Yourself)
   - Compression logic implemented once in `compression/`
   - CLI/UI call same core functions (`run_compress`, `run_eval`)
   - Evaluation metrics reuse `src/utils/metrics.py`

5. **Extensibility**
   - New methods: add to `compression/`, register in `registry.py`
   - New architectures: inherit from `SVDBlock` base class
   - New tasks: extend `GLUE_TASKS` list

6. **Documentation-Driven**
   - `CLAUDE.md`: Development must-read (execution contract, milestones, prohibitions)
   - `REPO_STRUCTURE.md`: Directory structure (this document)
   - `CONTRIBUTING.md`: Contribution guide
   - `DATASET_GUIDE.md`: Dataset usage
   - `M5_UI_GUIDE.md`: UI usage guide

---

## 🚀 Key Improvements (M0 → M6)

| Improvement | Before (Pre-M0) | After (M1-M6 Complete) | Milestone |
|-------------|-----------------|------------------------|-----------|
| **Package Structure** | None, scripts only | ✅ `pip install flashsvd` | M1 |
| **Command Line** | None | ✅ `flashsvd compress/eval/info/finetune` | M2-M4, M6 |
| **Web UI** | Old Gradio training UI | ✅ `flashsvd-ui` (compress/eval/info) | M5 |
| **Compression Methods** | Scattered in experiment dirs | ✅ Unified in `compression/` + registry | M2 |
| **Evaluation Pipeline** | Each script independently implemented | ✅ Unified `run_eval()` + JSON output | M3 |
| **Fine-tuning Pipeline** | Scattered implementations | ✅ Unified `finetune/` module + auto-organization | M6 |
| **Experiment Code** | Root directory chaos | ✅ `experiments/` organized, `legacy/` archived | M0 |
| **Import Paths** | Inconsistent (`from src.*`) | ✅ Unified `from flashsvd.*` | M1 |
| **Documentation** | README only | ✅ 8 documentation files (CLAUDE, CONTRIBUTING, etc.) | M4-M6 |
| **Testing** | None | ✅ Complete test suite (`test/scripts/`) | M4-M6 |
| **Checkpoint Format** | Inconsistent | ✅ HF `save_pretrained()` + metadata | M2 |

---

## 📝 Important Documentation Index

### Must-Read Documents (Before Development)

1. **CLAUDE.md**: Project execution contract, milestone definitions, prohibitions ⚠️
2. **REPO_STRUCTURE.md**: This document, directory structure and design principles
3. **README.md**: User documentation, installation and usage

### Quick Start (Users)

1. **README.md**: Installation and basic usage
2. **QUICK_START_UI.md**: Web UI quick start
3. **M5_UI_GUIDE.md**: Detailed UI usage guide

### Development Guide (Contributors)

1. **CONTRIBUTING.md**: Contribution guide and code standards
2. **CLAUDE.md**: Development constraints and milestones
3. **DATASET_GUIDE.md**: Dataset usage and extension

### Reference Documentation

1. **FINETUNED_MODEL_ORGANIZATION.md**: Fine-tuned model directory organization
2. **DATASET_QUICK_REFERENCE.md**: Dataset quick reference
3. **CHANGELOG.md**: Version changelog

---

## ✅ Current Status (2026-01-30)

### Completed (M1-M6)

- ✅ **M1**: Package structure (`pip install -e .` available)
- ✅ **M2**: Compression pipeline (standard SVD, FWSVD, Whiten, AdaSVD for encoders)
- ✅ **M3**: Evaluation pipeline (unified JSON output)
- ✅ **M4**: CLI interface (validation, progress bars, error handling)
- ✅ **M5**: Gradio UI (3 tabs: compress/eval/info)
- ✅ **M6**: Fine-tuning pipeline (auto-organization, best checkpoint saving)

### Future Extensions

- 🔜 **M2.1**: Decoder compression (ASVD for GPT/LLaMA)
- 🔜 **M2.2**: ModernBERT support
- 🔜 **M7**: PyPI release (`pip install flashsvd`)
- 🔜 **M8**: Docker image
- 🔜 **M9**: Multi-GPU training support

---

## 🔍 Quick File Reference

```bash
# Core product code
ls src/flashsvd/*.py                    # CLI, compress, eval, info
ls src/flashsvd/compression/*.py        # Compression method implementations
ls src/flashsvd/finetune/*.py           # Fine-tuning module
ls src/flashsvd/ui/*.py                 # Web UI

# Low-level implementation (research code)
ls src/kernels/*.py                     # Triton kernels
ls src/utils/*.py                       # SVD math and blocks

# Experiment code (reference)
ls experiments/BERT/*.py                # BERT experiments
ls experiments/BERTFW/*.py              # FWSVD experiments
ls decoders/gpt2/*.py                   # GPT-2 experiments
ls decoders/llama/*.py                  # LLaMA experiments

# Testing
ls test/scripts/*.sh                    # Shell test scripts
ls test/scripts/*.py                    # Python test suite

# Documentation
ls *.md                                 # All Markdown documentation

# Configuration
cat pyproject.toml                      # Package configuration
cat requirements.txt                    # Dependency list
```

---

**Last Updated**: 2026-01-30
**Version**: v0.1.0
**Status**: M1-M6 Complete, Production Ready
