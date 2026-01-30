# FlashSVD: Memory-Efficient Inference with Streaming for Low-Rank Models

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyPI version](https://img.shields.io/badge/pypi-v0.1.0-blue)](https://pypi.org/project/flashsvd/)
[![GitHub Stars](https://img.shields.io/github/stars/Zishan-Shao/FlashSVD?style=social)](https://github.com/Zishan-Shao/FlashSVD)

**FlashSVD** is an end-to-end rank-aware streaming inference framework for SVD-compressed large language models. It eliminates activation memory overhead during inference through fused Triton kernels, achieving up to **70.2% reduction** in peak activation memory and **75% reduction** in intermediate transient memory while preserving accuracy.

**📄 Paper**: [FlashSVD: Memory-Efficient Inference with Streaming for Low-Rank Models](https://arxiv.org/abs/2508.01506)

> 🙏 If you find FlashSVD useful, please consider **citing our paper** and **starring 🌟 this repository**!
>
> [![Cite FlashSVD](https://img.shields.io/badge/Cite-FlashSVD-brightgreen)](#citation) [![Star this repo](https://img.shields.io/badge/Star-This%20repo-yellow?logo=github)](https://github.com/Zishan-Shao/FlashSVD/stargazers)

---

## 🔥 What's New (v0.1.0)

- ✅ **Paper Published**: [FlashSVD paper](https://arxiv.org/abs/2508.01506) demonstrating 70.2% peak memory reduction and 75% transient memory reduction
- ✅ **Unified CLI**: Single `flashsvd` command with intuitive subcommands
- ✅ **Pip installable**: `pip install -e .` for easy setup
- ✅ **Web UI**: Interactive Gradio interface for compression and evaluation
- ✅ **Organized outputs**: Automatic directory structure for compressed and finetuned models
- ✅ **Comprehensive testing**: Complete test suite with automated benchmarks
- ✅ **4 compression methods**: Standard SVD, FWSVD, Whiten (DRONE), AdaSVD

---

## 📊 Quick Performance Overview

**Compression Methods Comparison** (BERT-base on SST-2, batch=64, 25% params):

| Method | Accuracy | Peak Memory | Transient Mem | Latency | Best For |
|--------|----------|-------------|---------------|---------|----------|
| **Baseline (Dense)** | 92.3% | 695 MiB | 277 MiB | 79.7 ms | - |
| **Vanilla SVD** | 85.2% | 742 MiB ❌ | 409 MiB | 161 ms | - |
| **FlashSVD (Standard)** | 85.2% | 577 MiB | 244 MiB | 189 ms | Memory-efficient |
| **FWSVD** | 89.9% | 914 MiB ❌ | 581 MiB | 83.8 ms | - |
| **FlashFWSVD** | 89.9% | **542 MiB** 🏆 | **209 MiB** 🏆 | 140 ms | Best overall |

📈 *FlashFWSVD achieves **70.2% reduction** in peak memory and **75% reduction** in transient memory vs. Vanilla SVD. See full benchmarks in the [paper](https://arxiv.org/abs/2508.01506).*

---

## 🚀 Quick Start

### Installation

**Prerequisites**: Install PyTorch with CUDA support first:

```bash
# CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Then install FlashSVD
git clone https://github.com/Zishan-Shao/FlashSVD.git
cd FlashSVD
pip install -e .
```

### Basic Usage (3 Steps)

```bash
# 1. Compress a model
flashsvd compress \
  --model textattack/bert-base-uncased-SST-2 \
  --task sst2 \
  --method fwsvd \
  --rank 64

# 2. Evaluate compressed model
flashsvd eval \
  --checkpoint ./compressed_models/bert-base-uncased-SST-2_fwsvd_r64 \
  --task sst2 \
  --batch-size 16

# 3. Fine-tune (optional, +2-5% accuracy boost)
flashsvd finetune \
  --checkpoint ./compressed_models/bert-base-uncased-SST-2_fwsvd_r64 \
  --task sst2 \
  --epochs 3
```

### Web UI (Recommended for Beginners)

```bash
flashsvd-ui
# Opens at http://localhost:7860
```

The UI provides an intuitive interface for:
- ✨ Model compression with all methods
- 📊 Evaluation and benchmarking
- 🔧 Fine-tuning compressed models
- 📁 Checkpoint management

---

## 🧩 Installation Options

### Option A: Quick Install (Recommended)

```bash
git clone https://github.com/Zishan-Shao/FlashSVD.git
cd FlashSVD
pip install torch --index-url https://download.pytorch.org/whl/cu121  # Choose your CUDA version
pip install -e .
```

### Option B: Using Install Script

```bash
./install_local.sh  # Creates .venv automatically
source .venv/bin/activate
```

### Option C: Conda Environment

```bash
conda env create -f environment.yml
conda activate flashsvd
pip install -e .
```

---

## 💻 CLI Usage

### Compress Command

Compress a pretrained model using SVD-based methods:

```bash
flashsvd compress \
  --model bert-base-uncased \
  --task sst2 \
  --method fwsvd \
  --rank 64 \
  --device cuda

# Output: ./compressed_models/bert-base-uncased_fwsvd_r64/
```

**Supported methods**:
- `standard`: Truncated SVD (baseline)
- `fwsvd` / `fw`: Fisher-Weighted SVD (best memory efficiency)
- `whiten` / `drone`: Data-aware whitening (best accuracy)
- `adasvd` / `ada`: Adaptive rank selection
- `asvd`: Activation-aware SVD (for decoders)

**Rank configuration**:
```bash
# Unified rank (recommended)
--rank 64  # Applies 6x scaling: attn=64, ffn=384, wo=384

# Separate ranks
--rank-attn 64 --rank-ffn 384 --rank-wo 384
```

### Evaluate Command

Evaluate a compressed model on GLUE tasks:

```bash
flashsvd eval \
  --checkpoint ./compressed_models/bert-base-uncased_fwsvd_r64 \
  --task sst2 \
  --batch-size 16 \
  --output ./results/eval.json

# Returns: accuracy, peak memory, latency
```

### Finetune Command

Fine-tune compressed models to recover accuracy:

```bash
flashsvd finetune \
  --checkpoint ./compressed_models/bert-base-uncased_fwsvd_r64 \
  --task sst2 \
  --epochs 3 \
  --learning-rate 3e-5 \
  --batch-size 32

# Output: models/finetuned/bert/fwsvd/bert-base-uncased_fwsvd_r64/
#   ├── best/                  # Best checkpoint (use this!)
#   ├── checkpoint-2-1500/     # Periodic checkpoints
#   └── tensorboard/           # Training logs
```

**Automatic organization**: Finetuned models are automatically organized by architecture and method:
```
models/finetuned/
├── bert/
│   ├── standard/
│   ├── fwsvd/
│   └── whiten/
├── modernbert/
└── roberta/
```

See [`FINETUNED_MODEL_ORGANIZATION.md`](docs/FINETUNED_MODEL_ORGANIZATION.md) for details.

### Info Command

Inspect checkpoint metadata:

```bash
flashsvd info ./compressed_models/bert-base-uncased_fwsvd_r64

# Displays:
# - Compression method and ranks
# - Model architecture and size
# - Timestamp and version info
# - File list and sizes
```

---

## 🎯 Supported Models & Methods

### Encoder Models

| Architecture | Standard SVD | FWSVD | Whiten | AdaSVD |
|--------------|--------------|-------|--------|--------|
| **BERT** | ✅ | ✅ | ✅ | ✅ |
| **RoBERTa** | ✅ | ✅ | ✅ | ✅ |
| **ModernBERT** | ✅ | ✅ | ✅ | - |

### Decoder Models

| Architecture | Standard SVD | ASVD | Notes |
|--------------|--------------|------|-------|
| **GPT-2** | ✅ | ✅ | KV-cache compression |
| **LLaMA-2-7B** | ✅ | ✅ | KV-cache compression |

### Compression Methods

1. **Standard SVD**: Baseline truncated SVD decomposition
2. **FWSVD**: Fisher-Weighted SVD - pre-conditions with Fisher information
3. **Whiten (DRONE)**: Data-aware whitening via Cholesky decomposition
4. **AdaSVD**: Adaptive per-layer rank selection via hypernetwork
5. **ASVD**: Activation-aware SVD for decoder KV-cache compression

---

## 🔍 Project Structure

```
FlashSVD/
├── src/
│   ├── flashsvd/              # Main package (pip installable)
│   │   ├── cli.py            # Unified CLI entry point
│   │   ├── compress.py       # Compression pipeline
│   │   ├── evaluate.py       # Evaluation pipeline
│   │   ├── finetune/         # Fine-tuning module
│   │   ├── compression/      # Compression methods (standard, fwsvd, etc.)
│   │   ├── ui/               # Gradio web interface
│   │   └── io.py             # Checkpoint loading/saving
│   ├── kernels/              # Triton streaming kernels
│   └── utils/                # SVD blocks and helpers
│
├── models/
│   └── finetuned/            # Finetuned models (auto-organized)
│       ├── bert/
│       ├── modernbert/
│       └── roberta/
│
├── compressed_models/         # Compressed model checkpoints
├── benchmark/                 # Performance evaluation scripts
├── experiments/               # Legacy experiment directories
└── docs/                      # Additional documentation
```

**Quick Links**:
- **Contributing**: [`CONTRIBUTING.md`](docs/CONTRIBUTING.md) - How to contribute
- **Developer guide**: [`CLAUDE.md`](docs/CLAUDE.md) - Architecture and development notes
- **Dataset guide**: [`DATASET_GUIDE.md`](docs/DATASET_GUIDE.md) - Dataset documentation
- **UI guide**: [`M5_UI_GUIDE.md`](docs/M5_UI_GUIDE.md) - Web UI usage guide

---

## 🧠 How FlashSVD Works

### The Problem

Traditional SVD compression reduces **parameter memory** but introduces significant **activation memory overhead** during inference:

```
Standard approach:
Input [B, M, d] → W_low [d, r] → Intermediate [B, M, r] → W_high [r, d] → Output [B, M, d]
                                      ↑ Materialized in memory!
```

This prevents real memory savings during inference.

### Our Solution

**FlashSVD** uses **fused streaming kernels** that process data in tiles, avoiding intermediate materialization:

```
FlashSVD approach:
Input [B, M, d] ──┐
                  ├─> Fused Triton Kernel (tile-based) ─> Output [B, M, d]
W_low, W_high ────┘   (Intermediate computed in SRAM, not DRAM)
```

**Key innovations**:
1. **Tile-based computation**: Small tiles of factors loaded into on-chip SRAM
2. **Fused operations**: Multiplication and reduction happen in one kernel
3. **Immediate eviction**: Tiles are discarded after use, no full-size buffers
4. **Rank-aware design**: Kernel optimized for low-rank projections

**Result**: Up to **70.2% reduction** in peak activation memory and **75% reduction** in intermediate transient memory with **no accuracy loss** compared to upstream compression methods.

---

## 📈 Benchmarks

### Memory Reduction (BERT-base, batch=64, 25% params)

| Task | Dense Peak | FlashSVD Peak | Dense Trans. | FlashSVD Trans. | Peak Reduction | Trans. Reduction |
|------|------------|---------------|--------------|-----------------|----------------|------------------|
| **SST-2** | 695 MiB | **577 MiB** | 277 MiB | **244 MiB** | **-17.0%** | **-11.9%** |
| **QQP** | 971 MiB | **786 MiB** | 554 MiB | **453 MiB** | **-19.1%** | **-18.2%** |
| **MNLI** | 1547 MiB | **1230 MiB** | 1130 MiB | **898 MiB** | **-20.5%** | **-20.5%** |

**FlashFWSVD** (Fisher-Weighted SVD + FlashSVD kernels):
| Task | Peak Memory | Transient Mem | Peak vs. Dense | Trans. vs. Dense |
|------|-------------|---------------|----------------|------------------|
| **SST-2** | **542 MiB** | **209 MiB** | **-22.0%** | **-24.5%** |
| **QQP** | **757 MiB** | **424 MiB** | **-22.0%** | **-23.5%** |
| **MNLI** | **1213 MiB** | **881 MiB** | **-21.6%** | **-22.0%** |

### Accuracy Preservation (25% params, training-free)

| Method | SST-2 | QQP | MNLI | Avg. |
|--------|-------|-----|------|------|
| **Dense (baseline)** | 92.3% | 90.9% | 84.1% | 89.1% |
| **Vanilla SVD** | 85.2% | 72.8% | 66.7% | 74.9% |
| **FlashSVD (Standard)** | 85.2% | 72.8% | 66.7% | 74.9% |
| **FWSVD** | 89.9% | 84.8% | 78.0% | 84.2% |
| **FlashFWSVD** | **89.9%** | **84.8%** | **77.9%** | **84.2%** |

*FlashSVD preserves accuracy identical to upstream compression methods while drastically reducing memory!*

### Latency (SST-2, batch=64, 25% params)

| Method | Latency | vs. Dense |
|--------|---------|-----------|
| Dense | 79.7 ms | 1.0× |
| Vanilla SVD | 161.1 ms | 2.0× |
| FlashSVD v1 | 188.8 ms | 2.4× |
| FWSVD | 83.8 ms | 1.1× |
| **FlashFWSVD** | **139.6 ms** | **1.8×** |

*FlashSVD achieves memory savings with acceptable latency trade-off. For long contexts (M≥512), FlashSVD shows up to 1.9× speedup over dense FFN (see paper Table S2-S3).*

See full benchmarks in the [paper](https://arxiv.org/abs/2508.01506) and [`benchmark/`](benchmark/) directory.

---

## 🔬 Advanced Usage

### Custom Rank Configuration

```bash
# Separate ranks for different components
flashsvd compress \
  --model bert-base-uncased \
  --task sst2 \
  --rank-attn 40 \
  --rank-ffn 240 \
  --rank-wo 240
```

### AdaSVD with Custom Ranks

```bash
# Use pre-computed adaptive ranks
flashsvd compress \
  --model bert-base-uncased \
  --task sst2 \
  --method adasvd \
  --ranks-json ./path/to/ranks.json
```

### Evaluation with Limited Samples

```bash
# Quick evaluation on subset
flashsvd eval \
  --checkpoint <path> \
  --task sst2 \
  --max-eval-samples 100
```

### Fine-tuning with Custom Settings

```bash
flashsvd finetune \
  --checkpoint <path> \
  --task sst2 \
  --epochs 5 \
  --learning-rate 2e-5 \
  --batch-size 32 \
  --warmup-ratio 0.1 \
  --early-stopping \
  --patience 3 \
  --output-dir ./my_finetuned_model
```

---

## 🧪 Testing

Test the installation with basic commands:

```bash
# Test compression
flashsvd compress --model bert-base-uncased --task sst2 --method fwsvd --rank 64

# Test evaluation
flashsvd eval --checkpoint ./compressed_models/bert-base-uncased_fwsvd_r64 --task sst2

# Test checkpoint info
flashsvd info ./compressed_models/bert-base-uncased_fwsvd_r64
```

---

## 📝 Citation

If you find FlashSVD useful in your research, please cite our paper:

```bibtex
@article{shao2025flashsvd,
  title={FlashSVD: Memory-Efficient Inference with Streaming for Low-Rank Models},
  author={Shao, Zishan and Wang, Yixiao and Wang, Qinsi and Jiang, Ting and Du, Zhixu and Ye, Hancheng and Zhuo, Danyang and Chen, Yiran and Li, Hai},
  journal={arXiv preprint arXiv:2508.01506},
  year={2025}
}
```

---

## 🤝 Contributing

We welcome contributions! Areas of interest:

- 🆕 Additional compression methods (SVD-LLM, Dobi-SVD, etc.)
- 🏗️ New model architectures (Qwen, Mistral, etc.)
- 📊 Custom dataset support for fine-tuning
- 🔧 Performance optimizations
- 📚 Documentation improvements

Please see [`CONTRIBUTING.md`](docs/CONTRIBUTING.md) for guidelines.

---

## 📞 Support & Contact

- **Issues & bugs**: [Open a GitHub issue](https://github.com/Zishan-Shao/FlashSVD/issues)
- **Feature requests**: [Open a feature request](https://github.com/Zishan-Shao/FlashSVD/issues/new)
- **Questions**: Email Zishan at zs89@duke.edu
- **Discussions**: [GitHub Discussions](https://github.com/Zishan-Shao/FlashSVD/discussions)

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- Paper implementations: Fisher-Weighted SVD, DRONE, AdaSVD, ASVD
- HuggingFace Transformers for model architectures
- Triton for GPU kernel framework
- GLUE benchmark for evaluation datasets

---

## 🔗 Related Projects

- **FlashAttention**: Efficient attention computation ([repo](https://github.com/Dao-AILab/flash-attention))
- **DRONE**: Data-aware low-rank compression ([paper](https://proceedings.neurips.cc/paper/2021/))
- **ASVD**: Activation-aware SVD ([paper](https://arxiv.org/abs/))
- **AdaSVD**: Adaptive rank selection ([paper](https://aclanthology.org/2024.naacl-long.13/))

---

**⭐ If you find this project helpful, please consider starring the repository!**

[![Star History Chart](https://api.star-history.com/svg?repos=Zishan-Shao/FlashSVD&type=Date)](https://star-history.com/#Zishan-Shao/FlashSVD&Date)
