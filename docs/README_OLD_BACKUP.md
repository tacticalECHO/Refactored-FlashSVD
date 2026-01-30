# FlashSVD: Memory-Efficient Inference with Streaming for Low-Rank Models

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![GitHub Stars](https://img.shields.io/github/stars/Zishan-Shao/FlashSVD?style=social)](https://github.com/Zishan-Shao/FlashSVD)

This repository contains the official implementation of **FlashSVD**, a novel end-to-end rank-aware streaming inference framework specifically designed for SVD-compressed large language models. FlashSVD addresses the critical limitation of previous SVD-based compression techniques by eliminating activation memory overhead during inference.


### Support & Contact

- Issues and bugs: please open a GitHub issue.
- Feature requests (e.g., new model support or SVD methods): open an issue with details.
- Collaboration or questions: email Zishan at zs89@duke.edu.

We aim to respond promptly. This is an active, long-term project, and we welcome community contributions.

**📄Paper**: [FlashSVD: Memory-Efficient Inference with Streaming for Low-Rank Models](https://arxiv.org/abs/2508.01506)

> 🙏 If you find FlashSVD useful in your research, we kindly ask that you `cite our paper` (see [Citation](#citation)). If this repository is helpful, please consider `starring 🌟` it to support the project — thank you!
>
> [![Cite FlashSVD](https://img.shields.io/badge/Cite-FlashSVD-brightgreen)](#citation) [![Star this repo](https://img.shields.io/badge/Star-This%20repo-yellow?logo=github)](https://github.com/Zishan-Shao/FlashSVD/stargazers)


## 🚀 Announcement
Our system involves have several popular SVD method (Unofficial) replication with code:

#### Encoders:

 - [Language model compression with weighted low-rank factorization](https://arxiv.org/abs/2207.00112): Fisher-Weighted SVD (FWSVD) is supported for BERT, RoBERTa, and ModernBERT
 - [DRONE: Data-aware Low-rank Compression for Large NLP Models](https://proceedings.neurips.cc/paper/2021/file/f56de5ef149cf0aedcc8f4797031e229-Paper.pdf): data whitening method enabled now on BERT
 - [Adaptive Rank Selections for Low-Rank Approximation of Language Models](https://aclanthology.org/2024.naacl-long.13/): AdaSVD code on BERT

#### Decoders:

- [ASVD: Activation-aware Singular Value Decomposition for Compressing Large Language Models](https://proceedings.neurips.cc/paper/2021/file/f56de5ef149cf0aedcc8f4797031e229-Paper.pdf): enabling low-rank kv-cache inference. We support asvd with uniform rank (hetero-rank case in development) on `Llama-2-7b model` and `gpt2` model

### 🛠️ TODO
- [✅] BERT (CLS / MLM) support  
- [✅] Fisher-Weighted SVD (FWSVD) replication  
- [✅] DRONE & Adaptive Rank Selection integration  
- [✅] LLaMA support (ASVD)  
- [ ] Qwen integration  
- [ ] LLaMA, GPT-2 support (SVD-LLM, Dobi-SVD etc.)  
- [✅] GPT-2 SVD (ASVD)  
- [✅] Add benchmark results and visualization tools


## 🔍 Overview

Singular Value Decomposition (SVD) has recently seen a surge of interest as a simple yet powerful tool for large language models (LLMs) compression, with a growing number of works demonstrating 20-80% parameter reductions at minimal accuracy loss. However, previous SVD-based approaches have focused primarily on reducing the memory footprint of model weights, largely overlooking the additional activation memory overhead incurred during inference when applying truncated factors via standard dense CUDA kernels.

Our experiments demonstrate that this activation overhead, scaling with sequence length and hidden dimension, prevents current SVD compression techniques from achieving any reduction in peak inference memory, thereby limiting their viability for real-world, on-device deployments.

### Pipeline

![FlashSVD Pipeline](figs/pipeline.png)

The figure above illustrates the FlashSVD computation pipeline, showing the efficient flow from input through low-rank attention and feed-forward layers.


### 🧰 Key Contributions

We introduce **FlashSVD**, a novel, end-to-end rank-aware streaming inference framework specifically designed for SVD-compressed large language models. FlashSVD can be seamlessly integrated with any model that employs SVD-based methods for parameter reduction. By fusing low-rank projection kernels directly into both the self-attention and feed-forward network (FFN) pipelines, FlashSVD avoids materializing full-size activation buffers. Instead, small tiles of the truncated factors are loaded into on-chip SRAM, multiplied and reduced on the fly, and immediately evicted, preserving high GPU occupancy and adding no extra latency.

- **End-to-End Streaming Framework**: Rank-aware inference system for SVD-compressed models
- **Fused Low-Rank Kernels**: Direct integration into attention and FFN pipelines  
- **Tile-Based Computation**: Avoids materializing full-size activation buffers
- **Memory-Efficient Deployment**: Up to 70.2% reduction in peak activation memory

## 🧠 Key Features

- **Universal Integration**: Seamlessly works with any SVD-compressed model
- **Streaming Inference**: Tile-based computation avoids activation buffer materialization
- **GPU Optimized**: Fused kernels preserve high GPU occupancy with no extra latency on medium-low ranked cases
- **Memory Efficient**: Up to 70.2% reduction in peak activation memory
- **Accuracy Preserving**: No accuracy loss with upstream compression methods

## 🧩 Installation

### Prerequisites
- Linux (WSL2 supported)
- Python 3.8 or higher
- CUDA-compatible GPU (recommended for Triton kernels)
- PyTorch 2.0+ with CUDA support
- Triton 2.0+ (typically installed with PyTorch)

### Setup

**⚠️ IMPORTANT**: You must install PyTorch with CUDA support **before** installing FlashSVD.

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Zishan-Shao/FlashSVD.git
   cd FlashSVD
   ```

2. **Install PyTorch first** (choose one based on your CUDA version):

   ```bash
   # CUDA 12.1 (recommended)
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

   # CUDA 11.8
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

   # CPU only (for testing, kernels require GPU)
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
   ```

3. **Install FlashSVD** (choose one option):

   ### Option A — Quick Install (recommended)
   ```bash
   pip install -e .
   ```

   ### Option B — Using install script (auto-creates venv)
   ```bash
   # Requires PyTorch pre-installed
   ./install_local.sh
   ```

   ### Option C — Conda environment
   ```bash
   conda create -n flashsvd python=3.10
   conda activate flashsvd

   # Install PyTorch (CUDA version)
   conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

   # Install FlashSVD
   pip install -e .
   ```

4. **Verify installation:**

   ```bash
   # Test package import
   python -c "import flashsvd; print(flashsvd.__version__)"

   # Test kernel/utils imports
   python -c "from flashsvd.kernels import flashsvdattn; from flashsvd.utils import FlashSVDBlocks; print('✓ All imports successful')"
   ```

   Expected output:
   ```
   0.1.0
   ✓ All imports successful
   ```

## 🎯 CLI Usage

FlashSVD provides a unified command-line interface for compressing, evaluating, and inspecting models.

### Compress a Model

Compress pretrained models using SVD:

```bash
# Compress with unified rank for all components
flashsvd compress --model bert-base-uncased --task sst2 --rank 64

# Compress with separate ranks for different components
flashsvd compress --model bert-base-uncased --task sst2 \
  --rank-attn 40 --rank-ffn 240 --rank-wo 240

# Compress a finetuned checkpoint
flashsvd compress --checkpoint ./models/bert-sst2-finetuned --task sst2 --rank 64

# Compress RoBERTa models
flashsvd compress --model roberta-base --task sst2 --rank 64
```

Output is saved to `./compressed_models/<model>_<method>_r<rank>/` with:
- HuggingFace format files (config.json, model weights)
- FlashSVD state dict (flashsvd_state_dict.pt)
- Compression metadata (compression_info.json)

### Evaluate Compressed Model

Evaluate compressed models on GLUE tasks:

```bash
# Evaluate on SST-2
flashsvd eval --checkpoint ./compressed_models/bert-base-uncased_standard_r64 --task sst2

# Custom batch size and sequence length
flashsvd eval --checkpoint ./compressed_models/bert_r64 --task sst2 \
  --batch-size 16 --seq-len 256

# Use CPU instead of GPU
flashsvd eval --checkpoint ./compressed_models/bert_r64 --task sst2 \
  --device cpu

# Save results to custom file
flashsvd eval --checkpoint ./compressed_models/bert_r64 --task sst2 \
  --output my_results.json

# Subsample validation set for quick testing
flashsvd eval --checkpoint ./compressed_models/bert_r64 --task sst2 \
  --max-eval-samples 1000
```

Results are saved to `eval_results.json` (or custom path) with metrics:
- Accuracy/Pearson score
- Peak memory usage (MiB)
- Latency (ms/batch)
- Model metadata and configuration

### Inspect Checkpoint

Display metadata and information about compressed checkpoints:

```bash
# Show checkpoint information
flashsvd info ./compressed_models/bert-base-uncased_standard_r64
```

Output includes:
- Compression method and ranks
- Base model and task
- File sizes and checksums
- Usage instructions

### Smoke Tests (Regression Protection)

After installation or updates, verify CLI functionality:

```bash
# Test help commands
flashsvd --help
flashsvd compress --help
flashsvd eval --help
flashsvd info --help

# Test actual workflow (requires GPU for CUDA kernels)
# 1. Compress a model
flashsvd compress --model bert-base-uncased --task sst2 --rank 64

# 2. Inspect the checkpoint
flashsvd info ./compressed_models/bert-base-uncased_standard_r64

# 3. Evaluate the compressed model
flashsvd eval --checkpoint ./compressed_models/bert-base-uncased_standard_r64 --task sst2
```

**Note**:
- Default device is `cuda`. Use `--device cpu` to force CPU execution.
- Triton kernels require CUDA. CPU evaluation will show warnings but won't block execution if you have alternative implementations.

### Web UI (Optional)

For an interactive web interface:

```bash
# Launch Gradio UI
flashsvd-ui
```

The UI provides three tabs:
- **Compress**: Compress models with visual parameter selection
- **Evaluate**: Evaluate compressed models with real-time logs
- **Info**: Inspect checkpoint metadata and files

Access at `http://localhost:7860` after launching.

## ⚡ Quick Start

### 1. Training BERT with FlashSVD in Gradio UI (Recommended)
This project provides an interactive **Gradio interface** to run the unified training script easily.
```bash
# Launch the Gradio app
python legacy/app.py
```
   ### UI Overview

   #### Mode
   Choose `cls` (classification) or `mlm` (masked language modeling).  
   The available tasks will automatically change depending on the selected mode.

   #### Task
   Supported GLUE tasks (e.g., `sst2`, `qnli`, `mnli`, `stsb`, etc.).

   #### Model Checkpoint
   Example — `bert-base-uncased`.

   #### Training Parameters
   - `epochs`
   - `batch size`
   - `learning rate`
   - `logging steps`
   - `evaluation steps`
   - `random seed`

   #### Advanced Options
   - **Force CPU (`--no_cuda`)**: Force CPU training.
   - **CUDA_VISIBLE_DEVICES**: e.g., `0` or `0,1` if multiple GPUs are available.
   - **Extra CLI Args**: Additional command-line arguments appended at runtime.

   #### Logging
   - **Log Directory**: default `runs/`
   - **Run Name**: leave blank to auto-generate a timestamp name.
   - **Append if log exists**: append logs instead of overwriting.
   - Logs are stored under `runs/<run_name>.log` and can be downloaded in the UI.

   #### Internal Execution
   Internally, the Gradio app executes a command equivalent to:

   ```bash
   python -u $TRAIN_UNIFIED_SCRIPT \
   --mode <mode> --task <task> --model <checkpoint> \
   --epochs <n> --batch_size <n> --learning_rate <lr> \
   --logging_steps <n> --eval_steps <n> --seed <n> \
   [--output_dir <path>] [--no_cuda] [<extra_args>]
   ```

### Command Line Usage (Without UI)
Train BERT models with specific GLUE tasks and rank configurations:

```bash
python legacy/train_bert_unified_min.py \
  --mode cls --task sst2 --model bert-base-uncased \
  --epochs 3 --batch_size 32 --learning_rate 2e-5 \
  --logging_steps 100 --eval_steps 0 --seed 0

# Masked Language Modeling example
python legacy/train_bert_unified_min.py \
  --mode mlm --task mnli --model bert-base-uncased \
  --epochs 3 --batch_size 32 --learning_rate 2e-5 \
  --logging_steps 100 --eval_steps 0 --seed 0
```

### 2. Inference and Profiling

After training, run inference with profiling in the experiments directories:

```bash
# Navigate to BERT directory for standard inference
cd experiments/BERT/
python profile_flashsvd.py  # or your specific profiling script

# Navigate to BERTFW directory for FlashSVD inference
cd experiments/BERTFW/
python profile_flashfwsvd.py  # or your specific profiling script
```

The profiling scripts will provide detailed performance metrics including:
- Inference latency
- Memory usage
- Comparison between standard and FlashSVD implementations



## 📊 Results

### Performance Comparison

FlashSVD achieves significant improvements in efficiency:

- **Memory Reduction**: Up to 70.2% reduction in peak activation memory
- **Intermediate Memory**: 75% reduction in transient memory usage
- **Accuracy Preservation**: No accuracy loss with upstream compression methods
- **Practical Deployment**: Enables memory-constrained deployment of low-rank LLMs

### Rank Loss Analysis

![Rank Loss Comparison](figs/rank_loss_comparison.png)

The figure above shows the trade-off between rank reduction and model performance across different tasks.

### Key Contributions

Our work addresses the critical limitation of previous SVD-based approaches by introducing:

- **End-to-end rank-aware streaming inference framework**
- **Fused low-rank projection kernels** for both attention and FFN
- **Tile-based computation** that avoids materializing full-size activation buffers
- **Seamless integration** with any SVD-compressed model

## Citation

If you find this work useful in your research, please cite our paper:

```bibtex
@article{shao2025flashsvd,
  title={FlashSVD: Memory-Efficient Inference with Streaming for Low-Rank Models},
  author={Shao, Zishan and Wang, Yixiao and Wang, Qinsi and Jiang, Ting and Du, Zhixu and Ye, Hancheng and Zhuo, Danyang and Chen, Yiran and Li, Hai},
  journal={arXiv preprint arXiv:2508.01506},
  year={2025}
}
```

<!-- **Paper**: [FlashSVD: Memory-Efficient Inference with Streaming for Low-Rank Models](https://arxiv.org/abs/2508.01506) -->
## Project Structure

```
FlashSVD/
├── src/                    # Core implementation
│   ├── flashsvd/          # Main package (pip installable)
│   ├── kernels/           # CUDA kernels and optimizations
│   └── utils/             # Utility functions and blocks
├── test/                   # Test suite (organized structure)
│   ├── scripts/           # Test execution scripts
│   ├── logs/              # Test execution logs
│   ├── results/           # Test results and reports
│   └── README.md          # Test organization guide
├── compression_test/       # Compressed model checkpoints
├── benchmark/             # Performance evaluation scripts
├── experiments/           # Legacy experiment directories
├── figs/                  # Paper figures and diagrams
├── train/                 # Training utilities
└── README.md             # This file
```

**Quick Links**:
- **Latest test results**: See [`test/results/COMPARISON_REPORT_FINAL.md`](test/results/COMPARISON_REPORT_FINAL.md)
- **Test organization**: See [`test/README.md`](test/README.md)
- **Dataset guide**: See [`DATASET_GUIDE.md`](DATASET_GUIDE.md) | Quick ref: [`DATASET_QUICK_REFERENCE.md`](DATASET_QUICK_REFERENCE.md)
- **Finetuned model organization**: See [`FINETUNED_MODEL_ORGANIZATION.md`](FINETUNED_MODEL_ORGANIZATION.md)
- **Project guide**: See [`CLAUDE.md`](CLAUDE.md)

## Contributing

We welcome contributions! Please feel free to submit issues and pull requests.

**Note**: For support, feature requests, or collaborations, please open a GitHub issue or email Zishan (zs89@duke.edu).
