# Dataset Quick Reference

## 📍 Dataset Location

### Auto-cache Location
```bash
~/.cache/huggingface/datasets/glue/
```

**Linux/Mac**:
```
/home/username/.cache/huggingface/datasets/glue/
├── sst2/     # 67K training samples, 188MB
├── cola/     # 8.5K training samples
├── mnli/     # 393K training samples (largest)
├── qnli/     # 105K training samples
├── qqp/      # 364K training samples
├── mrpc/     # 3.7K training samples
├── rte/      # 2.5K training samples
└── stsb/     # 5.7K training samples
```

**Windows**:
```
C:\Users\username\.cache\huggingface\datasets\glue\
```

---

## 📊 Supported Datasets

| Task | Type | Train Samples | Purpose |
|------|------|--------------|---------|
| **sst2** | Binary | 67K | Sentiment analysis ⭐ Recommended |
| **cola** | Binary | 8.5K | Grammar judgment |
| **mrpc** | Binary | 3.7K | Paraphrase detection |
| **qqp** | Binary | 364K | Question matching |
| **mnli** | 3-class | 393K | Natural language inference |
| **qnli** | Binary | 105K | Question inference |
| **rte** | Binary | 2.5K | Textual entailment ⭐ Quick test |
| **stsb** | Regression | 5.7K | Semantic similarity |

---

## 🚀 Usage Examples

### Basic Usage

```bash
# Fine-tune with SST-2 dataset (most common)
flashsvd finetune \
  --checkpoint ./compressed_models/bert-base_fwsvd_r64 \
  --task sst2 \
  --epochs 3

# First run will auto-download dataset to cache
# Downloading: 100%|████| 7.44M/7.44M [00:02<00:00]
# Subsequent runs use cache directly
```

### Quick Test (Limited Samples)

```bash
# Use only 1000 samples for quick testing
flashsvd finetune \
  --checkpoint <checkpoint_path> \
  --task sst2 \
  --max-train-samples 1000 \
  --epochs 1
```

### Different Tasks

```bash
# Sentiment analysis
flashsvd finetune --checkpoint <path> --task sst2

# Natural language inference
flashsvd finetune --checkpoint <path> --task mnli

# Question matching
flashsvd finetune --checkpoint <path> --task qqp

# Semantic similarity (regression task)
flashsvd finetune --checkpoint <path> --task stsb
```

---

## 🔍 View Datasets

### Method 1: Python Script

```python
from datasets import load_dataset

# Load dataset
ds = load_dataset("glue", "sst2")

# View information
print(f"Train samples: {len(ds['train'])}")        # 67349
print(f"Validation samples: {len(ds['validation'])}")   # 872

# View sample
print(ds["train"][0])
# {'sentence': 'hide new secretions...', 'label': 0}
```

### Method 2: Command Line

```bash
# View cached datasets
ls ~/.cache/huggingface/datasets/glue/

# View dataset size
du -sh ~/.cache/huggingface/datasets/glue/*
```

---

## ⚙️ Dataset Management

### Clear Cache

```bash
# Delete all dataset cache
rm -rf ~/.cache/huggingface/datasets/*

# Delete specific dataset
rm -rf ~/.cache/huggingface/datasets/glue/qqp
```

### Offline Usage

Datasets are cached after download and subsequent use doesn't require internet.

**For offline environments**:
1. Run once in networked environment (auto-download cache)
2. Copy cache directory: `~/.cache/huggingface/`
3. Paste to same location on offline machine

---

## 🎯 Dataset Selection Guide

### By Use Case

| Need | Recommended Dataset | Reason |
|------|-------------------|--------|
| **Quick Testing** | sst2, rte | Moderate samples, fast training |
| **Production Eval** | mnli, qqp | Many samples, reliable results |
| **Sentiment Analysis** | sst2 | Standard sentiment task |
| **Semantic Matching** | qqp, stsb | Question/sentence matching |

### By Training Time

| Dataset | Sample Count | Training Time (3 epochs) |
|---------|-------------|-------------------------|
| **rte** | 2.5K | ~5 minutes ⚡ |
| **cola** | 8.5K | ~10 minutes |
| **sst2** | 67K | ~20 minutes ⭐ |
| **qnli** | 105K | ~30 minutes |
| **qqp** | 364K | ~1 hour |

---

## ❌ Current Limitations

### No Custom Dataset Support

**Only GLUE tasks supported**, cannot use your own CSV/JSON data.

**Planned for future**:
```bash
# Planned (not implemented)
flashsvd finetune \
  --train-file my_train.csv \
  --val-file my_val.csv \
  --text-column "text" \
  --label-column "label"
```

**Current workaround**: See "Using Custom Datasets" section in `DATASET_GUIDE.md`

---

## 📚 Related Documentation

- **Detailed Guide**: `DATASET_GUIDE.md`
- **HuggingFace GLUE**: https://huggingface.co/datasets/glue
- **GLUE Benchmark**: https://gluebenchmark.com/

---

## 💡 Best Practices

1. **Prefer SST-2**: Moderate samples (67K), fast training (~20 min), good results
2. **RTE for quick validation**: Few samples (2.5K), very fast training (~5 min)
3. **Limit samples for dev**: `--max-train-samples 1000` for quick iteration
4. **Full training for deployment**: Use all data for best performance
5. **Monitor downloads**: Watch download progress on first run

---

**Quick Start**:
```bash
# Simplest fine-tuning command (using SST-2)
flashsvd finetune --checkpoint <checkpoint_path> --task sst2 --epochs 3
```

**Data will auto-download to**: `~/.cache/huggingface/datasets/glue/sst2/`

---

**Last Updated**: 2026-01-30
