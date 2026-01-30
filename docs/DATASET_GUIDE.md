# FlashSVD Dataset Usage Guide

## 📊 Currently Supported Datasets

FlashSVD fine-tuning currently supports **8 tasks from the GLUE benchmark**:

| Task | Full Name | Type | Train Samples | Val Samples | Metric |
|------|-----------|------|--------------|-------------|--------|
| **CoLA** | Corpus of Linguistic Acceptability | Binary | 8.5K | 1K | Matthews Corr |
| **SST-2** | Stanford Sentiment Treebank | Binary | 67K | 872 | Accuracy |
| **MRPC** | Microsoft Research Paraphrase Corpus | Binary | 3.7K | 408 | F1/Accuracy |
| **QQP** | Quora Question Pairs | Binary | 364K | 40K | F1/Accuracy |
| **MNLI** | Multi-Genre NLI | 3-class | 393K | 10K | Accuracy |
| **QNLI** | Question NLI | Binary | 105K | 5.4K | Accuracy |
| **RTE** | Recognizing Textual Entailment | Binary | 2.5K | 277 | Accuracy |
| **STS-B** | Semantic Textual Similarity | Regression | 5.7K | 1.5K | Pearson/Spearman |

---

## 📁 Dataset Storage Location

### 1. HuggingFace Cache Directory

Datasets are automatically downloaded and cached at:

```bash
~/.cache/huggingface/datasets/
```

**Linux/Mac**:
```
/home/username/.cache/huggingface/datasets/glue/
├── cola/        # 8.5K training samples
├── sst2/        # 67K training samples (188MB cache)
├── mrpc/        # 3.7K training samples
├── qqp/         # 364K training samples (largest)
├── mnli/        # 393K training samples
├── qnli/        # 105K training samples
├── rte/         # 2.5K training samples
└── stsb/        # 5.7K training samples
```

**Windows**:
```
C:\Users\username\.cache\huggingface\datasets\glue\
```

### 2. Dataset Format

Data is stored in **Apache Arrow** format (`.arrow` files), which is efficient and fast to load.

---

## 🔍 Viewing Dataset Information

### Method 1: Using Python Script

```python
from datasets import load_dataset

# Load SST-2 dataset
dataset = load_dataset("glue", "sst2")

# View dataset structure
print(dataset)
# DatasetDict({
#     train: Dataset({
#         features: ['sentence', 'label', 'idx'],
#         num_rows: 67349
#     })
#     validation: Dataset({
#         features: ['sentence', 'label', 'idx'],
#         num_rows: 872
#     })
# })

# View sample
print(dataset["train"][0])
# {'sentence': 'hide new secretions from the parental units', 'label': 0, 'idx': 0}

# View label distribution
from collections import Counter
labels = dataset["train"]["label"]
print(Counter(labels))
# Counter({1: 34299, 0: 33050})  # 1=positive, 0=negative
```

### Method 2: Using CLI Commands

```bash
# View cached datasets
ls ~/.cache/huggingface/datasets/glue/

# View dataset size
du -sh ~/.cache/huggingface/datasets/glue/*

# View specific dataset
python -c "from datasets import load_dataset; ds=load_dataset('glue','sst2'); print(ds)"
```

---

## 📥 Dataset Download Process

### Automatic Download on First Run

When you first run a fine-tuning command, datasets will automatically download from HuggingFace:

```bash
flashsvd finetune \
  --checkpoint <checkpoint_path> \
  --task sst2 \
  --epochs 3
```

**Download process**:
```
Downloading: 100%|████████| 7.44M/7.44M [00:02<00:00, 3.72MB/s]
Downloading: 100%|████████| 222k/222k [00:00<00:00, 1.11MB/s]
Generating train split: 67349 examples [00:01, 33674.50 examples/s]
Generating validation split: 872 examples [00:00, 8720.00 examples/s]
```

### Subsequent Runs Use Cache

After download, datasets are cached and subsequent runs use the cache directly without re-downloading.

---

## 🎯 Dataset Selection Recommendations

### By Task Type

| Goal | Recommended Dataset | Reason |
|------|-------------------|--------|
| **Quick Testing** | SST-2, RTE | Few samples, fast training |
| **Sentiment Analysis** | SST-2 | Standard sentiment classification |
| **Semantic Similarity** | QQP, STS-B | Question matching/similarity |
| **Natural Language Inference** | MNLI, QNLI, RTE | NLI tasks |
| **Linguistic Acceptability** | CoLA | Grammar correctness |
| **Paraphrase Detection** | MRPC | Sentence paraphrase detection |

### By Dataset Size

| Dataset Size | Tasks | Training Time (Est.) |
|-------------|-------|---------------------|
| **Small** (< 10K) | CoLA, RTE | 5-10 minutes |
| **Medium** (10K-100K) | SST-2, QNLI | 10-30 minutes |
| **Large** (> 100K) | QQP, MNLI | 30 minutes-2 hours |

---

## 🔧 Using Custom Datasets

### ❌ Current Limitation

**FlashSVD current version** only supports GLUE tasks, **does NOT support** custom datasets.

### ✅ Planned Support (Future Version)

We plan to add custom dataset support, allowing you to use your own data:

```bash
# Future feature (planned)
flashsvd finetune \
  --checkpoint <checkpoint_path> \
  --train-file ./my_data/train.csv \
  --val-file ./my_data/val.csv \
  --text-column "text" \
  --label-column "label" \
  --epochs 3
```

### 🛠️ Temporary Workarounds

If you need to use custom datasets now, here are some options:

#### Option 1: Convert to GLUE Format

1. Convert your data to GLUE CSV/JSON format
2. Replace data files in HuggingFace cache
3. Run fine-tuning with GLUE task name

**Example**:
```python
# Convert your data to GLUE SST-2 format
import pandas as pd
from datasets import Dataset, DatasetDict

# Read your data
df_train = pd.read_csv("my_train.csv")
df_val = pd.read_csv("my_val.csv")

# Convert to GLUE format (columns: sentence, label)
train_dataset = Dataset.from_pandas(df_train[["text", "label"]].rename(columns={"text": "sentence"}))
val_dataset = Dataset.from_pandas(df_val[["text", "label"]].rename(columns={"text": "sentence"}))

# Save as GLUE format
dataset_dict = DatasetDict({
    "train": train_dataset,
    "validation": val_dataset
})

# Save to HuggingFace cache location
dataset_dict.save_to_disk("~/.cache/huggingface/datasets/glue/custom_task")
```

#### Option 2: Modify Source Code

Modify the `load_glue_dataset()` function in `src/flashsvd/finetune/trainer.py`:

```python
def load_glue_dataset(config, tokenizer):
    """Load and tokenize dataset."""

    # Add custom dataset loading logic
    if config.task == "custom":
        # Load from CSV/JSON
        from datasets import load_dataset
        train_dataset = load_dataset("csv", data_files=config.train_file)
        val_dataset = load_dataset("csv", data_files=config.val_file)
    else:
        # Original GLUE loading logic
        if config.task == "mnli":
            train_dataset = load_dataset("glue", config.task, split="train")
            val_dataset = load_dataset("glue", config.task, split="validation_matched")
        else:
            train_dataset = load_dataset("glue", config.task, split="train")
            val_dataset = load_dataset("glue", config.task, split="validation")

    # ... subsequent tokenization logic
```

---

## 📊 Dataset Examples

### SST-2 (Sentiment Analysis)

```python
{
    'sentence': 'hide new secretions from the parental units',
    'label': 0,  # 0=negative, 1=positive
    'idx': 0
}
```

### MNLI (Natural Language Inference)

```python
{
    'premise': 'Conceptually cream skimming has two basic dimensions - product and geography.',
    'hypothesis': 'Product and geography are what make cream skimming work.',
    'label': 1,  # 0=entailment, 1=neutral, 2=contradiction
    'idx': 0
}
```

### QQP (Question Pair Matching)

```python
{
    'question1': 'What is the step by step guide to invest in share market in india?',
    'question2': 'What is the step by step guide to invest in share market?',
    'label': 0,  # 0=not duplicate, 1=duplicate
    'idx': 0
}
```

### STS-B (Semantic Similarity, Regression Task)

```python
{
    'sentence1': 'A plane is taking off.',
    'sentence2': 'An air plane is taking off.',
    'label': 5.0,  # Similarity score from 0.0-5.0
    'idx': 0
}
```

---

## 🔍 Common Questions

### Q1: Datasets are too large, what about disk space?

**A**: Clear unnecessary dataset cache:
```bash
# Delete specific task
rm -rf ~/.cache/huggingface/datasets/glue/qqp

# Clear all HuggingFace cache
rm -rf ~/.cache/huggingface/datasets/*
```

They will be re-downloaded on next run.

### Q2: Dataset download failed?

**A**: May be network issues, try:
```bash
# Set proxy (if needed)
export http_proxy=http://proxy.example.com:8080
export https_proxy=http://proxy.example.com:8080

# Or use mirror (China)
export HF_ENDPOINT=https://hf-mirror.com
```

### Q3: How to limit training sample count?

**A**: Use `--max-train-samples` parameter:
```bash
flashsvd finetune \
  --checkpoint <checkpoint_path> \
  --task sst2 \
  --max-train-samples 1000  # Only use 1000 training samples
  --epochs 3
```

### Q4: Where are datasets downloaded from?

**A**: From HuggingFace official servers:
- Official URL: `https://huggingface.co/datasets/glue`
- Data source: GLUE benchmark official data
- Auto-cached to: `~/.cache/huggingface/datasets/`

### Q5: Can datasets be used offline?

**A**: Yes, datasets are cached after download and subsequent use doesn't require internet.

For offline environment:
1. Run fine-tuning once on a networked machine (auto-download)
2. Copy cache directory: `~/.cache/huggingface/datasets/`
3. Paste to same location on offline machine

---

## 📝 Dataset Preprocessing

### Dataset Loading Pipeline

```python
# 1. Load raw data
dataset = load_dataset("glue", "sst2")

# 2. Tokenize
def tokenize_function(examples):
    return tokenizer(
        examples["sentence"],
        padding="max_length",
        truncation=True,
        max_length=128
    )

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# 3. Set format
tokenized_datasets.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "label"]
)
```

These steps are automatically done in `src/flashsvd/finetune/trainer.py`.

---

## 🎯 Best Practices

### 1. Choose Appropriate Dataset

- **Prototyping**: Use SST-2 or RTE (few samples, fast validation)
- **Final Evaluation**: Use full dataset (get accurate performance metrics)

### 2. Use Data Limits

During development use `--max-train-samples` for quick iteration:
```bash
# Quick test (1000 samples)
flashsvd finetune --checkpoint <path> --task sst2 --max-train-samples 1000 --epochs 1

# Full training (all samples)
flashsvd finetune --checkpoint <path> --task sst2 --epochs 3
```

### 3. Monitor Data Loading

Watch download progress on first run:
```
Downloading: 100%|████████| 7.44M/7.44M [00:02<00:00]
Generating train split: 67349 examples [00:01, 33674 examples/s]
```

### 4. Validate Data Quality

Check dataset before fine-tuning:
```python
from datasets import load_dataset
ds = load_dataset("glue", "sst2")
print(f"Train samples: {len(ds['train'])}")
print(f"Validation samples: {len(ds['validation'])}")
print(f"Sample: {ds['train'][0]}")
```

---

## 📚 Related Documentation

- **HuggingFace GLUE**: https://huggingface.co/datasets/glue
- **GLUE Benchmark**: https://gluebenchmark.com/
- **Datasets Library**: https://huggingface.co/docs/datasets/

---

## 🔜 Roadmap

We plan to add in future versions:

- [ ] Custom CSV/JSON dataset support
- [ ] Custom column mapping
- [ ] Data augmentation features
- [ ] Multi-task joint training
- [ ] More benchmark datasets (SuperGLUE, etc.)

---

**Last Updated**: 2026-01-30
**FlashSVD Version**: 0.1.0
