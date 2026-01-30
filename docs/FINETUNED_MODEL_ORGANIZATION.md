# Finetuned Model Auto-Organization Structure

## 📁 New Default Structure

When **not specifying `--output-dir`**, fine-tuned models will automatically be saved using the following structure:

```
models/finetuned/
├── bert/                          # BERT family
│   ├── standard/                  # Standard SVD
│   │   ├── bert-base-uncased_standard_r64/
│   │   │   ├── best/
│   │   │   ├── checkpoint-2-1500/
│   │   │   └── tensorboard/
│   │   └── bert-large-uncased_standard_r40/
│   │       └── best/
│   │
│   ├── fwsvd/                     # Fisher-Weighted SVD
│   │   ├── bert-base-uncased-SST-2_fwsvd_r64/
│   │   │   └── best/
│   │   └── bert-base-cased_fwsvd_r64/
│   │       └── best/
│   │
│   ├── whiten/                    # Whiten (DRONE)
│   │   └── bert-base-uncased_whiten_r64/
│   │       └── best/
│   │
│   └── adasvd/                    # Adaptive SVD
│       └── bert-base-uncased_adasvd_r64/
│           └── best/
│
├── modernbert/                    # ModernBERT family
│   ├── standard/
│   ├── fwsvd/
│   └── whiten/
│
├── roberta/                       # RoBERTa family
│   ├── standard/
│   │   └── roberta-base_standard_r64/
│   ├── fwsvd/
│   └── whiten/
│
├── llama/                         # LLaMA family
│   ├── asvd/
│   └── standard/
│
└── gpt2/                          # GPT-2 family
    ├── asvd/
    └── standard/
```

---

## 🎯 Path Generation Logic

### Automatic Metadata Reading

The system reads from the compressed model's `compression_info.json`:
- `arch`: Model architecture (bert, modernbert, roberta, llama, gpt2)
- `method`: Compression method (standard, fwsvd, whiten, adasvd, asvd)

**Path Generation Formula**:
```
models/finetuned/{arch}/{method}/{checkpoint_name}/
```

### Examples

#### Example 1: BERT + FWSVD

**Input**:
```bash
flashsvd finetune \
  --checkpoint ./compressed_models/bert-base-uncased-SST-2_fwsvd_r64 \
  --task sst2 \
  --epochs 3
# Note: No --output-dir specified
```

**Auto-generated Path**:
```
models/finetuned/bert/fwsvd/bert-base-uncased-SST-2_fwsvd_r64/
├── best/
│   ├── flashsvd_state_dict.pt
│   ├── compression_info.json
│   └── config.json
├── checkpoint-2-1500/
└── tensorboard/
```

#### Example 2: ModernBERT + Whiten

**Input**:
```bash
flashsvd finetune \
  --checkpoint ./compressed_models/modernbert-base_whiten_r64 \
  --task mnli \
  --epochs 5
```

**Auto-generated Path**:
```
models/finetuned/modernbert/whiten/modernbert-base_whiten_r64/
└── best/
```

#### Example 3: RoBERTa + Standard

**Input**:
```bash
flashsvd finetune \
  --checkpoint ./compressed_models/roberta-base_standard_r40 \
  --task qqp \
  --epochs 3
```

**Auto-generated Path**:
```
models/finetuned/roberta/standard/roberta-base_standard_r40/
└── best/
```

---

## 🔄 Fallback Mechanism

If `compression_info.json` doesn't exist, the system will infer architecture and method from **checkpoint path**:

### Architecture Inference Rules
```python
checkpoint_path → arch
"bert" (not modernbert/roberta) → "bert"
"modernbert" → "modernbert"
"roberta" → "roberta"
"llama" → "llama"
"gpt2" → "gpt2"
other → "unknown"
```

### Method Inference Rules
```python
checkpoint_path → method
"fwsvd" or "fw" → "fwsvd"
"whiten" or "drone" → "whiten"
"ada" → "adasvd"
"asvd" → "asvd"
"standard" → "standard"
other → "unknown"
```

---

## ✅ Advantages

### 1. Clear Hierarchy
```
✅ models/finetuned/bert/fwsvd/bert-base-SST-2_fwsvd_r64/
❌ ./compressed_models/bert-base-SST-2_fwsvd_r64/best/  (old way)
```

### 2. Easy to Find
```bash
# Find all BERT + FWSVD fine-tuned models
ls models/finetuned/bert/fwsvd/

# Find all Whiten method fine-tuned models
find models/finetuned -name "whiten" -type d
```

### 3. Easy Comparison
```bash
# Compare same architecture with different methods
models/finetuned/bert/
├── standard/bert-base_standard_r64/
├── fwsvd/bert-base_fwsvd_r64/
└── whiten/bert-base_whiten_r64/
```

### 4. Avoid Confusion
- Fine-tuned and compressed models completely separated
- Won't overwrite original compressed models
- Auto-categorized by architecture and method

---

## 🎨 Custom Path

If you want to use your own path, you can still specify `--output-dir`:

```bash
flashsvd finetune \
  --checkpoint <checkpoint_path> \
  --task sst2 \
  --output-dir ./my_custom_path/my_model  # Custom path
```

---

## 📊 Complete Workflow Example

### Step 1: Compress Model
```bash
flashsvd compress \
  --model textattack/bert-base-uncased-SST-2 \
  --task sst2 \
  --method fwsvd \
  --rank 64

# Output: ./compressed_models/bert-base-uncased-SST-2_fwsvd_r64/
```

### Step 2: Fine-tune Model (without specifying output-dir)
```bash
flashsvd finetune \
  --checkpoint ./compressed_models/bert-base-uncased-SST-2_fwsvd_r64 \
  --task sst2 \
  --epochs 3 \
  --learning-rate 3e-5

# Auto-output: models/finetuned/bert/fwsvd/bert-base-uncased-SST-2_fwsvd_r64/
```

### Step 3: Evaluate Best Model
```bash
flashsvd eval \
  --checkpoint models/finetuned/bert/fwsvd/bert-base-uncased-SST-2_fwsvd_r64/best \
  --task sst2 \
  --batch-size 16

# Result: Accuracy typically improves 2-5%
```

### Step 4: View All BERT + FWSVD Models
```bash
ls -lh models/finetuned/bert/fwsvd/
```

---

## 🗂️ Comparison with Compressed Models

| | Compressed Models | Finetuned Models |
|---|------------------|------------------|
| **Default Location** | `./compressed_models/` | `models/finetuned/` |
| **Organization** | Flat structure | Hierarchical by arch/method |
| **Naming** | `{model}_{method}_r{rank}` | Same |
| **Purpose** | Evaluate after compression | Production deployment after fine-tuning |
| **Overwrite** | May overwrite | Saved independently |

---

## 🔍 Finding Finetuned Models

### By Architecture
```bash
# All BERT fine-tuned models
find models/finetuned/bert -name "best" -type d

# All ModernBERT fine-tuned models
find models/finetuned/modernbert -name "best" -type d
```

### By Method
```bash
# All FWSVD fine-tuned models
find models/finetuned -path "*/fwsvd/*/best"

# All Whiten fine-tuned models
find models/finetuned -path "*/whiten/*/best"
```

### By Task (need to check compression_info.json)
```bash
# Find all SST-2 task fine-tuned models
grep -r "\"task\": \"sst2\"" models/finetuned/ | grep compression_info.json
```

---

## 📋 Summary

### ✨ New Auto-Organization Features

1. **Without output_dir**: Auto-save to `models/finetuned/{arch}/{method}/{checkpoint_name}/`
2. **Architecture classification**: bert, modernbert, roberta, llama, gpt2
3. **Method classification**: standard, fwsvd, whiten, adasvd, asvd
4. **Smart inference**: Infer from compression_info.json or path name
5. **Backward compatible**: Can still manually specify `--output-dir`

### 🎯 Recommended Usage

```bash
# ✅ Recommended: No output-dir, use auto-organization
flashsvd finetune --checkpoint <path> --task <task> --epochs 3

# ✅ Also OK: Manually specify custom path
flashsvd finetune --checkpoint <path> --task <task> --output-dir <custom_path>
```

### 📁 Directory Structure Overview

```
FlashSVD/
├── compressed_models/          # Compressed models (original)
│   ├── bert-base_fwsvd_r64/
│   └── bert-base_whiten_r64/
│
└── models/
    └── finetuned/              # Fine-tuned models (auto-organized) ⭐
        ├── bert/
        │   ├── standard/
        │   ├── fwsvd/
        │   └── whiten/
        ├── modernbert/
        └── roberta/
```

---

**Update Date**: 2026-01-30
**Effective Version**: FlashSVD 0.1.0+
