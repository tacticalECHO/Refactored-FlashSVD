# FlashSVD UI Quick Start Guide

## 🎯 Improvement Summary

### ✅ Completed Improvements

1. **Unified Import Paths** - All compression modules use unified `flashsvd.utils` imports
2. **Optimized Method Selection** - 4 main methods with clear descriptions
3. **Auto Local Model Detection** - Auto-scan `models/` directory
4. **Smart Model Selection** - Dropdown menu + custom path support

---

## 🚀 Launch UI

```bash
# Method 1: Using command line
flashsvd-ui

# Method 2: Using Python module
python -m flashsvd.ui.app

# Method 3: Direct run
python src/flashsvd/ui/app.py
```

Visit: http://localhost:7860

---

## 📖 Usage Guide

### Compress Tab - Compress

#### 1. Select Model

**Dropdown menu includes:**
- HuggingFace pretrained models:
  - bert-base-uncased
  - bert-base-cased
  - bert-large-uncased
  - roberta-base
  - roberta-large

- Local models (auto-detected from `models/` directory):
  - ./models/bert-sst2-finetuned
  - (Other local models appear automatically)

- Custom (custom path):
  - Select to show input box
  - Enter any HuggingFace model ID or local path

**Example:**
```
Model: [bert-base-uncased ▼]
```

If selecting "Custom":
```
Model: [Custom ▼]
Custom Model Path: [my-org/my-model___________]
```

#### 2. Select Compression Method

**4 main methods:**

| Display Name | Actual Value | Description |
|-------------|--------------|-------------|
| Standard SVD | standard | Standard SVD decomposition |
| Fisher-Weighted SVD (FWSVD) | fwsvd | Fisher information weighted |
| Adaptive Rank Selection (AdaSVD) | adasvd | Adaptive rank selection |
| Data-Aware Whitening (DRONE) | whiten | Data-aware whitening |

**Example:**
```
Method: [Standard SVD ▼]
  Standard SVD
  Fisher-Weighted SVD (FWSVD)
  Adaptive Rank Selection (AdaSVD)
  Data-Aware Whitening (DRONE)
```

#### 3. Set Rank Parameters

```
Unified Rank: [64]  (Set to 0 to use separate ranks below)

Or:

Rank Attn: [40]
Rank FFN:  [240]
Rank Wo:   [240]
```

#### 4. Method-Specific Settings (Optional)

**AdaSVD requires:**
- Ranks JSON: `./experiments/BERTAda/ars_out/ranks.json`

**FWSVD/Whiten require:**
- Calibration Samples: 128
- Calibration Task: sst2 (or leave blank to use main task)

#### 5. Output Settings

```
Output Directory: [./compressed_models]
Device: [cuda:0 ▼]
```

#### 6. Run Compression

Click **Compress Model** button and view:
- Log output (real-time progress display)
- Compression info JSON (metadata)
- Download button (save compression_info.json)

---

### Evaluate Tab - Evaluate

#### 1. Select Checkpoint

```
Checkpoint Directory: [./compressed_models/bert-base-uncased_standard_r64]
```

#### 2. Evaluation Settings

```
Task: [sst2 ▼]
Batch Size: [32]
Sequence Length: [128]
Max Eval Samples: [0]  (0=all)
```

#### 3. Run Evaluation

Click **Evaluate** button and view:
- Log output
- Evaluation result JSON (includes accuracy, memory, latency, etc.)
- Download button

---

### Info Tab - Info

#### View Checkpoint Information

```
Checkpoint Directory: [./compressed_models/bert-base-uncased_standard_r64]
```

Click **Show Info** to view:
- Model architecture
- Compression method
- Rank configuration
- Timestamp
- File list

---

## 🧪 Test Examples

### Example 1: Compress Using Local Model

```
1. Model: ./models/bert-sst2-finetuned
2. Task: sst2
3. Method: Standard SVD
4. Unified Rank: 64
5. Output Directory: ./compressed_models
6. Device: cuda:0
7. Click "Compress Model"
```

### Example 2: Use Custom Model Path

```
1. Model: Custom
2. Custom Model Path: google/bert_uncased_L-2_H-128_A-2
3. Task: cola
4. Method: Fisher-Weighted SVD (FWSVD)
5. Calibration Samples: 128
6. Unified Rank: 32
7. Click "Compress Model"
```

### Example 3: AdaSVD with Adaptive Ranks

```
1. Model: bert-base-uncased
2. Task: mnli
3. Method: Adaptive Rank Selection (AdaSVD)
4. Ranks JSON: ./experiments/BERTAda/ars_out/ranks.json
5. FFN Kernel: v1
6. Click "Compress Model"
```

---

## 📝 Notes

1. **Local Model Detection**
   - Only scans directories containing `config.json`
   - Model path format: `./models/model-name`
   - Auto-scans on UI startup

2. **Custom Models**
   - Input box only appears after selecting "Custom"
   - Supports HuggingFace model ID (e.g., `bert-base-uncased`)
   - Supports local paths (e.g., `./my-models/my-bert`)
   - Supports relative and absolute paths

3. **Method-Specific Requirements**
   - AdaSVD: Must provide ranks.json
   - FWSVD/Whiten: Optional calibration settings
   - Standard: No additional requirements

4. **GPU Detection**
   - Auto-detects CUDA on UI startup
   - If no GPU, prompts to use CPU (Triton kernels disabled)

---

## 🔍 Troubleshooting

### Issue 1: Local Model Not Appearing in Dropdown

**Solution:**
```bash
# Check models/ directory structure
ls -lh models/

# Ensure contains config.json
ls models/your-model/config.json

# Restart UI
```

### Issue 2: Custom Model Path Invalid

**Solution:**
- Ensure path is correct (relative paths start from current directory)
- Check model directory contains `config.json` and `pytorch_model.bin`/`model.safetensors`

### Issue 3: Import Errors

**Solution:**
```bash
# Reinstall
pip install -e .

# Verify imports
python -c "from flashsvd.compression import whiten, adasvd; print('OK')"
```

---

## 📚 More Resources

- **Full Documentation**: `CLAUDE.md`
- **Installation Guide**: `README.md`
- **UI Detailed Guide**: `M5_UI_GUIDE.md`
- **Improvement Summary**: `/tmp/ui_improvements_summary.md`
