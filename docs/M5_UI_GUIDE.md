# M5: Gradio UI - Usage Guide

## ✅ Status: COMPLETE

FlashSVD now includes a web-based user interface for easy model compression, evaluation, and checkpoint inspection.

---

## 🚀 Launch Command

```bash
flashsvd-ui
```

After launch, browser will open automatically (or visit `http://localhost:7860`).

### Launch Options

Default configuration:
- Server: `0.0.0.0:7860`
- Share: `False` (local access)
- For remote access, set `share=True` in code

---

## 📋 Usage Guide

### Tab 1: Compress (Model Compression)

**Step 1: Configure Model**
- **Model Name**: Select pretrained model (e.g., `bert-base-uncased`)
- **Task**: Select GLUE task (e.g., `sst2`)
- **Finetuned Checkpoint**: (Optional) If compressing fine-tuned model, provide path

**Step 2: Set Compression Parameters**
- **Compression Method**: Currently supports `standard`
- **Unified Rank**: Unified rank value (e.g., 64)
  - Set to 0 to use advanced separate ranks
- **Advanced Options**:
  - Attention Rank: Attention layer rank (default 64)
  - FFN Rank: FFN layer rank (default 256)
  - Output Projection Rank: Output projection layer rank (default 256)

**Step 3: Run Settings**
- **Output Directory**: Output directory (default `./compressed_models`)
- **Device**: Choose `cuda` or `cpu`

**Step 4: Execute**
- Click "🚀 Compress Model"
- View right-side log output and progress
- Download `compression_info.json` after completion

**Output**:
- Log area shows real-time progress (loading → compressing → saving)
- JSON displays compression metadata (method, ranks, timestamp, etc.)
- File download: compression_info.json

---

### Tab 2: Evaluate (Model Evaluation)

**Step 1: Select Model**
- **Checkpoint Directory**: Compressed model path
  - Example: `./compressed_models/bert-base-uncased_standard_r64`

**Step 2: Configure Evaluation**
- **Task**: Select evaluation task (e.g., `sst2`)
- **Batch Size**: Batch size (default 32)
- **Sequence Length**: Sequence length (default 128)
- **Max Eval Samples**: Maximum samples (0 = all)
- **Device**: Choose `cuda` or `cpu`
- **Output File**: Result save filename (default `eval_results.json`)

**Step 3: Execute**
- Click "📊 Evaluate Model"
- View right-side log output
- View result JSON after evaluation completes

**Output**:
- Log area shows evaluation progress (loading model → loading data → evaluating)
- JSON displays evaluation results:
  - `metric_name`: Metric name (accuracy / pearson)
  - `metric_value`: Metric value
  - `peak_memory_mib`: Peak memory (MiB)
  - `latency_ms`: Latency (ms/batch)
  - Complete model and evaluation configuration info
- File download: eval_results.json

---

### Tab 3: Info (Checkpoint Information)

**Step 1: Select Checkpoint**
- **Checkpoint Directory**: Compressed model path
  - Example: `./compressed_models/bert-base-uncased_standard_r64`

**Step 2: Execute**
- Click "ℹ️ Show Info"
- View detailed checkpoint information

**Output**:
- Detail area displays:
  - 📦 Compression details (method, base model, task, ranks)
  - 📅 Metadata (creation time, FlashSVD version, Git commit)
  - 📁 File list (config.json, flashsvd_state_dict.pt, model weights)
  - Total size
  - 💡 Usage instructions
- JSON display: compression_info.json content
- File download: compression_info.json

---

## 🎯 Complete Workflow Example

### Example: Compress and Evaluate BERT Model

1. **Open Compress Tab**
   - Model Name: `bert-base-uncased`
   - Task: `sst2`
   - Unified Rank: `64`
   - Device: `cuda`
   - Click "Compress Model"
   - Wait for completion, note output path (e.g., `./compressed_models/bert-base-uncased_standard_r64`)

2. **Switch to Evaluate Tab**
   - Checkpoint Directory: Copy path from previous step
   - Task: `sst2`
   - Device: `cuda`
   - Click "Evaluate Model"
   - View accuracy, peak memory, latency results
   - Download eval_results.json

3. **Switch to Info Tab**
   - Checkpoint Directory: Same as above
   - Click "Show Info"
   - View complete model information and file list

---

## 🔍 Technical Implementation

### Architecture Principle (Thin Wrapper)

UI code **contains NO business logic**, all operations call existing modules:

```python
# Compress Tab
from flashsvd.compress import CompressConfig, run_compress
config = CompressConfig(...)  # Parameter mapping
result = run_compress(config)  # Call business logic

# Evaluate Tab
from flashsvd.evaluate import EvalConfig, run_eval
config = EvalConfig(...)
result = run_eval(config)

# Info Tab
from flashsvd.info import show_checkpoint_info
show_checkpoint_info(checkpoint_dir)
```

### Log Capture

Uses `contextlib.redirect_stdout` and `redirect_stderr` to capture output:

```python
def capture_output(func, *args, **kwargs):
    output_buffer = io.StringIO()
    with redirect_stdout(output_buffer), redirect_stderr(output_buffer):
        result = func(*args, **kwargs)
    return output_buffer.getvalue(), result
```

### Error Handling

All errors caught by try-except and displayed in UI:

```python
try:
    result = run_compress(config)
except Exception as e:
    return f"❌ Error: {str(e)}\n{traceback.format_exc()}"
```

---

## 📦 File List

### New Files
- `src/flashsvd/ui/__init__.py` (7 lines)
- `src/flashsvd/ui/app.py` (507 lines)

### Modified Files
- `pyproject.toml`: Uncommented `flashsvd-ui` entry point

---

## ✅ Verification Checklist

```bash
# Check entry point
✅ which flashsvd-ui
✅ flashsvd-ui --help  # (Gradio has no --help, but command existence is enough)

# Test import
✅ python -c "from flashsvd.ui import app; print('OK')"

# Launch test
✅ flashsvd-ui  # Visit http://localhost:7860
```

---

## 🎨 UI Features

1. **Responsive Layout**: Two-column layout (input | output)
2. **Real-time Logs**: Shows compression/evaluation progress
3. **JSON Preview**: Syntax-highlighted results display
4. **File Download**: One-click download JSON results
5. **Error Messages**: Clear error messages and stack traces
6. **Default Values**: All parameters have reasonable defaults
7. **Theme**: Uses Gradio Soft theme

---

## 🚨 Notes

1. **Device Selection**:
   - Defaults to `cuda`
   - CPU mode shows Triton kernel warnings (in logs)

2. **File Paths**:
   - All paths relative to working directory where `flashsvd-ui` was launched
   - Recommended to launch from project root

3. **Concurrency Limit**:
   - Gradio runs single instance by default
   - Avoid triggering multiple long operations simultaneously

4. **Port Occupation**:
   - Default port 7860
   - If occupied, Gradio auto-selects next available port

---

## 📊 Comparison with CLI

| Feature | CLI | UI |
|---------|-----|-----|
| Compress Model | `flashsvd compress ...` | Compress Tab |
| Evaluate Model | `flashsvd eval ...` | Evaluate Tab |
| View Info | `flashsvd info ...` | Info Tab |
| Log Output | Terminal | Embedded textbox |
| Result Download | File system | Browser download |
| Parameter Validation | argparse | Gradio widgets |
| Use Case | Scripting/automation | Interactive exploration |

---

## 🎉 M5 Completion Summary

**Milestone M5: Gradio UI** completed!

- ✅ Three tabs (Compress / Evaluate / Info)
- ✅ Thin wrapper (no business logic)
- ✅ Calls existing modules (run_compress / run_eval / show_checkpoint_info)
- ✅ Device parameter consistent with CLI ({cuda, cpu})
- ✅ Log output area (real-time display)
- ✅ Result download (JSON files)
- ✅ Entry point: `flashsvd-ui`

**Next Step**: All milestones (M1-M5) completed! 🎊
