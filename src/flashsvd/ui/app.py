"""
FlashSVD Gradio UI

Thin web interface wrapper for compression, evaluation, and checkpoint info.
All business logic is delegated to flashsvd.compress, flashsvd.evaluate, flashsvd.info.
"""

import sys
import io
import json
import os
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path

try:
    import gradio as gr
except ImportError:
    raise ImportError(
        "gradio is required for UI. Install with: pip install gradio>=4.0.0"
    )

import torch

# Import FlashSVD modules (thin wrapper - no business logic here)
from flashsvd.compress import CompressConfig, run_compress
from flashsvd.evaluate import EvalConfig, run_eval
from flashsvd.finetune import FineTuneConfig, run_finetune
from flashsvd.info import show_checkpoint_info


# ============================================================================
# GPU Detection
# ============================================================================

def detect_gpu_info():
    """
    Detect GPU availability and information.

    Returns:
        dict: {
            "available": bool,
            "count": int,
            "devices": list[str],  # ["cuda:0 (RTX 4090)", "cuda:1 (RTX 4090)", ...]
            "device_choices": list[str],  # ["cuda:0", "cuda:1", ..., "cpu"]
            "default_device": str,  # "cuda:0" or "cpu"
            "status_message": str,  # Display message
            "status_type": str,  # "success" or "warning"
        }
    """
    if not torch.cuda.is_available():
        return {
            "available": False,
            "count": 0,
            "devices": [],
            "device_choices": ["cpu"],
            "default_device": "cpu",
            "status_message": "⚠️ CUDA not available, defaulting to CPU (Triton kernels disabled)",
            "status_type": "warning"
        }

    count = torch.cuda.device_count()
    devices = []
    device_choices = []

    for i in range(count):
        name = torch.cuda.get_device_name(i)
        devices.append(f"cuda:{i} ({name})")
        device_choices.append(f"cuda:{i}")

    # Add CPU option
    device_choices.append("cpu")

    # Status message
    if count == 1:
        gpu_name = torch.cuda.get_device_name(0)
        status_message = f"✅ CUDA available: {gpu_name}"
    else:
        gpu_names = ", ".join([torch.cuda.get_device_name(i) for i in range(count)])
        status_message = f"✅ CUDA available: {count} GPUs ({gpu_names})"

    return {
        "available": True,
        "count": count,
        "devices": devices,
        "device_choices": device_choices,
        "default_device": "cuda:0",
        "status_message": status_message,
        "status_type": "success"
    }


# Detect GPU info at module load time
GPU_INFO = detect_gpu_info()


def scan_local_models():
    """
    Scan local models/ directory for available models.

    Returns:
        list[str]: List of local model paths
    """
    models_dir = Path("models")
    if not models_dir.exists():
        return []

    local_models = []
    for item in models_dir.iterdir():
        if item.is_dir():
            # Check if it's a valid model directory (has config.json)
            if (item / "config.json").exists():
                local_models.append(f"./models/{item.name}")

    return sorted(local_models)


def get_model_choices():
    """
    Get combined list of model choices for dropdown.

    Returns:
        list[str]: Common HF models + local models + "Custom"
    """
    # Common HuggingFace models
    common_models = [
        "bert-base-uncased",
        "bert-base-cased",
        "bert-large-uncased",
        "roberta-base",
        "roberta-large",
    ]

    # Local models
    local_models = scan_local_models()

    # Combine: common + local + custom
    all_choices = common_models + local_models + ["Custom"]

    return all_choices


def capture_output(func, *args, **kwargs):
    """
    Capture stdout/stderr from a function call.

    Returns:
        tuple: (success: bool, output: str, result: any)
    """
    output_buffer = io.StringIO()
    error_buffer = io.StringIO()
    result = None
    success = False

    try:
        with redirect_stdout(output_buffer), redirect_stderr(error_buffer):
            result = func(*args, **kwargs)
        success = True
        captured_output = output_buffer.getvalue()
        if error_buffer.getvalue():
            captured_output += "\n" + error_buffer.getvalue()
    except Exception as e:
        captured_output = f"❌ Error: {str(e)}\n\n"
        captured_output += error_buffer.getvalue()
        import traceback
        captured_output += "\n" + traceback.format_exc()

    return success, captured_output, result


# ============================================================================
# Tab 1: Compress
# ============================================================================

def run_compress_ui(
    model_name,
    custom_model,
    task,
    checkpoint_dir,
    arch,
    method,
    rank,
    rank_attn,
    rank_ffn,
    rank_wo,
    ranks_json,
    calib_samples,
    calib_task,
    modernbert_variant,
    ffn_kernel,
    output_dir,
    device
):
    """
    Thin wrapper for run_compress(). No business logic.

    M6: Added arch, ranks_json, calib_samples, calib_task, modernbert_variant
    M7: Added custom_model parameter for custom model paths

    Returns:
        tuple: (log_text, compression_info_json, compression_info_file)
    """
    # Handle custom model path
    if model_name == "Custom":
        if not custom_model or custom_model.strip() == "":
            return (
                "❌ Error: Please enter a custom model path when 'Custom' is selected",
                "",
                None
            )
        model_name = custom_model.strip()

    # Handle unified rank vs separate ranks
    if rank is not None and rank > 0:
        rank_attn = rank_ffn = rank_wo = rank

    # Create config (thin wrapper - just parameter passing)
    config = CompressConfig(
        model_name=model_name,
        task=task,
        arch=arch,
        method=method,
        rank_attn=rank_attn,
        rank_ffn=rank_ffn,
        rank_wo=rank_wo,
        ranks_json=ranks_json if ranks_json else None,
        calib_samples=calib_samples,
        calib_task=calib_task if calib_task else task,
        modernbert_variant=modernbert_variant,
        ffn_kernel=ffn_kernel,
        checkpoint_dir=checkpoint_dir if checkpoint_dir else None,
        output_dir=output_dir,
        device=device,
    )

    # Call business logic
    success, log_output, result = capture_output(run_compress, config)

    if success and result:
        # Read compression_info.json for display/download
        compression_info_path = os.path.join(result, "compression_info.json")
        if os.path.exists(compression_info_path):
            with open(compression_info_path, "r") as f:
                compression_info = json.load(f)

            compression_info_str = json.dumps(compression_info, indent=2)
            return (
                log_output,
                compression_info_str,
                compression_info_path
            )

    # Error case
    return (
        log_output,
        "Compression failed. See logs above.",
        None
    )


def create_compress_tab():
    """Create Compress tab UI (M6: Method Selection)."""
    with gr.Tab("Compress"):
        gr.Markdown("""
        ## Compress a Model with SVD

        Compress pretrained or finetuned models using various SVD methods.
        Supports BERT, RoBERTa, and ModernBERT architectures.
        """)

        with gr.Row():
            with gr.Column():
                gr.Markdown("### Model Settings")
                model_name = gr.Dropdown(
                    label="Model",
                    choices=get_model_choices(),
                    value="bert-base-uncased",
                    info="Select pretrained model or local checkpoint"
                )
                custom_model = gr.Textbox(
                    label="Custom Model Path",
                    placeholder="Enter custom model path (only if 'Custom' is selected above)",
                    value="",
                    visible=False,
                    info="Path to custom model directory or HuggingFace model ID"
                )
                task = gr.Dropdown(
                    label="Task",
                    choices=["cola", "sst2", "mrpc", "qqp", "mnli", "qnli", "rte", "stsb"],
                    value="sst2"
                )
                checkpoint_dir = gr.Textbox(
                    label="Finetuned Checkpoint (optional)",
                    placeholder="Leave empty for pretrained model",
                    value=""
                )

                gr.Markdown("### Compression Method")
                method = gr.Dropdown(
                    label="Method",
                    choices=[
                        ("Standard SVD", "standard"),
                        ("Fisher-Weighted SVD (FWSVD)", "fwsvd"),
                        ("Adaptive Rank Selection (AdaSVD)", "adasvd"),
                        ("Data-Aware Whitening (DRONE)", "whiten"),
                    ],
                    value="standard",
                    info="Choose compression algorithm"
                )

                arch = gr.Dropdown(
                    label="Architecture (auto-detect if unspecified)",
                    choices=["auto", "bert", "roberta", "modernbert"],
                    value="auto",
                    info="Leave as 'auto' for automatic detection",
                    visible=False  # Hide by default, show in advanced settings
                )

                # Method-specific settings (dynamically shown based on method)
                with gr.Accordion("Method-Specific Settings", open=False):
                    # AdaSVD-specific
                    with gr.Group(visible=False) as adasvd_group:
                        gr.Markdown("**AdaSVD Settings**")
                        ranks_json = gr.Textbox(
                            label="Ranks JSON Path (required)",
                            placeholder="Path to ranks.json (e.g., ./BERTAda/ars_out/ranks.json)",
                            value="",
                            info="JSON file with per-layer rank specifications"
                        )
                        ffn_kernel = gr.Dropdown(
                            label="FFN Kernel Variant",
                            choices=["v1", "v2"],
                            value="v1",
                            info="v1 (two-stage fusion, default) or v2 (full-batched fusion)"
                        )

                    # FWSVD/Whiten-specific
                    with gr.Group(visible=False) as calib_group:
                        gr.Markdown("**Calibration Settings (FWSVD/Whiten)**")
                        calib_samples = gr.Number(
                            label="Calibration Samples",
                            value=128,
                            precision=0,
                            info="Number of samples for Fisher weight/covariance estimation"
                        )
                        calib_task = gr.Textbox(
                            label="Calibration Task (optional)",
                            placeholder="Leave empty to use same as Task",
                            value="",
                            info="GLUE task for calibration (defaults to Task)"
                        )

                    # ModernBERT-specific (hidden, for future use)
                    modernbert_variant = gr.Dropdown(
                        label="ModernBERT Variant",
                        choices=["mask", "fwmask", "long"],
                        value="mask",
                        info="Only used if architecture is ModernBERT",
                        visible=False
                    )

                gr.Markdown("### Rank Settings")
                with gr.Row():
                    rank = gr.Number(
                        label="Unified Rank (set to 0 to use separate ranks)",
                        value=64,
                        precision=0
                    )

                with gr.Accordion("Separate Ranks (Advanced)", open=False):
                    rank_attn = gr.Number(
                        label="Attention Rank",
                        value=64,
                        precision=0
                    )
                    rank_ffn = gr.Number(
                        label="FFN Rank",
                        value=256,
                        precision=0
                    )
                    rank_wo = gr.Number(
                        label="Output Projection Rank",
                        value=256,
                        precision=0
                    )

                gr.Markdown("### Settings")
                output_dir = gr.Textbox(
                    label="Output Directory",
                    value="./compressed_models"
                )
                device = gr.Dropdown(
                    label="Device",
                    choices=GPU_INFO["device_choices"],
                    value=GPU_INFO["default_device"],
                    info=f"{len(GPU_INFO['device_choices'])-1} GPU(s) + CPU" if GPU_INFO["available"] else "CPU only"
                )

                compress_btn = gr.Button("🚀 Compress Model", variant="primary")

            with gr.Column():
                gr.Markdown("### Output")
                log_output = gr.Textbox(
                    label="Logs",
                    lines=20,
                    max_lines=30,
                    interactive=False
                )

                compression_info_display = gr.Code(
                    label="Compression Info (JSON)",
                    language="json",
                    lines=10
                )

                compression_info_file = gr.File(
                    label="Download compression_info.json"
                )

        # Wire up model dropdown to show/hide custom model textbox
        def toggle_custom_model(selected_model):
            return gr.update(visible=(selected_model == "Custom"))

        model_name.change(
            fn=toggle_custom_model,
            inputs=[model_name],
            outputs=[custom_model]
        )

        # Wire up method dropdown to show/hide method-specific settings
        def toggle_method_settings(selected_method):
            """Show/hide method-specific settings based on selected compression method."""
            method_lower = selected_method.lower()

            # Show AdaSVD settings if AdaSVD is selected
            show_adasvd = method_lower in ("adasvd", "ada")

            # Show calibration settings if FWSVD or Whiten is selected
            show_calib = method_lower in ("fwsvd", "fw", "whiten", "drone")

            return (
                gr.update(visible=show_adasvd),  # adasvd_group
                gr.update(visible=show_calib),   # calib_group
            )

        method.change(
            fn=toggle_method_settings,
            inputs=[method],
            outputs=[adasvd_group, calib_group]
        )

        # Wire up the compress button (M6: added arch and method-specific inputs; M7: added custom_model)
        compress_btn.click(
            fn=run_compress_ui,
            inputs=[
                model_name, custom_model, task, checkpoint_dir, arch, method,
                rank, rank_attn, rank_ffn, rank_wo,
                ranks_json, calib_samples, calib_task, modernbert_variant, ffn_kernel,
                output_dir, device
            ],
            outputs=[log_output, compression_info_display, compression_info_file]
        )


# ============================================================================
# Tab 2: Evaluate
# ============================================================================

def run_eval_ui(
    checkpoint_dir,
    task,
    batch_size,
    seq_len,
    max_eval_samples,
    device,
    output_file
):
    """
    Thin wrapper for run_eval(). No business logic.

    Returns:
        tuple: (log_text, results_json, results_file)
    """
    # Create config (thin wrapper - just parameter passing)
    config = EvalConfig(
        checkpoint_dir=checkpoint_dir,
        task=task,
        batch_size=batch_size,
        seq_len=seq_len,
        max_eval_samples=max_eval_samples if max_eval_samples > 0 else None,
        device=device,
        output=output_file,
    )

    # Call business logic
    success, log_output, result = capture_output(run_eval, config)

    if success and result:
        # Display results JSON
        results_str = json.dumps(result, indent=2)

        # Return file path for download
        return (
            log_output,
            results_str,
            output_file if os.path.exists(output_file) else None
        )

    # Error case
    return (
        log_output,
        "Evaluation failed. See logs above.",
        None
    )


def create_evaluate_tab():
    """Create Evaluate tab UI."""
    with gr.Tab("Evaluate"):
        gr.Markdown("""
        ## Evaluate Compressed Model

        Evaluate compressed model on GLUE tasks with accuracy and performance metrics.
        """)

        with gr.Row():
            with gr.Column():
                gr.Markdown("### Model Settings")
                checkpoint_dir = gr.Textbox(
                    label="Checkpoint Directory",
                    placeholder="./compressed_models/bert-base-uncased_standard_r64"
                )
                task = gr.Dropdown(
                    label="Task",
                    choices=["cola", "sst2", "mrpc", "qqp", "mnli", "qnli", "rte", "stsb"],
                    value="sst2"
                )

                gr.Markdown("### Evaluation Settings")
                batch_size = gr.Number(
                    label="Batch Size",
                    value=32,
                    precision=0
                )
                seq_len = gr.Number(
                    label="Sequence Length",
                    value=128,
                    precision=0
                )
                max_eval_samples = gr.Number(
                    label="Max Eval Samples (0 = all)",
                    value=0,
                    precision=0
                )

                device = gr.Dropdown(
                    label="Device",
                    choices=GPU_INFO["device_choices"],
                    value=GPU_INFO["default_device"],
                    info=f"{len(GPU_INFO['device_choices'])-1} GPU(s) + CPU" if GPU_INFO["available"] else "CPU only"
                )

                output_file = gr.Textbox(
                    label="Output File",
                    value="eval_results.json"
                )

                eval_btn = gr.Button("📊 Evaluate Model", variant="primary")

            with gr.Column():
                gr.Markdown("### Results")
                log_output = gr.Textbox(
                    label="Logs",
                    lines=15,
                    max_lines=25,
                    interactive=False
                )

                results_display = gr.Code(
                    label="Results (JSON)",
                    language="json",
                    lines=15
                )

                results_file = gr.File(
                    label="Download eval_results.json"
                )

        # Wire up the evaluate button
        eval_btn.click(
            fn=run_eval_ui,
            inputs=[
                checkpoint_dir, task, batch_size, seq_len,
                max_eval_samples, device, output_file
            ],
            outputs=[log_output, results_display, results_file]
        )


# ============================================================================
# Tab 3: Info
# ============================================================================

def run_info_ui(checkpoint_dir):
    """
    Thin wrapper for show_checkpoint_info(). No business logic.

    Returns:
        tuple: (log_text, compression_info_json, compression_info_file)
    """
    if not checkpoint_dir or not os.path.exists(checkpoint_dir):
        return (
            f"❌ Checkpoint directory not found: {checkpoint_dir}",
            "",
            None
        )

    # Call business logic
    success, log_output, _ = capture_output(show_checkpoint_info, checkpoint_dir)

    # Try to read compression_info.json
    compression_info_path = os.path.join(checkpoint_dir, "compression_info.json")
    compression_info_str = ""
    compression_info_file = None

    if os.path.exists(compression_info_path):
        with open(compression_info_path, "r") as f:
            compression_info = json.load(f)
        compression_info_str = json.dumps(compression_info, indent=2)
        compression_info_file = compression_info_path

    return (
        log_output,
        compression_info_str,
        compression_info_file
    )


# ============================================================================
# Tab 3: Fine-tune
# ============================================================================

def run_finetune_ui(
    checkpoint_dir,
    task,
    epochs,
    learning_rate,
    batch_size,
    eval_batch_size,
    optimizer,
    weight_decay,
    max_grad_norm,
    lr_scheduler,
    warmup_ratio,
    warmup_steps,
    logging_steps,
    eval_steps,
    save_steps,
    early_stopping,
    patience,
    freeze_embeddings,
    freeze_attention,
    freeze_ffn,
    max_train_samples,
    max_eval_samples,
    max_seq_length,
    output_dir,
    device,
    seed,
    use_tensorboard,
):
    """
    Thin wrapper for run_finetune(). No business logic.

    Returns:
        tuple: (log_text, best_checkpoint_path)
    """
    # Create config (thin wrapper - just parameter passing)
    config = FineTuneConfig(
        checkpoint_dir=checkpoint_dir,
        task=task,
        epochs=epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
        eval_batch_size=eval_batch_size,
        optimizer=optimizer,
        weight_decay=weight_decay,
        max_grad_norm=max_grad_norm,
        lr_scheduler=lr_scheduler,
        warmup_ratio=warmup_ratio,
        warmup_steps=warmup_steps if warmup_steps > 0 else None,
        logging_steps=logging_steps,
        eval_steps=eval_steps,
        save_steps=save_steps,
        early_stopping=early_stopping,
        early_stopping_patience=patience,
        freeze_embeddings=freeze_embeddings,
        freeze_attention=freeze_attention,
        freeze_ffn=freeze_ffn,
        max_train_samples=max_train_samples if max_train_samples > 0 else None,
        max_eval_samples=max_eval_samples if max_eval_samples > 0 else None,
        max_seq_length=max_seq_length,
        output_dir=output_dir if output_dir else None,
        device=device,
        seed=seed,
        use_tensorboard=use_tensorboard,
    )

    # Call business logic
    success, log_output, best_checkpoint_path = capture_output(run_finetune, config)

    if success and best_checkpoint_path:
        return (
            log_output,
            f"✅ Fine-tuning complete!\n\nBest checkpoint: {best_checkpoint_path}"
        )

    # Error case
    return (
        log_output,
        "❌ Fine-tuning failed. See logs above."
    )


def create_finetune_tab():
    """Create Fine-tune tab UI."""
    with gr.Tab("Fine-tune"):
        gr.Markdown("""
        ## Fine-tune Compressed Model

        Fine-tune a compressed model to recover accuracy after SVD compression.
        Supports all GLUE tasks with comprehensive hyperparameter customization.
        """)

        with gr.Row():
            with gr.Column(scale=1):
                # Required inputs
                gr.Markdown("### Required Settings")
                checkpoint_dir = gr.Textbox(
                    label="Compressed Checkpoint Directory",
                    placeholder="./compressed_models/bert-base-uncased_standard_r64",
                    info="Path to compressed model checkpoint"
                )

                task = gr.Dropdown(
                    label="GLUE Task",
                    choices=["cola", "sst2", "mrpc", "qqp", "mnli", "qnli", "rte", "stsb"],
                    value="sst2",
                    info="Target downstream task for fine-tuning"
                )

                # Training hyperparameters
                gr.Markdown("### Training Hyperparameters")
                with gr.Row():
                    epochs = gr.Slider(1, 10, value=3, step=1, label="Epochs")
                    learning_rate = gr.Number(value=3e-5, label="Learning Rate", info="e.g., 3e-5, 2e-5, 5e-5")

                with gr.Row():
                    batch_size = gr.Slider(8, 64, value=32, step=8, label="Batch Size")
                    eval_batch_size = gr.Slider(8, 128, value=64, step=8, label="Eval Batch Size")

                # Optimizer settings
                gr.Markdown("### Optimizer & Scheduler")
                with gr.Row():
                    optimizer = gr.Dropdown(
                        label="Optimizer",
                        choices=["adamw", "adam", "sgd"],
                        value="adamw"
                    )
                    lr_scheduler = gr.Dropdown(
                        label="LR Scheduler",
                        choices=["linear", "cosine", "constant", "polynomial"],
                        value="linear"
                    )

                with gr.Row():
                    weight_decay = gr.Number(value=0.01, label="Weight Decay")
                    max_grad_norm = gr.Number(value=1.0, label="Max Grad Norm")

                with gr.Row():
                    warmup_ratio = gr.Slider(0.0, 0.3, value=0.1, step=0.05, label="Warmup Ratio")
                    warmup_steps = gr.Number(value=0, label="Warmup Steps (0=use ratio)", precision=0)

                # Training strategy
                gr.Markdown("### Training Strategy")
                with gr.Row():
                    logging_steps = gr.Number(value=50, label="Logging Steps", precision=0)
                    eval_steps = gr.Number(value=500, label="Eval Steps", precision=0)
                    save_steps = gr.Number(value=500, label="Save Steps", precision=0)

                with gr.Row():
                    early_stopping = gr.Checkbox(label="Early Stopping", value=False)
                    patience = gr.Number(value=3, label="Patience", precision=0, info="For early stopping")

                # Freezing strategy
                gr.Markdown("### Parameter Freezing")
                with gr.Row():
                    freeze_embeddings = gr.Checkbox(label="Freeze Embeddings", value=False)
                    freeze_attention = gr.Checkbox(label="Freeze Attention", value=False)
                    freeze_ffn = gr.Checkbox(label="Freeze FFN", value=False)

                # Data settings
                gr.Markdown("### Data Settings")
                with gr.Row():
                    max_train_samples = gr.Number(
                        value=0,
                        label="Max Train Samples (0=all)",
                        precision=0,
                        info="Limit training samples for quick testing"
                    )
                    max_eval_samples = gr.Number(
                        value=0,
                        label="Max Eval Samples (0=all)",
                        precision=0
                    )

                max_seq_length = gr.Slider(64, 512, value=128, step=64, label="Max Sequence Length")

                # Output & device
                gr.Markdown("### Output & Device")
                output_dir = gr.Textbox(
                    label="Output Directory (optional)",
                    placeholder="Leave empty to overwrite checkpoint",
                    info="If empty, saves to checkpoint directory"
                )

                with gr.Row():
                    device = gr.Dropdown(
                        label="Device",
                        choices=GPU_INFO["device_choices"],
                        value=GPU_INFO["default_device"],
                        info=f"💡 {GPU_INFO['status_message']}"
                    )
                    seed = gr.Number(value=42, label="Random Seed", precision=0)

                use_tensorboard = gr.Checkbox(label="Use TensorBoard Logging", value=False)

                # Run button
                finetune_btn = gr.Button("🚀 Start Fine-tuning", variant="primary", size="lg")

            with gr.Column(scale=1):
                gr.Markdown("### Output")
                log_output = gr.Textbox(
                    label="Training Logs",
                    lines=25,
                    max_lines=40,
                    interactive=False,
                    show_copy_button=True
                )

                result_display = gr.Textbox(
                    label="Result",
                    lines=4,
                    interactive=False,
                    show_copy_button=True
                )

                gr.Markdown("""
                ### Tips
                - **Quick Test**: Use `--max-train-samples 1000 --epochs 1` for fast validation
                - **Standard**: 3-5 epochs with full dataset usually works well
                - **Early Stopping**: Enable to prevent overfitting
                - **Freezing**: Freeze embeddings if training time is limited
                - **Learning Rate**: 2e-5 to 5e-5 is typical for fine-tuning
                - **Batch Size**: Larger batches (32-64) are more stable
                """)

        # Wire up the finetune button
        finetune_btn.click(
            fn=run_finetune_ui,
            inputs=[
                checkpoint_dir, task, epochs, learning_rate, batch_size, eval_batch_size,
                optimizer, weight_decay, max_grad_norm, lr_scheduler, warmup_ratio, warmup_steps,
                logging_steps, eval_steps, save_steps, early_stopping, patience,
                freeze_embeddings, freeze_attention, freeze_ffn,
                max_train_samples, max_eval_samples, max_seq_length,
                output_dir, device, seed, use_tensorboard
            ],
            outputs=[log_output, result_display]
        )


def create_info_tab():
    """Create Info tab UI."""
    with gr.Tab("Info"):
        gr.Markdown("""
        ## Checkpoint Information

        Display metadata and information about compressed model checkpoints.
        """)

        with gr.Row():
            with gr.Column():
                gr.Markdown("### Checkpoint")
                checkpoint_dir = gr.Textbox(
                    label="Checkpoint Directory",
                    placeholder="./compressed_models/bert-base-uncased_standard_r64"
                )

                info_btn = gr.Button("ℹ️ Show Info", variant="primary")

            with gr.Column():
                gr.Markdown("### Information")
                log_output = gr.Textbox(
                    label="Checkpoint Details",
                    lines=20,
                    max_lines=30,
                    interactive=False
                )

                compression_info_display = gr.Code(
                    label="compression_info.json",
                    language="json",
                    lines=10
                )

                compression_info_file = gr.File(
                    label="Download compression_info.json"
                )

        # Wire up the info button
        info_btn.click(
            fn=run_info_ui,
            inputs=[checkpoint_dir],
            outputs=[log_output, compression_info_display, compression_info_file]
        )


# ============================================================================
# Main App
# ============================================================================

def create_app():
    """Create the main Gradio app with all tabs."""
    with gr.Blocks(title="FlashSVD", theme=gr.themes.Soft()) as app:
        gr.Markdown("""
        # FlashSVD: Memory-Efficient Inference for Low-Rank Models

        Compress, evaluate, and inspect SVD-compressed language models.
        """)

        # GPU Status Banner
        if GPU_INFO["status_type"] == "success":
            status_html = f"""
            <div style="padding: 12px; background-color: #d4edda; border: 1px solid #c3e6cb;
                        border-radius: 6px; color: #155724; margin-bottom: 20px;">
                <strong>{GPU_INFO["status_message"]}</strong>
            """
            if GPU_INFO["count"] > 1:
                status_html += "<br><small>💡 Select GPU device in each tab's Device dropdown</small>"
            status_html += "</div>"
        else:
            status_html = f"""
            <div style="padding: 12px; background-color: #fff3cd; border: 1px solid #ffeaa7;
                        border-radius: 6px; color: #856404; margin-bottom: 20px;">
                <strong>{GPU_INFO["status_message"]}</strong>
            </div>
            """

        gr.HTML(status_html)

        with gr.Tabs():
            create_compress_tab()
            create_evaluate_tab()
            create_finetune_tab()
            create_info_tab()

        gr.Markdown("""
        ---

        **Documentation**: [GitHub](https://github.com/Zishan-Shao/FlashSVD) |
        **Paper**: [arXiv](https://arxiv.org/abs/2508.01506)
        """)

    return app


def main():
    """CLI entry point for flashsvd-ui."""
    print("=" * 60)
    print("FlashSVD Gradio UI")
    print("=" * 60)
    print("Starting web interface...")
    print()

    app = create_app()

    # Launch with share=False by default (user can change in code if needed)
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )


if __name__ == "__main__":
    main()
