"""
Core training logic for fine-tuning compressed models.
"""

import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_constant_schedule_with_warmup,
    get_polynomial_decay_schedule_with_warmup,
)
from datasets import load_dataset
from evaluate import load as load_metric
from tqdm import tqdm
from typing import Optional
import numpy as np
from datetime import datetime

from flashsvd.io import load_compressed
from .config import FineTuneConfig


def get_optimizer(model: nn.Module, config: FineTuneConfig) -> torch.optim.Optimizer:
    """Create optimizer based on config."""
    if config.optimizer == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            betas=(config.adam_beta1, config.adam_beta2),
            eps=config.adam_epsilon,
            weight_decay=config.weight_decay,
        )
    elif config.optimizer == "adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=config.learning_rate,
            betas=(config.adam_beta1, config.adam_beta2),
            eps=config.adam_epsilon,
        )
    elif config.optimizer == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=config.learning_rate,
            momentum=0.9,
            weight_decay=config.weight_decay,
        )
    else:
        raise ValueError(f"Unknown optimizer: {config.optimizer}")


def get_scheduler(optimizer: torch.optim.Optimizer, config: FineTuneConfig, num_training_steps: int):
    """Create learning rate scheduler."""
    # 计算预热步数
    if config.warmup_steps is not None:
        num_warmup_steps = config.warmup_steps
    else:
        num_warmup_steps = int(num_training_steps * config.warmup_ratio)

    if config.lr_scheduler == "linear":
        return get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)
    elif config.lr_scheduler == "cosine":
        return get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)
    elif config.lr_scheduler == "constant":
        return get_constant_schedule_with_warmup(optimizer, num_warmup_steps)
    elif config.lr_scheduler == "polynomial":
        return get_polynomial_decay_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, power=2.0)
    else:
        raise ValueError(f"Unknown scheduler: {config.lr_scheduler}")


def freeze_parameters(model: nn.Module, config: FineTuneConfig):
    """Freeze model parameters based on config."""
    if config.freeze_embeddings:
        print("  Freezing embeddings...")
        if hasattr(model, 'bert'):
            model.bert.embeddings.requires_grad_(False)
        elif hasattr(model, 'roberta'):
            model.roberta.embeddings.requires_grad_(False)

    if config.freeze_attention or config.freeze_ffn:
        print(f"  Freezing: attention={config.freeze_attention}, ffn={config.freeze_ffn}")
        encoder = model.bert.encoder if hasattr(model, 'bert') else model.roberta.encoder

        for layer in encoder.layer:
            if config.freeze_attention:
                # 冻结 attention 相关参数
                if hasattr(layer, 'block'):
                    for name, param in layer.block.named_parameters():
                        if any(x in name for x in ['Pq', 'Vq', 'bq', 'Pk', 'Vk', 'bk',
                                                     'Pv', 'Vv', 'bv', 'Uo', 'Vo', 'bo_attn']):
                            param.requires_grad = False

            if config.freeze_ffn:
                # 冻结 FFN 相关参数
                if hasattr(layer, 'block'):
                    for name, param in layer.block.named_parameters():
                        if any(x in name for x in ['U1', 'V1', 'b1', 'U2', 'V2', 'b2']):
                            param.requires_grad = False

    # 统计可训练参数
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Trainable parameters: {trainable_params:,} / {total_params:,} "
          f"({100 * trainable_params / total_params:.1f}%)")


def load_and_prepare_data(config: FineTuneConfig, tokenizer):
    """Load and tokenize GLUE dataset."""
    # 加载数据集
    if config.task == "mnli":
        train_dataset = load_dataset("glue", config.task, split="train")
        val_dataset = load_dataset("glue", config.task, split="validation_matched")
    else:
        train_dataset = load_dataset("glue", config.task, split="train")
        val_dataset = load_dataset("glue", config.task, split="validation")

    # 采样（如果指定）
    if config.max_train_samples:
        train_dataset = train_dataset.select(range(min(config.max_train_samples, len(train_dataset))))
    if config.max_eval_samples:
        val_dataset = val_dataset.select(range(min(config.max_eval_samples, len(val_dataset))))

    # Tokenize
    single_sent_tasks = {"cola", "sst2"}
    field_map = {
        "qqp": ("question1", "question2"),
        "mnli": ("premise", "hypothesis"),
        "qnli": ("question", "sentence"),
        "stsb": ("sentence1", "sentence2"),
        "rte": ("sentence1", "sentence2"),
        "mrpc": ("sentence1", "sentence2"),
    }

    def tokenize_fn(examples):
        if config.task in single_sent_tasks:
            return tokenizer(
                examples["sentence"],
                padding="max_length",
                truncation=True,
                max_length=config.max_seq_length,
            )
        else:
            f1, f2 = field_map[config.task]
            return tokenizer(
                examples[f1],
                examples[f2],
                padding="max_length",
                truncation=True,
                max_length=config.max_seq_length,
            )

    train_dataset = train_dataset.map(tokenize_fn, batched=True)
    val_dataset = val_dataset.map(tokenize_fn, batched=True)

    # 重命名 label -> labels（HuggingFace模型需要）
    train_dataset = train_dataset.rename_column("label", "labels")
    val_dataset = val_dataset.rename_column("label", "labels")

    # 设置格式
    train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    val_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    return train_dataset, val_dataset


class EarlyStopping:
    """Early stopping handler."""

    def __init__(self, patience: int = 3, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_metric = None
        self.counter = 0
        self.should_stop = False

    def __call__(self, metric: float) -> bool:
        """Returns True if should stop."""
        if self.best_metric is None:
            self.best_metric = metric
            return False

        if metric > self.best_metric + self.min_delta:
            self.best_metric = metric
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

        return self.should_stop


def evaluate_model(model, dataloader, metric, device):
    """Evaluate model on validation set."""
    model.eval()

    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}

        with torch.no_grad():
            outputs = model(**batch)

        predictions = outputs.logits.argmax(dim=-1)
        metric.add_batch(predictions=predictions, references=batch["labels"])

    result = metric.compute()
    model.train()

    return result


def save_checkpoint(model, config: FineTuneConfig, epoch: int, step: int, best: bool = False):
    """Save model checkpoint."""
    if best:
        save_dir = os.path.join(config.output_dir, "best")
    else:
        save_dir = os.path.join(config.output_dir, f"checkpoint-{epoch}-{step}")

    os.makedirs(save_dir, exist_ok=True)

    # 保存模型
    torch.save(
        model.state_dict(),
        os.path.join(save_dir, "flashsvd_state_dict.pt")
    )

    # 保存配置
    compression_info_path = os.path.join(config.checkpoint_dir, "compression_info.json")
    if os.path.exists(compression_info_path):
        with open(compression_info_path) as f:
            compression_info = json.load(f)

        # 更新元数据
        compression_info["finetuned"] = True
        compression_info["finetune_epochs"] = epoch + 1
        compression_info["finetune_timestamp"] = datetime.now().isoformat()

        with open(os.path.join(save_dir, "compression_info.json"), 'w') as f:
            json.dump(compression_info, f, indent=2)

    # 保存 config.json（如果存在）
    config_path = os.path.join(config.checkpoint_dir, "config.json")
    if os.path.exists(config_path):
        import shutil
        shutil.copy(config_path, os.path.join(save_dir, "config.json"))

    print(f"  ✓ Saved checkpoint to: {save_dir}")

    return save_dir


def run_finetune(config: FineTuneConfig) -> str:
    """
    Fine-tune a compressed model to recover accuracy.

    Returns:
        Path to fine-tuned checkpoint
    """
    print("=" * 60)
    print("FlashSVD Fine-tuning Pipeline")
    print("=" * 60)
    print(f"Checkpoint: {config.checkpoint_dir}")
    print(f"Task: {config.task}")
    print(f"Epochs: {config.epochs}")
    print(f"Learning rate: {config.learning_rate}")
    print(f"Batch size: {config.batch_size}")
    print(f"Device: {config.device}")
    print("=" * 60)

    # 设置随机种子
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    # 1. 加载压缩模型
    print("\n[1/6] Loading compressed model...")
    model = load_compressed(config.checkpoint_dir, device=config.device)

    # 读取 compression info
    with open(os.path.join(config.checkpoint_dir, "compression_info.json")) as f:
        compression_info = json.load(f)

    base_model = compression_info["base_model"]
    tokenizer = AutoTokenizer.from_pretrained(base_model)

    # 2. 冻结参数（如果需要）
    print("\n[2/6] Configuring trainable parameters...")
    freeze_parameters(model, config)

    # 3. 加载数据
    print("\n[3/6] Loading and tokenizing data...")
    train_dataset, val_dataset = load_and_prepare_data(config, tokenizer)
    print(f"  Training samples: {len(train_dataset)}")
    print(f"  Validation samples: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.dataloader_num_workers,
        pin_memory=config.dataloader_pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.eval_batch_size,
        shuffle=False,
        num_workers=config.dataloader_num_workers,
        pin_memory=config.dataloader_pin_memory,
    )

    # 4. 设置优化器和调度器
    print("\n[4/6] Setting up optimizer and scheduler...")
    optimizer = get_optimizer(model, config)

    num_training_steps = len(train_loader) * config.epochs // config.gradient_accumulation_steps
    scheduler = get_scheduler(optimizer, config, num_training_steps)

    print(f"  Optimizer: {config.optimizer}")
    print(f"  LR scheduler: {config.lr_scheduler}")
    print(f"  Total training steps: {num_training_steps}")
    warmup_steps = config.warmup_steps if config.warmup_steps else int(num_training_steps * config.warmup_ratio)
    print(f"  Warmup steps: {warmup_steps}")

    # 5. 设置 metric 和早停
    print("\n[5/6] Setting up evaluation...")
    if config.task == "stsb":
        metric = load_metric("pearsonr")
        metric_key = "pearsonr"
    else:
        metric = load_metric("accuracy")
        metric_key = "accuracy"

    early_stopping = None
    if config.early_stopping:
        early_stopping = EarlyStopping(
            patience=config.early_stopping_patience,
            min_delta=config.early_stopping_threshold
        )
        print(f"  Early stopping enabled (patience={config.early_stopping_patience})")

    # 6. TensorBoard
    writer = None
    if config.use_tensorboard:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(config.tensorboard_dir)
        print(f"  TensorBoard logging to: {config.tensorboard_dir}")

    # 7. 训练循环
    print("\n[6/6] Starting fine-tuning...")
    model.train()

    global_step = 0
    best_metric = 0.0
    best_checkpoint_path = None

    for epoch in range(config.epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{config.epochs}")
        print(f"{'='*60}")

        epoch_loss = 0.0
        progress_bar = tqdm(train_loader, desc="Training")

        for step, batch in enumerate(progress_bar):
            batch = {k: v.to(config.device) for k, v in batch.items()}

            # Forward
            outputs = model(**batch)
            loss = outputs.loss / config.gradient_accumulation_steps

            # Backward
            loss.backward()

            epoch_loss += loss.item()

            # 梯度累积
            if (step + 1) % config.gradient_accumulation_steps == 0:
                # 梯度裁剪
                if config.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                # Logging
                if global_step % config.logging_steps == 0:
                    current_lr = scheduler.get_last_lr()[0] if hasattr(scheduler, 'get_last_lr') else config.learning_rate
                    progress_bar.set_postfix({
                        'loss': f'{loss.item():.4f}',
                        'lr': f'{current_lr:.2e}'
                    })

                    if writer:
                        writer.add_scalar('train/loss', loss.item(), global_step)
                        writer.add_scalar('train/lr', current_lr, global_step)

                # Evaluation
                if global_step % config.eval_steps == 0:
                    print(f"\n  Evaluating at step {global_step}...")
                    eval_result = evaluate_model(model, val_loader, metric, config.device)
                    eval_metric = eval_result[metric_key]

                    print(f"  Validation {metric_key}: {eval_metric:.4f}")

                    if writer:
                        writer.add_scalar(f'eval/{metric_key}', eval_metric, global_step)

                    # 保存最佳模型
                    if eval_metric > best_metric:
                        best_metric = eval_metric
                        best_checkpoint_path = save_checkpoint(model, config, epoch, global_step, best=True)

                    # Early stopping
                    if early_stopping and early_stopping(eval_metric):
                        print(f"\n  Early stopping triggered at step {global_step}")
                        break

                # 定期保存
                if global_step % config.save_steps == 0:
                    save_checkpoint(model, config, epoch, global_step, best=False)

        # Epoch 结束
        avg_epoch_loss = epoch_loss / len(train_loader)
        print(f"\nEpoch {epoch + 1} completed. Average loss: {avg_epoch_loss:.4f}")

        if early_stopping and early_stopping.should_stop:
            break

    # 8. 最终评估和保存
    print("\n" + "=" * 60)
    print("Final evaluation...")
    print("=" * 60)

    final_result = evaluate_model(model, val_loader, metric, config.device)
    final_metric = final_result[metric_key]

    print(f"\nFinal {metric_key}: {final_metric:.4f}")
    print(f"Best {metric_key}: {best_metric:.4f}")

    # 保存最终模型
    final_checkpoint_path = save_checkpoint(model, config, config.epochs - 1, global_step, best=False)

    # 关闭 TensorBoard
    if writer:
        writer.close()

    print("\n" + "=" * 60)
    print("Fine-tuning complete!")
    print(f"Best checkpoint: {best_checkpoint_path}")
    print(f"Final checkpoint: {final_checkpoint_path}")
    print("=" * 60)

    return best_checkpoint_path or final_checkpoint_path
