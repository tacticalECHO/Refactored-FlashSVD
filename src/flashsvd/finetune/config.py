"""
Fine-tuning configuration for compressed models.
"""

from dataclasses import dataclass, field
from typing import Optional, Literal
import json


@dataclass
class FineTuneConfig:
    """Fine-tuning configuration for compressed models."""

    # ─── Required ───────────────────────────────────────────────────
    checkpoint_dir: str          # 压缩模型路径
    task: str                    # GLUE任务: cola, sst2, mrpc, qqp, mnli, qnli, rte, stsb

    # ─── Training Hyperparameters ──────────────────────────────────
    epochs: int = 3              # 训练轮数
    learning_rate: float = 3e-5  # 学习率
    batch_size: int = 32         # 训练批次大小
    eval_batch_size: int = 64    # 评估批次大小

    # ─── Optimizer Settings ────────────────────────────────────────
    optimizer: Literal["adamw", "adam", "sgd"] = "adamw"
    weight_decay: float = 0.01   # L2 正则化
    adam_beta1: float = 0.9      # AdamW beta1
    adam_beta2: float = 0.999    # AdamW beta2
    adam_epsilon: float = 1e-8   # AdamW epsilon
    max_grad_norm: float = 1.0   # 梯度裁剪阈值

    # ─── Learning Rate Scheduler ───────────────────────────────────
    lr_scheduler: Literal["linear", "cosine", "constant", "polynomial"] = "linear"
    warmup_ratio: float = 0.1    # 预热步数比例 (0.1 = 10% of total steps)
    warmup_steps: Optional[int] = None  # 或直接指定预热步数（优先级高于ratio）

    # ─── Training Strategy ─────────────────────────────────────────
    logging_steps: int = 50      # 每N步打印日志
    eval_steps: int = 500        # 每N步评估一次
    save_steps: int = 500        # 每N步保存checkpoint
    save_total_limit: int = 2    # 最多保存N个checkpoint

    # ─── Early Stopping ────────────────────────────────────────────
    early_stopping: bool = False          # 是否启用早停
    early_stopping_patience: int = 3      # 容忍N次评估不提升
    early_stopping_threshold: float = 0.0 # 最小提升阈值

    # ─── Model Fine-tuning Strategy ────────────────────────────────
    freeze_embeddings: bool = False       # 是否冻结 embeddings
    freeze_attention: bool = False        # 是否冻结 attention 层
    freeze_ffn: bool = False              # 是否冻结 FFN 层

    # ─── Data ──────────────────────────────────────────────────────
    max_train_samples: Optional[int] = None  # 限制训练样本数（测试用）
    max_eval_samples: Optional[int] = None   # 限制评估样本数
    max_seq_length: int = 128                # 最大序列长度

    # ─── Output & Device ───────────────────────────────────────────
    output_dir: Optional[str] = None      # 输出目录（None=覆盖原checkpoint）
    overwrite_output: bool = False        # 是否覆盖已存在的输出
    device: str = "cuda"                  # 设备
    seed: int = 42                        # 随机种子

    # ─── Logging & Monitoring ──────────────────────────────────────
    use_tensorboard: bool = False         # 是否使用 TensorBoard
    tensorboard_dir: Optional[str] = None # TensorBoard 日志目录

    # ─── Advanced ──────────────────────────────────────────────────
    fp16: bool = False                    # 是否使用混合精度训练
    gradient_accumulation_steps: int = 1  # 梯度累积步数
    dataloader_num_workers: int = 4       # DataLoader 工作进程数
    dataloader_pin_memory: bool = True    # 是否 pin memory

    def __post_init__(self):
        """Validation and computed fields."""
        # 自动设置输出目录（按架构和方法组织）
        if self.output_dir is None:
            self.output_dir = self._generate_output_dir()

        # 验证任务名称
        valid_tasks = ["cola", "sst2", "mrpc", "qqp", "mnli", "qnli", "rte", "stsb"]
        if self.task not in valid_tasks:
            raise ValueError(f"Invalid task: {self.task}. Must be one of {valid_tasks}")

        # TensorBoard 目录
        if self.use_tensorboard and self.tensorboard_dir is None:
            self.tensorboard_dir = f"{self.output_dir}/tensorboard"

    def _generate_output_dir(self) -> str:
        """
        Generate organized output directory structure:
        models/finetuned/{arch}/{method}/{checkpoint_name}/

        Examples:
            models/finetuned/bert/fwsvd/bert-base-uncased-SST-2_fwsvd_r64/
            models/finetuned/modernbert/whiten/modernbert-base_whiten_r64/
        """
        import os
        from pathlib import Path

        # 读取压缩模型的元数据
        compression_info_path = os.path.join(self.checkpoint_dir, "compression_info.json")

        if os.path.exists(compression_info_path):
            with open(compression_info_path) as f:
                compression_info = json.load(f)

            # 提取架构和方法
            arch = compression_info.get("arch", "unknown")
            method = compression_info.get("method", "unknown")
        else:
            # 如果没有compression_info.json，尝试从路径推断
            arch = "unknown"
            method = "unknown"

            # 从checkpoint路径推断架构
            checkpoint_path = self.checkpoint_dir.lower()
            if "bert" in checkpoint_path and "modernbert" not in checkpoint_path and "roberta" not in checkpoint_path:
                arch = "bert"
            elif "modernbert" in checkpoint_path:
                arch = "modernbert"
            elif "roberta" in checkpoint_path:
                arch = "roberta"
            elif "llama" in checkpoint_path:
                arch = "llama"
            elif "gpt2" in checkpoint_path:
                arch = "gpt2"

            # 从checkpoint路径推断方法
            if "fwsvd" in checkpoint_path or "fw" in checkpoint_path:
                method = "fwsvd"
            elif "whiten" in checkpoint_path or "drone" in checkpoint_path:
                method = "whiten"
            elif "ada" in checkpoint_path:
                method = "adasvd"
            elif "asvd" in checkpoint_path:
                method = "asvd"
            elif "standard" in checkpoint_path:
                method = "standard"

        # 获取checkpoint目录名称
        checkpoint_name = Path(self.checkpoint_dir).name

        # 生成组织化的路径
        output_dir = os.path.join(
            "models",
            "finetuned",
            arch,
            method,
            checkpoint_name
        )

        return output_dir

    def to_dict(self) -> dict:
        """Convert to dictionary for saving."""
        return {
            k: v for k, v in self.__dict__.items()
            if not k.startswith('_')
        }

    def save(self, path: str):
        """Save configuration to JSON."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
