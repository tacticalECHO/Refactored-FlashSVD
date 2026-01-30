"""
FlashSVD Unified CLI

Main entry point for flashsvd command with subcommands: compress, eval, info.
"""

import sys
import argparse
from typing import List, Optional


def create_parser():
    """Create the unified CLI parser with subcommands."""
    parser = argparse.ArgumentParser(
        prog="flashsvd",
        description="FlashSVD: Memory-Efficient Inference for Low-Rank Models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Subcommands:
  compress    Compress a model using SVD
  eval        Evaluate a compressed model on GLUE tasks
  finetune    Fine-tune compressed model to recover accuracy
  info        Display checkpoint metadata

Examples:
  # Compress a model
  flashsvd compress --model bert-base-uncased --task sst2 --rank 64

  # Fine-tune compressed model
  flashsvd finetune --checkpoint ./compressed_models/bert_r64 --task sst2 --epochs 3

  # Evaluate compressed model
  flashsvd eval --checkpoint ./compressed_models/bert_r64 --task sst2

  # Show checkpoint info
  flashsvd info ./compressed_models/bert_r64

For detailed help on each subcommand:
  flashsvd compress --help
  flashsvd finetune --help
  flashsvd eval --help
  flashsvd info --help

Documentation: https://github.com/Zishan-Shao/FlashSVD
        """
    )

    parser.add_argument("--version", action="version", version="%(prog)s 0.1.0")

    # Create subparsers
    subparsers = parser.add_subparsers(
        title="subcommands",
        dest="command",
        help="Available commands",
        required=True
    )

    # Compress subcommand
    compress_parser = subparsers.add_parser(
        "compress",
        help="Compress a model using SVD",
        description="Compress a pretrained model using standard SVD",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  flashsvd compress --model bert-base-uncased --task sst2 --rank 64
  flashsvd compress --model bert-base-uncased --task sst2 --rank-attn 40 --rank-ffn 240
        """
    )

    # Model settings
    compress_parser.add_argument("--model", type=str, default="bert-base-uncased",
                                 help="Model name or path (default: bert-base-uncased)")
    compress_parser.add_argument("--task", type=str, default="sst2",
                                 choices=["cola", "sst2", "mrpc", "qqp", "mnli", "qnli", "rte", "stsb"],
                                 help="GLUE task name (default: sst2)")
    compress_parser.add_argument("--checkpoint", type=str, default=None,
                                 help="Path to finetuned checkpoint (optional)")
    compress_parser.add_argument("--method", type=str, default="standard",
                                 choices=["standard", "fw", "fwsvd", "ada", "adasvd", "whiten", "drone"],
                                 help="Compression method: standard, fw/fwsvd (Fisher-Weighted), ada/adasvd (Adaptive), whiten/drone (Data-Aware)")

    # Ranks
    compress_parser.add_argument("--rank", type=int, default=None,
                                 help="Unified rank for all components")
    compress_parser.add_argument("--rank-attn", type=int, default=64,
                                 help="Rank for attention Q/K/V (default: 64)")
    compress_parser.add_argument("--rank-ffn", type=int, default=256,
                                 help="Rank for FFN layers (default: 256)")
    compress_parser.add_argument("--rank-wo", type=int, default=256,
                                 help="Rank for attention output projection (default: 256)")
    compress_parser.add_argument("--ranks-json", type=str, default=None,
                                 help="Path to ranks.json file (for AdaSVD method)")

    # I/O
    compress_parser.add_argument("--output-dir", type=str, default="./compressed_models",
                                 help="Output directory (default: ./compressed_models)")

    # Hardware
    compress_parser.add_argument("--device", type=str, default="cuda",
                                 choices=["cuda", "cpu"],
                                 help="Device to use: cuda or cpu (default: cuda)")

    # Eval subcommand
    eval_parser = subparsers.add_parser(
        "eval",
        help="Evaluate a compressed model",
        description="Evaluate compressed model on GLUE tasks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  flashsvd eval --checkpoint ./compressed_models/bert_r64 --task sst2
  flashsvd eval --checkpoint ./compressed_models/bert_r64 --task sst2 --batch-size 16
        """
    )

    eval_parser.add_argument("--checkpoint", type=str, required=True,
                            help="Path to compressed model checkpoint directory")
    eval_parser.add_argument("--task", type=str, default="sst2",
                            choices=["cola", "sst2", "mrpc", "qqp", "mnli", "qnli", "rte", "stsb"],
                            help="GLUE task name (default: sst2)")
    eval_parser.add_argument("--batch-size", type=int, default=32,
                            help="Batch size for evaluation (default: 32)")
    eval_parser.add_argument("--seq-len", type=int, default=128,
                            help="Max sequence length (default: 128)")
    eval_parser.add_argument("--max-eval-samples", type=int, default=None,
                            help="Max samples to evaluate (default: None = all)")
    eval_parser.add_argument("--device", type=str, default="cuda",
                            choices=["cuda", "cpu"],
                            help="Device to use: cuda or cpu (default: cuda)")
    eval_parser.add_argument("--output", type=str, default="eval_results.json",
                            help="Output JSON file (default: eval_results.json)")

    # Info subcommand
    info_parser = subparsers.add_parser(
        "info",
        help="Display checkpoint information",
        description="Display metadata and file information about a compressed checkpoint",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  flashsvd info ./compressed_models/bert-base-uncased_standard_r64
        """
    )

    info_parser.add_argument("checkpoint", type=str,
                            help="Path to compressed model checkpoint directory")

    # Finetune subcommand
    finetune_parser = subparsers.add_parser(
        "finetune",
        help="Fine-tune compressed model to recover accuracy",
        description="Fine-tune a compressed model on GLUE task to recover accuracy after SVD compression",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic fine-tuning
  flashsvd finetune --checkpoint ./compressed_models/bert_r64 --task sst2 --epochs 3

  # With custom learning rate
  flashsvd finetune --checkpoint ./compressed_models/bert_r64 --task sst2 --lr 2e-5 --epochs 5

  # Quick test
  flashsvd finetune --checkpoint ./compressed_models/bert_r64 --task sst2 --epochs 1 --max-train-samples 1000

  # With early stopping
  flashsvd finetune --checkpoint ./compressed_models/bert_r64 --task sst2 --early-stopping --patience 3
        """
    )

    # Required
    finetune_parser.add_argument("--checkpoint", required=True,
                                help="Path to compressed model checkpoint")
    finetune_parser.add_argument("--task", required=True,
                                choices=["cola", "sst2", "mrpc", "qqp", "mnli", "qnli", "rte", "stsb"],
                                help="GLUE task name")

    # Training hyperparameters
    finetune_parser.add_argument("--epochs", type=int, default=3,
                                help="Number of training epochs (default: 3)")
    finetune_parser.add_argument("--lr", "--learning-rate", type=float, default=3e-5, dest="learning_rate",
                                help="Learning rate (default: 3e-5)")
    finetune_parser.add_argument("--batch-size", type=int, default=32,
                                help="Training batch size (default: 32)")
    finetune_parser.add_argument("--eval-batch-size", type=int, default=64,
                                help="Evaluation batch size (default: 64)")

    # Optimizer
    finetune_parser.add_argument("--optimizer", choices=["adamw", "adam", "sgd"], default="adamw",
                                help="Optimizer type (default: adamw)")
    finetune_parser.add_argument("--weight-decay", type=float, default=0.01,
                                help="Weight decay (default: 0.01)")
    finetune_parser.add_argument("--max-grad-norm", type=float, default=1.0,
                                help="Max gradient norm for clipping (default: 1.0)")

    # LR scheduler
    finetune_parser.add_argument("--lr-scheduler", choices=["linear", "cosine", "constant", "polynomial"],
                                default="linear",
                                help="Learning rate scheduler (default: linear)")
    finetune_parser.add_argument("--warmup-ratio", type=float, default=0.1,
                                help="Warmup ratio (default: 0.1)")
    finetune_parser.add_argument("--warmup-steps", type=int,
                                help="Warmup steps (overrides warmup-ratio)")

    # Training strategy
    finetune_parser.add_argument("--logging-steps", type=int, default=50,
                                help="Log every N steps (default: 50)")
    finetune_parser.add_argument("--eval-steps", type=int, default=500,
                                help="Evaluate every N steps (default: 500)")
    finetune_parser.add_argument("--save-steps", type=int, default=500,
                                help="Save checkpoint every N steps (default: 500)")

    # Early stopping
    finetune_parser.add_argument("--early-stopping", action="store_true",
                                help="Enable early stopping")
    finetune_parser.add_argument("--patience", type=int, default=3,
                                help="Early stopping patience (default: 3)")

    # Freezing strategy
    finetune_parser.add_argument("--freeze-embeddings", action="store_true",
                                help="Freeze embedding layers")
    finetune_parser.add_argument("--freeze-attention", action="store_true",
                                help="Freeze attention layers")
    finetune_parser.add_argument("--freeze-ffn", action="store_true",
                                help="Freeze FFN layers")

    # Data
    finetune_parser.add_argument("--max-train-samples", type=int,
                                help="Limit training samples (for quick testing)")
    finetune_parser.add_argument("--max-eval-samples", type=int,
                                help="Limit evaluation samples")
    finetune_parser.add_argument("--max-seq-length", type=int, default=128,
                                help="Maximum sequence length (default: 128)")

    # Output
    finetune_parser.add_argument("--output", "--output-dir", dest="output_dir",
                                help="Output directory (default: overwrite checkpoint)")
    finetune_parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"],
                                help="Device (default: cuda)")
    finetune_parser.add_argument("--seed", type=int, default=42,
                                help="Random seed (default: 42)")

    # Logging
    finetune_parser.add_argument("--use-tensorboard", action="store_true",
                                help="Enable TensorBoard logging")
    finetune_parser.add_argument("--tensorboard-dir",
                                help="TensorBoard log directory")

    return parser


def main(argv: Optional[List[str]] = None):
    """
    Main entry point for unified flashsvd CLI.

    Args:
        argv: Command line arguments (default: sys.argv)
    """
    parser = create_parser()
    args = parser.parse_args(argv)

    # Dispatch to appropriate subcommand
    if args.command == "compress":
        from flashsvd.compress import CompressConfig, run_compress

        # Handle unified rank
        # Following experimental code convention: ffn_rank = 6x attn_rank, wo_rank = 6x attn_rank
        # (Original experiments use 40/240/240, where 240 = 6 * 40)
        if args.rank is not None:
            rank_attn = args.rank
            rank_ffn = args.rank * 6
            rank_wo = args.rank * 6
            print(f"Using unified rank: attn={rank_attn}, ffn={rank_ffn}, wo={rank_wo} (6x scaling)")
        else:
            rank_attn = args.rank_attn
            rank_ffn = args.rank_ffn
            rank_wo = args.rank_wo

        config = CompressConfig(
            model_name=args.model,
            task=args.task,
            method=args.method,
            rank_attn=rank_attn,
            rank_ffn=rank_ffn,
            rank_wo=rank_wo,
            checkpoint_dir=args.checkpoint,
            output_dir=args.output_dir,
            device=args.device,
            ranks_json=args.ranks_json,
        )

        try:
            output_dir = run_compress(config)
            print(f"\n🎉 Success! Compressed model saved to: {output_dir}")
            print(f"\nNext steps:")
            print(f"  • Show info:  flashsvd info {output_dir}")
            print(f"  • Evaluate:   flashsvd eval --checkpoint {output_dir} --task {args.task}")
            return 0
        except Exception as e:
            print(f"\n❌ Compression failed: {e}")
            import traceback
            traceback.print_exc()
            return 1

    elif args.command == "eval":
        from flashsvd.evaluate import EvalConfig, run_eval

        config = EvalConfig(
            checkpoint_dir=args.checkpoint,
            task=args.task,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            max_eval_samples=args.max_eval_samples,
            device=args.device,
            output=args.output,
        )

        try:
            results = run_eval(config)

            # Print summary
            print(f"\n{'=' * 60}")
            print("Results")
            print("=" * 60)

            metric_name = results["metric_name"]
            metric_value = results["metric_value"]
            peak_mem = results["peak_memory_mib"]
            latency = results["latency_ms"]

            if peak_mem is not None:
                print(f"{config.task} | {metric_name}={metric_value:.4f} | "
                      f"peak={peak_mem:.1f} MiB | latency={latency:.2f} ms/batch")
            else:
                print(f"{config.task} | {metric_name}={metric_value:.4f} | "
                      f"peak=N/A (CPU) | latency={latency:.2f} ms/batch")

            # Save results
            import json
            with open(config.output, "w") as f:
                json.dump(results, f, indent=2)

            print(f"\n✅ Results saved to: {config.output}")
            return 0

        except Exception as e:
            print(f"\n❌ Evaluation failed: {e}")
            import traceback
            traceback.print_exc()
            return 1

    elif args.command == "info":
        from flashsvd.info import show_checkpoint_info

        try:
            show_checkpoint_info(args.checkpoint)
            return 0
        except Exception as e:
            print(f"❌ Error: {e}")
            return 1

    elif args.command == "finetune":
        from flashsvd.finetune import FineTuneConfig, run_finetune

        config = FineTuneConfig(
            checkpoint_dir=args.checkpoint,
            task=args.task,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            eval_batch_size=args.eval_batch_size,
            optimizer=args.optimizer,
            weight_decay=args.weight_decay,
            max_grad_norm=args.max_grad_norm,
            lr_scheduler=args.lr_scheduler,
            warmup_ratio=args.warmup_ratio,
            warmup_steps=args.warmup_steps,
            logging_steps=args.logging_steps,
            eval_steps=args.eval_steps,
            save_steps=args.save_steps,
            early_stopping=args.early_stopping,
            early_stopping_patience=args.patience,
            freeze_embeddings=args.freeze_embeddings,
            freeze_attention=args.freeze_attention,
            freeze_ffn=args.freeze_ffn,
            max_train_samples=args.max_train_samples,
            max_eval_samples=args.max_eval_samples,
            max_seq_length=args.max_seq_length,
            output_dir=args.output_dir,
            device=args.device,
            seed=args.seed,
            use_tensorboard=args.use_tensorboard,
            tensorboard_dir=args.tensorboard_dir,
        )

        try:
            output_path = run_finetune(config)
            print(f"\n✅ Fine-tuned model saved to: {output_path}")
            print(f"\nNext steps:")
            print(f"  • Show info:  flashsvd info {output_path}")
            print(f"  • Evaluate:   flashsvd eval --checkpoint {output_path} --task {args.task}")
            return 0
        except Exception as e:
            print(f"\n❌ Fine-tuning failed: {e}")
            import traceback
            traceback.print_exc()
            return 1

    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
