"""
FlashSVD Info Command

Display metadata and information about compressed model checkpoints.
"""

import os
import json
from pathlib import Path


def show_checkpoint_info(checkpoint_dir: str):
    """
    Display checkpoint information in a pretty format.

    Args:
        checkpoint_dir: Path to compressed model checkpoint

    Raises:
        FileNotFoundError: If checkpoint or compression_info.json not found
    """
    checkpoint_dir = Path(checkpoint_dir).resolve()

    # Validate checkpoint exists
    if not checkpoint_dir.exists():
        raise FileNotFoundError(
            f"Checkpoint directory not found: {checkpoint_dir}\n"
            f"Make sure the path is correct."
        )

    if not checkpoint_dir.is_dir():
        raise NotADirectoryError(
            f"Path is not a directory: {checkpoint_dir}\n"
            f"Please provide a checkpoint directory path."
        )

    # Check for compression_info.json
    compression_info_path = checkpoint_dir / "compression_info.json"
    if not compression_info_path.exists():
        raise FileNotFoundError(
            f"compression_info.json not found in {checkpoint_dir}\n"
            f"This may not be a FlashSVD compressed model.\n"
            f"Expected file: {compression_info_path}"
        )

    # Load compression info
    with open(compression_info_path, "r") as f:
        info = json.load(f)

    # Check for required files
    files = {
        "config.json": checkpoint_dir / "config.json",
        "flashsvd_state_dict.pt": checkpoint_dir / "flashsvd_state_dict.pt",
        "model weights": checkpoint_dir / "model.safetensors" if (checkpoint_dir / "model.safetensors").exists()
                         else checkpoint_dir / "pytorch_model.bin",
    }

    # Print header
    print("=" * 60)
    print(f"FlashSVD Checkpoint Info: {checkpoint_dir.name}")
    print("=" * 60)

    # Compression details
    print("\n📦 Compression Details")
    print(f"  Method:      {info.get('method', 'N/A')}")
    print(f"  Base Model:  {info.get('base_model', 'N/A')}")
    print(f"  Task:        {info.get('task', 'N/A')}")

    # Ranks
    ranks = info.get('ranks', {})
    if ranks:
        print(f"  Ranks:")
        print(f"    - Attention: {ranks.get('attn', 'N/A')}")
        print(f"    - FFN:       {ranks.get('ffn', 'N/A')}")
        print(f"    - Output:    {ranks.get('wo', 'N/A')}")

    # Metadata
    print(f"\n📅 Metadata")
    print(f"  Created:         {info.get('timestamp', 'N/A')}")
    print(f"  FlashSVD Ver:    {info.get('flashsvd_version', 'N/A')}")
    if info.get('git_commit'):
        commit = info['git_commit'][:8]
        print(f"  Git Commit:      {commit}")

    # Files
    print(f"\n📁 Files")
    for name, path in files.items():
        if path.exists():
            size_mb = path.stat().st_size / (1024 ** 2)
            print(f"  ✓ {name:<25} ({size_mb:.1f} MB)")
        else:
            print(f"  ✗ {name:<25} (missing)")

    # Total size
    total_size = sum(
        path.stat().st_size for path in checkpoint_dir.glob("*") if path.is_file()
    )
    total_mb = total_size / (1024 ** 2)
    print(f"\n  Total Size: {total_mb:.1f} MB")

    # Usage instructions
    print(f"\n💡 Usage")
    print(f"  Evaluate:  flashsvd eval --checkpoint {checkpoint_dir}")
    print(f"  Load in Python:")
    print(f"    from flashsvd.io import load_compressed")
    print(f"    model = load_compressed('{checkpoint_dir}')")

    print("=" * 60)


def main():
    """CLI entry point for flashsvd info (standalone, for compatibility)."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Display FlashSVD checkpoint information",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  flashsvd info ./compressed_models/bert-base-uncased_standard_r64
        """
    )

    parser.add_argument("checkpoint", type=str,
                        help="Path to compressed model checkpoint directory")

    args = parser.parse_args()

    try:
        show_checkpoint_info(args.checkpoint)
        return 0
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
