"""
Train a GNN-based QEC decoder.

Examples
--------
    # Train logical head for 50 epochs
    uv run scripts/train_gnn.py --case logical_head --epochs 50

    # MWPM teacher with custom hidden dim
    uv run scripts/train_gnn.py --case mwpm_teacher --hidden-dim 64

    # Resume from checkpoint
    uv run scripts/train_gnn.py --case logical_head --resume runs/logical_head/best.pt
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Sequence

from gnn.train import TrainConfig, train


def setup_logging(verbose: bool = False) -> None:
    """
    Configure root logging.

    Parameters
    ----------
    verbose : bool
        Enable ``DEBUG`` level if True, otherwise ``INFO``.
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )


def parse_args(argv: Sequence[str] | None = None) -> TrainConfig:
    """
    Parse command-line arguments into a :class:`TrainConfig`.

    Parameters
    ----------
    argv : sequence of str or None
        Arguments to parse; defaults to ``sys.argv[1:]``.

    Returns
    -------
    TrainConfig
        Parsed training configuration.
    """
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--case",
        type=str,
        choices=["logical_head", "mwpm_teacher", "hybrid"],
        default="logical_head",
        help="Training case (default: logical_head)",
    )
    parser.add_argument(
        "--datasets-dir",
        type=Path,
        default=Path("data/datasets"),
        help="Root directory of packaged datasets",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs"),
        help="Output directory for checkpoints and logs",
    )

    # Model
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--dropout", type=float, default=0.1)

    # Optimisation
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)

    # Data loading
    parser.add_argument("--num-workers", type=int, default=4)

    # Edge-case specific
    parser.add_argument(
        "--edge-pos-weight",
        type=float,
        default=None,
        help="pos_weight for edge BCE (auto-estimated if omitted)",
    )

    # Misc
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--resume", type=Path, default=None, help="Resume from checkpoint"
    )
    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args(argv)
    setup_logging(verbose=args.verbose)

    return TrainConfig(
        case=args.case,
        datasets_dir=args.datasets_dir,
        output_dir=args.output_dir,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        lr=args.lr,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        edge_pos_weight=args.edge_pos_weight,
        seed=args.seed,
        resume=args.resume,
    )


def main(argv: Sequence[str] | None = None) -> None:
    """Entry point for GNN training."""
    cfg = parse_args(argv)
    train(cfg)


if __name__ == "__main__":
    main()
