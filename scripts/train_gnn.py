"""
Train a GNN-based QEC decoder.

Examples
--------
    # Train from config (recommended)
    uv run scripts/train_gnn.py -c configs/train.yaml

    # Override case and epochs
    uv run scripts/train_gnn.py -c configs/train.yaml --case mwpm_teacher --epochs 50

    # Pure CLI (no config file)
    uv run scripts/train_gnn.py --case logical_head --hidden-dim 64 --epochs 50

    # Resume from checkpoint
    uv run scripts/train_gnn.py -c configs/train.yaml --resume runs/logical_head/best.pt
"""

import argparse
import logging
import sys
from dataclasses import asdict
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
        "-c", "--config",
        type=Path,
        default=None,
        help="YAML config file (CLI args override config values)",
    )
    parser.add_argument(
        "--case",
        type=str,
        choices=["logical_head", "mwpm_teacher", "hybrid"],
        default=None,
    )
    parser.add_argument("--datasets-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)

    # Model
    parser.add_argument("--hidden-dim", type=int, default=None)
    parser.add_argument("--num-layers", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)

    # Optimisation
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)

    # Data loading
    parser.add_argument("--num-workers", type=int, default=None)

    # Edge-case specific
    parser.add_argument(
        "--edge-pos-weight", type=float, default=None,
        help="pos_weight for edge BCE (auto-estimated if omitted)",
    )

    # Misc
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args(argv)
    setup_logging(verbose=args.verbose)

    # Start from YAML if provided, else from defaults
    if args.config is not None:
        cfg = TrainConfig.from_yaml(args.config)
    else:
        cfg = TrainConfig()

    # CLI overrides (only non-None values)
    overrides = {
        "case": args.case,
        "datasets_dir": args.datasets_dir,
        "output_dir": args.output_dir,
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "edge_pos_weight": args.edge_pos_weight,
        "seed": args.seed,
        "resume": args.resume,
    }

    cfg_dict = asdict(cfg)
    for key, value in overrides.items():
        if value is not None:
            cfg_dict[key] = value

    # Path coercion
    for key in ("datasets_dir", "output_dir"):
        cfg_dict[key] = Path(cfg_dict[key])
    if cfg_dict["resume"] is not None:
        cfg_dict["resume"] = Path(cfg_dict["resume"])

    return TrainConfig(**cfg_dict)


def main(argv: Sequence[str] | None = None) -> None:
    """Entry point for GNN training."""
    cfg = parse_args(argv)
    train(cfg)


if __name__ == "__main__":
    main()
