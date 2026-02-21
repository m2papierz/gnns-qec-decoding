"""Train a GNN-based QEC decoder.

Examples
--------
    # Train from config (recommended)
    uv run scripts/train_gnn.py -c configs/train.yaml

    # Override case and epochs
    uv run scripts/train_gnn.py -c configs/train.yaml --case mwpm_teacher --epochs 50

    # Resume from checkpoint
    uv run scripts/train_gnn.py -c configs/train.yaml --resume outputs/runs/logical_head/best.pt
"""

import argparse
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

from cli import setup_logging
from gnn.trainer import TrainConfig, Trainer


def parse_args(argv: Sequence[str] | None = None) -> TrainConfig:
    """
    Parse command-line arguments into a :class:`TrainConfig`.

    CLI args override values loaded from YAML config.
    """
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "-c",
        "--config",
        type=Path,
        default=Path("configs/train.yaml"),
        help="YAML config file (CLI args override config values)",
    )
    parser.add_argument(
        "--case",
        type=str,
        default=None,
        choices=["logical_head", "mwpm_teacher", "hybrid"],
    )
    parser.add_argument("--datasets-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--hidden-dim", type=int, default=None)
    parser.add_argument("--num-layers", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--edge-pos-weight", type=float, default=None)
    parser.add_argument("--max-grad-norm", type=float, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--val-every", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args(argv)
    setup_logging(verbose=args.verbose)

    if args.config is not None and args.config.is_file():
        cfg = TrainConfig.from_yaml(args.config)
    elif args.config is not None and not args.config.is_file():
        logging.getLogger(__name__).warning(
            "Config file %s not found, using defaults",
            args.config,
        )
        cfg = TrainConfig()
    else:
        cfg = TrainConfig()

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
        "max_grad_norm": args.max_grad_norm,
        "patience": args.patience,
        "val_every": args.val_every,
        "seed": args.seed,
        "resume": args.resume,
        "max_samples": args.max_samples,
    }

    cfg_dict = asdict(cfg)
    for key, value in overrides.items():
        if value is not None:
            cfg_dict[key] = value

    for key in ("datasets_dir", "output_dir"):
        cfg_dict[key] = Path(cfg_dict[key])
    if cfg_dict["resume"] is not None:
        cfg_dict["resume"] = Path(cfg_dict["resume"])

    return TrainConfig(**cfg_dict)


def main(argv: Sequence[str] | None = None) -> None:
    """Entry point for GNN training."""
    cfg = parse_args(argv)
    Trainer(cfg).fit()


if __name__ == "__main__":
    main()
