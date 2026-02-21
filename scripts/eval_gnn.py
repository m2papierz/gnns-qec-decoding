"""
Evaluate a trained GNN decoder and compare with MWPM baseline.

Examples
--------
    # Evaluate on test split
    uv run scripts/eval_gnn.py --checkpoint outputs/runs/logical_head/best.pt

    # Compare with MWPM baseline
    uv run scripts/eval_gnn.py --checkpoint outputs/runs/logical_head/best.pt \\
        --baseline outputs/results/mwpm_baseline.json

    # Save JSON report to custom path
    uv run scripts/eval_gnn.py --checkpoint outputs/runs/logical_head/best.pt \\
        -o outputs/results/gnn_logical.json
"""

import argparse
from pathlib import Path
from typing import Sequence

from cli import setup_logging
from gnn.evaluator import Evaluator
from gnn.metrics import print_report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to model checkpoint (e.g. outputs/runs/logical_head/best.pt)",
    )
    parser.add_argument(
        "--datasets-dir",
        type=Path,
        default=Path("data/datasets"),
        help="Root directory of packaged datasets",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Data split to evaluate (default: test)",
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=None,
        help="MWPM baseline JSON report for comparison",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Shots per inference batch (default: 256)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Save JSON report to this path (default: next to checkpoint)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable DEBUG logging",
    )

    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Entry point for GNN evaluation."""
    args = parse_args(argv)
    setup_logging(verbose=args.verbose)

    evaluator = Evaluator(
        checkpoint=args.checkpoint,
        datasets_dir=args.datasets_dir,
        split=args.split,
        baseline_path=args.baseline,
        batch_size=args.batch_size,
    )
    report = evaluator.run()

    print_report(report)

    output_path = args.output
    if output_path is None:
        output_path = args.checkpoint.parent / f"eval_{args.split}.json"
    report.save(output_path)


if __name__ == "__main__":
    main()
