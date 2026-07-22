"""Generate evaluation and benchmark plots.

Loads evaluation JSONs from outputs/results/ and an optional benchmark
report, then produces publication-ready PDF+PNG figures.

Requires matplotlib: pip install matplotlib

Examples
--------
    # All plots (eval + benchmark)
    uv run scripts/plot_results.py

    # Eval-only plots (no benchmark data needed)
    uv run scripts/plot_results.py --no-benchmark

    # Custom paths and reference error probability
    uv run scripts/plot_results.py \
        --results-dir outputs/results \
        --benchmark outputs/benchmark_report.json \
        --reference-p 0.005

    # Custom output directory
    uv run scripts/plot_results.py -o outputs/figures
"""

import argparse
import logging
from pathlib import Path
from typing import Sequence

from benchmarks.plots import generate_all_plots


logger = logging.getLogger(__name__)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("outputs/results"),
        help="Directory containing evaluation JSON files.",
    )
    parser.add_argument(
        "--benchmark",
        type=Path,
        default=Path("outputs/benchmark_report.json"),
        help="Path to benchmark_report.json.",
    )
    parser.add_argument(
        "--no-benchmark",
        action="store_true",
        help="Skip benchmark-dependent plots (Pareto, throughput, speedup).",
    )
    parser.add_argument(
        "--reference-p",
        type=float,
        default=None,
        help="Error probability for LER scaling plot (auto-selects median if omitted).",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path("outputs/figures"),
        help="Where to save generated figures.",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    benchmark_path = None if args.no_benchmark else args.benchmark

    generate_all_plots(
        results_dir=args.results_dir,
        benchmark_path=benchmark_path,
        output_dir=args.output_dir,
        reference_p=args.reference_p,
    )


if __name__ == "__main__":
    main()
