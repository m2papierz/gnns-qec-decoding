"""Generate QEC datasets for GNN-based decoders."""

import argparse
import logging
import sys
from pathlib import Path

from qec_generator import CASES, Config, generate_datasets, generate_raw_data


def setup_logging(verbose: bool = False) -> None:
    """
    Configure logging for the application.

    Parameters
    ----------
    verbose : bool, default=False
        Enable DEBUG level logging if True.
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )


def main() -> None:
    """Run the QEC dataset generator CLI."""

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "-c",
        "--config",
        type=Path,
        default=Path("configs/data_generation.yaml"),
        help="Path to YAML configuration file",
    )

    parser.add_argument(
        "--mode",
        choices=["all", "raw-only", "dataset-only"],
        default="all",
        help="Generation mode: 'all' (raw + datasets), 'raw-only', or 'dataset-only'",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files",
    )

    parser.add_argument(
        "--cases",
        nargs="+",
        choices=CASES,
        default=list(CASES),
        help="Dataset cases to generate (default: all)",
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose (DEBUG) logging",
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(verbose=args.verbose)
    log = logging.getLogger(__name__)

    # Load configuration
    try:
        cfg = Config.from_yaml(args.config)
    except FileNotFoundError:
        log.error("Configuration file not found: %s", args.config)
        sys.exit(1)
    except Exception as e:
        log.error("Failed to load configuration: %s", e)
        sys.exit(1)

    total_settings = len(list(cfg.iter_settings()))
    log.info("Configuration loaded from %s", args.config)
    log.info("Family: %s", cfg.family)
    log.info("Distances: %s", cfg.distances)
    log.info("Rounds: %s", cfg.rounds)
    log.info("Error probabilities: %s", cfg.error_probs)
    log.info("Total settings: %d", total_settings)
    log.info("Samples per split:")
    for split, n_samples in cfg.num_samples.items():
        total = n_samples * total_settings
        log.info(
            "  %s: %d samples/setting x %d settings = %d total",
            split,
            n_samples,
            total_settings,
            total,
        )

    # Generate raw data
    if args.mode in ("all", "raw-only"):
        log.info("=" * 60)
        log.info("Generating raw data → %s", cfg.raw_data_dir)
        log.info("=" * 60)
        try:
            generate_raw_data(cfg, overwrite=args.overwrite)
        except Exception as e:
            log.error("Raw data generation failed: %s", e, exc_info=True)
            sys.exit(1)

    # Build datasets
    if args.mode in ("all", "dataset-only"):
        log.info("=" * 60)
        log.info("Building datasets → %s", cfg.datasets_dir)
        log.info("Cases: %s", args.cases)
        log.info("=" * 60)
        try:
            generate_datasets(
                cfg,
                cases=tuple(args.cases),
                overwrite=args.overwrite,
            )
        except Exception as e:
            log.error("Dataset generation failed: %s", e, exc_info=True)
            sys.exit(1)

    log.info("=" * 60)
    log.info("    Generation complete")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
