"""Evaluate MWPM baseline decoder across all settings.

Computes the logical error rate (LER) for PyMatching's minimum-weight
perfect matching decoder on every (distance, rounds, error_prob) setting
defined in the configuration.  This is the essential sanity check and
reference point before training any ML model.

Usage
-----
    python scripts/eval_mwpm.py -c configs/data_generation.yaml
    python scripts/eval_mwpm.py -c configs/data_generation.yaml --splits test val
    python scripts/eval_mwpm.py -c configs/data_generation.yaml -o results/mwpm_baseline.json

The script can operate in two modes (chosen automatically):

1. **From raw data** (default): loads saved ``.stim`` circuits, rebuilds
   the DEM and Matching decoder, then decodes saved syndromes.
2. **Regenerate circuits**: if ``--regenerate`` is passed, builds circuits
   from config parameters instead of loading saved files.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import pymatching
import stim
from tqdm import tqdm


_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from qec_generator.config import Config, load_config  # noqa: E402
from qec_generator.sampler import build_circuit  # noqa: E402


logger = logging.getLogger(__name__)


@dataclass
class SettingResult:
    """Evaluation result for a single (distance, rounds, error_prob) setting.

    Attributes
    ----------
    distance : int
        Code distance.
    rounds : int
        Number of syndrome measurement rounds.
    error_prob : float
        Physical error probability.
    split : str
        Evaluated split name (``"train"``, ``"val"``, or ``"test"``).
    num_shots : int
        Total number of shots evaluated.
    num_errors : int
        Number of shots where the decoder predicted the wrong logical
        observable (logical errors).
    logical_error_rate : float
        ``num_errors / num_shots``.
    elapsed_s : float
        Wall-clock seconds spent decoding this setting.
    """

    distance: int
    rounds: int
    error_prob: float
    split: str
    num_shots: int
    num_errors: int
    logical_error_rate: float
    elapsed_s: float

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a JSON-friendly dictionary."""
        return {
            "distance": self.distance,
            "rounds": self.rounds,
            "error_prob": self.error_prob,
            "split": self.split,
            "num_shots": self.num_shots,
            "num_errors": self.num_errors,
            "logical_error_rate": self.logical_error_rate,
            "elapsed_s": round(self.elapsed_s, 4),
        }


@dataclass
class EvalReport:
    """Aggregated MWPM evaluation report.

    Attributes
    ----------
    results : list of SettingResult
        Per-setting results.
    metadata : dict
        Run-level metadata (config path, decoder, etc.).
    """

    results: List[SettingResult] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the full report."""
        return {
            "metadata": self.metadata,
            "results": [r.to_dict() for r in self.results],
        }


def _load_or_build_circuit(
    cfg: Config,
    setting_dir: Path,
    distance: int,
    rounds: int,
    p: float,
    *,
    regenerate: bool,
) -> stim.Circuit:
    """Load a saved Stim circuit or rebuild it from config parameters.

    Parameters
    ----------
    cfg : Config
        Dataset configuration.
    setting_dir : Path
        Directory for this setting (may contain ``circuit.stim``).
    distance : int
        Code distance.
    rounds : int
        Number of syndrome measurement rounds.
    p : float
        Physical error probability.
    regenerate : bool
        If True, always rebuild from config instead of loading saved file.

    Returns
    -------
    stim.Circuit
        The Stim circuit for this setting.

    Raises
    ------
    FileNotFoundError
        If ``regenerate=False`` and no saved circuit exists.
    """
    circuit_path = setting_dir / "circuit.stim"

    if not regenerate and circuit_path.is_file():
        logger.debug("Loading circuit from %s", circuit_path)
        return stim.Circuit.from_file(str(circuit_path))

    if not regenerate and not circuit_path.is_file():
        logger.warning(
            "No saved circuit at %s; rebuilding from config", circuit_path
        )

    return build_circuit(cfg, distance, rounds, p)


def _load_split_arrays(
    setting_dir: Path,
    split: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Load syndrome and logical observable arrays for one split.

    Parameters
    ----------
    setting_dir : Path
        Setting directory containing ``{split}_syndrome.npy`` and
        ``{split}_logical.npy``.
    split : str
        Split name.

    Returns
    -------
    syndrome : ndarray, shape (N, num_detectors), dtype uint8
        Detector event vectors.
    logical : ndarray, shape (N, num_observables), dtype uint8
        Ground-truth logical observable flips.

    Raises
    ------
    FileNotFoundError
        If the split files do not exist.
    """
    syn_path = setting_dir / f"{split}_syndrome.npy"
    log_path = setting_dir / f"{split}_logical.npy"

    if not syn_path.is_file():
        raise FileNotFoundError(f"Syndrome file not found: {syn_path}")
    if not log_path.is_file():
        raise FileNotFoundError(f"Logical file not found: {log_path}")

    syndrome = np.load(syn_path).astype(np.uint8, copy=False)
    logical = np.load(log_path).astype(np.uint8, copy=False)

    # Ensure logical is 2-D: (N, num_observables)
    if logical.ndim == 1:
        logical = logical[:, np.newaxis]

    return syndrome, logical


def evaluate_setting(
    cfg: Config,
    distance: int,
    rounds: int,
    p: float,
    split: str,
    *,
    regenerate: bool = False,
    chunk_size: int = 10_000,
) -> SettingResult:
    """Run MWPM decoding on one setting and compute the logical error rate.

    Parameters
    ----------
    cfg : Config
        Dataset configuration.
    distance : int
        Code distance.
    rounds : int
        Number of syndrome measurement rounds.
    p : float
        Physical error probability.
    split : str
        Data split to evaluate.
    regenerate : bool
        Rebuild circuits from config rather than loading saved files.
    chunk_size : int
        Number of shots per ``decode_batch`` call.

    Returns
    -------
    SettingResult
        Evaluation result with logical error rate and timing.
    """
    setting_dir = cfg.setting_dir(distance, rounds, p)

    circuit = _load_or_build_circuit(
        cfg, setting_dir, distance, rounds, p, regenerate=regenerate
    )
    dem = circuit.detector_error_model(decompose_errors=True)
    matching = pymatching.Matching.from_detector_error_model(dem)

    syndrome, logical_true = _load_split_arrays(setting_dir, split)

    num_shots = syndrome.shape[0]
    num_obs = dem.num_observables

    # Validate dimensions
    if syndrome.shape[1] != dem.num_detectors:
        raise ValueError(
            f"Syndrome width {syndrome.shape[1]} != "
            f"num_detectors {dem.num_detectors} "
            f"(d={distance}, r={rounds}, p={p})"
        )
    if logical_true.shape[1] != num_obs:
        raise ValueError(
            f"Logical width {logical_true.shape[1]} != "
            f"num_observables {num_obs} "
            f"(d={distance}, r={rounds}, p={p})"
        )

    # Decode in chunks
    predicted = np.empty((num_shots, num_obs), dtype=np.uint8)
    t0 = time.perf_counter()

    for off in range(0, num_shots, chunk_size):
        end = min(off + chunk_size, num_shots)
        batch = syndrome[off:end]
        predicted[off:end] = matching.decode_batch(batch)[:, :num_obs]

    elapsed = time.perf_counter() - t0

    # A logical error occurs when any observable prediction disagrees
    shot_errors = np.any(predicted != logical_true, axis=1)
    num_errors = int(shot_errors.sum())
    ler = num_errors / num_shots

    return SettingResult(
        distance=distance,
        rounds=rounds,
        error_prob=p,
        split=split,
        num_shots=num_shots,
        num_errors=num_errors,
        logical_error_rate=ler,
        elapsed_s=elapsed,
    )


def evaluate_all(
    cfg: Config,
    splits: Sequence[str],
    *,
    regenerate: bool = False,
    chunk_size: int = 10_000,
) -> EvalReport:
    """Evaluate MWPM decoder across all settings and requested splits.

    Parameters
    ----------
    cfg : Config
        Dataset configuration.
    splits : sequence of str
        Splits to evaluate (e.g., ``["test"]`` or ``["test", "val"]``).
    regenerate : bool
        Rebuild circuits from config rather than loading saved files.
    chunk_size : int
        Shots per decoding batch.

    Returns
    -------
    EvalReport
        Full evaluation report with per-setting results and metadata.
    """
    settings = list(cfg.iter_settings())
    total_jobs = len(settings) * len(splits)

    report = EvalReport(
        metadata={
            "decoder": "pymatching_mwpm",
            "pymatching_version": getattr(pymatching, "__version__", "unknown"),
            "stim_version": getattr(stim, "__version__", "unknown"),
            "family": cfg.family,
            "splits": list(splits),
            "num_settings": len(settings),
            "regenerate": regenerate,
        }
    )

    logger.info(
        "Evaluating MWPM on %d settings x %d splits = %d jobs",
        len(settings),
        len(splits),
        total_jobs,
    )

    progress = tqdm(total=total_jobs, desc="MWPM eval", unit="job")

    for d, r, p in settings:
        for split in splits:
            try:
                result = evaluate_setting(
                    cfg,
                    d,
                    r,
                    p,
                    split,
                    regenerate=regenerate,
                    chunk_size=chunk_size,
                )
                report.results.append(result)
                progress.set_postfix_str(
                    f"d={d} r={r} p={p:.4f} LER={result.logical_error_rate:.4f}"
                )
            except FileNotFoundError as exc:
                logger.warning("Skipping d=%d r=%d p=%s split=%s: %s", d, r, p, split, exc)
            except Exception:
                logger.exception(
                    "Failed: d=%d r=%d p=%s split=%s", d, r, p, split
                )

            progress.update(1)

    progress.close()
    return report


def print_report(report: EvalReport) -> None:
    """Print a formatted summary table to stdout.

    Parameters
    ----------
    report : EvalReport
        Evaluation report to display.
    """
    results = report.results
    if not results:
        logger.warning("No results to display")
        return

    # Header
    header = (
        f"{'split':<6} {'d':>3} {'r':>4} {'p':>9} "
        f"{'shots':>8} {'errors':>8} {'LER':>10} {'time_s':>7}"
    )
    sep = "-" * len(header)

    print(f"\n{sep}")
    print("MWPM Baseline — Logical Error Rate")
    print(sep)
    print(header)
    print(sep)

    # Sort: split -> distance -> rounds -> error_prob
    sorted_results = sorted(
        results, key=lambda r: (r.split, r.distance, r.rounds, r.error_prob)
    )

    prev_group = None
    for r in sorted_results:
        group = (r.split, r.distance)
        if prev_group is not None and group != prev_group:
            print()  # visual separator between groups
        prev_group = group

        print(
            f"{r.split:<6} {r.distance:>3} {r.rounds:>4} {r.error_prob:>9.5f} "
            f"{r.num_shots:>8} {r.num_errors:>8} "
            f"{r.logical_error_rate:>10.6f} {r.elapsed_s:>7.2f}"
        )

    print(sep)

    # Summary statistics per distance
    print("\nSummary by distance:")
    for split in sorted(set(r.split for r in results)):
        print(f"  [{split}]")
        for d in sorted(set(r.distance for r in results)):
            subset = [
                r for r in results if r.distance == d and r.split == split
            ]
            if not subset:
                continue
            lers = [r.logical_error_rate for r in subset]
            print(
                f"    d={d}: "
                f"min_LER={min(lers):.6f}  "
                f"max_LER={max(lers):.6f}  "
                f"mean_LER={np.mean(lers):.6f}  "
                f"({len(subset)} settings)"
            )

    print()


def save_report(report: EvalReport, path: Path) -> None:
    """Save the evaluation report to a JSON file.

    Parameters
    ----------
    report : EvalReport
        Report to save.
    path : Path
        Output file path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report.to_dict(), indent=2),
        encoding="utf-8",
    )
    logger.info("Report saved to %s", path)


def setup_logging(verbose: bool = False) -> None:
    """Configure root logging.

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


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Parameters
    ----------
    argv : sequence of str or None
        Arguments to parse; defaults to ``sys.argv[1:]``.

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "-c",
        "--config",
        type=Path,
        default=Path("configs/data_generation.yaml"),
        help="Path to YAML configuration file (default: configs/data_generation.yaml)",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["test"],
        help="Splits to evaluate (default: test)",
    )
    parser.add_argument(
        "--regenerate",
        action="store_true",
        help="Rebuild circuits from config instead of loading saved .stim files",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=10_000,
        help="Shots per decode_batch call (default: 10000)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Save JSON report to this path",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable DEBUG logging",
    )

    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Entry point for MWPM baseline evaluation.

    Parameters
    ----------
    argv : sequence of str or None
        CLI arguments; defaults to ``sys.argv[1:]``.
    """
    args = parse_args(argv)
    setup_logging(verbose=args.verbose)

    try:
        cfg = load_config(args.config)
    except FileNotFoundError:
        logger.error("Config not found: %s", args.config)
        sys.exit(1)
    except Exception as exc:
        logger.error("Failed to load config: %s", exc)
        sys.exit(1)

    logger.info("Config: %s", args.config)
    logger.info("Family: %s", cfg.family)
    logger.info("Distances: %s", cfg.distances)
    logger.info("Rounds: %s", cfg.rounds)
    logger.info("Error probs: %s", cfg.error_probs)
    logger.info("Splits: %s", args.splits)

    report = evaluate_all(
        cfg,
        splits=args.splits,
        regenerate=args.regenerate,
        chunk_size=args.chunk_size,
    )

    print_report(report)

    if args.output is not None:
        save_report(report, args.output)

    # Sanity checks
    _run_sanity_checks(report)


def _ler_stderr(ler: float, n: int) -> float:
    """Binomial standard error of a logical error rate estimate.

    Parameters
    ----------
    ler : float
        Observed logical error rate.
    n : int
        Number of shots.

    Returns
    -------
    float
        Standard error: ``sqrt(ler * (1 - ler) / n)``.
    """
    if n <= 0:
        return float("inf")
    return np.sqrt(ler * (1.0 - ler) / n)


def _is_significant(ler_a: float, n_a: int, ler_b: float, n_b: int) -> bool:
    """Test whether two LER estimates differ by more than 2 sigma.

    Parameters
    ----------
    ler_a, ler_b : float
        Observed logical error rates.
    n_a, n_b : int
        Number of shots for each estimate.

    Returns
    -------
    bool
        True if ``|ler_a - ler_b| > 2 * combined_stderr``.
    """
    se = np.sqrt(_ler_stderr(ler_a, n_a) ** 2 + _ler_stderr(ler_b, n_b) ** 2)
    return abs(ler_a - ler_b) > 2.0 * se


def _estimate_threshold(results: List[SettingResult]) -> float:
    """Estimate the error-correction threshold from crossover data.

    The threshold is the physical error probability where LER stops
    decreasing with code distance.  We look for the smallest ``p``
    at which the largest distance has higher LER than the smallest
    distance (for any round count).

    Parameters
    ----------
    results : list of SettingResult
        All evaluation results.

    Returns
    -------
    float
        Estimated threshold ``p``.  Returns ``inf`` if no crossover
        is found (all ``p`` are below threshold).
    """
    distances = sorted(set(r.distance for r in results))
    if len(distances) < 2:
        return float("inf")

    d_min, d_max = distances[0], distances[-1]
    crossover_ps: list[float] = []

    for split in set(r.split for r in results):
        for r_val in set(r.rounds for r in results):
            low_d = [
                r for r in results
                if r.distance == d_min and r.rounds == r_val and r.split == split
            ]
            high_d = [
                r for r in results
                if r.distance == d_max and r.rounds == r_val and r.split == split
            ]
            low_map = {r.error_prob: r for r in low_d}
            high_map = {r.error_prob: r for r in high_d}

            for p in sorted(low_map.keys() & high_map.keys()):
                rl, rh = low_map[p], high_map[p]
                if rh.logical_error_rate > rl.logical_error_rate:
                    crossover_ps.append(p)
                    break  # smallest p for this (split, rounds)

    if not crossover_ps:
        return float("inf")

    return float(np.median(crossover_ps))


def _run_sanity_checks(report: EvalReport) -> None:
    """Log warnings for statistically significant anomalies.

    Performs two checks on the evaluation results:

    1. **LER(p) monotonicity** — for fixed ``(d, r)``, LER should
       increase with physical error probability ``p``.  Only warns
       when the drop is statistically significant (>2σ binomial).

    2. **LER(d) scaling below threshold** — for fixed ``(r, p)`` and
       ``p`` below the estimated error-correction threshold, LER should
       decrease with code distance ``d``.  Above threshold the inverse
       scaling is expected and silently ignored.

    Parameters
    ----------
    report : EvalReport
        Completed evaluation report.
    """
    results = report.results
    if not results:
        return

    threshold_p = _estimate_threshold(results)
    if threshold_p < float("inf"):
        logger.info(
            "Estimated error-correction threshold: p ≈ %.4f", threshold_p
        )

    # ------------------------------------------------------------------
    # Check 1: LER should increase with p for fixed (d, r)
    # ------------------------------------------------------------------
    for split in sorted(set(r.split for r in results)):
        for d in sorted(set(r.distance for r in results)):
            for r_val in sorted(set(r.rounds for r in results)):
                subset = sorted(
                    [
                        r
                        for r in results
                        if r.distance == d
                        and r.rounds == r_val
                        and r.split == split
                    ],
                    key=lambda x: x.error_prob,
                )
                if len(subset) < 2:
                    continue

                for i in range(1, len(subset)):
                    prev, curr = subset[i - 1], subset[i]
                    if curr.logical_error_rate >= prev.logical_error_rate:
                        continue
                    if not _is_significant(
                        prev.logical_error_rate,
                        prev.num_shots,
                        curr.logical_error_rate,
                        curr.num_shots,
                    ):
                        continue
                    logger.warning(
                        "Non-monotonic LER: [%s] d=%d r=%d "
                        "p=%.5f (LER=%.6f) > p=%.5f (LER=%.6f)",
                        split,
                        d,
                        r_val,
                        prev.error_prob,
                        prev.logical_error_rate,
                        curr.error_prob,
                        curr.logical_error_rate,
                    )

    # ------------------------------------------------------------------
    # Check 2: LER should decrease with d — only below threshold
    # ------------------------------------------------------------------
    for split in sorted(set(r.split for r in results)):
        for p in sorted(set(r.error_prob for r in results)):
            if p >= threshold_p:
                continue  # above threshold: inverse scaling is expected

            for r_val in sorted(set(r.rounds for r in results)):
                subset = sorted(
                    [
                        r
                        for r in results
                        if r.error_prob == p
                        and r.rounds == r_val
                        and r.split == split
                    ],
                    key=lambda x: x.distance,
                )
                if len(subset) < 2:
                    continue

                for i in range(1, len(subset)):
                    prev, curr = subset[i - 1], subset[i]
                    if curr.logical_error_rate <= prev.logical_error_rate:
                        continue
                    if not _is_significant(
                        prev.logical_error_rate,
                        prev.num_shots,
                        curr.logical_error_rate,
                        curr.num_shots,
                    ):
                        continue
                    logger.warning(
                        "LER increasing with d below threshold: "
                        "[%s] r=%d p=%.5f "
                        "d=%d (LER=%.6f) < d=%d (LER=%.6f)",
                        split,
                        r_val,
                        p,
                        prev.distance,
                        prev.logical_error_rate,
                        curr.distance,
                        curr.logical_error_rate,
                    )


if __name__ == "__main__":
    main()
