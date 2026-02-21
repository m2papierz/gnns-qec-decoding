"""
Evaluate MWPM baseline decoder across all settings.

Computes the logical error rate (LER) for PyMatching's minimum-weight
perfect matching decoder on every (distance, rounds, error_prob) setting
defined in the configuration.

Usage
-----
    uv run scripts/eval_mwpm.py -c configs/data_generation.yaml
    uv run scripts/eval_mwpm.py -c configs/data_generation.yaml --splits test val
    uv run scripts/eval_mwpm.py -c configs/data_generation.yaml -o outputs/results/mwpm_baseline.json
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import List, Sequence

import numpy as np
import pymatching
import stim
from tqdm import tqdm


_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from cli import setup_logging  # noqa: E402
from gnn.metrics import EvalReport, SettingResult, print_report  # noqa: E402
from qec_generator.config import Config, load_config  # noqa: E402
from qec_generator.sampler import build_circuit  # noqa: E402


logger = logging.getLogger(__name__)


def _load_or_build_circuit(
    cfg: Config,
    setting_dir: Path,
    distance: int,
    rounds: int,
    p: float,
    *,
    regenerate: bool,
) -> stim.Circuit:
    """Load a saved Stim circuit or rebuild it from config parameters."""
    circuit_path = setting_dir / "circuit.stim"

    if not regenerate and circuit_path.is_file():
        return stim.Circuit.from_file(str(circuit_path))

    if not regenerate:
        logger.warning(
            "No saved circuit at %s; rebuilding from config",
            circuit_path,
        )

    return build_circuit(cfg, distance, rounds, p)


def _load_split_arrays(
    setting_dir: Path,
    split: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Load syndrome and logical observable arrays for one split."""
    syn_path = setting_dir / f"{split}_syndrome.npy"
    log_path = setting_dir / f"{split}_logical.npy"

    if not syn_path.is_file():
        raise FileNotFoundError(f"Syndrome file not found: {syn_path}")
    if not log_path.is_file():
        raise FileNotFoundError(f"Logical file not found: {log_path}")

    syndrome = np.load(syn_path).astype(np.uint8, copy=False)
    logical = np.load(log_path).astype(np.uint8, copy=False)
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
    """Run MWPM decoding on one setting and compute the logical error rate."""
    setting_dir = cfg.setting_dir(distance, rounds, p)

    circuit = _load_or_build_circuit(
        cfg,
        setting_dir,
        distance,
        rounds,
        p,
        regenerate=regenerate,
    )
    dem = circuit.detector_error_model(decompose_errors=True)
    matching = pymatching.Matching.from_detector_error_model(dem)

    syndrome, logical_true = _load_split_arrays(setting_dir, split)
    num_shots = syndrome.shape[0]
    num_obs = dem.num_observables

    if syndrome.shape[1] != dem.num_detectors:
        raise ValueError(
            f"Syndrome width {syndrome.shape[1]} != "
            f"num_detectors {dem.num_detectors} (d={distance}, r={rounds}, p={p})"
        )
    if logical_true.shape[1] != num_obs:
        raise ValueError(
            f"Logical width {logical_true.shape[1]} != "
            f"num_observables {num_obs} (d={distance}, r={rounds}, p={p})"
        )

    predicted = np.empty((num_shots, num_obs), dtype=np.uint8)
    t0 = time.perf_counter()

    for off in range(0, num_shots, chunk_size):
        end = min(off + chunk_size, num_shots)
        predicted[off:end] = matching.decode_batch(syndrome[off:end])[:, :num_obs]

    elapsed = time.perf_counter() - t0

    shot_errors = np.any(predicted != logical_true, axis=1)
    num_errors = int(shot_errors.sum())

    return SettingResult(
        distance=distance,
        rounds=rounds,
        error_prob=p,
        split=split,
        num_shots=num_shots,
        num_errors=num_errors,
        logical_error_rate=num_errors / num_shots,
        elapsed_s=elapsed,
    )


def evaluate_all(
    cfg: Config,
    splits: Sequence[str],
    *,
    regenerate: bool = False,
    chunk_size: int = 10_000,
) -> EvalReport:
    """Evaluate MWPM decoder across all settings and requested splits."""
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
                logger.warning(
                    "Skipping d=%d r=%d p=%s split=%s: %s",
                    d,
                    r,
                    p,
                    split,
                    exc,
                )
            except Exception:
                logger.exception(
                    "Failed: d=%d r=%d p=%s split=%s",
                    d,
                    r,
                    p,
                    split,
                )
            progress.update(1)

    progress.close()
    return report


def _ler_stderr(ler: float, n: int) -> float:
    if n <= 0:
        return float("inf")
    return np.sqrt(ler * (1.0 - ler) / n)


def _is_significant(ler_a: float, n_a: int, ler_b: float, n_b: int) -> bool:
    se = np.sqrt(_ler_stderr(ler_a, n_a) ** 2 + _ler_stderr(ler_b, n_b) ** 2)
    return abs(ler_a - ler_b) > 2.0 * se


def _estimate_threshold(results: List[SettingResult]) -> float:
    """
    Estimate error-correction threshold from crossover data.

    Finds the smallest ``p`` where the largest code distance has
    higher LER than the smallest.
    """
    distances = sorted(set(r.distance for r in results))
    if len(distances) < 2:
        return float("inf")

    d_min, d_max = distances[0], distances[-1]
    crossover_ps: list[float] = []

    for split in set(r.split for r in results):
        for r_val in set(r.rounds for r in results):
            low_d = [
                r
                for r in results
                if r.distance == d_min and r.rounds == r_val and r.split == split
            ]
            high_d = [
                r
                for r in results
                if r.distance == d_max and r.rounds == r_val and r.split == split
            ]
            low_map = {r.error_prob: r for r in low_d}
            high_map = {r.error_prob: r for r in high_d}

            for p in sorted(low_map.keys() & high_map.keys()):
                if high_map[p].logical_error_rate > low_map[p].logical_error_rate:
                    crossover_ps.append(p)
                    break

    return float(np.median(crossover_ps)) if crossover_ps else float("inf")


def _run_sanity_checks(report: EvalReport) -> None:
    """Log warnings for statistically significant anomalies."""
    results = report.results
    if not results:
        return

    threshold_p = _estimate_threshold(results)
    if threshold_p < float("inf"):
        logger.info("Estimated error-correction threshold: p ≈ %.4f", threshold_p)

    # LER should increase with p for fixed (d, r)
    for split in sorted(set(r.split for r in results)):
        for d in sorted(set(r.distance for r in results)):
            for r_val in sorted(set(r.rounds for r in results)):
                subset = sorted(
                    [
                        r
                        for r in results
                        if r.distance == d and r.rounds == r_val and r.split == split
                    ],
                    key=lambda x: x.error_prob,
                )
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

    # LER should decrease with d — only below threshold
    for split in sorted(set(r.split for r in results)):
        for p in sorted(set(r.error_prob for r in results)):
            if p >= threshold_p:
                continue
            for r_val in sorted(set(r.rounds for r in results)):
                subset = sorted(
                    [
                        r
                        for r in results
                        if r.error_prob == p and r.rounds == r_val and r.split == split
                    ],
                    key=lambda x: x.distance,
                )
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
                        "[%s] r=%d p=%.5f d=%d (LER=%.6f) < d=%d (LER=%.6f)",
                        split,
                        r_val,
                        p,
                        prev.distance,
                        prev.logical_error_rate,
                        curr.distance,
                        curr.logical_error_rate,
                    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "-c",
        "--config",
        type=Path,
        default=Path("configs/data_generation.yaml"),
    )
    parser.add_argument("--splits", nargs="+", default=["test"])
    parser.add_argument("--regenerate", action="store_true")
    parser.add_argument("--chunk-size", type=int, default=10_000)
    parser.add_argument("-o", "--output", type=Path, default=None)
    parser.add_argument("-v", "--verbose", action="store_true")

    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Entry point for MWPM baseline evaluation."""
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
    logger.info("Distances: %s, Rounds: %s", cfg.distances, cfg.rounds)
    logger.info("Error probs: %s, Splits: %s", cfg.error_probs, args.splits)

    report = evaluate_all(
        cfg,
        splits=args.splits,
        regenerate=args.regenerate,
        chunk_size=args.chunk_size,
    )

    print_report(report)

    if args.output is not None:
        report.save(args.output)

    _run_sanity_checks(report)


if __name__ == "__main__":
    main()
