"""Generate frozen evaluation sets for all (d, p) evaluation points.

Adaptively samples shots from committed circuit files until each (d, p) point
accumulates ≥400 MWPM logical errors (target) or hits the 1,000,000 shot cap.
Each set is saved as compressed numpy archives with a manifest recording all
provenance information.

Usage
-----
    uv run python scripts/generate_eval_sets.py
    uv run python scripts/generate_eval_sets.py --target-errors 400 --cap 1000000
    uv run python scripts/generate_eval_sets.py --distances 3 5  # subset
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path

import numpy as np
import pymatching
import stim

from qec_generator.sampler import settings_from_circuit_dir
from qec_generator.utils import stable_seed


logger = logging.getLogger(__name__)

CIRCUIT_DIR = Path("data/circuits")
OUTPUT_DIR = Path("data/eval")
MASTER_SEED = 20240601
BATCH_SIZE = 50_000
TARGET_ERRORS = 400
SHOT_CAP = 1_000_000


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fired_count_stats(syndromes: np.ndarray) -> dict:
    """Compute fired-detector distribution statistics."""
    fired_counts = syndromes.sum(axis=1)
    return {
        "mean": float(np.mean(fired_counts)),
        "median": float(np.median(fired_counts)),
        "std": float(np.std(fired_counts)),
        "p99": float(np.percentile(fired_counts, 99)),
        "p99_9": float(np.percentile(fired_counts, 99.9)),
        "max": int(np.max(fired_counts)),
        "min": int(np.min(fired_counts)),
        "zero_count": int(np.sum(fired_counts == 0)),
    }


def _fired_count_buckets(syndromes: np.ndarray, observables: np.ndarray) -> dict:
    """Analyze val-set composition: positives per fired-count bucket."""
    fired_counts = syndromes.sum(axis=1)
    labels = observables.any(axis=1) if observables.ndim > 1 else observables.ravel()

    buckets = {
        "0_fired": (fired_counts == 0),
        "1_to_3_fired": (fired_counts >= 1) & (fired_counts <= 3),
        "4_to_10_fired": (fired_counts >= 4) & (fired_counts <= 10),
        "11_to_30_fired": (fired_counts >= 11) & (fired_counts <= 30),
        "31_plus_fired": (fired_counts >= 31),
    }

    result = {}
    for name, mask in buckets.items():
        n_total = int(mask.sum())
        n_positive = int(labels[mask].sum()) if n_total > 0 else 0
        result[name] = {
            "total": n_total,
            "positive": n_positive,
            "positive_rate": n_positive / n_total if n_total > 0 else 0.0,
        }
    return result


def generate_eval_set(
    circuit_path: Path,
    distance: int,
    rounds: int,
    error_prob: float,
    *,
    target_errors: int = TARGET_ERRORS,
    shot_cap: int = SHOT_CAP,
    batch_size: int = BATCH_SIZE,
    master_seed: int = MASTER_SEED,
    output_dir: Path = OUTPUT_DIR,
) -> dict:
    """Generate a frozen eval set for a single (d, p) point.

    Parameters
    ----------
    circuit_path : Path
        Path to the .stim circuit file.
    distance, rounds : int
        Code parameters.
    error_prob : float
        Physical error probability.
    target_errors : int
        Minimum MWPM errors to accumulate before stopping.
    shot_cap : int
        Maximum shots per point.
    batch_size : int
        Shots generated per sampling batch.
    master_seed : int
        Master seed for deterministic generation.
    output_dir : Path
        Root output directory.

    Returns
    -------
    dict
        Summary with error counts, fired-count stats, and paths.
    """
    circuit = stim.Circuit.from_file(str(circuit_path))
    dem = circuit.detector_error_model(decompose_errors=True)
    matching = pymatching.Matching.from_detector_error_model(dem)
    n_det = dem.num_detectors
    n_obs = dem.num_observables

    seed = stable_seed("eval", f"d={distance}", f"p={error_prob}", base=master_seed)
    sampler = circuit.compile_detector_sampler(seed=seed)

    syndrome_batches: list[np.ndarray] = []
    observable_batches: list[np.ndarray] = []
    mwpm_errors_total = 0
    shots_total = 0

    logger.info(
        "Generating d=%d p=%.4f (target=%d errors, cap=%d shots)",
        distance,
        error_prob,
        target_errors,
        shot_cap,
    )

    while mwpm_errors_total < target_errors and shots_total < shot_cap:
        remaining = shot_cap - shots_total
        n_batch = min(batch_size, remaining)

        raw = sampler.sample(shots=n_batch, bit_packed=False, append_observables=True)
        syndromes = raw[:, :n_det].astype(np.uint8)
        observables = raw[:, n_det : n_det + n_obs].astype(np.uint8)

        mwpm_pred = matching.decode_batch(syndromes)[:, :n_obs]
        batch_errors = int(np.any(mwpm_pred != observables, axis=1).sum())

        syndrome_batches.append(syndromes)
        observable_batches.append(observables)
        mwpm_errors_total += batch_errors
        shots_total += n_batch

        logger.info(
            "  batch +%d shots → %d/%d MWPM errors (total %d shots)",
            n_batch,
            mwpm_errors_total,
            target_errors,
            shots_total,
        )

    all_syndromes = np.concatenate(syndrome_batches, axis=0)
    all_observables = np.concatenate(observable_batches, axis=0)

    # Fired-count analysis
    fired_stats = _fired_count_stats(all_syndromes)
    bucket_composition = _fired_count_buckets(all_syndromes, all_observables)

    # Verify MWPM error count on full set (should match accumulated)
    full_mwpm_pred = matching.decode_batch(all_syndromes)[:, :n_obs]
    verified_errors = int(np.any(full_mwpm_pred != all_observables, axis=1).sum())
    assert (
        verified_errors == mwpm_errors_total
    ), f"Accumulated {mwpm_errors_total} != verified {verified_errors}"

    # Extract detector coordinates
    coord_dict = circuit.get_detector_coordinates()
    detector_coords = np.zeros((n_det, 3), dtype=np.float64)
    for det_id, c in coord_dict.items():
        if 0 <= det_id < n_det:
            detector_coords[det_id, : min(len(c), 3)] = c[:3]

    # Save
    p_str = f"{error_prob:.4f}".replace(".", "_")
    point_dir = output_dir / f"d{distance}_p{p_str}"
    point_dir.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        point_dir / "data.npz",
        syndromes=all_syndromes,
        observables=all_observables,
        detector_coords=detector_coords,
    )

    positive_count = int(all_observables.any(axis=1).sum())
    manifest = {
        "circuit_file": str(circuit_path),
        "circuit_sha256": _sha256(circuit_path),
        "stim_version": stim.__version__,
        "seed": seed,
        "master_seed": master_seed,
        "distance": distance,
        "rounds": rounds,
        "error_prob": error_prob,
        "num_shots": shots_total,
        "num_detectors": n_det,
        "num_observables": n_obs,
        "mwpm_errors": verified_errors,
        "mwpm_ler": verified_errors / shots_total,
        "positive_count": positive_count,
        "positive_rate": positive_count / shots_total,
        "target_errors": target_errors,
        "shot_cap": shot_cap,
        "target_met": mwpm_errors_total >= target_errors,
        "fired_count_stats": fired_stats,
        "fired_count_buckets": bucket_composition,
        "n_max_p99_9": fired_stats["p99_9"],
        "generation_command": "uv run python scripts/generate_eval_sets.py",
    }

    (point_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=False),
        encoding="utf-8",
    )

    met_str = "MET" if manifest["target_met"] else "BELOW TARGET"
    logger.info(
        "  → %s: %d shots, %d MWPM errors (LER=%.6f), N_max(p99.9)=%.0f [%s]",
        point_dir.name,
        shots_total,
        verified_errors,
        manifest["mwpm_ler"],
        fired_stats["p99_9"],
        met_str,
    )

    return manifest


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--circuit-dir", type=Path, default=CIRCUIT_DIR, help="Circuit file directory"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=OUTPUT_DIR, help="Output directory"
    )
    parser.add_argument(
        "--target-errors", type=int, default=TARGET_ERRORS, help="Target MWPM errors"
    )
    parser.add_argument("--cap", type=int, default=SHOT_CAP, help="Max shots per point")
    parser.add_argument(
        "--batch-size", type=int, default=BATCH_SIZE, help="Shots per sampling batch"
    )
    parser.add_argument("--seed", type=int, default=MASTER_SEED, help="Master seed")
    parser.add_argument(
        "--distances", type=int, nargs="*", default=None, help="Distances to generate"
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    settings = settings_from_circuit_dir(args.circuit_dir, distances=args.distances)
    logger.info("Generating eval sets for %d (d, p) points", len(settings))

    results = []
    t0 = time.time()

    for s in settings:
        manifest = generate_eval_set(
            circuit_path=s.circuit_path,
            distance=s.distance,
            rounds=s.rounds,
            error_prob=s.error_prob,
            target_errors=args.target_errors,
            shot_cap=args.cap,
            batch_size=args.batch_size,
            master_seed=args.seed,
            output_dir=args.output_dir,
        )
        results.append(manifest)

    elapsed = time.time() - t0

    # Summary table
    print()
    print("=" * 90)
    print("Frozen Eval Set Generation Summary")
    print("=" * 90)
    print(
        f"{'Point':<14} {'Shots':>8} {'MWPM Err':>9} {'MWPM LER':>10} "
        f"{'Pos Rate':>9} {'N_max(p99.9)':>13} {'Status':>8}"
    )
    print("-" * 90)
    for r in results:
        status = "OK" if r["target_met"] else "BELOW"
        print(
            f"d{r['distance']}_p{r['error_prob']:<7.4f} "
            f"{r['num_shots']:>8} {r['mwpm_errors']:>9} "
            f"{r['mwpm_ler']:>10.6f} {r['positive_rate']:>9.4f} "
            f"{r['n_max_p99_9']:>13.0f} {status:>8}"
        )

    # Composition check
    print()
    print("Val-set composition (positives per fired-count bucket):")
    print("-" * 90)
    print(
        f"{'Point':<14} {'0 fired':>10} {'1-3 fired':>12} "
        f"{'4-10 fired':>12} {'11-30 fired':>13} {'31+ fired':>12}"
    )
    print("-" * 90)
    for r in results:
        b = r["fired_count_buckets"]

        def _fmt(bucket: dict) -> str:
            return f"{bucket['positive']}/{bucket['total']}"

        print(
            f"d{r['distance']}_p{r['error_prob']:<7.4f} "
            f"{_fmt(b['0_fired']):>10} {_fmt(b['1_to_3_fired']):>12} "
            f"{_fmt(b['4_to_10_fired']):>12} {_fmt(b['11_to_30_fired']):>13} "
            f"{_fmt(b['31_plus_fired']):>12}"
        )

    # N_max table
    print()
    print("N_max (p99.9 fired detectors) per (d, p):")
    print("-" * 50)
    for r in results:
        print(
            f"  d={r['distance']} p={r['error_prob']:.4f}: "
            f"N_max(p99.9) = {r['n_max_p99_9']:.0f}, "
            f"max = {r['fired_count_stats']['max']}"
        )

    below = [r for r in results if not r["target_met"]]
    if below:
        print()
        print(
            f"WARNING: {len(below)} point(s) below {args.target_errors}-error target:"
        )
        for r in below:
            print(
                f"  d={r['distance']} p={r['error_prob']:.4f}: "
                f"{r['mwpm_errors']} errors in {r['num_shots']} shots"
            )

    print(f"\nTotal generation time: {elapsed:.1f}s")
    print(f"Output directory: {args.output_dir}")


if __name__ == "__main__":
    main()
