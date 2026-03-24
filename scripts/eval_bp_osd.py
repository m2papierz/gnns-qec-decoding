"""
Evaluate BP+OSD baseline decoder across all settings.

Computes the logical error rate (LER) for the NVIDIA CUDA-Q BP+OSD
decoder using static DEM weights (no GNN) on every (distance, rounds,
error_prob) setting defined in the configuration.

Usage
-----
    uv run scripts/eval_bp_osd_baseline.py -c configs/data_generation.yaml
    uv run scripts/eval_bp_osd_baseline.py -c configs/data_generation.yaml --splits test val
    uv run scripts/eval_bp_osd_baseline.py -c configs/data_generation.yaml -o outputs/results/bp_osd_baseline.json
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import stim
from tqdm import tqdm


_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from cli import setup_logging
from decoders.base import DecoderConfig
from decoders.bp_osd import BPOSDDecoder
from gnn.metrics import EvalReport, SettingResult, print_report
from qec_generator.config import Config, load_config
from qec_generator.graph import build_detector_graph
from qec_generator.sampler import build_circuit
from qec_generator.utils import undirected_edges


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


def _build_baseline_decoder(
    circuit: stim.Circuit,
    dem: stim.DetectorErrorModel,
) -> BPOSDDecoder:
    """Construct a BPOSDDecoder from static DEM weights (no GNN)."""
    graph = build_detector_graph(circuit, dem, include_boundary=True)

    und_pairs, dir_to_undir = undirected_edges(graph.edge_index)
    num_und = und_pairs.shape[0]

    # Collapse directed edge probs to undirected (first occurrence).
    first_idx = np.full(num_und, -1, dtype=np.int64)
    for e, uid in enumerate(dir_to_undir):
        if first_idx[uid] == -1:
            first_idx[uid] = e
    edge_probs = graph.edge_error_prob[first_idx]

    # Collapse directed observable_flips to undirected.
    obs_flips_und: np.ndarray
    if graph.observable_flips is not None:
        obs_flips_und = graph.observable_flips[first_idx]
    else:
        obs_flips_und = np.zeros((num_und, graph.num_observables), dtype=bool)

    config = DecoderConfig(
        num_detectors=graph.num_detectors,
        num_observables=graph.num_observables,
        has_boundary=graph.has_boundary,
        boundary_node=graph.num_detectors if graph.has_boundary else None,
    )

    return BPOSDDecoder(config, und_pairs, edge_probs, obs_flips_und)


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
    """Run BP+OSD decoding on one setting and compute the logical error rate."""
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
    decoder = _build_baseline_decoder(circuit, dem)

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
        predicted[off:end] = decoder.decode_batch(syndrome[off:end])

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
    """Evaluate BP+OSD decoder across all settings and requested splits."""
    settings = list(cfg.iter_settings())
    total_jobs = len(settings) * len(splits)

    report = EvalReport(
        metadata={
            "decoder": "cudaq_bp_osd_baseline",
            "stim_version": getattr(stim, "__version__", "unknown"),
            "family": cfg.family,
            "splits": list(splits),
            "num_settings": len(settings),
            "regenerate": regenerate,
        }
    )

    logger.info(
        "Evaluating BP+OSD on %d settings x %d splits = %d jobs",
        len(settings),
        len(splits),
        total_jobs,
    )

    progress = tqdm(total=total_jobs, desc="BP+OSD eval", unit="job")

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
    """Entry point for BP+OSD baseline evaluation."""
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


if __name__ == "__main__":
    main()
