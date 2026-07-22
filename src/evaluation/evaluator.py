"""
Evaluation harness: multi-decoder comparison on frozen eval sets.

Orchestrates paired evaluation of decoders (GNN, MWPM, BP+OSD) on identical
pre-sampled shots with adaptive early stopping via the Haybittle-Peto boundary.

The harness loads frozen eval sets, runs all decoders on the same syndromes in
check-interval increments, and stops per point once McNemar resolves or the
shot budget is exhausted.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from decoders import Decoder
from evaluation.stats import (
    CHECK_INTERVAL,
    EvalOutcome,
    McNemarResult,
    StoppingDecision,
    WilsonInterval,
    adaptive_stop,
    mcnemar_test,
    per_round_ler,
    wilson_interval,
)


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class EvalSet:
    """Frozen evaluation set loaded from disk.

    Parameters
    ----------
    syndromes : ndarray, shape ``(N, D)``, uint8
        Binary syndrome vectors.
    observables : ndarray, shape ``(N, O)``, uint8
        True logical observable flips.
    detector_coords : ndarray, shape ``(D, 3)``, float64
        Detector (x, y, t) coordinates.
    distance : int
        Code distance.
    rounds : int
        Syndrome measurement rounds.
    error_prob : float
        Physical error probability.
    num_shots : int
        Total available shots.
    circuit_file : str
        Relative path to source circuit file.
    manifest : dict
        Full manifest contents.
    """

    syndromes: NDArray[np.uint8]
    observables: NDArray[np.uint8]
    detector_coords: np.ndarray
    distance: int
    rounds: int
    error_prob: float
    num_shots: int
    circuit_file: str
    manifest: dict


@dataclass(frozen=True, slots=True)
class DecoderPointResult:
    """Per-decoder results for one (d, p) evaluation point.

    Parameters
    ----------
    decoder_name : str
        Decoder identifier.
    n_shots : int
        Number of shots processed.
    n_errors : int
        Number of logical errors (shot-level).
    ler : float
        Per-shot logical error rate.
    ler_interval : WilsonInterval
        Wilson 95% CI for per-shot LER.
    per_round_ler : float
        Per-round logical error rate (epsilon).
    per_round_interval : WilsonInterval
        Wilson 95% CI for per-round LER.
    correct : NDArray[np.bool_]
        Per-shot correctness vector (True = correct decode).
    """

    decoder_name: str
    n_shots: int
    n_errors: int
    ler: float
    ler_interval: WilsonInterval
    per_round_ler: float
    per_round_interval: WilsonInterval
    correct: NDArray[np.bool_]


@dataclass(frozen=True, slots=True)
class EvalPointResult:
    """Complete evaluation result for one (d, p) point.

    Parameters
    ----------
    distance : int
        Code distance.
    rounds : int
        Syndrome measurement rounds.
    error_prob : float
        Physical error probability.
    n_shots_used : int
        Shots actually processed (may be less than available due to early stop).
    decoder_results : dict[str, DecoderPointResult]
        Per-decoder results keyed by decoder name.
    mcnemar_results : dict[str, McNemarResult]
        McNemar test results for GNN vs each baseline, keyed by baseline name.
    stopping : StoppingDecision
        Final stopping decision.
    outcome : EvalOutcome
        Final evaluation outcome for this point.
    """

    distance: int
    rounds: int
    error_prob: float
    n_shots_used: int
    decoder_results: dict[str, DecoderPointResult]
    mcnemar_results: dict[str, McNemarResult]
    stopping: StoppingDecision
    outcome: EvalOutcome


@dataclass(slots=True)
class EvalReport:
    """Full evaluation report across all (d, p) points.

    Parameters
    ----------
    results : list[EvalPointResult]
        Per-point results.
    metadata : dict
        Report-level metadata (decoder names, config, etc).
    """

    results: list[EvalPointResult] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Serialize to JSON-compatible dict."""
        return {
            "metadata": self.metadata,
            "points": [_point_to_dict(r) for r in self.results],
        }

    def save(self, path: Path) -> None:
        """Write report as JSON."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# Eval set loading
# ---------------------------------------------------------------------------


def load_eval_set(eval_dir: Path) -> EvalSet:
    """Load a frozen evaluation set from a directory.

    Supports two formats:
    - Compressed npz (``data.npz`` with keys: syndromes, observables,
      detector_coords) + ``manifest.json``
    - Individual npy files (``syndromes.npy``, ``observables.npy``,
      ``detector_coords.npy``) + ``manifest.json``

    Parameters
    ----------
    eval_dir : Path
        Directory containing the eval set.

    Returns
    -------
    EvalSet

    Raises
    ------
    FileNotFoundError
        If required files are missing.
    ValueError
        If manifest is missing required fields or shapes mismatch.
    """
    manifest_path = eval_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"No manifest.json in eval set directory: {eval_dir}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    npz_path = eval_dir / "data.npz"
    if npz_path.exists():
        data = np.load(npz_path)
        syndromes = data["syndromes"].astype(np.uint8, copy=False)
        observables = data["observables"].astype(np.uint8, copy=False)
        detector_coords = data["detector_coords"].astype(np.float64, copy=False)
    else:
        syn_path = eval_dir / "syndromes.npy"
        obs_path = eval_dir / "observables.npy"
        coords_path = eval_dir / "detector_coords.npy"

        if not syn_path.exists():
            raise FileNotFoundError(f"No syndromes file in {eval_dir}")

        syndromes = np.load(syn_path).astype(np.uint8, copy=False)
        observables = np.load(obs_path).astype(np.uint8, copy=False)
        detector_coords = np.load(coords_path).astype(np.float64, copy=False)

    if observables.ndim == 1:
        observables = observables[:, np.newaxis]

    _required_fields = ["distance", "rounds", "error_prob", "circuit_file"]
    for f in _required_fields:
        if f not in manifest:
            raise ValueError(f"Manifest missing required field '{f}': {manifest_path}")

    return EvalSet(
        syndromes=syndromes,
        observables=observables,
        detector_coords=detector_coords,
        distance=manifest["distance"],
        rounds=manifest["rounds"],
        error_prob=manifest["error_prob"],
        num_shots=syndromes.shape[0],
        circuit_file=manifest["circuit_file"],
        manifest=manifest,
    )


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------


def evaluate_point(
    eval_set: EvalSet,
    decoders: dict[str, Decoder],
    reference_decoder: str,
    *,
    stopping_baseline: str | None = None,
    check_interval: int = CHECK_INTERVAL,
) -> EvalPointResult:
    """Evaluate all decoders on a single (d, p) point with adaptive stopping.

    Processes shots in increments of ``check_interval``. At each checkpoint,
    evaluates the Haybittle-Peto stopping criterion on the reference decoder
    vs ``stopping_baseline`` disagreement matrix only. McNemar results are
    computed for all comparison pairs at the end regardless of which pair
    drove the stopping decision.

    Parameters
    ----------
    eval_set : EvalSet
        Frozen evaluation set to decode.
    decoders : dict[str, Decoder]
        Decoders to evaluate, keyed by name.
    reference_decoder : str
        Name of the reference decoder (GNN in typical usage).
        Must be a key in ``decoders``.
    stopping_baseline : str or None
        Comparison decoder whose McNemar test drives adaptive stopping.
        If None, defaults to the first non-reference decoder. Only this
        pair is checked for the Haybittle-Peto boundary; other decoders
        are evaluated on the same shots but do not influence stopping.
    check_interval : int
        Shots between adaptive stopping checks.

    Returns
    -------
    EvalPointResult

    Raises
    ------
    ValueError
        If reference_decoder not in decoders, fewer than 2 decoders, or
        stopping_baseline not in decoders.
    """
    if reference_decoder not in decoders:
        raise ValueError(
            f"reference_decoder '{reference_decoder}' not in decoders: "
            f"{list(decoders.keys())}"
        )
    if len(decoders) < 2:
        raise ValueError("Need at least 2 decoders for paired comparison")

    # Resolve which comparison decoder drives stopping
    comparison_decoders = [k for k in decoders if k != reference_decoder]
    if stopping_baseline is None:
        stopping_baseline = comparison_decoders[0]
    elif stopping_baseline not in decoders:
        raise ValueError(
            f"stopping_baseline '{stopping_baseline}' not in decoders: "
            f"{list(decoders.keys())}"
        )
    elif stopping_baseline == reference_decoder:
        raise ValueError(
            f"stopping_baseline must differ from reference_decoder "
            f"(both are '{reference_decoder}')"
        )

    n_total = eval_set.num_shots
    n_obs = eval_set.observables.shape[1]

    # Preallocate correctness arrays
    correct_arrays: dict[str, NDArray[np.bool_]] = {
        name: np.empty(n_total, dtype=np.bool_) for name in decoders
    }

    # Determine check points (ceiling division to include tail shots)
    n_checks = max(1, -(-n_total // check_interval))
    shots_processed = 0
    stopping: StoppingDecision | None = None

    for check_idx in range(n_checks):
        start = shots_processed
        end = min(start + check_interval, n_total)
        chunk_syndromes = eval_set.syndromes[start:end]
        chunk_observables = eval_set.observables[start:end]

        for name, decoder in decoders.items():
            predictions = decoder.decode_batch(chunk_syndromes)
            if predictions.ndim == 1:
                predictions = predictions[:, np.newaxis]
            predictions = predictions[:, :n_obs]

            shot_correct = np.all(predictions == chunk_observables, axis=1)
            correct_arrays[name][start:end] = shot_correct

        shots_processed = end
        is_final = end >= n_total

        # Adaptive stopping checks only the primary comparison pair
        stopping = adaptive_stop(
            correct_arrays[reference_decoder][:shots_processed],
            correct_arrays[stopping_baseline][:shots_processed],
            is_final=is_final,
        )
        if stopping.action == "stop":
            break

    # If we ran all shots without a stopping decision
    if stopping is None or (
        stopping.action == "continue" and shots_processed >= n_total
    ):
        stopping = adaptive_stop(
            correct_arrays[reference_decoder][:shots_processed],
            correct_arrays[stopping_baseline][:shots_processed],
            is_final=True,
        )

    # Build per-decoder results
    decoder_results: dict[str, DecoderPointResult] = {}
    for name in decoders:
        correct_slice = correct_arrays[name][:shots_processed]
        n_errors = int(np.sum(~correct_slice))
        ler = n_errors / shots_processed if shots_processed > 0 else 0.0
        ler_ci = wilson_interval(n_errors, shots_processed)

        eps = per_round_ler(ler, eval_set.rounds)
        # Per-round CI: transform the per-shot CI bounds
        eps_lower = per_round_ler(ler_ci.lower, eval_set.rounds)
        eps_upper = per_round_ler(ler_ci.upper, eval_set.rounds)
        eps_interval = WilsonInterval(
            lower=eps_lower,
            upper=eps_upper,
            point=eps,
            n_errors=n_errors,
            n_total=shots_processed,
            alpha=0.05,
        )

        decoder_results[name] = DecoderPointResult(
            decoder_name=name,
            n_shots=shots_processed,
            n_errors=n_errors,
            ler=ler,
            ler_interval=ler_ci,
            per_round_ler=eps,
            per_round_interval=eps_interval,
            correct=correct_slice,
        )

    # McNemar results for reference decoder vs each other decoder
    mcnemar_results: dict[str, McNemarResult] = {}
    ref_correct = correct_arrays[reference_decoder][:shots_processed]
    for comp_name in comparison_decoders:
        comp_correct = correct_arrays[comp_name][:shots_processed]
        mcnemar_results[comp_name] = mcnemar_test(ref_correct, comp_correct)

    outcome = (
        stopping.outcome if stopping.outcome is not None else EvalOutcome.UNRESOLVED
    )

    return EvalPointResult(
        distance=eval_set.distance,
        rounds=eval_set.rounds,
        error_prob=eval_set.error_prob,
        n_shots_used=shots_processed,
        decoder_results=decoder_results,
        mcnemar_results=mcnemar_results,
        stopping=stopping,
        outcome=outcome,
    )


# ---------------------------------------------------------------------------
# Eval set discovery
# ---------------------------------------------------------------------------


def discover_eval_sets(
    eval_dir: Path,
    *,
    distances: list[int] | None = None,
    error_probs: list[float] | None = None,
) -> list[Path]:
    """Discover eval set directories matching the (d, p) naming convention.

    Parameters
    ----------
    eval_dir : Path
        Root eval directory containing ``d{d}_p{p_str}/`` subdirs.
    distances : list[int] or None
        Filter to these distances. None = all.
    error_probs : list[float] or None
        Filter to these error probabilities. None = all.

    Returns
    -------
    list[Path]
        Sorted list of eval set directories.
    """
    dirs: list[Path] = []
    if not eval_dir.is_dir():
        return dirs

    for sub in sorted(eval_dir.iterdir()):
        if not sub.is_dir():
            continue
        manifest_path = sub / "manifest.json"
        if not manifest_path.exists():
            continue

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        d = manifest.get("distance")
        p = manifest.get("error_prob")

        if distances is not None and d not in distances:
            continue
        if error_probs is not None and p not in error_probs:
            continue

        dirs.append(sub)

    return dirs


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------


def _point_to_dict(result: EvalPointResult) -> dict:
    """Serialize an EvalPointResult to a JSON-compatible dict."""
    decoders = {}
    for name, dr in result.decoder_results.items():
        decoders[name] = {
            "n_shots": dr.n_shots,
            "n_errors": dr.n_errors,
            "ler": dr.ler,
            "ler_ci_95": [dr.ler_interval.lower, dr.ler_interval.upper],
            "per_round_ler": dr.per_round_ler,
            "per_round_ler_ci_95": [
                dr.per_round_interval.lower,
                dr.per_round_interval.upper,
            ],
        }

    mcnemar = {}
    for name, mr in result.mcnemar_results.items():
        mcnemar[name] = {
            "statistic": mr.statistic,
            "p_value": mr.p_value,
            "n_discordant": mr.n_discordant,
            "gnn_wins": mr.gnn_wins,
            "baseline_wins": mr.baseline_wins,
        }

    return {
        "distance": result.distance,
        "rounds": result.rounds,
        "error_prob": result.error_prob,
        "n_shots_used": result.n_shots_used,
        "outcome": result.outcome.value,
        "stopping_reason": result.stopping.reason,
        "decoders": decoders,
        "mcnemar": mcnemar,
    }
