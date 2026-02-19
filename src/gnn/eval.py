"""
Evaluation of trained GNN decoders.

Computes per-setting logical error rates (LER) for a trained checkpoint,
with optional comparison against MWPM baseline results.

Three evaluation protocols match the three training cases:

- ``logical_head``: threshold graph-level logits at 0 => observable flip.
- ``mwpm_teacher``: convert per-edge logits to MWPM weights => decode
  with PyMatching => LER.  Also reports edge accuracy for diagnostics.
- ``hybrid``: identical to ``mwpm_teacher`` (same architecture, same
  eval path).
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pymatching
import torch
from torch_geometric.data import Batch
from tqdm import tqdm

from gnn.models.heads import Case, QECDecoder, build_model


logger = logging.getLogger(__name__)


@dataclass
class SettingResult:
    """
    Evaluation result for a single (distance, rounds, error_prob) setting.

    Attributes
    ----------
    distance : int
        Code distance.
    rounds : int
        Number of syndrome measurement rounds.
    error_prob : float
        Physical error probability.
    split : str
        Evaluated split name.
    num_shots : int
        Total number of shots evaluated.
    num_errors : int
        Shots where the decoder predicted the wrong observable.
    logical_error_rate : float
        ``num_errors / num_shots``.
    elapsed_s : float
        Wall-clock seconds for this setting.
    mwpm_ler : float or None
        MWPM baseline LER for comparison (if loaded).
    edge_acc : float or None
        Per-edge accuracy (edge cases only).
    """

    distance: int
    rounds: int
    error_prob: float
    split: str
    num_shots: int
    num_errors: int
    logical_error_rate: float
    elapsed_s: float
    mwpm_ler: float | None = None
    edge_acc: float | None = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to JSON-friendly dictionary."""
        d: Dict[str, Any] = {
            "distance": self.distance,
            "rounds": self.rounds,
            "error_prob": self.error_prob,
            "split": self.split,
            "num_shots": self.num_shots,
            "num_errors": self.num_errors,
            "logical_error_rate": self.logical_error_rate,
            "elapsed_s": round(self.elapsed_s, 4),
        }
        if self.mwpm_ler is not None:
            d["mwpm_ler"] = self.mwpm_ler
        if self.edge_acc is not None:
            d["edge_acc"] = self.edge_acc
        return d


@dataclass
class EvalReport:
    """
    Aggregated GNN evaluation report.

    Attributes
    ----------
    results : list of SettingResult
        Per-setting results.
    metadata : dict
        Run-level metadata (checkpoint, case, device, etc.).
    """

    results: List[SettingResult] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the full report."""
        return {
            "metadata": self.metadata,
            "results": [r.to_dict() for r in self.results],
        }


def load_model(
    checkpoint: Path,
    device: torch.device,
) -> tuple[QECDecoder, Dict[str, Any]]:
    """
    Load a trained QECDecoder from a checkpoint.

    Parameters
    ----------
    checkpoint : Path
        Path to the ``best.pt`` checkpoint file.
    device : torch.device
        Device to load the model onto.

    Returns
    -------
    model : QECDecoder
        Model in eval mode.
    config : dict
        Training config stored in the checkpoint.
    """
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    cfg = ckpt["config"]

    model = build_model(
        cfg["case"],
        hidden_dim=cfg["hidden_dim"],
        num_layers=cfg["num_layers"],
        dropout=cfg.get("dropout", 0.0),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()

    logger.info(
        "Loaded model: case=%s, hidden_dim=%d, num_layers=%d (epoch %d)",
        cfg["case"],
        cfg["hidden_dim"],
        cfg["num_layers"],
        ckpt.get("epoch", -1),
    )
    return model, cfg


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


@dataclass
class _SettingMeta:
    """Internal metadata for one setting."""

    setting_id: int
    relpath: str
    distance: int
    rounds: int
    error_prob: float
    num_nodes: int
    num_detectors: int
    num_observables: int
    has_boundary: bool


def _load_settings(datasets_dir: Path, case: str) -> List[_SettingMeta]:
    """
    Load setting metadata from settings.json.

    Parameters
    ----------
    datasets_dir : Path
        Root datasets directory.
    case : str
        Training case name.

    Returns
    -------
    list of _SettingMeta
        Per-setting metadata.
    """
    case_root = datasets_dir / case
    settings_obj = _read_json(case_root / "settings.json")

    metas = []
    for s in settings_obj["settings"]:
        g = s["graph"]
        metas.append(
            _SettingMeta(
                setting_id=int(s["setting_id"]),
                relpath=str(s["relpath"]),
                distance=int(s["distance"]),
                rounds=int(s["rounds"]),
                error_prob=float(s["error_prob"]),
                num_nodes=int(g["num_nodes"]),
                num_detectors=int(g["num_detectors"]),
                num_observables=int(g["num_observables"]),
                has_boundary=bool(g.get("has_boundary", False)),
            )
        )
    return metas


def _load_setting_data(
    datasets_dir: Path,
    case: str,
    meta: _SettingMeta,
    split: str,
) -> tuple[torch.Tensor, torch.Tensor, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load graph and split arrays for one setting.

    Parameters
    ----------
    datasets_dir : Path
        Root datasets directory.
    case : str
        Training case name.
    meta : _SettingMeta
        Setting metadata.
    split : str
        Data split name.

    Returns
    -------
    edge_index : Tensor, shape (2, E)
    edge_attr : Tensor, shape (E, 2)
    syndrome : ndarray, shape (N_shots, num_detectors), uint8
    logical : ndarray, shape (N_shots, num_observables), uint8
    edge_index_np : ndarray, shape (2, E), int64
        NumPy version for undirected edge mapping in hybrid eval.
    """
    shard = datasets_dir / case / "shards" / meta.relpath
    graph_dir = shard / "graph"

    ei_np = np.load(graph_dir / "edge_index.npy").astype(np.int64)
    ea_np = np.load(graph_dir / "edge_attr.npy").astype(np.float32)

    edge_index = torch.from_numpy(ei_np).long()
    edge_attr = torch.from_numpy(ea_np).float()

    syndrome = np.load(shard / f"{split}_syndrome.npy").astype(np.uint8)
    logical = np.load(shard / f"{split}_logical.npy").astype(np.uint8)
    if logical.ndim == 1:
        logical = logical[:, np.newaxis]

    return edge_index, edge_attr, syndrome, logical, ei_np


def _build_batch(
    syndrome: np.ndarray,
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
    num_nodes: int,
    num_detectors: int,
    start: int,
    end: int,
) -> Batch:
    """
    Build a PyG Batch from a slice of syndrome shots (vectorized).

    All shots share the same graph structure; only node features differ.
    Constructs the batch directly with tensor ops instead of iterating
    over individual Data objects.

    Parameters
    ----------
    syndrome : ndarray, shape (N, num_detectors)
        Full syndrome array.
    edge_index : Tensor, shape (2, E)
        Graph edges (shared across shots).
    edge_attr : Tensor, shape (E, 2)
        Edge attributes (shared across shots).
    num_nodes : int
        Total node count (detectors + boundary).
    num_detectors : int
        Detector node count.
    start, end : int
        Slice indices into the syndrome array.

    Returns
    -------
    Batch
        Batched PyG graphs ready for model forward.
    """
    n = end - start

    # Node features: (n, num_detectors) => pad => (n, num_nodes) => (n*num_nodes, 1)
    det = torch.from_numpy(
        syndrome[start:end].astype(np.float32, copy=False)
    )  # (n, num_detectors)
    if num_nodes > num_detectors:
        pad = torch.zeros(n, num_nodes - num_detectors, dtype=torch.float32)
        det = torch.cat([det, pad], dim=1)  # (n, num_nodes)
    x = det.reshape(-1, 1)  # (n * num_nodes, 1)

    # Replicate edge_index with per-graph node offsets
    offsets = torch.arange(n, dtype=torch.long).unsqueeze(1) * num_nodes  # (n, 1)
    ei_rep = edge_index.unsqueeze(0).expand(n, -1, -1) + offsets.unsqueeze(1)
    ei_batch = ei_rep.permute(1, 0, 2).reshape(2, -1)  # (2, n * E)

    # Replicate edge_attr
    ea_batch = edge_attr.unsqueeze(0).expand(n, -1, -1).reshape(-1, edge_attr.shape[1])

    # Batch vector
    batch_vec = torch.arange(n, dtype=torch.long).repeat_interleave(num_nodes)

    return Batch(
        x=x,
        edge_index=ei_batch,
        edge_attr=ea_batch,
        batch=batch_vec,
    )


@torch.no_grad()
def _eval_logical_head(
    model: QECDecoder,
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
    syndrome: np.ndarray,
    logical: np.ndarray,
    num_nodes: int,
    num_detectors: int,
    device: torch.device,
    batch_size: int = 256,
) -> tuple[int, int]:
    """
    Evaluate logical_head: direct observable prediction.

    Parameters
    ----------
    model : QECDecoder
        Trained model in eval mode.
    edge_index, edge_attr : Tensor
        Graph structure.
    syndrome : ndarray, shape (N, D)
        Syndrome vectors.
    logical : ndarray, shape (N, O)
        Ground-truth observables.
    num_nodes, num_detectors : int
        Node counts.
    device : torch.device
        Compute device.
    batch_size : int
        Shots per forward pass.

    Returns
    -------
    num_shots : int
    num_errors : int
    """
    n = syndrome.shape[0]
    num_errors = 0

    for off in range(0, n, batch_size):
        end = min(off + batch_size, n)
        batch = _build_batch(
            syndrome,
            edge_index,
            edge_attr,
            num_nodes,
            num_detectors,
            off,
            end,
        )
        batch = batch.to(device)

        logits = model(batch)  # (B, num_obs)
        pred = (logits > 0.0).cpu().numpy().astype(np.uint8)

        target = logical[off:end]
        errors = np.any(pred != target, axis=1)
        num_errors += int(errors.sum())

    return n, num_errors


def _undirected_edges(edge_index: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (unique undirected pairs, directed=>undirected index map)."""
    src, dst = edge_index[0], edge_index[1]
    pairs = np.stack([np.minimum(src, dst), np.maximum(src, dst)], axis=1)
    unique, inverse = np.unique(pairs, axis=0, return_inverse=True)
    return unique.astype(np.int64), inverse.astype(np.int64)


def _build_matching_from_weights(
    und_pairs: np.ndarray,
    weights: np.ndarray,
    num_detectors: int,
    has_boundary: bool,
) -> pymatching.Matching:
    """
    Build PyMatching decoder from predicted undirected edge weights.

    Parameters
    ----------
    und_pairs : ndarray, shape (U, 2)
        Undirected edge endpoint pairs.
    weights : ndarray, shape (U,)
        Predicted MWPM weights per undirected edge.
    num_detectors : int
        Number of detector nodes.
    has_boundary : bool
        Whether the graph includes a boundary node.

    Returns
    -------
    pymatching.Matching
        Decoder configured with predicted weights.
    """
    boundary_node = num_detectors if has_boundary else None
    m = pymatching.Matching()

    for eid in range(und_pairs.shape[0]):
        u, v = int(und_pairs[eid, 0]), int(und_pairs[eid, 1])
        w = float(max(weights[eid], 1e-8))  # clamp to avoid zero weight

        if boundary_node is not None and boundary_node in (u, v):
            det = v if u == boundary_node else u
            if 0 <= det < num_detectors:
                m.add_boundary_edge(det, weight=w)
        elif u < num_detectors and v < num_detectors:
            m.add_edge(u, v, weight=w)

    return m


def _logits_to_weights(logits: np.ndarray) -> np.ndarray:
    """Logit => sigmoid => log-likelihood ratio, clamped positive."""
    prob = 1.0 / (1.0 + np.exp(-logits))
    prob = np.clip(prob, 1e-7, 1.0 - 1e-7)
    weights = np.log((1.0 - prob) / prob)
    return np.maximum(weights, 1e-8)


def _directed_to_undirected_logits(
    directed_logits: np.ndarray,
    dir_to_undir: np.ndarray,
    num_undirected: int,
) -> np.ndarray:
    """Average directed edge logits into per-undirected-edge means."""
    logit_sum = np.zeros(num_undirected, dtype=np.float64)
    logit_count = np.zeros(num_undirected, dtype=np.int32)
    np.add.at(logit_sum, dir_to_undir, directed_logits)
    np.add.at(logit_count, dir_to_undir, 1)
    return logit_sum / np.maximum(logit_count, 1)


@torch.no_grad()
def _eval_edge_case(
    model: QECDecoder,
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
    edge_index_np: np.ndarray,
    syndrome: np.ndarray,
    logical: np.ndarray,
    num_nodes: int,
    num_detectors: int,
    has_boundary: bool,
    num_observables: int,
    device: torch.device,
    batch_size: int = 256,
) -> tuple[int, int, float]:
    """
    Evaluate edge case: per-shot GNN weights => PyMatching => LER.

    For each shot, the model produces syndrome-conditional edge logits.
    These are converted to MWPM weights and decoded individually,
    preserving the model's ability to produce different weights for
    different syndromes.

    Parameters
    ----------
    model : QECDecoder
        Trained model in eval mode.
    edge_index, edge_attr : Tensor
        Graph structure.
    edge_index_np : ndarray
        NumPy edge index for undirected mapping.
    syndrome : ndarray, shape (N, D)
        Syndrome vectors.
    logical : ndarray, shape (N, O)
        Ground-truth observables.
    num_nodes, num_detectors : int
        Node counts.
    has_boundary : bool
        Whether graph has boundary node.
    num_observables : int
        Number of logical observables.
    device : torch.device
        Compute device.
    batch_size : int
        Shots per forward pass.

    Returns
    -------
    num_shots : int
    num_errors : int
    edge_acc : float
        NaN (teacher labels not available in this eval path).
    """
    n = syndrome.shape[0]
    edges_per_graph = edge_index_np.shape[1]
    und_pairs, dir_to_undir = _undirected_edges(edge_index_np)
    num_und = und_pairs.shape[0]

    predicted = np.empty((n, num_observables), dtype=np.uint8)

    # Cache: weight tuple => Matching object (avoids rebuilding for
    # identical weight vectors, which happen when syndromes produce
    # similar logit patterns).
    matching_cache: Dict[bytes, pymatching.Matching] = {}

    for off in range(0, n, batch_size):
        end = min(off + batch_size, n)
        batch = _build_batch(
            syndrome,
            edge_index,
            edge_attr,
            num_nodes,
            num_detectors,
            off,
            end,
        )
        batch = batch.to(device)

        logits = model(batch)  # (E_total,)
        logits_np = logits.cpu().numpy()

        num_graphs = end - off

        for g in range(num_graphs):
            start_e = g * edges_per_graph
            end_e = start_e + edges_per_graph
            graph_logits = logits_np[start_e:end_e]

            # Directed => undirected mean => weights
            und_logits = _directed_to_undirected_logits(
                graph_logits, dir_to_undir, num_und
            )
            weights = _logits_to_weights(und_logits)

            # Cache matching by weight vector bytes
            cache_key = weights.tobytes()
            if cache_key not in matching_cache:
                matching_cache[cache_key] = _build_matching_from_weights(
                    und_pairs, weights, num_detectors, has_boundary
                )
            matching = matching_cache[cache_key]

            shot_syn = syndrome[off + g]
            pred = matching.decode(shot_syn)
            predicted[off + g] = pred[:num_observables]

    shot_errors = np.any(predicted != logical, axis=1)
    num_errors = int(shot_errors.sum())

    return n, num_errors, float("nan")


def evaluate_all(
    checkpoint: Path,
    datasets_dir: Path,
    split: str = "test",
    baseline_path: Path | None = None,
    batch_size: int = 256,
) -> EvalReport:
    """
    Evaluate a trained GNN decoder on all settings.

    Parameters
    ----------
    checkpoint : Path
        Path to the model checkpoint.
    datasets_dir : Path
        Root datasets directory.
    split : str
        Split to evaluate.
    baseline_path : Path or None
        Path to MWPM baseline JSON report for comparison.
    batch_size : int
        Shots per forward pass during inference.

    Returns
    -------
    EvalReport
        Full evaluation report with per-setting results.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg = load_model(checkpoint, device)
    case: Case = cfg["case"]

    settings = _load_settings(datasets_dir, case)
    logger.info(
        "Evaluating %s on %d settings (split=%s, device=%s)",
        case,
        len(settings),
        split,
        device,
    )

    # Load MWPM baseline if provided
    baseline: Dict[tuple, float] = {}
    if baseline_path is not None and baseline_path.is_file():
        bl = _read_json(baseline_path)
        for r in bl.get("results", []):
            key = (r["distance"], r["rounds"], r["error_prob"], r["split"])
            baseline[key] = r["logical_error_rate"]
        logger.info("Loaded MWPM baseline: %d entries", len(baseline))

    report = EvalReport(
        metadata={
            "decoder": f"gnn_{case}",
            "checkpoint": str(checkpoint),
            "case": case,
            "split": split,
            "hidden_dim": cfg["hidden_dim"],
            "num_layers": cfg["num_layers"],
            "device": str(device),
        }
    )

    for meta in tqdm(settings, desc="GNN eval", unit="setting"):
        t0 = time.perf_counter()

        try:
            edge_index, edge_attr, syndrome, logical, ei_np = _load_setting_data(
                datasets_dir, case, meta, split
            )
        except FileNotFoundError as exc:
            logger.warning(
                "Skipping d=%d r=%d p=%s: %s",
                meta.distance,
                meta.rounds,
                meta.error_prob,
                exc,
            )
            continue

        if case == "logical_head":
            num_shots, num_errors = _eval_logical_head(
                model,
                edge_index,
                edge_attr,
                syndrome,
                logical,
                meta.num_nodes,
                meta.num_detectors,
                device,
                batch_size,
            )
            edge_acc = None
        else:
            num_shots, num_errors, edge_acc = _eval_edge_case(
                model,
                edge_index,
                edge_attr,
                ei_np,
                syndrome,
                logical,
                meta.num_nodes,
                meta.num_detectors,
                meta.has_boundary,
                meta.num_observables,
                device,
                batch_size,
            )

        elapsed = time.perf_counter() - t0
        ler = num_errors / num_shots if num_shots > 0 else 0.0

        mwpm_ler = baseline.get((meta.distance, meta.rounds, meta.error_prob, split))

        report.results.append(
            SettingResult(
                distance=meta.distance,
                rounds=meta.rounds,
                error_prob=meta.error_prob,
                split=split,
                num_shots=num_shots,
                num_errors=num_errors,
                logical_error_rate=ler,
                elapsed_s=elapsed,
                mwpm_ler=mwpm_ler,
                edge_acc=edge_acc,
            )
        )

    return report


def print_report(report: EvalReport) -> None:
    """
    Log a formatted summary table.

    Parameters
    ----------
    report : EvalReport
        Evaluation report to display.
    """
    results = report.results
    if not results:
        logger.warning("No results to display")
        return

    has_baseline = any(r.mwpm_ler is not None for r in results)
    case = report.metadata.get("case", "unknown")

    if has_baseline:
        header = (
            f"{'d':>3} {'r':>4} {'p':>9} "
            f"{'shots':>7} {'GNN_LER':>10} {'MWPM_LER':>10} {'delta':>8}"
        )
    else:
        header = f"{'d':>3} {'r':>4} {'p':>9} " f"{'shots':>7} {'GNN_LER':>10}"

    sep = "-" * len(header)
    lines: list[str] = []

    lines.append("")
    lines.append(sep)
    lines.append(f"GNN Evaluation — {case}")
    lines.append(sep)
    lines.append(header)
    lines.append(sep)

    sorted_results = sorted(results, key=lambda r: (r.distance, r.rounds, r.error_prob))

    prev_d = None
    total_gnn_better = 0
    total_compared = 0

    for r in sorted_results:
        if prev_d is not None and r.distance != prev_d:
            lines.append("")
        prev_d = r.distance

        if has_baseline and r.mwpm_ler is not None:
            delta = r.logical_error_rate - r.mwpm_ler
            marker = " +" if delta > 0 else " "
            if delta < -1e-9:
                marker = " *"  # GNN is better
                total_gnn_better += 1
            total_compared += 1
            lines.append(
                f"{r.distance:>3} {r.rounds:>4} {r.error_prob:>9.5f} "
                f"{r.num_shots:>7} {r.logical_error_rate:>10.6f} "
                f"{r.mwpm_ler:>10.6f} {delta:>+8.4f}{marker}"
            )
        else:
            lines.append(
                f"{r.distance:>3} {r.rounds:>4} {r.error_prob:>9.5f} "
                f"{r.num_shots:>7} {r.logical_error_rate:>10.6f}"
            )

    lines.append(sep)

    # Summary
    if has_baseline and total_compared > 0:
        lines.append(
            f"\nGNN better in {total_gnn_better}/{total_compared} settings "
            f"(marked with *)"
        )

    # Per-distance summary
    lines.append("\nSummary by distance:")
    for d in sorted(set(r.distance for r in results)):
        subset = [r for r in results if r.distance == d]
        lers = [r.logical_error_rate for r in subset]
        line = (
            f"  d={d}: "
            f"min_LER={min(lers):.6f}  "
            f"max_LER={max(lers):.6f}  "
            f"mean_LER={np.mean(lers):.6f}"
        )
        if has_baseline:
            bl_lers = [r.mwpm_ler for r in subset if r.mwpm_ler is not None]
            if bl_lers:
                line += f"  (MWPM mean={np.mean(bl_lers):.6f})"
        lines.append(line)

    lines.append("")

    output = "\n".join(lines)
    print(output)


def save_report(report: EvalReport, path: Path) -> None:
    """
    Save the evaluation report to a JSON file.

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
