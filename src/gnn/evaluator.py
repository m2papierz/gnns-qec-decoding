"""
Evaluator for trained GNN decoders.

Computes per-setting logical error rates (LER) for a trained checkpoint,
with optional comparison against MWPM baseline results.

Three evaluation protocols match the three training cases:

- ``logical_head``: threshold graph-level logits at 0 => observable flip.
- ``mwpm_teacher`` / ``hybrid``: convert per-edge logits to MWPM
  weights => decode with PyMatching => LER.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pymatching
import torch
from torch_geometric.data import Batch
from tqdm import tqdm

from gnn.metrics import EvalReport, SettingResult
from gnn.models.heads import Case, QECDecoder, build_model
from qec_generator.utils import read_json, undirected_edges


logger = logging.getLogger(__name__)


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


def _build_matching_from_weights(
    und_pairs: np.ndarray,
    weights: np.ndarray,
    num_detectors: int,
    has_boundary: bool,
) -> pymatching.Matching:
    """Build PyMatching decoder from predicted undirected edge weights."""
    boundary_node = num_detectors if has_boundary else None
    m = pymatching.Matching()

    for eid in range(und_pairs.shape[0]):
        u, v = int(und_pairs[eid, 0]), int(und_pairs[eid, 1])
        w = float(max(weights[eid], 1e-8))

        if boundary_node is not None and boundary_node in (u, v):
            det = v if u == boundary_node else u
            if 0 <= det < num_detectors:
                m.add_boundary_edge(det, weight=w)
        elif u < num_detectors and v < num_detectors:
            m.add_edge(u, v, weight=w)

    return m


@dataclass(frozen=True)
class _SettingMeta:
    """Metadata for one (distance, rounds, error_prob) setting."""

    setting_id: int
    relpath: str
    distance: int
    rounds: int
    error_prob: float
    num_nodes: int
    num_detectors: int
    num_observables: int
    has_boundary: bool


class Evaluator:
    """
    Evaluate a trained GNN decoder across all settings.

    Parameters
    ----------
    checkpoint : Path
        Path to the ``best.pt`` checkpoint file.
    datasets_dir : Path
        Root datasets directory.
    split : str
        Data split to evaluate (default: ``"test"``).
    baseline_path : Path or None
        Path to MWPM baseline JSON report for comparison.
    batch_size : int
        Shots per forward pass during inference.
    """

    def __init__(
        self,
        checkpoint: Path,
        datasets_dir: Path,
        split: str = "test",
        baseline_path: Path | None = None,
        batch_size: int = 256,
    ) -> None:
        self.checkpoint = checkpoint
        self.datasets_dir = datasets_dir
        self.split = split
        self.baseline_path = baseline_path
        self.batch_size = batch_size

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model, self.cfg = self._load_model()
        self.case: Case = self.cfg["case"]

    def _load_model(self) -> tuple[QECDecoder, Dict[str, Any]]:
        """Load model from checkpoint."""
        ckpt = torch.load(
            self.checkpoint, map_location=self.device, weights_only=False
        )
        cfg = ckpt["config"]

        model = build_model(
            cfg["case"],
            hidden_dim=cfg["hidden_dim"],
            num_layers=cfg["num_layers"],
            dropout=cfg.get("dropout", 0.0),
        )
        model.load_state_dict(ckpt["model_state_dict"])
        model = model.to(self.device)
        model.eval()

        logger.info(
            "Loaded model: case=%s, hidden_dim=%d, num_layers=%d (epoch %d)",
            cfg["case"],
            cfg["hidden_dim"],
            cfg["num_layers"],
            ckpt.get("epoch", -1),
        )
        return model, cfg

    def _load_settings(self) -> List[_SettingMeta]:
        """Load setting metadata from settings.json."""
        case_root = self.datasets_dir / self.case
        settings_obj = read_json(case_root / "settings.json")

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

    def _load_baseline(self) -> Dict[tuple, float]:
        """Load MWPM baseline LER values if available."""
        if self.baseline_path is None or not self.baseline_path.is_file():
            return {}

        bl = read_json(self.baseline_path)
        baseline: Dict[tuple, float] = {}
        for r in bl.get("results", []):
            key = (r["distance"], r["rounds"], r["error_prob"], r["split"])
            baseline[key] = r["logical_error_rate"]
        logger.info("Loaded MWPM baseline: %d entries", len(baseline))
        return baseline

    def _load_setting_data(
        self, meta: _SettingMeta
    ) -> tuple[torch.Tensor, torch.Tensor, np.ndarray, np.ndarray, np.ndarray]:
        """Load graph and split arrays for one setting."""
        shard = self.datasets_dir / self.case / "shards" / meta.relpath
        graph_dir = shard / "graph"

        ei_np = np.load(graph_dir / "edge_index.npy").astype(np.int64)
        ea_np = np.load(graph_dir / "edge_attr.npy").astype(np.float32)

        edge_index = torch.from_numpy(ei_np).long()
        edge_attr = torch.from_numpy(ea_np).float()

        syndrome = np.load(
            shard / f"{self.split}_syndrome.npy"
        ).astype(np.uint8)
        logical = np.load(
            shard / f"{self.split}_logical.npy"
        ).astype(np.uint8)
        if logical.ndim == 1:
            logical = logical[:, np.newaxis]

        return edge_index, edge_attr, syndrome, logical, ei_np

    @staticmethod
    def _build_batch(
        syndrome: np.ndarray,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        num_nodes: int,
        num_detectors: int,
        start: int,
        end: int,
    ) -> Batch:
        """Build a PyG Batch from a slice of syndrome shots."""
        n = end - start

        det = torch.from_numpy(
            syndrome[start:end].astype(np.float32, copy=False)
        )
        if num_nodes > num_detectors:
            pad = torch.zeros(n, num_nodes - num_detectors, dtype=torch.float32)
            det = torch.cat([det, pad], dim=1)
        x = det.reshape(-1, 1)

        offsets = (
            torch.arange(n, dtype=torch.long).unsqueeze(1) * num_nodes
        )
        ei_rep = (
            edge_index.unsqueeze(0).expand(n, -1, -1) + offsets.unsqueeze(1)
        )
        ei_batch = ei_rep.permute(1, 0, 2).reshape(2, -1)

        ea_batch = (
            edge_attr.unsqueeze(0)
            .expand(n, -1, -1)
            .reshape(-1, edge_attr.shape[1])
        )

        batch_vec = torch.arange(n, dtype=torch.long).repeat_interleave(
            num_nodes
        )

        return Batch(
            x=x,
            edge_index=ei_batch,
            edge_attr=ea_batch,
            batch=batch_vec,
        )

    @torch.no_grad()
    def _eval_logical_head(
        self,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        syndrome: np.ndarray,
        logical: np.ndarray,
        num_nodes: int,
        num_detectors: int,
    ) -> tuple[int, int]:
        """Evaluate logical_head: direct observable prediction."""
        n = syndrome.shape[0]
        num_errors = 0

        for off in range(0, n, self.batch_size):
            end = min(off + self.batch_size, n)
            batch = self._build_batch(
                syndrome, edge_index, edge_attr,
                num_nodes, num_detectors, off, end,
            ).to(self.device)

            logits = self.model(batch)
            pred = (logits > 0.0).cpu().numpy().astype(np.uint8)
            errors = np.any(pred != logical[off:end], axis=1)
            num_errors += int(errors.sum())

        return n, num_errors

    @torch.no_grad()
    def _eval_edge_case(
        self,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        edge_index_np: np.ndarray,
        syndrome: np.ndarray,
        logical: np.ndarray,
        num_nodes: int,
        num_detectors: int,
        has_boundary: bool,
        num_observables: int,
    ) -> tuple[int, int, float]:
        """Evaluate edge case: per-shot GNN weights => PyMatching => LER."""
        n = syndrome.shape[0]
        edges_per_graph = edge_index_np.shape[1]
        und_pairs, dir_to_undir = undirected_edges(edge_index_np)
        num_und = und_pairs.shape[0]

        predicted = np.empty((n, num_observables), dtype=np.uint8)
        matching_cache: Dict[bytes, pymatching.Matching] = {}

        for off in range(0, n, self.batch_size):
            end = min(off + self.batch_size, n)
            batch = self._build_batch(
                syndrome, edge_index, edge_attr,
                num_nodes, num_detectors, off, end,
            ).to(self.device)

            logits_np = self.model(batch).cpu().numpy()

            for g in range(end - off):
                start_e = g * edges_per_graph
                end_e = start_e + edges_per_graph
                graph_logits = logits_np[start_e:end_e]

                und_logits = _directed_to_undirected_logits(
                    graph_logits, dir_to_undir, num_und
                )
                weights = _logits_to_weights(und_logits)

                cache_key = weights.tobytes()
                if cache_key not in matching_cache:
                    matching_cache[cache_key] = _build_matching_from_weights(
                        und_pairs, weights, num_detectors, has_boundary
                    )

                pred = matching_cache[cache_key].decode(syndrome[off + g])
                predicted[off + g] = pred[:num_observables]

        shot_errors = np.any(predicted != logical, axis=1)
        return n, int(shot_errors.sum()), float("nan")

    def run(self) -> EvalReport:
        """
        Evaluate model on all settings.

        Returns
        -------
        EvalReport
            Full evaluation report with per-setting results.
        """
        settings = self._load_settings()
        baseline = self._load_baseline()

        logger.info(
            "Evaluating %s on %d settings (split=%s, device=%s)",
            self.case,
            len(settings),
            self.split,
            self.device,
        )

        report = EvalReport(
            metadata={
                "decoder": f"gnn_{self.case}",
                "checkpoint": str(self.checkpoint),
                "case": self.case,
                "split": self.split,
                "hidden_dim": self.cfg["hidden_dim"],
                "num_layers": self.cfg["num_layers"],
                "device": str(self.device),
            }
        )

        for meta in tqdm(settings, desc="GNN eval", unit="setting"):
            t0 = time.perf_counter()

            try:
                (
                    edge_index,
                    edge_attr,
                    syndrome,
                    logical,
                    ei_np,
                ) = self._load_setting_data(meta)
            except FileNotFoundError as exc:
                logger.warning(
                    "Skipping d=%d r=%d p=%s: %s",
                    meta.distance,
                    meta.rounds,
                    meta.error_prob,
                    exc,
                )
                continue

            if self.case == "logical_head":
                num_shots, num_errors = self._eval_logical_head(
                    edge_index, edge_attr, syndrome, logical,
                    meta.num_nodes, meta.num_detectors,
                )
                edge_acc = None
            else:
                num_shots, num_errors, edge_acc = self._eval_edge_case(
                    edge_index, edge_attr, ei_np, syndrome, logical,
                    meta.num_nodes, meta.num_detectors,
                    meta.has_boundary, meta.num_observables,
                )

            elapsed = time.perf_counter() - t0
            ler = num_errors / num_shots if num_shots > 0 else 0.0

            mwpm_ler = baseline.get(
                (meta.distance, meta.rounds, meta.error_prob, self.split)
            )
            report.results.append(
                SettingResult(
                    distance=meta.distance,
                    rounds=meta.rounds,
                    error_prob=meta.error_prob,
                    split=self.split,
                    num_shots=num_shots,
                    num_errors=num_errors,
                    logical_error_rate=ler,
                    elapsed_s=elapsed,
                    mwpm_ler=mwpm_ler,
                    edge_acc=edge_acc,
                )
            )

        return report
