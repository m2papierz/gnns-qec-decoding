"""
Evaluator for trained GNN decoders.

Computes per-setting logical error rates (LER) for a trained checkpoint,
with optional comparison against MWPM baseline results.

Evaluation protocols by training case:

- ``logical_head``: threshold graph-level logits at 0 => observable flip.
- ``mwpm_teacher`` / ``hybrid`` / ``tn_teacher``: convert per-edge logits
  to decoder weights => decode with pluggable decoder => LER.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from torch_geometric.data import Batch
from tqdm import tqdm

from decoders.base import BaseDecoder, DecoderConfig
from decoders.mwpm import MWPMDecoder
from gnn.metrics import EvalReport, SettingResult
from gnn.models.heads import Case, QECDecoder, build_model
from qec_generator.utils import read_json, undirected_edges


logger = logging.getLogger(__name__)


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
        Data split to evaluate.
    decoder_type : str
        Decoder to use for edge-based cases: ``"mwpm"`` (default).
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
        decoder_type: str = "mwpm",
        baseline_path: Path | None = None,
        batch_size: int = 256,
    ) -> None:
        self.checkpoint = checkpoint
        self.datasets_dir = datasets_dir
        self.split = split
        self.decoder_type = decoder_type
        self.baseline_path = baseline_path
        self.batch_size = batch_size

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.cfg = self._load_model()
        self.case: Case = self.cfg["case"]

    def _load_model(self) -> tuple[QECDecoder, Dict[str, Any]]:
        """Load model from checkpoint."""
        ckpt = torch.load(
            self.checkpoint,
            map_location=self.device,
            weights_only=False,
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

    def _make_decoder(
        self,
        meta: _SettingMeta,
        und_pairs: np.ndarray,
        directed_logits: np.ndarray,
        dir_to_undir: np.ndarray,
        num_undirected: int,
    ) -> BaseDecoder:
        """Build a decoder from GNN edge logits for a single graph.

        Parameters
        ----------
        meta : _SettingMeta
            Setting metadata (detectors, observables, boundary).
        und_pairs : ndarray, shape ``(U, 2)``
            Undirected edge endpoints.
        directed_logits : ndarray, shape ``(E,)``
            Raw logits from the GNN edge head.
        dir_to_undir : ndarray, shape ``(E,)``
            Directed-to-undirected edge mapping.
        num_undirected : int
            Number of unique undirected edges.

        Returns
        -------
        BaseDecoder
        """
        config = DecoderConfig(
            num_detectors=meta.num_detectors,
            num_observables=meta.num_observables,
            has_boundary=meta.has_boundary,
            boundary_node=meta.num_detectors if meta.has_boundary else None,
        )
        if self.decoder_type == "mwpm":
            return MWPMDecoder.from_gnn_logits(
                config,
                und_pairs,
                directed_logits,
                dir_to_undir,
                num_undirected,
            )
        raise ValueError(f"Unknown decoder type: {self.decoder_type!r}")

    def _load_settings(self) -> List[_SettingMeta]:
        """Load setting metadata from settings.json."""
        case_root = self.datasets_dir / self.case
        settings_obj = read_json(case_root / "settings.json")

        return [
            _SettingMeta(
                setting_id=int(s["setting_id"]),
                relpath=str(s["relpath"]),
                distance=int(s["distance"]),
                rounds=int(s["rounds"]),
                error_prob=float(s["error_prob"]),
                num_nodes=int(s["graph"]["num_nodes"]),
                num_detectors=int(s["graph"]["num_detectors"]),
                num_observables=int(s["graph"]["num_observables"]),
                has_boundary=bool(s["graph"].get("has_boundary", False)),
            )
            for s in settings_obj["settings"]
        ]

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
        self,
        meta: _SettingMeta,
    ) -> tuple[torch.Tensor, torch.Tensor, np.ndarray, np.ndarray, np.ndarray]:
        """Load graph and split arrays for one setting."""
        shard = self.datasets_dir / self.case / "shards" / meta.relpath
        graph_dir = shard / "graph"

        ei_np = np.load(graph_dir / "edge_index.npy").astype(np.int64)
        ea_np = np.load(graph_dir / "edge_attr.npy").astype(np.float32)

        edge_index = torch.from_numpy(ei_np).long()
        edge_attr = torch.from_numpy(ea_np).float()

        syndrome = np.load(shard / f"{self.split}_syndrome.npy").astype(np.uint8)
        logical = np.load(shard / f"{self.split}_logical.npy").astype(np.uint8)
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
        """Build a PyG Batch from a slice of syndrome shots (vectorized)."""
        n = end - start

        det = torch.from_numpy(
            syndrome[start:end].astype(np.float32, copy=False),
        )
        if num_nodes > num_detectors:
            pad = torch.zeros(n, num_nodes - num_detectors, dtype=torch.float32)
            det = torch.cat([det, pad], dim=1)
        x = det.reshape(-1, 1)

        offsets = torch.arange(n, dtype=torch.long).unsqueeze(1) * num_nodes
        ei_rep = edge_index.unsqueeze(0).expand(n, -1, -1) + offsets.unsqueeze(1)
        ei_batch = ei_rep.permute(1, 0, 2).reshape(2, -1)

        ea_batch = (
            edge_attr.unsqueeze(0).expand(n, -1, -1).reshape(-1, edge_attr.shape[1])
        )

        batch_vec = torch.arange(n, dtype=torch.long).repeat_interleave(num_nodes)

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
                syndrome,
                edge_index,
                edge_attr,
                num_nodes,
                num_detectors,
                off,
                end,
            ).to(self.device)

            logits = self.model(batch)
            pred = (logits > 0.0).cpu().numpy().astype(np.uint8)
            num_errors += int(np.any(pred != logical[off:end], axis=1).sum())

        return n, num_errors

    @torch.no_grad()
    def _eval_edge_case(
        self,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        edge_index_np: np.ndarray,
        syndrome: np.ndarray,
        logical: np.ndarray,
        meta: _SettingMeta,
    ) -> tuple[int, int, float]:
        """Evaluate edge case: per-shot GNN weights => decoder => LER."""
        n = syndrome.shape[0]
        edges_per_graph = edge_index_np.shape[1]
        und_pairs, dir_to_undir = undirected_edges(edge_index_np)
        num_und = und_pairs.shape[0]

        predicted = np.empty((n, meta.num_observables), dtype=np.uint8)
        decoder_cache: Dict[bytes, BaseDecoder] = {}

        for off in range(0, n, self.batch_size):
            end = min(off + self.batch_size, n)
            batch = self._build_batch(
                syndrome,
                edge_index,
                edge_attr,
                meta.num_nodes,
                meta.num_detectors,
                off,
                end,
            ).to(self.device)

            logits_np = self.model(batch).cpu().numpy()

            for g in range(end - off):
                start_e = g * edges_per_graph
                graph_logits = logits_np[start_e : start_e + edges_per_graph]

                cache_key = graph_logits.tobytes()
                if cache_key not in decoder_cache:
                    decoder_cache[cache_key] = self._make_decoder(
                        meta,
                        und_pairs,
                        graph_logits,
                        dir_to_undir,
                        num_und,
                    )

                predicted[off + g] = decoder_cache[cache_key].decode(syndrome[off + g])

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
                ei, ea, syndrome, logical, ei_np = self._load_setting_data(meta)
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
                    ei,
                    ea,
                    syndrome,
                    logical,
                    meta.num_nodes,
                    meta.num_detectors,
                )
                edge_acc = None
            else:
                num_shots, num_errors, edge_acc = self._eval_edge_case(
                    ei,
                    ea,
                    ei_np,
                    syndrome,
                    logical,
                    meta,
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
