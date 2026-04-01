"""
PyG dataset for mixed-setting QEC decoding.

All tensors are returned on CPU.  Move to device in the training loop
via ``batch = batch.to(device)`` — this keeps ``DataLoader(num_workers>0)``
safe (workers cannot share CUDA state).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data

from constants import Case, Split
from qec_generator.utils import read_json


logger = logging.getLogger(__name__)

# Feature dimensions (fixed).
NODE_DIM: int = 5  # syndrome + is_boundary + d_horizontal + d_vertical + d_temporal
EDGE_DIM: int = 3  # error_prob + weight + inv_dist_sq


@dataclass(frozen=True)
class SettingInfo:
    """Metadata for a single (distance, rounds, error_prob) shard."""

    relpath: str
    num_nodes: int
    num_detectors: int
    distance: int
    rounds: int
    error_prob: float


def compute_relative_node_features(
    coords: np.ndarray,
    is_boundary: np.ndarray,
    distance: int,
    rounds: int,
) -> np.ndarray:
    """Build relative node features from raw Stim detector coordinates.

    For each detector node, computes position relative to code boundaries:

    - ``d_horizontal = x / (2 * distance)``
    - ``d_vertical   = y / (2 * distance)``
    - ``d_temporal    = t / rounds``

    Boundary nodes (NaN coordinates) get all-zero features.

    Parameters
    ----------
    coords : ndarray, shape ``(N, C)``, float32
        Raw node coordinates from Stim (at least 3 columns: x, y, t).
    is_boundary : ndarray, shape ``(N,)``, bool
        True for the virtual boundary node.
    distance : int
        Code distance (spatial normalisation uses ``2 * d``).
    rounds : int
        Syndrome measurement rounds (temporal normalisation).

    Returns
    -------
    ndarray, shape ``(N, 4)``, float32
        ``[is_boundary, d_horizontal, d_vertical, d_temporal]`` per node.
    """
    n = coords.shape[0]
    out = np.zeros((n, 4), dtype=np.float32)

    out[:, 0] = is_boundary.astype(np.float32)

    mask = ~is_boundary & np.all(np.isfinite(coords[:, :3]), axis=1)

    spatial_scale = max(2.0 * distance, 1.0)
    temporal_scale = max(float(rounds), 1.0)

    if mask.any():
        out[mask, 1] = coords[mask, 0] / spatial_scale  # d_horizontal
        out[mask, 2] = coords[mask, 1] / spatial_scale  # d_vertical
        out[mask, 3] = coords[mask, 2] / temporal_scale  # d_temporal

    return out


def compute_inv_dist_sq(
    edge_index: np.ndarray,
    coords: np.ndarray,
    is_boundary: np.ndarray,
) -> np.ndarray:
    """Compute squared inverse Chebyshev distance for each directed edge.

    .. math::

        e_{ij} = \\frac{1}{\\max(|x_i - x_j|, |y_i - y_j|, |t_i - t_j|)^2}

    Edges touching a boundary node get ``inv_dist_sq = 0``.

    Parameters
    ----------
    edge_index : ndarray, shape ``(2, E)``, int64
    coords : ndarray, shape ``(N, C)``, float32
        Raw node coordinates (at least 3 columns).
    is_boundary : ndarray, shape ``(N,)``, bool

    Returns
    -------
    ndarray, shape ``(E,)``, float32
    """
    src, dst = edge_index[0], edge_index[1]
    n_edges = len(src)
    result = np.zeros(n_edges, dtype=np.float32)

    bnd_mask = is_boundary[src] | is_boundary[dst]
    valid = ~bnd_mask

    if valid.any():
        c_src = coords[src[valid], :3]
        c_dst = coords[dst[valid], :3]
        diff = np.abs(c_src - c_dst)
        chebyshev = diff.max(axis=1)
        chebyshev = np.maximum(chebyshev, 1e-6)
        result[valid] = 1.0 / (chebyshev**2)

    return result


class MixedSurfaceCodeDataset(Dataset):
    """
    Mixed-setting PyG dataset for rotated surface code decoding.

    Mixes samples across all settings ``(d, r, p)`` using a global index
    ``(setting_id, shot_id)``.  Each ``__getitem__`` returns a PyG
    :class:`~torch_geometric.data.Data` object.

    Returned ``Data`` fields
    ------------------------
    x : FloatTensor, shape ``(num_nodes, 5)``
        ``[syndrome, is_boundary, d_horizontal, d_vertical, d_temporal]``
    edge_index : LongTensor, shape ``(2, E)``
        Directed COO edge indices (bidirectional).
    edge_attr : FloatTensor, shape ``(E, 3)``
        ``[error_prob, weight, inv_dist_sq]``
    y : FloatTensor
        Target depends on *case*:
        ``"direct"`` => ``(num_observables,)``;
        ``"edge"`` => ``(E,)`` float BP marginals per directed edge.
    logical : FloatTensor, shape ``(num_observables,)``
        Ground-truth observable flip (always present for eval).
    setting_id : LongTensor, scalar
        Setting index for conditioning / diagnostics.
    """

    node_dim: int = NODE_DIM
    edge_dim: int = EDGE_DIM

    def __init__(
        self,
        *,
        datasets_dir: Path,
        case: Case,
        split: Split,
    ) -> None:
        self.datasets_dir = Path(datasets_dir)
        self.case = case
        self.split = split

        self.case_root = self.datasets_dir / case
        self.shards_root = self.case_root / "shards"
        self.splits_root = self.case_root / "splits"

        # Load settings table
        settings_obj = read_json(self.case_root / "settings.json")
        self.settings: Dict[int, SettingInfo] = {}
        for s in settings_obj["settings"]:
            sid = int(s["setting_id"])
            self.settings[sid] = SettingInfo(
                relpath=str(s["relpath"]),
                num_nodes=int(s["graph"]["num_nodes"]),
                num_detectors=int(s["splits"][split]["num_detectors"]),
                distance=int(s["distance"]),
                rounds=int(s["rounds"]),
                error_prob=float(s["error_prob"]),
            )

        # Load global mixed index
        idx = np.load(self.splits_root / f"{split}_index.npz")
        self._setting_id = idx["setting_id"].astype(np.int32, copy=False)
        self._shot_id = idx["shot_id"].astype(np.int32, copy=False)

        # Caches keyed by setting_id
        self._graph_cache: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        self._split_cache: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
        self._bp_cache: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
        self._static_node_feat_cache: Dict[int, torch.Tensor] = {}

    def __len__(self) -> int:
        return int(self._setting_id.shape[0])

    def _setting_dir(self, sid: int) -> Path:
        return self.shards_root / self.settings[sid].relpath

    def _get_graph(self, sid: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return cached ``(edge_index, edge_attr)`` for *sid*.

        ``edge_attr`` has shape ``(E, 3)``: ``[error_prob, weight, inv_dist_sq]``.
        """
        if sid in self._graph_cache:
            return self._graph_cache[sid]

        gdir = self._setting_dir(sid) / "graph"

        ei_np = np.load(gdir / "edge_index.npy").astype(np.int64, copy=False)
        ea_np = np.load(gdir / "edge_attr.npy").astype(np.float32, copy=False)

        coords = np.load(gdir / "node_coords.npy").astype(np.float32, copy=False)
        boundary = np.load(gdir / "node_is_boundary.npy")

        ids = compute_inv_dist_sq(ei_np, coords, boundary)
        ea_np = np.concatenate([ea_np, ids[:, None]], axis=1)  # (E, 3)

        ei = torch.from_numpy(ei_np).long()
        ea = torch.from_numpy(ea_np).float()

        self._graph_cache[sid] = (ei, ea)
        return ei, ea

    def _get_static_node_features(self, sid: int) -> torch.Tensor:
        """Return cached ``(num_nodes, 4)`` relative node features.

        Layout: ``[is_boundary, d_horizontal, d_vertical, d_temporal]``.
        """
        if sid in self._static_node_feat_cache:
            return self._static_node_feat_cache[sid]

        info = self.settings[sid]
        gdir = self._setting_dir(sid) / "graph"

        coords = np.load(gdir / "node_coords.npy").astype(np.float32, copy=False)
        boundary = np.load(gdir / "node_is_boundary.npy")

        static = compute_relative_node_features(
            coords,
            boundary,
            info.distance,
            info.rounds,
        )
        tensor = torch.from_numpy(static)
        self._static_node_feat_cache[sid] = tensor
        return tensor

    def _get_split_arrays(self, sid: int) -> Tuple[np.ndarray, np.ndarray]:
        """Return memory-mapped ``(syndrome, logical)`` for *sid*."""
        if sid in self._split_cache:
            return self._split_cache[sid]

        sdir = self._setting_dir(sid)
        syndrome = np.load(sdir / f"{self.split}_syndrome.npy", mmap_mode="r")
        logical = np.load(sdir / f"{self.split}_logical.npy", mmap_mode="r")

        self._split_cache[sid] = (syndrome, logical)
        return syndrome, logical

    def _get_bp_arrays(self, sid: int) -> Tuple[np.ndarray, np.ndarray]:
        """Return cached BP soft labels for *sid*."""
        if sid in self._bp_cache:
            return self._bp_cache[sid]

        sdir = self._setting_dir(sid)
        marginals = np.load(sdir / f"{self.split}_bp_soft_labels.npy", mmap_mode="r")
        dir_to_undir = np.load(sdir / "mwpm" / "dir_to_undir.npy")

        self._bp_cache[sid] = (marginals, dir_to_undir)
        return marginals, dir_to_undir

    def __getitem__(self, idx: int) -> Data:
        sid = int(self._setting_id[idx])
        shot = int(self._shot_id[idx])
        info = self.settings[sid]

        edge_index, edge_attr = self._get_graph(sid)
        syndrome_mm, logical_mm = self._get_split_arrays(sid)

        # --- Build node features: [syndrome, is_boundary, d_h, d_v, d_t] ---
        det = np.asarray(syndrome_mm[shot], dtype=np.float32)
        syndrome_full = np.zeros(info.num_nodes, dtype=np.float32)
        syndrome_full[: info.num_detectors] = det

        static = self._get_static_node_features(sid)  # (N, 4)
        syndrome_col = torch.from_numpy(syndrome_full).unsqueeze(1)  # (N, 1)
        x = torch.cat([syndrome_col, static], dim=1)  # (N, 5)

        logical = torch.from_numpy(np.asarray(logical_mm[shot], dtype=np.float32))

        # Target depends on training case.
        if self.case == "direct":
            y = logical
        else:
            marginals, dir_to_undir = self._get_bp_arrays(sid)
            und_marginals = np.asarray(marginals[shot], dtype=np.float32)
            y = torch.from_numpy(und_marginals[dir_to_undir])

        return Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=y,
            logical=logical,
            setting_id=torch.tensor(sid, dtype=torch.int64),
        )
