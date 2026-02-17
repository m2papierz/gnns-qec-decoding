"""
PyG dataset for mixed-setting QEC decoding.

All tensors are returned on CPU.  Move to device in the training loop
via ``batch = batch.to(device)`` — this keeps ``DataLoader(num_workers>0)``
safe (workers cannot share CUDA state).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Literal, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data


Case = Literal["logical_head", "mwpm_teacher", "hybrid"]
Split = Literal["train", "val", "test"]


@dataclass(frozen=True)
class SettingInfo:
    """
    Metadata for a single (distance, rounds, error_prob) shard.

    Attributes
    ----------
    relpath : str
        Shard directory relative to ``shards/``.
    num_nodes : int
        Total graph nodes (detectors + optional boundary).
    num_detectors : int
        Number of detector nodes.
    """

    relpath: str
    num_nodes: int
    num_detectors: int


def _read_json(path: Path) -> dict:
    """Read a UTF-8 JSON file and return its contents as a dict."""
    return json.loads(path.read_text(encoding="utf-8"))


def _unpack_bits_row(packed_row: np.ndarray, num_bits: int) -> np.ndarray:
    """
    Unpack a single row of packed MWPM teacher labels.

    Parameters
    ----------
    packed_row : ndarray, shape (ceil(num_bits/8),), dtype=uint8
        Bit-packed row (bitorder ``"little"``).
    num_bits : int
        Number of meaningful bits to retain.

    Returns
    -------
    ndarray, shape (num_bits,), dtype=uint8
        Unpacked bits in {0, 1}.
    """
    bits = np.unpackbits(packed_row[None, :], axis=1, bitorder="little")[0]
    return bits[:num_bits].astype(np.uint8, copy=False)


class MixedSurfaceCodeDataset(Dataset):
    """
    Mixed-setting PyG dataset for rotated surface code decoding.

    Mixes samples across all settings ``(d, r, p)`` using a global index
    ``(setting_id, shot_id)``.  Each ``__getitem__`` returns a PyG
    :class:`~torch_geometric.data.Data` object containing the setting's
    detector graph and the shot's syndrome as node features.

    Returned ``Data`` fields
    ------------------------
    x : FloatTensor, shape ``(num_nodes, 1)``
        Node features: syndrome bits for detectors, 0 for boundary.
    edge_index : LongTensor, shape ``(2, E)``
        Directed COO edge indices (bidirectional).
    edge_attr : FloatTensor, shape ``(E, 2)``
        ``[error_prob, weight]`` per directed edge.
    y : FloatTensor
        Target depends on *case*:
        ``"logical_head"`` → ``(num_observables,)``;
        ``"mwpm_teacher"`` / ``"hybrid"`` → ``(E,)`` directed edge labels.
    logical : FloatTensor, shape ``(num_observables,)``
        Ground-truth observable flip (always present for eval).
    setting_id : LongTensor, scalar
        Setting index for conditioning / diagnostics.

    Parameters
    ----------
    datasets_dir : Path
        Root directory produced by ``generate_datasets``.
    case : Case
        Training variant.
    split : Split
        Data split to load.

    Notes
    -----
    Graph tensors are cached in memory (small); split arrays are
    memory-mapped (large).  All tensors are returned on CPU — transfer
    to device in the training loop for ``DataLoader`` compatibility.
    """

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

        # Load settings table (setting_id → relpath + node counts)
        settings_obj = _read_json(self.case_root / "settings.json")
        self.settings: Dict[int, SettingInfo] = {}
        for s in settings_obj["settings"]:
            sid = int(s["setting_id"])
            self.settings[sid] = SettingInfo(
                relpath=str(s["relpath"]),
                num_nodes=int(s["graph"]["num_nodes"]),
                num_detectors=int(s["splits"][split]["num_detectors"]),
            )

        # Load global mixed index
        idx = np.load(self.splits_root / f"{split}_index.npz")
        self._setting_id = idx["setting_id"].astype(np.int32, copy=False)
        self._shot_id = idx["shot_id"].astype(np.int32, copy=False)

        # Caches keyed by setting_id
        self._graph_cache: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        self._split_cache: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
        self._mwpm_cache: Dict[int, Tuple[np.ndarray, np.ndarray, int]] = {}

    def __len__(self) -> int:
        return int(self._setting_id.shape[0])

    def _setting_dir(self, sid: int) -> Path:
        return self.shards_root / self.settings[sid].relpath

    def _get_graph(self, sid: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return cached ``(edge_index, edge_attr)`` for *sid*."""
        if sid in self._graph_cache:
            return self._graph_cache[sid]

        gdir = self._setting_dir(sid) / "graph"
        ei = torch.from_numpy(
            np.load(gdir / "edge_index.npy").astype(np.int64, copy=False)
        ).long()
        ea = torch.from_numpy(
            np.load(gdir / "edge_attr.npy").astype(np.float32, copy=False)
        ).float()

        self._graph_cache[sid] = (ei, ea)
        return ei, ea

    def _get_split_arrays(self, sid: int) -> Tuple[np.ndarray, np.ndarray]:
        """Return memory-mapped ``(syndrome, logical)`` for *sid*."""
        if sid in self._split_cache:
            return self._split_cache[sid]

        sdir = self._setting_dir(sid)
        syndrome = np.load(sdir / f"{self.split}_syndrome.npy", mmap_mode="r")
        logical = np.load(sdir / f"{self.split}_logical.npy", mmap_mode="r")

        self._split_cache[sid] = (syndrome, logical)
        return syndrome, logical

    def _get_mwpm_arrays(self, sid: int) -> Tuple[np.ndarray, np.ndarray, int]:
        """Return cached MWPM labels ``(packed, dir_to_undir, num_und)`` for *sid*."""
        if sid in self._mwpm_cache:
            return self._mwpm_cache[sid]

        sdir = self._setting_dir(sid)
        packed = np.load(
            sdir / f"{self.split}_mwpm_edge_selected_packed.npy", mmap_mode="r"
        )
        dir_to_undir = np.load(sdir / "mwpm" / "dir_to_undir.npy")
        meta = _read_json(sdir / "mwpm" / "teacher_meta.json")
        num_und = int(meta["num_undirected_edges"])

        self._mwpm_cache[sid] = (packed, dir_to_undir, num_und)
        return packed, dir_to_undir, num_und

    def __getitem__(self, idx: int) -> Data:
        sid = int(self._setting_id[idx])
        shot = int(self._shot_id[idx])
        info = self.settings[sid]

        edge_index, edge_attr = self._get_graph(sid)
        syndrome_mm, logical_mm = self._get_split_arrays(sid)

        # Node features: syndrome bits for detectors, 0-padded for boundary.
        det = np.asarray(syndrome_mm[shot], dtype=np.float32)
        x = torch.from_numpy(det).view(-1, 1)
        if info.num_nodes > info.num_detectors:
            pad = torch.zeros(
                info.num_nodes - info.num_detectors, 1, dtype=torch.float32
            )
            x = torch.cat([x, pad], dim=0)

        logical = torch.from_numpy(np.asarray(logical_mm[shot], dtype=np.float32))

        # Target depends on training case.
        if self.case == "logical_head":
            y = logical
        else:
            packed, dir_to_undir, num_und = self._get_mwpm_arrays(sid)
            und = _unpack_bits_row(np.asarray(packed[shot], dtype=np.uint8), num_und)
            y = torch.from_numpy(und[dir_to_undir].astype(np.float32, copy=False))

        return Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=y,
            logical=logical,
            setting_id=torch.tensor(sid, dtype=torch.int64),
        )
