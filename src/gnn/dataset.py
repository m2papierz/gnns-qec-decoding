from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Literal, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data


Case = Literal["logical_head", "mwpm_teacher", "hybrid"]
Split = Literal["train", "val", "test"]


@dataclass(frozen=True)
class SettingInfo:
    """Metadata needed to read one setting shard."""

    relpath: str
    num_nodes: int
    num_detectors: int


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _unpack_bits_row(packed_row: np.ndarray, num_bits: int) -> np.ndarray:
    """
    Unpack a single packed bit row.

    Parameters
    ----------
    packed_row : ndarray, shape (ceil(M/8),), dtype=uint8
        Packed bits (bitorder="little").
    num_bits : int
        Number of bits to keep.

    Returns
    -------
    bits : ndarray, shape (M,), dtype=uint8
        Unpacked bits in {0,1}.
    """
    bits = np.unpackbits(packed_row[None, :], axis=1, bitorder="little")[0]
    return bits[:num_bits].astype(np.uint8, copy=False)


class MixedSurfaceCodeDataset(Dataset):
    """
    Mixed-setting PyG dataset for rotated surface code decoding.

    This dataset mixes samples across *all* settings (d, r, p) using a global
    index (setting_id, shot_id). Each __getitem__ returns a PyG `Data` object
    containing the setting's detector graph and the shot's syndrome as node
    features.

    Returned Data fields
    --------------------
    x : torch.FloatTensor, shape (num_nodes, 1)
        Node features: detector events (0/1) for detector nodes.
        If a boundary node exists, its feature is 0.
    edge_index : torch.LongTensor, shape (2, E)
        Directed COO edge indices.
    edge_attr : torch.FloatTensor, shape (E, 2)
        Edge attributes [error_prob, weight].
    y : torch.FloatTensor
        Targets depend on `case`:
          - "logical_head": (num_observables,)
          - "mwpm_teacher"/"hybrid": (E,) directed edge labels in {0,1}
    logical : torch.FloatTensor, shape (num_observables,)
        Always included for evaluation convenience.
    setting_id : int
        Setting id for debugging/conditioning (stored as a tensor scalar).

    Notes
    -----
    - Graph tensors are cached per setting (small).
    - Split arrays are memory-mapped per setting (large).
    """

    def __init__(
        self,
        *,
        datasets_dir: Path,
        case: Case,
        split: Split,
        device: Optional[torch.device] = None,
    ) -> None:
        self.datasets_dir = Path(datasets_dir)
        self.case = case
        self.split = split
        self.device = device

        self.case_root = self.datasets_dir / case
        self.shards_root = self.case_root / "shards"
        self.splits_root = self.case_root / "splits"

        # Load settings table (setting_id -> relpath + node counts)
        settings_obj = _read_json(self.case_root / "settings.json")
        self.settings: Dict[int, SettingInfo] = {}
        for s in settings_obj["settings"]:
            sid = int(s["setting_id"])
            relpath = str(s["relpath"])
            num_nodes = int(s["graph"]["num_nodes"])
            num_det = int(
                s["splits"][split]["num_detectors"]
            )  # split-specific but usually same
            self.settings[sid] = SettingInfo(
                relpath=relpath, num_nodes=num_nodes, num_detectors=num_det
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
        """
        Load and cache (edge_index, edge_attr) for a setting.
        """
        if sid in self._graph_cache:
            return self._graph_cache[sid]

        gdir = self._setting_dir(sid) / "graph"
        edge_index = np.load(gdir / "edge_index.npy").astype(np.int64, copy=False)
        edge_attr = np.load(gdir / "edge_attr.npy").astype(np.float32, copy=False)

        ei = torch.from_numpy(edge_index).long()
        ea = torch.from_numpy(edge_attr).float()

        if self.device is not None:
            ei = ei.to(self.device)
            ea = ea.to(self.device)

        self._graph_cache[sid] = (ei, ea)
        return ei, ea

    def _get_split_arrays(self, sid: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load and cache memmaps for (syndrome, logical) for a setting+split.
        """
        if sid in self._split_cache:
            return self._split_cache[sid]

        sdir = self._setting_dir(sid)
        syndrome = np.load(sdir / f"{self.split}_syndrome.npy", mmap_mode="r")
        logical = np.load(sdir / f"{self.split}_logical.npy", mmap_mode="r")

        self._split_cache[sid] = (syndrome, logical)
        return syndrome, logical

    def _get_mwpm_arrays(self, sid: int) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        Load and cache memmaps for MWPM teacher labels for a setting+split.
        """
        if sid in self._mwpm_cache:
            return self._mwpm_cache[sid]

        sdir = self._setting_dir(sid)
        packed = np.load(
            sdir / f"{self.split}_mwpm_edge_selected_packed.npy", mmap_mode="r"
        )
        dir_to_undir = np.load(sdir / "mwpm" / "dir_to_undir.npy")
        meta = _read_json(sdir / "mwpm" / "teacher_meta.json")
        M = int(meta["num_undirected_edges"])

        self._mwpm_cache[sid] = (packed, dir_to_undir, M)
        return packed, dir_to_undir, M

    def __getitem__(self, idx: int) -> Data:
        sid = int(self._setting_id[idx])
        shot = int(self._shot_id[idx])
        info = self.settings[sid]

        edge_index, edge_attr = self._get_graph(sid)
        syndrome_mm, logical_mm = self._get_split_arrays(sid)

        # x: detector events as node features
        det = np.asarray(syndrome_mm[shot], dtype=np.float32)  # (num_detectors,)
        x = torch.from_numpy(det).view(-1, 1)

        # pad boundary node feature if graph has one
        if info.num_nodes > info.num_detectors:
            pad = torch.zeros(
                (info.num_nodes - info.num_detectors, 1), dtype=torch.float32
            )
            x = torch.cat([x, pad], dim=0)

        logical = torch.from_numpy(np.asarray(logical_mm[shot], dtype=np.float32))

        if self.device is not None:
            x = x.to(self.device)
            logical = logical.to(self.device)

        if self.case == "logical_head":
            y = logical
        else:
            packed, dir_to_undir, M = self._get_mwpm_arrays(sid)
            und = _unpack_bits_row(np.asarray(packed[shot], dtype=np.uint8), M)  # (M,)
            dir_labels = und[dir_to_undir]  # (E,)
            y = torch.from_numpy(dir_labels.astype(np.float32, copy=False))
            if self.device is not None:
                y = y.to(self.device)

        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=y,
            logical=logical,
            setting_id=torch.tensor(sid, dtype=torch.int64),
        )
        return data
