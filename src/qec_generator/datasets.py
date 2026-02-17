"""Dataset packaging for ML training pipelines."""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pymatching
from tqdm import tqdm

from constants import CASES, MWPM_BITORDER, MWPM_LABEL, SPLITS, Case
from qec_generator.config import Config
from qec_generator.utils import read_json, save_json, save_npy


logger = logging.getLogger(__name__)


def _copy_graph(
    raw_dir: Path,
    out_dir: Path,
    overwrite: bool,
) -> Dict[str, Any]:
    """
    Copy and transform graph tensors from raw to dataset format.

    Parameters
    ----------
    raw_dir : Path
        Source directory containing raw graph data.
    out_dir : Path
        Destination directory for processed graph.
    overwrite : bool
        Overwrite existing files.

    Returns
    -------
    Dict
        Graph metadata including num_nodes, num_detectors, etc.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    edge_index = np.load(raw_dir / "edge_index.npy").astype(np.int64)
    edge_p = np.load(raw_dir / "edge_error_prob.npy").astype(np.float32)
    edge_w = np.load(raw_dir / "edge_weight.npy").astype(np.float32)

    # Combine edge attributes: [error_prob, weight]
    edge_attr = np.stack([edge_p, edge_w], axis=1)

    save_npy(out_dir / "edge_index.npy", edge_index, overwrite)
    save_npy(out_dir / "edge_attr.npy", edge_attr, overwrite)

    # Optional node arrays
    for name in ("node_coords", "node_is_boundary"):
        src = raw_dir / f"{name}.npy"
        if src.exists():
            dtype = bool if "boundary" in name else np.float32
            save_npy(out_dir / f"{name}.npy", np.load(src).astype(dtype), overwrite)

    # Load metadata
    meta = read_json(raw_dir / "meta.json") if (raw_dir / "meta.json").exists() else {}
    num_nodes = meta.get(
        "num_nodes", int(edge_index.max()) + 1 if edge_index.size else 0
    )

    graph_meta = {
        "num_nodes": num_nodes,
        "num_detectors": meta.get("num_detectors", -1),
        "num_observables": meta.get("num_observables", -1),
        "num_edges": int(edge_index.shape[1]),
        "has_boundary": meta.get("has_boundary", False),
        "edge_attr_columns": ["error_prob", "weight"],
    }
    save_json(out_dir / "graph_meta.json", graph_meta, overwrite)
    return graph_meta


def _copy_split(
    raw_dir: Path,
    out_dir: Path,
    split: str,
    overwrite: bool,
) -> Dict[str, int]:
    """
    Copy split arrays from raw to dataset shard.

    Parameters
    ----------
    raw_dir : Path
        Source directory containing raw split data.
    out_dir : Path
        Destination directory for processed split.
    split : str
        Split name ("train", "val", or "test").
    overwrite : bool
        Overwrite existing files.

    Returns
    -------
    Dict
        Split statistics: num_shots, num_detectors, num_observables.

    Raises
    ------
    FileNotFoundError
        If split files are missing.
    """
    syn_src = raw_dir / f"{split}_syndrome.npy"
    log_src = raw_dir / f"{split}_logical.npy"

    if not syn_src.exists() or not log_src.exists():
        raise FileNotFoundError(f"Missing split files for '{split}' in {raw_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)

    syn = np.load(syn_src).astype(np.uint8)
    log = np.load(log_src).astype(np.uint8)

    save_npy(out_dir / f"{split}_syndrome.npy", syn, overwrite)
    save_npy(out_dir / f"{split}_logical.npy", log, overwrite)

    return {
        "num_shots": syn.shape[0],
        "num_detectors": syn.shape[1],
        "num_observables": log.shape[1] if log.ndim == 2 else 1,
    }


def _undirected_edges(edge_index: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract unique undirected edges and mapping from directed edges.

    Parameters
    ----------
    edge_index : ndarray, shape (2, E)
        Directed edge list.

    Returns
    -------
    unique_pairs : ndarray, shape (U, 2)
        Unique undirected edge pairs.
    dir_to_undir : ndarray, shape (E,)
        Mapping from directed edge index to undirected edge index.
    """
    src, dst = edge_index[0], edge_index[1]
    pairs = np.stack([np.minimum(src, dst), np.maximum(src, dst)], axis=1)
    unique, inverse = np.unique(pairs, axis=0, return_inverse=True)
    return unique.astype(np.int64), inverse.astype(np.int64)


def _build_teacher_matching(
    und_pairs: np.ndarray,
    und_attr: np.ndarray,
    num_detectors: int,
    boundary_node: int | None,
) -> pymatching.Matching:
    """
    Build PyMatching graph with unique fault IDs per undirected edge.

    Parameters
    ----------
    und_pairs : ndarray, shape (U, 2)
        Undirected edge pairs.
    und_attr : ndarray, shape (U, 2)
        Edge attributes [error_prob, weight].
    num_detectors : int
        Number of detector nodes.
    boundary_node : int or None
        Index of virtual boundary node, or None.

    Returns
    -------
    pymatching.Matching
        Configured matching decoder.
    """
    m = pymatching.Matching()
    num_edges = und_pairs.shape[0]

    for eid in range(num_edges):
        u, v = int(und_pairs[eid, 0]), int(und_pairs[eid, 1])
        p, w = float(und_attr[eid, 0]), float(und_attr[eid, 1])

        if boundary_node is not None and boundary_node in (u, v):
            det = v if u == boundary_node else u
            if 0 <= det < num_detectors:
                m.add_boundary_edge(det, fault_ids={eid}, weight=w, error_probability=p)
        elif u < num_detectors and v < num_detectors:
            m.add_edge(u, v, fault_ids={eid}, weight=w, error_probability=p)

    m.ensure_num_fault_ids(num_edges)
    return m


def _write_mwpm_labels(
    shard_dir: Path,
    graph_dir: Path,
    split: str,
    chunk_size: int,
    num_detectors: int,
    has_boundary: bool,
    overwrite: bool,
) -> None:
    """
    Generate packed MWPM teacher labels for one split.

    Parameters
    ----------
    shard_dir : Path
        Shard directory containing split data.
    graph_dir : Path
        Graph directory with edge_index and edge_attr.
    split : str
        Split name.
    chunk_size : int
        Batch size for MWPM decoding.
    num_detectors : int
        Number of detector nodes.
    has_boundary : bool
        Whether graph has boundary node.
    overwrite : bool
        Overwrite existing labels.
    """
    out_path = shard_dir / f"{split}_mwpm_edge_selected_packed.npy"
    if out_path.exists() and not overwrite:
        logger.debug("MWPM labels already exist for split '%s'", split)
        return

    edge_index = np.load(graph_dir / "edge_index.npy")
    edge_attr = np.load(graph_dir / "edge_attr.npy")

    und_pairs, dir_to_undir = _undirected_edges(edge_index)
    num_und = und_pairs.shape[0]

    # Get undirected attributes (first occurrence)
    first_idx = np.full(num_und, -1, dtype=np.int64)
    for e, uid in enumerate(dir_to_undir):
        if first_idx[uid] == -1:
            first_idx[uid] = e
    und_attr = edge_attr[first_idx]

    boundary_node = num_detectors if has_boundary else None
    teacher = _build_teacher_matching(und_pairs, und_attr, num_detectors, boundary_node)

    # Save MWPM metadata
    mwpm_dir = shard_dir / "mwpm"
    mwpm_dir.mkdir(parents=True, exist_ok=True)

    endpoints = und_pairs.astype(np.int32)
    if has_boundary and boundary_node is not None:
        endpoints = endpoints.copy()
        endpoints[endpoints == boundary_node] = -1

    save_npy(mwpm_dir / "undirected_edge_endpoints.npy", endpoints, overwrite)
    save_npy(mwpm_dir / "dir_to_undir.npy", dir_to_undir, overwrite)
    save_json(
        mwpm_dir / "teacher_meta.json",
        {
            "num_undirected_edges": num_und,
            "bitorder": MWPM_BITORDER,
            "label": MWPM_LABEL,
        },
        overwrite,
    )

    # Decode syndromes in batches
    syndrome = np.load(shard_dir / f"{split}_syndrome.npy", mmap_mode="r")
    n_samples = syndrome.shape[0]
    if syndrome.shape[1] != num_detectors:
        raise ValueError(
            f"MWPM label gen: syndrome width {syndrome.shape[1]} != "
            f"num_detectors {num_detectors} for split '{split}'"
        )
    packed_cols = math.ceil(num_und / 8)
    packed = np.empty((n_samples, packed_cols), dtype=np.uint8)

    for off in tqdm(range(0, n_samples, chunk_size), desc=f"MWPM {split}", leave=False):
        batch = np.asarray(syndrome[off : off + chunk_size], dtype=np.uint8)
        selected = teacher.decode_batch(batch).astype(np.uint8)[:, :num_und]
        packed[off : off + selected.shape[0]] = np.packbits(
            selected, axis=1, bitorder=MWPM_BITORDER
        )

    save_npy(out_path, packed, overwrite=True)


def generate_datasets(
    cfg: Config,
    cases: tuple[Case, ...] = CASES,
    overwrite: bool = False,
) -> None:
    """
    Generate training datasets from raw sampled data.

    Creates directory structure for each case:
    datasets_dir/{case}/
        shards/
            d{distance}_r{rounds}_p{error_prob}/
                graph/
                    edge_index.npy, edge_attr.npy, graph_meta.json
                {split}_syndrome.npy
                {split}_logical.npy
                {split}_mwpm_edge_selected_packed.npy
                mwpm/
                    undirected_edge_endpoints.npy
                    dir_to_undir.npy
                    teacher_meta.json
        splits/
            {split}_index.npz  # global index: setting_id, shot_id
        settings.json
        build_meta.json

    Parameters
    ----------
    cfg : Config
        Configuration with raw_data_dir and datasets_dir.
    cases : tuple of Case, default=("logical_head", "mwpm_teacher", "hybrid")
        Dataset variants to generate.
    overwrite : bool, default=False
        Overwrite existing files.
    """
    cfg.datasets_dir.mkdir(parents=True, exist_ok=True)
    settings = list(cfg.iter_settings())

    logger.info("Generating datasets for cases: %s", cases)

    for case in cases:
        logger.info("Processing case: %s", case)

        case_root = cfg.datasets_dir / case
        shards_root = case_root / "shards"
        splits_root = case_root / "splits"
        shards_root.mkdir(parents=True, exist_ok=True)
        splits_root.mkdir(parents=True, exist_ok=True)

        split_indices: dict[str, dict[str, list[int]]] = {
            s: {"setting": [], "shot": []} for s in SPLITS
        }
        settings_table: list[dict[str, Any]] = []

        for sid, (d, r, p) in enumerate(tqdm(settings, desc=f"{case}")):
            raw_dir = cfg.setting_dir(d, r, p)
            if not raw_dir.exists():
                raise FileNotFoundError(f"Missing raw dir: {raw_dir}")

            rel = raw_dir.relative_to(cfg.raw_data_dir)
            shard_dir = shards_root / rel
            graph_dir = shard_dir / "graph"

            # Copy graph
            graph_meta_path = graph_dir / "graph_meta.json"
            if overwrite or not graph_meta_path.exists():
                graph_meta = _copy_graph(raw_dir / "graph", graph_dir, overwrite)
            else:
                graph_meta = read_json(graph_meta_path)

            split_stats: dict[str, dict[str, int]] = {}

            # Copy splits and cross-validate against graph metadata.
            for split in SPLITS:
                try:
                    if overwrite or not (shard_dir / f"{split}_syndrome.npy").exists():
                        stats = _copy_split(raw_dir, shard_dir, split, overwrite)
                    else:
                        # Load existing to get stats
                        syn = np.load(
                            shard_dir / f"{split}_syndrome.npy", mmap_mode="r"
                        )
                        log = np.load(shard_dir / f"{split}_logical.npy", mmap_mode="r")
                        stats = {
                            "num_shots": syn.shape[0],
                            "num_detectors": syn.shape[1],
                            "num_observables": log.shape[1] if log.ndim == 2 else 1,
                        }
                    split_stats[split] = stats

                    # Cross-validate: graph and split must agree on detector count.
                    graph_nd = graph_meta.get("num_detectors", -1)
                    if graph_nd >= 0 and stats["num_detectors"] != graph_nd:
                        raise ValueError(
                            f"Detector count mismatch for setting d={d} r={r} "
                            f"p={p} split={split}: graph says {graph_nd}, "
                            f"syndrome has {stats['num_detectors']}"
                        )

                    # Build global index
                    n = stats["num_shots"]
                    split_indices[split]["setting"].extend([sid] * n)
                    split_indices[split]["shot"].extend(range(n))

                except FileNotFoundError as e:
                    logger.warning("Skipping split '%s': %s", split, e)
                    continue

            # MWPM labels
            if case in ("mwpm_teacher", "hybrid"):
                for split, stats in split_stats.items():
                    _write_mwpm_labels(
                        shard_dir,
                        graph_dir,
                        split,
                        cfg.chunk_size,
                        stats["num_detectors"],
                        graph_meta.get("has_boundary", False),
                        overwrite,
                    )

            settings_table.append(
                {
                    "setting_id": sid,
                    "relpath": str(rel),
                    "distance": d,
                    "rounds": r,
                    "error_prob": p,
                    "graph": graph_meta,
                    "splits": split_stats,
                }
            )

        # Write global indices
        for split in SPLITS:
            out = splits_root / f"{split}_index.npz"
            if out.exists() and not overwrite:
                logger.debug("Split index exists: %s", out)
                continue
            np.savez_compressed(
                out,
                setting_id=np.array(split_indices[split]["setting"], dtype=np.int32),
                shot_id=np.array(split_indices[split]["shot"], dtype=np.int32),
            )

        # Write metadata
        save_json(case_root / "settings.json", {"settings": settings_table}, overwrite)
        save_json(
            case_root / "build_meta.json",
            {
                "case": case,
                "raw_data_dir": str(cfg.raw_data_dir),
                "datasets_dir": str(cfg.datasets_dir),
            },
            overwrite,
        )

    logger.info("Dataset generation complete")
