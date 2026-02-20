"""Shared utility functions for QEC dataset generation."""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict

import numpy as np


logger = logging.getLogger(__name__)


def stable_seed(*parts: str, base: int | None) -> int | None:
    """
    Derive a reproducible seed from base seed and string identifiers.

    Uses BLAKE2b hashing to combine base seed with arbitrary string parts,
    ensuring deterministic seed generation across runs.

    Parameters
    ----------
    *parts : str
        String identifiers to incorporate into the seed (e.g., "d=5", "p=0.01").
    base : int or None
        Base seed value. If None, returns None.

    Returns
    -------
    int or None
        Derived seed, or None if base is None.
    """
    if base is None:
        return None

    h = hashlib.blake2b(digest_size=8)
    h.update(str(base).encode())
    for part in parts:
        h.update(b"|")
        h.update(part.encode())
    return int.from_bytes(h.digest(), "little", signed=False)


def save_npy(
    path: Path,
    arr: np.ndarray,
    overwrite: bool = False,
) -> None:
    """
    Save NumPy array to .npy file with directory creation.

    Parameters
    ----------
    path : Path
        Output file path.
    arr : ndarray
        Array to save.
    overwrite : bool, default=False
        If False and file exists, skip saving.
    """
    if path.exists() and not overwrite:
        logger.debug("Skipping existing file: %s", path)
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, arr)
    logger.debug("Saved array to %s", path)


def save_json(
    path: Path,
    data: Dict[str, Any],
    overwrite: bool = False,
) -> None:
    """
    Save dictionary to JSON file with pretty printing.

    Parameters
    ----------
    path : Path
        Output file path.
    data : Dict
        Dictionary to serialize.
    overwrite : bool, default=False
        If False and file exists, skip saving.
    """
    if path.exists() and not overwrite:
        logger.debug("Skipping existing file: %s", path)
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    logger.debug("Saved JSON to %s", path)


def read_json(path: Path) -> Dict[str, Any]:
    """
    Read JSON file into dictionary.

    Parameters
    ----------
    path : Path
        Input file path.

    Returns
    -------
    Dict
        Parsed JSON data.

    Raises
    ------
    FileNotFoundError
        If file does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")

    return json.loads(path.read_text(encoding="utf-8"))


def ensure_dir(path: Path) -> Path:
    """
    Ensure directory exists, creating it if necessary.

    Parameters
    ----------
    path : Path
        Directory path.

    Returns
    -------
    Path
        The same path, for chaining.
    """
    path.mkdir(parents=True, exist_ok=True)
    return path


def undirected_edges(edge_index: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
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
