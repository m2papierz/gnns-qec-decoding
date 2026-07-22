"""Detector graph construction for GNN-based QEC decoding.

``FiredDetectorGraph`` / ``build_fired_detector_graph``: fired detectors
only, complete graph with learned features. Primary representation for
GNN training and inference.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import stim


NODE_DIM: int = 6
"""Node feature dimensionality: x_norm, y_norm, t_norm, d_x, d_y, basis."""

EDGE_DIM: int = 5
"""Edge feature dimensionality: dx, dy, dt, euclidean, chebyshev."""


@dataclass(frozen=True, slots=True)
class CircuitMetadata:
    """Precomputed per-circuit metadata for the graph builder.

    Extracted once from a ``stim.Circuit`` via ``extract_circuit_metadata``
    and shared read-only across DataLoader workers.

    Parameters
    ----------
    detector_coords : ndarray, shape ``(D, 3)``, float64
        ``(x, y, t)`` coordinates for each detector.
    distance : int
        Code distance.
    rounds : int
        Number of syndrome measurement rounds.
    num_detectors : int
        Total detector count (must equal ``detector_coords.shape[0]``).
    """

    detector_coords: np.ndarray
    distance: int
    rounds: int
    num_detectors: int

    def __post_init__(self) -> None:
        if self.detector_coords.ndim != 2 or self.detector_coords.shape[1] < 3:
            raise ValueError(
                f"detector_coords must have shape (D, >=3), "
                f"got {self.detector_coords.shape}"
            )
        if self.detector_coords.shape[0] != self.num_detectors:
            raise ValueError(
                f"detector_coords rows {self.detector_coords.shape[0]} != "
                f"num_detectors {self.num_detectors}"
            )
        if self.distance < 1:
            raise ValueError(f"distance must be >= 1, got {self.distance}")
        if self.rounds < 1:
            raise ValueError(f"rounds must be >= 1, got {self.rounds}")


@dataclass(frozen=True, slots=True)
class FiredDetectorGraph:
    """Fired-detector complete graph with learned features.

    Parameters
    ----------
    node_features : ndarray, shape ``(N, 6)``, float32
        Per-node: ``[x_norm, y_norm, t_norm, d_x, d_y, basis]``.
        ``x_norm = x / (2d)``, ``y_norm = y / (2d)``, ``t_norm = t / r``.
        ``d_x = (x - d) / d`` (signed boundary distance, x-axis).
        ``d_y = (y - d) / d`` (signed boundary distance, y-axis).
        ``basis`` = stabilizer type flag (0 or 1).
    edge_index : ndarray, shape ``(2, E)``, int64
        Directed COO edges for the complete graph (both directions).
    edge_features : ndarray, shape ``(E, 5)``, float32
        Per-edge: ``[dx, dy, dt, euclidean, chebyshev]`` where deltas are
        normalized ``(dst - src)`` and distances are of normalized coords.
    num_fired : int
        Number of fired detectors (nodes in the graph).
    fired_indices : ndarray, shape ``(N,)``, int64
        Original detector indices that fired.
    """

    node_features: np.ndarray
    edge_index: np.ndarray
    edge_features: np.ndarray
    num_fired: int
    fired_indices: np.ndarray

    def __post_init__(self) -> None:
        N = self.num_fired
        E = N * (N - 1) if N > 1 else 0

        if self.node_features.shape != (N, NODE_DIM):
            raise ValueError(
                f"node_features shape {self.node_features.shape} != "
                f"expected ({N}, {NODE_DIM})"
            )
        if self.edge_index.shape != (2, E):
            raise ValueError(
                f"edge_index shape {self.edge_index.shape} != expected (2, {E})"
            )
        if self.edge_features.shape != (E, EDGE_DIM):
            raise ValueError(
                f"edge_features shape {self.edge_features.shape} != "
                f"expected ({E}, {EDGE_DIM})"
            )
        if self.fired_indices.shape != (N,):
            raise ValueError(
                f"fired_indices shape {self.fired_indices.shape} != " f"expected ({N},)"
            )


def extract_circuit_metadata(
    circuit: stim.Circuit,
    distance: int,
    rounds: int,
) -> CircuitMetadata:
    """Extract graph builder metadata from a Stim circuit.

    Parameters
    ----------
    circuit : stim.Circuit
        Circuit with detector coordinates annotated.
    distance : int
        Code distance (used for spatial normalization as ``2 * d``).
    rounds : int
        Number of syndrome measurement rounds (temporal normalization).

    Returns
    -------
    CircuitMetadata
    """
    coord_dict = circuit.get_detector_coordinates()
    num_det = circuit.num_detectors

    coords = np.zeros((num_det, 3), dtype=np.float64)
    for det_id, c in coord_dict.items():
        if 0 <= det_id < num_det:
            coords[det_id, : min(len(c), 3)] = c[:3]

    return CircuitMetadata(
        detector_coords=coords,
        distance=distance,
        rounds=rounds,
        num_detectors=num_det,
    )


_EMPTY_NODE_FEATURES = np.zeros((0, NODE_DIM), dtype=np.float32)
_EMPTY_EDGE_INDEX = np.zeros((2, 0), dtype=np.int64)
_EMPTY_EDGE_FEATURES = np.zeros((0, EDGE_DIM), dtype=np.float32)
_EMPTY_FIRED = np.zeros((0,), dtype=np.int64)


def build_fired_detector_graph(
    syndrome: np.ndarray,
    metadata: CircuitMetadata,
) -> FiredDetectorGraph:
    """Build a fired-detector complete graph from a syndrome vector.

    Parameters
    ----------
    syndrome : ndarray, shape ``(D,)``
        Binary syndrome bit-vector (1 = fired).
    metadata : CircuitMetadata
        Precomputed circuit metadata.

    Returns
    -------
    FiredDetectorGraph
        Complete graph over fired detectors with learned features.
        For empty syndromes (zero fired detectors), returns a graph
        with ``num_fired=0`` — the caller short-circuits to no-flip.
    """
    fired = np.flatnonzero(syndrome)
    N = len(fired)

    if N == 0:
        return FiredDetectorGraph(
            node_features=_EMPTY_NODE_FEATURES,
            edge_index=_EMPTY_EDGE_INDEX,
            edge_features=_EMPTY_EDGE_FEATURES,
            num_fired=0,
            fired_indices=_EMPTY_FIRED,
        )

    coords = metadata.detector_coords[fired]  # (N, 3)
    x = coords[:, 0]
    y = coords[:, 1]
    t = coords[:, 2]

    d = metadata.distance
    r = metadata.rounds
    spatial_scale = 2.0 * d
    temporal_scale = float(r)

    # Normalized coordinates
    x_norm = x / spatial_scale
    y_norm = y / spatial_scale
    t_norm = t / temporal_scale

    # Signed boundary distances: -1 at min boundary, +1 at max boundary
    d_x = (x - d) / d
    d_y = (y - d) / d

    # Basis flag: distinguishes X-check vs Z-check stabilizers
    basis = (((x + y) / 2).astype(np.intp) % 2).astype(np.float32)

    node_features = np.column_stack([x_norm, y_norm, t_norm, d_x, d_y, basis]).astype(
        np.float32
    )

    if N == 1:
        return FiredDetectorGraph(
            node_features=node_features,
            edge_index=np.zeros((2, 0), dtype=np.int64),
            edge_features=np.zeros((0, EDGE_DIM), dtype=np.float32),
            num_fired=1,
            fired_indices=fired.astype(np.int64),
        )

    # Complete graph edges via broadcasting (vectorized, no Python loop)
    idx = np.arange(N, dtype=np.int64)
    all_src = np.repeat(idx, N)  # (N*N,)
    all_dst = np.tile(idx, N)  # (N*N,)
    not_self = all_src != all_dst  # mask out diagonal
    src = all_src[not_self]  # (N*(N-1),)
    dst = all_dst[not_self]  # (N*(N-1),)
    edge_index = np.stack([src, dst], axis=0)

    # Edge features from normalized coordinates (vectorized)
    norm_coords = node_features[:, :3]  # (N, 3): x_norm, y_norm, t_norm
    src_coords = norm_coords[src]  # (E, 3)
    dst_coords = norm_coords[dst]  # (E, 3)
    delta = dst_coords - src_coords  # (E, 3): signed (dx, dy, dt)

    abs_delta = np.abs(delta)
    euclidean = np.sqrt((delta**2).sum(axis=1))
    chebyshev = abs_delta.max(axis=1)

    edge_features = np.column_stack([delta, euclidean, chebyshev]).astype(np.float32)

    return FiredDetectorGraph(
        node_features=node_features,
        edge_index=edge_index,
        edge_features=edge_features,
        num_fired=N,
        fired_indices=fired.astype(np.int64),
    )
