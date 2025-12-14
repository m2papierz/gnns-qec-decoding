"""Detector graph construction for GNN-based QEC decoding."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import networkx as nx
import numpy as np
import pymatching
import stim


@dataclass(frozen=True)
class DetectorGraph:
    """
    Detector graph for GNN-based decoding.

    Represents a quantum error correction decoding problem as a graph where
    nodes correspond to syndrome detectors and edges represent error correlations.

    Attributes
    ----------
    edge_index : ndarray, shape (2, E), int64
        Directed COO edges (bidirectional: both u→v and v→u stored).
    edge_error_prob : ndarray, shape (E,), float32
        Per-edge error probability.
    edge_weight : ndarray, shape (E,), float32
        Per-edge MWPM weight (typically -log(p/(1-p))).
    node_coords : ndarray, shape (N, C), float32
        Node coordinates (NaN if unavailable).
    node_is_boundary : ndarray, shape (N,), bool
        True for virtual boundary node.
    num_nodes : int
        Total nodes (detectors + optional boundary).
    num_detectors : int
        Number of detector nodes.
    num_observables : int
        Number of logical observables.
    has_boundary : bool
        Whether a virtual boundary node is included.
    """

    edge_index: np.ndarray
    edge_error_prob: np.ndarray
    edge_weight: np.ndarray
    node_coords: np.ndarray
    node_is_boundary: np.ndarray
    num_nodes: int
    num_detectors: int
    num_observables: int
    has_boundary: bool


def build_detector_graph(
    circuit: stim.Circuit,
    dem: stim.DetectorErrorModel,
    include_boundary: bool = True,
) -> DetectorGraph:
    """
    Build a detector graph from a Stim circuit and error model.

    Constructs a graph representation suitable for GNN-based decoding by:
    1. Converting the detector error model to a matching graph
    2. Collapsing boundary nodes into a single virtual node (optional)
    3. Extracting node coordinates from circuit
    4. Creating bidirectional edge representation

    Parameters
    ----------
    circuit : stim.Circuit
        Circuit for detector coordinates.
    dem : stim.DetectorErrorModel
        Graphlike detector error model from
        circuit.detector_error_model(decompose_errors=True).
    include_boundary : bool, default=True
        Include virtual boundary node if PyMatching reports boundary edges.

    Returns
    -------
    DetectorGraph
        Graph suitable for GNN input with bidirectional edges.

    Raises
    ------
    ValueError
        If detector node IDs are invalid or edge attributes are missing.

    Notes
    -----
    The graph is stored in COO (coordinate) format with both directions
    of each edge explicitly stored (u→v and v→u).

    Examples
    --------
    >>> circuit = stim.Circuit.generated("surface_code:rotated_memory_x", ...)
    >>> dem = circuit.detector_error_model(decompose_errors=True)
    >>> graph = build_detector_graph(circuit, dem)
    >>> graph.num_nodes
    42
    """
    num_det = dem.num_detectors
    num_obs = dem.num_observables

    matching = pymatching.Matching.from_detector_error_model(dem)
    G: nx.Graph = matching.to_networkx()

    def is_boundary(n: Any) -> bool:
        """Check if node is marked as boundary in PyMatching graph."""
        return bool(G.nodes[n].get("is_boundary", False))

    # Validate detector nodes
    detector_nodes = [n for n in G.nodes if not is_boundary(n)]
    invalid = [
        n
        for n in detector_nodes
        if not (isinstance(n, (int, np.integer)) and 0 <= n < num_det)
    ]
    if invalid:
        raise ValueError(f"Invalid detector node IDs: {invalid[:5]}")

    boundary_nodes = [n for n in G.nodes if is_boundary(n)]
    has_boundary = include_boundary and len(boundary_nodes) > 0
    boundary_idx = num_det if has_boundary else None

    # Map nodes to contiguous indices
    node_map: dict[Any, int | None] = {}
    for n in G.nodes:
        if not is_boundary(n):
            node_map[n] = int(n)
        elif has_boundary:
            node_map[n] = boundary_idx  # Collapse all boundary nodes
        else:
            node_map[n] = None

    num_nodes = num_det + (1 if has_boundary else 0)

    # Extract node coordinates from Stim circuit
    coord_dict = circuit.get_detector_coordinates()
    coord_dim = max((len(v) for v in coord_dict.values()), default=0)
    node_coords = np.full((num_nodes, coord_dim), np.nan, dtype=np.float32)
    for det_id, coords in coord_dict.items():
        if 0 <= det_id < num_det:
            node_coords[det_id, : len(coords)] = coords

    node_is_boundary = np.zeros(num_nodes, dtype=bool)
    if has_boundary:
        node_is_boundary[boundary_idx] = True

    # Deduplicate edges after boundary collapse (keep smallest weight)
    best: dict[tuple[int, int], tuple[float, float]] = {}
    for u, v, data in G.edges(data=True):
        iu, iv = node_map.get(u), node_map.get(v)
        if iu is None or iv is None or iu == iv:
            continue

        p = data.get("error_probability")
        if p is None:
            raise ValueError("Missing 'error_probability' on edge")
        w = data.get("weight", np.nan)

        key = (min(iu, iv), max(iu, iv))
        if key not in best or (
            not np.isnan(w) and (np.isnan(best[key][1]) or w < best[key][1])
        ):
            best[key] = (float(p), float(w))

    # Expand to directed COO format (bidirectional)
    edges, probs, weights = [], [], []
    for (a, b), (p, w) in best.items():
        edges.extend([(a, b), (b, a)])
        probs.extend([p, p])
        weights.extend([w, w])

    edge_index = (
        np.array(edges, dtype=np.int64).T if edges else np.zeros((2, 0), dtype=np.int64)
    )

    return DetectorGraph(
        edge_index=edge_index,
        edge_error_prob=np.array(probs, dtype=np.float32),
        edge_weight=np.array(weights, dtype=np.float32),
        node_coords=node_coords,
        node_is_boundary=node_is_boundary,
        num_nodes=num_nodes,
        num_detectors=num_det,
        num_observables=num_obs,
        has_boundary=has_boundary,
    )
