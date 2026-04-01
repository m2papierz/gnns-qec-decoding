"""Tests for the data generation and packaging pipeline."""

import numpy as np
import pytest
import stim

from qec_generator.graph import build_detector_graph
from qec_generator.utils import undirected_edges


class TestUndirectedEdges:
    """Mapping from directed COO edges to unique undirected pairs."""

    def test_simple_bidirectional(self) -> None:
        """Two directed edges (0=>1, 1=>0) collapse to one undirected."""
        ei = np.array([[0, 1], [1, 0]], dtype=np.int64)
        pairs, mapping = undirected_edges(ei)

        assert pairs.shape == (1, 2)
        np.testing.assert_array_equal(pairs[0], [0, 1])
        np.testing.assert_array_equal(mapping, [0, 0])

    def test_multiple_edges(self) -> None:
        """Four directed edges from two undirected edges."""
        ei = np.array([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=np.int64)
        pairs, mapping = undirected_edges(ei)

        assert pairs.shape == (2, 2)
        assert mapping.shape == (4,)
        # Both directions of an edge map to same undirected index
        assert mapping[0] == mapping[1]  # 0 <-> 1
        assert mapping[2] == mapping[3]  # 1 <-> 2
        assert mapping[0] != mapping[2]  # different edges

    def test_canonical_ordering(self) -> None:
        """Undirected pairs always have smaller node first."""
        ei = np.array([[5, 3, 3, 5], [3, 5, 5, 3]], dtype=np.int64)
        pairs, _ = undirected_edges(ei)

        assert pairs.shape == (1, 2)
        assert pairs[0, 0] < pairs[0, 1]

    def test_self_loop(self) -> None:
        """Self-loops collapse to a single undirected edge."""
        ei = np.array([[2, 2], [2, 2]], dtype=np.int64)
        pairs, mapping = undirected_edges(ei)

        assert pairs.shape == (1, 2)
        np.testing.assert_array_equal(pairs[0], [2, 2])
        np.testing.assert_array_equal(mapping, [0, 0])

    def test_empty(self) -> None:
        """Zero edges => empty output."""
        ei = np.zeros((2, 0), dtype=np.int64)
        pairs, mapping = undirected_edges(ei)

        assert pairs.shape[0] == 0
        assert mapping.shape[0] == 0

    def test_dtypes(self) -> None:
        """Output dtypes are int64."""
        ei = np.array([[0, 1], [1, 0]], dtype=np.int32)
        pairs, mapping = undirected_edges(ei)

        assert pairs.dtype == np.int64
        assert mapping.dtype == np.int64

    def test_roundtrip_with_edge_attr(self) -> None:
        """Mapping correctly indexes undirected attributes back to directed."""
        ei = np.array([[0, 1, 1, 2, 2, 0], [1, 0, 2, 1, 0, 2]], dtype=np.int64)
        pairs, d2u = undirected_edges(ei)

        # Simulate per-undirected-edge weights
        und_weights = np.array([0.1, 0.2, 0.3])
        assert und_weights.shape[0] == pairs.shape[0]

        # Expand back to directed: both directions get same weight
        dir_weights = und_weights[d2u]
        assert dir_weights.shape[0] == 6
        # Edge 0=>1 and 1=>0 must have same weight
        assert dir_weights[0] == dir_weights[1]
        assert dir_weights[2] == dir_weights[3]
        assert dir_weights[4] == dir_weights[5]


class TestBuildDetectorGraph:
    """Stim DEM => GNN input graph construction."""

    @pytest.fixture
    def d3_circuit(self) -> stim.Circuit:
        """Minimal d=3 rotated surface code circuit."""
        return stim.Circuit.generated(
            "surface_code:rotated_memory_x",
            distance=3,
            rounds=3,
            after_clifford_depolarization=0.01,
            after_reset_flip_probability=0.01,
            before_measure_flip_probability=0.01,
            before_round_data_depolarization=0.01,
        )

    def test_basic_properties(self, d3_circuit: stim.Circuit) -> None:
        dem = d3_circuit.detector_error_model(decompose_errors=True)
        graph = build_detector_graph(d3_circuit, dem)

        assert graph.num_detectors == dem.num_detectors
        assert graph.num_observables == dem.num_observables
        assert graph.num_nodes == graph.num_detectors + (1 if graph.has_boundary else 0)

    def test_edges_are_bidirectional(self, d3_circuit: stim.Circuit) -> None:
        """Every (u,v) has a matching (v,u)."""
        dem = d3_circuit.detector_error_model(decompose_errors=True)
        graph = build_detector_graph(d3_circuit, dem)

        ei = graph.edge_index
        forward = set(zip(ei[0].tolist(), ei[1].tolist()))
        reverse = set(zip(ei[1].tolist(), ei[0].tolist()))
        assert forward == reverse

    def test_edge_count_is_even(self, d3_circuit: stim.Circuit) -> None:
        """Bidirectional edges means even total count."""
        dem = d3_circuit.detector_error_model(decompose_errors=True)
        graph = build_detector_graph(d3_circuit, dem)

        assert graph.edge_index.shape[1] % 2 == 0

    def test_edge_attributes_positive(self, d3_circuit: stim.Circuit) -> None:
        dem = d3_circuit.detector_error_model(decompose_errors=True)
        graph = build_detector_graph(d3_circuit, dem)

        assert np.all(graph.edge_error_prob > 0)
        assert np.all(graph.edge_error_prob < 1)
        assert np.all(graph.edge_weight > 0)

    def test_boundary_node_is_last(self, d3_circuit: stim.Circuit) -> None:
        dem = d3_circuit.detector_error_model(decompose_errors=True)
        graph = build_detector_graph(d3_circuit, dem, include_boundary=True)

        if graph.has_boundary:
            assert graph.node_is_boundary[-1] is True or graph.node_is_boundary[-1]
            assert graph.node_is_boundary[:-1].sum() == 0

    def test_no_boundary_option(self, d3_circuit: stim.Circuit) -> None:
        dem = d3_circuit.detector_error_model(decompose_errors=True)
        graph = build_detector_graph(d3_circuit, dem, include_boundary=False)

        assert graph.has_boundary is False
        assert graph.num_nodes == graph.num_detectors

    def test_node_coords_shape(self, d3_circuit: stim.Circuit) -> None:
        dem = d3_circuit.detector_error_model(decompose_errors=True)
        graph = build_detector_graph(d3_circuit, dem)

        assert graph.node_coords.shape[0] == graph.num_nodes

    def test_no_duplicate_edges(self, d3_circuit: stim.Circuit) -> None:
        """No repeated (u,v) pairs after boundary collapse."""
        dem = d3_circuit.detector_error_model(decompose_errors=True)
        graph = build_detector_graph(d3_circuit, dem)

        ei = graph.edge_index
        edge_set = set(zip(ei[0].tolist(), ei[1].tolist()))
        assert len(edge_set) == ei.shape[1]

    def test_node_indices_in_range(self, d3_circuit: stim.Circuit) -> None:
        dem = d3_circuit.detector_error_model(decompose_errors=True)
        graph = build_detector_graph(d3_circuit, dem)

        assert graph.edge_index.min() >= 0
        assert graph.edge_index.max() < graph.num_nodes

    def test_scales_with_distance(self) -> None:
        """d=5 graph has more nodes and edges than d=3."""
        kwargs = dict(
            after_clifford_depolarization=0.01,
            after_reset_flip_probability=0.01,
            before_measure_flip_probability=0.01,
            before_round_data_depolarization=0.01,
        )
        c3 = stim.Circuit.generated(
            "surface_code:rotated_memory_x",
            distance=3,
            rounds=3,
            **kwargs,
        )
        c5 = stim.Circuit.generated(
            "surface_code:rotated_memory_x",
            distance=5,
            rounds=3,
            **kwargs,
        )
        dem3 = c3.detector_error_model(decompose_errors=True)
        dem5 = c5.detector_error_model(decompose_errors=True)
        g3 = build_detector_graph(c3, dem3)
        g5 = build_detector_graph(c5, dem5)

        assert g5.num_detectors > g3.num_detectors
        assert g5.edge_index.shape[1] > g3.edge_index.shape[1]

    def test_observable_flips_present(self, d3_circuit: stim.Circuit) -> None:
        """Graph has observable_flips array."""
        dem = d3_circuit.detector_error_model(decompose_errors=True)
        graph = build_detector_graph(d3_circuit, dem)

        assert graph.observable_flips is not None

    def test_observable_flips_shape(self, d3_circuit: stim.Circuit) -> None:
        """observable_flips has shape (E, num_observables)."""
        dem = d3_circuit.detector_error_model(decompose_errors=True)
        graph = build_detector_graph(d3_circuit, dem)

        E = graph.edge_index.shape[1]
        assert graph.observable_flips is not None
        assert graph.observable_flips.shape == (E, graph.num_observables)
        assert graph.observable_flips.dtype == bool

    def test_observable_flips_not_all_false(self, d3_circuit: stim.Circuit) -> None:
        """At least some edges flip an observable (surface code has boundary errors)."""
        dem = d3_circuit.detector_error_model(decompose_errors=True)
        graph = build_detector_graph(d3_circuit, dem)

        assert graph.observable_flips is not None
        assert (
            graph.observable_flips.any()
        ), "Expected at least one edge to flip an observable"

    def test_observable_flips_bidirectional(self, d3_circuit: stim.Circuit) -> None:
        """Both directions of an undirected edge have identical observable_flips."""
        dem = d3_circuit.detector_error_model(decompose_errors=True)
        graph = build_detector_graph(d3_circuit, dem)

        assert graph.observable_flips is not None
        ei = graph.edge_index
        obs = graph.observable_flips
        # Build lookup: (u,v) -> obs_flips row index
        edge_to_idx: dict[tuple[int, int], int] = {}
        for e in range(ei.shape[1]):
            edge_to_idx[(int(ei[0, e]), int(ei[1, e]))] = e

        for e in range(ei.shape[1]):
            u, v = int(ei[0, e]), int(ei[1, e])
            rev = edge_to_idx.get((v, u))
            assert rev is not None, f"Missing reverse edge for ({u}, {v})"
            np.testing.assert_array_equal(
                obs[e],
                obs[rev],
                err_msg=f"Mismatch for edge ({u},{v}) vs ({v},{u})",
            )
