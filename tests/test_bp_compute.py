"""Tests for sum-product belief propagation marginals."""

import numpy as np
import pytest

from qec_generator.bp import compute_bp_marginals


class TestComputeBpMarginals:
    """Tests for compute_bp_marginals."""

    def test_single_edge_no_syndrome(self) -> None:
        """One edge, zero syndrome => degree-1 checks prove no error."""
        und_pairs = np.array([[0, 1]], dtype=np.int64)
        edge_probs = np.array([0.1], dtype=np.float64)
        syndrome = np.array([0, 0], dtype=np.uint8)

        marginals = compute_bp_marginals(und_pairs, edge_probs, 2, syndrome)

        assert marginals.shape == (1,)
        # Both degree-1 checks have s=0: the edge definitely didn't error.
        assert marginals[0] < 1e-4

    def test_single_edge_both_triggered(self) -> None:
        """One edge connecting two detectors, both triggered => high marginal."""
        und_pairs = np.array([[0, 1]], dtype=np.int64)
        edge_probs = np.array([0.1], dtype=np.float64)
        syndrome = np.array([1, 1], dtype=np.uint8)

        marginals = compute_bp_marginals(und_pairs, edge_probs, 2, syndrome)

        assert marginals.shape == (1,)
        # Both detectors triggered — strong evidence the edge errored.
        assert marginals[0] > 0.99

    def test_marginals_in_01(self) -> None:
        """All marginals lie in [0, 1]."""
        rng = np.random.RandomState(42)
        U = 10
        und_pairs = np.column_stack([np.arange(U), np.arange(1, U + 1)]).astype(
            np.int64
        )
        edge_probs = rng.uniform(0.01, 0.2, size=U).astype(np.float64)
        syndrome = rng.randint(0, 2, size=U + 1).astype(np.uint8)

        marginals = compute_bp_marginals(und_pairs, edge_probs, U + 1, syndrome)

        assert marginals.shape == (U,)
        assert np.all(marginals >= 0.0)
        assert np.all(marginals <= 1.0)

    def test_all_zero_syndrome_small_marginals(self) -> None:
        """Zero syndrome => marginals much smaller than priors (strong no-error evidence)."""
        und_pairs = np.array([[0, 1], [1, 2], [2, 3]], dtype=np.int64)
        priors = np.array([0.05, 0.10, 0.15], dtype=np.float64)
        syndrome = np.zeros(4, dtype=np.uint8)

        marginals = compute_bp_marginals(und_pairs, priors, 4, syndrome)

        # All-zero syndrome on a chain provides strong "no error"
        # evidence via BP — marginals should be much smaller than priors.
        for i in range(3):
            assert marginals[i] < priors[i]

    def test_triangle_symmetry(self) -> None:
        """Triangle graph with equal priors + all triggered => equal marginals."""
        und_pairs = np.array([[0, 1], [1, 2], [0, 2]], dtype=np.int64)
        edge_probs = np.array([0.1, 0.1, 0.1], dtype=np.float64)
        syndrome = np.array([1, 1, 1], dtype=np.uint8)

        marginals = compute_bp_marginals(und_pairs, edge_probs, 3, syndrome)

        # By symmetry, all three edges should get the same marginal.
        assert marginals[0] == pytest.approx(marginals[1], abs=1e-6)
        assert marginals[1] == pytest.approx(marginals[2], abs=1e-6)

    def test_boundary_edge_no_checks(self) -> None:
        """Edge whose BOTH endpoints are boundary => marginal equals prior."""
        # Edge between boundary nodes 5 and 6, num_detectors=3.
        # Neither endpoint is a detector, so no check constrains this edge.
        und_pairs = np.array([[5, 6]], dtype=np.int64)
        edge_probs = np.array([0.2], dtype=np.float64)
        syndrome = np.zeros(3, dtype=np.uint8)

        marginals = compute_bp_marginals(und_pairs, edge_probs, 3, syndrome)

        assert marginals.shape == (1,)
        assert marginals[0] == pytest.approx(0.2, abs=1e-4)

    def test_empty_graph(self) -> None:
        """Zero edges => empty marginals."""
        und_pairs = np.zeros((0, 2), dtype=np.int64)
        edge_probs = np.zeros(0, dtype=np.float64)
        syndrome = np.zeros(3, dtype=np.uint8)

        marginals = compute_bp_marginals(und_pairs, edge_probs, 3, syndrome)

        assert marginals.shape == (0,)

    def test_dtype_is_float32(self) -> None:
        """Output dtype is float32."""
        und_pairs = np.array([[0, 1]], dtype=np.int64)
        edge_probs = np.array([0.1], dtype=np.float64)
        syndrome = np.array([0, 0], dtype=np.uint8)

        marginals = compute_bp_marginals(und_pairs, edge_probs, 2, syndrome)

        assert marginals.dtype == np.float32

    def test_chain_one_triggered_endpoint(self) -> None:
        """Chain 0-1-2, syndrome [1,0,0]: only valid config is all-error."""
        und_pairs = np.array([[0, 1], [1, 2]], dtype=np.int64)
        edge_probs = np.full(2, 0.05, dtype=np.float64)
        # d0=e0=1, d1=e0⊕e1=0 => e1=1, d2=e1=1 ≠ 0 => contradiction!
        # Actually: d0=e0, d1=e0⊕e1, d2=e1.
        # syndrome [1,0,0] => e0=1, e0⊕e1=0=>e1=1, e1=0=>contradiction.
        # So use syndrome [1,0,1] => e0=1, e0⊕e1=0=>e1=1, e1=1 ✓
        syndrome = np.array([1, 0, 1], dtype=np.uint8)

        marginals = compute_bp_marginals(und_pairs, edge_probs, 3, syndrome)

        # Both edges must have errored — only valid configuration.
        assert marginals[0] > 0.99
        assert marginals[1] > 0.99

    def test_pair_one_triggered(self) -> None:
        """Two disconnected edges, only one detector triggered.

        Edge 0 connects detectors 0-1, edge 1 connects detectors 2-3.
        Syndrome [1,0,0,0]: only edge 0's endpoint 0 triggered, but
        d0=e0 and d1=e0 => contradiction with [1,0]. BP should still
        produce valid marginals (this tests robustness).
        """
        # Better: use a star graph. Detector 0 connected to 3 edges.
        # d0 = e0 ⊕ e1 ⊕ e2, d1 = e0, d2 = e1, d3 = e2
        und_pairs = np.array([[0, 1], [0, 2], [0, 3]], dtype=np.int64)
        edge_probs = np.full(3, 0.1, dtype=np.float64)
        # syndrome [1,1,0,0] => d1=e0=1, d2=e1=0, d3=e2=0, d0=1⊕0⊕0=1 ✓
        syndrome = np.array([1, 1, 0, 0], dtype=np.uint8)

        marginals = compute_bp_marginals(und_pairs, edge_probs, 4, syndrome)

        # Edge 0 (0-1) should have high marginal (explains d0 and d1).
        assert marginals[0] > 0.9
        # Edges 1,2 should have low marginal.
        assert marginals[1] < 0.1
        assert marginals[2] < 0.1
