"""Tests for the BP+OSD decoder."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.sparse import issparse

from decoders.base import DecoderConfig
from decoders.bp_osd import _build_parity_check_matrix


class TestBuildParityCheckMatrix:
    """Tests for _build_parity_check_matrix."""

    def test_simple_chain(self) -> None:
        """Chain 0—1—2: H has shape (3, 2), correct entries."""
        und_pairs = np.array([[0, 1], [1, 2]], dtype=np.int64)
        H = _build_parity_check_matrix(und_pairs, num_detectors=3)

        assert issparse(H)
        assert H.shape == (3, 2)

        Hd = H.toarray()
        # Edge 0 connects detectors 0, 1
        assert Hd[0, 0] == 1
        assert Hd[1, 0] == 1
        assert Hd[2, 0] == 0
        # Edge 1 connects detectors 1, 2
        assert Hd[1, 1] == 1
        assert Hd[2, 1] == 1
        assert Hd[0, 1] == 0

    def test_boundary_excluded(self) -> None:
        """Boundary node (idx >= num_detectors) produces no row entry."""
        # Edge 0—1 (both detectors), edge 1—3 (3 is boundary, num_det=3)
        und_pairs = np.array([[0, 1], [1, 3]], dtype=np.int64)
        H = _build_parity_check_matrix(und_pairs, num_detectors=3)

        assert H.shape == (3, 2)
        Hd = H.toarray()
        # Edge 1 only touches detector 1 (boundary node 3 excluded).
        assert Hd[1, 1] == 1
        assert Hd[3 - 1, 1] == 0  # detector 2 not connected
        # No row for node 3 (it's beyond num_detectors).

    def test_empty(self) -> None:
        """Zero edges =>shape (num_det, 0)."""
        und_pairs = np.zeros((0, 2), dtype=np.int64)
        H = _build_parity_check_matrix(und_pairs, num_detectors=4)
        assert H.shape == (4, 0)

    def test_single_edge(self) -> None:
        """One edge =>shape (num_det, 1), two nonzeros."""
        und_pairs = np.array([[2, 5]], dtype=np.int64)
        H = _build_parity_check_matrix(und_pairs, num_detectors=8)
        assert H.shape == (8, 1)
        assert H.nnz == 2
        Hd = H.toarray()
        assert Hd[2, 0] == 1
        assert Hd[5, 0] == 1

    def test_syndrome_check(self) -> None:
        """H @ error should reproduce syndrome (mod 2) for a simple case."""
        # Triangle: edges 0-1, 1-2, 0-2
        und_pairs = np.array([[0, 1], [1, 2], [0, 2]], dtype=np.int64)
        H = _build_parity_check_matrix(und_pairs, num_detectors=3)

        # Single error on edge 0 (0-1): detectors 0 and 1 trigger.
        error = np.array([1, 0, 0], dtype=np.int32)
        syndrome = (H.toarray() @ error) % 2
        np.testing.assert_array_equal(syndrome, [1, 1, 0])


@pytest.fixture
def bp_osd_config() -> DecoderConfig:
    """Minimal config for BP+OSD tests."""
    return DecoderConfig(
        num_detectors=4,
        num_observables=1,
        has_boundary=True,
        boundary_node=4,
    )


@pytest.fixture
def bp_osd_graph() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Small graph with edge probs and observable flips.

    Returns ``(und_pairs, edge_probs, observable_flips_und)``.
    """
    #  0 — 1 — 2 — 3, plus 0 — boundary(4)
    und_pairs = np.array(
        [[0, 1], [1, 2], [2, 3], [0, 4]],
        dtype=np.int64,
    )
    edge_probs = np.array([0.05, 0.05, 0.05, 0.05], dtype=np.float64)
    # Only boundary edge flips observable 0
    obs_flips = np.array(
        [[False], [False], [False], [True]],
        dtype=bool,
    )
    return und_pairs, edge_probs, obs_flips


class TestBPOSDDecoder:
    """Integration tests — require cudaq-qec runtime."""

    def test_decode_shape(
        self,
        bp_osd_config: DecoderConfig,
        bp_osd_graph: tuple,
    ) -> None:
        from decoders.bp_osd import BPOSDDecoder

        und_pairs, probs, obs_flips = bp_osd_graph
        decoder = BPOSDDecoder(bp_osd_config, und_pairs, probs, obs_flips)

        syndrome = np.zeros(4, dtype=np.uint8)
        result = decoder.decode(syndrome)
        assert result.shape == (1,)
        assert result.dtype == np.uint8

    def test_decode_batch_shape(
        self,
        bp_osd_config: DecoderConfig,
        bp_osd_graph: tuple,
    ) -> None:
        from decoders.bp_osd import BPOSDDecoder

        und_pairs, probs, obs_flips = bp_osd_graph
        decoder = BPOSDDecoder(bp_osd_config, und_pairs, probs, obs_flips)

        syndromes = np.zeros((10, 4), dtype=np.uint8)
        results = decoder.decode_batch(syndromes)
        assert results.shape == (10, 1)

    def test_zero_syndrome_gives_zero(
        self,
        bp_osd_config: DecoderConfig,
        bp_osd_graph: tuple,
    ) -> None:
        from decoders.bp_osd import BPOSDDecoder

        und_pairs, probs, obs_flips = bp_osd_graph
        decoder = BPOSDDecoder(bp_osd_config, und_pairs, probs, obs_flips)

        result = decoder.decode(np.zeros(4, dtype=np.uint8))
        np.testing.assert_array_equal(result, [0])

    def test_from_gnn_logits(
        self,
        bp_osd_config: DecoderConfig,
        bp_osd_graph: tuple,
    ) -> None:
        from decoders.bp_osd import BPOSDDecoder

        und_pairs, _, obs_flips = bp_osd_graph
        # 4 undirected edges => 8 directed edges
        directed_logits = np.array(
            [-2.0, -2.0, -1.5, -1.5, -1.0, -1.0, 0.5, 0.5],
            dtype=np.float64,
        )
        dir_to_undir = np.array([0, 0, 1, 1, 2, 2, 3, 3], dtype=np.int64)

        decoder = BPOSDDecoder.from_gnn_logits(
            bp_osd_config,
            und_pairs,
            directed_logits,
            dir_to_undir,
            num_undirected=4,
            observable_flips=obs_flips,
        )
        assert isinstance(decoder, BPOSDDecoder)
        result = decoder.decode(np.zeros(4, dtype=np.uint8))
        assert result.shape == (1,)

    def test_name(
        self,
        bp_osd_config: DecoderConfig,
        bp_osd_graph: tuple,
    ) -> None:
        from decoders.bp_osd import BPOSDDecoder

        und_pairs, probs, obs_flips = bp_osd_graph
        decoder = BPOSDDecoder(bp_osd_config, und_pairs, probs, obs_flips)
        assert decoder.name == "BPOSDDecoder"


class TestBPOSDValidation:
    """Input-validation tests that do not require the cudaq runtime."""

    def test_observable_flips_shape_mismatch_raises(self) -> None:
        """Mismatched observable_flips rows should raise immediately."""
        from decoders.bp_osd import BPOSDDecoder

        config = DecoderConfig(
            num_detectors=3,
            num_observables=1,
            has_boundary=False,
        )
        und_pairs = np.array([[0, 1], [1, 2]], dtype=np.int64)
        probs = np.array([0.05, 0.05], dtype=np.float64)
        bad_flips = np.array([[False], [False], [False]], dtype=bool)  # 3 != 2

        with pytest.raises(ValueError, match="observable_flips rows"):
            BPOSDDecoder(config, und_pairs, probs, bad_flips)
