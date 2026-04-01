"""Tests for feature engineering: relative node features and edge features."""

import numpy as np
import pytest

from gnn.dataset import compute_inv_dist_sq, compute_relative_node_features


class TestComputeRelativeNodeFeatures:
    """Tests for d_horizontal, d_vertical, d_temporal node features."""

    def _make_coords(
        self,
        d: int = 5,
        r: int = 3,
        with_boundary: bool = True,
    ) -> tuple[np.ndarray, np.ndarray, int, int]:
        """Create synthetic coords mimicking Stim output for d, r."""
        xs = np.arange(0, 2 * d + 1, 2, dtype=np.float32)
        ys = np.arange(0, 2 * d + 1, 2, dtype=np.float32)
        ts = np.arange(0, r + 1, dtype=np.float32)
        det_coords = []
        for t in ts:
            for x in xs:
                for y in ys:
                    det_coords.append([x, y, t])
        det_coords = np.array(det_coords, dtype=np.float32)
        nd = len(det_coords)

        if with_boundary:
            bnd = np.full((1, 3), np.nan, dtype=np.float32)
            coords = np.concatenate([det_coords, bnd], axis=0)
            boundary = np.zeros(nd + 1, dtype=bool)
            boundary[nd] = True
            return coords, boundary, d, r
        else:
            boundary = np.zeros(nd, dtype=bool)
            return det_coords, boundary, d, r

    def test_output_shape(self) -> None:
        coords, bnd, d, r = self._make_coords(d=3, r=3)
        out = compute_relative_node_features(coords, bnd, d, r)
        assert out.shape == (len(coords), 4)
        assert out.dtype == np.float32

    def test_boundary_node_zeros(self) -> None:
        coords, bnd, d, r = self._make_coords(d=5, r=3)
        out = compute_relative_node_features(coords, bnd, d, r)
        bnd_idx = np.argmax(bnd)
        assert out[bnd_idx, 0] == 1.0  # is_boundary
        assert out[bnd_idx, 1] == 0.0
        assert out[bnd_idx, 2] == 0.0
        assert out[bnd_idx, 3] == 0.0

    def test_range_01(self) -> None:
        coords, bnd, d, r = self._make_coords(d=5, r=5)
        out = compute_relative_node_features(coords, bnd, d, r)
        assert out.min() >= 0.0
        assert out.max() <= 1.0

    def test_normalisation_is_distance_invariant(self) -> None:
        """d_horizontal=0.5 means 'middle of code' for any d."""
        for d in [3, 5, 7]:
            coords, bnd, _, r = self._make_coords(d=d, r=3)
            out = compute_relative_node_features(coords, bnd, d, r)
            mid_mask = (~bnd) & (coords[:, 0] == float(d))
            if mid_mask.any():
                idx = np.argmax(mid_mask)
                assert out[idx, 1] == pytest.approx(0.5, abs=1e-6)

    def test_corner_values(self) -> None:
        coords, bnd, d, r = self._make_coords(d=5, r=5)
        out = compute_relative_node_features(coords, bnd, d, r)
        assert out[0, 1] == pytest.approx(0.0, abs=1e-6)
        assert out[0, 2] == pytest.approx(0.0, abs=1e-6)
        assert out[0, 3] == pytest.approx(0.0, abs=1e-6)

    def test_max_coord_values(self) -> None:
        coords, bnd, d, r = self._make_coords(d=5, r=5)
        out = compute_relative_node_features(coords, bnd, d, r)
        det_mask = ~bnd
        max_mask = (
            det_mask
            & (coords[:, 0] == 2.0 * d)
            & (coords[:, 1] == 2.0 * d)
            & (coords[:, 2] == float(r))
        )
        if max_mask.any():
            idx = np.argmax(max_mask)
            assert out[idx, 1] == pytest.approx(1.0, abs=1e-6)
            assert out[idx, 2] == pytest.approx(1.0, abs=1e-6)
            assert out[idx, 3] == pytest.approx(1.0, abs=1e-6)

    def test_is_boundary_flag(self) -> None:
        coords, bnd, d, r = self._make_coords(d=3, r=3)
        out = compute_relative_node_features(coords, bnd, d, r)
        np.testing.assert_array_equal(out[:, 0], bnd.astype(np.float32))

    def test_no_boundary(self) -> None:
        coords, bnd, d, r = self._make_coords(d=3, r=3, with_boundary=False)
        out = compute_relative_node_features(coords, bnd, d, r)
        assert out[:, 0].sum() == 0.0


class TestComputeInvDistSq:
    """Tests for squared inverse Chebyshev edge features."""

    def _simple_graph(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        coords = np.array(
            [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [np.nan, np.nan, np.nan]],
            dtype=np.float32,
        )
        boundary = np.array([False, False, True])
        edge_index = np.array([[0, 1, 0, 2, 1, 2], [1, 0, 2, 0, 2, 1]], dtype=np.int64)
        return edge_index, coords, boundary

    def test_output_shape(self) -> None:
        ei, coords, bnd = self._simple_graph()
        result = compute_inv_dist_sq(ei, coords, bnd)
        assert result.shape == (ei.shape[1],)
        assert result.dtype == np.float32

    def test_boundary_edges_are_zero(self) -> None:
        ei, coords, bnd = self._simple_graph()
        result = compute_inv_dist_sq(ei, coords, bnd)
        assert result[2] == 0.0
        assert result[3] == 0.0
        assert result[4] == 0.0
        assert result[5] == 0.0

    def test_detector_edge_positive(self) -> None:
        ei, coords, bnd = self._simple_graph()
        result = compute_inv_dist_sq(ei, coords, bnd)
        assert result[0] > 0.0
        assert result[1] > 0.0

    def test_chebyshev_formula(self) -> None:
        coords = np.array(
            [[0, 0, 0], [2, 0, 0], [0, 0, 1], [4, 2, 1]], dtype=np.float32
        )
        boundary = np.zeros(4, dtype=bool)
        ei = np.array([[0, 0, 0], [1, 2, 3]], dtype=np.int64)
        result = compute_inv_dist_sq(ei, coords, boundary)
        assert result[0] == pytest.approx(1.0 / 4.0, abs=1e-6)
        assert result[1] == pytest.approx(1.0, abs=1e-6)
        assert result[2] == pytest.approx(1.0 / 16.0, abs=1e-6)

    def test_bidirectional_symmetry(self) -> None:
        ei, coords, bnd = self._simple_graph()
        result = compute_inv_dist_sq(ei, coords, bnd)
        assert result[0] == result[1]

    def test_empty_edges(self) -> None:
        coords = np.array([[0, 0, 0]], dtype=np.float32)
        bnd = np.array([False])
        ei = np.zeros((2, 0), dtype=np.int64)
        result = compute_inv_dist_sq(ei, coords, bnd)
        assert result.shape == (0,)

    def test_realistic_chebyshev_values(self) -> None:
        coords = np.array(
            [[0, 0, 0], [0, 0, 1], [2, 2, 0], [0, 0, 4]], dtype=np.float32
        )
        boundary = np.zeros(4, dtype=bool)
        ei = np.array([[0, 0, 0], [1, 2, 3]], dtype=np.int64)
        result = compute_inv_dist_sq(ei, coords, boundary)
        assert result[0] == pytest.approx(1.0, abs=1e-5)
        assert result[1] == pytest.approx(0.25, abs=1e-5)
        assert result[2] == pytest.approx(0.0625, abs=1e-5)
