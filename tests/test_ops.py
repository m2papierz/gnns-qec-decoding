"""Verify ops.py functions produce correct results and match original implementations."""

import pytest
import torch
import torch.nn as nn

from gnn.models.ops import (
    Backend,
    fused_norm_residual_dropout,
    get_backend,
    set_backend,
    symmetric_edge_features,
)


def _make_graph(n_nodes: int = 10, n_edges: int = 20, hidden: int = 16):
    x = torch.randn(n_nodes, hidden)
    edge_index = torch.randint(0, n_nodes, (2, n_edges))
    edge_h = torch.randn(n_edges, hidden)
    return x, edge_index, edge_h


class TestSymmetricEdgeFeatures:
    def test_shape(self) -> None:
        x, ei, eh = _make_graph()
        out = symmetric_edge_features(x, ei, eh)
        assert out.shape == (ei.shape[1], 3 * x.shape[1])

    def test_content_matches_manual(self) -> None:
        x, ei, eh = _make_graph()
        out = symmetric_edge_features(x, ei, eh)
        H = x.shape[1]

        h_src = x[ei[0]]
        h_dst = x[ei[1]]
        expected = torch.cat([h_src + h_dst, (h_src - h_dst).abs(), eh], dim=-1)
        torch.testing.assert_close(out, expected)

    def test_symmetry_under_edge_reversal(self) -> None:
        """Sum and |diff| components are symmetric; edge_h is not reversed."""
        x, ei, eh = _make_graph()
        H = x.shape[1]
        out = symmetric_edge_features(x, ei, eh)

        ei_rev = torch.stack([ei[1], ei[0]], dim=0)
        out_rev = symmetric_edge_features(x, ei_rev, eh)

        # sum(src, dst) symmetric
        torch.testing.assert_close(out[:, :H], out_rev[:, :H])
        # |src - dst| symmetric
        torch.testing.assert_close(out[:, H : 2 * H], out_rev[:, H : 2 * H])

    def test_zero_edges(self) -> None:
        x = torch.randn(5, 8)
        ei = torch.zeros(2, 0, dtype=torch.long)
        eh = torch.zeros(0, 8)
        out = symmetric_edge_features(x, ei, eh)
        assert out.shape == (0, 24)

    def test_self_loops(self) -> None:
        x = torch.randn(4, 8)
        ei = torch.tensor([[0, 1, 2], [0, 1, 2]])  # all self-loops
        eh = torch.randn(3, 8)
        out = symmetric_edge_features(x, ei, eh)
        H = 8
        # For self-loop: src == dst, so |diff| == 0 and sum == 2*x[i]
        for i in range(3):
            node = ei[0, i].item()
            torch.testing.assert_close(out[i, :H], 2 * x[node])
            torch.testing.assert_close(out[i, H : 2 * H], torch.zeros(H))

    def test_odd_hidden_dim(self) -> None:
        """Ensure no crash when hidden_dim is not divisible by 4."""
        x, ei, eh = _make_graph(hidden=7)
        out = symmetric_edge_features(x, ei, eh)
        assert out.shape == (ei.shape[1], 21)


class TestFusedNormResidualDropout:
    def test_eval_matches_norm_plus_residual(self) -> None:
        x = torch.randn(8, 16)
        residual = torch.randn(8, 16)
        norm = nn.LayerNorm(16)
        dropout = nn.Dropout(0.1)

        result = fused_norm_residual_dropout(x, residual, norm, dropout, training=False)
        expected = norm(x) + residual
        torch.testing.assert_close(result, expected)

    def test_training_mode_has_dropout_effect(self) -> None:
        """With high dropout, many elements should be zeroed before residual."""
        torch.manual_seed(0)
        x = torch.randn(100, 32)
        residual = torch.zeros(100, 32)
        norm = nn.LayerNorm(32)
        dropout = nn.Dropout(0.5)

        result = fused_norm_residual_dropout(x, residual, norm, dropout, training=True)
        normed = norm(x)
        # With p=0.5 and residual=0, roughly half elements should differ
        # from just norm(x) (they'll be scaled or zeroed)
        diff = (result - normed).abs().sum().item()
        assert diff > 0, "Dropout had no effect"


class TestBackendManagement:
    def test_default_is_pytorch(self) -> None:
        set_backend("pytorch")
        assert get_backend() == Backend.PYTORCH

    def test_set_compiled(self) -> None:
        set_backend("compiled")
        assert get_backend() == Backend.COMPILED
        set_backend("pytorch")  # restore

    def test_cuda_fallback_when_unavailable(self) -> None:
        """Requesting cuda falls back to pytorch if extension is missing,
        or stays on cuda if kernels are built."""
        set_backend("cuda")
        try:
            import kernels

            if kernels.AVAILABLE:
                assert get_backend() == Backend.CUDA
            else:
                assert get_backend() == Backend.PYTORCH
        except ImportError:
            assert get_backend() == Backend.PYTORCH
        set_backend("pytorch")  # restore

    def test_invalid_backend_raises(self) -> None:
        with pytest.raises(ValueError):
            set_backend("nonexistent")

    def test_case_insensitive(self) -> None:
        set_backend("PyTorch")
        assert get_backend() == Backend.PYTORCH
        set_backend("COMPILED")
        assert get_backend() == Backend.COMPILED
        set_backend("pytorch")  # restore


def _cuda_ops_available() -> bool:
    """Check if CUDA kernels are built and GPU is present."""
    if not torch.cuda.is_available():
        return False
    try:
        import kernels

        return kernels.AVAILABLE
    except ImportError:
        return False


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="no GPU",
)
class TestCUDABackendEquivalence:
    """Verify CUDA kernels match PyTorch reference."""

    _skip = pytest.mark.skipif(not _cuda_ops_available(), reason="CUDA ops not built")

    @_skip
    def test_symmetric_edge_features_matches(self) -> None:
        x, ei, eh = _make_graph()
        x, ei, eh = x.cuda(), ei.cuda(), eh.cuda()

        set_backend("pytorch")
        ref = symmetric_edge_features(x, ei, eh)

        set_backend("cuda")
        out = symmetric_edge_features(x, ei, eh)
        set_backend("pytorch")

        torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)

    @_skip
    def test_symmetric_edge_features_zero_edges(self) -> None:
        x = torch.randn(5, 8).cuda()
        ei = torch.zeros(2, 0, dtype=torch.long).cuda()
        eh = torch.zeros(0, 8).cuda()

        set_backend("cuda")
        out = symmetric_edge_features(x, ei, eh)
        set_backend("pytorch")

        assert out.shape == (0, 24)

    @_skip
    def test_symmetric_edge_features_odd_hidden(self) -> None:
        """hidden_dim=7 forces scalar path (not divisible by 4)."""
        x, ei, eh = _make_graph(hidden=7)
        x, ei, eh = x.cuda(), ei.cuda(), eh.cuda()

        set_backend("pytorch")
        ref = symmetric_edge_features(x, ei, eh)

        set_backend("cuda")
        out = symmetric_edge_features(x, ei, eh)
        set_backend("pytorch")

        torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)

    @_skip
    def test_symmetric_edge_features_self_loops(self) -> None:
        x = torch.randn(4, 16).cuda()
        ei = torch.tensor([[0, 1, 2], [0, 1, 2]], dtype=torch.long).cuda()
        eh = torch.randn(3, 16).cuda()

        set_backend("pytorch")
        ref = symmetric_edge_features(x, ei, eh)

        set_backend("cuda")
        out = symmetric_edge_features(x, ei, eh)
        set_backend("pytorch")

        torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)

    @_skip
    def test_symmetric_edge_features_single_edge(self) -> None:
        x = torch.randn(2, 32).cuda()
        ei = torch.tensor([[0], [1]], dtype=torch.long).cuda()
        eh = torch.randn(1, 32).cuda()

        set_backend("pytorch")
        ref = symmetric_edge_features(x, ei, eh)

        set_backend("cuda")
        out = symmetric_edge_features(x, ei, eh)
        set_backend("pytorch")

        torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)

    @_skip
    def test_symmetric_edge_features_large_hidden(self) -> None:
        """hidden_dim=256 to stress vectorized path."""
        x, ei, eh = _make_graph(n_nodes=20, n_edges=50, hidden=256)
        x, ei, eh = x.cuda(), ei.cuda(), eh.cuda()

        set_backend("pytorch")
        ref = symmetric_edge_features(x, ei, eh)

        set_backend("cuda")
        out = symmetric_edge_features(x, ei, eh)
        set_backend("pytorch")

        torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)

    @_skip
    def test_fused_norm_residual_dropout_matches(self) -> None:
        x = torch.randn(8, 16).cuda()
        residual = torch.randn(8, 16).cuda()
        norm = nn.LayerNorm(16).cuda()
        dropout = nn.Dropout(0.0)

        set_backend("pytorch")
        ref = fused_norm_residual_dropout(x, residual, norm, dropout, training=False)

        set_backend("cuda")
        out = fused_norm_residual_dropout(x, residual, norm, dropout, training=False)
        set_backend("pytorch")

        torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)

    @_skip
    def test_fused_norm_residual_dropout_single_row(self) -> None:
        x = torch.randn(1, 32).cuda()
        residual = torch.randn(1, 32).cuda()
        norm = nn.LayerNorm(32).cuda()
        dropout = nn.Dropout(0.0)

        set_backend("pytorch")
        ref = fused_norm_residual_dropout(x, residual, norm, dropout, training=False)

        set_backend("cuda")
        out = fused_norm_residual_dropout(x, residual, norm, dropout, training=False)
        set_backend("pytorch")

        torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)

    @_skip
    def test_fused_norm_residual_dropout_odd_hidden(self) -> None:
        x = torch.randn(4, 7).cuda()
        residual = torch.randn(4, 7).cuda()
        norm = nn.LayerNorm(7).cuda()
        dropout = nn.Dropout(0.0)

        set_backend("pytorch")
        ref = fused_norm_residual_dropout(x, residual, norm, dropout, training=False)

        set_backend("cuda")
        out = fused_norm_residual_dropout(x, residual, norm, dropout, training=False)
        set_backend("pytorch")

        torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)
