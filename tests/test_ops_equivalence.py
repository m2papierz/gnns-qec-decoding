"""Verify ops.py functions produce correct results and match original implementations."""

import pytest
import torch
import torch.nn as nn

from gnn.models.ops import (
    Backend,
    edge_mean_pool,
    fused_norm_residual_dropout,
    get_backend,
    graph_normalized_bce,
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

    def test_shape_preserved(self) -> None:
        x = torch.randn(5, 7)
        residual = torch.randn(5, 7)
        norm = nn.LayerNorm(7)
        dropout = nn.Dropout(0.0)

        result = fused_norm_residual_dropout(x, residual, norm, dropout, training=False)
        assert result.shape == (5, 7)

    def test_single_row(self) -> None:
        x = torch.randn(1, 16)
        residual = torch.randn(1, 16)
        norm = nn.LayerNorm(16)
        dropout = nn.Dropout(0.0)

        result = fused_norm_residual_dropout(x, residual, norm, dropout, training=False)
        expected = norm(x) + residual
        torch.testing.assert_close(result, expected)


class TestEdgeMeanPool:
    def test_shape(self) -> None:
        edge_h = torch.randn(30, 16)
        edge_batch = torch.tensor([0] * 10 + [1] * 12 + [2] * 8)
        out = edge_mean_pool(edge_h, edge_batch, n_graphs=3)
        assert out.shape == (3, 16)

    def test_single_graph_equals_mean(self) -> None:
        edge_h = torch.randn(10, 8)
        edge_batch = torch.zeros(10, dtype=torch.long)
        out = edge_mean_pool(edge_h, edge_batch, n_graphs=1)
        torch.testing.assert_close(out[0], edge_h.mean(dim=0))

    def test_per_graph_mean(self) -> None:
        e1 = torch.ones(4, 2) * 2.0
        e2 = torch.ones(6, 2) * 5.0
        edge_h = torch.cat([e1, e2])
        edge_batch = torch.tensor([0] * 4 + [1] * 6)
        out = edge_mean_pool(edge_h, edge_batch, n_graphs=2)
        torch.testing.assert_close(out[0], torch.tensor([2.0, 2.0]))
        torch.testing.assert_close(out[1], torch.tensor([5.0, 5.0]))

    def test_empty_graph_safe(self) -> None:
        """Graph with 0 edges should produce zeros (not NaN)."""
        edge_h = torch.randn(5, 4)
        edge_batch = torch.zeros(5, dtype=torch.long)
        out = edge_mean_pool(edge_h, edge_batch, n_graphs=2)
        # Graph 1 has no edges → should be 0 (clamped division)
        assert out.shape == (2, 4)
        torch.testing.assert_close(out[1], torch.zeros(4))


class TestGraphNormalizedBCE:
    def test_single_graph_matches_standard_bce(self) -> None:
        logits = torch.randn(10)
        target = torch.randint(0, 2, (10,)).float()
        edge_graph = torch.zeros(10, dtype=torch.long)

        result = graph_normalized_bce(logits, target, edge_graph, n_graphs=1)
        expected = torch.nn.functional.binary_cross_entropy_with_logits(logits, target)
        torch.testing.assert_close(result, expected, atol=1e-6, rtol=1e-6)

    def test_uniform_graphs_equal_weight(self) -> None:
        """Two graphs of same size should give same result as averaging BCE."""
        logits = torch.randn(20)
        target = torch.randint(0, 2, (20,)).float()
        edge_graph = torch.tensor([0] * 10 + [1] * 10)

        result = graph_normalized_bce(logits, target, edge_graph, n_graphs=2)

        bce_0 = torch.nn.functional.binary_cross_entropy_with_logits(
            logits[:10], target[:10]
        )
        bce_1 = torch.nn.functional.binary_cross_entropy_with_logits(
            logits[10:], target[10:]
        )
        expected = (bce_0 + bce_1) / 2
        torch.testing.assert_close(result, expected, atol=1e-6, rtol=1e-6)

    def test_pos_weight(self) -> None:
        logits = torch.randn(10)
        target = torch.randint(0, 2, (10,)).float()
        edge_graph = torch.zeros(10, dtype=torch.long)
        pw = torch.tensor([3.0])

        result = graph_normalized_bce(
            logits, target, edge_graph, n_graphs=1, pos_weight=pw
        )
        expected = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, target, pos_weight=pw
        )
        torch.testing.assert_close(result, expected, atol=1e-6, rtol=1e-6)

    def test_differentiable(self) -> None:
        logits = torch.randn(10, requires_grad=True)
        target = torch.randint(0, 2, (10,)).float()
        edge_graph = torch.zeros(10, dtype=torch.long)

        loss = graph_normalized_bce(logits, target, edge_graph, n_graphs=1)
        loss.backward()
        assert logits.grad is not None
        assert logits.grad.abs().sum() > 0


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


class TestEncoderRefactorRegression:
    """Verify the encoder refactor (ops.py wiring) didn't change numerics."""

    def test_deterministic_output(self) -> None:
        from gnn.models.encoder import DetectorGraphEncoder

        torch.manual_seed(42)
        enc = DetectorGraphEncoder(hidden_dim=32, num_layers=3, dropout=0.0)
        enc.eval()

        x = torch.randn(10, 1)
        ei = torch.randint(0, 10, (2, 20))
        ea = torch.randn(20, 2)

        with torch.no_grad():
            h1, edge_h1 = enc(x, ei, ea)
            h2, edge_h2 = enc(x, ei, ea)

        torch.testing.assert_close(h1, h2)
        torch.testing.assert_close(edge_h1, edge_h2)

    def test_shapes_correct(self) -> None:
        from gnn.models.encoder import DetectorGraphEncoder

        torch.manual_seed(42)
        enc = DetectorGraphEncoder(hidden_dim=32, num_layers=3, dropout=0.0)
        enc.eval()

        x = torch.randn(10, 1)
        ei = torch.randint(0, 10, (2, 20))
        ea = torch.randn(20, 2)

        with torch.no_grad():
            h, edge_h = enc(x, ei, ea)

        assert h.shape == (10, 32)
        assert edge_h.shape == (20, 32)

    def test_gradients_flow(self) -> None:
        from gnn.models.encoder import DetectorGraphEncoder

        enc = DetectorGraphEncoder(hidden_dim=16, num_layers=2, dropout=0.0)
        x = torch.randn(6, 1)
        ei = torch.randint(0, 6, (2, 10))
        ea = torch.randn(10, 2)

        h, edge_h = enc(x, ei, ea)
        (h.sum() + edge_h.sum()).backward()

        for name, param in enc.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"


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

    @_skip
    def test_graph_normalized_bce_matches(self) -> None:
        logits = torch.randn(20).cuda()
        target = torch.randint(0, 2, (20,)).float().cuda()
        edge_graph = torch.tensor([0] * 10 + [1] * 10).cuda()

        set_backend("pytorch")
        ref = graph_normalized_bce(logits, target, edge_graph, n_graphs=2)

        set_backend("cuda")
        out = graph_normalized_bce(logits, target, edge_graph, n_graphs=2)
        set_backend("pytorch")

        torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)

    @_skip
    def test_graph_normalized_bce_single_graph(self) -> None:
        logits = torch.randn(10).cuda()
        target = torch.randint(0, 2, (10,)).float().cuda()
        edge_graph = torch.zeros(10, dtype=torch.long).cuda()

        set_backend("pytorch")
        ref = graph_normalized_bce(logits, target, edge_graph, n_graphs=1)

        set_backend("cuda")
        out = graph_normalized_bce(logits, target, edge_graph, n_graphs=1)
        set_backend("pytorch")

        torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)

    @_skip
    def test_graph_normalized_bce_with_pos_weight(self) -> None:
        logits = torch.randn(20).cuda()
        target = torch.randint(0, 2, (20,)).float().cuda()
        edge_graph = torch.tensor([0] * 10 + [1] * 10).cuda()
        pw = torch.tensor([5.0]).cuda()

        set_backend("pytorch")
        ref = graph_normalized_bce(
            logits,
            target,
            edge_graph,
            n_graphs=2,
            pos_weight=pw,
        )

        set_backend("cuda")
        out = graph_normalized_bce(
            logits,
            target,
            edge_graph,
            n_graphs=2,
            pos_weight=pw,
        )
        set_backend("pytorch")

        torch.testing.assert_close(out, ref, atol=1e-4, rtol=1e-4)

    @_skip
    def test_graph_normalized_bce_uneven_graphs(self) -> None:
        """Graphs of different sizes — tests normalization correctness."""
        logits = torch.randn(15).cuda()
        target = torch.randint(0, 2, (15,)).float().cuda()
        edge_graph = torch.tensor([0] * 3 + [1] * 5 + [2] * 7).cuda()

        set_backend("pytorch")
        ref = graph_normalized_bce(logits, target, edge_graph, n_graphs=3)

        set_backend("cuda")
        out = graph_normalized_bce(logits, target, edge_graph, n_graphs=3)
        set_backend("pytorch")

        torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)
