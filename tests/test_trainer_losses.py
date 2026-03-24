"""Tests for trainer loss functions and metrics."""

import numpy as np
import pytest
import torch
from torch_geometric.data import Batch, Data

from gnn.trainer import _roc_auc_score, _SoftTeacherLoss


def _make_edge_batch(
    n_graphs: int = 3,
    n_nodes: int = 6,
    n_edges: int = 10,
) -> tuple[torch.Tensor, torch.Tensor, Batch]:
    """Create synthetic edge-level logits, targets, and a PyG batch.

    Returns
    -------
    logits : Tensor, shape ``(n_graphs * n_edges,)``
    targets : Tensor, shape ``(n_graphs * n_edges,)``
    batch : Batch
    """
    graphs = []
    for _ in range(n_graphs):
        graphs.append(
            Data(
                x=torch.randn(n_nodes, 1),
                edge_index=torch.randint(0, n_nodes, (2, n_edges)),
                edge_attr=torch.rand(n_edges, 2),
            )
        )
    batch = Batch.from_data_list(graphs)
    total_edges = n_graphs * n_edges
    logits = torch.randn(total_edges)
    targets = torch.rand(total_edges)  # continuous [0, 1] for soft labels
    return logits, targets, batch


class TestSoftTeacherLoss:
    """Tests for _SoftTeacherLoss."""

    def test_output_is_scalar(self) -> None:
        """Loss is a scalar tensor."""
        logits, targets, batch = _make_edge_batch()
        loss = _SoftTeacherLoss()(logits, targets, batch)
        assert loss.ndim == 0

    def test_loss_nonnegative(self) -> None:
        """MSE-based loss is always >= 0."""
        logits, targets, batch = _make_edge_batch()
        loss = _SoftTeacherLoss()(logits, targets, batch)
        assert loss.item() >= 0.0

    def test_perfect_prediction_zero_loss(self) -> None:
        """When sigmoid(logits) == targets, loss ~ 0."""
        _, targets, batch = _make_edge_batch()
        # Invert sigmoid to get logits that produce exact targets.
        targets = targets.clamp(1e-6, 1.0 - 1e-6)
        logits = torch.log(targets / (1.0 - targets))
        loss = _SoftTeacherLoss()(logits, targets, batch)
        assert loss.item() == pytest.approx(0.0, abs=1e-5)

    def test_gradient_flows(self) -> None:
        """Logits receive gradients through the loss."""
        logits, targets, batch = _make_edge_batch()
        logits.requires_grad_(True)
        loss = _SoftTeacherLoss()(logits, targets, batch)
        loss.backward()
        assert logits.grad is not None
        assert not torch.all(logits.grad == 0)

    def test_graph_normalization(self) -> None:
        """Each graph contributes equally regardless of edge count.

        Build a batch where graph 0 has many edges and graph 1 has few.
        Verify the loss is not dominated by graph 0.
        """
        g_big = Data(
            x=torch.randn(10, 1),
            edge_index=torch.randint(0, 10, (2, 100)),
            edge_attr=torch.rand(100, 2),
        )
        g_small = Data(
            x=torch.randn(3, 1),
            edge_index=torch.randint(0, 3, (2, 4)),
            edge_attr=torch.rand(4, 2),
        )
        batch = Batch.from_data_list([g_big, g_small])
        total_edges = 104

        # Set all predictions wrong by the same amount.
        logits = torch.zeros(total_edges)  # sigmoid(0) = 0.5
        targets = torch.ones(total_edges)  # target = 1.0

        loss = _SoftTeacherLoss()(logits, targets, batch)

        # Without graph normalization, the big graph would dominate.
        # With it, each graph's MSE is averaged independently, then
        # the two averages are averaged. Both graphs have the same
        # per-edge error (0.25), so the result should be ~ 0.25.
        assert loss.item() == pytest.approx(0.25, abs=1e-5)

    def test_mixed_size_batch(self) -> None:
        """Works with graphs of varying sizes."""
        graphs = [
            Data(
                x=torch.randn(n, 1),
                edge_index=torch.randint(0, n, (2, e)),
                edge_attr=torch.rand(e, 2),
            )
            for n, e in [(4, 6), (8, 14), (3, 4)]
        ]
        batch = Batch.from_data_list(graphs)
        total = sum(g.edge_attr.shape[0] for g in graphs)
        logits = torch.randn(total)
        targets = torch.rand(total)

        loss = _SoftTeacherLoss()(logits, targets, batch)
        assert loss.ndim == 0
        assert loss.item() >= 0.0


class TestRocAucScore:
    """Tests of ROC AUC implementation."""

    def test_perfect_separation(self) -> None:
        """All positives scored higher than all negatives => AUC=1."""
        targets = np.array([0, 0, 0, 1, 1, 1])
        scores = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
        assert _roc_auc_score(targets, scores) == pytest.approx(1.0)

    def test_perfect_inverse(self) -> None:
        """All positives scored lower than all negatives => AUC=0."""
        targets = np.array([1, 1, 1, 0, 0, 0])
        scores = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
        assert _roc_auc_score(targets, scores) == pytest.approx(0.0)

    def test_random_baseline(self) -> None:
        """Identical scores => AUC=0.5."""
        targets = np.array([0, 1, 0, 1])
        scores = np.array([0.5, 0.5, 0.5, 0.5])
        assert _roc_auc_score(targets, scores) == pytest.approx(0.5)

    def test_all_positive_returns_half(self) -> None:
        targets = np.array([1, 1, 1])
        scores = np.array([0.1, 0.5, 0.9])
        assert _roc_auc_score(targets, scores) == 0.5

    def test_all_negative_returns_half(self) -> None:
        targets = np.array([0, 0, 0])
        scores = np.array([0.1, 0.5, 0.9])
        assert _roc_auc_score(targets, scores) == 0.5

    def test_known_value(self) -> None:
        """Hand-computed: 2 pos at ranks 3,4 out of 4 items, 2 neg.
        U = (3+4) - 2*3/2 = 4. AUC = 4/(2*2) = 1.0.
        """
        targets = np.array([0, 0, 1, 1])
        scores = np.array([0.1, 0.2, 0.8, 0.9])
        assert _roc_auc_score(targets, scores) == pytest.approx(1.0)

    def test_tied_scores_handled(self) -> None:
        """Ties must be averaged, not arbitrarily broken."""
        targets = np.array([0, 1, 0, 1])
        scores = np.array([0.5, 0.5, 0.5, 0.5])
        # All tied => AUC should be exactly 0.5
        assert _roc_auc_score(targets, scores) == pytest.approx(0.5)

    def test_matches_sklearn_on_nontrivial(self) -> None:
        """Cross-check against sklearn on a non-trivial case."""
        try:
            from sklearn.metrics import roc_auc_score as sklearn_auc
        except ImportError:
            pytest.skip("sklearn not installed")

        rng = np.random.RandomState(42)
        targets = rng.randint(0, 2, size=200).astype(float)
        scores = rng.randn(200)

        ours = _roc_auc_score(targets, scores)
        theirs = sklearn_auc(targets, scores)
        assert ours == pytest.approx(theirs, abs=1e-10)

    def test_two_samples(self) -> None:
        """Minimal case: one pos, one neg."""
        assert _roc_auc_score(np.array([0, 1]), np.array([0.3, 0.7])) == pytest.approx(
            1.0
        )
        assert _roc_auc_score(np.array([1, 0]), np.array([0.3, 0.7])) == pytest.approx(
            0.0
        )
