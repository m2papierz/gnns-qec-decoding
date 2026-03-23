"""Tests for trainer loss functions and metrics."""

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from torch_geometric.data import Batch, Data

from gnn.trainer import _roc_auc_score, _SoftTeacherLoss


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


def _make_edge_batch(
    n_graphs: int = 2,
    edges_per_graph: tuple[int, ...] = (6, 10),
    nodes_per_graph: tuple[int, ...] = (4, 6),
) -> Batch:
    """Build a Batch with known edge-to-graph assignment."""
    graphs = []
    for i in range(n_graphs):
        ne = edges_per_graph[i]
        nn_ = nodes_per_graph[i]
        graphs.append(
            Data(
                x=torch.randn(nn_, 1),
                edge_index=torch.randint(0, nn_, (2, ne)),
                edge_attr=torch.randn(ne, 2),
            )
        )
    return Batch.from_data_list(graphs)


class TestSoftTeacherLoss:
    """Graph-normalized MSE on sigmoid(logits) vs continuous targets."""

    def test_zero_loss_at_target(self) -> None:
        """If sigmoid(logits) == target, loss is 0."""
        loss_fn = _SoftTeacherLoss()
        batch = _make_edge_batch(
            n_graphs=1,
            edges_per_graph=(8,),
            nodes_per_graph=(4,),
        )

        target = torch.full((8,), 0.5)
        # sigmoid(0) = 0.5
        logits = torch.zeros(8)

        loss = loss_fn(logits, target, batch)
        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_positive_loss(self) -> None:
        """Mismatched logits and targets produce positive loss."""
        loss_fn = _SoftTeacherLoss()
        batch = _make_edge_batch(
            n_graphs=1,
            edges_per_graph=(8,),
            nodes_per_graph=(4,),
        )

        logits = torch.randn(8)
        target = torch.rand(8)

        loss = loss_fn(logits, target, batch)
        assert loss.item() > 0

    def test_graph_normalization(self) -> None:
        """Small and large graphs contribute equally to the loss."""
        loss_fn = _SoftTeacherLoss()

        torch.manual_seed(99)
        batch = _make_edge_batch(
            n_graphs=2,
            edges_per_graph=(4, 100),
            nodes_per_graph=(3, 20),
        )
        total_edges = batch.edge_attr.shape[0]

        logits = torch.randn(total_edges)
        target = torch.rand(total_edges)

        loss = loss_fn(logits, target, batch)

        # Compute per-graph losses manually
        edge_graph = batch.batch[batch.edge_index[0]]
        pred = torch.sigmoid(logits)
        raw = F.mse_loss(pred, target, reduction="none")

        g0_mask = edge_graph == 0
        g1_mask = edge_graph == 1
        g0_loss = raw[g0_mask].mean()
        g1_loss = raw[g1_mask].mean()
        expected = (g0_loss + g1_loss) / 2

        assert loss.item() == pytest.approx(expected.item(), abs=1e-6)

    def test_differentiable(self) -> None:
        loss_fn = _SoftTeacherLoss()
        batch = _make_edge_batch(
            n_graphs=1,
            edges_per_graph=(10,),
            nodes_per_graph=(4,),
        )

        logits = torch.randn(10, requires_grad=True)
        target = torch.rand(10)

        loss = loss_fn(logits, target, batch)
        loss.backward()

        assert logits.grad is not None
        assert logits.grad.abs().sum() > 0

    def test_output_is_scalar(self) -> None:
        loss_fn = _SoftTeacherLoss()
        batch = _make_edge_batch(
            n_graphs=2,
            edges_per_graph=(6, 8),
            nodes_per_graph=(4, 5),
        )
        total = batch.edge_attr.shape[0]

        loss = loss_fn(torch.randn(total), torch.rand(total), batch)
        assert loss.ndim == 0
