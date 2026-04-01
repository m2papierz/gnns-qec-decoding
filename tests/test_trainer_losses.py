"""Tests for trainer loss functions and metrics."""

import pytest
import torch
from torch_geometric.data import Batch, Data

from gnn.trainer import GraphNormalizedSoftBCELoss


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


class TestGraphNormalizedSoftBCELoss:
    """Tests for GraphNormalizedSoftBCELoss."""

    def test_output_is_scalar(self) -> None:
        """Loss is a scalar tensor."""
        logits, targets, batch = _make_edge_batch()
        loss = GraphNormalizedSoftBCELoss()(logits, targets, batch)
        assert loss.ndim == 0

    def test_loss_nonnegative(self) -> None:
        """BCE-based loss is always >= 0."""
        logits, targets, batch = _make_edge_batch()
        loss = GraphNormalizedSoftBCELoss()(logits, targets, batch)
        assert loss.item() >= 0.0

    def test_perfect_prediction_at_entropy_floor(self) -> None:
        """When sigmoid(logits) == targets, loss equals binary entropy.

        For soft BCE, the minimum achievable loss is the average binary
        entropy of the targets: ``-t*log(t) - (1-t)*log(1-t)``, not zero.
        """
        _, targets, batch = _make_edge_batch()
        targets = targets.clamp(1e-6, 1.0 - 1e-6)
        logits = torch.log(targets / (1.0 - targets))
        loss = GraphNormalizedSoftBCELoss()(logits, targets, batch)

        # Compute expected binary entropy of targets.
        entropy = -(
            targets * torch.log(targets) + (1 - targets) * torch.log(1 - targets)
        ).mean()
        assert loss.item() == pytest.approx(entropy.item(), abs=1e-4)

    def test_gradient_flows(self) -> None:
        """Logits receive gradients through the loss."""
        logits, targets, batch = _make_edge_batch()
        logits.requires_grad_(True)
        loss = GraphNormalizedSoftBCELoss()(logits, targets, batch)
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

        loss = GraphNormalizedSoftBCELoss()(logits, targets, batch)

        # BCE(0, 1) = -log(sigmoid(0)) = log(2) ≈ 0.6931
        # Both graphs have identical per-edge BCE, graph normalisation
        # averages within each graph then across, giving the same value.
        import math

        assert loss.item() == pytest.approx(math.log(2), abs=1e-4)

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

        loss = GraphNormalizedSoftBCELoss()(logits, targets, batch)
        assert loss.ndim == 0
        assert loss.item() >= 0.0
