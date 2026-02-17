"""Tests for decoding heads and QECDecoder."""

import pytest
import torch
from torch_geometric.data import Batch, Data

from gnn.models.heads import (
    EdgeHead,
    LogicalHead,
    QECDecoder,
    build_model,
)


def _make_graph(num_nodes: int = 6, num_edges: int = 10) -> Data:
    """Create a minimal detector graph for testing."""
    return Data(
        x=torch.randint(0, 2, (num_nodes, 1)).float(),
        edge_index=torch.randint(0, num_nodes, (2, num_edges)),
        edge_attr=torch.rand(num_edges, 2),
        y=torch.zeros(1),  # placeholder
        logical=torch.zeros(1),
        setting_id=torch.tensor(0),
    )


@pytest.fixture
def single_graph() -> Data:
    return _make_graph(num_nodes=6, num_edges=10)


@pytest.fixture
def batched() -> Batch:
    """Batch of 3 graphs with different sizes."""
    g1 = _make_graph(num_nodes=6, num_edges=10)
    g2 = _make_graph(num_nodes=10, num_edges=18)
    g3 = _make_graph(num_nodes=8, num_edges=14)
    return Batch.from_data_list([g1, g2, g3])


class TestLogicalHead:
    """Tests for graph-level observable prediction."""

    def test_output_shape_single_observable(self) -> None:
        head = LogicalHead(hidden_dim=32, num_observables=1, dropout=0.0)
        h = torch.randn(18, 32)
        batch = torch.tensor([0] * 6 + [1] * 6 + [2] * 6)
        logits = head(h, batch)
        assert logits.shape == (3, 1)

    def test_output_shape_multi_observable(self) -> None:
        head = LogicalHead(hidden_dim=32, num_observables=3, dropout=0.0)
        h = torch.randn(10, 32)
        batch = torch.tensor([0] * 4 + [1] * 6)
        logits = head(h, batch)
        assert logits.shape == (2, 3)

    def test_ignores_extra_kwargs(self) -> None:
        head = LogicalHead(hidden_dim=16, dropout=0.0)
        h = torch.randn(5, 16)
        batch = torch.zeros(5, dtype=torch.long)
        # Should not raise on unexpected kwargs
        logits = head(h, batch, edge_index=None, edge_attr=None)
        assert logits.shape == (1, 1)

    def test_gradient_flow(self) -> None:
        head = LogicalHead(hidden_dim=16, dropout=0.0)
        h = torch.randn(8, 16, requires_grad=True)
        batch = torch.tensor([0] * 4 + [1] * 4)
        logits = head(h, batch)
        logits.sum().backward()
        assert h.grad is not None


class TestEdgeHead:
    """Tests for per-edge prediction."""

    def test_output_shape(self) -> None:
        head = EdgeHead(hidden_dim=32, edge_dim=2, dropout=0.0)
        h = torch.randn(10, 32)
        ei = torch.randint(0, 10, (2, 20))
        ea = torch.randn(20, 2)
        logits = head(h, edge_index=ei, edge_attr=ea)
        assert logits.shape == (20,)

    def test_custom_edge_dim(self) -> None:
        head = EdgeHead(hidden_dim=16, edge_dim=5, dropout=0.0)
        h = torch.randn(6, 16)
        ei = torch.randint(0, 6, (2, 12))
        ea = torch.randn(12, 5)
        logits = head(h, edge_index=ei, edge_attr=ea)
        assert logits.shape == (12,)

    def test_ignores_extra_kwargs(self) -> None:
        head = EdgeHead(hidden_dim=16, dropout=0.0)
        h = torch.randn(4, 16)
        ei = torch.randint(0, 4, (2, 6))
        ea = torch.randn(6, 2)
        logits = head(h, edge_index=ei, edge_attr=ea, batch=None)
        assert logits.shape == (6,)

    def test_gradient_flow(self) -> None:
        head = EdgeHead(hidden_dim=16, dropout=0.0)
        h = torch.randn(6, 16, requires_grad=True)
        ei = torch.randint(0, 6, (2, 10))
        ea = torch.randn(10, 2)
        logits = head(h, edge_index=ei, edge_attr=ea)
        logits.sum().backward()
        assert h.grad is not None


class TestQECDecoder:
    """Tests for the full encoder + head pipeline."""

    @pytest.mark.parametrize("case", ["logical_head", "mwpm_teacher", "hybrid"])
    def test_forward_runs(self, case: str, batched: Batch) -> None:
        model = build_model(case, hidden_dim=32, num_layers=2, dropout=0.0)
        model.eval()
        with torch.no_grad():
            logits = model(batched)
        assert logits is not None
        assert logits.ndim >= 1

    def test_logical_head_output_shape(self, batched: Batch) -> None:
        model = build_model(
            "logical_head", hidden_dim=32, num_layers=2, dropout=0.0
        )
        model.eval()
        with torch.no_grad():
            logits = model(batched)
        # 3 graphs, 1 observable each
        assert logits.shape == (3, 1)

    def test_edge_head_output_shape(self, batched: Batch) -> None:
        model = build_model(
            "mwpm_teacher", hidden_dim=32, num_layers=2, dropout=0.0
        )
        model.eval()
        with torch.no_grad():
            logits = model(batched)
        # Total edges across all 3 graphs
        total_edges = batched.edge_attr.shape[0]
        assert logits.shape == (total_edges,)

    def test_hybrid_same_as_teacher(self, batched: Batch) -> None:
        """hybrid and mwpm_teacher produce same architecture."""
        m1 = build_model("mwpm_teacher", hidden_dim=16, num_layers=2)
        m2 = build_model("hybrid", hidden_dim=16, num_layers=2)
        # Same parameter structure
        p1 = {n for n, _ in m1.named_parameters()}
        p2 = {n for n, _ in m2.named_parameters()}
        assert p1 == p2

    def test_backward_pass(self, batched: Batch) -> None:
        model = build_model(
            "logical_head", hidden_dim=32, num_layers=2, dropout=0.0
        )
        logits = model(batched)
        logits.sum().backward()
        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"


class TestBuildModel:
    """Tests for the model factory."""

    def test_invalid_case(self) -> None:
        with pytest.raises(ValueError, match="Unknown case"):
            build_model("nonexistent")  # type: ignore[arg-type]

    def test_returns_qec_decoder(self) -> None:
        model = build_model("logical_head")
        assert isinstance(model, QECDecoder)

    def test_encoder_config_propagates(self) -> None:
        model = build_model(
            "logical_head", hidden_dim=64, num_layers=4, node_dim=3
        )
        assert model.encoder.hidden_dim == 64
        assert model.encoder.num_layers == 4
        assert model.encoder.node_dim == 3

    def test_logical_head_observables(self) -> None:
        model = build_model("logical_head", hidden_dim=32, num_observables=5)
        assert isinstance(model.head, LogicalHead)

    @pytest.mark.parametrize("case", ["mwpm_teacher", "hybrid"])
    def test_edge_cases_use_edge_head(self, case: str) -> None:
        model = build_model(case, hidden_dim=32)
        assert isinstance(model.head, EdgeHead)
