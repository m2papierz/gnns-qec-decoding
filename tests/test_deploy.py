"""Tests for deploy.engine — inference engine and helpers."""

import pytest
import torch
from torch_geometric.data import Batch, Data

from constants import CASES
from deploy.engine import (
    InferenceBackend,
    InferenceEngine,
    make_synthetic_batch,
)
from gnn.models.heads import build_model


def _make_batch(n_graphs=2, n_nodes=10, n_edges=20, device="cpu"):
    graphs = [
        Data(
            x=torch.randn(n_nodes, 1),
            edge_index=torch.randint(0, n_nodes, (2, n_edges)),
            edge_attr=torch.randn(n_edges, 2),
        )
        for _ in range(n_graphs)
    ]
    return Batch.from_data_list(graphs).to(device)


def _device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def _trt_available() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        from deploy.engine import _setup_tensorrt_libs

        _setup_tensorrt_libs()
        import torch_tensorrt  # noqa: F401

        return True
    except Exception:
        return False


_skip_no_gpu = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="no CUDA GPU",
)

_skip_no_trt = pytest.mark.skipif(
    not _trt_available(),
    reason="torch_tensorrt not available or no GPU",
)


class TestInferenceBackend:
    def test_values(self) -> None:
        assert InferenceBackend("pytorch") == InferenceBackend.PYTORCH
        assert InferenceBackend("compiled") == InferenceBackend.COMPILED
        assert InferenceBackend("tensorrt") == InferenceBackend.TENSORRT

    def test_invalid(self) -> None:
        with pytest.raises(ValueError):
            InferenceBackend("invalid_backend")


class TestPytorchBackend:
    """Baseline backend — runs on CPU or GPU."""

    @pytest.mark.parametrize("case", CASES)
    def test_forward_all_cases(self, case: str) -> None:
        """Every training case produces output of correct shape."""
        model = build_model(case, hidden_dim=16, num_layers=2, dropout=0.0)
        engine = InferenceEngine(model, backend="pytorch", device="cpu")

        batch = _make_batch(n_graphs=3, device="cpu")
        out = engine.predict(batch)

        if case == "direct":
            assert out.shape == (3, 1)
        else:
            # EdgeHead: one logit per directed edge
            assert out.shape == (batch.edge_index.shape[1],)

    def test_deterministic_eval_mode(self) -> None:
        """Repeated calls produce identical output (no dropout)."""
        model = build_model("direct", hidden_dim=16, num_layers=2, dropout=0.0)
        engine = InferenceEngine(model, backend="pytorch", device="cpu")
        batch = _make_batch(device="cpu")

        out1 = engine.predict(batch)
        out2 = engine.predict(batch)
        torch.testing.assert_close(out1, out2)

    def test_benchmark_returns_metrics(self) -> None:
        """Benchmark dict has all expected keys with sane values."""
        model = build_model("direct", hidden_dim=16, num_layers=2, dropout=0.0)
        engine = InferenceEngine(
            model,
            backend="pytorch",
            device="cpu",
            warmup_iters=2,
        )
        batch = _make_batch(device="cpu")
        metrics = engine.benchmark(batch, n_iters=5)

        assert metrics["backend"] == "pytorch"
        assert metrics["n_iters"] == 5
        assert metrics["n_graphs"] == 2
        assert metrics["mean_ms"] > 0
        assert metrics["std_ms"] >= 0
        assert metrics["min_ms"] > 0
        assert metrics["min_ms"] <= metrics["mean_ms"] <= metrics["max_ms"]
        assert metrics["throughput_graphs_per_sec"] > 0


@_skip_no_gpu
class TestCompiledBackend:
    """torch.compile backend — GPU only for reduce-overhead."""

    def test_output_matches_pytorch(self) -> None:
        """Compiled backend is numerically identical to PyTorch."""
        torch.manual_seed(42)
        model = build_model("direct", hidden_dim=32, num_layers=3, dropout=0.0)
        batch = _make_batch(n_graphs=2, device="cuda")

        pt_engine = InferenceEngine(model, backend="pytorch", device="cuda")
        pt_out = pt_engine.predict(batch)

        compiled_engine = InferenceEngine(model, backend="compiled", device="cuda")
        compiled_out = compiled_engine.predict(batch)

        torch.testing.assert_close(pt_out, compiled_out, atol=1e-5, rtol=1e-5)

    @pytest.mark.parametrize("case", CASES)
    def test_all_cases_compile(self, case: str) -> None:
        """All 4 model architectures survive torch.compile."""
        model = build_model(case, hidden_dim=16, num_layers=2, dropout=0.0)
        engine = InferenceEngine(model, backend="compiled", device="cuda")
        batch = _make_batch(device="cuda")
        out = engine.predict(batch)
        assert out is not None
        assert not torch.isnan(out).any()


@_skip_no_trt
class TestTensorRTBackend:
    """TensorRT backend — requires torch-tensorrt + GPU."""

    def test_output_close_to_pytorch(self) -> None:
        """TRT output within FP16 tolerance of PyTorch."""
        torch.manual_seed(42)
        model = build_model("direct", hidden_dim=32, num_layers=3, dropout=0.0)
        batch = _make_batch(n_graphs=2, device="cuda")

        pt_engine = InferenceEngine(model, backend="pytorch", device="cuda")
        pt_out = pt_engine.predict(batch)

        trt_engine = InferenceEngine(
            model,
            backend="tensorrt",
            device="cuda",
            precision="fp16",
        )
        trt_out = trt_engine.predict(batch)

        # FP16 tolerance — wider than compiled backend
        torch.testing.assert_close(pt_out, trt_out, atol=1e-2, rtol=1e-2)

    def test_fp32_precision(self) -> None:
        """TRT with fp32 is tighter than fp16."""
        torch.manual_seed(42)
        model = build_model("direct", hidden_dim=16, num_layers=2, dropout=0.0)
        batch = _make_batch(n_graphs=2, device="cuda")

        pt_engine = InferenceEngine(model, backend="pytorch", device="cuda")
        pt_out = pt_engine.predict(batch)

        trt_engine = InferenceEngine(
            model,
            backend="tensorrt",
            device="cuda",
            precision="fp32",
        )
        trt_out = trt_engine.predict(batch)

        torch.testing.assert_close(pt_out, trt_out, atol=5e-4, rtol=5e-4)

    @pytest.mark.parametrize("case", CASES)
    def test_all_cases_compile(self, case: str) -> None:
        """All 4 model architectures survive TRT compilation."""
        model = build_model(case, hidden_dim=16, num_layers=2, dropout=0.0)
        engine = InferenceEngine(
            model,
            backend="tensorrt",
            device="cuda",
            precision="fp32",
        )
        batch = _make_batch(device="cuda")
        out = engine.predict(batch)
        assert not torch.isnan(out).any()


class TestMakeSyntheticBatch:
    def test_shape(self) -> None:
        batch = make_synthetic_batch(
            n_graphs=3,
            n_nodes_per_graph=20,
            n_edges_per_graph=50,
            device="cpu",
        )
        assert batch.x.shape == (60, 1)
        assert batch.edge_attr.shape[1] == 2
        assert int(batch.batch.max()) + 1 == 3

    def test_on_gpu(self) -> None:
        if not torch.cuda.is_available():
            pytest.skip("no GPU")
        batch = make_synthetic_batch(device="cuda")
        assert batch.x.is_cuda
