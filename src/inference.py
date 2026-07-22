"""
Unified inference engine for GNN-based QEC decoders.

Supports three backends:

``"pytorch"``
    Vanilla eager-mode PyTorch.  Baseline for correctness.
``"compiled"``
    ``torch.compile`` with ``reduce-overhead`` mode.  Fuses ops
    via Triton; first call is slow (compilation), subsequent calls
    are faster.
``"tensorrt"``
    ``torch.compile`` with the ``torch_tensorrt`` backend.  Dense
    subgraphs (Linear, LayerNorm, MLP) are lowered to TensorRT
    engines; sparse GNN ops (scatter, gather) stay in PyTorch.
    Requires ``torch-tensorrt`` to be installed.

All backends accept the same :class:`~torch_geometric.data.Batch`
input and produce numerically close outputs (within FP tolerance).
"""

from __future__ import annotations

import logging
import os
import time
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch_geometric.data import Batch, Data

from model.decoder import QECDecoder, build_model


logger = logging.getLogger(__name__)


def _setup_tensorrt_libs() -> None:
    """Ensure TensorRT native libraries are discoverable.

    ``tensorrt-cu12-libs`` installs ``libnvinfer.so.10`` inside the
    Python environment (``site-packages/tensorrt_libs/``), but
    ``torch_tensorrt`` searches system paths and
    ``LD_LIBRARY_PATH``.  This function locates the libs directory
    and adds it to ``LD_LIBRARY_PATH`` before the first
    ``import torch_tensorrt``.
    """
    import sysconfig

    site_packages = Path(sysconfig.get_path("purelib"))
    trt_libs = site_packages / "tensorrt_libs"

    if not trt_libs.is_dir():
        return

    ld_path = os.environ.get("LD_LIBRARY_PATH", "")
    trt_str = str(trt_libs)
    if trt_str not in ld_path:
        os.environ["LD_LIBRARY_PATH"] = f"{trt_str}:{ld_path}" if ld_path else trt_str
        logger.debug("Added %s to LD_LIBRARY_PATH", trt_str)


def _deregister_scatter_reduce_decomp() -> None:
    """Remove torch_tensorrt's broken scatter_reduce decomposition.

    ``torch_tensorrt`` registers a custom decomposition for
    ``aten.scatter_reduce.two`` that raises ``AssertionError`` when
    ``include_self=False`` — a pattern used by PyG's ``softmax``.
    Removing the decomposition lets ``scatter_reduce`` stay as an
    opaque ATen op during AOT tracing so the TRT partitioner routes
    it to a PyTorch subgraph instead of crashing.
    """
    try:
        from torch_tensorrt.dynamo.lowering._decompositions import (
            TORCH_TRT_DECOMPOSITIONS,
        )

        key = torch.ops.aten.scatter_reduce.two
        if key in TORCH_TRT_DECOMPOSITIONS:
            del TORCH_TRT_DECOMPOSITIONS[key]
            logger.debug("Removed scatter_reduce decomposition from TRT table")
    except (ImportError, AttributeError) as exc:
        logger.debug("Could not deregister scatter_reduce decomp: %s", exc)


class InferenceBackend(Enum):
    """Available inference backends."""

    PYTORCH = "pytorch"
    COMPILED = "compiled"
    TENSORRT = "tensorrt"


class InferenceEngine:
    """Unified inference wrapper with benchmarking.

    Wraps a :class:`QECDecoder` and applies the selected backend
    optimisation.  The ``benchmark`` method uses CUDA events for
    accurate latency measurement.

    Parameters
    ----------
    model : QECDecoder
        Trained decoder model.
    backend : str or InferenceBackend
        One of ``"pytorch"``, ``"compiled"``, ``"tensorrt"``.
    device : str or torch.device
        Target device (default ``"cuda"``).
    precision : str
        ``"fp32"`` or ``"fp16"`` (only affects TensorRT).
    min_block_size : int
        Minimum consecutive TRT-convertible ops to form a TRT
        segment.  Smaller values create more TRT segments (more
        overhead, more ops accelerated).
    warmup_iters : int
        Number of warmup forward passes before benchmarking.
    """

    def __init__(
        self,
        model: QECDecoder,
        *,
        backend: str | InferenceBackend = "pytorch",
        device: str | torch.device = "cuda",
        precision: str = "fp16",
        min_block_size: int = 3,
        warmup_iters: int = 10,
    ) -> None:
        if isinstance(backend, str):
            backend = InferenceBackend(backend.lower())

        self.backend = backend
        self.device = torch.device(device)
        self.precision = precision
        self.warmup_iters = warmup_iters
        self._warmed_up = False

        self.model = self._prepare_model(
            model,
            min_block_size=min_block_size,
        )

    def _prepare_model(
        self,
        model: QECDecoder,
        *,
        min_block_size: int,
    ) -> torch.nn.Module:
        """Move model to device and apply backend-specific compilation."""
        model = model.to(self.device).eval()

        if self.backend == InferenceBackend.COMPILED:
            model = torch.compile(
                model,
                mode="reduce-overhead",
                fullgraph=False,  # GNN scatter ops cause graph breaks
            )
            logger.info("torch.compile enabled (reduce-overhead)")

        elif self.backend == InferenceBackend.TENSORRT:
            _setup_tensorrt_libs()
            # GNNs have dynamic shapes (variable nodes/edges per batch).
            # torch_tensorrt's dynamic shape extraction crashes on
            # unbounded dims (sympy int_oo). Force static shapes:
            # each unique input shape triggers recompilation, which is
            # fine for benchmarking at fixed batch geometry.
            import torch._dynamo as dynamo
            import torch_tensorrt  # noqa: F401 — registers backend

            dynamo.config.assume_static_by_default = True

            # torch_tensorrt registers a custom decomposition for
            # scatter_reduce that doesn't support include_self=False,
            # which PyG's softmax uses internally. Removing it lets
            # the op stay opaque during AOT tracing so the partitioner
            # correctly routes it to the PyTorch subgraph while dense
            # ops (Linear, LayerNorm, MLP) go to TRT engines.
            _deregister_scatter_reduce_decomp()

            precisions = {torch.float32}
            if self.precision == "fp16":
                precisions.add(torch.float16)

            model = torch.compile(
                model,
                backend="torch_tensorrt",
                dynamic=False,
                options={
                    "enabled_precisions": precisions,
                    "min_block_size": min_block_size,
                    "use_fp32_acc": True,
                    "pass_through_build_failures": False,
                },
            )
            logger.info(
                "torch_tensorrt backend enabled (precision=%s, " "min_block_size=%d)",
                self.precision,
                min_block_size,
            )

        return model

    @torch.no_grad()
    def predict(self, batch: Batch) -> torch.Tensor:
        """Run inference on a batched graph.

        Parameters
        ----------
        batch : Batch
            PyG batched graph (moved to device internally).

        Returns
        -------
        Tensor
            Model output — shape depends on the head type.
        """
        return self.model(batch.to(self.device))

    def warmup(self, sample_batch: Batch, n: int | None = None) -> None:
        """Run warmup forward passes to trigger compilation.

        Parameters
        ----------
        sample_batch : Batch
            Representative input batch.
        n : int or None
            Override for number of warmup iterations.
        """
        n = n or self.warmup_iters
        sample = sample_batch.to(self.device)

        with torch.no_grad():
            for _ in range(n):
                self.model(sample)

        if self.device.type == "cuda":
            torch.cuda.synchronize()

        self._warmed_up = True
        logger.debug("Warmup complete (%d iterations)", n)

    @torch.no_grad()
    def benchmark(
        self,
        batch: Batch,
        n_iters: int = 100,
    ) -> dict[str, Any]:
        """Timed inference benchmark using CUDA events.

        Automatically warms up on first call.

        Parameters
        ----------
        batch : Batch
            Input batch to benchmark on.
        n_iters : int
            Number of timed iterations.

        Returns
        -------
        dict
            Timing statistics: ``mean_ms``, ``std_ms``,
            ``median_ms``, ``min_ms``, ``max_ms``,
            ``throughput_graphs_per_sec``, and metadata.
        """
        if not self._warmed_up:
            self.warmup(batch)

        batch = batch.to(self.device)
        n_graphs = int(batch.batch.max()) + 1

        if self.device.type == "cuda":
            timings = self._benchmark_cuda(batch, n_iters)
        else:
            timings = self._benchmark_cpu(batch, n_iters)

        t = np.array(timings)
        mean_ms = float(t.mean())
        return {
            "backend": self.backend.value,
            "precision": self.precision,
            "n_graphs": n_graphs,
            "n_iters": n_iters,
            "mean_ms": mean_ms,
            "std_ms": float(t.std()),
            "median_ms": float(np.median(t)),
            "min_ms": float(t.min()),
            "max_ms": float(t.max()),
            "throughput_graphs_per_sec": (
                n_graphs / (mean_ms / 1000.0) if mean_ms > 0 else 0.0
            ),
        }

    def _benchmark_cuda(
        self,
        batch: Batch,
        n_iters: int,
    ) -> list[float]:
        """Benchmark with CUDA events (sub-ms precision)."""
        timings: list[float] = []
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        for _ in range(n_iters):
            start.record()
            self.model(batch)
            end.record()
            torch.cuda.synchronize()
            timings.append(start.elapsed_time(end))

        return timings

    def _benchmark_cpu(
        self,
        batch: Batch,
        n_iters: int,
    ) -> list[float]:
        """Benchmark with perf_counter (CPU fallback)."""
        timings: list[float] = []

        for _ in range(n_iters):
            t0 = time.perf_counter()
            self.model(batch)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            timings.append(elapsed_ms)

        return timings


def load_model_from_checkpoint(
    checkpoint: Path,
    *,
    device: str | torch.device = "cuda",
    dropout: float = 0.0,
) -> tuple[QECDecoder, dict[str, Any]]:
    """Load a trained QECDecoder from a training checkpoint.

    Parameters
    ----------
    checkpoint : Path
        Path to ``best.pt`` checkpoint file.
    device : str or torch.device
        Device to load weights onto.
    dropout : float
        Dropout probability (0.0 for inference).

    Returns
    -------
    model : QECDecoder
        Model with loaded weights, in eval mode.
    config : dict
        Training configuration from the checkpoint.
    """
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    cfg = ckpt["config"]

    model = build_model(
        node_dim=cfg["node_dim"],
        edge_dim=cfg["edge_dim"],
        hidden_dim=cfg["hidden_dim"],
        num_layers=cfg["num_layers"],
        dropout=dropout,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device).eval()

    logger.info(
        "Loaded model: hidden_dim=%d, num_layers=%d, "
        "node_dim=%d, edge_dim=%d (epoch %d)",
        cfg["hidden_dim"],
        cfg["num_layers"],
        cfg["node_dim"],
        cfg["edge_dim"],
        ckpt.get("epoch", -1),
    )
    return model, cfg


def make_synthetic_batch(
    n_graphs: int = 4,
    n_nodes_per_graph: int = 50,
    n_edges_per_graph: int = 120,
    device: str | torch.device = "cuda",
) -> Batch:
    """Create a synthetic PyG batch for warmup / benchmarking.

    Generates random tensors matching the enriched feature dimensions
    (``node_dim=5``, ``edge_dim=3``).

    Parameters
    ----------
    n_graphs : int
        Number of graphs in the batch.
    n_nodes_per_graph : int
        Nodes per graph (roughly d=5 surface code).
    n_edges_per_graph : int
        Directed edges per graph.
    device : str or torch.device
        Target device.

    Returns
    -------
    Batch
    """
    from model.dataset import EDGE_DIM, NODE_DIM

    graphs = []
    for _ in range(n_graphs):
        graphs.append(
            Data(
                x=torch.randn(n_nodes_per_graph, NODE_DIM),
                edge_index=torch.randint(
                    0,
                    n_nodes_per_graph,
                    (2, n_edges_per_graph),
                ),
                edge_attr=torch.randn(n_edges_per_graph, EDGE_DIM),
            )
        )
    return Batch.from_data_list(graphs).to(device)
