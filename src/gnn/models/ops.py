"""
Swappable compute operations for GNN encoder and heads.

Every function here isolates a compute-intensive pattern that appears in
the encoder or head forward passes. The active backend is selected via
:func:`set_backend` or the ``QECDEC_BACKEND`` environment variable.

Backends
--------
``"pytorch"``
    Pure PyTorch reference implementations (default, always available).
``"compiled"``
    ``torch.compile``-wrapped PyTorch — identical numerics, kernel fusion
    handled by the compiler.
``"cuda"``
    Hand-written CUDA kernels loaded from ``cuda_ops`` (requires build).
"""

from __future__ import annotations

import logging
import os
from enum import Enum
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


logger = logging.getLogger(__name__)


class Backend(Enum):
    """Available compute backends."""

    PYTORCH = "pytorch"
    COMPILED = "compiled"
    CUDA = "cuda"


_active_backend: Backend = Backend.PYTORCH
_cuda_module = None  # lazily imported and cached


def get_backend() -> Backend:
    """Return the currently active compute backend."""
    return _active_backend


def set_backend(backend: str | Backend) -> None:
    """Set the compute backend globally.

    Parameters
    ----------
    backend : str or Backend
        One of ``"pytorch"``, ``"compiled"``, ``"cuda"``.
        If ``"cuda"`` is requested but the CUDA extension is not
        installed, falls back to ``"pytorch"`` with a warning.
    """
    global _active_backend, _cuda_module

    if isinstance(backend, str):
        backend = Backend(backend.lower())

    if backend == Backend.CUDA:
        try:
            import cuda_ops.ops as _mod

            _cuda_module = _mod
        except ImportError:
            logger.warning("CUDA ops not available — falling back to pytorch backend")
            backend = Backend.PYTORCH

    _active_backend = backend
    logger.info("Compute backend set to: %s", backend.value)


def _init_backend_from_env() -> None:
    """Initialise the backend from ``QECDEC_BACKEND`` env var (if set)."""
    env = os.environ.get("QECDEC_BACKEND", "pytorch").lower()
    try:
        set_backend(env)
    except ValueError:
        logger.warning(
            "Unknown QECDEC_BACKEND value '%s' — falling back to pytorch", env
        )


def symmetric_edge_features(
    x: torch.Tensor,
    edge_index: torch.Tensor,
    edge_h: torch.Tensor,
) -> torch.Tensor:
    """Compute symmetric edge feature vector.

    Gathers source and destination node embeddings, then concatenates
    three components: element-wise sum, element-wise absolute difference,
    and the current edge embedding.

    Parameters
    ----------
    x : Tensor, shape ``(N, H)``
        Node embeddings.
    edge_index : Tensor, shape ``(2, E)``
        Source/destination indices in COO format.
    edge_h : Tensor, shape ``(E, H)``
        Current edge embeddings.

    Returns
    -------
    Tensor, shape ``(E, 3*H)``
        ``[h_src + h_dst | |h_src - h_dst| | edge_h]``
    """
    if _active_backend == Backend.CUDA and _cuda_module is not None:
        try:
            return _cuda_module.symmetric_edge_features(x, edge_index, edge_h)
        except Exception:
            pass  # fall through to PyTorch

    h_src = x[edge_index[0]]
    h_dst = x[edge_index[1]]
    return torch.cat([h_src + h_dst, (h_src - h_dst).abs(), edge_h], dim=-1)


def fused_norm_residual_dropout(
    x: torch.Tensor,
    residual: torch.Tensor,
    norm: nn.Module,
    dropout: nn.Module,
    training: bool,
) -> torch.Tensor:
    """Normalise, optionally drop, and add residual.

    Applies ``norm(x)``, then dropout (only when *training*), then adds
    the *residual* connection.

    Parameters
    ----------
    x : Tensor, shape ``(N, H)``
        Input to normalise.
    residual : Tensor, shape ``(N, H)``
        Residual tensor to add after norm + dropout.
    norm : nn.Module
        Normalisation layer (``LayerNorm`` or compatible).
    dropout : nn.Module
        Dropout layer (used only when *training* is ``True``).
    training : bool
        Whether the model is in training mode.

    Returns
    -------
    Tensor, shape ``(N, H)``
    """
    if _active_backend == Backend.CUDA and _cuda_module is not None:
        try:
            return _cuda_module.fused_norm_residual_dropout(
                x,
                residual,
                norm.weight,
                norm.bias,
                float(dropout.p),
                training,
            )
        except Exception:
            pass

    x = norm(x)
    if training:
        x = dropout(x)
    return x + residual


def edge_mean_pool(
    edge_h: torch.Tensor,
    edge_batch: torch.Tensor,
    n_graphs: int,
) -> torch.Tensor:
    """Mean-pool edge embeddings per graph.

    Parameters
    ----------
    edge_h : Tensor, shape ``(E_total, H)``
        Edge embeddings across the full batch.
    edge_batch : Tensor, shape ``(E_total,)``
        Graph membership index for each edge.
    n_graphs : int
        Number of graphs in the batch.

    Returns
    -------
    Tensor, shape ``(n_graphs, H)``
    """
    from torch_geometric.nn import global_add_pool

    g_edge = global_add_pool(edge_h, edge_batch)
    edge_counts = torch.zeros(n_graphs, device=edge_h.device)
    edge_counts.scatter_add_(
        0, edge_batch, torch.ones(edge_batch.shape[0], device=edge_h.device)
    )
    return g_edge / edge_counts.clamp(min=1).unsqueeze(-1)


def graph_normalized_bce(
    logits: torch.Tensor,
    target: torch.Tensor,
    edge_graph: torch.Tensor,
    n_graphs: int,
    pos_weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Per-graph normalised BCE with logits for edge predictions.

    Computes element-wise BCE, averages within each graph, then averages
    across graphs.  This prevents larger graphs from dominating the
    gradient in mixed-distance batches.

    Parameters
    ----------
    logits : Tensor, shape ``(E_total,)``
        Raw edge logits.
    target : Tensor, shape ``(E_total,)``
        Binary edge targets.
    edge_graph : Tensor, shape ``(E_total,)``
        Graph index for each edge.
    n_graphs : int
        Number of graphs in the batch.
    pos_weight : Tensor or None
        Positive-class weight for BCE.

    Returns
    -------
    Tensor, scalar
    """
    if _active_backend == Backend.CUDA and _cuda_module is not None:
        try:
            return _cuda_module.graph_normalized_bce(
                logits, target, edge_graph, n_graphs, pos_weight
            )
        except Exception:
            pass

    raw = F.binary_cross_entropy_with_logits(
        logits, target, pos_weight=pos_weight, reduction="none"
    )
    graph_loss = torch.zeros(n_graphs, device=logits.device)
    graph_count = torch.zeros(n_graphs, device=logits.device)
    graph_loss.scatter_add_(0, edge_graph, raw)
    graph_count.scatter_add_(0, edge_graph, torch.ones_like(raw))
    return (graph_loss / graph_count.clamp(min=1)).mean()


_init_backend_from_env()
