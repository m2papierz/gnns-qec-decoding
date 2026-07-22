"""
Swappable compute operations for GNN encoder and heads.

Every function here isolates a compute-intensive pattern that appears in
the encoder or head forward passes. The active backend is selected via
:func:`set_backend` or the ``QECDEC_BACKEND`` environment variable.

Backends
--------
``"pytorch"``
    Pure PyTorch reference implementations (default, always available).
    Full autograd support — safe for training and inference.
``"compiled"``
    ``torch.compile``-wrapped PyTorch — identical numerics, kernel fusion
    handled by the compiler.  Full autograd support — recommended for
    training on GPU.
``"cuda"``
    Hand-written CUDA kernels loaded from ``kernels`` (requires build).
    **Inference only** — these are forward-pass kernels without autograd
    backward implementations.  Using this backend for training will
    silently break gradient propagation.
"""

from __future__ import annotations

import logging
import os
from enum import Enum

import torch
import torch.nn as nn


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
            import kernels.ops as _mod

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


_init_backend_from_env()
