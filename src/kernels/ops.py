"""CUDA kernel wrappers matching model.ops signatures."""

from __future__ import annotations

import torch

from kernels._C import fused_norm_residual_dropout as _cuda_norm_res_drop
from kernels._C import fused_symmetric_edge_features as _cuda_sym_edge_feat
from kernels._C import graph_normalized_bce as _cuda_graph_bce


def symmetric_edge_features(
    x: torch.Tensor,
    edge_index: torch.Tensor,
    edge_h: torch.Tensor,
) -> torch.Tensor:
    """CUDA fused symmetric edge features."""
    if x.is_cuda and x.is_contiguous() and edge_h.is_contiguous():
        return _cuda_sym_edge_feat(x, edge_index, edge_h)
    from model.ops import symmetric_edge_features as _pt

    return _pt(x, edge_index, edge_h)


def fused_norm_residual_dropout(
    x: torch.Tensor,
    residual: torch.Tensor,
    gamma: torch.Tensor,
    beta: torch.Tensor,
    dropout_p: float,
    training: bool,
) -> torch.Tensor:
    """CUDA fused LayerNorm + residual + dropout."""
    if x.is_cuda and x.is_contiguous():
        return _cuda_norm_res_drop(x, residual, gamma, beta, dropout_p, training)
    import torch.nn as nn

    norm = nn.LayerNorm(x.size(-1))
    norm.weight.data = gamma
    norm.bias.data = beta
    drop = nn.Dropout(dropout_p)
    from model.ops import fused_norm_residual_dropout as _pt

    return _pt(x, residual, norm, drop, training)


def graph_normalized_bce(
    logits: torch.Tensor,
    target: torch.Tensor,
    edge_graph: torch.Tensor,
    n_graphs: int,
    pos_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """CUDA per-graph normalized BCE."""
    if logits.is_cuda and logits.is_contiguous():
        return _cuda_graph_bce(logits, target, edge_graph, n_graphs, pos_weight)
    from model.ops import graph_normalized_bce as _pt

    return _pt(logits, target, edge_graph, n_graphs, pos_weight)
