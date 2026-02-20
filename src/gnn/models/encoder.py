"""
Message-passing encoder for detector-graph decoding.

Provides the shared GNN backbone used by all three decoding heads.
Transforms raw node features (syndrome bits) and edge attributes
(error probability, MWPM weight) into learned node *and edge*
embeddings via neighbourhood message passing with explicit edge
co-evolution.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.nn import GINEConv, LayerNorm


class _GINEBlock(nn.Module):
    """
    Single message-passing layer: node update via GINEConv followed by
    explicit edge update, with residual connections, norms, and dropout.

    Parameters
    ----------
    hidden_dim : int
        Dimensionality of node and edge embeddings.
    edge_dim : int
        Dimensionality of *raw* edge features (before projection).
    dropout : float
        Dropout probability applied after norms.
    """

    def __init__(
        self,
        hidden_dim: int,
        edge_dim: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.conv = GINEConv(nn=mlp, edge_dim=hidden_dim)
        self.edge_proj = nn.Linear(edge_dim, hidden_dim)
        self.norm = LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

        self.edge_update = nn.Sequential(
            nn.Linear(3 * hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.edge_norm = LayerNorm(hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        edge_h: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        One round of node + edge message passing.

        Parameters
        ----------
        x : Tensor, shape (N, hidden_dim)
            Node embeddings.
        edge_index : Tensor, shape (2, E)
            Edge indices in COO format.
        edge_attr : Tensor, shape (E, edge_dim)
            Raw edge features (projected internally per layer).
        edge_h : Tensor or None, shape (E, hidden_dim)
            Learned edge embeddings from previous layer.

        Returns
        -------
        x : Tensor, shape (N, hidden_dim)
        edge_h : Tensor, shape (E, hidden_dim)
        """
        e = self.edge_proj(edge_attr)
        if edge_h is not None:
            e = e + edge_h

        residual = x
        x = self.conv(x, edge_index, e)
        x = self.norm(x)
        x = self.dropout(x)
        x = x + residual

        h_src = x[edge_index[0]]
        h_dst = x[edge_index[1]]
        e_new = self.edge_update(
            torch.cat([h_src + h_dst, (h_src - h_dst).abs(), e], dim=-1)
        )
        e_new = self.edge_norm(e_new)
        e_new = self.dropout(e_new) + e

        return x, e_new


class DetectorGraphEncoder(nn.Module):
    """
    GNN encoder producing node and edge embeddings from detector graphs.

    Parameters
    ----------
    node_dim : int
        Input node feature dimensionality (default: 1).
    edge_dim : int
        Input edge feature dimensionality (default: 2).
    hidden_dim : int
        Hidden embedding dimensionality (default: 128).
    num_layers : int
        Number of message-passing layers (default: 6).
    dropout : float
        Dropout probability (default: 0.1).
    """

    def __init__(
        self,
        node_dim: int = 1,
        edge_dim: int = 2,
        hidden_dim: int = 128,
        num_layers: int = 6,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")
        if hidden_dim < 1:
            raise ValueError(f"hidden_dim must be >= 1, got {hidden_dim}")

        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.node_proj = nn.Linear(node_dim, hidden_dim)
        self.layers = nn.ModuleList(
            [
                _GINEBlock(hidden_dim, edge_dim=edge_dim, dropout=dropout)
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Encode detector graph into learned node and edge embeddings.

        Parameters
        ----------
        x : Tensor, shape (N, node_dim)
            Raw node features (syndrome bits).
        edge_index : Tensor, shape (2, E)
            Directed edge indices in COO format (bidirectional).
        edge_attr : Tensor, shape (E, edge_dim)
            Raw edge attributes ``[error_prob, weight]``.

        Returns
        -------
        h : Tensor, shape (N, hidden_dim)
            Node embeddings.
        edge_h : Tensor, shape (E, hidden_dim)
            Edge embeddings.
        """
        h = self.node_proj(x)
        edge_h = None

        for layer in self.layers:
            h, edge_h = layer(h, edge_index, edge_attr, edge_h)

        return h, edge_h
