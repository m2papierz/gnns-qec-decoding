"""
Message-passing encoder for detector-graph decoding.

This module provides the shared GNN backbone used by all three decoding
heads (``logical_head``, ``mwpm_teacher``, ``hybrid``). The encoder
transforms raw node features (syndrome bits) and edge attributes (error
probability, MWPM weight) into learned node embeddings via several rounds
of neighbourhood message passing.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.nn import GINEConv, LayerNorm


class _GINEBlock(nn.Module):
    """
    Single GINEConv layer with residual connection, norm, and dropout.

    Parameters
    ----------
    hidden_dim : int
        Dimensionality of node and edge embeddings.
    dropout : float
        Dropout probability applied after activation.
    """

    def __init__(self, hidden_dim: int, dropout: float = 0.1) -> None:
        super().__init__()

        mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.conv = GINEConv(nn=mlp, edge_dim=hidden_dim)
        self.norm = LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        """
        Run one message-passing step with residual.

        Parameters
        ----------
        x : Tensor, shape (N, hidden_dim)
            Node embeddings.
        edge_index : Tensor, shape (2, E)
            Edge indices in COO format.
        edge_attr : Tensor, shape (E, hidden_dim)
            Projected edge features.

        Returns
        -------
        Tensor, shape (N, hidden_dim)
            Updated node embeddings.
        """
        residual = x
        x = self.conv(x, edge_index, edge_attr)
        x = self.norm(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = x + residual
        return x


class DetectorGraphEncoder(nn.Module):
    """
    GNN encoder that produces node embeddings from detector graphs.

    Takes raw node features (syndrome bits, boundary flags) and edge
    attributes (error probability, MWPM weight) and outputs a learned
    embedding per node. Downstream heads consume these embeddings for
    graph-level classification, edge-level prediction, or weight
    rewriting.

    Parameters
    ----------
    node_dim : int, optional
        Input node feature dimensionality.  Default is 1 (syndrome bit
        only).
    edge_dim : int, optional
        Input edge feature dimensionality.  Default is 2
        (``[error_prob, weight]``).
    hidden_dim : int, optional
        Hidden embedding dimensionality for all layers.  Default is 128.
    num_layers : int, optional
        Number of message-passing layers.  Default is 6.
    dropout : float, optional
        Dropout probability.  Default is 0.1.

    Attributes
    ----------
    node_proj : nn.Linear
        Projects raw node features to ``hidden_dim``.
    edge_proj : nn.Linear
        Projects raw edge attributes to ``hidden_dim``.
    layers : nn.ModuleList
        Sequence of :class:`_GINEBlock` message-passing layers.

    Examples
    --------
    >>> encoder = DetectorGraphEncoder(node_dim=1, edge_dim=2, hidden_dim=64)
    >>> # Single graph: 10 nodes, 30 edges
    >>> x = torch.randn(10, 1)
    >>> edge_index = torch.randint(0, 10, (2, 30))
    >>> edge_attr = torch.randn(30, 2)
    >>> h = encoder(x, edge_index, edge_attr)
    >>> h.shape
    torch.Size([10, 64])
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

        # Input projections
        self.node_proj = nn.Linear(node_dim, hidden_dim)
        self.edge_proj = nn.Linear(edge_dim, hidden_dim)

        # Message-passing layers
        self.layers = nn.ModuleList(
            [_GINEBlock(hidden_dim, dropout=dropout) for _ in range(num_layers)]
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode detector graph nodes into learned embeddings.

        Parameters
        ----------
        x : Tensor, shape (N, node_dim)
            Raw node features.  For detector graphs this is typically
            a single syndrome bit per node (0 or 1), with boundary nodes
            set to 0.
        edge_index : Tensor, shape (2, E)
            Directed edge indices in COO format.  The detector graph
            stores both directions (u→v and v→u).
        edge_attr : Tensor, shape (E, edge_dim)
            Raw edge attributes, typically ``[error_prob, weight]``.

        Returns
        -------
        Tensor, shape (N, hidden_dim)
            Node embeddings after ``num_layers`` rounds of message
            passing.  These can be consumed by downstream heads for
            graph-level pooling, edge-level prediction, etc.
        """
        h = self.node_proj(x)
        e = self.edge_proj(edge_attr)

        for layer in self.layers:
            h = layer(h, edge_index, e)

        return h
