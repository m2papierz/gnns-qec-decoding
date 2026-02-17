"""
Decoding heads and full decoder for QEC graph decoding.

This module provides task-specific heads that sit on top of the shared
:class:`~gnn.models.encoder.DetectorGraphEncoder` backbone, plus a
:class:`QECDecoder` wrapper that combines encoder and head into a single
``forward(batch)`` interface.

Two head architectures cover all three training modes:

- :class:`LogicalHead` — graph-level classification of observable flips.
  Used by the ``logical_head`` case.
- :class:`EdgeHead` — per-edge binary prediction.  Used by both
  ``mwpm_teacher`` (edge labels from MWPM solution) and ``hybrid``
  (predicted logits are converted to MWPM weights at eval time).

The ``mwpm_teacher`` and ``hybrid`` cases share the same model
architecture; they differ only in loss computation and evaluation
protocol, which is handled by the training loop, not here.
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
from torch_geometric.data import Batch
from torch_geometric.nn import global_mean_pool

from gnn.models.encoder import DetectorGraphEncoder


class LogicalHead(nn.Module):
    """
    Graph-level head for predicting logical observable flips.

    Pools node embeddings into a single graph-level vector via mean
    pooling, then maps through a two-layer MLP to produce one logit
    per logical observable.

    Parameters
    ----------
    hidden_dim : int
        Dimensionality of encoder output / input embeddings.
    num_observables : int, optional
        Number of logical observables to predict.  Default is 1
        (standard for ``rotated_memory_x``).
    dropout : float, optional
        Dropout probability in the MLP.  Default is 0.1.

    Examples
    --------
    >>> head = LogicalHead(hidden_dim=64, num_observables=1)
    >>> h = torch.randn(18, 64)          # 3 graphs batched, 6 nodes each
    >>> batch = torch.tensor([0]*6 + [1]*6 + [2]*6)
    >>> logits = head(h, batch)
    >>> logits.shape
    torch.Size([3, 1])
    """

    def __init__(
        self,
        hidden_dim: int,
        num_observables: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_observables),
        )

    def forward(
        self,
        h: torch.Tensor,
        batch: torch.Tensor,
        **kwargs: object,
    ) -> torch.Tensor:
        """Predict logical observable logits.

        Parameters
        ----------
        h : Tensor, shape (N_total, hidden_dim)
            Node embeddings from the encoder (batched).
        batch : Tensor, shape (N_total,)
            PyG batch vector mapping each node to its graph index.
        **kwargs
            Ignored.  Accepts extra keyword arguments so that the
            :class:`QECDecoder` can pass a uniform interface.

        Returns
        -------
        Tensor, shape (B, num_observables)
            Raw logits (pre-sigmoid) per graph.
        """
        graph_emb = global_mean_pool(h, batch)  # (B, hidden_dim)
        return self.mlp(graph_emb)


class EdgeHead(nn.Module):
    """
    Per-edge head for predicting edge activations.

    For each directed edge ``(u, v)``, concatenates the endpoint node
    embeddings with the *raw* edge attributes and maps through a
    two-layer MLP to produce a single logit.

    Used by both ``mwpm_teacher`` (supervised by MWPM edge selections)
    and ``hybrid`` (logits converted to MWPM weights at eval time).
    The two modes share this architecture; the difference is in how the
    training loop computes the loss and how evaluation interprets the
    outputs.

    Parameters
    ----------
    hidden_dim : int
        Dimensionality of encoder node embeddings.
    edge_dim : int, optional
        Dimensionality of raw edge attributes.  Default is 2
        (``[error_prob, weight]``).
    dropout : float, optional
        Dropout probability in the MLP.  Default is 0.1.

    Examples
    --------
    >>> head = EdgeHead(hidden_dim=64, edge_dim=2)
    >>> h = torch.randn(10, 64)
    >>> edge_index = torch.randint(0, 10, (2, 30))
    >>> edge_attr = torch.randn(30, 2)
    >>> logits = head(h, edge_index=edge_index, edge_attr=edge_attr)
    >>> logits.shape
    torch.Size([30])
    """

    def __init__(
        self,
        hidden_dim: int,
        edge_dim: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        mlp_in = 2 * hidden_dim + edge_dim

        self.mlp = nn.Sequential(
            nn.Linear(mlp_in, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        h: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        **kwargs: object,
    ) -> torch.Tensor:
        """
        Predict per-edge logits.

        Parameters
        ----------
        h : Tensor, shape (N_total, hidden_dim)
            Node embeddings from the encoder (batched).
        edge_index : Tensor, shape (2, E_total)
            Directed edge indices (batched, with PyG offsets).
        edge_attr : Tensor, shape (E_total, edge_dim)
            Raw edge attributes.
        **kwargs
            Ignored.

        Returns
        -------
        Tensor, shape (E_total,)
            Raw logit (pre-sigmoid) per directed edge.
        """
        h_u = h[edge_index[0]]  # (E, hidden_dim)
        h_v = h[edge_index[1]]  # (E, hidden_dim)
        inp = torch.cat([h_u, h_v, edge_attr], dim=-1)  # (E, 2H + edge_dim)
        return self.mlp(inp).squeeze(-1)  # (E,)


class QECDecoder(nn.Module):
    """
    Full decoder: encoder backbone + task-specific head.

    Provides a clean ``forward(batch)`` interface that runs the encoder
    and head in sequence on a PyG :class:`~torch_geometric.data.Batch`.

    Parameters
    ----------
    encoder : DetectorGraphEncoder
        Shared message-passing backbone.
    head : LogicalHead or EdgeHead
        Task-specific prediction head.

    Examples
    --------
    >>> model = build_model("logical_head", hidden_dim=64)
    >>> batch = Batch.from_data_list([data1, data2])
    >>> logits = model(batch)
    """

    def __init__(
        self,
        encoder: DetectorGraphEncoder,
        head: nn.Module,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.head = head

    def forward(self, batch: Batch) -> torch.Tensor:
        """Run full decode pipeline on a batched graph.

        Parameters
        ----------
        batch : Batch
            PyG batch containing ``x``, ``edge_index``, ``edge_attr``,
            and ``batch`` fields.

        Returns
        -------
        Tensor
            Prediction logits.  Shape depends on the head:

            - :class:`LogicalHead`: ``(B, num_observables)``
            - :class:`EdgeHead`: ``(E_total,)``
        """
        h = self.encoder(batch.x, batch.edge_index, batch.edge_attr)
        return self.head(
            h,
            batch=batch.batch,
            edge_index=batch.edge_index,
            edge_attr=batch.edge_attr,
        )


Case = Literal["logical_head", "mwpm_teacher", "hybrid"]


def build_model(
    case: Case,
    *,
    node_dim: int = 1,
    edge_dim: int = 2,
    hidden_dim: int = 128,
    num_layers: int = 6,
    num_observables: int = 1,
    dropout: float = 0.1,
) -> QECDecoder:
    """
    Construct a full :class:`QECDecoder` for a given training case.

    Parameters
    ----------
    case : {"logical_head", "mwpm_teacher", "hybrid"}
        Training case.  Determines which head is attached.
    node_dim : int, optional
        Input node feature dimensionality.  Default is 1.
    edge_dim : int, optional
        Input edge feature dimensionality.  Default is 2.
    hidden_dim : int, optional
        Hidden embedding dimensionality.  Default is 128.
    num_layers : int, optional
        Number of message-passing layers.  Default is 6.
    num_observables : int, optional
        Number of logical observables (only used for ``logical_head``).
        Default is 1.
    dropout : float, optional
        Dropout probability.  Default is 0.1.

    Returns
    -------
    QECDecoder
        Encoder + head ready for training.

    Raises
    ------
    ValueError
        If *case* is not one of the supported values.

    Examples
    --------
    >>> model = build_model("logical_head", hidden_dim=64, num_layers=4)
    >>> model = build_model("mwpm_teacher", hidden_dim=128)
    >>> model = build_model("hybrid", hidden_dim=128, num_layers=8)
    """
    encoder = DetectorGraphEncoder(
        node_dim=node_dim,
        edge_dim=edge_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
    )

    if case == "logical_head":
        head: nn.Module = LogicalHead(
            hidden_dim=hidden_dim,
            num_observables=num_observables,
            dropout=dropout,
        )
    elif case in ("mwpm_teacher", "hybrid"):
        head = EdgeHead(
            hidden_dim=hidden_dim,
            edge_dim=edge_dim,
            dropout=dropout,
        )
    else:
        raise ValueError(
            f"Unknown case {case!r}. "
            f"Expected one of: 'logical_head', 'mwpm_teacher', 'hybrid'"
        )

    return QECDecoder(encoder, head)
