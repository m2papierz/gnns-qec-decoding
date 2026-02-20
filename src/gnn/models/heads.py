"""
Decoding heads and full decoder for QEC graph decoding.

Two head architectures cover all three training modes:

- :class:`LogicalHead` — graph-level classification of observable flips
  using attention-weighted pooling.
- :class:`EdgeHead` — per-edge binary prediction using learned edge
  embeddings from the encoder.

The ``mwpm_teacher`` and ``hybrid`` cases share the same model
architecture; they differ only in loss computation and evaluation
protocol.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.data import Batch
from torch_geometric.nn import global_add_pool, global_max_pool
from torch_geometric.utils import softmax

from constants import Case
from gnn.models.encoder import DetectorGraphEncoder


class LogicalHead(nn.Module):
    """
    Graph-level head for predicting logical observable flips.

    Uses attention-weighted sum pooling (letting the model focus on
    informative detector nodes rather than being dominated by
    zero-syndrome background) concatenated with max pooling.

    Parameters
    ----------
    hidden_dim : int
        Dimensionality of encoder output embeddings.
    num_observables : int
        Number of logical observables to predict (default: 1).
    dropout : float
        Dropout probability in the MLP (default: 0.1).
    """

    def __init__(
        self,
        hidden_dim: int,
        num_observables: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        self.mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
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

        Returns
        -------
        Tensor, shape (B, num_observables)
        """
        attn = softmax(self.gate(h), batch)
        g_attn = global_add_pool(h * attn, batch)
        g_max = global_max_pool(h, batch)
        return self.mlp(torch.cat([g_attn, g_max], dim=-1))


class EdgeHead(nn.Module):
    """
    Per-edge head for predicting edge activations.

    Combines symmetric node pair features with learned edge embeddings
    from the encoder.

    Parameters
    ----------
    hidden_dim : int
        Dimensionality of encoder node/edge embeddings.
    dropout : float
        Dropout probability in the MLP (default: 0.1).
    """

    def __init__(
        self,
        hidden_dim: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(3 * hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        h: torch.Tensor,
        edge_index: torch.Tensor,
        edge_h: torch.Tensor,
        **kwargs: object,
    ) -> torch.Tensor:
        """Predict per-edge logits.

        Parameters
        ----------
        h : Tensor, shape (N_total, hidden_dim)
            Node embeddings from the encoder.
        edge_index : Tensor, shape (2, E_total)
            Directed edge indices.
        edge_h : Tensor, shape (E_total, hidden_dim)
            Learned edge embeddings from the encoder.

        Returns
        -------
        Tensor, shape (E_total,)
        """
        h_u = h[edge_index[0]]
        h_v = h[edge_index[1]]
        inp = torch.cat([h_u + h_v, (h_u - h_v).abs(), edge_h], dim=-1)
        return self.mlp(inp).squeeze(-1)


class QECDecoder(nn.Module):
    """
    Full decoder: encoder backbone + task-specific head.

    Parameters
    ----------
    encoder : DetectorGraphEncoder
        Shared message-passing backbone.
    head : LogicalHead or EdgeHead
        Task-specific prediction head.
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

        Returns
        -------
        Tensor
            - :class:`LogicalHead`: ``(B, num_observables)``
            - :class:`EdgeHead`: ``(E_total,)``
        """
        h, edge_h = self.encoder(batch.x, batch.edge_index, batch.edge_attr)
        return self.head(
            h,
            batch=batch.batch,
            edge_index=batch.edge_index,
            edge_attr=batch.edge_attr,
            edge_h=edge_h,
        )


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
    node_dim, edge_dim, hidden_dim, num_layers, dropout
        Encoder architecture parameters.
    num_observables : int
        Logical observables (only used for ``logical_head``).

    Returns
    -------
    QECDecoder
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
            dropout=dropout,
        )
    else:
        raise ValueError(
            f"Unknown case {case!r}. "
            f"Expected one of: 'logical_head', 'mwpm_teacher', 'hybrid'"
        )

    return QECDecoder(encoder, head)
