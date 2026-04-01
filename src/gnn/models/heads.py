"""
Decoding heads and full decoder for QEC graph decoding.

Two head architectures cover the training modes:

- :class:`LogicalHead` — graph-level classification of observable flips
  using attention-weighted node pooling and edge embedding pooling.
- :class:`EdgeHead` — per-edge binary prediction using learned edge
  embeddings from the encoder.

The ``edge`` case uses :class:`EdgeHead`; ``direct`` uses
:class:`LogicalHead`.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.data import Batch
from torch_geometric.nn import global_add_pool, global_max_pool
from torch_geometric.utils import softmax

from constants import Case
from gnn.models.encoder import DetectorGraphEncoder
from gnn.models.ops import edge_mean_pool, symmetric_edge_features


class LogicalHead(nn.Module):
    """
    Graph-level head for predicting logical observable flips.

    Combines attention-weighted node pooling with mean edge embedding
    pooling, giving the head access to both node- and edge-level
    representations from the encoder.

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
        # Input: attn_pool (H) | max_pool (H) | edge_mean_pool (H)
        self.mlp = nn.Sequential(
            nn.Linear(3 * hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_observables),
        )

    def forward(
        self,
        h: torch.Tensor,
        batch: torch.Tensor,
        edge_index: torch.Tensor,
        edge_h: torch.Tensor,
        **kwargs: object,
    ) -> torch.Tensor:
        """Predict logical observable logits.

        Parameters
        ----------
        h : Tensor, shape (N_total, hidden_dim)
            Node embeddings from the encoder (batched).
        batch : Tensor, shape (N_total,)
            PyG batch vector mapping each node to its graph index.
        edge_index : Tensor, shape (2, E_total)
            Directed edge indices (used to assign edges to graphs).
        edge_h : Tensor, shape (E_total, hidden_dim)
            Learned edge embeddings from the encoder.

        Returns
        -------
        Tensor, shape (B, num_observables)
        """
        attn = softmax(self.gate(h), batch)
        g_attn = global_add_pool(h * attn, batch)
        g_max = global_max_pool(h, batch)

        edge_batch = batch[edge_index[0]]
        n_graphs = int(batch.max()) + 1
        g_edge = edge_mean_pool(edge_h, edge_batch, n_graphs)

        return self.mlp(torch.cat([g_attn, g_max, g_edge], dim=-1))


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
        inp = symmetric_edge_features(h, edge_index, edge_h)
        return self.mlp(inp).squeeze(-1)


class QECDecoder(nn.Module):
    """
    Full decoder: encoder backbone + task-specific head.

    For the ``edge`` case the head predicts a *delta* logit which is
    added to the prior edge logit ``log(p / (1-p))`` from the dataset,
    so the model only needs to learn the correction to the baseline.

    Parameters
    ----------
    encoder : DetectorGraphEncoder
        Shared message-passing backbone.
    head : LogicalHead or EdgeHead
        Task-specific prediction head.
    case : str
        Training case (``"direct"`` or ``"edge"``).
    """

    def __init__(
        self,
        encoder: DetectorGraphEncoder,
        head: nn.Module,
        case: Case = "direct",
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.head = head
        self.case = case

    def forward(self, batch: Batch) -> torch.Tensor:
        """Run full decode pipeline on a batched graph.

        Returns
        -------
        Tensor
            - :class:`LogicalHead`: ``(B, num_observables)``
            - :class:`EdgeHead`: ``(E_total,)`` final logits
              (``prior_edge_logit + delta`` for edge case)
        """
        h, edge_h = self.encoder(batch.x, batch.edge_index, batch.edge_attr)
        out = self.head(
            h,
            batch=batch.batch,
            edge_index=batch.edge_index,
            edge_attr=batch.edge_attr,
            edge_h=edge_h,
        )

        # Residual prediction: model outputs delta, add prior baseline.
        if self.case == "edge":
            prior = getattr(batch, "prior_edge_logit", None)
            if prior is not None:
                out = prior + out

        return out


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
    case : {"direct", "edge"}
        Training case.  Determines which head is attached.
    node_dim, edge_dim, hidden_dim, num_layers, dropout
        Encoder architecture parameters.
    num_observables : int
        Logical observables (only used for ``direct``).

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

    if case == "direct":
        head: nn.Module = LogicalHead(
            hidden_dim=hidden_dim,
            num_observables=num_observables,
            dropout=dropout,
        )
    elif case == "edge":
        head = EdgeHead(
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
    else:
        raise ValueError(
            f"Unknown case {case!r}. " f"Expected one of: 'direct', 'edge'"
        )

    return QECDecoder(encoder, head, case=case)
