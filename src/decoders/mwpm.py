"""Minimum Weight Perfect Matching decoder using PyMatching.

Accepts either static weights (from a detector error model) or
dynamic weights derived from GNN edge logits via the
:meth:`~MWPMDecoder.from_gnn_logits` class method.
"""

from __future__ import annotations

import numpy as np
import pymatching

from decoders.base import BaseDecoder, DecoderConfig


class MWPMDecoder(BaseDecoder):
    """MWPM decoder backed by PyMatching.

    Parameters
    ----------
    config : DecoderConfig
        Shared decoder configuration.
    und_pairs : ndarray, shape ``(U, 2)``
        Undirected edge endpoints.
    weights : ndarray, shape ``(U,)``
        Per-undirected-edge weights (positive, log-likelihood ratios).
    """

    def __init__(
        self,
        config: DecoderConfig,
        und_pairs: np.ndarray,
        weights: np.ndarray,
    ) -> None:
        super().__init__(config)
        self._matching = self._build_matching(und_pairs, weights)

    def _build_matching(
        self,
        und_pairs: np.ndarray,
        weights: np.ndarray,
    ) -> pymatching.Matching:
        """Construct a ``pymatching.Matching`` graph."""
        m = pymatching.Matching()
        num_edges = und_pairs.shape[0]

        for eid in range(num_edges):
            u, v = int(und_pairs[eid, 0]), int(und_pairs[eid, 1])
            w = float(max(weights[eid], 1e-8))

            if self.config.boundary_node is not None and self.config.boundary_node in (
                u,
                v,
            ):
                det = v if u == self.config.boundary_node else u
                if 0 <= det < self.config.num_detectors:
                    m.add_boundary_edge(det, fault_ids={eid}, weight=w)
            elif u < self.config.num_detectors and v < self.config.num_detectors:
                m.add_edge(u, v, fault_ids={eid}, weight=w)

        m.ensure_num_fault_ids(num_edges)
        return m

    def decode(self, syndrome: np.ndarray) -> np.ndarray:
        """Decode a single syndrome via MWPM."""
        pred = self._matching.decode(syndrome)
        return pred[: self.config.num_observables]

    def decode_batch(self, syndromes: np.ndarray) -> np.ndarray:
        """Decode a batch of syndromes via MWPM."""
        results = self._matching.decode_batch(syndromes)
        return results[:, : self.config.num_observables]

    @classmethod
    def from_gnn_logits(
        cls,
        config: DecoderConfig,
        und_pairs: np.ndarray,
        directed_logits: np.ndarray,
        dir_to_undir: np.ndarray,
        num_undirected: int,
    ) -> MWPMDecoder:
        """Construct from GNN-predicted directed edge logits.

        Averages directed logits into per-undirected-edge logits,
        converts to log-likelihood-ratio weights, and builds the
        matching graph.

        Parameters
        ----------
        config : DecoderConfig
            Shared decoder configuration.
        und_pairs : ndarray, shape ``(U, 2)``
            Undirected edge endpoints.
        directed_logits : ndarray, shape ``(E,)``
            Raw logits from the GNN edge head (directed edges).
        dir_to_undir : ndarray, shape ``(E,)``
            Mapping from directed edge index to undirected edge index.
        num_undirected : int
            Number of unique undirected edges ``U``.

        Returns
        -------
        MWPMDecoder
        """
        und_logits = _directed_to_undirected_logits(
            directed_logits, dir_to_undir, num_undirected
        )
        weights = _logits_to_weights(und_logits)
        return cls(config, und_pairs, weights)


def _directed_to_undirected_logits(
    directed_logits: np.ndarray,
    dir_to_undir: np.ndarray,
    num_undirected: int,
) -> np.ndarray:
    """Average directed edge logits into per-undirected-edge means."""
    logit_sum = np.zeros(num_undirected, dtype=np.float64)
    logit_count = np.zeros(num_undirected, dtype=np.int32)
    np.add.at(logit_sum, dir_to_undir, directed_logits)
    np.add.at(logit_count, dir_to_undir, 1)
    return logit_sum / np.maximum(logit_count, 1)


def _logits_to_weights(logits: np.ndarray) -> np.ndarray:
    """Convert logits to positive log-likelihood-ratio weights."""
    prob = 1.0 / (1.0 + np.exp(-logits))
    prob = np.clip(prob, 1e-7, 1.0 - 1e-7)
    weights = np.log((1.0 - prob) / prob)
    return np.maximum(weights, 1e-8)
