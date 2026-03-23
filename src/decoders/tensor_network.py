"""
Tensor-network decoder using NVIDIA cuTensorNet.

Factor-graph formulation: each undirected edge *i* in the detector
graph has a binary error variable ``e_i`` with prior
``P(e_i=1) = p_i``. Each detector *d* imposes a parity constraint:

    XOR(adjacent edge variables) == syndrome bit s_d

Per-edge marginals ``P(e_i=1 | syndrome)`` are computed by
contracting the factor graph with cuTensorNet, keeping one edge
index free at a time. For small codes (d <= 7) this is exact and
fast.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Tuple

import numpy as np

from decoders.base import BaseDecoder, DecoderConfig


logger = logging.getLogger(__name__)


class TNDecoder(BaseDecoder):
    """GPU-accelerated tensor-network decoder.

    Parameters
    ----------
    config : DecoderConfig
        Shared decoder configuration.
    und_pairs : ndarray, shape ``(U, 2)``
        Undirected edge endpoints.
    edge_probs : ndarray, shape ``(U,)``
        Per-edge error probabilities.
    device_id : int
        CUDA device to use for cuTensorNet contractions.
    """

    def __init__(
        self,
        config: DecoderConfig,
        und_pairs: np.ndarray,
        edge_probs: np.ndarray,
        device_id: int = 0,
    ) -> None:
        super().__init__(config)

        self.und_pairs = und_pairs
        self.edge_probs = np.clip(edge_probs.astype(np.float64), 1e-10, 1.0 - 1e-10)
        self.device_id = device_id
        self.num_und = und_pairs.shape[0]

        # Which edges touch each detector (boundary node excluded)
        self._detector_edges: Dict[int, List[int]] = {}
        for eid in range(self.num_und):
            u, v = int(und_pairs[eid, 0]), int(und_pairs[eid, 1])
            for node in (u, v):
                if node < config.num_detectors:
                    self._detector_edges.setdefault(node, []).append(eid)

    @staticmethod
    def _build_check_tensor(degree: int, target_parity: int) -> np.ndarray:
        """Build a parity-check tensor.

        Returns a rank-*degree* tensor of shape ``(2,) * degree``
        with value 1 where the XOR of all indices equals
        *target_parity*, and 0 elsewhere.

        Parameters
        ----------
        degree : int
            Number of adjacent edge variables.
        target_parity : int
            Required parity (0 or 1), from the syndrome bit.
        """
        check = np.zeros([2] * degree, dtype=np.float64)
        for bits in range(2**degree):
            indices = tuple((bits >> j) & 1 for j in range(degree))
            if sum(indices) % 2 == target_parity:
                check[indices] = 1.0
        return check

    def _compute_edge_marginals(self, syndrome: np.ndarray) -> np.ndarray:
        """Exact per-edge marginals via factor-graph contraction.

        Builds a factor graph with:

        - One rank-1 prior tensor ``[1-p_i, p_i]`` per edge variable.
        - One rank-k parity-check tensor per detector, enforcing
          ``XOR(adjacent edges) == syndrome[d]``.

        For each target edge, contracts the full network keeping that
        edge's index free, yielding a 2-element vector proportional
        to ``[P(e=0, syn), P(e=1, syn)]``. Normalising gives the
        conditional marginal.

        Parameters
        ----------
        syndrome : ndarray, shape ``(num_detectors,)``
            Binary syndrome vector.

        Returns
        -------
        ndarray, shape ``(num_undirected,)``, dtype float32
        """
        import cupy as cp
        from cuquantum.tensornet import contract

        with cp.cuda.Device(self.device_id):
            # Prior tensors: one per edge variable
            priors = []
            for eid in range(self.num_und):
                p = self.edge_probs[eid]
                priors.append(cp.array([1.0 - p, p], dtype=cp.float64))

            # Parity-check tensors: one per detector
            checks: List[Tuple[cp.ndarray, List[int]]] = []
            for d in range(self.config.num_detectors):
                adj = self._detector_edges.get(d, [])
                if not adj:
                    continue
                s_d = int(syndrome[d])
                check_np = self._build_check_tensor(len(adj), s_d)
                checks.append((cp.asarray(check_np), list(adj)))

            # Compute per-edge marginals
            marginals = np.zeros(self.num_und, dtype=np.float64)

            for target in range(self.num_und):
                # Interleaved operand list for cuquantum contract:
                #   prior_0, [0], prior_1, [1], ...,
                #   check_d, [adj_edges], ...,
                #   [target]                        <- output modes
                operands: list = []
                for eid in range(self.num_und):
                    operands.append(priors[eid])
                    operands.append([eid])
                for check_tensor, adj in checks:
                    operands.append(check_tensor)
                    operands.append(adj)
                operands.append([target])

                result = cp.asnumpy(contract(*operands)).astype(np.float64)

                z = result[0] + result[1]
                if z > 0:
                    marginals[target] = result[1] / z
                else:
                    # Syndrome inconsistent with graph — use prior
                    marginals[target] = float(self.edge_probs[target])

        return marginals.astype(np.float32)

    def decode(self, syndrome: np.ndarray) -> np.ndarray:
        """Decode via TN marginals -> threshold -> observable prediction."""
        marginals = self._compute_edge_marginals(syndrome)
        predicted = (marginals > 0.5).astype(np.uint8)
        return predicted[: self.config.num_observables]

    def decode_batch(self, syndromes: np.ndarray) -> np.ndarray:
        """Decode a batch of syndromes."""
        results = np.empty(
            (syndromes.shape[0], self.config.num_observables), dtype=np.uint8
        )
        for i in range(syndromes.shape[0]):
            results[i] = self.decode(syndromes[i])
        return results


class TNSoftLabelGenerator:
    """Generate soft teacher labels via TN marginals.

    Produces continuous per-edge error probabilities that serve as
    richer supervision than binary MWPM labels.

    Parameters
    ----------
    config : DecoderConfig
        Decoder configuration.
    und_pairs : ndarray, shape ``(U, 2)``
        Undirected edge endpoints.
    edge_probs : ndarray, shape ``(U,)``
        Per-edge error probabilities.
    device_id : int
        CUDA device for cuTensorNet.
    """

    def __init__(
        self,
        config: DecoderConfig,
        und_pairs: np.ndarray,
        edge_probs: np.ndarray,
        device_id: int = 0,
    ) -> None:
        self._decoder = TNDecoder(config, und_pairs, edge_probs, device_id)

    def generate_soft_labels(
        self,
        syndromes: np.ndarray,
        chunk_size: int = 100,
    ) -> np.ndarray:
        """Compute soft per-edge labels for a batch of syndromes.

        Parameters
        ----------
        syndromes : ndarray, shape ``(N, num_detectors)``
            Binary syndrome vectors.
        chunk_size : int
            Processing chunk size (for progress / memory).

        Returns
        -------
        ndarray, shape ``(N, num_undirected_edges)``, dtype float32
        """
        n_samples = syndromes.shape[0]
        num_und = self._decoder.num_und
        soft_labels = np.empty((n_samples, num_und), dtype=np.float32)

        for off in range(0, n_samples, chunk_size):
            end = min(off + chunk_size, n_samples)
            for i in range(off, end):
                soft_labels[i] = self._decoder._compute_edge_marginals(syndromes[i])

        return soft_labels
