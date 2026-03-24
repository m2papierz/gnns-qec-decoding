"""BP+OSD decoder using NVIDIA CUDA-Q QEC.

Wraps the ``cudaq-qec`` QLDPC decoder (belief-propagation with ordered
statistics decoding post-processing) behind the :class:`BaseDecoder`
interface.
"""

from __future__ import annotations

import logging
from typing import Any

import cudaq_qec
import numpy as np
from scipy.sparse import csr_matrix

from decoders.base import BaseDecoder, DecoderConfig


logger = logging.getLogger(__name__)


def _build_parity_check_matrix(
    und_pairs: np.ndarray,
    num_detectors: int,
) -> csr_matrix:
    """Build a sparse parity-check matrix from undirected edges.

    Constructs ``H`` of shape ``(num_detectors, U)`` where ``H[d, e] = 1``
    iff detector *d* is an endpoint of undirected edge *e*. Boundary
    nodes (index >= *num_detectors*) are excluded.

    Parameters
    ----------
    und_pairs : ndarray, shape ``(U, 2)``
        Undirected edge endpoints.
    num_detectors : int
        Number of detector nodes.

    Returns
    -------
    csr_matrix, shape ``(num_detectors, U)``
    """
    U = und_pairs.shape[0]
    rows: list[int] = []
    cols: list[int] = []

    for e in range(U):
        for endpoint in (int(und_pairs[e, 0]), int(und_pairs[e, 1])):
            if 0 <= endpoint < num_detectors:
                rows.append(endpoint)
                cols.append(e)

    data = np.ones(len(rows), dtype=np.uint8)
    return csr_matrix((data, (rows, cols)), shape=(num_detectors, U))


class BPOSDDecoder(BaseDecoder):
    """BP+OSD decoder backed by NVIDIA CUDA-Q QEC.

    Decodes syndromes into an error estimate vector, then projects onto
    logical observables via the ``observable_flips`` mask.

    Parameters
    ----------
    config : DecoderConfig
        Shared decoder configuration.
    und_pairs : ndarray, shape ``(U, 2)``
        Undirected edge endpoints.
    edge_probs : ndarray, shape ``(U,)``
        Per-undirected-edge error probabilities in ``(0, 1)``.
    observable_flips : ndarray, shape ``(U, num_obs)``, bool
        Per-undirected-edge observable flip mask.
    bp_method : int
        BP variant selector (0 = product-sum, 1 = min-sum).
    max_iterations : int
        Maximum BP iterations.
    osd_order : int
        OSD post-processing order (0 = OSD-0 / combination sweep).
    """

    def __init__(
        self,
        config: DecoderConfig,
        und_pairs: np.ndarray,
        edge_probs: np.ndarray,
        observable_flips: np.ndarray,
        *,
        bp_method: int = 0,
        max_iterations: int = 30,
        osd_order: int = 0,
    ) -> None:
        super().__init__(config)

        if observable_flips.shape[0] != und_pairs.shape[0]:
            raise ValueError(
                f"observable_flips rows ({observable_flips.shape[0]}) "
                f"!= num undirected edges ({und_pairs.shape[0]})"
            )

        self._observable_flips = observable_flips.astype(np.int32, copy=False)
        self._edge_probs = np.clip(
            edge_probs.astype(np.float64, copy=False),
            1e-15,
            1.0 - 1e-15,
        )

        H = _build_parity_check_matrix(und_pairs, config.num_detectors)
        self._num_edges = und_pairs.shape[0]

        self._decoder = cudaq_qec.get_decoder(
            "nv-qldpc-decoder",
            H.toarray().astype(np.uint8),
            error_rate=self._edge_probs.tolist(),
            bp_method=bp_method,
            max_iterations=max_iterations,
            osd_order=osd_order,
        )

        logger.debug(
            "BPOSDDecoder: %d detectors, %d edges, bp_method=%d, "
            "max_iter=%d, osd_order=%d",
            config.num_detectors,
            self._num_edges,
            bp_method,
            max_iterations,
            osd_order,
        )

    def decode(self, syndrome: np.ndarray) -> np.ndarray:
        """Decode a single syndrome via BP+OSD.

        Parameters
        ----------
        syndrome : ndarray, shape ``(num_detectors,)``

        Returns
        -------
        ndarray, shape ``(num_observables,)``
            Predicted observable flips in ``{0, 1}``.
        """
        syn = syndrome.astype(np.float64).tolist()

        result = self._decoder.decode(syn)

        error_est = np.asarray(result.result, dtype=np.int32)[: self._num_edges]

        obs = (error_est @ self._observable_flips) % 2
        return obs[: self.config.num_observables].astype(np.uint8)

    def decode_batch(self, syndromes: np.ndarray) -> np.ndarray:
        """Decode a batch of syndromes.

        Parameters
        ----------
        syndromes : ndarray, shape ``(N, num_detectors)``

        Returns
        -------
        ndarray, shape ``(N, num_observables)``
        """
        n = syndromes.shape[0]
        out = np.empty((n, self.config.num_observables), dtype=np.uint8)
        for i in range(n):
            out[i] = self.decode(syndromes[i])
        return out

    @classmethod
    def from_gnn_logits(
        cls,
        config: DecoderConfig,
        und_pairs: np.ndarray,
        directed_logits: np.ndarray,
        dir_to_undir: np.ndarray,
        num_undirected: int,
        observable_flips: np.ndarray,
        **kwargs: Any,
    ) -> BPOSDDecoder:
        """Construct from GNN-predicted directed edge logits.

        Averages directed logits per undirected edge, applies sigmoid to
        obtain error probabilities, then constructs the decoder.

        Parameters
        ----------
        config : DecoderConfig
            Shared decoder configuration.
        und_pairs : ndarray, shape ``(U, 2)``
            Undirected edge endpoints.
        directed_logits : ndarray, shape ``(E,)``
            Raw logits from the GNN edge head.
        dir_to_undir : ndarray, shape ``(E,)``
            Directed-to-undirected edge mapping.
        num_undirected : int
            Number of unique undirected edges.
        observable_flips : ndarray, shape ``(U, num_obs)``
            Per-undirected-edge observable flip mask.
        **kwargs
            Forwarded to :class:`BPOSDDecoder` constructor
            (``bp_method``, ``max_iterations``, ``osd_order``).

        Returns
        -------
        BPOSDDecoder
        """
        from decoders.mwpm import _directed_to_undirected_logits

        und_logits = _directed_to_undirected_logits(
            directed_logits, dir_to_undir, num_undirected
        )
        probs = 1.0 / (1.0 + np.exp(-und_logits))
        probs = np.clip(probs, 1e-7, 1.0 - 1e-7)
        return cls(config, und_pairs, probs, observable_flips, **kwargs)
