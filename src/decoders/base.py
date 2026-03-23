"""Abstract decoder interface for QEC."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class DecoderConfig:
    """Shared configuration for all QEC decoders.

    Parameters
    ----------
    num_detectors : int
        Number of detector nodes in the graph.
    num_observables : int
        Number of logical observables to predict.
    has_boundary : bool
        Whether the graph contains a boundary node.
    boundary_node : int or None
        Index of the boundary node (typically ``num_detectors``).
    """

    num_detectors: int
    num_observables: int
    has_boundary: bool
    boundary_node: Optional[int] = None


class BaseDecoder(ABC):
    """Base class for all QEC decoders.

    Subclasses implement :meth:`decode` for single-syndrome decoding
    and :meth:`decode_batch` for vectorised decoding.

    Parameters
    ----------
    config : DecoderConfig
        Decoder configuration shared across all implementations.
    """

    def __init__(self, config: DecoderConfig) -> None:
        self.config = config

    @abstractmethod
    def decode(self, syndrome: np.ndarray) -> np.ndarray:
        """Decode a single syndrome into predicted observable flips.

        Parameters
        ----------
        syndrome : ndarray, shape ``(num_detectors,)``
            Binary syndrome vector.

        Returns
        -------
        ndarray, shape ``(num_observables,)``
            Predicted observable flips in ``{0, 1}``.
        """
        ...

    @abstractmethod
    def decode_batch(self, syndromes: np.ndarray) -> np.ndarray:
        """Decode a batch of syndromes.

        Parameters
        ----------
        syndromes : ndarray, shape ``(N, num_detectors)``
            Batch of syndrome vectors.

        Returns
        -------
        ndarray, shape ``(N, num_observables)``
            Predicted observable flips for each shot.
        """
        ...

    @property
    def name(self) -> str:
        """Human-readable decoder name."""
        return self.__class__.__name__
