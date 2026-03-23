"""Deployment and inference tooling for trained GNN decoders.

Provides :class:`InferenceEngine` — a unified inference wrapper
supporting PyTorch, ``torch.compile``, and TensorRT backends with
built-in benchmarking via CUDA events.
"""

from deploy.engine import InferenceBackend, InferenceEngine


__all__ = ["InferenceBackend", "InferenceEngine"]
