"""Deterministic seed derivation for reproducible sampling."""

from __future__ import annotations

import hashlib


def stable_seed(*parts: str, base: int | None) -> int | None:
    """
    Derive a reproducible seed from base seed and string identifiers.

    Uses BLAKE2b hashing to combine base seed with arbitrary string parts,
    ensuring deterministic seed generation across runs.

    Parameters
    ----------
    *parts : str
        String identifiers to incorporate into the seed (e.g., "d=5", "p=0.01").
    base : int or None
        Base seed value. If None, returns None.

    Returns
    -------
    int or None
        Derived seed, or None if base is None.
    """
    if base is None:
        return None

    h = hashlib.blake2b(digest_size=8)
    h.update(str(base).encode())
    for part in parts:
        h.update(b"|")
        h.update(part.encode())
    return int.from_bytes(h.digest(), "little", signed=False)
