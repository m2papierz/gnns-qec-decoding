"""Shared CLI utilities for QEC scripts."""

import logging
import sys


def setup_logging(verbose: bool = False) -> None:
    """
    Configure root logging.

    Parameters
    ----------
    verbose : bool
        Enable ``DEBUG`` level if True, otherwise ``INFO``.
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )
