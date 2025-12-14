"""Raw data generation using Stim circuit sampling."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import stim
from tqdm import tqdm

from constants import DEM_EXT, STIM_EXT
from qec_generator.config import Config
from qec_generator.graph import DetectorGraph, build_detector_graph
from qec_generator.utils import save_json, save_npy, stable_seed


logger = logging.getLogger(__name__)


def build_circuit(
    cfg: Config,
    distance: int,
    rounds: int,
    p: float,
) -> stim.Circuit:
    """
    Build a Stim surface code circuit for given parameters.

    Parameters
    ----------
    cfg : Config
        Configuration with family and noise parameters.
    distance : int
        Code distance.
    rounds : int
        Number of syndrome measurement rounds.
    p : float
        Physical error probability.

    Returns
    -------
    stim.Circuit
        Generated surface code circuit.
    """
    return stim.Circuit.generated(
        f"surface_code:{cfg.family}",
        distance=distance,
        rounds=rounds,
        **cfg.resolve_noise(p),
    )


def _write_graph(
    cfg: Config,
    out_dir: Path,
    graph: DetectorGraph,
    distance: int,
    rounds: int,
    p: float,
    overwrite: bool,
) -> None:
    """
    Write graph tensors and metadata to disk.

    Parameters
    ----------
    cfg : Config
        Configuration object.
    out_dir : Path
        Output directory for this setting.
    graph : DetectorGraph
        Constructed detector graph.
    distance : int
        Code distance.
    rounds : int
        Number of rounds.
    p : float
        Error probability.
    overwrite : bool
        Whether to overwrite existing files.
    """
    gdir = out_dir / "graph"
    gdir.mkdir(parents=True, exist_ok=True)

    # Save graph tensors
    save_npy(gdir / "edge_index.npy", graph.edge_index, overwrite)
    save_npy(gdir / "edge_error_prob.npy", graph.edge_error_prob, overwrite)
    save_npy(gdir / "edge_weight.npy", graph.edge_weight, overwrite)
    save_npy(gdir / "node_coords.npy", graph.node_coords, overwrite)
    save_npy(gdir / "node_is_boundary.npy", graph.node_is_boundary, overwrite)

    # Save metadata
    save_json(
        gdir / "meta.json",
        {
            "family": cfg.family,
            "distance": distance,
            "rounds": rounds,
            "error_prob": p,
            "num_nodes": graph.num_nodes,
            "num_detectors": graph.num_detectors,
            "num_observables": graph.num_observables,
            "has_boundary": graph.has_boundary,
        },
        overwrite,
    )


def _generate_split(
    cfg: Config,
    out_dir: Path,
    circuit: stim.Circuit,
    dem: stim.DetectorErrorModel,
    split: str,
    n_samples: int,
    seed: int | None,
    overwrite: bool,
    show_progress: bool,
) -> None:
    """
    Generate syndrome/logical arrays for one split.

    Parameters
    ----------
    cfg : Config
        Configuration object.
    out_dir : Path
        Output directory for this setting.
    circuit : stim.Circuit
        Compiled circuit.
    dem : stim.DetectorErrorModel
        Detector error model.
    split : str
        Split name ("train", "val", or "test").
    n_samples : int
        Number of samples to generate.
    seed : int or None
        Random seed for this split.
    overwrite : bool
        Whether to overwrite existing files.
    show_progress : bool
        Show tqdm progress bar.
    """
    if n_samples <= 0:
        logger.warning("Skipping split '%s' with n_samples=%d", split, n_samples)
        return

    syn_path = out_dir / f"{split}_syndrome.npy"
    log_path = out_dir / f"{split}_logical.npy"
    meta_path = out_dir / f"{split}_meta.json"

    if syn_path.exists() and log_path.exists() and not overwrite:
        logger.debug("Split '%s' already exists, skipping", split)
        return

    num_det, num_obs = dem.num_detectors, dem.num_observables
    sampler = circuit.compile_detector_sampler(seed=seed)

    syndromes = np.empty((n_samples, num_det), dtype=np.uint8)
    logicals = np.empty((n_samples, num_obs), dtype=np.uint8)

    chunk = max(1, cfg.chunk_size)
    iterator = range(0, n_samples, chunk)
    if show_progress:
        iterator = tqdm(iterator, desc=f"{split}", leave=False)

    offset = 0
    for _ in iterator:
        batch_size = min(chunk, n_samples - offset)
        dets, obs = sampler.sample(
            shots=batch_size, separate_observables=True, bit_packed=False
        )
        syndromes[offset : offset + batch_size] = dets.astype(np.uint8, copy=False)
        logicals[offset : offset + batch_size] = obs.astype(np.uint8, copy=False)
        offset += batch_size

    # Save arrays
    save_npy(syn_path, syndromes, overwrite=True)
    save_npy(log_path, logicals, overwrite=True)

    # Save metadata
    save_json(
        meta_path,
        {
            "split": split,
            "num_samples": n_samples,
            "num_detectors": num_det,
            "num_observables": num_obs,
            "seed": seed,
            "chunk_size": cfg.chunk_size,
            "stim_version": getattr(stim, "__version__", "unknown"),
        },
        overwrite=True,
    )


def generate_for_setting(
    cfg: Config,
    distance: int,
    rounds: int,
    p: float,
    overwrite: bool = False,
    save_artifacts: bool = True,
    show_progress: bool = True,
) -> None:
    """
    Generate graph and all splits for a single setting.

    Parameters
    ----------
    cfg : Config
        Configuration object.
    distance : int
        Code distance.
    rounds : int
        Number of syndrome measurement rounds.
    p : float
        Physical error probability.
    overwrite : bool, default=False
        Overwrite existing files.
    save_artifacts : bool, default=True
        Save circuit and DEM files for debugging.
    show_progress : bool, default=True
        Display progress bars.
    """
    out_dir = cfg.setting_dir(distance, rounds, p)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build circuit and error model
    circuit = build_circuit(cfg, distance, rounds, p)
    dem = circuit.detector_error_model(decompose_errors=True)

    # Save circuit/DEM for debugging
    if save_artifacts:
        circuit_path = out_dir / f"circuit{STIM_EXT}"
        dem_path = out_dir / f"model{DEM_EXT}"
        if overwrite or not circuit_path.exists():
            circuit.to_file(circuit_path)
            logger.debug("Saved circuit to %s", circuit_path)
        if overwrite or not dem_path.exists():
            dem.to_file(dem_path)
            logger.debug("Saved DEM to %s", dem_path)

    # Build and save detector graph
    graph = build_detector_graph(circuit, dem, include_boundary=True)
    _write_graph(
        cfg,
        out_dir=out_dir,
        graph=graph,
        distance=distance,
        rounds=rounds,
        p=p,
        overwrite=overwrite,
    )

    # Generate each split
    for split, n_samples in cfg.num_samples.items():
        split_seed = stable_seed(
            f"family={cfg.family}",
            f"d={distance}",
            f"r={rounds}",
            f"p={p:.12g}",
            f"split={split}",
            base=cfg.seed,
        )
        _generate_split(
            cfg,
            out_dir=out_dir,
            circuit=circuit,
            dem=dem,
            split=split,
            n_samples=n_samples,
            seed=split_seed,
            overwrite=overwrite,
            show_progress=show_progress,
        )


def generate_raw_data(
    cfg: Config,
    overwrite: bool = False,
    save_artifacts: bool = True,
) -> None:
    """
    Generate raw data for all settings in configuration.

    Creates directory structure:
    raw_data_dir/
        d{distance}_r{rounds}_p{error_prob}/
            graph/
                edge_index.npy, edge_error_prob.npy, ...
            {split}_syndrome.npy
            {split}_logical.npy
            circuit.stim
            model.dem

    Parameters
    ----------
    cfg : Config
        Configuration with all settings to generate.
    overwrite : bool, default=False
        Overwrite existing files.
    save_artifacts : bool, default=True
        Save Stim circuit and DEM files.
    """
    cfg.raw_data_dir.mkdir(parents=True, exist_ok=True)
    settings = list(cfg.iter_settings())

    logger.info("Generating raw data for %d settings", len(settings))

    for d, r, p in tqdm(settings, desc="Settings"):
        generate_for_setting(
            cfg,
            distance=d,
            rounds=r,
            p=p,
            overwrite=overwrite,
            save_artifacts=save_artifacts,
            show_progress=True,
        )

    logger.info("Raw data generation complete")
