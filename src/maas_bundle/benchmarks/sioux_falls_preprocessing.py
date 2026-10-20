"""Persist all 1,056 OD/segment vertex sets before any bundle-design solve."""

from pathlib import Path
from time import perf_counter

import numpy as np

from maas_bundle.examples.sioux_falls import (
    MODE_NAMES,
    _od_route_oracles,
    sioux_falls_geometry_signature,
    sioux_falls_od_matrix,
)
from maas_bundle.examples.sioux_falls_config import DEFAULT_SIOUX_FALLS_CONFIG
from maas_bundle.oracle.vertex_cache import load_vertex_cache, save_vertex_cache

DEFAULT_VERTEX_CACHE = "results/sioux_falls_five_modes/dual_vertices.npz"


def preprocess_sioux_falls_oracles(
    output_path=DEFAULT_VERTEX_CACHE,
    *,
    config=DEFAULT_SIOUX_FALLS_CONFIG,
    progress=None,
):
    """Resume matching checkpoints and atomically save after each 25 OD pairs."""

    started = perf_counter()
    signature = sioux_falls_geometry_signature(config)
    path = Path(output_path)
    oracles = (
        load_vertex_cache(path, signature=signature, mode_count=len(MODE_NAMES))
        if path.exists()
        else {}
    )
    demand = sioux_falls_od_matrix()
    pairs = sorted(
        [(float(demand[o, d]), o + 1, d + 1) for o, d in zip(*np.nonzero(demand), strict=True)],
        reverse=True,
    )
    expected = {(o, d, s) for _, o, d in pairs for s in range(2)}
    if not set(oracles) <= expected:
        raise ValueError("vertex cache contains unexpected OD/segment keys")
    metadata = {"experiment": config.metadata(), "complete": False}
    for number, (_, origin, destination) in enumerate(pairs, 1):
        if any((origin, destination, segment) not in oracles for segment in range(2)):
            for segment, oracle in enumerate(_od_route_oracles(origin, destination, config)):
                oracles[origin, destination, segment] = oracle
        if number % 25 == 0 or number == len(pairs):
            metadata["complete"] = set(oracles) == expected
            save_vertex_cache(path, oracles, signature=signature, metadata=metadata)
            if progress is not None:
                progress(f"{number}/{len(pairs)} OD pairs; {len(oracles)} vertex sets saved")
    counts = [oracle.vertex_count for oracle in oracles.values()]
    return {
        "cache_path": str(path),
        "geometry_signature": signature,
        "positive_od_pairs": len(pairs),
        "traveler_types": len(oracles),
        "mode_count": len(MODE_NAMES),
        "total_vertices": sum(counts),
        "minimum_vertices": min(counts),
        "maximum_vertices": max(counts),
        "complete": set(oracles) == expected,
        "preprocessing_seconds": perf_counter() - started,
    }
