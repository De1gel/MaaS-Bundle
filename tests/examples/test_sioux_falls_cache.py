"""Persistent vertices must reproduce values without any geometric rebuild."""

from dataclasses import replace

import numpy as np
import pytest

import maas_bundle.examples.sioux_falls as sioux
from maas_bundle.examples.sioux_falls_config import DEFAULT_SIOUX_FALLS_CONFIG as CONFIG
from maas_bundle.oracle.vertex_cache import load_vertex_cache, save_vertex_cache


def test_saved_vertices_load_without_reprocessing_and_reject_stale_cache(tmp_path, monkeypatch):
    demand = sioux.sioux_falls_od_matrix()
    flow, origin, destination = max(
        (demand[o, d], o + 1, d + 1) for o, d in zip(*np.nonzero(demand), strict=True)
    )
    # build_sioux_falls_problem breaks equal-demand ties by increasing OD indices.
    origin, destination = min(
        (o + 1, d + 1) for o, d in zip(*np.nonzero(demand == flow), strict=True)
    )
    originals = sioux._od_route_oracles(origin, destination)
    oracles = {(origin, destination, s): oracle for s, oracle in enumerate(originals)}
    path = tmp_path / "vertices.npz"
    signature = sioux.sioux_falls_geometry_signature()
    save_vertex_cache(path, oracles, signature=signature, metadata={"complete": False})

    def no_rebuild(*args, **kwargs):
        pytest.fail("loading saved vertices must not rerun network preprocessing")

    monkeypatch.setattr(sioux, "_od_route_oracles", no_rebuild)
    problem = sioux.build_sioux_falls_problem(maximum_od_pairs=1, oracle_cache_path=path)
    for segment in range(2):
        np.testing.assert_array_equal(
            problem.route_oracles[segment][0].vertices, originals[segment].vertices
        )
    with pytest.raises(ValueError, match="incomplete"):
        sioux.build_sioux_falls_problem(oracle_cache_path=path)
    changed = replace(CONFIG, travel_time_factors=(1, 1, 1, 1, 1))
    with pytest.raises(ValueError, match="does not match"):
        sioux.build_sioux_falls_problem(maximum_od_pairs=1, config=changed, oracle_cache_path=path)
    prices_only = replace(CONFIG, initial_prices_usd_per_month=(0, 12, 24))
    assert sioux.sioux_falls_geometry_signature(prices_only) == signature
    restored = load_vertex_cache(path, signature=signature, mode_count=5)
    assert set(restored) == set(oracles)
