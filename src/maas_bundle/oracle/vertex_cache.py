"""Portable, validated storage of complete route-dual vertex sets."""

from __future__ import annotations

import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np

from maas_bundle.oracle.route_value import RouteValueOracle


def save_vertex_cache(path, oracles, *, signature: str, metadata: dict) -> None:
    """Atomically replace a compressed cache; keys are (origin, destination, segment)."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {"metadata": np.asarray(json.dumps({**metadata, "signature": signature, "schema": 1}))}
    arrays.update({f"od_{o}_{d}_{s}": oracle.vertices for (o, d, s), oracle in oracles.items()})
    with NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as stream:
        temporary = Path(stream.name)
        try:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


def load_vertex_cache(path, *, signature: str, mode_count: int):
    """Reject stale geometry, nonfinite values and unsafe pickle-based archives."""

    oracles = {}
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata"]))
        if metadata.get("schema") != 1 or metadata.get("signature") != signature:
            raise ValueError("vertex cache does not match the network and economic geometry")
        for name in archive.files:
            if name == "metadata":
                continue
            prefix, origin, destination, segment = name.split("_")
            if prefix != "od":
                raise ValueError("invalid vertex cache key")
            vertices = archive[name]
            if vertices.ndim != 2 or vertices.shape[1] != mode_count + 1 or not len(vertices):
                raise ValueError("invalid cached vertex dimensions")
            oracles[int(origin), int(destination), int(segment)] = RouteValueOracle(vertices)
    return oracles
