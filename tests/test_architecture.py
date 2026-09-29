"""Enforce dependency boundaries between instance data and experiment drivers."""

import ast
from pathlib import Path

import pytest

SOURCE = Path(__file__).resolve().parents[1] / "src" / "maas_bundle"


@pytest.mark.parametrize(
    "files,forbidden",
    [
        (
            "examples/*.py",
            (
                "maas_bundle.benchmarks",
                "maas_bundle.cli",
                "maas_bundle.design.primal",
                "maas_bundle.design.regional",
                "maas_bundle.design.validation",
            ),
        ),
        ("design/*.py", ("maas_bundle.examples", "maas_bundle.benchmarks", "maas_bundle.cli")),
        (
            "oracle/*.py",
            (
                "maas_bundle.design",
                "maas_bundle.examples",
                "maas_bundle.benchmarks",
                "maas_bundle.cli",
            ),
        ),
        ("cli.py", ("maas_bundle.design.primal", "maas_bundle.design.regional")),
    ],
)
def test_dependencies_point_from_experiments_toward_models(files, forbidden):
    for path in SOURCE.glob(files):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Import):
                imports = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports = [node.module] + [f"{node.module}.{alias.name}" for alias in node.names]
            else:
                continue
            for name in imports:
                assert not any(
                    name == prefix or name.startswith(prefix + ".") for prefix in forbidden
                ), f"{path.name} imports {name}"
