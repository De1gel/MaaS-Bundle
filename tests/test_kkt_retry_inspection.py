"""Ensure retry diagnostics serialize and independently enforce complementarity."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from maas_bundle.design.menu import BundleMenu
from maas_bundle.design.regional import RegionalFirstOrderCertificate

_PATH = Path(__file__).parents[1] / "scripts/retry-nlp-with-kkt.py"
_SPEC = importlib.util.spec_from_file_location("kkt_retry_inspection", _PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def diagnostic_fixture(slack):
    def certificate(vector, *, active_tolerance, **kwargs):
        active = active_tolerance >= slack
        return RegionalFirstOrderCertificate(
            primal_violation=0.0,
            stationarity_residual=0.0 if active else 1.0,
            normalized_stationarity_residual=0.0 if active else 0.5,
            active_inequality_count=int(active),
            active_lower_bound_count=0,
            feasibility_tolerance=1e-7,
            stationarity_tolerance=1e-5,
            active_inequality_indices=(0,) if active else (),
            inequality_multipliers=(1.0,) if active else (),
        )

    regional = SimpleNamespace(
        layout=SimpleNamespace(pack=lambda menu: np.ones(1)),
        constraint_values=lambda x: np.array([-slack]),
        maximum_constraint_violation=lambda x: 0.0,
        profit=lambda x: 1.0,
        first_order_certificate=certificate,
    )
    menu = BundleMenu(np.array([0.0, 1.0]), np.zeros((2, 1)))
    return _MODULE.diagnose(regional, regional, menu)


def test_passing_independent_check_is_json_serializable():
    record = diagnostic_fixture(1e-7)
    decoded = json.loads(json.dumps(record))
    assert decoded["first_order_checks"][2]["independent_screen_passed"] is True


def test_adapter_stationarity_pass_does_not_replace_complementarity_check():
    record = diagnostic_fixture(1e-3)
    loose = record["first_order_checks"][-1]
    assert loose["adapter_first_order_passed"]
    assert not loose["independent_screen_passed"]
