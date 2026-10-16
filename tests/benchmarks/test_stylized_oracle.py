"""Independent anchors and artifact checks for the A1 oracle experiment."""

import json

import numpy as np
import pytest

from maas_bundle.benchmarks import StylizedOracleSettings, run_stylized_oracle
from maas_bundle.cli import main


def test_expected_value_includes_the_demand_kink_and_uses_sm_pt_array_order() -> None:
    result = run_stylized_oracle(settings=StylizedOracleSettings(grid_points=5, gradient_samples=8))
    # At (20,20), Phi=3*mu below 20 and Phi=9*mu-120 above 20.
    # Integrating U[16,24] gives (54+78)/2=66, rather than Phi(E[mu])=60.
    assert result.expected_values[2, 2] == pytest.approx(66)
    np.testing.assert_allclose(result.expected_omega[2, 2], [1, 2])
    assert result.realized_values[2, 2] == pytest.approx(60)
    assert result.expected_values[0, 0] == pytest.approx(180)
    assert result.expected_values[-1, -1] == pytest.approx(60)
    assert result.expected_values[0, -1] == pytest.approx(104)  # (PT,SM)=(40,0)
    assert result.expected_values[-1, 0] == pytest.approx(91)  # (PT,SM)=(0,40)
    assert result.diagnostics["passed"]
    assert result.diagnostics["maximum_expected_primal_oracle_error"] < 1e-7
    assert result.diagnostics["maximum_gradient_error_at_smooth_samples"] < 1e-7
    assert result.diagnostics["smooth_gradient_sample_count"] == 8


def test_dual_regime_map_marks_degenerate_maximizers_separately() -> None:
    result = run_stylized_oracle(settings=StylizedOracleSettings(grid_points=5, gradient_samples=2))
    pt, sm = np.meshgrid(result.allowance_axis, result.allowance_axis)
    dual_values = 20 * result.vertices[:, 0] - (
        pt[..., None] * result.vertices[:, 1] + sm[..., None] * result.vertices[:, 2]
    )
    ties = (
        np.sum(
            np.isclose(dual_values, result.realized_values[..., None], atol=1e-9, rtol=0), axis=-1
        )
        > 1
    )
    assert np.any(ties)
    np.testing.assert_array_equal(result.realized_dual_indices == -1, ties)


def test_cli_exports_only_two_panel_figure_and_cleans_retired_outputs(
    tmp_path, monkeypatch, capsys
) -> None:
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "matplotlib-cache"))
    directory = tmp_path / "a1"
    directory.mkdir()
    (directory / "a1_overview.png").write_text("retired figure")
    (directory / "a1_oracle_grid.csv").write_text("retired export")
    (directory / "README.md").write_text("# A1 — Oracle structure\n")
    (directory / "user_notes.txt").write_text("preserve unrelated user notes")
    status = main(["oracle", "stylized", "--grid-points", "5", "--output-dir", str(directory)])
    payload = json.loads(capsys.readouterr().out)
    assert status == 0
    assert payload["grid_evaluations"] == 25
    assert payload["diagnostics_passed"]
    assert payload["mu_values"] == [16, 20, 24]
    assert payload["active_dual_vertex_counts"] == {"16": 5, "20": 5, "24": 5}
    assert set(payload["figures"]) == {"a1_valuation_structure.pdf", "a1_valuation_structure.png"}
    for filename in payload["figures"]:
        assert (directory / filename).stat().st_size > 1000
    assert set(path.name for path in directory.iterdir()) == {
        "a1_valuation_structure.pdf",
        "a1_valuation_structure.png",
        "user_notes.txt",
    }
    assert (directory / "user_notes.txt").read_text() == "preserve unrelated user notes"


def test_moving_boundaries_are_oracle_maximizers_on_both_sides() -> None:
    from maas_bundle.benchmarks.stylized_oracle_plots import (
        prepare_stylized_regimes,
        validate_stylized_figure,
    )

    result = run_stylized_oracle(
        settings=StylizedOracleSettings(grid_points=41, gradient_samples=4)
    )
    regimes = prepare_stylized_regimes(result)
    checks = validate_stylized_figure(result, regimes)
    assert checks["passed"]
    assert checks["original_realized_partition_matches"]
    assert checks["same_cached_oracle_vertices"]
    assert checks["verified_junctions"] == [[16, 16], [20, 20], [24, 24]]
    assert checks["G_M_origin"] == 180
    assert checks["G_M_large_allowance"] == 60
    assert checks["boundary_segment_counts"] == {"16": 5, "20": 5, "24": 5}
    for regime in regimes:
        for boundary in regime.boundaries:
            midpoint = boundary.endpoints.mean(axis=0)
            first, second = boundary.vertex_pair
            normal = result.vertices[first, 1:] - result.vertices[second, 1:]
            normal /= np.linalg.norm(normal)
            negative = midpoint - 1e-4 * normal
            positive = midpoint + 1e-4 * normal
            pi = regime.frequency * result.vertices[:, 0]
            assert np.argmax(pi - result.vertices[:, 1:] @ negative) == first
            assert np.argmax(pi - result.vertices[:, 1:] @ positive) == second


def test_final_figure_has_two_equal_allowance_panels() -> None:
    from maas_bundle.benchmarks.stylized_oracle_plots import (
        build_stylized_figure,
        plt,
        prepare_stylized_regimes,
    )

    result = run_stylized_oracle(settings=StylizedOracleSettings(grid_points=5, gradient_samples=2))
    figure = build_stylized_figure(result, prepare_stylized_regimes(result))
    try:
        panels = [axis for axis in figure.axes if axis.get_title(loc="left")]
        assert len(panels) == 2
        assert panels[0].get_xlim() == panels[1].get_xlim() == (0, 40)
        assert panels[0].get_ylim() == panels[1].get_ylim() == (0, 40)
        assert panels[0].get_aspect() == panels[1].get_aspect() == 1
        assert len(panels[1].get_legend().get_texts()) == 3
    finally:
        plt.close(figure)


@pytest.mark.parametrize(
    "changes",
    [{"grid_points": 2}, {"allowance_max": 0}, {"allowance_max": np.inf}, {"gradient_samples": 0}],
)
def test_invalid_scan_settings_are_rejected(changes) -> None:
    with pytest.raises(ValueError):
        StylizedOracleSettings(**changes)
