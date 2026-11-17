"""Tests for the calibrated ranges, the recommendation backtest results and the 'never recommend worse' rule.

The full calibration and backtest take minutes (tools/build_trust_report.py); these tests check the stored
results and the small helpers, so they run in seconds.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ecobuild_agent import OBJECTIVES, Constraints, run_agent  # noqa: E402
from ecobuild_ai import EcoBuildTools  # noqa: E402
from ecobuild_core import FEATURES, EcoBuildModels, gap_table, pair_gap, range_half_width  # noqa: E402
from ecobuild_econ import Assumptions, business_case  # noqa: E402
from ecobuild_trust import HeldOutModels, conformal_quantile, with_ids  # noqa: E402


@pytest.fixture(scope="module")
def models():
    return EcoBuildModels(ROOT / "models")


def design(models, form, o=4, g=0.25, d=3):
    x = models.shapes().loc[form].to_dict()
    x.update(Orientation=o, Glazing_Area=g, Glazing_Area_Distribution=d if g else 0)
    return x


# ------------------------------------------------------------------------------------------------ ranges
def test_conformal_quantile_is_finite_sample_corrected():
    s = np.arange(1, 20)                        # n = 19 -> ceil(20 * 0.9) = 18th smallest
    assert conformal_quantile(s, 0.9) == 18
    assert conformal_quantile(np.arange(1, 5), 0.9) == 4      # small n -> largest score


def test_ranges_are_per_building_type_and_held_on_unseen_designs(models):
    r = models.meta["ranges"]
    for t in ("Heating_Load", "Cooling_Load", "Total_Load"):
        assert r["by_type"]["two_story"][t] > r["by_type"]["one_story"][t]
        for k in ("one_story", "two_story"):
            assert r["repeated"]["range"][k][t]["mean"] >= r["level"]        # held on average over random splits
    assert r["repeated"]["n_splits"] >= 20


def test_range_helpers_fall_back_for_old_bundles():
    old = {"half_width": {"Total_Load": 2.0}, "gap": {"Total_Load": 3.0}}
    assert range_half_width(old, 3.5, "Total_Load") == 2.0
    assert pair_gap(old, 3.5, 7.0, "Total_Load") == 3.0
    assert range_half_width(None, 3.5, "Total_Load") is None


def test_pair_gap_depends_on_building_types(models):
    r = models.meta["ranges"]
    one, two, mixed = (pair_gap(r, 3.5, 3.5, "Total_Load"), pair_gap(r, 7, 7, "Total_Load"),
                       pair_gap(r, 3.5, 7, "Total_Load"))
    assert one < mixed < two


def test_business_case_uses_per_row_gaps(models):
    df = pd.DataFrame([design(models, 2), design(models, 3, o=2), design(models, 9)], index=["A", "B", "C"])[FEATURES]
    pred = models.predict(df)
    g = gap_table(models.meta["ranges"], df, "A")
    bc = business_case(df, pred, Assumptions(), "A", g)
    assert g.loc["B", "Total_Load"] < g.loc["C", "Total_Load"]      # one vs one-story is tighter than one vs two
    width_b = bc.loc["B", "savings_high"] - bc.loc["B", "savings_low"]
    width_c = bc.loc["C", "savings_high"] - bc.loc["C", "savings_low"]
    assert width_b < width_c
    assert not bc.loc["A", "within_error"]


# ------------------------------------------------------------------------------------------------ agent rule
@pytest.mark.parametrize("form", [1, 3, 6, 9, 12])
@pytest.mark.parametrize("objective", ["Lowest energy use", "Best value (highest NPV)", "Lowest CO2"])
def test_agent_never_recommends_a_design_worse_than_the_baseline(models, form, objective):
    base = design(models, form, o=2, g=0.10, d=1)
    R = run_agent(models, base, Constraints(objective=objective), Assumptions(), explain=False)
    col, asc = OBJECTIVES[objective]
    b = R.table.loc[R.baseline_key, col]
    for label, r in R.recommendations.items():
        if "Recommended" in label or "Quick win" in label:
            assert (r[col] < b) if asc else (r[col] > b)
    if not R.recommendations:
        assert any("Keep the baseline" in n for n in R.notes)


# ------------------------------------------------------------------------------------------------ backtest
def test_backtest_results_are_stored_and_sensible(models):
    bt = models.meta["backtest"]["by_design"]
    s = bt["suggest"]
    assert s["certain"]["saves_energy"] > s["uncertain"]["saves_energy"]   # the "too close to call" flag means something
    assert s["certain"]["saves_energy"] >= 0.9
    e = bt["agent"]["Lowest energy use"]
    assert e["constraints_met"] == 1.0
    assert e["beats_baseline"] >= 0.9


def test_held_out_models_look_up_predictions(models):
    df = with_ids(models.data())
    fake = pd.DataFrame({"Heating_Load": np.arange(len(df)), "Cooling_Load": 0.0}, index=df.index)
    fake["Total_Load"] = fake.Heating_Load
    h = HeldOutModels(models, fake, df)
    out = h.predict(df.iloc[[5, 100]][FEATURES])
    assert list(out.Heating_Load) == [5, 100]


# ------------------------------------------------------------------------------------------------ AI tools
def test_ai_tools_report_calibrated_ranges(models):
    tools = EcoBuildTools(models, Assumptions())
    one = tools.predict_design({"form": 2, "orientation": "south", "glazing_pct": 25})
    two = tools.predict_design({"form": 9, "orientation": "south", "glazing_pct": 25})
    assert one["range_90pct_plus_minus_kwh_m2"]["total"] < two["range_90pct_plus_minus_kwh_m2"]["total"]
    info = tools.get_model_info()
    assert info["recommendation_backtest"]["clear_suggestions_truly_saving_energy"] >= 0.9
    cmp = tools.compare_designs({"form": 2, "orientation": "south", "glazing_pct": 25},
                                {"form": 2, "orientation": "north", "glazing_pct": 25})
    assert "too close to call" in cmp["verdict"]
