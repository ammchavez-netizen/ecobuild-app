"""Tests for the decision-support features: economics, design agent, assistant tools and the LLM chat loop.

The LLM is replaced by a scripted fake client, so the tests run offline and without an API key.
Run:  pytest -q
"""
import json
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ecobuild_agent import Constraints, pareto_front, run_agent  # noqa: E402
from ecobuild_ai import (TOOLS, EcoBuildAssistant, EcoBuildTools, apply_brief, parse_brief,  # noqa: E402
                         summarize_agent)
from ecobuild_core import FEATURES, EcoBuildModels  # noqa: E402
from ecobuild_econ import Assumptions, business_case, envelope_cost, geometry, present_value_factor  # noqa: E402


@pytest.fixture(scope="module")
def models():
    return EcoBuildModels(ROOT / "models")


def design(models, form, o=4, g=0.25, d=3):
    x = models.shapes().loc[form].to_dict()
    x.update(Orientation=o, Glazing_Area=g, Glazing_Area_Distribution=d if g else 0)
    return x


# ------------------------------------------------------------------------------------------------ economics
def test_geometry_matches_dataset_identity(models):
    s = models.shapes()
    assert np.allclose(s.Surface_Area, s.Wall_Area + 2 * s.Roof_Area)       # footprint = roof area
    g = geometry(s.assign(Glazing_Area=0.4))
    assert (g.window_per_m2 == 0.4).all()
    assert set(g.stories) == {1, 2}


def test_present_value_factor_no_growth_no_discount():
    a = Assumptions(horizon_years=10, discount_rate=0.0, energy_escalation=0.0)
    assert present_value_factor(a) == pytest.approx(10.0)


def test_business_case_baseline_is_zero_and_signs(models):
    df = pd.DataFrame([design(models, 6), design(models, 10)], index=["A", "B"])[FEATURES]
    pred = models.predict(df)
    bc = business_case(df, pred, Assumptions(), "A", models.meta["ranges"]["gap"])
    assert bc.loc["A", ["extra_construction", "annual_savings", "npv"]].abs().max() == pytest.approx(0)
    assert bc.loc["B", "annual_savings"] < 0          # two-story uses more energy
    assert bc.loc["B", "savings_low"] < bc.loc["B", "annual_savings"] < bc.loc["B", "savings_high"]
    assert bc.loc["B", "co2_saved_t_per_yr"] < 0


def test_heat_pump_and_carbon_price_change_costs(models):
    df = pd.DataFrame([design(models, 6)])[FEATURES]
    pred = models.predict(df)
    gas = business_case(df, pred, Assumptions(), df.index[0]).energy_cost.iloc[0]
    co2 = business_case(df, pred, Assumptions(carbon_price=100), df.index[0]).energy_cost.iloc[0]
    assert co2 > gas
    hp = Assumptions(heating_system="Heat pump")
    assert hp.heat_price_per_kwh == pytest.approx(hp.elec_price / hp.heat_pump_cop)


def test_more_windows_cost_more_to_build(models):
    df = pd.DataFrame([design(models, 6, g=0.10), design(models, 6, g=0.40)])[FEATURES]
    c = envelope_cost(df, Assumptions())
    assert c.iloc[1] > c.iloc[0]


# ------------------------------------------------------------------------------------------------ agent
def test_pareto_front_is_non_dominated():
    df = pd.DataFrame({"construction_cost": [1, 2, 3, 2.5], "energy_cost": [5, 3, 1, 4]}, index=list("abcd"))
    assert list(pareto_front(df).index) == ["a", "b", "c"]


def test_agent_respects_constraints(models):
    base = design(models, 6, g=0.40)
    c = Constraints(stories="One-story", orientation=4, min_glazing=0.25, max_glazing=0.40, max_extra_budget=0.0)
    R = run_agent(models, base, c, Assumptions(), explain=False)
    feas = R.table.drop(index=R.baseline_key)
    assert (feas.Overall_Height < 7).all()
    assert (feas.Orientation == 4).all()
    assert feas.Glazing_Area.between(0.25, 0.40).all()
    assert (feas.extra_construction <= 1e-6).all()
    assert R.recommendations and len(R.steps) >= 8


def test_agent_default_finds_cheaper_and_more_efficient_design(models):
    R = run_agent(models, design(models, 6, g=0.40), Constraints(), Assumptions())
    top = next(iter(R.recommendations.values()))
    assert top.Total_Load < R.table.loc[R.baseline_key, "Total_Load"]
    assert top.extra_construction <= 0
    assert R.robust is not None


# ------------------------------------------------------------------------------------------------ tools
def test_every_declared_tool_runs(models):
    T = EcoBuildTools(models, Assumptions())
    d = {"form": 6, "orientation": "south", "glazing_pct": 40, "glazing_distribution": "east"}
    args = {"list_building_forms": {}, "predict_design": {"design": d},
            "compare_designs": {"design_a": d, "design_b": {**d, "form": 10}},
            "suggest_improvements": {"design": d}, "run_design_agent": {"baseline": d},
            "get_feature_importance": {}, "get_model_info": {}}
    for t in TOOLS:
        name = t["function"]["name"]
        out = T.call(name, args[name])
        assert "error" not in out, (name, out)
        json.dumps(out)                                   # must be JSON-serialisable


def test_tools_reject_bad_input_gracefully(models):
    T = EcoBuildTools(models, Assumptions())
    assert "error" in T.call("predict_design", {"design": {"form": 99, "orientation": "south", "glazing_pct": 25}})
    assert "error" in T.call("predict_design", {"design": {"form": 6, "orientation": "up", "glazing_pct": 25}})
    assert "error" in T.call("not_a_tool", {})
    # off-grid glazing snaps to the nearest option the model knows
    assert "error" not in T.call("predict_design", {"design": {"form": 6, "orientation": "east", "glazing_pct": 30}})


def test_compare_matches_app_numbers(models):
    T = EcoBuildTools(models, Assumptions())
    a = {"form": 6, "orientation": "south", "glazing_pct": 25, "glazing_distribution": "east"}
    out = T.call("compare_designs", {"design_a": a, "design_b": {**a, "form": 10}})
    assert out["design_a"]["heating"] == pytest.approx(12.2, abs=0.05)
    assert out["design_b"]["heating"] == pytest.approx(29.2, abs=0.05)
    assert out["verdict"] == "design_a uses less energy"
    assert list(out["shap_drivers_of_difference_kwh_m2"])[0] in ("Overall height", "Roof area")


# ------------------------------------------------------------------------------------------------ chat loop
class FakeClient:
    """Mimics groq.Groq().chat.completions.create with a scripted list of responses; records requests."""

    def __init__(self, script):
        self.script, self.requests = list(script), []
        self.chat = NS(completions=NS(create=self._create))

    def _create(self, **kw):
        self.requests.append(kw)
        return self.script.pop(0)


def tool_turn(*calls):
    tcs = [NS(id=f"call_{i}", type="function", function=NS(name=n, arguments=json.dumps(a)))
           for i, (n, a) in enumerate(calls)]
    return NS(choices=[NS(message=NS(content=None, tool_calls=tcs))])


def text_turn(text):
    return NS(choices=[NS(message=NS(content=text, tool_calls=None))])


def test_chat_loop_runs_tools_and_feeds_results_back(models):
    d = {"form": 6, "orientation": "south", "glazing_pct": 40, "glazing_distribution": "east"}
    client = FakeClient([tool_turn(("suggest_improvements", {"design": d})), text_turn("Cut the windows to 10%.")])
    bot = EcoBuildAssistant(client, EcoBuildTools(models, Assumptions()), "test-model")
    answer, log = bot.reply([{"role": "user", "content": "How do I improve form 6?"}])
    assert answer == "Cut the windows to 10%."
    assert log[0]["tool"] == "suggest_improvements" and log[0]["result"]["options"]
    second = client.requests[1]["messages"]
    assert second[0]["role"] == "system"
    assert any(m["role"] == "tool" and "Change the windows" in m["content"] for m in second)
    assert client.requests[0]["tools"] == TOOLS


def test_chat_loop_handles_api_errors(models):
    class Boom:
        chat = NS(completions=NS(create=lambda **kw: (_ for _ in ()).throw(RuntimeError("Error code: 401 invalid_api_key"))))
    answer, log = EcoBuildAssistant(Boom(), EcoBuildTools(models, Assumptions()), "m").reply(
        [{"role": "user", "content": "hi"}])
    assert "key" in answer.lower() and log == []


def test_brief_parsing_and_validation(models):
    settings = {"baseline_form": 3, "baseline_orientation": "south", "baseline_glazing_pct": 40,
                "stories": "one-story", "fixed_orientation": "south", "min_glazing_pct": 25,
                "max_glazing_pct": 10, "max_extra_budget": 0, "objective": "best_value", "floor_area_m2": 1500,
                "carbon_price": 50}
    client = FakeClient([tool_turn(("configure_design_agent", settings))])
    parsed = parse_brief(client, "m", "1,500 m² one-story building facing south ...")
    assert parsed == settings
    b, c, a = apply_brief(parsed, design(models, 6), Constraints(), Assumptions(), models.shapes())
    assert b["Roof_Area"] == models.shapes().loc[3, "Roof_Area"] and b["Glazing_Area"] == 0.40
    assert c.stories == "One-story" and c.orientation == 4 and c.objective == "Best value (highest NPV)"
    assert c.max_glazing >= c.min_glazing                  # inconsistent limits are repaired
    assert a.floor_area_m2 == 1500 and a.carbon_price == 50


def test_summary_only_sends_agent_numbers(models):
    client = FakeClient([text_turn("Memo.")])
    assert summarize_agent(client, "m", "brief", {"recommendations": {"x": "y"}, "v": np.float64(1.5)}) == "Memo."
    assert "1.5" in client.requests[0]["messages"][1]["content"]


# ------------------------------------------------------------------------------------------------ model handling
class NotFoundError(Exception):
    """Stand-in for groq.NotFoundError (the code matches on the class name)."""


def test_retired_model_falls_back_to_next(models):
    from ecobuild_ai import GROQ_MODELS, chat_create

    calls = []

    def create(**kw):
        calls.append(kw)
        if kw["model"] == "llama-3.3-70b-versatile":
            raise NotFoundError("Error code: 404 - model_not_found")
        return text_turn(f"answered by {kw['model']}")

    client = NS(chat=NS(completions=NS(create=create)))
    resp, used = chat_create(client, "llama-3.3-70b-versatile", messages=[])
    assert used == GROQ_MODELS[0] and "answered by" in resp.choices[0].message.content
    assert calls[1].get("reasoning_effort") == "low"          # gpt-oss gets short reasoning


def test_assistant_reports_when_no_model_is_available(models):
    def create(**kw):
        raise NotFoundError("Error code: 404 - The model does not exist or you do not have access to it.")
    bot = EcoBuildAssistant(NS(chat=NS(completions=NS(create=create))), EcoBuildTools(models, Assumptions()), "x")
    answer, _ = bot.reply([{"role": "user", "content": "hi"}])
    assert "Pick another model" in answer


def test_model_list_filters_non_chat_models():
    from ecobuild_ai import list_chat_models
    ids = ["whisper-large-v3", "openai/gpt-oss-20b", "meta-llama/llama-prompt-guard-2-22m", "qwen/qwen3.6-27b",
           "groq/compound", "openai/gpt-oss-120b", "some/new-model"]
    client = NS(models=NS(list=lambda: NS(data=[NS(id=i) for i in ids])))
    assert list_chat_models(client) == ["openai/gpt-oss-120b", "openai/gpt-oss-20b", "qwen/qwen3.6-27b", "some/new-model"]
    broken = NS(models=NS(list=lambda: (_ for _ in ()).throw(RuntimeError("offline"))))
    assert list_chat_models(broken)[0] == "openai/gpt-oss-120b"


def test_reasoning_tags_are_removed():
    from ecobuild_ai import _strip_reasoning
    assert _strip_reasoning("<think>internal</think>\nFinal answer.") == "Final answer."
