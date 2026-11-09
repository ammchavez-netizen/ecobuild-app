"""EcoBuild AI assistant: natural-language questions answered by an LLM (Groq) that calls EcoBuild's own tools.

Design principle: the language model *never* produces energy or money numbers by itself. Every number comes
from a tool call into the trained models (`ecobuild_core`), the economics module (`ecobuild_econ`) or the
design agent (`ecobuild_agent`). The model plans which tools to call, then explains the results in plain
language, including the uncertainty and the limits of the data.

The Groq API is OpenAI-compatible. Set GROQ_API_KEY in `.streamlit/secrets.toml` or the environment.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import replace

import numpy as np
import pandas as pd

from ecobuild_agent import (OBJECTIVES, Constraints, describe_design, design_key, form_id, recommendation_text,
                            run_agent)
from ecobuild_core import (FEATURE_LABELS, FEATURES, ORIENTATION_LABELS, TARGETS, EcoBuildModels, suggest_changes)
from ecobuild_econ import Assumptions, business_case, fmt_money, fmt_payback, geometry

# Groq retired llama-3.3-70b-versatile from its free/Developer tiers on 2026-08-16 and recommends gpt-oss-120b
# or qwen3.6-27b instead. Models change often, so the app also asks Groq which models the key can use
# (list_chat_models) and falls back to the next model when one is not found.
GROQ_MODELS = ["openai/gpt-oss-120b", "openai/gpt-oss-20b", "qwen/qwen3.6-27b"]
DEFAULT_MODEL = os.environ.get("GROQ_MODEL", GROQ_MODELS[0])
MAX_TOOL_ROUNDS = 6
_NOT_CHAT = ("whisper", "guard", "tts", "orpheus", "playai", "compound", "prompt", "distil")


def list_chat_models(client) -> list[str]:
    """Chat models this API key can use, preferred ones first. Falls back to GROQ_MODELS on any error."""
    try:
        ids = [m.id for m in client.models.list().data]
    except Exception:
        return list(GROQ_MODELS)
    chat = [i for i in ids if not any(x in i.lower() for x in _NOT_CHAT)]
    preferred = [m for m in GROQ_MODELS if m in chat]
    return preferred + sorted(m for m in chat if m not in preferred) or list(GROQ_MODELS)


def _model_not_available(e: Exception) -> bool:
    s = str(e).lower()
    return (type(e).__name__ == "NotFoundError" or "model_not_found" in s or "decommissioned" in s
            or ("model" in s and "does not exist" in s))


def chat_create(client, model: str, **kw):
    """chat.completions.create with model-specific settings and automatic fallback to the next available model.

    Returns (response, model_used). Reasoning models get settings that keep their hidden reasoning short/out of the
    answer text.
    """
    tried, last = [], None
    for m in [model] + [x for x in GROQ_MODELS if x != model]:
        args = dict(kw, model=m)
        if "gpt-oss" in m:
            args["reasoning_effort"] = "low"
        elif "qwen3" in m:
            args["reasoning_format"] = "hidden"
        try:
            return client.chat.completions.create(**args), m
        except Exception as e:                      # only a missing/retired model moves on to the next one
            if not _model_not_available(e):
                raise
            tried.append(m)
            last = e
    raise last


def _strip_reasoning(text: str | None) -> str:
    """Remove any <think>...</think> block a reasoning model leaves in its answer."""
    return re.sub(r"<think>.*?</think>", "", text or "", flags=re.S).strip()

ORIENT_IDS = {v.lower(): k for k, v in ORIENTATION_LABELS.items()}           # "north" -> 2
DIST_IDS = {"none": 0, "uniform": 1, "north": 2, "east": 3, "south": 4, "west": 5}
DIST_NAMES = {v: k for k, v in DIST_IDS.items()}

SYSTEM_PROMPT = """You are EcoBuild Assistant, an expert building-energy advisor for architects working on early designs.

You answer with EcoBuild's tools, which run trained machine-learning models (Gradient Boosting) and a cost model.
Rules:
1. Every energy, cost, CO2, payback or NPV number you state must come from a tool result in this conversation. Never estimate numbers yourself.
2. Designs are described by: form (1-12), orientation (north/east/south/west), glazing_pct (0, 10, 25 or 40, % of floor area) and glazing_distribution (none, uniform, north, east, south, west). Forms 1-6 are one-story, 7-12 are two-story; call list_building_forms if you need details. If the user describes a shape that is not one of the 12 forms, pick the closest form and say so.
3. Loads are kWh per m² of floor area per year (heating and cooling). Lower is better.
4. Always respect uncertainty: if a tool says a difference is within the model's typical error, call it "too close to call", not a win.
5. Explain WHY using the SHAP drivers from the tools (e.g. height, roof area, glazing area). Give one or two concrete recommendations.
6. Money figures are planning estimates based on editable assumptions (energy prices, placeholder unit construction costs, project floor area). Say so briefly when you give dollar figures.
7. Limits to mention when relevant: the model was trained on 768 simulated buildings (12 forms) in one climate; it predicts heating and cooling load only (not daylight, comfort, structure or code compliance).
8. Be concise: short paragraphs or bullet points, plain language, no code, no JSON. Use the user's language.
9. If a question is unrelated to building design or energy, say briefly that you can only help with EcoBuild topics."""


# =============================================================================================== tool schemas
_DESIGN = {
    "type": "object",
    "properties": {
        "form": {"type": "integer", "minimum": 1, "maximum": 12, "description": "Building form 1-12 (1-6 one-story, 7-12 two-story)"},
        "orientation": {"type": "string", "enum": ["north", "east", "south", "west"]},
        "glazing_pct": {"type": "integer", "enum": [0, 10, 25, 40], "description": "Window area as % of floor area"},
        "glazing_distribution": {"type": "string", "enum": list(DIST_IDS), "description": "Where the windows are; 'none' only when glazing_pct is 0"},
    },
    "required": ["form", "orientation", "glazing_pct"],
}


def _fn(name, description, properties=None, required=None):
    return {"type": "function", "function": {"name": name, "description": description,
                                             "parameters": {"type": "object", "properties": properties or {},
                                                            "required": required or []}}}


TOOLS = [
    _fn("list_building_forms", "List the 12 building forms the model knows (stories, compactness, wall/roof areas)."),
    _fn("predict_design", "Predict heating, cooling and total load for one design, with typical error and annual energy cost/CO2.",
        {"design": _DESIGN}, ["design"]),
    _fn("compare_designs", "Compare two designs: loads, whether the difference is real or within model error, the SHAP "
        "drivers of the difference, and the business case of design_b versus design_a.",
        {"design_a": _DESIGN, "design_b": _DESIGN}, ["design_a", "design_b"]),
    _fn("suggest_improvements", "For one design, try every orientation and window option for the SAME building form and "
        "return the best changes with savings and certainty.",
        {"design": _DESIGN, "min_glazing_pct": {"type": "integer", "enum": [0, 10, 25, 40],
                                                "description": "Smallest acceptable window share (default 10)"}},
        ["design"]),
    _fn("run_design_agent", "Search all 768 candidate designs under constraints (stories, fixed orientation, window limits, "
        "construction budget) and recommend the best by an objective, with costs, payback, NPV and CO2.",
        {"baseline": _DESIGN,
         "stories": {"type": "string", "enum": ["any", "one-story", "two-story"]},
         "fixed_orientation": {"type": "string", "enum": ["free", "north", "east", "south", "west"]},
         "min_glazing_pct": {"type": "integer", "enum": [0, 10, 25, 40]},
         "max_glazing_pct": {"type": "integer", "enum": [0, 10, 25, 40]},
         "max_extra_budget": {"type": "number", "description": "Max extra construction $ versus the baseline; omit for no cap; 0 = no more than baseline"},
         "objective": {"type": "string", "enum": ["best_value", "lowest_energy", "lowest_construction_cost", "lowest_co2"]}},
        ["baseline"]),
    _fn("get_feature_importance", "Which design features drive heating and cooling load the most overall (mean |SHAP|)."),
    _fn("get_model_info", "Model accuracy on unseen designs, typical error ranges, data limits and the cost assumptions in use."),
]

OBJECTIVE_IDS = {"best_value": "Best value (highest NPV)", "lowest_energy": "Lowest energy use",
                 "lowest_construction_cost": "Lowest construction cost", "lowest_co2": "Lowest CO2"}


# =============================================================================================== tool runtime
class ToolError(ValueError):
    pass


class EcoBuildTools:
    """Executes tool calls against the models. Returns plain dicts (JSON-serialisable, rounded)."""

    def __init__(self, models: EcoBuildModels, assumptions: Assumptions):
        self.models, self.a = models, assumptions
        self.shapes = models.shapes()
        self.ranges = models.meta.get("ranges")
        self.gap = (self.ranges or {}).get("gap")

    # ---------------------------------------------------------------- helpers
    def design(self, d: dict) -> dict:
        if not isinstance(d, dict):
            raise ToolError("design must be an object with form, orientation, glazing_pct")
        try:
            form = int(d["form"])
        except (KeyError, TypeError, ValueError):
            raise ToolError("design.form must be an integer 1-12")
        if form not in self.shapes.index:
            raise ToolError("design.form must be 1-12 (the model only knows 12 building forms)")
        o = str(d.get("orientation", "south")).lower()
        if o not in ORIENT_IDS:
            raise ToolError("orientation must be north, east, south or west")
        g = int(round(float(d.get("glazing_pct", 25))))
        if g not in (0, 10, 25, 40):
            g = min((0, 10, 25, 40), key=lambda v: abs(v - g))      # snap to the nearest option the model knows
        dist = str(d.get("glazing_distribution", "uniform" if g else "none")).lower()
        if g == 0:
            dist = "none"
        elif dist not in DIST_IDS or dist == "none":
            dist = "uniform"
        out = self.shapes.loc[form].to_dict()
        out.update(Orientation=ORIENT_IDS[o], Glazing_Area=g / 100, Glazing_Area_Distribution=DIST_IDS[dist])
        return out

    def label(self, design: dict) -> str:
        return describe_design({**design, "Form": form_id(self.shapes, design)})

    def _err(self, t):
        return round(self.ranges["half_width"][t], 2) if self.ranges else None

    # ---------------------------------------------------------------- tools
    def list_building_forms(self):
        g = geometry(self.shapes.assign(Glazing_Area=0.0))
        return {"forms": [
            {"form": int(i), "stories": int(g.loc[i, "stories"]), "relative_compactness": r.Relative_Compactness,
             "surface_area_m2": r.Surface_Area, "wall_area_m2": r.Wall_Area, "roof_area_m2": r.Roof_Area,
             "height_m": r.Overall_Height} for i, r in self.shapes.iterrows()],
            "note": "Forms 1-6 are one-story (3.5 m), 7-12 two-story (7 m). Higher compactness = less surface per volume."}

    def predict_design(self, design):
        d = self.design(design)
        df = pd.DataFrame([d])[FEATURES]
        p = self.models.predict(df).iloc[0]
        bc = business_case(df, self.models.predict(df), self.a, df.index[0], self.gap).iloc[0]
        return {"design": self.label(d),
                "heating_kwh_m2": round(p.Heating_Load, 2), "cooling_kwh_m2": round(p.Cooling_Load, 2),
                "total_kwh_m2": round(p.Total_Load, 2),
                "typical_error_kwh_m2": {"heating": self._err("Heating_Load"), "cooling": self._err("Cooling_Load"),
                                         "total": self._err("Total_Load")},
                "annual_energy_cost_usd": round(bc.energy_cost), "annual_co2_tonnes": round(bc.co2_t, 2),
                "envelope_construction_cost_usd": round(bc.construction_cost),
                "project_floor_area_m2": self.a.floor_area_m2}

    def compare_designs(self, design_a, design_b):
        a, b = self.design(design_a), self.design(design_b)
        df = pd.DataFrame([a, b], index=["A", "B"])[FEATURES]
        pred = self.models.predict(df)
        bc = business_case(df, pred, self.a, "A", self.gap)
        diff = float(pred.loc["B", "Total_Load"] - pred.loc["A", "Total_Load"])
        thr = self.gap["Total_Load"] if self.gap else None
        drivers = {}
        if self.models.shap_available():
            tot = pd.Series(0.0, index=FEATURES)
            for t in TARGETS:
                sv, _ = self.models.explain(df, t)
                tot += sv.loc["B"] - sv.loc["A"]
            changed = [f for f in FEATURES if abs(float(df.loc["A", f]) - float(df.loc["B", f])) > 1e-9] or FEATURES
            tot = tot[changed].reindex(tot[changed].abs().sort_values(ascending=False).index)
            drivers = {FEATURE_LABELS[f]: round(float(v), 2) for f, v in tot.items() if abs(v) >= 0.05}
        r = bc.loc["B"]
        verdict = ("too close to call (within the model's typical error)" if thr and abs(diff) <= thr
                   else ("design_a uses less energy" if diff > 0 else "design_b uses less energy"))
        return {
            "design_a": {"design": self.label(a), **{k: round(float(pred.loc["A", c]), 2) for k, c in
                         [("heating", "Heating_Load"), ("cooling", "Cooling_Load"), ("total", "Total_Load")]}},
            "design_b": {"design": self.label(b), **{k: round(float(pred.loc["B", c]), 2) for k, c in
                         [("heating", "Heating_Load"), ("cooling", "Cooling_Load"), ("total", "Total_Load")]}},
            "difference_b_minus_a_kwh_m2": round(diff, 2),
            "percent_difference_vs_a": round(diff / float(pred.loc["A", "Total_Load"]) * 100, 1),
            "typical_error_for_a_difference_kwh_m2": round(thr, 2) if thr else None,
            "verdict": verdict,
            "shap_drivers_of_difference_kwh_m2": drivers,
            "business_case_b_vs_a": {
                "annual_energy_savings_usd": round(r.annual_savings),
                "annual_savings_range_usd": [round(r.savings_low), round(r.savings_high)],
                "extra_construction_cost_usd": round(r.extra_construction),
                "payback": fmt_payback(r.payback_years),
                f"npv_{self.a.horizon_years}yr_usd": round(r.npv),
                "co2_saved_tonnes_per_year": round(r.co2_saved_t_per_yr, 2)},
        }

    def suggest_improvements(self, design, min_glazing_pct=10):
        d = self.design(design)
        shape_row = self.shapes.loc[form_id(self.shapes, d)]
        res = suggest_changes(self.models, d, shape_row, (min_glazing_pct or 0) / 100)
        cur, thr = res["current"], (self.gap or {}).get("Total_Load")
        opts = []
        for name, o in res["options"].items():
            saving = float(cur.Total_Load - o.Total_Load)
            new = {**d, "Orientation": int(o.Orientation), "Glazing_Area": float(o.Glazing_Area),
                   "Glazing_Area_Distribution": int(o.Glazing_Area_Distribution)}
            opts.append({"option": name, "new_design": self.label(new), "total_kwh_m2": round(float(o.Total_Load), 2),
                         "saving_kwh_m2": round(saving, 2), "saving_pct": round(saving / float(cur.Total_Load) * 100, 1),
                         "certain": bool(thr is None or saving > thr)})
        return {"current_design": self.label(d), "current_total_kwh_m2": round(float(cur.Total_Load), 2),
                "options": opts, "note": "Building form is kept fixed; 'certain' = saving larger than the model's typical error."}

    def run_design_agent(self, baseline, stories="any", fixed_orientation="free", min_glazing_pct=10,
                         max_glazing_pct=40, max_extra_budget=None, objective="lowest_energy"):
        b = self.design(baseline)
        c = Constraints(
            stories={"any": "Any", "one-story": "One-story", "two-story": "Two-story"}.get(str(stories).lower(), "Any"),
            orientation=None if str(fixed_orientation).lower() in ("free", "", "none") else ORIENT_IDS.get(str(fixed_orientation).lower()),
            min_glazing=(min_glazing_pct if min_glazing_pct is not None else 10) / 100,
            max_glazing=(max_glazing_pct if max_glazing_pct is not None else 40) / 100,
            max_extra_budget=None if max_extra_budget is None else float(max_extra_budget),
            objective=OBJECTIVE_IDS.get(str(objective), "Lowest energy use"))
        r = run_agent(self.models, b, c, self.a)
        return {"steps": [f"{t}: {d}" for t, d in r.steps],
                "recommendations": {k: recommendation_text(v) for k, v in r.recommendations.items()},
                "robust_to_cost_estimates": r.robust, "notes": r.notes,
                "feasible_designs": int(len(r.table)) if r.table is not None else 0}

    def get_feature_importance(self):
        imp = pd.DataFrame(self.models.meta["shap_importance"])
        return {t.replace("_", " "): {FEATURE_LABELS[f]: round(v, 2) for f, v in
                                      imp[imp.Target == t].sort_values("Mean_abs_SHAP", ascending=False)
                                      [["Feature", "Mean_abs_SHAP"]].values}
                for t in TARGETS}

    def get_model_info(self):
        m = pd.DataFrame(self.models.meta["test_metrics"])
        best = m[m.Model == "Gradient Boosting"]
        return {"accuracy_on_unseen_designs": {r.Target: {"MAE": round(r.MAE, 2), "R2": round(r.R2, 3)} for r in best.itertuples()},
                "typical_error_kwh_m2": self.ranges["half_width"] if self.ranges else None,
                "limits": ["768 simulated buildings from 12 forms, one climate (not real measured buildings)",
                           "predicts heating and cooling load only (not daylight, comfort, structure, cost or code)",
                           "money figures use editable planning assumptions, not quotes"],
                "cost_assumptions": {k: v for k, v in self.a.as_dict().items()}}

    # ---------------------------------------------------------------- dispatcher
    def call(self, name: str, args: dict):
        fn = getattr(self, name, None)
        if name not in {t["function"]["name"] for t in TOOLS} or fn is None:
            return {"error": f"unknown tool {name}"}
        try:
            return _clean(fn(**(args or {})))
        except ToolError as e:
            return {"error": str(e)}
        except TypeError as e:
            return {"error": f"bad arguments for {name}: {e}"}


def _clean(x):
    """Make tool output JSON-safe (numpy types, inf/nan)."""
    if isinstance(x, dict):
        return {str(k): _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if isinstance(x, (np.floating, float)):
        return None if not np.isfinite(x) else round(float(x), 4)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return x


# =============================================================================================== chat loop
def get_api_key(secrets=None) -> str | None:
    try:
        if secrets is not None and "GROQ_API_KEY" in secrets:
            return secrets["GROQ_API_KEY"]
    except Exception:
        pass
    return os.environ.get("GROQ_API_KEY")


def make_client(api_key: str):
    from groq import Groq
    return Groq(api_key=api_key)


class EcoBuildAssistant:
    """Tool-calling chat loop. `client` is a Groq client (or any object with the same chat.completions API)."""

    def __init__(self, client, tools: EcoBuildTools, model: str = DEFAULT_MODEL):
        self.client, self.tools, self.model = client, tools, model
        self.model_used = model

    def _create(self, messages, use_tools=True):
        kw = dict(messages=messages, temperature=0.2, max_completion_tokens=2048)
        if use_tools:
            kw.update(tools=TOOLS, tool_choice="auto")
        resp, self.model_used = chat_create(self.client, self.model_used, **kw)
        return resp

    def reply(self, history: list[dict]) -> tuple[str, list[dict]]:
        """history: [{"role": "user"|"assistant", "content": str}, ...]. Returns (answer, tool_log)."""
        messages = [{"role": "system", "content": SYSTEM_PROMPT}] + [
            {"role": m["role"], "content": m["content"]} for m in history if m["role"] in ("user", "assistant")]
        log = []
        for _ in range(MAX_TOOL_ROUNDS):
            try:
                resp = self._create(messages)
            except Exception as e:                       # malformed tool call, rate limit, network
                if "tool" in str(e).lower() and log:     # retry once, answering from the tool results we have
                    try:
                        resp = self._create(messages, use_tools=False)
                    except Exception as e2:
                        return _friendly_error(e2), log
                else:
                    return _friendly_error(e), log
            msg = resp.choices[0].message
            calls = getattr(msg, "tool_calls", None) or []
            if not calls:
                return _strip_reasoning(msg.content), log
            messages.append({"role": "assistant", "content": msg.content or "",
                             "tool_calls": [{"id": c.id, "type": "function",
                                             "function": {"name": c.function.name, "arguments": c.function.arguments}}
                                            for c in calls]})
            for c in calls:
                try:
                    args = json.loads(c.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}
                result = self.tools.call(c.function.name, args)
                log.append({"tool": c.function.name, "args": args, "result": result})
                messages.append({"role": "tool", "tool_call_id": c.id, "content": json.dumps(result)})
        resp = self._create(messages, use_tools=False)    # out of tool rounds: answer with what we have
        return _strip_reasoning(resp.choices[0].message.content), log


def _friendly_error(e: Exception) -> str:
    s = str(e)
    if "401" in s or "invalid_api_key" in s.lower() or "authentication" in s.lower():
        return "The Groq API key was rejected. Check the key in the sidebar or in Streamlit secrets."
    if _model_not_available(e):
        return ("None of the AI models is available to this Groq key (Groq retires models from time to time). "
                "Pick another model in the sidebar under AI settings.")
    if "429" in s or "rate" in s.lower():
        return "The AI service is rate-limited right now. Wait a few seconds and try again."
    detail = getattr(e, "message", None) or s
    return f"The AI service returned an error, so I could not answer. ({type(e).__name__}: {detail[:200]})"


# =============================================================================================== agent helpers
BRIEF_TOOL = _fn(
    "configure_design_agent", "Translate the architect's brief into design-agent settings.",
    {"baseline_form": {"type": "integer", "minimum": 1, "maximum": 12},
     "baseline_orientation": {"type": "string", "enum": ["north", "east", "south", "west"]},
     "baseline_glazing_pct": {"type": "integer", "enum": [0, 10, 25, 40]},
     "stories": {"type": "string", "enum": ["any", "one-story", "two-story"]},
     "fixed_orientation": {"type": "string", "enum": ["free", "north", "east", "south", "west"]},
     "min_glazing_pct": {"type": "integer", "enum": [0, 10, 25, 40]},
     "max_glazing_pct": {"type": "integer", "enum": [0, 10, 25, 40]},
     "max_extra_budget": {"type": ["number", "null"], "description": "Max extra construction $ vs baseline; null = no cap"},
     "objective": {"type": "string", "enum": list(OBJECTIVE_IDS)},
     "floor_area_m2": {"type": ["number", "null"]},
     "carbon_price": {"type": ["number", "null"], "description": "$ per tonne CO2 if the user wants carbon valued"}},
    ["objective"])


def parse_brief(client, model: str, brief: str) -> dict:
    """Use the LLM to turn a plain-English brief into agent settings (validated by the caller)."""
    resp, _ = chat_create(
        client, model, temperature=0, max_completion_tokens=1024, tools=[BRIEF_TOOL],
        tool_choice={"type": "function", "function": {"name": "configure_design_agent"}},
        messages=[{"role": "system", "content":
                   "Extract design-agent settings from the brief. Forms 1-6 are one-story, 7-12 two-story. "
                   "Only set fields the brief supports; daylight requirements imply min_glazing_pct >= 10. "
                   "'Don't spend more than the current design' means max_extra_budget = 0."},
                  {"role": "user", "content": brief}])
    calls = resp.choices[0].message.tool_calls or []
    return json.loads(calls[0].function.arguments) if calls else {}


def apply_brief(settings: dict, baseline: dict, constraints: Constraints, assumptions: Assumptions, shapes: pd.DataFrame):
    """Validate LLM-extracted settings and merge them into the current baseline/constraints/assumptions."""
    s = settings or {}
    b = dict(baseline)
    if isinstance(s.get("baseline_form"), (int, float)) and int(s["baseline_form"]) in shapes.index:
        b.update(shapes.loc[int(s["baseline_form"])].to_dict())
    if str(s.get("baseline_orientation", "")).lower() in ORIENT_IDS:
        b["Orientation"] = ORIENT_IDS[s["baseline_orientation"].lower()]
    if s.get("baseline_glazing_pct") in (0, 10, 25, 40):
        b["Glazing_Area"] = s["baseline_glazing_pct"] / 100
        b["Glazing_Area_Distribution"] = 0 if b["Glazing_Area"] == 0 else (b.get("Glazing_Area_Distribution") or 1)
    c = replace(constraints)
    if s.get("stories") in ("any", "one-story", "two-story"):
        c.stories = {"any": "Any", "one-story": "One-story", "two-story": "Two-story"}[s["stories"]]
    if "fixed_orientation" in s:
        c.orientation = ORIENT_IDS.get(str(s["fixed_orientation"]).lower())
    if s.get("min_glazing_pct") in (0, 10, 25, 40):
        c.min_glazing = s["min_glazing_pct"] / 100
    if s.get("max_glazing_pct") in (0, 10, 25, 40):
        c.max_glazing = s["max_glazing_pct"] / 100
    if c.max_glazing < c.min_glazing:
        c.max_glazing = c.min_glazing
    if "max_extra_budget" in s:
        c.max_extra_budget = None if s["max_extra_budget"] is None else float(s["max_extra_budget"])
    if s.get("objective") in OBJECTIVE_IDS:
        c.objective = OBJECTIVE_IDS[s["objective"]]
    a = replace(assumptions)
    if isinstance(s.get("floor_area_m2"), (int, float)) and s["floor_area_m2"] > 0:
        a.floor_area_m2 = float(s["floor_area_m2"])
    if isinstance(s.get("carbon_price"), (int, float)) and s["carbon_price"] >= 0:
        a.carbon_price = float(s["carbon_price"])
    return b, c, a


def summarize_agent(client, model: str, brief: str, result_dict: dict) -> str:
    """LLM executive summary of the agent's results — numbers only from `result_dict`."""
    resp, _ = chat_create(
        client, model, temperature=0.2, max_completion_tokens=1500,
        messages=[{"role": "system", "content":
                   "You write a short decision memo (max 150 words) for an architect and their client. Use ONLY numbers "
                   "from the JSON. Lead with the recommendation, then why (drivers), the money (savings, extra cost, "
                   "payback, NPV), and one caution (uncertainty, cost assumptions or data limits). Plain language, bullets ok."},
                  {"role": "user", "content": f"Brief: {brief or '(none)'}\n\nAgent results:\n{json.dumps(_clean(result_dict))}"}])
    return _strip_reasoning(resp.choices[0].message.content)
