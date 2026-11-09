"""EcoBuild design agent: evaluate every valid design against an architect's constraints and recommend options.

The agent is deterministic: it plans and executes the same steps every time, logs each step, and only uses
numbers produced by the trained models and the economics module. A language model (see ecobuild_ai.py) can
turn a plain-English brief into `Constraints` and write a summary, but it never invents the numbers.

Steps
1. Read the brief (baseline design + constraints)
2. Generate every valid design: 12 building forms × 64 orientation/window options = 768 candidates
3. Apply the constraints (stories, forms, fixed orientation, window limits, budget)
4. Predict heating and cooling load for every candidate
5. Estimate construction cost, energy cost, CO2, payback and NPV against the baseline
6. Keep the Pareto frontier (nothing else is both cheaper to build and cheaper to run)
7. Rank by the chosen objective and flag savings that are within the model's typical error
8. Stress-test the pick against ±25% construction costs
9. Explain the pick versus the baseline with SHAP
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace

import numpy as np
import pandas as pd

from ecobuild_core import (FEATURE_LABELS, FEATURES, GLAZING_AREA_LABELS, GLAZING_DIST_LABELS, ORIENTATION_LABELS,
                           TARGETS, EcoBuildModels, design_variants)
from ecobuild_econ import Assumptions, business_case, fmt_money, fmt_payback

OBJECTIVES = {
    "Best value (highest NPV)": ("npv", False),
    "Lowest energy use": ("Total_Load", True),
    "Lowest construction cost": ("construction_cost", True),
    "Lowest CO2": ("co2_t", True),
}


@dataclass
class Constraints:
    stories: str = "Any"                    # "Any" | "One-story" | "Two-story"
    forms: list[int] | None = None          # allowed building-form IDs (1-12); None = all
    orientation: int | None = None          # fixed by the site (2=N, 3=E, 4=S, 5=W); None = free
    min_glazing: float = 0.10               # daylight/view requirement (share of floor area)
    max_glazing: float = 0.40
    max_extra_budget: float | None = 0.0    # max extra construction $ over the baseline; None = no cap
    objective: str = "Lowest energy use"

    def as_dict(self) -> dict:
        return asdict(self)

    def describe(self) -> str:
        parts = [f"stories: {self.stories}"]
        if self.forms:
            parts.append(f"forms {', '.join(map(str, self.forms))}")
        parts.append("orientation: " + (ORIENTATION_LABELS[self.orientation] + " (fixed by site)"
                                        if self.orientation else "free"))
        parts.append(f"glazing {self.min_glazing:.0%}–{self.max_glazing:.0%} of floor area")
        if self.max_extra_budget is not None:
            parts.append(f"extra construction budget ≤ {fmt_money(self.max_extra_budget)}")
        parts.append(f"objective: {self.objective}")
        return "; ".join(parts)


@dataclass
class AgentResult:
    steps: list = field(default_factory=list)          # [(title, detail)]
    table: pd.DataFrame | None = None                   # every feasible candidate with its business case
    pareto: pd.DataFrame | None = None
    recommendations: dict = field(default_factory=dict)  # "label · label" -> row (Series); same design merged
    baseline_key: str = ""
    explanation: list = field(default_factory=list)     # [(feature label, kWh/m² contribution)]
    robust: bool | None = None
    notes: list = field(default_factory=list)


def design_key(row) -> str:
    o = ORIENTATION_LABELS[int(row["Orientation"])]
    g = row["Glazing_Area"]
    w = "no windows" if g == 0 else f"{g:.0%} {GLAZING_DIST_LABELS[int(row['Glazing_Area_Distribution'])].lower()}"
    return f"Form {int(row['Form'])} · {o} · {w}"


def describe_design(row, shapes: pd.DataFrame | None = None) -> str:
    stories = "two-story" if row["Overall_Height"] >= 7 else "one-story"
    g = row["Glazing_Area"]
    w = ("no windows" if g == 0 else
         f"{GLAZING_AREA_LABELS[g]} glazing, {GLAZING_DIST_LABELS[int(row['Glazing_Area_Distribution'])].lower()}")
    form = f"form {int(row['Form'])}, " if "Form" in row else ""
    return f"{form}{stories}, faces {ORIENTATION_LABELS[int(row['Orientation'])]}, {w}"


def form_id(shapes: pd.DataFrame, design: dict) -> int:
    m = (shapes == pd.Series({c: design[c] for c in shapes.columns})).all(axis=1)
    return int(shapes.index[m][0])


def all_candidates(models: EcoBuildModels) -> pd.DataFrame:
    shapes = models.shapes()
    frames = []
    for i, row in shapes.iterrows():
        v = design_variants(row)
        v.insert(0, "Form", i)
        frames.append(v)
    c = pd.concat(frames, ignore_index=True)
    c.index = [design_key(r) for _, r in c.iterrows()]
    return c


def pareto_front(df: pd.DataFrame, x: str = "construction_cost", y: str = "energy_cost") -> pd.DataFrame:
    """Designs for which no other design is cheaper to build AND cheaper to run."""
    d = df.sort_values([x, y])
    keep, best_y = [], np.inf
    for k, r in d.iterrows():
        if r[y] < best_y - 1e-9:
            keep.append(k)
            best_y = r[y]
    return df.loc[keep]


def explain_vs_baseline(models: EcoBuildModels, base: pd.Series, pick: pd.Series, top: int = 3):
    """Top SHAP contributions (heating + cooling) to the load difference pick − baseline."""
    if not models.shap_available():
        return []
    two = pd.DataFrame([base[FEATURES], pick[FEATURES]], index=["base", "pick"]).astype(float)
    total = pd.Series(0.0, index=FEATURES)
    for t in TARGETS:
        sv, _ = models.explain(two, t)
        total += sv.loc["pick"] - sv.loc["base"]
    # Attribute the difference only to inputs that actually changed; SHAP gives unchanged inputs small
    # interaction shares that would confuse a reader ("height +0.4" when the height did not change).
    changed = [f for f in FEATURES if abs(float(base[f]) - float(pick[f])) > 1e-9]
    total = total[changed] if changed else total
    total = total.reindex(total.abs().sort_values(ascending=False).index)
    return [(FEATURE_LABELS[f], float(v)) for f, v in total.head(top).items() if abs(v) >= 0.05]


def run_agent(models: EcoBuildModels, baseline: dict, constraints: Constraints, assumptions: Assumptions,
              explain: bool = True) -> AgentResult:
    R = AgentResult()
    gap = (models.meta.get("ranges") or {}).get("gap")
    shapes = models.shapes()
    base = {**baseline, "Form": form_id(shapes, baseline)}
    base_key = design_key(base)
    R.baseline_key = base_key

    # 1. brief
    R.steps.append(("Read the brief",
                    f"Baseline: {describe_design(base)}. Constraints: {constraints.describe()}. "
                    f"Project floor area {assumptions.floor_area_m2:,.0f} m²."))

    # 2. candidates
    cand = all_candidates(models)
    R.steps.append(("Generated candidates",
                    f"{len(cand)} designs = 12 building forms × 4 orientations × 16 window options. "
                    "The model has only seen these 12 forms, so it does not invent new shapes."))

    # 3. constraints
    n0, removed = len(cand), []
    mask = pd.Series(True, index=cand.index)
    if constraints.stories != "Any":
        two = constraints.stories == "Two-story"
        m = (cand.Overall_Height >= 7) == two
        removed.append(f"{(~m & mask).sum()} not {constraints.stories.lower()}"); mask &= m
    if constraints.forms:
        m = cand.Form.isin(constraints.forms)
        removed.append(f"{(~m & mask).sum()} outside forms {constraints.forms}"); mask &= m
    if constraints.orientation:
        m = cand.Orientation == constraints.orientation
        removed.append(f"{(~m & mask).sum()} not facing {ORIENTATION_LABELS[constraints.orientation]}"); mask &= m
    m = (cand.Glazing_Area >= constraints.min_glazing - 1e-9) & (cand.Glazing_Area <= constraints.max_glazing + 1e-9)
    removed.append(f"{(~m & mask).sum()} outside the window limits"); mask &= m
    mask.loc[base_key] = True                       # always keep the baseline for comparison
    cand = cand[mask]

    # 4. predict
    pred = models.predict(cand[FEATURES])
    ranges = models.meta.get("ranges")
    err = (f" Typical error ±{ranges['half_width']['Heating_Load']:.1f} (heating) and "
           f"±{ranges['half_width']['Cooling_Load']:.1f} (cooling) kWh/m²." if ranges else "")
    R.steps.append(("Predicted energy loads",
                    f"Ran the heating and cooling models on {len(cand)} candidates.{err}"))

    # 5. business case
    bc = business_case(cand[FEATURES], pred, assumptions, base_key, gap)
    table = pd.concat([cand[["Form"] + FEATURES], bc], axis=1)
    if constraints.max_extra_budget is not None:
        over = table.extra_construction > constraints.max_extra_budget
        over.loc[base_key] = False
        removed.append(f"{int(over.sum())} over the extra budget")
        table = table[~over]
    R.steps.append(("Applied constraints",
                    f"{n0} → {len(table)} feasible designs (removed " + ", ".join(removed) + ")."))
    R.steps.append(("Estimated costs and returns",
                    f"Envelope construction cost, annual energy cost, CO2, payback and {assumptions.horizon_years}-year "
                    f"NPV at {assumptions.discount_rate:.0%} discount rate, compared with the baseline "
                    f"(baseline energy cost {fmt_money(table.loc[base_key, 'energy_cost'])}/yr)."))
    R.table = table

    # 6. pareto
    R.pareto = pareto_front(table)
    R.steps.append(("Found the trade-off frontier",
                    f"{len(R.pareto)} designs are on the Pareto frontier: no other feasible design is both cheaper "
                    "to build and cheaper to run."))

    # 7. rank
    col, asc = OBJECTIVES[constraints.objective]
    others = table.drop(index=base_key)
    if others.empty:
        R.steps.append(("Ranked options", "No feasible alternative to the baseline."))
        return R
    ranked = others.sort_values(col, ascending=asc)
    confident = ranked[~ranked.within_error]
    pick_key = (confident if len(confident) else ranked).index[0]
    picks = [("Recommended", pick_key),
             ("Most energy-efficient", others.sort_values("Total_Load").index[0]),
             ("Lowest construction cost", others.sort_values("construction_cost").index[0])]
    same_form = others[others.Form == base["Form"]]
    if len(same_form):
        picks.append(("Quick win (keep the form)", same_form.sort_values(col, ascending=asc).index[0]))
    merged: dict = {}
    for label, key in picks:                       # one card per design, with every label it earned
        merged.setdefault(key, []).append(label)
    R.recommendations = {" · ".join(labels): table.loc[key] for key, labels in merged.items()}
    n_unc = int(ranked.head(10).within_error.sum())
    R.steps.append(("Ranked options",
                    f"Sorted by '{constraints.objective}'. Top pick: {describe_design(table.loc[pick_key])}."
                    + (f" {n_unc} of the top 10 options save less than the model's typical error and are marked uncertain."
                       if n_unc else "")))

    # 8. sensitivity
    picks = []
    for f in (0.75, 1.25):
        a2 = replace(assumptions, wall_cost=assumptions.wall_cost * f, window_cost=assumptions.window_cost * f,
                     roof_cost=assumptions.roof_cost * f, foundation_cost=assumptions.foundation_cost * f,
                     upper_floor_cost=assumptions.upper_floor_cost * f)
        bc2 = business_case(cand.loc[table.index, FEATURES], pred.loc[table.index], a2, base_key, gap).drop(index=base_key)
        bc2 = bc2[~bc2.within_error] if (~bc2.within_error).any() else bc2
        picks.append(bc2.sort_values(col, ascending=asc).index[0])
    R.robust = all(p == pick_key for p in picks)
    R.steps.append(("Stress-tested the pick",
                    "The same design stays on top when construction costs are 25% lower or higher."
                    if R.robust else
                    "The top pick changes when construction costs move ±25%, so the choice depends on cost estimates. "
                    "Get real quotes before deciding."))

    # 9. explain
    if explain:
        R.explanation = explain_vs_baseline(models, table.loc[base_key], table.loc[pick_key])
        if R.explanation:
            R.steps.append(("Explained the pick with SHAP",
                            "Largest drivers of the load difference vs. baseline: "
                            + ", ".join(f"{f} ({v:+.1f} kWh/m²)" for f, v in R.explanation) + "."))
    p = table.loc[pick_key]
    if p.annual_savings < 0 and constraints.objective.startswith("Best value"):
        R.notes.append("The best-value design uses more energy than the baseline: its lower construction cost outweighs "
                       "higher energy bills under these assumptions. Add a carbon price or a higher energy price to see "
                       "how the trade-off changes.")
    return R


def recommendation_text(row: pd.Series) -> str:
    """One-line business summary of a recommended design versus the baseline."""
    sav = row.annual_savings
    txt = (f"{describe_design(row)}: total load {row.Total_Load:.1f} kWh/m², "
           f"energy {'saves' if sav >= 0 else 'costs'} {fmt_money(abs(sav))}/yr "
           f"({fmt_money(row.savings_low)} to {fmt_money(row.savings_high)}), "
           f"construction {'+' if row.extra_construction >= 0 else '−'}{fmt_money(abs(row.extra_construction))}, "
           f"NPV {fmt_money(row.npv)}, payback {fmt_payback(row.payback_years)}, "
           f"CO2 {row.co2_saved_t_per_yr:+.1f} t/yr saved")
    if row.within_error:
        txt += " (energy difference within model error — treat as a tie)"
    return txt
