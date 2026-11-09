"""EcoBuild — Building Energy Efficiency Predictor (Streamlit prototype).

Run locally:   streamlit run app.py
"""
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from ecobuild_agent import OBJECTIVES, Constraints, describe_design, recommendation_text, run_agent
from ecobuild_ai import (GROQ_MODELS, EcoBuildAssistant, EcoBuildTools, apply_brief, get_api_key, list_chat_models,
                         make_client, parse_brief, summarize_agent)
from ecobuild_core import (FEATURE_LABELS, FEATURES, GLAZING_AREA_LABELS, GLAZING_DIST_LABELS,
                           ORIENTATION_LABELS, TARGETS, EcoBuildModels, suggest_changes)
from ecobuild_econ import Assumptions, business_case, fmt_money, fmt_payback

st.set_page_config(page_title="EcoBuild", page_icon="🏢", layout="wide")

MODEL_DIR = Path(__file__).parent / "models"
COLOR_A, COLOR_B = "#2a78d6", "#eb6834"          # Design A / Design B (used everywhere)
COLOR_UP, COLOR_DOWN = "#e34948", "#1baf7a"      # raises load / lowers load
TARGET_NAMES = {"Heating_Load": "Heating", "Cooling_Load": "Cooling", "Total_Load": "Total"}


@st.cache_data(ttl=3600, show_spinner=False)
def available_groq_models(api_key: str) -> list[str]:
    return list_chat_models(make_client(api_key))


def esc(text: str) -> str:
    """Escape $ so Streamlit markdown does not render dollar amounts as LaTeX math."""
    return str(text).replace("$", "\\$")


@st.cache_resource(show_spinner="Loading EcoBuild models…")
def get_models() -> EcoBuildModels:
    return EcoBuildModels(MODEL_DIR)


models = get_models()
meta = models.meta
shapes = models.shapes()
RANGES = meta.get("ranges")   # prediction ranges; bundles exported before Update 4 don't have them


def half_width(target: str):
    return RANGES["half_width"][target] if RANGES else None


def gap_threshold(target: str):
    return RANGES["gap"][target] if RANGES else None


def shape_label(i: int) -> str:
    r = shapes.loc[i]
    stories = "Two-story" if r.Overall_Height >= 7 else "One-story"
    return (f"Shape {i} · {stories} · compactness {r.Relative_Compactness:.2f} · "
            f"surface {r.Surface_Area:g} m² · roof {r.Roof_Area:g} m²")


# ============================================================================ sidebar
with st.sidebar:
    st.markdown("## 🏢 EcoBuild")
    st.caption("Building Energy Efficiency Predictor")
    st.markdown(
        "Predicts **heating** and **cooling loads** (kWh/m²) from early design choices, "
        "so architects can compare options before construction."
    )
    st.divider()
    st.markdown("**Models**")
    test = pd.DataFrame(meta["test_metrics"])
    for t in TARGETS:
        name = meta["best_model"][t]
        row = test[(test.Target == t.replace("_", " ")) & (test.Model == name)].iloc[0]
        st.markdown(f"- {TARGET_NAMES[t]}: {name} — R² {row.R2:.3f}, MAE {row.MAE:.2f}")
    retrained = [t for t, s in models.status.items() if s != "loaded"]
    if retrained:
        st.info("Saved models were built with a different library version, so they were "
                "retrained from the bundled data with the same settings.", icon="ℹ️")
    st.divider()
    with st.expander("💲 Cost & ROI assumptions", expanded=False):
        st.caption("Used for every dollar figure. Defaults are rough U.S. averages or placeholders. "
                   "Replace them with your project's numbers.")
        _d = Assumptions()
        floor_area = st.number_input("Project floor area (m²)", 50.0, 100000.0, _d.floor_area_m2, step=50.0)
        heating_system = st.selectbox("Heating system", ["Gas furnace", "Heat pump"])
        c1, c2 = st.columns(2)
        elec_price = c1.number_input("Electricity ($/kWh)", 0.01, 1.0, _d.elec_price, step=0.01, format="%.3f",
                                     help="Default: EIA Short-Term Energy Outlook (Oct 2026), U.S. commercial 2026.")
        gas_price = c2.number_input("Gas ($/Mcf)", 1.0, 60.0, _d.gas_price_mcf, step=0.5,
                                    help="Natural gas, $ per thousand cubic feet. Default: EIA STEO (Oct 2026), commercial 2026.")
        st.markdown("**Envelope unit costs ($/m², placeholders)**")
        c1, c2 = st.columns(2)
        wall_cost = c1.number_input("Wall", 0.0, 5000.0, _d.wall_cost, step=10.0)
        window_cost = c2.number_input("Window", 0.0, 5000.0, _d.window_cost, step=10.0)
        roof_cost = c1.number_input("Roof", 0.0, 5000.0, _d.roof_cost, step=10.0)
        foundation_cost = c2.number_input("Foundation", 0.0, 5000.0, _d.foundation_cost, step=10.0)
        upper_floor_cost = c1.number_input("Upper floor", 0.0, 5000.0, _d.upper_floor_cost, step=10.0,
                                           help="Structure of floors above ground (slab, stairs), per m² of upper floor.")
        carbon_price = c2.number_input("Carbon ($/t CO₂)", 0.0, 1000.0, _d.carbon_price, step=10.0,
                                       help="Optional. Adds a cost per tonne of CO₂ to the energy bill.")
        c1, c2 = st.columns(2)
        horizon = c1.number_input("Horizon (years)", 1, 60, _d.horizon_years)
        discount = c2.number_input("Discount rate", 0.0, 0.3, _d.discount_rate, step=0.01, format="%.2f")
    ASSUMPTIONS = Assumptions(floor_area_m2=floor_area, elec_price=elec_price, gas_price_mcf=gas_price,
                              heating_system=heating_system, wall_cost=wall_cost, window_cost=window_cost,
                              roof_cost=roof_cost, foundation_cost=foundation_cost, upper_floor_cost=upper_floor_cost,
                              carbon_price=carbon_price, horizon_years=int(horizon), discount_rate=discount)

    with st.expander("🤖 AI settings (Groq)", expanded=False):
        _secret_key = get_api_key(st.secrets if hasattr(st, "secrets") else None)
        if _secret_key:
            st.success("Groq API key loaded from secrets.", icon="🔑")
            GROQ_KEY = _secret_key
        else:
            GROQ_KEY = st.text_input("Groq API key", type="password",
                                     help="Free key at console.groq.com. Stored only in this browser session.") or None
        _models_for_key = available_groq_models(GROQ_KEY) if GROQ_KEY else GROQ_MODELS
        GROQ_MODEL = st.selectbox("Model", _models_for_key,
                                  help="Models your key can use, recommended first. If a model is retired, the "
                                       "assistant automatically falls back to the next recommended one.")
    st.divider()
    st.caption("Group 3 — Veronica Broadway · Namita Moses · Anthony Chavez  \n"
               "Advisor: Dr. Clifford Whitworth  \n"
               "Course: DTSC 5082  \n"
               "Department: UNT College of Artificial Intelligence and Advanced Analytics")

st.title("EcoBuild: Building Energy Efficiency Predictor")
tab_compare, tab_improve, tab_agent, tab_ai, tab_drivers, tab_perf, tab_about, tab_road = st.tabs(
    ["Compare designs", "Suggest improvements", "🧭 Design agent", "🤖 AI assistant", "What drives energy use",
     "Model performance", "About the data", "Roadmap"])


# ============================================================================ compare tab
def design_inputs(key: str, defaults: dict, color: str, title: str = None) -> dict:
    st.markdown(f"<h3 style='color:{color};margin-bottom:0'>{title or 'Design ' + key}</h3>", unsafe_allow_html=True)
    shape_id = st.selectbox("Building form", shapes.index, index=defaults["shape"] - 1,
                            format_func=shape_label, key=f"shape_{key}")
    c1, c2 = st.columns(2)
    orientation = c1.selectbox("Orientation", list(ORIENTATION_LABELS),
                               index=list(ORIENTATION_LABELS).index(defaults["orientation"]),
                               format_func=ORIENTATION_LABELS.get, key=f"orient_{key}")
    glazing = c2.selectbox("Glazing area (% of floor area)", list(GLAZING_AREA_LABELS),
                           index=list(GLAZING_AREA_LABELS).index(defaults["glazing"]),
                           format_func=GLAZING_AREA_LABELS.get, key=f"glaz_{key}")
    if glazing == 0:
        st.selectbox("Glazing distribution", [0], format_func=GLAZING_DIST_LABELS.get,
                     disabled=True, key=f"dist0_{key}", help="No windows, so no distribution.")
        dist = 0
    else:
        opts = [1, 2, 3, 4, 5]
        dist = st.selectbox("Glazing distribution", opts, index=opts.index(defaults["dist"]),
                            format_func=GLAZING_DIST_LABELS.get, key=f"dist_{key}")
    design = shapes.loc[shape_id].to_dict()
    design.update(Orientation=orientation, Glazing_Area=glazing, Glazing_Area_Distribution=dist)
    return design


with tab_compare:
    st.markdown("Pick two design options. EcoBuild predicts their energy loads and explains the difference.")
    col_a, col_b = st.columns(2, gap="large")
    with col_a:
        design_a = design_inputs("A", {"shape": 6, "orientation": 4, "glazing": 0.25, "dist": 3}, COLOR_A)
    with col_b:
        design_b = design_inputs("B", {"shape": 10, "orientation": 4, "glazing": 0.25, "dist": 3}, COLOR_B)

    designs = pd.DataFrame([design_a, design_b], index=["Design A", "Design B"])[FEATURES]
    pred = models.predict(designs)
    a, b = pred.loc["Design A"], pred.loc["Design B"]

    st.divider()
    st.subheader("Predicted energy loads")
    cols = st.columns(3)
    for col, t in zip(cols, ["Heating_Load", "Cooling_Load", "Total_Load"]):
        with col:
            st.markdown(f"**{TARGET_NAMES[t]} load**")
            ca, cb = st.columns(2)
            ca.metric("Design A", f"{a[t]:.1f}")
            cb.metric("Design B", f"{b[t]:.1f}", delta=f"{b[t] - a[t]:+.1f} vs A", delta_color="inverse")
            if RANGES:
                st.caption(f"Typical error of each prediction: ± {half_width(t):.1f}")
    st.caption("Loads in kWh/m². Lower is more energy-efficient. Green delta = Design B uses less than Design A."
               + (f" The typical error is the range that about {RANGES['level']:.0%} of designs the model had never seen "
                  "landed inside." if RANGES else ""))

    diff_pct = (b.Total_Load - a.Total_Load) / a.Total_Load * 100
    too_close = RANGES and abs(b.Total_Load - a.Total_Load) <= gap_threshold("Total_Load")
    if abs(diff_pct) < 2:
        st.info(f"The two designs have nearly the same total load ({diff_pct:+.1f}%).")
    elif too_close:
        st.info(f"**Too close to call.** The total loads differ by {abs(b.Total_Load - a.Total_Load):.1f} kWh/m², "
                f"which is within the model's typical error when comparing two designs "
                f"(±{gap_threshold('Total_Load'):.1f}). Treat these two as a tie.")
    else:
        better, worse = ("A", "B") if diff_pct > 0 else ("B", "A")
        saving = abs(b.Total_Load - a.Total_Load) / max(a.Total_Load, b.Total_Load) * 100
        st.success(f"**Design {better} is more energy-efficient**: it needs about **{saving:.0f}% less** "
                   f"combined heating and cooling load than Design {worse} "
                   f"({abs(b.Total_Load - a.Total_Load):.1f} kWh/m² less).")

    # ------------------------------------------------------------- business view
    bc = business_case(designs, pred, ASSUMPTIONS, "Design A", RANGES["gap"] if RANGES else None)
    rb = bc.loc["Design B"]
    with st.container(border=True):
        st.markdown(f"**💲 What it means in dollars** · {ASSUMPTIONS.floor_area_m2:,.0f} m² project, "
                    f"{ASSUMPTIONS.heating_system.lower()} · *planning estimate, see assumptions in the sidebar*")
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Energy cost / yr, A", fmt_money(bc.loc["Design A", "energy_cost"]))
        m2.metric("Energy cost / yr, B", fmt_money(rb.energy_cost),
                  delta=f"{fmt_money(-rb.annual_savings)} vs A", delta_color="inverse")
        m3.metric("Extra construction, B vs A", fmt_money(rb.extra_construction),
                  help="Envelope only: walls, windows, roof, foundation, upper floors.")
        m4.metric(f"{ASSUMPTIONS.horizon_years}-yr NPV of choosing B", fmt_money(rb.npv),
                  help="Positive = choosing B over A is worth it financially over the horizon.")
        m5.metric("CO₂, B vs A", f"{-rb.co2_saved_t_per_yr:+.1f} t/yr", delta_color="off")
        st.caption(esc(f"Energy savings of B vs A: {fmt_money(rb.annual_savings)}/yr "
                       f"(range {fmt_money(rb.savings_low)} to {fmt_money(rb.savings_high)} from the model's typical error). "
                       f"Payback: {fmt_payback(rb.payback_years)}. Construction costs are placeholders: edit them in the sidebar."))

    chart_df = (pred[TARGETS].rename(columns=TARGET_NAMES).reset_index(names="Design")
                .melt(id_vars="Design", var_name="Load", value_name="kWh/m²"))
    bars = alt.Chart(chart_df).mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
        x=alt.X("Load:N", title=None, sort=["Heating", "Cooling"], axis=alt.Axis(labelAngle=0)),
        xOffset=alt.XOffset("Design:N"),
        y=alt.Y("kWh/m²:Q", title="Predicted load (kWh/m²)"),
        color=alt.Color("Design:N", scale=alt.Scale(range=[COLOR_A, COLOR_B]), legend=alt.Legend(orient="top", title=None)),
        tooltip=["Design", "Load", alt.Tooltip("kWh/m²:Q", format=".2f")],
    )
    labels = bars.mark_text(dy=-8, fontSize=12).encode(text=alt.Text("kWh/m²:Q", format=".1f"), color=alt.value("#444"))
    st.altair_chart((bars + labels).properties(height=320), width="stretch")

    # ------------------------------------------------------------- SHAP difference
    st.subheader("Why do the designs differ?")
    if not models.shap_available():
        st.warning("Install the `shap` package to see explanations.")
    else:
        st.markdown("SHAP splits the difference **Design B − Design A** into the contribution of each design "
                    "feature. Red bars push Design B's load **up**, and green bars push it **down**. "
                    "The bars add up to the total difference.")
        with st.spinner("Computing SHAP explanations…"):
            exp = {t: models.explain(designs, t) for t in TARGETS}
        c1, c2 = st.columns(2, gap="large")
        for col, t in zip((c1, c2), TARGETS):
            sv, base = exp[t]
            d = (sv.loc["Design B"] - sv.loc["Design A"]).rename("Contribution").rename_axis("Feature").reset_index()
            d["Feature"] = d["Feature"].map(FEATURE_LABELS)
            d["Effect"] = d["Contribution"].apply(lambda v: "Raises B's load" if v > 0 else "Lowers B's load")
            chart = alt.Chart(d).mark_bar(cornerRadius=3).encode(
                y=alt.Y("Feature:N", sort=alt.EncodingSortField("Contribution", op="sum", order="descending"), title=None,
                        axis=alt.Axis(labelOverlap=False, labelLimit=200)),
                x=alt.X("Contribution:Q", title="Contribution to B − A (kWh/m²)"),
                color=alt.Color("Effect:N", scale=alt.Scale(domain=["Raises B's load", "Lowers B's load"],
                                                            range=[COLOR_UP, COLOR_DOWN]),
                                legend=alt.Legend(orient="bottom", title=None)),
                tooltip=["Feature", alt.Tooltip("Contribution:Q", format="+.2f")],
            )
            rule = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color="#888").encode(x="x:Q")
            with col:
                st.markdown(f"**{TARGET_NAMES[t]} load: B − A = {d.Contribution.sum():+.2f} kWh/m²**")
                st.altair_chart((chart + rule).properties(height=280), width="stretch")

        with st.expander("SHAP values for each design"):
            for t in TARGETS:
                sv, base = exp[t]
                tbl = sv.T.rename(index=FEATURE_LABELS)
                tbl.loc["Base value (average building)"] = base
                tbl.loc["= Prediction"] = sv.sum(axis=1) + base
                st.markdown(f"**{TARGET_NAMES[t]} load (kWh/m²)**")
                st.dataframe(tbl.style.format("{:+.2f}"), width="stretch")

    with st.expander("Design inputs used for the prediction"):
        st.dataframe(designs.T.rename(index=FEATURE_LABELS), width="stretch")


# ============================================================================ improvements tab
def window_text(g: float, d: int) -> str:
    return "no windows" if g == 0 else f"{GLAZING_AREA_LABELS[g]} glazing, {GLAZING_DIST_LABELS[d].lower()}"


def change_text(cur: dict, new) -> str:
    parts = []
    if int(new.Orientation) != int(cur["Orientation"]):
        parts.append(f"face {ORIENTATION_LABELS[int(new.Orientation)]} instead of {ORIENTATION_LABELS[int(cur['Orientation'])]}")
    if (new.Glazing_Area, int(new.Glazing_Area_Distribution)) != (cur["Glazing_Area"], int(cur["Glazing_Area_Distribution"])):
        parts.append(f"windows: {window_text(new.Glazing_Area, int(new.Glazing_Area_Distribution))} "
                     f"(now {window_text(cur['Glazing_Area'], int(cur['Glazing_Area_Distribution']))})")
    return "; ".join(parts)


with tab_improve:
    st.markdown("Start from a design. EcoBuild tries every orientation and window setup for the **same building form** "
                "and shows the changes that lower the predicted energy load the most.")
    left, right = st.columns([1.2, 1], gap="large")
    with left:
        current = design_inputs("C", {"shape": 6, "orientation": 4, "glazing": 0.40, "dist": 3}, COLOR_A,
                                title="Your current design")
    with right:
        st.markdown("<h3 style='margin-bottom:0'>Limits</h3>", unsafe_allow_html=True)
        min_glaz = st.selectbox("Smallest window share you would accept", list(GLAZING_AREA_LABELS), index=1,
                                format_func=GLAZING_AREA_LABELS.get,
                                help="Removing every window always lowers the predicted load, but it is rarely "
                                     "realistic. Raise this to keep daylight and views.")
        st.caption("The model does not know about daylight, comfort, cost or building codes. "
                   "It only predicts heating and cooling load.")

    shape_row = shapes[(shapes == pd.Series({c: current[c] for c in shapes.columns})).all(axis=1)].iloc[0]
    res = suggest_changes(models, current, shape_row, min_glaz)
    cur = res["current"]
    st.divider()
    st.subheader("Best changes")
    if not res["options"]:
        st.success("No change we tested lowers the predicted load for this design. It is already the best "
                   "setup within your limits.")
    else:
        thr = gap_threshold("Total_Load") if RANGES else None
        cols = st.columns(len(res["options"]), gap="large")
        for col, (name, opt) in zip(cols, res["options"].items()):
            saving = cur.Total_Load - opt.Total_Load
            with col:
                st.markdown(f"**{name}**")
                st.markdown(change_text(current, opt))
                st.metric("Predicted total load", f"{opt.Total_Load:.1f}", delta=f"{-saving:.1f} vs now",
                          delta_color="inverse")
                if RANGES:
                    st.caption(f"± {half_width('Total_Load'):.1f} · heating {opt.Heating_Load:.1f}, cooling {opt.Cooling_Load:.1f}")
                    if saving > thr:
                        st.success(f"Saves about {saving / cur.Total_Load:.0%}. Larger than the model's typical error.")
                    else:
                        st.warning(f"Saves about {saving / cur.Total_Load:.0%}, but that is within the model's "
                                   f"typical error (±{thr:.1f}). Treat it as uncertain.")
                else:
                    st.caption(f"heating {opt.Heating_Load:.1f}, cooling {opt.Cooling_Load:.1f}")

        bars_df = pd.DataFrame(
            [{"Design": "Current", "Load": "Heating", "kWh/m²": cur.Heating_Load},
             {"Design": "Current", "Load": "Cooling", "kWh/m²": cur.Cooling_Load}] +
            [{"Design": n, "Load": "Heating", "kWh/m²": o.Heating_Load} for n, o in res["options"].items()] +
            [{"Design": n, "Load": "Cooling", "kWh/m²": o.Cooling_Load} for n, o in res["options"].items()])
        order = ["Current"] + list(res["options"])
        ch = alt.Chart(bars_df).mark_bar(cornerRadiusTopRight=3, cornerRadiusBottomRight=3, stroke="white", strokeWidth=2).encode(
            y=alt.Y("Design:N", sort=order, title=None, axis=alt.Axis(labelLimit=220, labelOverlap=False)),
            x=alt.X("kWh/m²:Q", title="Predicted load (kWh/m²), heating + cooling"),
            color=alt.Color("Load:N", scale=alt.Scale(domain=["Heating", "Cooling"], range=[COLOR_A, COLOR_B]),
                            legend=alt.Legend(orient="top", title=None)),
            order=alt.Order("Load:N", sort="descending"),
            tooltip=["Design", "Load", alt.Tooltip("kWh/m²:Q", format=".2f")])
        st.altair_chart(ch.properties(height=alt.Step(36)), width="stretch")

    with st.expander("All tested designs for this building form, best first"):
        allv = res["all"].copy()
        allv["Orientation"] = allv["Orientation"].map(ORIENTATION_LABELS)
        allv["Windows"] = [window_text(g, int(d)) for g, d in zip(allv.Glazing_Area, allv.Glazing_Area_Distribution)]
        st.dataframe(allv.sort_values("Total_Load")[["Orientation", "Windows", "Heating_Load", "Cooling_Load", "Total_Load"]]
                     .rename(columns={"Heating_Load": "Heating", "Cooling_Load": "Cooling", "Total_Load": "Total"})
                     .style.format({"Heating": "{:.1f}", "Cooling": "{:.1f}", "Total": "{:.1f}"}),
                     width="stretch", hide_index=True)
    st.caption("The building form (shape and height) is fixed here because the model has only seen 12 forms. "
               "Savings are predictions from simulated data for one climate, so use them to rank options, not to promise savings.")


# ============================================================================ design agent tab
@st.cache_data(show_spinner=False)
def cached_agent(_models, baseline_items: tuple, constraints_items: tuple, assumptions_items: tuple):
    c = dict(constraints_items)
    c["forms"] = list(c["forms"]) if c["forms"] else None
    return run_agent(_models, dict(baseline_items), Constraints(**c), Assumptions(**dict(assumptions_items)))


def memo_text(R, constraints: Constraints, a: Assumptions) -> str:
    lines = [f"# EcoBuild decision memo", "",
             f"**Baseline:** {describe_design(R.table.loc[R.baseline_key])}",
             f"**Constraints:** {constraints.describe()}",
             f"**Project:** {a.floor_area_m2:,.0f} m², {a.heating_system.lower()}, "
             f"{a.horizon_years}-year horizon at {a.discount_rate:.0%}", "", "## Recommendations"]
    lines += [f"- **{k}:** {recommendation_text(v)}" for k, v in R.recommendations.items()]
    if R.explanation:
        lines += ["", "## Why (SHAP drivers vs. baseline)"] + [f"- {f}: {v:+.1f} kWh/m²" for f, v in R.explanation]
    lines += ["", "## How the agent decided"] + [f"{i}. **{t}.** {d}" for i, (t, d) in enumerate(R.steps, 1)]
    lines += ["", "## Cautions",
              "- Loads come from models trained on 768 simulated buildings (12 forms, one climate).",
              "- Construction costs cover the envelope only and use placeholder unit costs; replace them with quotes.",
              "- " + ("The pick is the same when construction costs move ±25%." if R.robust else
                      "The pick changes when construction costs move ±25%; get real quotes before deciding.")]
    lines += [f"- {n}" for n in R.notes]
    return "\n".join(lines)


with tab_agent:
    st.markdown("The design agent **evaluates every design the model knows (768)** against your constraints, estimates "
                "construction cost, energy cost, CO₂ and return on investment, and **recommends the best options**. "
                "Every step is logged below, and every number comes from the trained models and the cost assumptions "
                "in the sidebar.")

    if GROQ_KEY:
        with st.expander("✍️ Describe your project in plain English (AI fills the settings)", expanded=False):
            brief = st.text_area("Brief", key="agent_brief", height=90,
                                 placeholder="e.g. 1,500 m² community center, site forces it to face south, we need "
                                             "decent daylight, and the client won't pay more to build than the current "
                                             "one-story scheme. Minimize energy.")
            if st.button("Apply brief", disabled=not brief):
                try:
                    st.session_state["agent_brief_settings"] = parse_brief(make_client(GROQ_KEY), GROQ_MODEL, brief)
                except Exception as e:
                    st.error(f"Could not read the brief with the AI service ({type(e).__name__}).")
            if st.session_state.get("agent_brief_settings"):
                st.caption("Settings extracted by the AI (applied below): "
                           f"`{st.session_state['agent_brief_settings']}`")

    left, right = st.columns([1.15, 1], gap="large")
    with left:
        baseline = design_inputs("G", {"shape": 6, "orientation": 4, "glazing": 0.40, "dist": 3}, COLOR_A,
                                 title="Baseline design")
    with right:
        st.markdown("<h3 style='margin-bottom:0'>Constraints and goal</h3>", unsafe_allow_html=True)
        stories = st.radio("Stories", ["Any", "One-story", "Two-story"], horizontal=True, key="ag_stories")
        orient_opts = [None] + list(ORIENTATION_LABELS)
        fixed_o = st.selectbox("Orientation", orient_opts, key="ag_orient",
                               format_func=lambda o: "Free (agent may rotate)" if o is None else f"Fixed: {ORIENTATION_LABELS[o]}")
        g_opts = list(GLAZING_AREA_LABELS)
        gmin, gmax = st.select_slider("Window share allowed (daylight)", options=g_opts, value=(0.10, 0.40),
                                      format_func=lambda g: f"{g:.0%}", key="ag_glaz")
        budget_mode = st.radio("Extra construction budget vs. baseline",
                               ["No more than baseline", "Cap at…", "No cap"], horizontal=True, key="ag_budget")
        cap = (0.0 if budget_mode == "No more than baseline" else
               st.number_input("Max extra construction ($)", 0.0, 1e8, 50000.0, step=5000.0, key="ag_cap")
               if budget_mode == "Cap at…" else None)
        objective = st.selectbox("Goal", list(OBJECTIVES), index=list(OBJECTIVES).index("Lowest energy use"),
                                 key="ag_obj")

    constraints = Constraints(stories=stories, orientation=fixed_o, min_glazing=gmin, max_glazing=gmax,
                              max_extra_budget=cap, objective=objective)
    agent_assumptions = ASSUMPTIONS
    if st.session_state.get("agent_brief_settings"):
        baseline, constraints, agent_assumptions = apply_brief(st.session_state["agent_brief_settings"], baseline,
                                                               constraints, ASSUMPTIONS, shapes)
        st.info(esc(f"Using the AI-read brief: {constraints.describe()}. "
                    f"Project floor area {agent_assumptions.floor_area_m2:,.0f} m²."), icon="✍️")
        if st.button("Clear brief settings"):
            st.session_state.pop("agent_brief_settings")
            st.rerun()

    with st.spinner("Agent is evaluating 768 designs…"):
        R = cached_agent(models, tuple(sorted(baseline.items())),
                         tuple(sorted({**constraints.as_dict(),
                                       "forms": tuple(constraints.forms) if constraints.forms else None}.items())),
                         tuple(sorted(agent_assumptions.as_dict().items())))

    st.divider()
    with st.status(f"Agent finished: {len(R.steps)} steps", state="complete", expanded=False):
        for i, (t, d) in enumerate(R.steps, 1):
            st.markdown(esc(f"**{i}. {t}.** {d}"))

    st.subheader("Recommendations")
    if not R.recommendations:
        st.warning("No feasible alternative to the baseline under these constraints. Relax a constraint and try again.")
    else:
        base_row = R.table.loc[R.baseline_key]
        cols = st.columns(len(R.recommendations), gap="medium")
        for col, (label, r) in zip(cols, R.recommendations.items()):
            with col, st.container(border=True):
                st.markdown(f"**{label}**")
                st.markdown(describe_design(r).capitalize())
                st.metric("Total load (kWh/m²)", f"{r.Total_Load:.1f}",
                          delta=f"{r.Total_Load - base_row.Total_Load:+.1f} vs baseline", delta_color="inverse")
                st.metric("Energy cost / yr", fmt_money(r.energy_cost),
                          delta=f"{fmt_money(-r.annual_savings)} vs baseline", delta_color="inverse")
                st.metric("Extra construction", fmt_money(r.extra_construction))
                st.caption(esc(f"Payback: {fmt_payback(r.payback_years)} · NPV {fmt_money(r.npv)} · "
                               f"CO₂ {-r.co2_saved_t_per_yr:+.1f} t/yr"))
                if r.within_error:
                    st.warning("Energy difference is within the model's typical error: treat it as a tie.")
        if R.robust is not None:
            (st.success if R.robust else st.warning)(
                "✔ Robust: the same design stays on top when construction costs move ±25%." if R.robust else
                "⚠ Cost-sensitive: the top pick changes when construction costs move ±25%. Get real quotes.")
        for n in R.notes:
            st.info(esc(n))
        if R.explanation:
            st.markdown("**Why the recommended design is different (SHAP, vs. baseline):** " +
                        ", ".join(f"{f} {v:+.1f} kWh/m²" for f, v in R.explanation))

        # trade-off chart
        st.subheader("Trade-off: cost to build vs. cost to run")
        plot = R.table.reset_index(names="Design").assign(
            Kind="Feasible design", Stories=lambda d: np.where(d.Overall_Height >= 7, "Two-story", "One-story"))
        plot.loc[plot.Design.isin(R.pareto.index), "Kind"] = "Pareto frontier"
        plot.loc[plot.Design == R.baseline_key, "Kind"] = "Baseline"
        rec_keys = {r.name for r in R.recommendations.values()}
        plot.loc[plot.Design.isin(rec_keys), "Kind"] = "Recommended"
        kinds = ["Feasible design", "Pareto frontier", "Recommended", "Baseline"]
        pts = alt.Chart(plot).mark_point(filled=True, opacity=0.85).encode(
            x=alt.X("construction_cost:Q", title="Envelope construction cost ($)", scale=alt.Scale(zero=False),
                    axis=alt.Axis(format="$.3~s", tickCount=6)),
            y=alt.Y("energy_cost:Q", title="Energy cost per year ($)", scale=alt.Scale(zero=False),
                    axis=alt.Axis(format="$,.0f")),
            color=alt.Color("Kind:N", scale=alt.Scale(domain=kinds, range=["#c9cdd2", COLOR_A, "#1baf7a", "#222"]),
                            legend=alt.Legend(orient="top", title=None)),
            shape=alt.Shape("Stories:N", legend=alt.Legend(orient="top", title=None)),
            size=alt.condition(alt.datum.Kind == "Feasible design", alt.value(30), alt.value(140)),
            order=alt.Order("Kind:N", sort="ascending"),
            tooltip=["Design", alt.Tooltip("Total_Load:Q", format=".1f", title="Total load"),
                     alt.Tooltip("energy_cost:Q", format="$,.0f", title="Energy $/yr"),
                     alt.Tooltip("construction_cost:Q", format="$,.0f", title="Construction $"),
                     alt.Tooltip("npv:Q", format="$,.0f", title="NPV vs baseline")])
        front = alt.Chart(plot[plot.Design.isin(R.pareto.index)].sort_values("construction_cost")).mark_line(
            color=COLOR_A, strokeDash=[4, 3]).encode(x="construction_cost:Q", y="energy_cost:Q")
        st.altair_chart((front + pts).properties(height=360), width="stretch")
        st.caption("Each point is one feasible design. Designs on the dashed frontier are the only sensible choices: "
                   "for every other design, something on the frontier is cheaper to build and cheaper to run.")

        # memo
        st.subheader("Decision memo")
        memo = memo_text(R, constraints, agent_assumptions)
        if GROQ_KEY and st.button("✨ Write an executive summary with AI"):
            with st.spinner("Writing…"):
                try:
                    payload = {"baseline": describe_design(R.table.loc[R.baseline_key]),
                               "constraints": constraints.describe(),
                               "recommendations": {k: recommendation_text(v) for k, v in R.recommendations.items()},
                               "shap_drivers": R.explanation, "robust_to_cost_estimates": R.robust, "notes": R.notes}
                    st.session_state["agent_summary"] = summarize_agent(make_client(GROQ_KEY), GROQ_MODEL,
                                                                        st.session_state.get("agent_brief", ""), payload)
                except Exception as e:
                    st.error(f"The AI service returned an error ({type(e).__name__}).")
        if st.session_state.get("agent_summary"):
            with st.container(border=True):
                st.markdown(esc(st.session_state["agent_summary"]))
                st.caption("Written by an LLM from the agent's numbers above.")
        with st.expander("Full memo (no AI needed)"):
            st.markdown(esc(memo))
        st.download_button("Download memo (.md)", memo, file_name="ecobuild_decision_memo.md", mime="text/markdown")


# ============================================================================ AI assistant tab
EXAMPLES = [
    "Why does a one-story building use less energy than a two-story one with the same windows?",
    "I have form 6 facing south with 40% glazing mostly east. How can I cut its energy use?",
    "Compare form 3 and form 9, both facing west with 25% uniform glazing. Which is the better investment?",
    "Find the best design for a 2,000 m² building that must face north, without spending more to build than form 1 with 25% glazing.",
]

with tab_ai:
    st.markdown("Ask in plain English. The assistant answers by **calling EcoBuild's own tools** (the trained models, "
                "the SHAP explainer, the improvement search, the design agent and the cost model), so every number "
                "in its answer comes from the models, not from the language model's memory.")
    if not GROQ_KEY:
        st.info("To chat, add a free Groq API key: open **🤖 AI settings** in the sidebar, or set `GROQ_API_KEY` in "
                "`.streamlit/secrets.toml` (Streamlit Cloud: *Settings → Secrets*). Get a key at "
                "[console.groq.com](https://console.groq.com/keys). The **Design agent** tab works without a key.",
                icon="🔑")
    st.session_state.setdefault("chat", [])
    st.markdown("**Try:**")
    ex_cols = st.columns(2)
    clicked = None
    for i, q in enumerate(EXAMPLES):
        if ex_cols[i % 2].button(q, key=f"ex_{i}", disabled=not GROQ_KEY, width="stretch"):
            clicked = q

    for msg in st.session_state["chat"]:
        with st.chat_message(msg["role"], avatar="🏢" if msg["role"] == "assistant" else None):
            st.markdown(esc(msg["content"]))
            if msg.get("tools"):
                with st.expander(f"🔧 {len(msg['tools'])} tool call(s) behind this answer"):
                    for t in msg["tools"]:
                        st.markdown(f"**{t['tool']}** `{t['args']}`")
                        st.json(t["result"], expanded=False)

    prompt = st.chat_input("Ask about a design, a comparison, savings or ROI…", disabled=not GROQ_KEY) or clicked
    if prompt and GROQ_KEY:
        st.session_state["chat"].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant", avatar="🏢"):
            with st.spinner("Thinking and running EcoBuild tools…"):
                assistant = EcoBuildAssistant(make_client(GROQ_KEY), EcoBuildTools(models, ASSUMPTIONS), GROQ_MODEL)
                answer, log = assistant.reply(st.session_state["chat"])
            st.markdown(esc(answer))
            if log:
                with st.expander(f"🔧 {len(log)} tool call(s) behind this answer"):
                    for t in log:
                        st.markdown(f"**{t['tool']}** `{t['args']}`")
                        st.json(t["result"], expanded=False)
        st.session_state["chat"].append({"role": "assistant", "content": answer, "tools": log})
    if st.session_state["chat"] and st.button("Clear conversation"):
        st.session_state["chat"] = []
        st.rerun()
    st.caption("AI answers can be wrong. Check the tool calls behind each answer. Loads come from simulated data "
               "(12 forms, one climate); dollar figures use the sidebar assumptions.")


# ============================================================================ drivers tab
with tab_drivers:
    st.markdown("Global feature importance from the notebook's SHAP analysis: the **average size of each "
                "feature's effect** on a building's predicted load (mean |SHAP| on the test set).")
    imp = pd.DataFrame(meta["shap_importance"])
    imp["Feature"] = imp["Feature"].map(FEATURE_LABELS)
    imp["Target"] = imp["Target"].map(TARGET_NAMES)
    c1, c2 = st.columns(2, gap="large")
    for col, t, color in [(c1, "Heating", COLOR_A), (c2, "Cooling", COLOR_B)]:
        sub = imp[imp.Target == t]
        ch = alt.Chart(sub).mark_bar(color=color, cornerRadiusTopRight=4, cornerRadiusBottomRight=4).encode(
            y=alt.Y("Feature:N", sort="-x", title=None, axis=alt.Axis(labelOverlap=False, labelLimit=200)),
            x=alt.X("Mean_abs_SHAP:Q", title="Mean |SHAP| (kWh/m²)"),
            tooltip=["Feature", alt.Tooltip("Mean_abs_SHAP:Q", format=".2f")])
        txt = ch.mark_text(align="left", dx=4, color="#444").encode(text=alt.Text("Mean_abs_SHAP:Q", format=".2f"))
        with col:
            st.markdown(f"**{t} load**")
            st.altair_chart((ch + txt).properties(height=300), width="stretch")
    top = {t: imp[imp.Target == t].sort_values("Mean_abs_SHAP", ascending=False)["Feature"].tolist()
           for t in ("Heating", "Cooling")}
    st.markdown(
        f"- **{top['Heating'][0]}** has the largest effect on heating load and **{top['Cooling'][0]}** on cooling "
        f"load. Next largest: {top['Heating'][1].lower()} for heating, {top['Cooling'][1].lower()} for cooling.\n"
        "- **Roof, wall and surface area** describe the building form. They are strongly correlated, "
        "so read them together.\n"
        "- **Glazing area** is the most actionable lever once the form is fixed: more windows mean higher loads.\n"
        "- **Orientation** and **glazing distribution** have only small effects."
    )

    st.divider()
    st.markdown("**Explore the data:** average actual load for each value of a design feature")
    data = models.data()
    feat = st.selectbox("Design feature", FEATURES, index=FEATURES.index("Glazing_Area"),
                        format_func=FEATURE_LABELS.get)
    g = (data.groupby(feat)[TARGETS].mean().rename(columns=TARGET_NAMES).reset_index()
         .melt(id_vars=feat, var_name="Load", value_name="Average load (kWh/m²)"))
    g[feat] = g[feat].astype(str)
    ch = alt.Chart(g).mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3).encode(
        x=alt.X(f"{feat}:N", title=FEATURE_LABELS[feat], axis=alt.Axis(labelAngle=0)),
        xOffset=alt.XOffset("Load:N", sort=["Heating", "Cooling"]),
        y="Average load (kWh/m²):Q",
        color=alt.Color("Load:N", scale=alt.Scale(domain=["Heating", "Cooling"], range=[COLOR_A, COLOR_B]),
                        legend=alt.Legend(orient="top", title=None)),
        tooltip=[feat, "Load", alt.Tooltip("Average load (kWh/m²):Q", format=".2f")])
    st.altair_chart(ch.properties(height=320), width="stretch")


# ============================================================================ performance tab
with tab_perf:
    split = meta.get("split")
    if split:
        st.markdown(f"Results on **{split['n_test_designs']} held-out designs** ({split['n_test_rows']} rows: all four "
                    "orientations of each design). These designs were left out of both training and tuning, so the "
                    "models are tested on designs they have never seen. The Gradient Boosting models were tuned with "
                    "5-fold cross-validation that keeps each design's orientations together.")
    else:
        st.markdown("Results on the **held-out 20% test set** (154 buildings). The tree-based models were tuned "
                    "with 5-fold cross-validation on the training set.")
    m = pd.DataFrame(meta["test_metrics"])
    c1, c2 = st.columns([1, 1.3], gap="large")
    with c1:
        for t in ["Heating Load", "Cooling Load"]:
            st.markdown(f"**{t}**")
            sub = m[m.Target == t].drop(columns="Target").set_index("Model").sort_values("RMSE")
            st.dataframe(sub.style.format("{:.3f}").highlight_min(["MAE", "RMSE"], color="#dcebfb")
                         .highlight_max(["R2"], color="#dcebfb"), width="stretch")
    with c2:
        metric = st.radio("Metric", ["RMSE", "MAE", "R2"], horizontal=True,
                          format_func=lambda x: "R²" if x == "R2" else x)
        ch = alt.Chart(m).mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3).encode(
            x=alt.X("Target:N", title=None, sort=["Heating Load", "Cooling Load"], axis=alt.Axis(labelAngle=0)),
            xOffset=alt.XOffset("Model:N", sort=["Linear Regression", "Random Forest", "Gradient Boosting"]),
            y=alt.Y(f"{metric}:Q", title="R²" if metric == "R2" else f"{metric} (kWh/m²)",
                    scale=alt.Scale(zero=metric != "R2")),
            color=alt.Color("Model:N", sort=["Linear Regression", "Random Forest", "Gradient Boosting"],
                            scale=alt.Scale(range=["#2a78d6", "#eb6834", "#1baf7a"]),
                            legend=alt.Legend(orient="top", title=None)),
            tooltip=["Target", "Model", alt.Tooltip(f"{metric}:Q", format=".3f")])
        st.altair_chart(ch.properties(height=340), width="stretch")
    st.caption("MAE = mean absolute error · RMSE = root mean squared error (both kWh/m², lower is better) · "
               "R² = share of variance explained (higher is better).")


# ============================================================================ about tab
with tab_about:
    st.markdown(
        """
**Dataset:** UCI Machine Learning Repository, *Energy Efficiency* (Tsanas & Xifara, 2012). It has 768 buildings
simulated in Ecotect from **12 building forms**, each with 4 orientations and several glazing options. There are no missing values.

| Variable | Description | Values |
|---|---|---|
| X1 | Relative compactness | 0.62 – 0.98 |
| X2 | Surface area (m²) | 514.5 – 808.5 |
| X3 | Wall area (m²) | 245 – 416.5 |
| X4 | Roof area (m²) | 110.25 – 220.5 |
| X5 | Overall height (m) | 3.5 or 7.0 |
| X6 | Orientation | North, East, South, West |
| X7 | Glazing area | 0%, 10%, 25%, 40% of floor area |
| X8 | Glazing distribution | None, Uniform, or mostly N/E/S/W |
| Y1 | Heating load (kWh/m²) | 6.0 – 43.1 |
| Y2 | Cooling load (kWh/m²) | 10.9 – 48.0 |

**Why you choose a building form instead of typing in areas:** the models learned from only 12 forms.
Picking one of these forms keeps predictions within what the models have seen.

**Limitations:** the data is simulated for a single climate, so predictions are for *comparing* early design
options, not for estimating real energy bills.
"""
    )


# ============================================================================ roadmap tab
with tab_road:
    st.markdown("Where EcoBuild is today, what limits it, and what we plan to do next. "
                "Items under *Next steps* are **plans, not results**.")

    st.subheader("Where we are today")
    sp = meta.get("split")
    st.markdown(
        "- Two models (heating and cooling) trained on 768 simulated buildings, tested on "
        + (f"**{sp['n_test_designs']} whole designs they never saw**." if sp else "a held-out test set.") + "\n"
        "- Predictions come with a typical-error range, and the app says when two designs are too close to call.\n"
        "- Users can compare two designs, see why they differ, and get suggested changes for a building form.\n"
        "- **New: decision support.** A cost model turns loads into energy cost, construction cost, payback, NPV and CO₂; "
        "a **design agent** searches all 768 designs under constraints and recommends options; an **AI assistant** "
        "(Groq LLM with tool calling) answers plain-English questions using EcoBuild's own models."
    )

    st.subheader("Known limits")
    st.markdown(
        "- **Only 12 building forms and 2 heights.** The model has never seen other shapes, so the app only lets you pick "
        "from these 12.\n"
        "- **Simulated data, one climate.** Results rank design options. They are not real energy bills.\n"
        "- **Honest testing roughly doubled the error.** Each design appears four times (one per orientation) with nearly "
        "identical loads, so a random split flatters the model. We now hold out whole designs.\n"
        "- **Some designs are harder.** In our cross-validation, two-story and no-window designs had larger errors than "
        "one-story designs with windows.\n"
        "- **The model predicts load only.** Costs come from a simple envelope cost model with placeholder unit costs; daylight, "
        "comfort and building codes are not modeled.\n"
        "- **The AI assistant explains, it does not predict.** Its numbers come from tool calls; it can still word things "
        "badly, so each answer shows the tool calls behind it."
    )

    st.subheader("Next steps")
    st.markdown(
        """
| When | What | Why | How we will know it worked |
|---|---|---|---|
| Next | **Leave-one-form-out test**: train on 11 forms, test on the 12th, for all 12 | Measures how wrong the model is on a shape it has never seen | An error number per form, shown in the app |
| Next | **Automated checks** that the saved models load and give the same answers | The app should not break silently when a library updates | Tests pass before every release |
| Next | **CSV upload and download** for many designs at once | Lets a team screen a whole list of options | A file of predictions with ranges |
| Later | **More building forms and heights** from a new data source | 12 forms is the biggest limit | Lower leave-one-form-out error on new shapes |
| Later | **More climates** | Heating and cooling needs depend on the weather | Separate validation per climate |
| Later | **Test on real, measured buildings** | Simulation is not the same as reality | Error on measured data, reported next to simulated |
| Next | **Real cost data** in place of placeholder unit costs (e.g. regional cost books or firm quotes) | ROI rankings depend on construction costs | Recommendations checked against real project budgets |
| Next | **Evaluate the AI assistant**: a set of test questions with known answers | LLM wording can drift even with grounded tools | Share of answers with correct numbers and caveats |
| Later | **Daylight and comfort** next to energy and cost | An energy-only answer can push designs somewhere unbuildable | A design is ranked on more than one measure |
"""
    )

    st.subheader("Data sources we would explore")
    st.markdown(
        """
None of these can be added to our 768 rows directly. They measure different things, so each would need its own
check and probably its own model. We have only read their descriptions, not tested them.

| Source | What it is | Why it could help | Open question |
|---|---|---|---|
| [NREL End-Use Load Profiles (ResStock / ComStock)](https://registry.opendata.aws/nrel-pds-building-stock/) | Simulated US building stock, 15-minute loads, many climates, efficiency-upgrade scenarios | Many more building types and climates, same simulated style | Which geometry and heating/cooling fields it has |
| [ThermBuild](https://arxiv.org/html/2606.01994v1) | 960 houses (958 simulated, 2 real), 5 Central European climates | Adds climate, construction age and materials | Outputs are heat-pump power and temperatures, and we found no license on the paper page |
| [Building Data Genome Project 2](https://arxiv.org/pdf/2006.02273) | 1,636 real commercial buildings, hourly meter data | Real measured energy to test against | No geometry or window details, so it would test a simpler model |
"""
    )
    st.caption("We will say \"explore\" until we have checked each source's fields and license.")
