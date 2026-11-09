"""EcoBuild — Building Energy Efficiency Predictor (Streamlit prototype).

Run locally:   streamlit run app.py
"""
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from ecobuild_core import (FEATURE_LABELS, FEATURES, GLAZING_AREA_LABELS, GLAZING_DIST_LABELS,
                           ORIENTATION_LABELS, TARGETS, EcoBuildModels, suggest_changes)

st.set_page_config(page_title="EcoBuild", page_icon="🏢", layout="wide")

MODEL_DIR = Path(__file__).parent / "models"
COLOR_A, COLOR_B = "#2a78d6", "#eb6834"          # Design A / Design B (used everywhere)
COLOR_UP, COLOR_DOWN = "#e34948", "#1baf7a"      # raises load / lowers load
TARGET_NAMES = {"Heating_Load": "Heating", "Cooling_Load": "Cooling", "Total_Load": "Total"}


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
    st.caption("Group 3 — Veronica Broadway · Namita Moses · Anthony Chavez  \n"
               "Advisor: Dr. Clifford Whitworth  \n"
               "Course: DTSC 5082  \n"
               "Department: UNT College of Artificial Intelligence and Advanced Analytics")

st.title("EcoBuild: Building Energy Efficiency Predictor")
tab_compare, tab_improve, tab_drivers, tab_perf, tab_about, tab_road = st.tabs(
    ["Compare designs", "Suggest improvements", "What drives energy use", "Model performance",
     "About the data", "Roadmap"])


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
        "- Users can compare two designs, see why they differ, and get suggested changes for a building form."
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
        "- **The model predicts load only.** It knows nothing about cost, daylight, comfort or building codes."
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
| Later | **Cost, daylight and comfort** next to energy | An energy-only answer can push designs somewhere unbuildable | A design is ranked on more than one measure |
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
