"""EcoBuild — Building Energy Efficiency Predictor (Streamlit prototype).

Run locally:   streamlit run app.py
"""
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from ecobuild_core import (FEATURE_LABELS, FEATURES, GLAZING_AREA_LABELS, GLAZING_DIST_LABELS,
                           ORIENTATION_LABELS, TARGETS, EcoBuildModels)

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
               "UNT Anuradha and Vikas Sinha Department of Data Science")

st.title("EcoBuild: Building Energy Efficiency Predictor")
tab_compare, tab_drivers, tab_perf, tab_about = st.tabs(
    ["Compare designs", "What drives energy use", "Model performance", "About the data"])


# ============================================================================ compare tab
def design_inputs(key: str, defaults: dict, color: str) -> dict:
    st.markdown(f"<h3 style='color:{color};margin-bottom:0'>Design {key}</h3>", unsafe_allow_html=True)
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
    st.caption("Loads in kWh/m². Lower is more energy-efficient. Green delta = Design B uses less than Design A.")

    diff_pct = (b.Total_Load - a.Total_Load) / a.Total_Load * 100
    if abs(diff_pct) < 2:
        st.info(f"The two designs have nearly the same total load ({diff_pct:+.1f}%).")
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
    st.markdown(
        "- **Overall height** (one- vs. two-story) has by far the largest effect on both loads.\n"
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
