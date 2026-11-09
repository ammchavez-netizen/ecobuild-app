"""EcoBuild core logic: feature engineering, model loading, prediction and SHAP explanations.

Kept separate from app.py so the same code can be reused in notebooks or tests.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

FEATURES = [
    "Relative_Compactness", "Surface_Area", "Wall_Area", "Roof_Area",
    "Overall_Height", "Orientation", "Glazing_Area", "Glazing_Area_Distribution",
]
TARGETS = ["Heating_Load", "Cooling_Load"]
SHAPE_COLS = ["Relative_Compactness", "Surface_Area", "Wall_Area", "Roof_Area", "Overall_Height"]

ORIENTATION_LABELS = {2: "North", 3: "East", 4: "South", 5: "West"}
GLAZING_AREA_LABELS = {0.0: "0% (no windows)", 0.10: "10%", 0.25: "25%", 0.40: "40%"}
GLAZING_DIST_LABELS = {0: "None", 1: "Uniform (all sides)", 2: "Mostly North", 3: "Mostly East",
                       4: "Mostly South", 5: "Mostly West"}
FEATURE_LABELS = {
    "Relative_Compactness": "Relative compactness", "Surface_Area": "Surface area",
    "Wall_Area": "Wall area", "Roof_Area": "Roof area", "Overall_Height": "Overall height",
    "Orientation": "Orientation", "Glazing_Area": "Glazing area",
    "Glazing_Area_Distribution": "Glazing distribution",
}


def engineer_features(data: pd.DataFrame) -> pd.DataFrame:
    """Identical to the notebook's feature engineering (Section 7)."""
    out = data.copy()
    out["Wall_to_Roof_Ratio"] = out["Wall_Area"] / out["Roof_Area"]
    out["Envelope_per_Height"] = out["Surface_Area"] / out["Overall_Height"]
    out["Glazing_x_Height"] = out["Glazing_Area"] * out["Overall_Height"]
    out["Is_Two_Story"] = (out["Overall_Height"] >= 7.0).astype(int)
    out["Orientation"] = out["Orientation"].astype(int).astype(str)
    out["Glazing_Area_Distribution"] = out["Glazing_Area_Distribution"].astype(int).astype(str)
    return out


# A "design" is every input except orientation. Each design appears four times in the data (once per
# orientation) with nearly identical loads, so train/test splits must keep those four rows together.
DESIGN_COLS = [c for c in FEATURES if c != "Orientation"]


def split_by_design(df: pd.DataFrame, test_size: float, random_state: int):
    """Hold out whole designs: all four orientations of a design stay on the same side of the split."""
    from sklearn.model_selection import GroupShuffleSplit
    groups = df.groupby(DESIGN_COLS).ngroup()
    train_idx, test_idx = next(
        GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state).split(df, groups=groups))
    return df.iloc[train_idx], df.iloc[test_idx]


class EcoBuildModels:
    """Loads the saved model bundle; retrains from the bundled data if the saved models can't be loaded."""

    def __init__(self, model_dir: str | Path):
        self.dir = Path(model_dir)
        self.meta = json.loads((self.dir / "metadata.json").read_text())
        self.num_features = self.meta["num_features"]
        self.cat_features = self.meta["cat_features"]
        self.models, self.status = {}, {}
        for t in TARGETS:
            try:
                self.models[t] = joblib.load(self.dir / f"{t.lower()}_model.joblib")
                self.models[t].predict(self._model_input(self.shapes().head(1).assign(
                    Orientation=2, Glazing_Area=0.0, Glazing_Area_Distribution=0)))
                self.status[t] = "loaded"
            except Exception as exc:  # version mismatch, missing file, etc.
                self.models[t] = self._retrain(t)
                self.status[t] = f"retrained ({type(exc).__name__})"
        self.background = pd.read_csv(self.dir / "shap_background.csv")[FEATURES]
        self._explainers = {}

    # ------------------------------------------------------------------ data helpers
    def data(self) -> pd.DataFrame:
        return pd.read_csv(self.dir / "ENB2012_data.csv")

    def shapes(self) -> pd.DataFrame:
        s = pd.DataFrame(self.meta["valid_shapes"])[SHAPE_COLS]
        s.index = np.arange(1, len(s) + 1)
        return s

    def _model_input(self, raw: pd.DataFrame) -> pd.DataFrame:
        return engineer_features(raw[FEATURES])[self.num_features + self.cat_features]

    # ------------------------------------------------------------------ fallback training
    def _retrain(self, target: str):
        from sklearn.compose import ColumnTransformer
        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.impute import SimpleImputer
        from sklearn.model_selection import train_test_split
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OneHotEncoder

        df = self.data()
        if "split" in self.meta:   # bundles exported with grouped validation hold out whole designs
            train, _ = split_by_design(df, self.meta["split"]["test_size"], self.meta["random_state"])
        else:                      # older bundles used a plain random split
            train, _ = train_test_split(df, test_size=0.20, random_state=self.meta["random_state"])
        prep = ColumnTransformer([
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), self.num_features),
            ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                              ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), self.cat_features),
        ])
        params = self.meta.get("best_params", {}).get(target, {})
        pipe = Pipeline([("prep", prep),
                         ("model", GradientBoostingRegressor(random_state=self.meta["random_state"], **params))])
        pipe.fit(self._model_input(train), train[target])
        return pipe

    # ------------------------------------------------------------------ prediction
    def predict(self, designs: pd.DataFrame) -> pd.DataFrame:
        X = self._model_input(designs)
        out = pd.DataFrame({t: self.models[t].predict(X) for t in TARGETS}, index=designs.index)
        out["Total_Load"] = out[TARGETS].sum(axis=1)
        return out

    # ------------------------------------------------------------------ SHAP
    def shap_available(self) -> bool:
        try:
            import shap  # noqa: F401
            return True
        except ImportError:
            return False

    def explain(self, designs: pd.DataFrame, target: str):
        """Exact SHAP values on the 8 original design features (same method as notebook Section 14.2)."""
        import shap
        if target not in self._explainers:
            model = self.models[target]
            f = lambda arr: model.predict(self._model_input(pd.DataFrame(arr, columns=FEATURES)))
            bg = self.background.sample(min(50, len(self.background)), random_state=0)
            self._explainers[target] = shap.explainers.Exact(f, shap.maskers.Independent(bg, max_samples=50))
        sv = self._explainers[target](designs[FEATURES])
        return pd.DataFrame(sv.values, columns=FEATURES, index=designs.index), float(sv.base_values[0])


# ---------------------------------------------------------------------------------- ranges and suggestions
def design_variants(shape_row: pd.Series) -> pd.DataFrame:
    """Every orientation / glazing amount / glazing layout available for one building form (64 designs)."""
    rows = []
    for o in ORIENTATION_LABELS:
        for g in GLAZING_AREA_LABELS:
            for d in ([0] if g == 0 else [1, 2, 3, 4, 5]):
                rows.append({**shape_row.to_dict(), "Orientation": o, "Glazing_Area": g, "Glazing_Area_Distribution": d})
    return pd.DataFrame(rows)[FEATURES]


def suggest_changes(models: "EcoBuildModels", current: dict, shape_row: pd.Series, min_glazing: float = 0.0) -> dict:
    """Best alternatives for one building form (the form itself never changes).

    Options: rotate only, change the windows only (amount and layout), and the best of everything.
    Each is the lowest predicted total load of its kind. Only designs with at least `min_glazing`
    window share are considered, so the tool does not just recommend removing every window.
    """
    v = design_variants(shape_row)
    v = pd.concat([v, models.predict(v)], axis=1)
    v = v[v.Glazing_Area >= min_glazing]
    cur = models.predict(pd.DataFrame([current])[FEATURES]).iloc[0]
    same_o = v.Orientation == current["Orientation"]
    same_w = (v.Glazing_Area == current["Glazing_Area"]) & (v.Glazing_Area_Distribution == current["Glazing_Area_Distribution"])
    pools = {"Rotate the building": v[same_w & ~same_o],
             "Change the windows": v[same_o & ~same_w],
             "Best overall": v[~(same_o & same_w)]}
    out, seen = {}, set()
    for name, sub in pools.items():
        if not len(sub):
            continue
        best = sub.sort_values("Total_Load").iloc[0]
        key = (best.Orientation, best.Glazing_Area, best.Glazing_Area_Distribution)
        if best.Total_Load < cur.Total_Load and key not in seen:
            out[name] = best
            seen.add(key)
    return {"current": cur, "options": out, "all": v}
