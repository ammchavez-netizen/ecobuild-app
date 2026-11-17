"""EcoBuild trust checks: calibrated prediction ranges and a backtest of the recommendations.

Two questions a user should be able to ask before trusting EcoBuild:

1. "How far off can a prediction be?"  -> `calibrate_ranges`
   Conformal prediction ranges, calibrated separately for one-story and two-story buildings (their errors differ
   a lot). Calibration uses out-of-fold errors on the training designs only; every design's four orientations
   stay in the same fold. The score for a design is its worst error over the four orientations, so a range covers
   the whole design, not just one row. The ranges are then checked on the held-out test designs and on repeated
   random splits.

2. "Do the recommendations actually save energy?"  -> `backtest_recommendations`
   Every design in the dataset was simulated, so we know the true loads of every option the app can recommend.
   We predict each design with a model that never saw it (grouped cross-validation), run the app's own
   recommendation code on those predictions, and score the picks against the simulated truth.

Run `python tools/build_trust_report.py` to recompute both and store them in models/metadata.json.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

from ecobuild_core import DESIGN_COLS, FEATURES, SHAPE_COLS, TARGETS, EcoBuildModels, design_variants, suggest_changes

ALL = TARGETS + ["Total_Load"]
TYPES = {0: "one_story", 1: "two_story"}
PAIR_TYPES = ("one_story", "two_story", "mixed")


# ================================================================================================ helpers
def conformal_quantile(scores, level: float) -> float:
    """Finite-sample split-conformal quantile: the ceil((n+1)·level)-th smallest score."""
    s = np.sort(np.asarray(scores, dtype=float))
    n = len(s)
    k = int(np.ceil((n + 1) * level))
    return float(s[min(k, n) - 1])


def with_ids(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["design_id"] = out.groupby(DESIGN_COLS).ngroup()
    out["form_id"] = out.groupby(SHAPE_COLS).ngroup()
    out["two_story"] = (out.Overall_Height >= 7).astype(int)
    return out


def fit_predict(models: EcoBuildModels, train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """Refit both saved pipelines (same settings) on `train` and predict `test`."""
    Xtr, Xte = models._model_input(train), models._model_input(test)
    out = pd.DataFrame({t: clone(models.models[t]).fit(Xtr, train[t]).predict(Xte) for t in TARGETS}, index=test.index)
    out["Total_Load"] = out[TARGETS].sum(axis=1)
    return out


def grouped_oof(models: EcoBuildModels, data: pd.DataFrame, groups: pd.Series, n_splits: int = 5) -> pd.DataFrame:
    """Out-of-fold predictions: each row is predicted by a model that never saw its group."""
    parts = []
    for tr, te in GroupKFold(n_splits=n_splits).split(data, groups=groups):
        parts.append(fit_predict(models, data.iloc[tr], data.iloc[te]))
    return pd.concat(parts).loc[data.index]


def residuals(data: pd.DataFrame, pred: pd.DataFrame) -> pd.DataFrame:
    r = pd.DataFrame({t: data[t] - pred[t] for t in TARGETS}, index=data.index)
    r["Total_Load"] = r[TARGETS].sum(axis=1)
    return r


def pair_type(two_a: int, two_b: int) -> str:
    return TYPES[two_a] if two_a == two_b else "mixed"


def _pairs(data: pd.DataFrame, res: pd.DataFrame, t: str) -> pd.DataFrame:
    """|error of A − error of B| for every pair of rows from two different designs, labelled by pair type."""
    e, s, g = res[t].to_numpy(), data.two_story.to_numpy(), data.design_id.to_numpy()
    i, j = np.triu_indices(len(e), 1)
    keep = g[i] != g[j]
    i, j = i[keep], j[keep]
    kind = np.where(s[i] != s[j], "mixed", np.where(s[i] == 1, "two_story", "one_story"))
    return pd.DataFrame({"score": np.abs(e[i] - e[j]), "kind": kind})


def widths_from_residuals(data: pd.DataFrame, res: pd.DataFrame, level: float) -> dict:
    """Range half-widths per building type (design-level conformal) and gap half-widths per pair type."""
    half, gap = {}, {}
    for code, name in TYPES.items():
        m = data.two_story == code
        worst = res[m].abs().groupby(data.design_id[m]).max()       # one score per design
        half[name] = {t: conformal_quantile(worst[t], level) for t in ALL}
        half[name]["n_designs"] = int(worst.shape[0])
    for t in ALL:
        p = _pairs(data, res, t)
        for k in PAIR_TYPES:
            gap.setdefault(k, {})[t] = conformal_quantile(p.score[p.kind == k], level)
    return {"half_width": half, "gap": gap}


def coverage(data: pd.DataFrame, res: pd.DataFrame, w: dict) -> dict:
    """Share of rows inside their range, and of design pairs whose true difference is inside the gap range."""
    unit, pair = {}, {}
    for code, name in TYPES.items():
        m = data.two_story == code
        unit[name] = {t: float((res[t][m].abs() <= w["half_width"][name][t]).mean()) for t in ALL}
        unit[name]["n_rows"] = int(m.sum())
    for t in ALL:
        p = _pairs(data, res, t)
        for k in PAIR_TYPES:
            sub = p[p.kind == k]
            pair.setdefault(k, {})[t] = float((sub.score <= w["gap"][k][t]).mean())
    return {"range": unit, "gap": pair}


# ================================================================================================ ranges
def calibrate_ranges(models: EcoBuildModels, level: float = 0.9, n_repeats: int = 30, test_size: float = 0.2,
                     random_state: int = 42) -> dict:
    """Mondrian (per building type), group-aware conformal ranges + checks on unseen designs."""
    df = with_ids(models.data())
    split = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    tr_idx, te_idx = next(split.split(df, groups=df.design_id))
    train, test = df.iloc[tr_idx], df.iloc[te_idx]

    # 1) calibrate on the training designs only (the deployed models were trained on these)
    res_oof = residuals(train, grouped_oof(models, train, train.design_id))
    w = widths_from_residuals(train, res_oof, level)

    # 2) check on the held-out designs with the deployed models
    res_test = residuals(test, models.predict(test))
    held = coverage(test, res_test, w)

    # 3) repeat the whole procedure on many random design splits (more test designs -> steadier estimate)
    reps = []
    for seed in range(n_repeats):
        a, b = next(GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=1000 + seed)
                    .split(df, groups=df.design_id))
        tr, te = df.iloc[a], df.iloc[b]
        w_r = widths_from_residuals(tr, residuals(tr, grouped_oof(models, tr, tr.design_id)), level)
        reps.append(coverage(te, residuals(te, fit_predict(models, tr, te)), w_r))

    def avg(path):
        vals = [r for r in reps]
        for p in path:
            vals = [v[p] for v in vals]
        return float(np.mean(vals)), float(np.min(vals))

    repeated = {"n_splits": n_repeats, "range": {}, "gap": {}}
    for name in TYPES.values():
        repeated["range"][name] = {t: dict(zip(("mean", "min"), avg(("range", name, t)))) for t in ALL}
    for k in PAIR_TYPES:
        repeated["gap"][k] = {t: dict(zip(("mean", "min"), avg(("gap", k, t)))) for t in ALL}

    # Backward-compatible single numbers (the larger, two-story values: safe for code that ignores building type)
    return {
        "level": level,
        "method": ("Split-conformal ranges calibrated separately for one-story and two-story buildings, from "
                   "5-fold GroupKFold out-of-fold errors on the training designs (each design's 4 orientations "
                   "kept together). Score per design = its largest error over the 4 orientations. "
                   "Gap ranges (for comparing two designs) use the error of the difference, per pair type."),
        "meaning": (f"About {level:.0%} of designs the model has never seen fall inside the range, on average. "
                    "Not a guarantee for any single design, for other building shapes or for real buildings."),
        "by_type": w["half_width"],
        "gap_by_pair": w["gap"],
        "half_width": {t: w["half_width"]["two_story"][t] for t in ALL},
        "gap": {t: w["gap"]["two_story"][t] for t in ALL},
        "heldout": {"n_designs": int(test.design_id.nunique()), **held},
        "repeated": repeated,
    }


# ================================================================================================ backtest
class HeldOutModels:
    """Stands in for EcoBuildModels: every prediction comes from a model that never saw that design."""

    def __init__(self, models: EcoBuildModels, pred: pd.DataFrame, data: pd.DataFrame):
        self.meta, self._m = models.meta, models
        keyed = pd.concat([data[FEATURES].astype(float).reset_index(drop=True), pred.reset_index(drop=True)], axis=1)
        self._lookup = keyed.set_index(FEATURES)

    def shapes(self):
        return self._m.shapes()

    def shap_available(self):
        return False

    def predict(self, designs: pd.DataFrame) -> pd.DataFrame:
        keys = pd.MultiIndex.from_frame(designs[FEATURES].astype(float))
        out = self._lookup.reindex(keys)[ALL]
        out.index = designs.index
        return out


def _truth(data: pd.DataFrame) -> pd.DataFrame:
    t = data[FEATURES + TARGETS].copy()
    t[FEATURES] = t[FEATURES].astype(float)
    t["Total_Load"] = t[TARGETS].sum(axis=1)
    return t.set_index(FEATURES)[ALL]


def backtest_recommendations(models: EcoBuildModels, by: str = "design", min_glazing: float = 0.10,
                             n_splits: int = 5) -> dict:
    """Score 'Suggest improvements' and the design agent against the simulated truth.

    by="design": the model never saw the recommended design or the starting design (grouped CV by design).
    by="form":   harder stress test: the model never saw the building form at all (leave-one-form-out).
    """
    from ecobuild_agent import Constraints, run_agent
    from ecobuild_econ import Assumptions

    df = with_ids(models.data())
    groups = df.design_id if by == "design" else df.form_id
    pred = grouped_oof(models, df, groups, n_splits=n_splits if by == "design" else df.form_id.nunique())
    held = HeldOutModels(models, pred, df)
    truth = _truth(df)
    shapes = models.shapes()
    ranges = models.meta.get("ranges") or {}

    def true_total(rows: pd.DataFrame) -> pd.Series:
        return truth.reindex(pd.MultiIndex.from_frame(rows[FEATURES].astype(float)))["Total_Load"].to_numpy()

    def gap_total(two_a, two_b):
        g = (ranges.get("gap_by_pair") or {}).get(pair_type(two_a, two_b))
        return g["Total_Load"] if g else (ranges.get("gap") or {}).get("Total_Load", 0.0)

    # ---------------------------------------------------------------- Suggest improvements (form fixed)
    rows = []
    starts = df[df.Glazing_Area >= min_glazing - 1e-9]
    for _, s in starts.iterrows():
        cur = s[FEATURES].to_dict()
        shape_row = shapes[(shapes == pd.Series({c: cur[c] for c in SHAPE_COLS})).all(axis=1)].iloc[0]
        res = suggest_changes(held, cur, shape_row, min_glazing)
        if not res["options"]:
            continue
        pool = res["all"]
        pool = pool[~((pool.Orientation == cur["Orientation"]) & (pool.Glazing_Area == cur["Glazing_Area"])
                      & (pool.Glazing_Area_Distribution == cur["Glazing_Area_Distribution"]))]
        true_pool = np.sort(true_total(pool))
        cur_true = float(true_total(pd.DataFrame([cur]))[0])
        two = int(cur["Overall_Height"] >= 7)
        for name, opt in res["options"].items():
            sub = pool if name == "Best overall" else None
            pred_saving = float(res["current"].Total_Load - opt.Total_Load)
            true_load = float(true_total(pd.DataFrame([opt[FEATURES]]))[0])
            row = {"option": name, "pred_saving": pred_saving, "true_saving": cur_true - true_load,
                   "certain": pred_saving > gap_total(two, two), "two_story": two}
            if sub is not None:
                row["regret"] = true_load - true_pool[0]
                row["near_best"] = bool(true_load - true_pool[0] <= 1.0)      # within 1 kWh/m² of the true best
                row["pool_spread"] = float(cur_true - true_pool[0])
            rows.append(row)
    S = pd.DataFrame(rows)

    def summarize(d: pd.DataFrame) -> dict:
        out = {"n": int(len(d)),
               "saves_energy": float((d.true_saving > 0).mean()),
               "mean_pred_saving": float(d.pred_saving.mean()), "mean_true_saving": float(d.true_saving.mean()),
               "mae_saving": float((d.pred_saving - d.true_saving).abs().mean())}
        if "regret" in d and d.regret.notna().any():
            dd = d.dropna(subset=["regret"])
            out.update(near_best=float(dd.near_best.mean()), mean_regret=float(dd.regret.mean()), p90_regret=float(dd.regret.quantile(0.9)),
                       share_of_best_possible=float(dd.true_saving.sum() / dd.pool_spread.sum()))
        return out

    suggest = {name: summarize(S[S.option == name]) for name in S.option.unique()}
    suggest["certain"] = summarize(S[S.certain])
    suggest["uncertain"] = summarize(S[~S.certain]) if (~S.certain).any() else {"n": 0}
    suggest["all"] = summarize(S)
    suggest["by_option_and_flag"] = {f"{n} · {'clear' if c else 'too close to call'}": summarize(d)
                                     for (n, c), d in S.groupby(["option", "certain"])}

    # ---------------------------------------------------------------- Design agent (all forms, default brief)
    from ecobuild_agent import OBJECTIVES
    from ecobuild_econ import business_case
    a = Assumptions()
    agent_rows = []
    for objective in ("Lowest energy use", "Best value (highest NPV)"):
        col, asc = OBJECTIVES[objective]
        sign = 1 if asc else -1                       # lower is better after multiplying by sign
        for _, s in starts.iterrows():
            base = s[FEATURES].to_dict()
            R = run_agent(held, base, Constraints(objective=objective), a, explain=False)
            tab = R.table
            true_pred = truth.reindex(pd.MultiIndex.from_frame(tab[FEATURES].astype(float)))
            true_pred.index = tab.index
            true_bc = business_case(tab[FEATURES], true_pred, a, R.baseline_key)
            score = sign * true_bc[col]                # true objective, lower = better
            others = score.drop(index=R.baseline_key)
            base_score = score.loc[R.baseline_key]
            truly_better_exists = bool((others < base_score - 1e-9).any())
            rec = next((r for k, r in R.recommendations.items() if k.startswith("Recommended")), None)
            row = {"objective": objective, "recommended": rec is not None,
                   "better_exists": truly_better_exists, "n_feasible": int(len(others))}
            if rec is not None:
                k = rec.name
                row.update(beats_baseline=bool(score.loc[k] < base_score - 1e-9),
                           regret=float(score.loc[k] - others.min()),
                           true_rank=int((others < score.loc[k] - 1e-9).sum()) + 1,
                           pred_gain=float(sign * (tab.loc[R.baseline_key, col] - tab.loc[k, col])),
                           true_gain=float(base_score - score.loc[k]),
                           near_best=bool(score.loc[k] - others.min() <= (1.0 if col == "Total_Load" else 0.0)),
                           constraints_met=bool(0.10 - 1e-9 <= rec.Glazing_Area <= 0.40 + 1e-9
                                                and rec.extra_construction <= 1e-6))
            agent_rows.append(row)
    A = pd.DataFrame(agent_rows)
    agent = {}
    for objective, d in A.groupby("objective"):
        r = d[d.recommended]
        unit = "kWh/m²" if OBJECTIVES[objective][0] == "Total_Load" else "$ NPV"
        agent[objective] = {
            "n": int(len(d)), "unit": unit,
            "recommended": int(len(r)),
            "beats_baseline": float(r.beats_baseline.mean()),
            "mean_pred_gain": float(r.pred_gain.mean()), "mean_true_gain": float(r.true_gain.mean()),
            "mean_regret": float(r.regret.mean()), "median_true_rank": float(r.true_rank.median()),
            "near_best": float(r.near_best.mean()) if unit == "kWh/m²" else None,
            "median_feasible": float(d.n_feasible.median()),
            "constraints_met": float(r.constraints_met.mean()),
            "kept_baseline": int((~d.recommended).sum()),
            "kept_baseline_correct": float((~d[~d.recommended].better_exists).mean()) if (~d.recommended).any() else None,
        }

    held_r = residuals(df, pred)
    return {
        "held_out_by": by,
        "description": ("Each design predicted by a model trained without it (5-fold GroupKFold by design)"
                        if by == "design" else
                        "Each design predicted by a model trained without its building form (leave-one-form-out)"),
        "model_mae": {t: float(held_r[t].abs().mean()) for t in ALL},
        "starting_designs": int(len(starts)),
        "recommendations_in_dataset": True,   # every option is one of the 768 simulated designs
        "min_glazing": min_glazing,
        "suggest": suggest,
        "agent": agent,
        "units": "kWh/m² of total (heating + cooling) load; shares are 0-1",
    }
