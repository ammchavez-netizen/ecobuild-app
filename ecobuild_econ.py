"""EcoBuild economics: turn predicted loads (kWh/m²) into dollars, CO2 and return on investment.

Everything here is a *planning estimate*. The model predicts simulated heating and cooling loads; this module
adds clearly labeled, user-editable assumptions (energy prices, equipment efficiency, unit construction
costs) so that two designs can be compared in business terms. Uncertainty from the model's typical error is
carried through to every dollar figure.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

KWH_PER_MCF = 1.037 * 293.07      # 1 thousand cubic feet of natural gas ≈ 1.037 MMBtu ≈ 303.9 kWh (EIA heat content)
GAS_KG_CO2_PER_KWH = 53.06 / 293.07  # EPA emission factor for natural gas: 53.06 kg CO2 per MMBtu


@dataclass
class Assumptions:
    """Editable assumptions. Defaults are rough U.S. averages or placeholders — replace with local numbers."""
    floor_area_m2: float = 1000.0          # the real project's gross floor area
    elec_price: float = 0.140              # $/kWh — EIA STEO (Oct 2026) U.S. commercial average, 2026 forecast
    gas_price_mcf: float = 11.46           # $/thousand cubic feet — EIA STEO (Oct 2026) U.S. commercial, 2026
    heating_system: str = "Gas furnace"    # or "Heat pump"
    furnace_eff: float = 0.90              # delivered heat / fuel energy
    heat_pump_cop: float = 3.0
    cooling_cop: float = 3.0               # air conditioner coefficient of performance
    grid_kg_co2_per_kwh: float = 0.37      # approx. U.S. grid average (EPA eGRID)
    # Placeholder unit costs for the building envelope ($ per m² of element). Replace with an estimator's numbers.
    wall_cost: float = 250.0
    window_cost: float = 650.0
    roof_cost: float = 200.0
    foundation_cost: float = 180.0
    upper_floor_cost: float = 150.0        # structure of each floor above ground (slab, stairs), $ per m² of upper floor
    carbon_price: float = 0.0              # optional $ per tonne CO2 (0 = ignore carbon in the money figures)
    horizon_years: int = 20
    discount_rate: float = 0.05
    energy_escalation: float = 0.02

    def as_dict(self) -> dict:
        return asdict(self)

    # ---- derived prices per kWh of *delivered* heat / cooling
    @property
    def heat_price_per_kwh(self) -> float:
        if self.heating_system == "Heat pump":
            return self.elec_price / self.heat_pump_cop
        return (self.gas_price_mcf / KWH_PER_MCF) / self.furnace_eff

    @property
    def cool_price_per_kwh(self) -> float:
        return self.elec_price / self.cooling_cop

    @property
    def heat_co2_per_kwh(self) -> float:
        if self.heating_system == "Heat pump":
            return self.grid_kg_co2_per_kwh / self.heat_pump_cop
        return GAS_KG_CO2_PER_KWH / self.furnace_eff

    @property
    def cool_co2_per_kwh(self) -> float:
        return self.grid_kg_co2_per_kwh / self.cooling_cop


# ------------------------------------------------------------------------------------------- geometry & cost
def geometry(designs: pd.DataFrame) -> pd.DataFrame:
    """Envelope quantities per m² of floor area for each design in the dataset's reference geometry.

    In the dataset, surface area = wall area + 2 × roof area, so the footprint equals the roof area and the
    floor area is roof area × number of stories. Glazing area is defined as a share of floor area.
    """
    stories = np.where(designs["Overall_Height"] >= 7.0, 2, 1)
    floor = designs["Roof_Area"] * stories
    window = designs["Glazing_Area"] * floor
    opaque_wall = (designs["Wall_Area"] - window).clip(lower=0)
    return pd.DataFrame({
        "stories": stories,
        "ref_floor_m2": floor,
        "wall_per_m2": opaque_wall / floor,
        "window_per_m2": window / floor,
        "roof_per_m2": designs["Roof_Area"] / floor,
        "foundation_per_m2": designs["Roof_Area"] / floor,
        "upper_floor_per_m2": (stories - 1) / stories,
    }, index=designs.index)


def envelope_cost(designs: pd.DataFrame, a: Assumptions) -> pd.Series:
    """Envelope construction cost ($) for each design, scaled to the project's floor area."""
    g = geometry(designs)
    per_m2 = (g.wall_per_m2 * a.wall_cost + g.window_per_m2 * a.window_cost
              + g.roof_per_m2 * a.roof_cost + g.foundation_per_m2 * a.foundation_cost
              + g.upper_floor_per_m2 * a.upper_floor_cost)
    return per_m2 * a.floor_area_m2


# ------------------------------------------------------------------------------------------- energy & CO2
def annual_energy(pred: pd.DataFrame, a: Assumptions) -> pd.DataFrame:
    """Annual energy cost ($) and CO2 (tonnes) from predicted loads (kWh/m² per year).

    `energy_cost` includes the optional carbon price so that rankings can value CO2 when the user asks.
    """
    heat_kwh = pred["Heating_Load"] * a.floor_area_m2
    cool_kwh = pred["Cooling_Load"] * a.floor_area_m2
    co2_t = (heat_kwh * a.heat_co2_per_kwh + cool_kwh * a.cool_co2_per_kwh) / 1000
    heat_cost = heat_kwh * a.heat_price_per_kwh
    cool_cost = cool_kwh * a.cool_price_per_kwh
    return pd.DataFrame({
        "heat_cost": heat_cost,
        "cool_cost": cool_cost,
        "carbon_cost": co2_t * a.carbon_price,
        "energy_cost": heat_cost + cool_cost + co2_t * a.carbon_price,
        "co2_t": co2_t,
    }, index=pred.index)


def money_uncertainty(a: Assumptions, gap: dict | None) -> float:
    """Dollar-per-year uncertainty of a *difference* between two designs, from the model's typical gap error.

    Adds the heating and cooling gaps (conservative). Returns 0 when the bundle has no ranges.
    """
    if not gap:
        return 0.0
    heat = a.heat_price_per_kwh + a.carbon_price * a.heat_co2_per_kwh / 1000
    cool = a.cool_price_per_kwh + a.carbon_price * a.cool_co2_per_kwh / 1000
    return a.floor_area_m2 * (gap["Heating_Load"] * heat + gap["Cooling_Load"] * cool)


def present_value_factor(a: Assumptions) -> float:
    """PV of $1/year of savings over the horizon, growing with energy escalation, discounted at the discount rate."""
    t = np.arange(1, a.horizon_years + 1)
    return float(np.sum((1 + a.energy_escalation) ** (t - 1) / (1 + a.discount_rate) ** t))


def business_case(designs: pd.DataFrame, pred: pd.DataFrame, a: Assumptions, baseline: str | int,
                  gap: dict | None = None) -> pd.DataFrame:
    """Compare every design to a baseline: extra construction cost, energy savings, payback, NPV, CO2.

    `designs` and `pred` share an index; `baseline` is the index label of the reference design.
    Positive savings = cheaper to run than the baseline. NPV > 0 = the design pays back within the horizon.
    """
    cap = envelope_cost(designs, a)
    en = annual_energy(pred, a)
    out = pd.concat([pred[["Heating_Load", "Cooling_Load", "Total_Load"]], en], axis=1)
    out["construction_cost"] = cap
    out["extra_construction"] = cap - cap.loc[baseline]
    out["annual_savings"] = en.energy_cost.loc[baseline] - en.energy_cost
    out["co2_saved_t_per_yr"] = en.co2_t.loc[baseline] - en.co2_t
    pvf = present_value_factor(a)
    out["npv"] = out.annual_savings * pvf - out.extra_construction
    # 0 = cheaper to build and to run; inf = costs more to build and to run;
    # nan = cheaper to build but costs more to run (payback does not apply)
    out["payback_years"] = [
        0.0 if (x <= 0 and s > 0) else (x / s if s > 0 else (np.nan if x <= 0 else np.inf))
        for x, s in zip(out.extra_construction, out.annual_savings)
    ]
    u = money_uncertainty(a, gap)
    out["savings_low"] = out.annual_savings - u
    out["savings_high"] = out.annual_savings + u
    out["npv_low"] = out.savings_low * pvf - out.extra_construction
    out["npv_high"] = out.savings_high * pvf - out.extra_construction
    if gap:
        out["within_error"] = (pred["Total_Load"] - pred.loc[baseline, "Total_Load"]).abs() <= gap["Total_Load"]
    else:
        out["within_error"] = False
    out.loc[baseline, "within_error"] = False
    return out


def fmt_money(x: float) -> str:
    if not np.isfinite(x):
        return "n/a"
    sign = "-" if x < 0 else ""
    x = abs(x)
    if x >= 1e6:
        return f"{sign}${x / 1e6:.2f}M"
    if x >= 1e4:
        return f"{sign}${x / 1e3:.0f}k"
    return f"{sign}${x:,.0f}"


def fmt_payback(y: float) -> str:
    if y == 0:
        return "immediate (cheaper to build and to run)"
    if np.isnan(y):
        return "n/a (cheaper to build, costs more to run)"
    if not np.isfinite(y):
        return "never (costs more to build and to run)"
    return f"{y:.1f} years"
