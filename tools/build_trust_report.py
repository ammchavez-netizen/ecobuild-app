"""Recompute EcoBuild's calibrated prediction ranges and the recommendation backtest.

    python tools/build_trust_report.py            # about 6 minutes on a laptop

Writes metadata["ranges"] and metadata["backtest"] in models/metadata.json. The saved models are not changed.
See ecobuild_trust.py for the method.
"""
import json
import sys
import time
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

from ecobuild_core import EcoBuildModels  # noqa: E402
from ecobuild_trust import backtest_recommendations, calibrate_ranges  # noqa: E402

META = ROOT / "models" / "metadata.json"


def main():
    t0 = time.time()
    models = EcoBuildModels(ROOT / "models")
    if any(s != "loaded" for s in models.status.values()):
        sys.exit(f"Saved models did not load ({models.status}). Install the versions in metadata['versions'] first.")

    print("1/3 Calibrating 90% ranges (one-story / two-story) and checking them on unseen designs…")
    ranges = calibrate_ranges(models, level=0.9, n_repeats=30)
    models.meta["ranges"] = ranges          # the backtest uses the new ranges for its "too close to call" flags

    print("2/3 Backtesting recommendations with models that never saw the design…")
    by_design = backtest_recommendations(models, by="design")
    print("3/3 Stress test: models that never saw the building form…")
    by_form = backtest_recommendations(models, by="form")

    meta = json.loads(META.read_text())
    meta["ranges"] = ranges
    meta["backtest"] = {"created": time.strftime("%Y-%m-%d"), "by_design": by_design, "by_form": by_form}
    META.write_text(json.dumps(meta, indent=1))

    s, g = by_design["suggest"], by_design["agent"]["Lowest energy use"]
    print(f"\nDone in {time.time() - t0:.0f}s. Highlights (model never saw the design):")
    print(f"  Suggest improvements: {s['all']['saves_energy']:.0%} of suggestions truly save energy; "
          f"{s['certain']['saves_energy']:.0%} of the ones marked clear, "
          f"{s['uncertain']['saves_energy']:.0%} of the ones marked too close to call.")
    print(f"  Design agent (lowest energy): {g['beats_baseline']:.0%} of picks truly beat the baseline; "
          f"mean regret {g['mean_regret']:.2f} kWh/m².")


if __name__ == "__main__":
    main()
