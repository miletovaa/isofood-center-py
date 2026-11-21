"""A synthetic (made-up) honey-provenance dataset, so the UI and docs have
something to run on without a database connection. Not real measurements."""

from __future__ import annotations

import numpy as np
import pandas as pd


def example_dataset(seed: int = 7) -> pd.DataFrame:
    """60 jars × 3 regions × 2 harvest years, measured fresh and after
    6 months: stable isotopes (δ13C, δ15N, δ2H, δ18O), elements (K, Ca, Mg,
    Fe, Zn, Mn, Sr in mg/kg) and a presence/absence pollen marker.

    Long in time (one row per jar × storage time) so the paired tests have
    something to work with; filter `storage == "fresh"` for one row per jar.
    """
    rng = np.random.default_rng(seed)
    regions = {
        "Coastal":  dict(d13c=-25.2, d15n=1.5, d2h=-95, d18o=-3.0, K=900,  Ca=80,  Mg=35, Fe=3.0, Zn=1.5, Mn=1.2, Sr=0.9, pollen=0.8),
        "Pannonian": dict(d13c=-26.1, d15n=2.6, d2h=-110, d18o=-5.0, K=1300, Ca=60, Mg=25, Fe=4.5, Zn=2.2, Mn=2.8, Sr=0.4, pollen=0.4),
        "Alpine":   dict(d13c=-26.4, d15n=0.8, d2h=-125, d18o=-7.5, K=700,  Ca=110, Mg=45, Fe=2.0, Zn=1.0, Mn=4.0, Sr=0.6, pollen=0.2),
    }
    rows = []
    jar = 0
    for region, mu in regions.items():
        for year in (2024, 2025):
            for _ in range(10):
                jar += 1
                base = {
                    "d13c": rng.normal(mu["d13c"], 0.6),
                    "d15n": rng.normal(mu["d15n"], 0.6),
                    "d2h": rng.normal(mu["d2h"] + (4 if year == 2025 else 0), 12),
                    "d18o": rng.normal(mu["d18o"] + (0.5 if year == 2025 else 0), 1.8),
                    **{el: float(rng.lognormal(np.log(mu[el]), 0.45)) for el in ("K", "Ca", "Mg", "Fe", "Zn", "Mn", "Sr")},
                }
                base["Zn"] += 0.4 * (base["Fe"] - mu["Fe"])  # Fe and Zn co-vary
                pollen = "present" if rng.random() < mu["pollen"] else "absent"
                for storage, shift in (("fresh", 0.0), ("6 months", 1.0)):
                    # every re-measurement carries a little analytical noise (~1-2 %)
                    measured = {k: v + rng.normal(0, 0.05) if k.startswith("d") else v * rng.lognormal(0, 0.015)
                                for k, v in base.items()}
                    row = {"sample_id": f"J{jar:03d}", "region": region, "harvest_year": year,
                           "storage": storage, "pollen_marker": pollen, **measured}
                    # storage drifts d18O slightly and lowers Fe (oxidation/precipitation, illustrative only)
                    row["d18o"] = base["d18o"] + shift * 0.35 + rng.normal(0, 0.1)
                    row["Fe"] = base["Fe"] * (1 - 0.08 * shift) * rng.lognormal(0, 0.03)
                    rows.append(row)
    df = pd.DataFrame(rows)
    df.loc[rng.choice(len(df), 6, replace=False), "Mn"] = np.nan  # a few missing values, like real data
    return df
