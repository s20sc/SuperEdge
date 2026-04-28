#!/usr/bin/env python3
"""
McNemar's test on the n=50 paired grasp trials.

Backs Section III-F paragraph "6-DoF robotic arm" of the SuperEdge T-IM
submission. Reproduces:

    n = 50 paired trials
    baseline (YOLO only)         : 39 / 50  (78%)
    YOLO + SuperEdge edge-refine : 45 / 50  (90%)
    discordant counts            : b = 6, c = 0
    concordant counts            : 39 (both succeed) + 5 (both fail) = 44
    chi^2 (Yates' continuity)    : (|b - c| - 1)^2 / (b + c) = 25/6 ~= 4.17
    p-value (two-sided)          : 0.041
    significant at alpha = 0.05  : YES

Reads the released paired-trial log (one row per trial) and emits:
    - the 2x2 contingency table
    - chi^2 with Yates' continuity correction
    - exact-binomial p-value as a cross-check (preferred when b+c is small)
    - the same numbers in markdown for the supplementary

Usage
-----
    python scripts/mcnemar_grasp.py \\
        --trials data/grasp_trials/trials.csv \\
        --out    results/mcnemar.json

Trial CSV schema (each row = one paired trial)
    trial_id        : int    -- 1..n
    scene           : str    -- short scene descriptor
    object          : str    -- object identifier
    baseline_success: int    -- 0/1, YOLO-only attempt
    superedge_success: int   -- 0/1, YOLO + SuperEdge edge-refine attempt

Dependencies
------------
    pandas, statsmodels (statsmodels.stats.contingency_tables.mcnemar),
    scipy>=1.7 (binom_test as exact cross-check).
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--trials", type=Path, required=True,
                   help="CSV with paired-trial outcomes (schema in module docstring).")
    p.add_argument("--out", type=Path, default=Path("results/mcnemar.json"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    import pandas as pd
    from scipy import stats
    from statsmodels.stats.contingency_tables import mcnemar

    df = pd.read_csv(args.trials, comment="#", skip_blank_lines=True)
    n = len(df)

    bs = df["baseline_success"].astype(int).to_numpy()
    se = df["superedge_success"].astype(int).to_numpy()

    # 2x2 contingency table for McNemar:
    #                 SuperEdge=success | SuperEdge=fail
    #   baseline=succ        a                   c
    #   baseline=fail        b                   d
    a = int(((bs == 1) & (se == 1)).sum())  # both succeed (concordant)
    b = int(((bs == 0) & (se == 1)).sum())  # baseline fail, superedge success
    c = int(((bs == 1) & (se == 0)).sum())  # baseline success, superedge fail
    d = int(((bs == 0) & (se == 0)).sum())  # both fail (concordant)

    # McNemar with Yates' continuity correction (preferred when b+c >= 10)
    table = np.array([[a, c], [b, d]], dtype=int)
    res_yates = mcnemar(table, exact=False, correction=True)
    chi2_yates = float(res_yates.statistic)
    p_yates    = float(res_yates.pvalue)

    # Exact binomial test (preferred when b+c is small, paper has b+c=6)
    res_exact  = mcnemar(table, exact=True)
    p_exact    = float(res_exact.pvalue)

    record = {
        "n_trials": n,
        "baseline_success": int(bs.sum()),
        "baseline_pct":     float(bs.mean()) * 100.0,
        "superedge_success": int(se.sum()),
        "superedge_pct":     float(se.mean()) * 100.0,
        "concordant_both_success": a,
        "concordant_both_fail":    d,
        "discordant_b_baseline_fail_only":   b,
        "discordant_c_baseline_success_only": c,
        "chi_squared_yates":  chi2_yates,
        "p_value_yates":      p_yates,
        "p_value_exact":      p_exact,
        "significant_alpha_0_05": bool(p_yates < 0.05),
    }
    args.out.write_text(json.dumps(record, indent=2))
    logger.info("Wrote %s", args.out)
    logger.info("n=%d  baseline %d/%d (%.0f%%)  ours %d/%d (%.0f%%)",
                n, bs.sum(), n, record["baseline_pct"],
                se.sum(), n, record["superedge_pct"])
    logger.info("b=%d  c=%d  concordant=%d", b, c, a + d)
    logger.info("McNemar (Yates):  chi^2 = %.2f, p = %.3f", chi2_yates, p_yates)
    logger.info("McNemar (exact):  p = %.3f", p_exact)
    if p_yates < 0.05:
        logger.info("Significant at alpha = 0.05")
    else:
        logger.info("NOT significant at alpha = 0.05")


if __name__ == "__main__":
    main()
