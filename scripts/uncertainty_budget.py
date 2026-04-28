#!/usr/bin/env python3
"""
GUM Type-A / Type-B uncertainty budget for SuperEdge edge measurements.

Backs Section II-G (Metrological Traceability and Uncertainty Budget) and
Table II of the SuperEdge T-IM submission. Implements the structure of
ISO/IEC Guide 98-3 (JCGM 100:2008, "GUM 1995 with minor corrections"),
combining the dominant noise sources via root-sum-square.

The 6-row table (paper Section II-G):

    Source                          | Type | Distribution        | Estimate
    --------------------------------+------+---------------------+----------
    Bootstrap detector f(.) noise   | A    | Gaussian (1/sqrt N) | sigma_r = 4.4e-4
    Camera read-out noise           | A    | Additive Gaussian   | DeltaF <= 6.8%
    Inter-annotator bias            | B    | N/A                 | 0 (removed)
    Homography sampling bias        | B    | Bounded uniform     | traceable
    Illumination drift              | B    | Gamma sweep         | DeltaF <= 1.4%
    H/8 x W/8 quantization          | B    | Discrete            | run-invariant

Combined relative standard uncertainty (only the two characterized
disturbance sources are RSS-combined as independent contributors):

    u_combined = sqrt(1.4^2 + 6.8^2) % ~= 6.9 %

Paper reports 6.9% as the combined figure of merit (Section II-G).

Usage
-----
    python scripts/uncertainty_budget.py \\
        --repeatability results/repeatability.csv \\
        --robustness-gamma results/robustness_gamma.csv \\
        --robustness-noise results/robustness_noise.csv \\
        --out results/uncertainty_budget.csv

This script consumes the CSVs emitted by repeatability_sweep.py and
robustness_sweep.py to verify the Type-A entries, and prints a markdown
table that matches paper Table II.

Dependencies
------------
    pandas, numpy
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ----- GUM table rows (paper Table II) ---------------------------------------

@dataclass
class UncertaintySource:
    name: str
    type: str               # "A" or "B"
    distribution: str
    estimate: str           # human-readable; verbatim from paper
    notes: str
    numeric_pct: Optional[float] = None  # for the RSS combiner


def build_budget(
    sigma_r_at_nh100: float,
    delta_f_noise_pct: float,
    delta_f_gamma_pct: float,
) -> List[UncertaintySource]:
    """
    Construct the 6-row GUM Type-A/B table from per-experiment numbers.

    Numbers fed in:
        sigma_r_at_nh100   -- from repeatability.csv  (paper: 4.4e-4)
        delta_f_noise_pct  -- from robustness_noise.csv  (paper: 6.8)
        delta_f_gamma_pct  -- from robustness_gamma.csv  (paper: 1.4)
    """
    return [
        UncertaintySource(
            name="Bootstrap detector $f(\\cdot)$ noise",
            type="A",
            distribution="Gaussian ($1/\\sqrt{N_h}$)",
            estimate=f"$\\sigma_r = {sigma_r_at_nh100:.1e}$ at $N_h{{=}}100$",
            notes="Eq. (1); §III-D",
            numeric_pct=None,  # not in same units as F-measure %
        ),
        UncertaintySource(
            name="Camera read-out noise",
            type="A",
            distribution="Additive Gaussian ($\\sigma^2$)",
            estimate=f"$\\Delta F \\le {delta_f_noise_pct:.1f}\\%$ on $\\sigma^2 \\in [0, 0.05]$",
            notes="§III-E",
            numeric_pct=delta_f_noise_pct,
        ),
        UncertaintySource(
            name="Inter-annotator bias",
            type="B",
            distribution="N/A",
            estimate="0 (removed by design)",
            notes="analytical synthetic ground truth",
            numeric_pct=None,
        ),
        UncertaintySource(
            name="Homography sampling bias",
            type="B",
            distribution="Bounded uniform",
            estimate="traceable to declared ranges",
            notes="rotation $\\pm\\pi/2$, scale 0.8--1.2, perspective 0.2, patch 0.85",
            numeric_pct=None,
        ),
        UncertaintySource(
            name="Illumination drift",
            type="B",
            distribution="Gamma sweep ($[0.4, 2.0]$)",
            estimate=f"$\\Delta F \\le {delta_f_gamma_pct:.1f}\\%$",
            notes="§III-E",
            numeric_pct=delta_f_gamma_pct,
        ),
        UncertaintySource(
            name="$H/8 \\times W/8$ quantization",
            type="B",
            distribution="Discrete",
            estimate="deterministic, run-invariant",
            notes="not in repeatability indicator",
            numeric_pct=None,
        ),
    ]


def combine_rss(sources: List[UncertaintySource]) -> float:
    """
    Root-sum-square the characterized contributors that have a numeric_pct.

    Paper Section II-G: independent RSS of (gamma drift, sigma^2 noise) ->
        sqrt(1.4^2 + 6.8^2)% ~= 6.94%
    """
    contributions = [s.numeric_pct for s in sources if s.numeric_pct is not None]
    return float(np.sqrt(np.sum(np.square(contributions))))


def format_markdown(sources: List[UncertaintySource], combined: float) -> str:
    head = ("| Source | Type | Distribution | Estimate | Notes |\n"
            "|---|---|---|---|---|\n")
    body = "\n".join(
        f"| {s.name} | {s.type} | {s.distribution} | {s.estimate} | {s.notes} |"
        for s in sources
    )
    tail = f"\n\n**Combined relative standard uncertainty (RSS):** {combined:.2f}%"
    return head + body + tail


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repeatability",      type=Path, required=True)
    p.add_argument("--robustness-gamma",   type=Path, required=True)
    p.add_argument("--robustness-noise",   type=Path, required=True)
    p.add_argument("--out",                type=Path,
                   default=Path("results/uncertainty_budget.csv"))
    p.add_argument("--out-md",             type=Path,
                   default=Path("results/uncertainty_budget.md"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    import pandas as pd

    rep = pd.read_csv(args.repeatability)
    sigma_r = float(rep.loc[rep["N_h"] == 100, "sigma_r_mean"].iloc[0])

    g = pd.read_csv(args.robustness_gamma)
    delta_g = float(g.loc[g["method"] == "SuperEdge_fused", "Delta_max_pct"].iloc[0])

    n = pd.read_csv(args.robustness_noise)
    delta_n = float(n.loc[n["method"] == "SuperEdge_fused", "Delta_max_pct"].iloc[0])

    sources = build_budget(sigma_r, delta_n, delta_g)
    combined = combine_rss(sources)

    # CSV (machine-readable)
    df = pd.DataFrame([{
        "Source":       s.name.replace("$", "").replace("\\", ""),
        "Type":         s.type,
        "Distribution": s.distribution.replace("$", "").replace("\\", ""),
        "Estimate":     s.estimate.replace("$", "").replace("\\", ""),
        "Notes":        s.notes.replace("$", "").replace("\\", ""),
        "numeric_pct":  s.numeric_pct,
    } for s in sources])
    df.to_csv(args.out, index=False)
    logger.info("Wrote %s", args.out)

    # Markdown (human-readable, paste-able into supplementary)
    args.out_md.write_text(format_markdown(sources, combined))
    logger.info("Wrote %s", args.out_md)
    logger.info("Combined RSS uncertainty: %.2f%% (paper claims 6.9%%)", combined)


if __name__ == "__main__":
    main()
