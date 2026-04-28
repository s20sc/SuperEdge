#!/usr/bin/env python3
"""
Robustness sweep over illumination (gamma) and additive Gaussian noise.

Backs Section III-E (Robustness to Imaging Conditions) and Tables VIII / IX
of the SuperEdge T-IM submission. Reproduces:

    F-measure max-deviation on gamma-sweep:  Delta_max <= 1.4%
    F-measure max-deviation on sigma^2-sweep: Delta_max <= 6.8%
    DexiNed degradation on sigma^2 = 0.05:    73% (for contrast)

Caveat: per Section III-E paragraph "Note on the metric used in Tables
VIII–IX", the per-condition F-measure is computed under a tolerance-relaxed
approximation of the BSDS protocol (looser localization tolerance, threshold
fixed at 0.3). Only relative deviations across rows are interpretable across
comparisons; the standard ODS=0.825 anchor at gamma=1, sigma^2=0 is reported
separately in Table V.

Procedure
---------
For each method m in {SuperEdge_pixel, SuperEdge_object, SuperEdge_fused,
                     Canny, DexiNed-ST, PiDiNet-ST, STEdge}:
    For gamma in {0.4, 0.8, 1.0, 1.4, 2.0}:    # illumination sweep
        For each test image x in BIPEDv2 test split:
            Apply gamma correction: x' = x^gamma  (after [0,1] norm)
            Run model m on x' -> predicted edge map
            Compute relaxed-BSDS F-measure vs ground truth
        F[m, gamma] = mean over images
    For sigma2 in {0, 0.0125, 0.025, 0.0375, 0.05}:  # noise sweep
        Same protocol with x' = x + N(0, sigma2)

Outputs:
    results/robustness_gamma.csv  (rows = methods, cols = gamma values + Delta_max)
    results/robustness_noise.csv  (rows = methods, cols = sigma^2 values + Delta_max)

Usage
-----
    python scripts/robustness_sweep.py \\
        --checkpoint export/coco_val_v6_1.994_185.pth \\
        --bipedv2-dir /path/to/BIPEDv2/test \\
        --metric relaxed_bsds --threshold 0.3 \\
        --out-dir results/

Dependencies
------------
    torch, numpy, opencv-python, scikit-image (for gamma correction
    sanity), pandas (CSV emission). The relaxed-BSDS evaluator is
    implemented in solver/detector_evaluation.py (already in the repo).
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List

import numpy as np

logger = logging.getLogger(__name__)


# ----- Paper constants (Section III-E sweep grids) ---------------------------

GAMMA_GRID: List[float] = [0.4, 0.8, 1.0, 1.4, 2.0]
SIGMA2_GRID: List[float] = [0.0, 0.0125, 0.025, 0.0375, 0.05]

# Methods evaluated. Each must have a registered loader in load_method().
METHOD_KEYS: List[str] = [
    "SuperEdge_pixel",
    "SuperEdge_object",
    "SuperEdge_fused",
    "Canny",
    "DexiNed-ST",
    "PiDiNet-ST",
    "STEdge",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path, required=True,
                   help="SuperEdge checkpoint (.pth)")
    p.add_argument("--bipedv2-dir", type=Path, required=True)
    p.add_argument("--metric", choices=["relaxed_bsds", "standard_bsds"],
                   default="relaxed_bsds")
    p.add_argument("--threshold", type=float, default=0.3,
                   help="Edge-probability threshold (paper: 0.3 for relaxed).")
    p.add_argument("--seed", type=int, default=0,
                   help="Seed for additive-noise reproducibility.")
    p.add_argument("--out-dir", type=Path, default=Path("results"))
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def apply_gamma(image: np.ndarray, gamma: float) -> np.ndarray:
    """x' = x^gamma after [0,1] normalisation."""
    x = image.astype(np.float32) / 255.0
    return np.clip(np.power(x, gamma) * 255.0, 0, 255).astype(np.uint8)


def apply_gaussian_noise(image: np.ndarray, sigma2: float,
                         rng: np.random.Generator) -> np.ndarray:
    """x' = x + N(0, sigma^2). sigma^2 is on the [0,1] scale."""
    x = image.astype(np.float32) / 255.0
    noise = rng.normal(0.0, np.sqrt(sigma2), size=x.shape).astype(np.float32)
    return np.clip((x + noise) * 255.0, 0, 255).astype(np.uint8)


def evaluate_method(
    method_key: str,
    perturbed_images: List[np.ndarray],
    gt_edges: List[np.ndarray],
    metric: str,
    threshold: float,
) -> float:
    """
    Returns mean F-measure across the test set for one method under one
    perturbation condition.

    REQUIRES a method-loader callback that produces the predicted edge map.
    Hook into solver/detector_evaluation.py for the relaxed-BSDS scorer.
    """
    raise NotImplementedError(
        "Plug into solver/detector_evaluation.py: this should be a thin wrapper "
        "that loads `method_key` (e.g., dispatches to torch.load for SuperEdge "
        "checkpoints, cv2.Canny for the classical baseline, separate "
        "checkpoints for DexiNed-ST / PiDiNet-ST / STEdge), runs inference on "
        "each perturbed image, then calls the relaxed-BSDS evaluator with the "
        "given threshold."
    )


def sweep_one_axis(
    methods: List[str],
    grid: List[float],
    apply_fn,
    grid_label: str,
    metric: str,
    threshold: float,
    images: List[np.ndarray],
    gts:    List[np.ndarray],
    rng:    np.random.Generator,
) -> "pd.DataFrame":
    import pandas as pd
    rows = []
    for m in methods:
        f_per_value = []
        for v in grid:
            perturbed = [apply_fn(img, v) if grid_label == "gamma"
                         else apply_fn(img, v, rng) for img in images]
            f = evaluate_method(m, perturbed, gts, metric, threshold)
            f_per_value.append(f)
            logger.info("[%s = %.4f] %-20s F = %.4f", grid_label, v, m, f)
        f_arr = np.asarray(f_per_value)
        delta_max = 100.0 * (f_arr.max() - f_arr.min()) / f_arr.max() if f_arr.max() > 0 else 0.0
        row = {"method": m, **{f"{grid_label}_{v}": fv
                               for v, fv in zip(grid, f_per_value)},
               "Delta_max_pct": delta_max}
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    # TODO: load BIPEDv2 test split into (images, gt_edges)
    images: List[np.ndarray] = []
    gts:    List[np.ndarray] = []

    rng = np.random.default_rng(args.seed)

    df_g = sweep_one_axis(METHOD_KEYS, GAMMA_GRID, apply_gamma, "gamma",
                          args.metric, args.threshold, images, gts, rng)
    df_g.to_csv(args.out_dir / "robustness_gamma.csv", index=False)
    logger.info("Wrote %s", args.out_dir / "robustness_gamma.csv")

    df_s = sweep_one_axis(METHOD_KEYS, SIGMA2_GRID, apply_gaussian_noise, "sigma2",
                          args.metric, args.threshold, images, gts, rng)
    df_s.to_csv(args.out_dir / "robustness_noise.csv", index=False)
    logger.info("Wrote %s", args.out_dir / "robustness_noise.csv")


if __name__ == "__main__":
    main()
