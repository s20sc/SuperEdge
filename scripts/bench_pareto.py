#!/usr/bin/env python3
"""
Cost-accuracy operating-point benchmark for the §III-C Pareto analysis.

Backs Section III-C (Cost-Accuracy Operating-Point Analysis Against
Foundation-Model Boundary Extractors), Table VI, and Figure 4 (the TikZ
double-panel Pareto scatter) of the SuperEdge T-IM submission.

Reproduces the three-axis Pareto frontier:
    accuracy      -- ODS on BIPEDv2 test split
    throughput    -- FPS at 480 x 640 (median over warm GPU runs)
    footprint     -- model parameters (M)

Plus reports for each method:
    p50 / p95 latency in ms
    peak GPU memory (MiB)
    FLOPs (G)

The Pareto-domination criterion (paper §III-C):
    A point P dominates Q iff P is no worse on every axis AND strictly
    better on at least one. The script flags which methods are
    non-dominated.

Methods compared (paper Table VI):
    SuperEdge (ours)            -- 0.926 ODS / 67.6 FPS / 1.29 M
    SAM (ViT-H)                 -- 0.821 ODS /  0.5 FPS / 641 M
    SAM (ViT-L)
    SAM2
    MobileSAM                   -- 0.690 ODS / 11   FPS / 10  M
    FastSAM                     -- 0.726 ODS / 52   FPS / 68  M
    EdgeSAM

Usage
-----
    python scripts/bench_pareto.py \\
        --model superedge --checkpoint export/coco_val_v6_1.994_185.pth \\
        --resolution 480x640 --warmup 20 --iters 200 --device cuda \\
        --out results/pareto/superedge.json
    # ... repeat for each baseline checkpoint, then merge:
    python scripts/bench_pareto.py --merge results/pareto/*.json \\
        --ods-csv results/ods_table_vi.csv \\
        --out results/pareto.json

Dependencies
------------
    torch (>=1.13 for accurate CUDA event timing),
    numpy, pandas,
    fvcore (FLOPs)  OR  thop (fallback FLOPs)
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="mode", required=False)

    # Single-method profile mode
    g = p.add_argument_group("single-method profiling")
    g.add_argument("--model", type=str,
                   help="Method key (superedge, sam_vit_h, sam2, mobilesam, ...).")
    g.add_argument("--checkpoint", type=Path)
    g.add_argument("--resolution", default="480x640",
                   help="Input resolution H x W (paper: 480x640).")
    g.add_argument("--warmup", type=int, default=20)
    g.add_argument("--iters",  type=int, default=200,
                   help="Timed iterations (paper protocol: 200 warm runs).")
    g.add_argument("--device", default="cuda")

    # Merge mode
    g2 = p.add_argument_group("merge mode")
    g2.add_argument("--merge", nargs="+", type=Path,
                    help="Per-method JSON files to merge into a single Pareto JSON.")
    g2.add_argument("--ods-csv", type=Path,
                    help="CSV with method, ods columns (paper Table VI).")

    p.add_argument("--out", type=Path, required=True)
    return p.parse_args()


def load_method(model: str, checkpoint: Path, device: str):
    """
    Method-key -> (callable forward, parameter count).

    Implementations expected (extend as needed):
        superedge       -- torch.load(checkpoint), forward(image)
        sam_vit_h       -- official segment_anything package
        sam2            -- sam2 package
        mobilesam       -- MobileSAM package
        fastsam         -- FastSAM package
        edgesam         -- EdgeSAM package
    Returns (forward_fn, n_params).
    """
    raise NotImplementedError(
        "Wire each baseline through its official inference API. SuperEdge "
        "loads from torch.load(checkpoint); SAM-family loads via their "
        "official packages with automatic-mask-generation enabled (paper "
        "protocol — that's where the 0.5 FPS comes from)."
    )


def measure_latency(forward_fn, dummy_input,
                    warmup: int, iters: int) -> Dict[str, float]:
    """
    Warm GPU, then time iters runs using torch.cuda.Event.
    Returns p50 / p95 latency (ms) and median FPS.
    """
    import torch
    for _ in range(warmup):
        forward_fn(dummy_input)
    torch.cuda.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends   = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        starts[i].record()
        forward_fn(dummy_input)
        ends[i].record()
    torch.cuda.synchronize()

    ms = [s.elapsed_time(e) for s, e in zip(starts, ends)]
    ms_sorted = sorted(ms)
    p50 = ms_sorted[iters // 2]
    p95 = ms_sorted[int(0.95 * iters)]
    fps = 1000.0 / statistics.median(ms)
    return {"p50_ms": p50, "p95_ms": p95, "fps_median": fps}


def measure_flops(forward_fn, dummy_input) -> Optional[float]:
    """G FLOPs via fvcore (preferred) or thop. Returns None on failure."""
    try:
        from fvcore.nn import FlopCountAnalysis
        return float(FlopCountAnalysis(forward_fn, dummy_input).total()) / 1e9
    except Exception:
        try:
            from thop import profile
            macs, _ = profile(forward_fn, inputs=(dummy_input,))
            return float(macs * 2) / 1e9  # MACs -> FLOPs (factor of 2)
        except Exception as e:
            logger.warning("FLOP counting failed: %s", e)
            return None


def measure_peak_gpu_mem_mib(forward_fn, dummy_input) -> float:
    import torch
    torch.cuda.reset_peak_memory_stats()
    forward_fn(dummy_input)
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / (1024 * 1024)


def is_pareto_optimal(point: Dict[str, float],
                      others: List[Dict[str, float]]) -> bool:
    """
    Pareto-domination test on (ODS up, FPS up, params down).
    """
    for q in others:
        if q is point:
            continue
        # q dominates point iff: ODS >=, FPS >=, params <=, and strictly
        # better on at least one axis.
        no_worse = (q["ods"] >= point["ods"] and
                    q["fps"] >= point["fps"] and
                    q["params_M"] <= point["params_M"])
        strictly_better = (q["ods"] > point["ods"] or
                           q["fps"] > point["fps"] or
                           q["params_M"] < point["params_M"])
        if no_worse and strictly_better:
            return False
    return True


def main() -> None:
    args = parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    # Merge mode -------------------------------------------------------------
    if args.merge:
        records = []
        for p in args.merge:
            records.append(json.loads(p.read_text()))
        if args.ods_csv:
            import pandas as pd
            ods_df = pd.read_csv(args.ods_csv).set_index("method")
            for r in records:
                r["ods"] = float(ods_df.loc[r["method"], "ods"])
        for r in records:
            r["pareto_optimal"] = is_pareto_optimal(r, records)
        args.out.write_text(json.dumps(records, indent=2))
        logger.info("Merged %d methods into %s", len(records), args.out)
        for r in records:
            tag = "[PARETO]" if r["pareto_optimal"] else "        "
            logger.info("%s %-15s ODS=%.3f  FPS=%5.1f  params=%6.2f M",
                        tag, r["method"], r["ods"], r["fps"], r["params_M"])
        return

    # Single-method profile mode --------------------------------------------
    import torch

    h, w = (int(x) for x in args.resolution.lower().split("x"))
    dummy = torch.randn(1, 3, h, w, device=args.device)

    forward_fn, n_params = load_method(args.model, args.checkpoint, args.device)

    flops_g     = measure_flops(forward_fn, dummy)
    peak_mib    = measure_peak_gpu_mem_mib(forward_fn, dummy)
    latency     = measure_latency(forward_fn, dummy, args.warmup, args.iters)

    record = {
        "method":      args.model,
        "resolution":  args.resolution,
        "params_M":    n_params / 1e6,
        "flops_G":     flops_g,
        "peak_mem_MiB": peak_mib,
        "fps":         latency["fps_median"],
        **latency,
    }
    args.out.write_text(json.dumps(record, indent=2))
    logger.info("Wrote %s", args.out)


if __name__ == "__main__":
    main()
