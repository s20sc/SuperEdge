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


def _auto_mask_forward(generator):
    """Adapter: torch [1, C, H, W] tensor -> numpy uint8 RGB -> AMG output.

    All SAM-family baselines compare under automatic-mask-generation per the
    paper protocol (§III-C); the cost of the dense prompt grid is what
    drives 0.5 FPS for SAM-ViT-H.
    """
    def forward(x):
        import numpy as _np
        arr = x.detach().clamp(0, 1).cpu().numpy()[0]
        arr = (arr.transpose(1, 2, 0) * 255).astype(_np.uint8)
        if arr.shape[2] == 1:
            arr = _np.repeat(arr, 3, axis=2)
        return generator.generate(arr)
    return forward


def _superedge_forward(net):
    def forward(x):
        import torch
        with torch.no_grad():
            if x.shape[1] == 3:
                x = x.mean(dim=1, keepdim=True)
            return net(x)
    return forward


def load_method(model: str, checkpoint: Path, device: str):
    """Method-key -> (callable forward, parameter count).

    SuperEdge loads via :mod:`model.superedge`. SAM-family baselines load
    through their official packages with automatic-mask-generation
    enabled — the paper protocol — so the timed forward includes the dense
    prompt grid that drives the 0.5 FPS figure for SAM-ViT-H.
    """
    import torch

    key = model.lower()
    ckpt_str = str(checkpoint) if checkpoint is not None else None

    if key == "superedge":
        from model.superedge import SuperEdge
        state = torch.load(checkpoint, map_location=device)
        if hasattr(state, "eval"):
            net = state.to(device).eval()
        else:
            cfg = {"name": "superedge", "using_bn": True}
            net = SuperEdge(cfg, device=device, using_bn=True)
            net.load_state_dict(state)
            net = net.to(device).eval()
        return _superedge_forward(net), sum(p.numel() for p in net.parameters())

    if key in ("sam_vit_h", "sam_vit_l", "sam_vit_b"):
        try:
            from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
        except ImportError as e:
            raise RuntimeError(
                "Install Meta SAM: pip install git+https://github.com/facebookresearch/segment-anything"
            ) from e
        key_map = {"sam_vit_h": "vit_h", "sam_vit_l": "vit_l", "sam_vit_b": "vit_b"}
        sam = sam_model_registry[key_map[key]](checkpoint=ckpt_str).to(device).eval()
        return (_auto_mask_forward(SamAutomaticMaskGenerator(sam)),
                sum(p.numel() for p in sam.parameters()))

    if key == "sam2":
        try:
            from sam2.build_sam import build_sam2
            from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
        except ImportError as e:
            raise RuntimeError("Install SAM2: pip install sam2") from e
        cfg_path = checkpoint.with_suffix(".yaml")
        if not cfg_path.exists():
            raise FileNotFoundError(
                f"SAM2 config not found at {cfg_path}; place the matching "
                f".yaml next to the checkpoint or pass via SAM2_CFG env var.")
        sam2 = build_sam2(str(cfg_path), ckpt_str, device=device).eval()
        return (_auto_mask_forward(SAM2AutomaticMaskGenerator(sam2)),
                sum(p.numel() for p in sam2.parameters()))

    if key == "mobilesam":
        try:
            from mobile_sam import sam_model_registry, SamAutomaticMaskGenerator
        except ImportError as e:
            raise RuntimeError("Install MobileSAM: pip install mobile-sam") from e
        sam = sam_model_registry["vit_t"](checkpoint=ckpt_str).to(device).eval()
        return (_auto_mask_forward(SamAutomaticMaskGenerator(sam)),
                sum(p.numel() for p in sam.parameters()))

    if key == "fastsam":
        try:
            from ultralytics import FastSAM
        except ImportError as e:
            raise RuntimeError("Install ultralytics: pip install ultralytics") from e
        net = FastSAM(ckpt_str)
        n_params = sum(p.numel() for p in net.model.parameters()) \
            if hasattr(net, "model") else 0
        def forward(x):
            import numpy as _np
            arr = x.detach().clamp(0, 1).cpu().numpy()[0]
            arr = (arr.transpose(1, 2, 0) * 255).astype(_np.uint8)
            if arr.shape[2] == 1:
                arr = _np.repeat(arr, 3, axis=2)
            return net(arr, device=device, retina_masks=True, verbose=False)
        return forward, n_params

    if key == "edgesam":
        try:
            from edge_sam import sam_model_registry, SamAutomaticMaskGenerator
        except ImportError as e:
            raise RuntimeError("Install EdgeSAM: pip install edge-sam") from e
        sam = sam_model_registry["edge_sam"](checkpoint=ckpt_str).to(device).eval()
        return (_auto_mask_forward(SamAutomaticMaskGenerator(sam)),
                sum(p.numel() for p in sam.parameters()))

    raise ValueError(f"Unknown method key: '{model}'. Supported: "
                     "superedge, sam_vit_h, sam_vit_l, sam_vit_b, sam2, "
                     "mobilesam, fastsam, edgesam.")


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
