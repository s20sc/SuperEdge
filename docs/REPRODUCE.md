# Reproducing the T-IM Submission

This document walks a reviewer (or future you) from `git clone` to each
numerical artifact in the SuperEdge T-IM submission, in the order the paper
presents them.

All scripts use pinned seeds and pinned data manifests. CSV/JSON outputs are
emitted under `results/`. Expected runtimes assume an NVIDIA RTX 5090 at 480x640
input resolution unless stated otherwise.

## 0. Setup (one-time)

```bash
git clone https://github.com/s20sc/SuperEdge.git
cd SuperEdge

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # training stack
pip install -r requirements-eval.txt     # measurement-systems stack

# Download the BIPEDv2-trained checkpoint
mkdir -p export
wget https://github.com/s20sc/SuperEdge/releases/download/v1.0/superedge_bipedv2.pth \
     -O export/superedge_bipedv2.pth
```

## 1. Tables I and III–V (accuracy: ODS / OIS / AP)

These come straight from the existing `evaluate.py` and don't need new code.

```bash
python evaluate.py --config config/superedge_train.yaml \
                   --weights export/superedge_bipedv2.pth \
                   --datasets BIPED BIPEDv2 BSDS500 NYUD BSDS-RIND \
                   --out results/accuracy.csv
```

Cross-check against paper:
- BIPEDv2 ODS = 0.926 (Table V row "SuperEdge (ours)")
- BIPED ODS = 0.811 (Table III row "+pix +obj +postprocess")
- BSDS-RIND ODS deltas vs supervised baselines per Table V

**Expected runtime**: ~10 minutes per dataset.

## 2. Table II—GUM Type-A / Type-B Uncertainty Budget (§II-G)

This script consumes the outputs of steps 4 and 5 below, so run those first.

```bash
python scripts/uncertainty_budget.py \
    --repeatability      results/repeatability.csv \
    --robustness-gamma   results/robustness_gamma.csv \
    --robustness-noise   results/robustness_noise.csv \
    --out                results/uncertainty_budget.csv \
    --out-md             results/uncertainty_budget.md
```

Cross-check against paper Section II-G:
- Bootstrap detector noise: $\sigma_r = 4.4 \times 10^{-4}$ at $N_h = 100$
- Camera read-out noise: $\Delta F \le 6.8\%$
- Illumination drift: $\Delta F \le 1.4\%$
- **Combined RSS: 6.9%**

## 3. Table VI and Figure 4—Cost–accuracy Pareto Analysis (§III-C)

Profile each method individually, then merge into a Pareto report.

```bash
# Per-method profiles (one JSON per method)
python scripts/bench_pareto.py --model superedge \
       --checkpoint export/superedge_bipedv2.pth \
       --resolution 480x640 --warmup 20 --iters 200 \
       --out results/pareto/superedge.json

python scripts/bench_pareto.py --model sam_vit_h \
       --checkpoint export/baselines/sam_vit_h.pth \
       --resolution 480x640 --warmup 20 --iters 200 \
       --out results/pareto/sam_vit_h.json
# ... repeat for sam2, mobilesam, fastsam, edgesam

# Merge into a single Pareto report (and flag non-dominated points)
python scripts/bench_pareto.py \
       --merge results/pareto/*.json \
       --ods-csv results/accuracy_table_vi.csv \
       --out results/pareto.json
```

Cross-check Table VI:
- SuperEdge: 0.926 ODS / 67.6 FPS / 1.29 M params
- SAM (ViT-H): 0.821 / 0.5 / 641 (the only foundation model that loses on **all
  three** axes)
- FastSAM: only baseline that retains a niche on the FPS axis (and SuperEdge
  is still faster)

**Expected runtime**: ~5 minutes per method (200 warm GPU iterations).

## 4. Table VII—Repeatability with Bootstrap-1000 95% CIs (§III-D)

```bash
python scripts/repeatability_sweep.py \
    --checkpoint export/superedge_bipedv2.pth \
    --manifest   data/manifests/coco_repeatability_12.txt \
    --coco-dir   /path/to/COCO/val2017 \
    --K 50 --bootstrap 1000 --seed 0 \
    --out results/repeatability.csv
```

Cross-check against paper Table VII:
- $N_h = 100$: mean $\sigma_r = 0.000437$, CI = [0.000411, 0.000461]
- log-log slope = $-0.505$ (theoretical $1/\sqrt{N_h}$ predicts $-0.5$)
- $R^2 = 0.99998$

**Expected runtime**: ~45 minutes (7 N_h levels × 50 trials × 12 scenes).

The 12 pinned COCO scenes are listed (by image id) in
`data/manifests/coco_repeatability_12.txt`.

## 5. Tables VIII and IX—Robustness Sweeps (§III-E)

```bash
python scripts/robustness_sweep.py \
    --checkpoint    export/superedge_bipedv2.pth \
    --bipedv2-dir   /path/to/BIPEDv2/test \
    --metric        relaxed_bsds --threshold 0.3 \
    --seed 0 \
    --out-dir       results/
```

Outputs:
- `results/robustness_gamma.csv` (Table VIII): per-method F-measure across
  $\gamma \in \{0.4, 0.8, 1.0, 1.4, 2.0\}$, plus $\Delta_{\max}$ %
- `results/robustness_noise.csv` (Table IX): per-method F-measure across
  $\sigma^2 \in \{0, 0.0125, 0.025, 0.0375, 0.05\}$, plus $\Delta_{\max}$ %

Cross-check:
- SuperEdge (fused): $\Delta_{\max} = 1.4\%$ on $\gamma$ sweep
- SuperEdge (fused): $\Delta_{\max} = 6.8\%$ on $\sigma^2$ sweep
- DexiNed-ST: $\Delta_{\max} \approx 73\%$ on $\sigma^2$ sweep (the contrast
  that motivates the robustness framing)

> **Note on the metric.** Per Section III-E paragraph "Note on the metric used
> in Tables VIII–IX", the per-condition F-measure uses a tolerance-relaxed
> approximation of the BSDS protocol (looser localization tolerance, threshold
> 0.3). Only relative deviations across rows are interpretable; the standard
> ODS = 0.825 anchor at $\gamma = 1$, $\sigma^2 = 0$ is reported separately
> in Table V.

**Expected runtime**: ~30 minutes per method × 7 methods × (5 + 5) conditions.

## 6. §III-F—6-DoF Arm McNemar's Test (n = 50)

```bash
python scripts/mcnemar_grasp.py \
    --trials data/grasp_trials/trials.csv \
    --out    results/mcnemar.json
```

Cross-check against paper Section III-F:
- $n = 50$ paired trials
- Baseline (YOLO only): $39 / 50$ (78%)
- YOLO + SuperEdge: $45 / 50$ (90%)
- $b = 6$, $c = 0$
- McNemar with continuity correction: $\chi^2 = 4.17$, $p = 0.041$ (two-sided)
- Significant at $\alpha = 0.05$

The arm-control / grasp-execution code lives in a separate sister repository
(see [README.md](../README.md)). The paired-trial outcome log released here is
sufficient to reproduce the statistical test.

**Expected runtime**: < 1 second.

## End-to-end Smoke Test

A single Make target runs steps 4 and 5 (which produce the inputs for step 2)
and then step 2:

```bash
make repro_uncertainty
```

If every cross-check above passes, the paper's central numerical claims are
fully reproducible from this repository.
