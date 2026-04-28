# SuperEdge: A Measurement-Systems Framework for Self-Supervised Edge Detection

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Paper: T-IM](https://img.shields.io/badge/Paper-IEEE_T--IM-red.svg)](#citing)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org)

SuperEdge is a self-supervised edge detector packaged with a **measurement-systems
evaluation framework**. Beyond conventional ODS / OIS accuracy, this repository
quantifies the four first-class indicators an instrumentation engineer applies
to any measurand:

| Indicator | What it measures | Where it lives |
| --- | --- | --- |
| **Accuracy** | ODS, OIS, AP across BIPED, BIPEDv2, BSDS500, NYUD, BSDS-RIND | `evaluate.py` |
| **Repeatability** | $\sigma_r$ under homographic perturbation, with bootstrap-1000 95% CIs | [`scripts/repeatability_sweep.py`](scripts/repeatability_sweep.py) |
| **Robustness** | F-measure under $\gamma \in [0.4, 2.0]$ illumination drift and $\sigma^2 \in [0, 0.05]$ additive Gaussian noise | [`scripts/robustness_sweep.py`](scripts/robustness_sweep.py) |
| **Deployment footprint** | Latency p50/p95, GPU memory, parameters, FLOPs, Pareto-domination across the foundation-model comparison set | [`scripts/bench_pareto.py`](scripts/bench_pareto.py) |

A GUM-aligned (ISO/IEC Guide 98-3, JCGM 100:2008) Type-A / Type-B uncertainty
budget combines the dominant noise sources into a **combined relative standard
uncertainty of 6.9%** for downstream traceability—see
[`scripts/uncertainty_budget.py`](scripts/uncertainty_budget.py).

A 6-DoF arm grasp study with **n = 50 paired trials** (McNemar's test,
$\chi^2 = 4.17$, $p = 0.041$) is reproduced from released trial logs by
[`scripts/mcnemar_grasp.py`](scripts/mcnemar_grasp.py).

> **Reproducibility note.** Every numerical claim in §§II-G, III-C, III-D,
> III-E, III-F of the accompanying T-IM submission is backed by exactly one
> script in `scripts/`, each with pinned seeds, a documented invocation
> block, and CSV/JSON output that matches a paper table or figure.
> See [`docs/REPRODUCE.md`](docs/REPRODUCE.md) for the step-by-step pipeline
> from `git clone` to each numerical artifact.

## Quick Start

```bash
git clone https://github.com/s20sc/SuperEdge.git
cd SuperEdge

# Training stack (matches the published checkpoint)
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Evaluation / measurement-systems stack (additional deps)
pip install -r requirements-eval.txt

# Reproduce paper Tables / Figures (see docs/REPRODUCE.md for full walkthrough)
python scripts/repeatability_sweep.py  --checkpoint export/<ckpt>.pth ...
python scripts/robustness_sweep.py     --checkpoint export/<ckpt>.pth ...
python scripts/bench_pareto.py         --model superedge ...
python scripts/uncertainty_budget.py   --repeatability results/repeatability.csv ...
python scripts/mcnemar_grasp.py        --trials data/grasp_trials/trials.csv ...
```

## Repository Layout

```
.
├── model/                  # SuperEdge architecture (encoder + dual decoder)
├── dataset/                # COCO, BIPED, BSDS500 loaders + synthetic shapes
├── solver/                 # loss, NMS, BSDS-style F-measure evaluator
├── config/                 # Hydra-style YAML configs for each training stage
├── train.py                # 3-stage training pipeline
├── evaluate.py             # accuracy evaluation (ODS / OIS / AP)
├── homography_adaptation.py
├── object_level_label.py
├── scripts/                # T-IM measurement-systems evaluation (this release)
│   ├── repeatability_sweep.py   # §III-D, Table VII
│   ├── robustness_sweep.py      # §III-E, Tables VIII / IX
│   ├── uncertainty_budget.py    # §II-G, Table II
│   ├── bench_pareto.py          # §III-C, Table VI, Fig. 4
│   └── mcnemar_grasp.py         # §III-F (6-DoF arm)
├── data/
│   ├── manifests/
│   │   └── coco_repeatability_12.txt   # pinned 12-image scene list
│   └── grasp_trials/
│       └── trials.csv                  # released n=50 paired-trial log
├── docs/
│   └── REPRODUCE.md                    # step-by-step recipe per paper claim
├── requirements.txt              # training stack
├── requirements-eval.txt         # additional eval-only deps (scipy, statsmodels, fvcore, ...)
├── CITATION.cff
└── LICENSE                       # Apache-2.0
```

## Pre-trained Weights

Download the BIPEDv2-trained checkpoint (1.29 M parameters):

```bash
# From the GitHub release page:
wget https://github.com/s20sc/SuperEdge/releases/download/v1.0/superedge_bipedv2.pth \
     -O export/superedge_bipedv2.pth
```

## Citing

If you use this code or the released measurement protocol, please cite the
T-IM paper:

```bibtex
@article{qin2026superedge,
  author  = {Qin, Xue and Leng, Kai and Zhang, Yuqi and Liu, Xin and
             Li, Tao and Chao, Pingfu and Li, Zhijun},
  title   = {{SuperEdge}: A Self-Calibrating Edge Detector
             for Vision-Based Measurement},
  journal = {IEEE Transactions on Instrumentation and Measurement},
  year    = {2026},
  note    = {In submission}
}
```

A machine-readable `CITATION.cff` is included.

## License

Released under the [Apache License 2.0](LICENSE). The trained checkpoints are
released under the same terms.

## Contact

Issues and PRs welcome on this repository. For research correspondence:
**Pingfu Chao** (pfchao@suda.edu.cn) and **Zhijun Li**
(lizhijun_os@hit.edu.cn), corresponding authors.
