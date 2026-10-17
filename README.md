# Land-surface-conditioned memory for ungauged basins

Research code for **Flow-regime performance and training repeatability of a
land-surface-conditioned LSTM for ungauged basins**, by Xiong Wu, Xiaoran Fu,
Jinlun Jiang, Haijia Zhang, Hongquan Sun, and Zhonggen Wang.

The experiment compares three LSTMs on a fixed CAMELS-US split of 370 training,
50 validation, and 107 test basins. B0 concatenates static attributes with daily
meteorological inputs. A1 adds a shared low-rank term to the forget gate. M1
conditions that term on land-surface attributes. The analysis examines where
errors improve or increase across flow regimes, how those changes relate to
environmental groups, and whether they recur across training seeds and persist
across evaluation periods.

This repository contains the model, data preparation, training, fixed-weight
evaluation, gate diagnostics, analysis, frozen basin lists, and compact reference
results. See [中文使用说明](README_zh.md) for a short Chinese guide.

## Installation

Use Python 3.12 in a fresh environment. For the CUDA 13.0 build used in the study:

```bash
python -m venv .venv
# Activate .venv using your shell.
python -m pip install --upgrade pip
python -m pip install torch==2.12.1 --index-url https://download.pytorch.org/whl/cu130
python -m pip install -r requirements.txt
```

For CPU-only checks, replace `cu130` with `cpu` in the PyTorch command. Both
builds are provided by the official [PyTorch wheel index](https://download.pytorch.org/whl/).
The recorded training environment was Linux, Python 3.12.11, PyTorch
2.12.1+cu130, NumPy 2.3.2, and an NVIDIA GeForce RTX 5090. Analysis dependencies
are pinned in `requirements.txt`. GPU training uses float32, disables TF32 and
cuDNN benchmarking, and enables deterministic algorithms. Each run records its
actual software and device information.

## Quick verification without CAMELS data

Run commands from this directory:

```bash
python scripts/verify_release.py
python -m unittest discover -s tests -v
python scripts/smoke_test.py
```

These checks verify the release fingerprints, frozen basin membership, data
boundaries, paired initialization, A1's function-class equivalence, three-model
training on synthetic inputs, exact checkpoint resume, and fixed-weight
evaluation. They do not retrain the full hydrological experiment.

## Reproduce the experiment

### 1. Prepare the original data

Obtain CAMELS-US Daymet time series v1.2 and attributes v2.0 from the
[official sources](docs/data.md). Supply your local directories:

```bash
python scripts/prepare_data.py --forcing-root /path/to/basin_mean_forcing/daymet --streamflow-root /path/to/usgs_streamflow --attributes-root /path/to/camels_attributes_v2.0
python scripts/verify_release.py --cache
```

The builder preserves the published split, checks original-file hashes,
recomputes the climate attributes, fits preprocessing on training data, and
creates `data/formal/`. It does not put test streamflow in the training cache.
No machine-specific path is committed. See [data.md](docs/data.md) for file
layouts, units, provenance, and missing-data handling.

### 2. Train the paired models

```bash
python scripts/train.py --models all --seeds all --device cuda
```

The ten seeds are `11, 29, 47, 65, 83, 103, 119, 137, 155, 173`.
For a single run, use `--models M1 --seeds 11`. Resume interrupted runs with
`--resume`; code, configuration, and cache fingerprints must still match.
`--backend auto` uses the study's native fused B0 and CUDA Graph A1/M1 execution
on a GPU. `--backend reference --device cpu` provides the transparent manual
implementation, mainly useful for debugging small examples.

All models have hidden size 64, rank 4 where applicable, a 365-day input window,
output dropout 0.2, and a Softplus output. Training uses AdamW, batch size 64,
learning rate 0.001, weight decay 1e-5, and gradient clipping at 1.0. Validation
median NSE selects the checkpoint, with a maximum of 30 epochs and patience of
eight epochs. Training labels span 1 October 1989–30 September 1999; validation
labels span 1 October 1999–30 September 2003. Full settings are in
[`configs/formal.json`](configs/formal.json).

### 3. Evaluate selected weights

```bash
python scripts/evaluate.py --period test --device cuda
python scripts/evaluate.py --period historical --device cuda
```

Evaluation first checks that selected weights recover the saved validation
predictions. It then evaluates the same 107 held-out basins with fixed weights.
The test period is 1 October 2003–30 September 2008; the historical comparison
is 1 October 1989–30 September 1999. Historical evaluation neither retrains
the model nor selects another checkpoint. Valid dates are shared across models
and seeds; nonfinite predictions fail the run instead of changing its mask.

### 4. Compute diagnostics and summaries

```bash
python scripts/analysis_core.py
python scripts/diagnose_gates.py --device cuda
python scripts/analysis_gates.py
python scripts/analysis_temporal.py
python scripts/analysis_plots.py
```

See [analysis.md](docs/analysis.md) for the output tables, flow-regime boundary
rules, paired-seed aggregation, both directions of the joint trade-off, rainfall
controls, HUC2-region sign-flip tests, and cross-period threshold schemes.
The plotting script produces scientific regenerations in SVG and 600 dpi PNG;
the journal manuscript's manual layout edits are not required for reproducing
the numerical results.

## Repository layout

```text
src/                 Models, windows, preprocessing, training, evaluation
scripts/             Preparation, experiment, diagnostics, analysis, verification
configs/             Current ten-seed experiment configuration
splits/              Exact 370/50/107 membership and grouping attributes
data_contract/       Input definitions, raw-file fingerprints, basin screening
reference_results/   Compact results and preprocessing values for comparison
tests/               Data-independent numerical and protocol checks
docs/                Data, analysis, verification, and publication instructions
```

Generated caches and checkpoints remain local under `data/` and `runs/` and are
excluded from Git. This source release contains no pretrained checkpoints;
the commands above generate them. Training seeds and deterministic settings
control repeatability within a fixed setup; changing hardware or numerical
backends may change floating-point trajectories and selected epochs.

## Verification and citation

The integrity checks, unit tests, synthetic training example, and reference
comparisons are described in [verification.md](docs/verification.md). These
checks support running and evaluating the experiment on another system.

Use [`CITATION.cff`](CITATION.cff) to cite the software and cite the CAMELS-US
providers listed in [data.md](docs/data.md). The code is available under the
[MIT license](LICENSE). Raw datasets and third-party dependencies retain their
own terms.

