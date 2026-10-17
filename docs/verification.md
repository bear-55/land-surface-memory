# Verification

Run the checks from the repository root after installing the dependencies.

## Source integrity and frozen membership

```bash
python scripts/verify_release.py
```

This checks the files listed in `MANIFEST.sha256`, the frozen split fingerprints,
and the 370/50/107 basin counts. Git text conversion is disabled in
`.gitattributes` so that byte fingerprints remain unchanged across platforms.

## Data-independent tests

```bash
python -m unittest discover -s tests -v
python scripts/smoke_test.py
```

The unit tests cover basin membership and features, training-only preprocessing,
missingness indicators, streamflow conversion, label isolation, metric edge
cases, and causal windows. The synthetic example exercises B0, A1, and M1,
paired initialization, A1 function equivalence, native/manual B0 execution,
checkpoint resume, and fixed-weight evaluation. It requires no CAMELS files.
GitHub Actions runs these checks using the CPU build of PyTorch.

## Prepared data

After running `scripts/prepare_data.py`:

```bash
python scripts/verify_release.py --cache
```

The expected cache contains 527 basins, 1,341,831 training windows, and 73,050
validation windows. The common streamflow scale is
`q_ref = 1.2906279207521238` mm/day. Five daily inputs and 21 static inputs
(20 numerical attributes plus one missingness indicator) enter the backbone;
17 static inputs enter the conditioning branch. The cache contains training
and validation streamflow but no test streamflow.

The check compares scalers and metadata with `reference_results/preprocessing.json`
and `reference_results/cache_metadata.json`. Raw-file fingerprints and data
definitions are provided in `data_contract/`.

## Experiment outputs

After training, evaluating both periods, and running the analysis:

```bash
python scripts/analysis_verify.py
```

This compares regenerated summaries with the archived reference tables at
explicit numerical tolerances. Examine differences in conjunction with the
actual environment, seed-level outcomes, and backend settings. Replotting the
reference tables or passing the synthetic example does not retrain the full
experiment. See [analysis.md](analysis.md) for aggregation and comparison rules.
