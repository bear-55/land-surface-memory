"""Compare regenerated numerical outputs with the archived reference summaries."""
from pathlib import Path
import argparse
import json
import numpy as np
from analysis_core import ROOT, read

TABLES = {
    'attribute_group_summary.csv': (['feature', 'level', 'comparison', 'state', 'metric'],
        ['median_seedmean_delta', 'median_relative_change_pct', 'group_median_improved_seeds',
         'group_median_worsened_seeds', 'mid_improve_high_worsen_group_median_seeds',
         'mid_worsen_high_improve_group_median_seeds']),
    'tertile_boundaries.csv': (['feature'], ['lower_tercile', 'upper_tercile', 'n_low', 'n_middle', 'n_high']),
    'flow_error_budget.csv': (['comparison', 'metric', 'state'], ['mean_delta', 'median_delta', 'mean_day_weighted_contribution']),
    'flow_tradeoffs.csv': (['comparison', 'metric'], ['opposite_sign_basins', 'all_regimes_improved']),
    'gates/rainfall_group_summary.csv': (['flow', 'predictor', 'group'], ['mean', 'p', 'q_fdr']),
    'gates/paired_state_contrasts.csv': (['feature', 'level'], ['median_pp', 'positive_seeds', 'negative_seeds']),
    'gates/basin_seed_bin_curves.csv': (['basin_id', 'flow', 'predictor', 'seed', 'bin'], ['n_days', 'mean_adjusted_gate']),
    'temporal/cross_period_summary.csv': (['threshold_scheme', 'comparison', 'flow', 'metric', 'group'],
        ['period_rank_rho', 'same_direction_rate_nonzero', 'median_original_delta', 'median_confirmation_delta']),
    'temporal/cross_period_joint_pattern_consistency.csv': (['threshold_scheme', 'comparison', 'group'],
        ['n_original_pattern', 'n_confirmation_pattern', 'n_both_periods_pattern'])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis', type=Path, default=ROOT/'runs/analysis')
    parser.add_argument('--reference', type=Path, default=ROOT/'reference_results')
    parser.add_argument('--atol', type=float, default=2e-10)
    parser.add_argument('--rtol', type=float, default=2e-10)
    args = parser.parse_args()
    results = []
    for name, (keys, columns) in TABLES.items():
        actual = read(args.analysis/name).set_index(keys).sort_index()
        expected = read(args.reference/name).set_index(keys).sort_index()
        if not actual.index.equals(expected.index):
            raise ValueError('Rows differ: '+name)
        x, y = actual[columns].to_numpy(float), expected[columns].to_numpy(float)
        results.append(dict(table=name, rows=len(actual),
            matches=bool(np.allclose(x, y, atol=args.atol, rtol=args.rtol, equal_nan=True)),
            maximum_absolute_difference=float(np.nanmax(np.abs(x-y)))))
    print(json.dumps(results, indent=2))
    if not all(row['matches'] for row in results):
        raise SystemExit('Numerical differences found. Inspect the report; do not substitute reference results.')


if __name__ == '__main__':
    main()
