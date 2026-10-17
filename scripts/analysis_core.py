"""Paired ten-seed metrics, flow-regime budgets and environmental groups."""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SEEDS = [11, 29, 47, 65, 83, 103, 119, 137, 155, 173]
MODELS = ['B0', 'A1', 'M1']
PAIRS = [('M1', 'B0'), ('M1', 'A1'), ('A1', 'B0')]
STATES = ['low', 'mid', 'high', 'all']
FEATURES = ['elev_mean', 'slope_mean', 'area_geospa_fabric', 'frac_forest',
            'lai_max', 'lai_diff', 'soil_depth_pelletier', 'soil_depth_statsgo',
            'soil_porosity', 'soil_conductivity', 'max_water_content',
            'sand_frac', 'silt_frac', 'clay_frac', 'geol_porosity', 'geol_permeability']


def read(path):
    return pd.read_csv(path, dtype={'basin_id': str, 'huc2': str}, float_precision='round_trip')


def save(frame, path):
    pd.DataFrame(frame).to_csv(path, index=False)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def rho(x, y):
    return float(pd.Series(x).rank().corr(pd.Series(y).rank()))


def load_predictions(source, period='test'):
    predictions, canonical, hashes = {}, None, {}
    for model in MODELS:
        for seed in SEEDS:
            path = source / model / f'seed_{seed}' / f'predictions_{period}.npz'
            with np.load(path, allow_pickle=False) as z:
                identity = tuple(z[k].copy() for k in ['basin_ids', 'dates', 'q_observed_mm_day'])
                values = z['q_predicted_mm_day'].astype(float)
            if canonical is None:
                canonical = identity
            else:
                for x, y in zip(canonical, identity):
                    np.testing.assert_array_equal(x, y)
            assert np.isfinite(values).all() and (values >= 0).all()
            predictions[model, seed] = values
            hashes[str(path.resolve())] = sha(path)
    ids, dates, observed = canonical
    assert len(np.unique(ids)) == 107 and np.isfinite(observed).all()
    assert not pd.MultiIndex.from_arrays([ids, dates]).has_duplicates
    return ids, dates, observed.astype(float), predictions, hashes


def state_metrics(ids, observed, predictions):
    rows, thresholds, overall = [], [], []
    for basin in np.unique(ids):
        ix = np.flatnonzero(ids == basin)
        q = observed[ix]
        q20, q95 = np.quantile(q, [.2, .95], method='linear')
        high = q >= q95
        low = (~high) & (q <= q20)
        masks = [low, ~(low | high), high, np.ones(len(q), bool)]
        assert all(m.any() for m in masks)
        thresholds.append(dict(basin_id=basin, q20_mm_day=q20, q95_mm_day=q95,
            n_days=len(q), n_low=int(low.sum()), n_mid=int(masks[1].sum()), n_high=int(high.sum())))
        for (model, seed), prediction in predictions.items():
            error = prediction[ix] - q
            nse = 1 - np.square(error).sum() / np.square(q - q.mean()).sum()
            overall.append(dict(basin_id=basin, model=model, seed=seed, nse=nse,
                                mae_mm_day=np.abs(error).mean()))
            for state, mask in zip(STATES, masks):
                rows.append(dict(model=model, seed=seed, basin_id=basin, state=state,
                    n_days=int(mask.sum()), mae_mm_day=np.abs(error[mask]).mean(),
                    mse_mm2_day2=np.square(error[mask]).mean(),
                    mae_contribution=np.abs(error[mask]).sum() / len(q),
                    mse_contribution=np.square(error[mask]).sum() / len(q)))
    return pd.DataFrame(rows), pd.DataFrame(thresholds), pd.DataFrame(overall)


def attribute_groups(split, metrics):
    attrs = read(split / 'candidate_attributes.csv').set_index('basin_id')
    train = attrs.loc[(split / 'train.txt').read_text().split()]
    test = attrs.loc[sorted((split / 'test.txt').read_text().split()), FEATURES].copy()
    assert len(train) == 370 and len(test) == 107
    ix = pd.MultiIndex.from_product([MODELS, SEEDS, test.index, STATES],
                                    names=['model', 'seed', 'basin_id', 'state'])
    errors = metrics.set_index(ix.names).loc[ix, ['mae_mm_day', 'mse_mm2_day2']].to_numpy().reshape(3, 10, 107, 4, 2)
    cuts, groups, associations = [], [], []
    forest_cut = np.quantile(train.frac_forest.dropna(), 2 / 3, method='linear')
    high_forest = test.frac_forest.to_numpy() > forest_cut
    for feature in FEATURES:
        valid_train = train[feature].dropna().to_numpy()
        q1, q2 = np.quantile(valid_train, [1 / 3, 2 / 3], method='linear')
        x = test[feature].to_numpy(float)
        valid = np.isfinite(x)
        levels = np.where(~valid, 'missing', np.where(x <= q1, 'low', np.where(x <= q2, 'middle', 'high')))
        test[feature + '_group'] = levels
        cuts.append(dict(feature=feature, lower_tercile=q1, upper_tercile=q2,
            train_valid_n=len(valid_train), test_valid_n=int(valid.sum()), test_missing_n=int((~valid).sum()),
            **{f'n_{level}': int((levels == level).sum()) for level in ['low', 'middle', 'high']}))
        for left, right in PAIRS:
            base = errors[MODELS.index(right)]
            delta = errors[MODELS.index(left)] - base
            mean = delta.mean(axis=0)
            for ei, metric in enumerate(['mae', 'mse']):
                for fi, state in enumerate(STATES):
                    sr = np.array([rho(x[valid], delta[si, valid, fi, ei]) for si in range(10)])
                    associations.append(dict(feature=feature, comparison=left+'-'+right, state=state, metric=metric,
                        n_basins=int(valid.sum()), rho_seedmean_effect=rho(x[valid], mean[valid, fi, ei]),
                        negative_rho_seeds=int((sr < 0).sum()), positive_rho_seeds=int((sr > 0).sum()),
                        **{f'rho_seed_{s}': v for s, v in zip(SEEDS, sr)}))
                    for level in ['low', 'middle', 'high']:
                        mask = levels == level
                        d, m = delta[:, mask, fi, ei], mean[mask, fi, ei]
                        b = base[:, mask, fi, ei].mean(axis=0)
                        sm = np.median(d, axis=1) if mask.any() else np.full(10, np.nan)
                        medium = np.median(delta[:, mask, 1, ei], axis=1) if mask.any() else np.full(10, np.nan)
                        high = np.median(delta[:, mask, 2, ei], axis=1) if mask.any() else np.full(10, np.nan)
                        relative = np.divide(m, b, out=np.full_like(m, np.nan), where=b > 0) * 100
                        groups.append(dict(feature=feature, level=level, comparison=left+'-'+right, state=state,
                            metric=metric, n_basins=int(mask.sum()), n_high_forest_overlap=int((mask & high_forest).sum()),
                            median_seedmean_delta=float(np.median(m)) if len(m) else np.nan,
                            mean_seedmean_delta=float(np.mean(m)) if len(m) else np.nan,
                            median_relative_change_pct=float(np.nanmedian(relative)) if np.isfinite(relative).any() else np.nan,
                            group_median_improved_seeds=int((sm < 0).sum()), group_median_worsened_seeds=int((sm > 0).sum()),
                            mid_improve_high_worsen_group_median_seeds=int(((medium < 0) & (high > 0)).sum()),
                            mid_worsen_high_improve_group_median_seeds=int(((medium > 0) & (high < 0)).sum()),
                            **{f'group_median_delta_seed_{s}': v for s, v in zip(SEEDS, sm)}))
    return pd.DataFrame(cuts), test.reset_index(), pd.DataFrame(groups), pd.DataFrame(associations)


def summarize(metrics, overall):
    run = overall.groupby(['model', 'seed']).nse.median().rename('median_nse').reset_index()
    model = run.groupby('model').median_nse.agg(['mean', 'std', 'min', 'max']).reset_index()
    pairs, budgets, tradeoffs = [], [], []
    for left, right in PAIRS:
        wide = overall.pivot(index=['basin_id', 'seed'], columns='model', values='nse')
        delta = (wide[left] - wide[right]).unstack('seed').loc[:, SEEDS]
        for basin, values in delta.iterrows():
            pairs.append(dict(comparison=left+'-'+right, basin_id=basin,
                mean_delta=values.mean(), improved_seeds=int((values > 0).sum()),
                **{f'seed_{s}_delta': values[s] for s in SEEDS}))
        index = ['basin_id', 'seed', 'state']
        a = metrics[metrics.model == left].set_index(index)
        b = metrics[metrics.model == right].set_index(index)
        for metric, contribution in [('mae_mm_day', 'mae_contribution'), ('mse_mm2_day2', 'mse_contribution')]:
            difference = a[[metric, contribution]] - b[[metric, contribution]]
            means = difference.groupby(['basin_id', 'state']).mean()
            wide = means[metric].unstack('state')
            mixed = (wide[STATES[:3]].min(axis=1) < 0) & (wide[STATES[:3]].max(axis=1) > 0)
            tradeoffs.append(dict(comparison=left+'-'+right, metric=metric,
                n_basins=len(wide), opposite_sign_basins=int(mixed.sum()),
                all_regimes_improved=int((wide[STATES[:3]] < 0).all(axis=1).sum())))
            for state in STATES:
                part = means.xs(state, level='state')
                budgets.append(dict(comparison=left+'-'+right, metric=metric, state=state,
                    mean_delta=part[metric].mean(), median_delta=part[metric].median(),
                    mean_day_weighted_contribution=part[contribution].mean()))
            parts = means[contribution].unstack('state')
            np.testing.assert_allclose(parts[STATES[:3]].sum(axis=1), parts['all'], atol=2e-11, rtol=1e-10)
    return run, model, pd.DataFrame(pairs), pd.DataFrame(budgets), pd.DataFrame(tradeoffs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictions', type=Path, default=ROOT / 'runs/evaluation_test')
    parser.add_argument('--split', type=Path, default=ROOT / 'splits/formal_pub_527')
    parser.add_argument('--output', type=Path, default=ROOT / 'runs/analysis')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    ids, dates, q, pred, hashes = load_predictions(args.predictions)
    metrics, thresholds, overall = state_metrics(ids, q, pred)
    cuts, attrs, groups, associations = attribute_groups(args.split, metrics)
    run, model, paired, budgets, tradeoffs = summarize(metrics, overall)
    outputs = dict(basin_state_metrics=metrics, flow_thresholds=thresholds, overall_metrics=overall,
        tertile_boundaries=cuts, basin_attributes_and_groups=attrs, attribute_group_summary=groups,
        attribute_associations=associations, run_summary=run, model_summary=model, paired_nse=paired,
        flow_error_budget=budgets, flow_tradeoffs=tradeoffs)
    for name, frame in outputs.items():
        save(frame, args.output / (name+'.csv'))
    (args.output / 'core_manifest.json').write_text(json.dumps(dict(seeds=SEEDS, models=MODELS,
        basins=107, basin_days=len(ids), input_sha256=hashes,
        aggregation='Paired seed differences, seed mean within basin, then equal-basin summaries.'), indent=2))
    print(f'Wrote {len(outputs)} tables to {args.output}')


if __name__ == '__main__':
    main()
