"""Rainfall-history partial correlations, regional inference and gate summaries."""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
from scipy import stats
from analysis_core import ROOT, SEEDS, STATES, read, save, rho, sha

BOOT = 2000
RNG_SEED = 20260928
GROUPS = {'all': None, 'high_forest': 'frac_forest', 'high_slope': 'slope_mean', 'high_clay': 'clay_frac'}

def rolling_sum(x, n):
    x = np.asarray(x, float)
    sums = np.r_[0., np.cumsum(np.nan_to_num(x, nan=0.))]
    counts = np.r_[0, np.cumsum(np.isfinite(x))]
    out = np.full(len(x), np.nan)
    out[n-1:] = sums[n:] - sums[:-n]
    out[n-1:][counts[n:] - counts[:-n] != n] = np.nan
    return out


def antecedent_inputs(p, temp):
    a30 = rolling_sum(p, 30)
    return {'p3': rolling_sum(p, 3), 'ap30': np.r_[np.full(3, np.nan), a30[:-3]],
            't7': rolling_sum(temp, 7) / 7}


def bh(values):
    p = np.asarray(values, float)
    # NA planned cells still count in family size, conservatively as untestable p=1.
    order = np.argsort(np.nan_to_num(p, nan=1.))
    ps = np.nan_to_num(p[order], nan=1.)
    qs = np.minimum.accumulate((ps * len(p) / np.arange(1, len(p)+1))[::-1])[::-1]
    out = np.empty(len(p)); out[order] = np.minimum(qs, 1)
    out[~np.isfinite(p)] = np.nan
    return out


def partial_corr(x, y, controls):
    x, y = np.asarray(x, float), np.asarray(y, float)
    controls = np.asarray(controls, float).reshape(len(x), -1)
    ranks = stats.rankdata(np.column_stack([x, y, controls]), axis=0)
    c = np.column_stack([np.ones(len(x)), ranks[:, 2:]])
    residual = ranks[:, :2] - c @ np.linalg.lstsq(c, ranks[:, :2], rcond=None)[0]
    rx, ry = residual.T
    denom = np.linalg.norm(rx) * np.linalg.norm(ry)
    r = float(rx @ ry / denom) if denom > 1e-10 else np.nan
    return r, rx, ry


_signs = {}


def signflip_p(cluster_scores):
    s = np.asarray(cluster_scores, float)
    g = len(s)
    if g < 2 or not np.isfinite(s).all():
        return np.nan
    if g not in _signs:
        _signs[g] = (((np.arange(2**g, dtype=np.uint32)[:, None] >> np.arange(g)) & 1).astype(float)*2 - 1)
    null = np.abs(_signs[g] @ s)
    return float(np.mean(null >= abs(s.sum()) - 1e-12 * max(1, np.abs(s).sum())))


def summarize_matrix(values, hucs, seed=RNG_SEED):
    v = np.asarray(values, float)
    if v.ndim == 1:
        v = v[:, None]
    valid = np.isfinite(v).all(axis=1)
    v, hucs = v[valid], np.asarray(hucs)[valid]
    if len(v) < 3:
        return dict(n_basins=len(v), n_huc2=len(np.unique(hucs)), mean=np.nan, median=np.nan,
                    ci_low=np.nan, ci_high=np.nan, p=np.nan, seed_negative=0, seed_positive=0,
                    seed_median_negative=0, seed_median_positive=0, loo_min=np.nan, loo_max=np.nan)
    regions, inv = np.unique(hucs, return_inverse=True)
    n = np.bincount(inv)
    sums = np.array([v[inv == j].sum(axis=0) for j in range(len(regions))])
    rng = np.random.default_rng(seed)
    hw = rng.multinomial(len(regions), np.full(len(regions), 1/len(regions)), size=BOOT)
    sw = rng.multinomial(v.shape[1], np.full(v.shape[1], 1/v.shape[1]), size=BOOT)
    b = np.einsum('bg,gs,bs->b', hw, sums, sw, optimize=True) / ((hw @ n) * v.shape[1])
    ci = np.quantile(b, [.025, .975])
    leave = [(v.sum()-sums[j].sum())/((len(v)-n[j])*v.shape[1]) for j in range(len(regions)) if len(v) > n[j]]
    return dict(n_basins=len(v), n_huc2=len(regions), mean=v.mean(), median=np.median(v.mean(axis=1)),
                ci_low=ci[0], ci_high=ci[1], p=signflip_p(sums.mean(axis=1)),
                seed_negative=int((v.mean(axis=0) < 0).sum()), seed_positive=int((v.mean(axis=0) > 0).sum()),
                seed_median_negative=int((np.median(v, axis=0) < 0).sum()),
                seed_median_positive=int((np.median(v, axis=0) > 0).sum()),
                loo_min=min(leave) if leave else np.nan, loo_max=max(leave) if leave else np.nan)


def summarize_gates(gates, attrs, metrics, output):
    rows, scatter, association = [], [], []
    joined = gates.merge(attrs.reset_index(), on='basin_id', validate='many_to_one')
    for feature in ['frac_forest', 'slope_mean', 'clay_frac']:
        for level in ['low', 'middle', 'high']:
            selected = joined[joined[feature+'_group'] == level]
            p = selected.pivot(index=['seed', 'basin_id'], columns='state', values='net_gate_shift')
            delta = p.high - p.mid
            mean = delta.groupby('basin_id').mean()
            sm = delta.groupby('seed').median()
            rows.append(dict(feature=feature, level=level, n_basins=len(mean),
                median_pp=float(mean.median()*100), q25_pp=float(mean.quantile(.25)*100),
                q75_pp=float(mean.quantile(.75)*100), positive_seeds=int((sm > 0).sum()),
                negative_seeds=int((sm < 0).sum()),
                **{f'seed_{seed}_median_pp': sm.loc[seed]*100 for seed in SEEDS}))
    save(rows, output/'paired_state_contrasts.csv')
    p = gates.pivot(index=['seed', 'basin_id'], columns='state', values='net_gate_shift')
    contrasts = (p.high-p.mid).groupby('basin_id').mean().mul(100).rename('high_minus_mid_pp')
    save(attrs.join(contrasts).reset_index(), output/'gate_density_data.csv')
    metrics_wide = metrics.pivot(index=['seed', 'basin_id', 'state'], columns='model', values='mae_mm_day')
    for baseline in ['B0', 'A1']:
        error = (metrics_wide.M1-metrics_wide[baseline]).rename('mae_change').reset_index()
        matched = gates.merge(error, on=['seed', 'basin_id', 'state'], validate='one_to_one')
        for state in ['low', 'mid', 'high']:
            sub = matched[matched.state == state]
            means = sub.groupby('basin_id')[['net_gate_shift', 'mae_change']].mean()
            association.append(dict(comparison='M1-'+baseline, state=state,
                rho_seedmean=rho(means.net_gate_shift, means.mae_change),
                **{f'rho_seed_{seed}': rho(s.net_gate_shift, s.mae_change) for seed, s in sub.groupby('seed')}))
            means['comparison'], means['state'] = 'M1-'+baseline, state
            scatter.append(means.reset_index())
    save(association, output/'gate_error_associations.csv')
    save(pd.concat(scatter), output/'gate_error_scatter.csv')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis', type=Path, default=ROOT/'runs/analysis')
    parser.add_argument('--gates', type=Path, default=ROOT/'runs/gates')
    parser.add_argument('--cache', type=Path, default=ROOT/'data/formal')
    parser.add_argument('--assignment', type=Path, default=ROOT/'splits/formal_pub_527/basin_assignment.csv')
    parser.add_argument('--predictions', type=Path, default=ROOT/'runs/evaluation_test')
    parser.add_argument('--output', type=Path, default=ROOT/'runs/analysis/gates')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    attrs = read(args.analysis/'basin_attributes_and_groups.csv').set_index('basin_id')
    attrs = attrs.join(read(args.assignment).set_index('basin_id')[['huc2']])
    thresholds = read(args.analysis/'flow_thresholds.csv').set_index('basin_id')
    gate = []
    identity = None
    for seed in SEEDS:
        with np.load(args.gates/f'M1_seed_{seed}_daily.npz', allow_pickle=False) as z:
            current = z['basin_ids'].copy(), z['dates'].copy()
            if identity is None:
                identity = current
            else:
                for x, y in zip(identity, current):
                    np.testing.assert_array_equal(x, y)
            gate.append(z['values'][:, z['metric_names'].tolist().index('net_gate_shift')].astype(float)*100)
    ids, dates = identity
    with np.load(args.predictions/'M1/seed_11/predictions_test.npz', allow_pickle=False) as z:
        np.testing.assert_array_equal(z['basin_ids'], ids)
        np.testing.assert_array_equal(z['dates'], dates)
        observed = z['q_observed_mm_day'].astype(float)
    gate = np.asarray(gate)
    with np.load(args.cache/'formal.npz', allow_pickle=False) as z:
        cache_ids, cache_dates = z['basin_ids'], z['dates']
        dynamic, valid = z['x_dynamic'], z['dynamic_valid']
    scaler = json.loads((args.cache/'scaler.json').read_text())['dynamic']
    correlations, curves, state_rows = [], [], []
    for basin in np.unique(ids):
        ix = np.flatnonzero(ids == basin)
        d, q, gg = dates[ix], observed[ix], gate[:, ix]
        ci = int(np.flatnonzero(cache_ids == basin)[0])
        di = np.searchsorted(cache_dates, d)
        np.testing.assert_array_equal(cache_dates[di], d)
        raw = dynamic[ci].astype(float)*np.array(scaler['scale'])+np.array(scaler['mean'])
        raw[~valid[ci]] = np.nan
        weather = antecedent_inputs(np.maximum(raw[:, 0], 0), (raw[:, 2]+raw[:, 3])/2)
        p3, ap, t7 = (weather[k][di] for k in ['p3', 'ap30', 't7'])
        assert np.isfinite(np.column_stack([p3, ap, t7])).all()
        q20, q95 = thresholds.loc[basin, ['q20_mm_day', 'q95_mm_day']]
        high = q >= q95
        low = ~high & (q <= q20)
        flow = np.where(low, 0, np.where(high, 2, 1))
        doy = pd.DatetimeIndex(d).dayofyear.to_numpy()
        season = np.column_stack([np.sin(2*np.pi*doy/365.25), np.cos(2*np.pi*doy/365.25)])
        for state, fi in [('low', 0), ('mid', 1), ('high', 2), ('all', None)]:
            mask = np.ones(len(q), bool) if fi is None else flow == fi
            for si, seed in enumerate(SEEDS):
                state_rows.append(dict(basin_id=basin, seed=seed, state=state,
                    n_days=int(mask.sum()), net_gate_shift=gg[si, mask].mean()/100))
            if state == 'low':
                continue
            for predictor, x, other in [('ap30', ap, p3), ('p3', p3, ap)]:
                controls = np.column_stack([other, q, t7, season])[mask]
                shared_bins = None
                for si, seed in enumerate(SEEDS):
                    if mask.sum() >= 30 and np.ptp(x[mask]) > 1e-10:
                        r, rx, ry = partial_corr(x[mask], gg[si, mask], controls)
                    else:
                        r, rx, ry = np.nan, None, None
                    correlations.append(dict(basin_id=basin, seed=seed, flow=state,
                        predictor=predictor, n_days=int(mask.sum()), partial_rho=r))
                    if state != 'all' and rx is not None:
                        if shared_bins is None:
                            predictor_ranks = stats.rankdata(np.column_stack([x[mask], controls]), axis=0)
                            design = np.column_stack([np.ones(mask.sum()), predictor_ranks[:, 1:]])
                            rx = predictor_ranks[:, 0]-design@np.linalg.lstsq(design, predictor_ranks[:, 0], rcond=None)[0]
                            rx = (rx-rx.mean())/rx.std(ddof=0)
                            shared_bins = np.searchsorted(np.quantile(rx, np.linspace(0, 1, 6)[1:-1], method='linear'), rx, side='right')
                        ry = (ry-ry.mean())/ry.std(ddof=0)
                        for bin_number in range(5):
                            choose = shared_bins == bin_number
                            curves.append(dict(basin_id=basin, seed=seed, flow=state, predictor=predictor,
                                bin=bin_number+1, percentile_center=10+20*bin_number,
                                n_days=int(choose.sum()), mean_adjusted_gate=ry[choose].mean()))
    correlations = pd.DataFrame(correlations)
    save(correlations, args.output/'gate_partial_basin_seed.csv')
    save(curves, args.output/'basin_seed_bin_curves.csv')
    save(state_rows, args.output/'basin_state_by_seed.csv')
    summaries = []
    for (flow, predictor), sub in correlations.groupby(['flow', 'predictor'], sort=True):
        matrix = sub.pivot(index='basin_id', columns='seed', values='partial_rho').reindex(index=attrs.index, columns=SEEDS)
        for group, feature in GROUPS.items():
            mask = np.ones(len(attrs), bool) if feature is None else attrs[feature+'_group'].to_numpy() == 'high'
            summaries.append(dict(flow=flow, predictor=predictor, group=group,
                **summarize_matrix(matrix.to_numpy()[mask], attrs.huc2.to_numpy()[mask])))
    summary = pd.DataFrame(summaries)
    assert len(summary) == 24
    summary['q_fdr'] = bh(summary.p)
    save(summary, args.output/'rainfall_group_summary.csv')
    summarize_gates(pd.DataFrame(state_rows), attrs,
                    read(args.analysis/'basin_state_metrics.csv'), args.output)
    manifest = dict(seeds=SEEDS, bh_family_size=24, bootstrap_replicates=BOOT, bootstrap_seed=RNG_SEED,
        definition='Local sigmoid(u_f+delta)-sigmoid(u_f) at the same M1 hidden state.',
        rainfall='P3=t-2..t; AP30=t-32..t-3; precipitation reconstructed from normalized float32 cache.',
        controls=['other rainfall window', 'observed discharge', 'T7', 'annual sine', 'annual cosine'],
        inference='Equal basin/seed means; exact two-sided HUC2 sign flips; BH over all 24 cells.',
        figures='Adjusted rank residual quintiles; same predictor bins for ten seeds; right-bin boundary ties.')
    (args.output/'gate_manifest.json').write_text(json.dumps(manifest, indent=2))
    print(summary[(summary.predictor == 'ap30') & (summary.flow == 'high')][['group', 'mean', 'q_fdr']].to_string(index=False))


if __name__ == '__main__':
    main()


