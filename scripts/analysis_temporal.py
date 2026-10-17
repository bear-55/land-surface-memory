"""Fixed-weight cross-period comparison of all ten paired seeds."""
from pathlib import Path
import hashlib
import json
import argparse
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'runs/analysis/temporal'
ANALYSIS = ROOT / 'runs/analysis'
SEEDS = [11,29,47,65,83,103,119,137,155,173]
MODELS = ['B0','A1','M1']
COMPS = ['M1_B0','M1_A1','A1_B0']
FLOWS = ['all','low','mid','high']
ERAS = ['original_2003_2008','confirmation_1989_1999']
SOURCES = [ROOT / 'runs/evaluation_test', ROOT / 'runs/evaluation_historical']
PERIODS = [['2003-10-01','2008-09-30'], ['1989-10-01','1999-09-30']]
HISTORICAL_FILENAME = 'predictions_historical.npz'


def read(path):
    return pd.read_csv(path, dtype={'basin_id':str}, float_precision='round_trip')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(rows, name):
    pd.DataFrame(rows).to_csv(OUT / name, index=False, encoding='utf-8-sig')


def rho(a,b):
    return float(pd.Series(a).rank().corr(pd.Series(b).rank()))


def stats(x,y):
    x,y = np.asarray(x),np.asarray(y)
    valid = np.isfinite(x)&np.isfinite(y)
    x,y = x[valid],y[valid]
    nonzero = (x!=0)&(y!=0)
    same = ((x<0)&(y<0))|((x>0)&(y>0))
    return dict(n_valid=len(x), mean_original_delta=float(np.mean(x)), median_original_delta=float(np.median(x)),
                mean_confirmation_delta=float(np.mean(y)), median_confirmation_delta=float(np.median(y)),
                mean_paired_change=float(np.mean(y-x)), median_paired_change=float(np.median(y-x)),
                period_rank_rho=rho(x,y), n_both_negative=int(((x<0)&(y<0)).sum()),
                n_both_positive=int(((x>0)&(y>0)).sum()), n_sign_reversed=int((x*y<0).sum()),
                n_nonzero_both=int(nonzero.sum()), same_direction_rate_nonzero=float(same[nonzero].mean()))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    metrics, threshold_rows, coverage_rows, input_hashes = [],[],[],{}
    original_thresholds = {}
    for era, source, period in zip(ERAS,SOURCES,PERIODS):
        predictions,canonical = {},None
        for model in MODELS:
            for seed in SEEDS:
                filename = 'predictions_test.npz' if era == ERAS[0] else HISTORICAL_FILENAME
                path=source/model/f'seed_{seed}'/filename
                h=sha(path)
                input_hashes[str(path.resolve())]=h
                with np.load(path,allow_pickle=False) as z:
                    rec=tuple(z[k].copy() for k in ('basin_ids','dates','q_observed_mm_day'))
                    prediction=z['q_predicted_mm_day'].astype(float)
                assert np.isfinite(prediction).all() and (prediction>=0).all()
                if canonical is None: canonical=rec
                else: assert all(np.array_equal(a,b) for a,b in zip(canonical,rec))
                predictions[model,seed]=prediction
        ids,dates,observed=canonical
        observed=observed.astype(float)
        assert len(np.unique(ids))==107 and np.isfinite(observed).all()
        expected_days=int((np.datetime64(period[1])-np.datetime64(period[0])).astype(int))+1
        for basin in np.unique(ids):
            ix=np.flatnonzero(ids==basin)
            q=observed[ix]
            quantiles=np.quantile(q,[.2,.95],method='linear')
            if era==ERAS[0]: original_thresholds[basin]=quantiles
            coverage=len(ix)/expected_days
            coverage_rows.append(dict(era=era,basin_id=basin,period='all',period_n_days=len(ix),
                                      expected_period_days=expected_days,coverage=coverage))
            for scheme in (['period_relative'] if era==ERAS[0] else ['period_relative','original_absolute']):
                q20,q95=quantiles if scheme=='period_relative' else original_thresholds[basin]
                high=q>=q95
                low=~high&(q<=q20)
                masks=dict(all=np.ones(len(ix),bool),low=low,mid=~low&~high,high=high)
                threshold_rows.append(dict(era=era,threshold_scheme=scheme,basin_id=basin,
                    q20_mm_day=q20,q95_mm_day=q95,n_days=len(ix),
                    **{f'n_{f}':int(masks[f].sum()) for f in FLOWS[1:]}))
                for flow,mask in masks.items():
                    qo=q[mask]
                    common=dict(era=era,threshold_scheme=scheme,basin_id=basin,period='all',flow=flow,
                                n_days=len(qo),period_n_days=len(q),expected_period_days=expected_days,coverage=coverage)
                    for (model,seed),pred in predictions.items():
                        error=pred[ix[mask]]-qo
                        metrics.append(dict(**common,model=model,seed=seed,
                            mae=float(np.abs(error).mean()) if len(qo) else np.nan,
                            mse=float(np.square(error).mean()) if len(qo) else np.nan,
                            mean_error=float(error.mean()) if len(qo) else np.nan,
                            mae_contribution=float(np.abs(error).sum()/len(q)),
                            mse_contribution=float(np.square(error).sum()/len(q))))
        print(f'{era}: metrics calculated for all 30 models/seeds',flush=True)
    metrics=pd.DataFrame(metrics)
    save(metrics,'temporal_metrics.csv')
    save(threshold_rows,'temporal_thresholds.csv')
    save(coverage_rows,'temporal_coverage.csv')
    idx=['era','threshold_scheme','basin_id','period','flow','seed']
    pairs=[]
    for comp in COMPS:
        a,b=comp.split('_')
        left=metrics[metrics.model==a].set_index(idx).sort_index()
        right=metrics[metrics.model==b].set_index(idx).sort_index()
        out=left[['n_days','period_n_days','coverage']].copy()
        for metric in ['mae','mse','mean_error','mae_contribution','mse_contribution']:
            out['delta_'+metric]=left[metric]-right[metric]
        out['comparison']=comp
        pairs.append(out.reset_index())
    pairs=pd.concat(pairs,ignore_index=True)
    save(pairs,'temporal_pairs.csv')
    attrs=read(ANALYSIS/'basin_attributes_and_groups.csv').set_index('basin_id')
    # Preserve the training-derived groups, including all three thematic attributes.
    groups={'all':set(attrs.index)}
    for feature in ['frac_forest','slope_mean','clay_frac']:
        for group in ['low','middle','high']:
            groups[f'{feature}_{group}']=set(attrs.index[attrs[f'{feature}_group']==group])
    summaries,seed_summaries,coordinates,group_rows=[],[],[],[]
    for scheme in ['period_relative','original_absolute']:
        for comp in COMPS:
            for flow in FLOWS:
                part=pairs[(pairs.comparison==comp)&(pairs.flow==flow)&
                           (((pairs.era==ERAS[0])&(pairs.threshold_scheme=='period_relative'))|
                            ((pairs.era==ERAS[1])&(pairs.threshold_scheme==scheme)))]
                for metric in ['delta_mae','delta_mse']:
                    pivot=part.pivot(index=['basin_id','seed'],columns='era',values=metric)
                    average=pivot.groupby('basin_id').mean()
                    assert len(average)==107 and len(pivot)==1070
                    for basin,row in average.iterrows():
                        coordinates.append(dict(threshold_scheme=scheme,comparison=comp,flow=flow,metric=metric,
                            basin_id=basin,original_delta=row[ERAS[0]],confirmation_delta=row[ERAS[1]],
                            paired_change=row[ERAS[1]]-row[ERAS[0]]))
                    for group,ids in groups.items():
                        selected=average[average.index.isin(ids)]
                        summary=dict(threshold_scheme=scheme,comparison=comp,flow=flow,metric=metric,
                                     cohort='full',group=group,**stats(selected[ERAS[0]],selected[ERAS[1]]))
                        summaries.append(summary)
                        for seed in SEEDS:
                            s=pivot.xs(seed,level='seed')
                            s=s[s.index.isin(ids)]
                            seed_summaries.append(dict(threshold_scheme=scheme,comparison=comp,flow=flow,metric=metric,
                                cohort='full',group=group,seed=seed,**stats(s[ERAS[0]],s[ERAS[1]])))
    summary=pd.DataFrame(summaries)
    seed_summary=pd.DataFrame(seed_summaries)
    main_ref=read(ANALYSIS/'attribute_group_summary.csv')
    main_errors=[]
    for row in summary[(summary.threshold_scheme=='period_relative')&(summary.flow!='all')&
                       (summary.group!='all')].itertuples():
        feature,level=row.group.rsplit('_',1)
        ref=main_ref[(main_ref.feature==feature)&(main_ref.level==level)&
                     (main_ref.comparison==row.comparison.replace('_','-'))&
                     (main_ref.state==row.flow)&(main_ref.metric==row.metric.removeprefix('delta_'))]
        assert len(ref)==1
        difference=abs(row.median_original_delta-ref.iloc[0].median_seedmean_delta)
        assert difference<1e-12
        main_errors.append(difference)
    save(summary,'cross_period_summary.csv')
    save(seed_summary,'cross_period_seed_statistics.csv')
    save(coordinates,'cross_period_basin_effects.csv')
    consistency=[]
    for key,part in seed_summary.groupby(['threshold_scheme','comparison','flow','metric','group'],sort=False):
        x,y=part.median_original_delta,part.median_confirmation_delta
        consistency.append(dict(zip(['threshold_scheme','comparison','flow','metric','group'],key),
            n_seeds=len(part),n_original_negative=int((x<0).sum()),n_confirmation_negative=int((y<0).sum()),
            n_both_negative=int(((x<0)&(y<0)).sum()),n_both_positive=int(((x>0)&(y>0)).sum()),
            n_sign_reversed=int((x*y<0).sum()),seed_rho_min=float(part.period_rank_rho.min()),
            seed_rho_max=float(part.period_rank_rho.max()),seed_rho_positive=int((part.period_rank_rho>0).sum()),
            seed_same_direction_rate_min=float(part.same_direction_rate_nonzero.min()),
            seed_same_direction_rate_max=float(part.same_direction_rate_nonzero.max())))
    save(consistency,'cross_period_seed_consistency.csv')
    joint=[]
    for scheme in ['period_relative','original_absolute']:
        for comp in ['M1_B0','M1_A1']:
            for feature in ['frac_forest','slope_mean','clay_frac']:
                s=seed_summary[(seed_summary.threshold_scheme==scheme)&(seed_summary.comparison==comp)&
                               (seed_summary.metric=='delta_mae')&(seed_summary.group==feature+'_high')&
                               seed_summary.flow.isin(['mid','high'])]
                old=s.pivot(index='seed',columns='flow',values='median_original_delta')
                early=s.pivot(index='seed',columns='flow',values='median_confirmation_delta')
                if feature=='clay_frac':
                    a=(old.mid>0)&(old.high<0)
                    b=(early.mid>0)&(early.high<0)
                    label='mid degradation and high improvement'
                else:
                    a=(old.mid<0)&(old.high>0)
                    b=(early.mid<0)&(early.high>0)
                    label='mid improvement and high degradation'
                joint.append(dict(threshold_scheme=scheme,comparison=comp,group=feature+'_high',
                    target_pattern=label,n_seeds=10,n_original_pattern=int(a.sum()),
                    n_confirmation_pattern=int(b.sum()),n_both_periods_pattern=int((a&b).sum()),
                    **{f'seed{seed}_both_periods_pattern':bool(a.loc[seed]&b.loc[seed]) for seed in SEEDS}))
    save(joint,'cross_period_joint_pattern_consistency.csv')
    protocol=dict(seeds=SEEDS,models=MODELS,test_basins=107,periods=PERIODS,
        threshold_schemes=['period_relative','original_absolute'],
        aggregation='Paired within-seed metric difference, then arithmetic mean across all ten seeds within each basin; equal-basin summaries',
        repeatability='Per-seed basin rank correlations, sign agreement and attribute-group medians reported separately from seed-mean coordinates',
        scope='Full-period fixed-weight retrospective comparison on the same 107 spatial holdouts',
        audit=dict(existing_ten_seed_group_summary_max_error=max(main_errors),shared_masks_all_30=True,
                   source_sha256=input_hashes,script_sha256=sha(Path(__file__))))
    (OUT/'temporal_protocol.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2),encoding='utf-8')
    panel=summary[(summary.threshold_scheme=='period_relative')&(summary.group=='all')&
                  (summary.metric=='delta_mae')&(summary.flow!='all')]
    print(panel[['comparison','flow','period_rank_rho','n_both_negative','n_both_positive','n_sign_reversed']].to_string(index=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--test', type=Path, default=SOURCES[0])
    parser.add_argument('--historical', type=Path, default=SOURCES[1])
    parser.add_argument('--analysis', type=Path, default=ANALYSIS)
    parser.add_argument('--output', type=Path, default=OUT)
    parser.add_argument('--historical-filename', default=HISTORICAL_FILENAME)
    args = parser.parse_args()
    SOURCES, ANALYSIS, OUT = [args.test, args.historical], args.analysis, args.output
    HISTORICAL_FILENAME = args.historical_filename
    main()

