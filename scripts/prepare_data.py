"""Build the paper cache from the original CAMELS-US files."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from project import digest, save_json
from dataset import read_forcing, DevelopmentStreamflow, valid_window_ends
from preprocessing import fit_static, apply_static, fit_dynamic, apply_dynamic


def index_files(folder, suffix, ids):
    result = {}
    for path in folder.rglob('*' + suffix):
        basin = path.name[:8]
        if basin in ids:
            if basin in result:
                raise ValueError(f'Duplicate files for {basin}: narrow the supplied root')
            result[basin] = path.resolve()
    missing = set(ids) - set(result)
    if missing:
        raise FileNotFoundError(f'Missing {suffix} files: {sorted(missing)}')
    return result


def climate_statistics(days, values, period):
    selected = (days >= np.datetime64(period[0])) & (days <= np.datetime64(period[1]))
    if selected.sum() != 3287:
        raise ValueError('Climate reference must contain 3287 calendar days')
    precipitation, tmax, tmin = values[selected, 0], values[selected, 2], values[selected, 3]
    pg = np.isfinite(precipitation) & (precipitation >= 0)
    tg = np.isfinite(tmax) & np.isfinite(tmin) & (tmin > -999) & (tmax >= tmin)
    joint = pg & tg
    temperature = (tmax + tmin) / 2
    total = precipitation[joint].sum()
    if not pg.any() or not tg.any() or total <= 0:
        raise ValueError('Invalid climate reference')
    return {
        'climate_mean_precipitation': precipitation[pg].mean(),
        'climate_mean_temperature': temperature[tg].mean(),
        'climate_std_precipitation': precipitation[pg].std(ddof=0),
        'climate_cold_precipitation_fraction': precipitation[joint & (temperature < 0)].sum() / total,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--forcing-root', type=Path, required=True)
    parser.add_argument('--streamflow-root', type=Path, required=True)
    parser.add_argument('--attributes-root', type=Path, required=True)
    args = parser.parse_args()
    cfg = json.loads((ROOT / 'configs/formal.json').read_text(encoding='utf-8'))
    split_dir = ROOT / 'splits/formal_pub_527'
    manifest = json.loads((split_dir / 'manifest.json').read_text(encoding='utf-8'))
    for name, checksum in manifest['sha256'].items():
        if digest(split_dir / name) != checksum:
            raise ValueError('Frozen split input changed: ' + name)
    splits = {r: (split_dir / (r + '.txt')).read_text().split() for r in ('train', 'val', 'test')}
    if {r: len(v) for r, v in splits.items()} != cfg['split_counts']:
        raise ValueError('Unexpected split sizes')
    ids = sorted(sum(splits.values(), []))
    if len(set(ids)) != 527:
        raise ValueError('Basin splits overlap')
    out = ROOT / 'data/formal'
    if (out / 'metadata.json').exists():
        raise FileExistsError('Cache already exists; use scripts/verify_release.py --cache to verify it')
    expected = json.loads((ROOT / 'data_contract/source_hashes.json').read_text(encoding='utf-8'))
    raw_hashes = {r['basin_id']: r for r in expected['basins']}
    forcing = index_files(args.forcing_root, '_lump_cida_forcing_leap.txt', ids)
    streamflow = index_files(args.streamflow_root, '_streamflow_qc.txt', ids)
    fields = pd.read_csv(ROOT / 'data_contract/feature_manifest.csv')
    static_fields = fields[fields.role.isin(['static_climate', 'static_land'])]
    names = static_fields.standard_name.tolist()
    land_names = static_fields.loc[static_fields.group == 'L', 'standard_name'].tolist()
    raw_static = pd.DataFrame(index=pd.Index(ids, name='basin_id'), columns=names, dtype=float)
    for filename, group in static_fields[static_fields.group == 'L'].groupby('source_file'):
        path = args.attributes_root / filename
        if digest(path) != expected['attributes'][filename]:
            raise ValueError('Raw attribute version differs from the paper: ' + filename)
        table = pd.read_csv(path, sep=';', dtype={'gauge_id': str}).set_index('gauge_id')
        for row in group.itertuples():
            raw_static[row.standard_name] = pd.to_numeric(table[row.source_column], errors='coerce').reindex(ids)
    dates = np.arange(np.datetime64(cfg['periods']['train_labels'][0]) - np.timedelta64(364, 'D'),
                      np.datetime64(cfg['periods']['final_labels'][1]) + np.timedelta64(1, 'D'))
    dynamic = np.empty((len(ids), len(dates), 5), dtype=np.float64)
    valid = np.empty((len(ids), len(dates)), dtype=bool)
    areas = {}
    for i, basin in enumerate(ids):
        if digest(forcing[basin]) != raw_hashes[basin]['forcing_sha256']:
            raise ValueError('Raw forcing version differs from the paper: ' + basin)
        days, values, mask, areas[basin] = read_forcing(forcing[basin])
        pos = np.searchsorted(days, dates)
        if np.any(pos >= len(days)) or not np.array_equal(days[pos], dates):
            raise ValueError('Forcing calendar does not cover the experiment: ' + basin)
        dynamic[i], valid[i] = values[pos], mask[pos]
        for name, value in climate_statistics(days, values, cfg['periods']['climate_reference']).items():
            raw_static.loc[basin, name] = value
        if (i + 1) % 100 == 0:
            print(f'Prepared forcing and climate attributes: {i + 1}/527', flush=True)
    reference = pd.read_csv(split_dir / 'candidate_attributes.csv', dtype={'basin_id': str}).set_index('basin_id')
    np.testing.assert_allclose(raw_static[names], reference.loc[ids, names], rtol=1e-12, atol=1e-12, equal_nan=True)
    train_rows = [ids.index(b) for b in splits['train']]
    static = raw_static[names].to_numpy(dtype=np.float64)
    ss = fit_static(static, names, train_rows, cfg['features']['static_missing_indicators'], cfg['features']['static_log1p'])
    ds = fit_dynamic(dynamic, valid, dates, train_rows, *cfg['periods']['train_labels'])
    xs, xd = apply_static(static, ss), apply_dynamic(dynamic, valid, ds)
    full_names = names + [x + '_missing' for x in cfg['features']['static_missing_indicators']]
    full_land = land_names + [x + '_missing' for x in cfg['features']['static_missing_indicators'] if x in land_names]
    periods = {'train': cfg['periods']['train_labels'], 'val': cfg['periods']['development_labels']}
    reader = DevelopmentStreamflow({b: streamflow[b] for b in splits['train'] + splits['val']}, splits, periods)
    sigma, means, windows, arrays = {}, {}, {}, {}
    reference_meta = json.loads((ROOT / 'reference_results/cache_metadata.json').read_text(encoding='utf-8'))
    for role in ('train', 'val'):
        arr = np.full((len(splits[role]), len(dates)), np.nan, dtype=np.float32)
        for j, basin in enumerate(splits[role]):
            if digest(streamflow[basin]) != raw_hashes[basin]['q_sha256']:
                raise ValueError('Raw streamflow version differs from the paper: ' + basin)
            days, values = reader.read(basin, role, areas[basin])
            arr[j, np.searchsorted(dates, days)] = values
            ends = valid_window_ends(valid[ids.index(basin)], dates, *periods[role], 365, arr[j])
            windows[basin] = {'role': role, 'valid_windows': len(ends)}
            if windows[basin] != reference_meta['basin_windows'][basin]:
                raise ValueError('Window count differs from the paper: ' + basin)
            if role == 'train':
                observed = values[np.isfinite(values)]
                sigma[basin], means[basin] = float(observed.std(ddof=0)), float(observed.mean())
        arrays['q_' + role] = arr
    qref = max(float(np.median(list(means.values()))), .1)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / 'formal.npz', basin_ids=np.array(ids), dates=dates,
                        x_dynamic=xd, dynamic_valid=valid, x_static=xs, **arrays)
    save_json(out / 'scaler.json', {'source_basin_ids': splits['train'], 'source_label_period': periods['train'],
              'dynamic': ds, 'static': ss, 'q_ref_mm_day': qref, 'source_sigma_mm_day': sigma, 'sigma_ddof': 0})
    save_json(out / 'label_access.json', {'scope': 'train_and_validation_only', 'test_q_read_by_this_builder': False,
              'records': reader.accesses})
    source_files = {b: {'forcing': str(forcing[b]), 'streamflow': str(streamflow[b]), 'area_m2': areas[b],
                        'streamflow_sha256': raw_hashes[b]['q_sha256']} for b in ids}
    rows = [{'basin_id': b, 'kind': kind, 'path': str(paths[b])}
            for kind, paths in [('daymet', forcing), ('streamflow_metadata_only', streamflow)] for b in ids]
    manifest_dir = ROOT / 'data/manifests'
    manifest_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(manifest_dir / 'source_files.csv', index=False)
    tracked = ['configs/formal.json', 'data_contract/feature_manifest.csv', 'splits/formal_pub_527/manifest.json',
               'splits/formal_pub_527/candidate_attributes.csv', 'scripts/prepare_data.py', 'src/dataset.py',
               'src/preprocessing.py', 'data/formal/formal.npz', 'data/formal/scaler.json']
    tracked += ['splits/formal_pub_527/' + role + '.txt' for role in splits]
    save_json(out / 'metadata.json', {'splits': splits, 'sequence_length': 365,
              'dynamic_names': cfg['features']['dynamic'], 'static_names': full_names, 'land_names': full_land,
              'land_indices': [full_names.index(x) for x in full_land], 'label_periods': periods,
              'q_ref_mm_day': qref, 'sigma_train_mm': sigma, 'basin_windows': windows, 'target_unit': 'mm/day',
              'test_q_in_cache': False, 'test_q_read_by_this_builder': False, 'prior_test_quality_audit': True,
              'config_hash': digest(ROOT / 'configs/formal.json'), 'cache_sha256': digest(out / 'formal.npz'),
              'scaler_sha256': digest(out / 'scaler.json'), 'feature_manifest_hash': digest(ROOT / 'data_contract/feature_manifest.csv'),
              'source_files': source_files, 'raw_source_hashes': raw_hashes,
              'source_hashes': {f: digest(ROOT / f) for f in tracked}})
    save_json(ROOT / 'configs/paths.local.json', {k: str(getattr(args, k).resolve())
              for k in ['forcing_root', 'streamflow_root', 'attributes_root']})
    print(json.dumps({'status': 'prepared', 'basins': len(ids), 'q_ref_mm_day': qref,
                     'train_windows': sum(v['valid_windows'] for v in windows.values() if v['role'] == 'train'),
                     'val_windows': sum(v['valid_windows'] for v in windows.values() if v['role'] == 'val'),
                     'test_q_read': False}))


if __name__ == '__main__':
    main()
