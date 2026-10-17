"""Fixed-checkpoint evaluation on common valid dates."""
from datetime import date
from pathlib import Path
import numpy as np
import torch
from dataset import valid_window_ends
from evaluate import basin_metrics, summarize
from graph_backend import GraphBackend
from project import ROOT, digest, read_json, save_json
from training import make_model, predict_scaled

PRED_RTOL, PRED_ATOL, NSE_ATOL = 1e-4, 1e-5, 1e-5

def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def load_completed_model(directory, cfg, meta, device, hashes=None):
    directory = Path(directory)
    spec, result = (read_json(directory / name) for name in ("run.json", "result.json"))
    require(spec["protocol"] == cfg, "Run protocol differs from the current configuration")
    require(result["status"] == "completed" and not result["test_q_read"], "Completed validation-selected run required")
    if hashes is not None:
        require(spec["provenance"] == hashes, "Training code or prepared data changed")
    checkpoint = torch.load(directory / "best.pt", map_location="cpu", weights_only=True)
    require(checkpoint["identity"] == spec["identity"] == result["identity"], "Checkpoint identity mismatch")
    require(checkpoint["epoch"] == result["best_epoch"], "Checkpoint was not selected by validation")
    backend = "native_fused" if spec["backend"] == "native_fused" else "reference_manual_loop"
    model = make_model(cfg, meta, spec["model"], spec["seed"], device, backend)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    require(all(bool(torch.isfinite(v).all()) for v in model.state_dict().values()), "Nonfinite checkpoint")
    return model, spec, result

def parse_streamflow(text, basin, period, area_m2):
    """Parse only the frozen period; same units/flag rules as development labels."""
    if not np.isfinite(area_m2) or area_m2 <= 0:
        raise ValueError('Finite positive forcing-header area required')
    start, end = map(date.fromisoformat, period)
    if end < start:
        raise ValueError('Invalid evaluation period')
    dates = np.arange(np.datetime64(start), np.datetime64(end) + np.timedelta64(1, 'D'))
    q = np.full(len(dates), np.nan, dtype=np.float32)
    previous = None
    for line in text.splitlines():
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 6 or fields[0] != basin:
            raise ValueError('Unexpected streamflow schema/basin')
        day = date(*map(int, fields[1:4]))
        if previous is not None and day <= previous:
            raise ValueError('Duplicate or unordered streamflow dates')
        previous = day
        if day < start:
            continue
        if day > end:
            break
        flag = fields[5]
        if not (flag == 'M' or flag.startswith('A') or flag.startswith('P')):
            raise ValueError('Unreviewed streamflow flag: ' + flag)
        observed = float(fields[4])
        if flag == 'M' or not np.isfinite(observed) or observed < 0:
            continue
        q[(day - start).days] = observed * 0.028316846592 * 86400 * 1000 / area_m2
    return dates, q


class EvaluationWindows:
    """One immutable common mask. Only meteorology/static covariates enter batches."""
    def __init__(self, cache, meta, ids, period, q):
        self.ids = list(ids)
        self.dates = cache['dates']
        self.dynamic, self.static = cache['x_dynamic'], cache['x_static']
        self.land_indices = meta['land_indices']
        self.length = meta['sequence_length']
        require(q.shape == (len(ids), len(self.dates)), 'Label shape mismatch')
        all_ids = cache['basin_ids'].tolist()
        rows = [all_ids.index(b) for b in ids]
        ends = [valid_window_ends(cache['dynamic_valid'][row], self.dates, *period, self.length, q[j])
                for j, row in enumerate(rows)]
        require(all(len(e) > 0 for e in ends), 'A basin has no common valid dates')
        self.basins = np.repeat(np.arange(len(ids)), [len(e) for e in ends])
        self.ends = np.concatenate(ends)
        self.rows = np.asarray(rows)[self.basins]
        self.observed = q[self.basins, self.ends]
        self.lags = np.arange(1 - self.length, 1)

    def __len__(self):
        return len(self.ends)

    def batch(self, start, stop, device):
        rows, ends = self.rows[start:stop], self.ends[start:stop]
        dynamic = self.dynamic[rows[:, None], ends[:, None] + self.lags]
        static = self.static[rows]
        arrays = {'x_dynamic': dynamic, 'x_static': static, 'x_land': static[:, self.land_indices]}
        require(all(np.isfinite(x).all() for x in arrays.values()), 'Nonfinite evaluation inputs')
        return {k: torch.from_numpy(np.ascontiguousarray(v)).to(device) for k, v in arrays.items()}

    def records(self, prediction):
        return {'basin_ids': np.asarray(self.ids)[self.basins], 'dates': self.dates[self.ends],
                'q_observed_mm_day': self.observed, 'q_predicted_mm_day': prediction}


def metric_rows(dataset, prediction, cfg):
    rows = []
    for j, basin in enumerate(dataset.ids):
        mask = dataset.basins == j
        rows.append({'basin_id': basin, **basin_metrics(dataset.observed[mask], prediction[mask],
                    min_days=cfg['evaluation']['min_valid_days'],
                    quantile=cfg['evaluation']['high_flow_quantile'])})
    return rows


@torch.no_grad()
def predict(model, dataset, cfg, meta, device, label="Evaluation", *, use_graph=True):
    size = cfg['training']['batch_size']
    if use_graph and model.adapter_mode != 'none' and device.type == 'cuda':
        samples = []
        for n in sorted({min(size, len(dataset)), len(dataset) % size} - {0}):
            batch = dataset.batch(0, n, device)
            samples.append(tuple(batch[k] for k in ('x_dynamic', 'x_static', 'x_land')))
        runner = GraphBackend(model, [], samples)
        model._graph_backend = runner
        model.forward = runner.forward
        for sample in samples:
            torch.testing.assert_close(model(*sample), runner.raw(*sample), rtol=1e-5, atol=1e-6)
    result = np.empty(len(dataset), dtype=np.float32)
    for start in range(0, len(dataset), size):
        stop = min(start + size, len(dataset))
        output = (predict_scaled(model, dataset.batch(start, stop, device)) * meta['q_ref_mm_day']).cpu().numpy()[:, 0]
        require(bool(np.isfinite(output).all() and (output >= 0).all()), 'Invalid prediction; never shrink the mask')
        result[start:stop] = output
        if start and (start // size) % 1000 == 0:
            print(f'{label}: {stop:,}/{len(dataset):,} windows', flush=True)
    return result


def compare_validation(records, saved, rows, expected_summary):
    for key in ('basin_ids', 'dates', 'q_observed_mm_day'):
        require(np.array_equal(records[key], saved[key]), 'Validation alignment mismatch: ' + key)
    a, b = records['q_predicted_mm_day'], saved['q_predicted_mm_day']
    require(bool(np.allclose(a, b, rtol=PRED_RTOL, atol=PRED_ATOL)),
            'Validation prediction mismatch: stop before opening Test labels')
    local = summarize(rows)
    for key in ('basins', 'valid_nse_basins', 'invalid_nse_basins', 'negative_nse_basins'):
        require(local[key] == expected_summary[key], 'Validation summary mismatch: ' + key)
    for key in ('median_nse', 'mean_nse'):
        require(abs(local[key] - expected_summary[key]) <= NSE_ATOL, 'Validation NSE mismatch: ' + key)
    return {'max_absolute_prediction_difference_mm_day': float(np.max(np.abs(a - b))),
            'local_validation': local, 'saved_validation': expected_summary}

