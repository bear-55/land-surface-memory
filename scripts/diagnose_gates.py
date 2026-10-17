"""Extract local target-day forget-gate changes from the ten selected M1 runs."""
from pathlib import Path
import argparse
import json
import sys
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from evaluation import load_completed_model
from project import config, digest, read_json, save_json
from training import setup_device

SEEDS = [11, 29, 47, 65, 83, 103, 119, 137, 155, 173]
METRICS = ['net_gate_shift', 'absolute_gate_shift', 'positive_unit_fraction',
           'base_gate_mean', 'adapted_gate_mean', 'median_unit_shift']


class LastGate(torch.nn.Module):
    """Use the fitted M1 trajectory; the base gate shares its previous state."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, dynamic, static, land):
        m = self.model
        h = dynamic.new_zeros(dynamic.shape[0], m.hidden_size)
        c = dynamic.new_zeros(dynamic.shape[0], m.hidden_size)
        inputs = F.linear(dynamic, m.core.weight_ih_l0[:, :m.dynamic_size])
        st = F.linear(static, m.core.weight_ih_l0[:, m.dynamic_size:], m.core.bias_ih_l0)
        inputs = inputs + st.unsqueeze(1)
        coeff = torch.tanh(m.land_encoder(land))
        for t in range(dynamic.shape[1]):
            gates = inputs[:, t] + F.linear(h, m.core.weight_hh_l0, m.core.bias_hh_l0)
            il, fl, gl, ol = gates.chunk(4, dim=-1)
            context = m.context_projection(torch.cat((dynamic[:, t], h), dim=-1))
            delta = m.up_projection(coeff * context)
            i, f, g, o = torch.sigmoid(il), torch.sigmoid(fl+delta), torch.tanh(gl), torch.sigmoid(ol)
            c = f*c + i*g
            h = o*torch.tanh(c)
        q = F.softplus(m.head(m.head_dropout(h)))
        return torch.cat((q, f, torch.sigmoid(fl)), dim=1)


class GraphRunner:
    def __init__(self, diagnostic, sample):
        self.buffers = tuple(v.clone() for v in sample)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream), torch.no_grad():
            for _ in range(3):
                diagnostic(*self.buffers)
        torch.cuda.current_stream().wait_stream(stream)
        self.graph = torch.cuda.CUDAGraph()
        with torch.no_grad(), torch.cuda.graph(self.graph):
            self.output = diagnostic(*self.buffers)

    def __call__(self, sample):
        for dst, src in zip(self.buffers, sample):
            dst.copy_(src)
        self.graph.replay()
        return self.output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path, default=ROOT/'runs/formal_pub_527')
    parser.add_argument('--predictions', type=Path, default=ROOT/'runs/evaluation_test')
    parser.add_argument('--cache', type=Path, default=ROOT/'data/formal')
    parser.add_argument('--output', type=Path, default=ROOT/'runs/gates')
    parser.add_argument('--seeds', type=int, nargs='+', default=SEEDS)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--batch-size', type=int, default=64)
    args = parser.parse_args()
    if args.batch_size < 1 or not set(args.seeds).issubset(SEEDS):
        parser.error('Use a positive batch size and the documented seed roster.')
    args.output.mkdir(parents=True, exist_ok=True)
    cfg, meta = config(), read_json(args.cache/'metadata.json')
    device = setup_device(args.device)
    with np.load(args.cache/'formal.npz', allow_pickle=False) as z:
        dynamic, static, cache_dates = z['x_dynamic'], z['x_static'], z['dates']
        cache_ids, valid = z['basin_ids'].tolist(), z['dynamic_valid']
    with np.load(args.predictions/f'M1/seed_{args.seeds[0]}/predictions_test.npz', allow_pickle=False) as z:
        identity = {k: z[k].copy() for k in ['basin_ids', 'dates', 'q_observed_mm_day']}
    ids, dates = identity['basin_ids'], identity['dates']
    ends = np.searchsorted(cache_dates, dates)
    np.testing.assert_array_equal(cache_dates[ends], dates)
    rows = np.array([cache_ids.index(b) for b in ids])
    lags = np.arange(1-meta['sequence_length'], 1)
    assert ends.min() >= meta['sequence_length']-1
    def batch(start, stop):
        index = np.arange(start, stop)
        if len(index) < args.batch_size:
            index = np.pad(index, (0, args.batch_size-len(index)), mode='edge')
        r, e = rows[index], ends[index]
        assert valid[r[:, None], e[:, None]+lags].all()
        s = static[r]
        arrays = (dynamic[r[:, None], e[:, None]+lags], s, s[:, meta['land_indices']])
        return tuple(torch.from_numpy(np.ascontiguousarray(a)).to(device) for a in arrays)
    checks = []
    for seed in args.seeds:
        model, spec, result = load_completed_model(args.runs/f'M1/seed_{seed}', cfg, meta, device)
        diag = LastGate(model).eval()
        h = model.hidden_size
        with np.load(args.predictions/f'M1/seed_{seed}/predictions_test.npz', allow_pickle=False) as z:
            for key, values in identity.items():
                np.testing.assert_array_equal(values, z[key])
            reference = z['q_predicted_mm_day'].copy()
        daily = np.empty((len(ids), len(METRICS)), dtype=np.float32)
        predicted = np.empty(len(ids), dtype=np.float32)
        with torch.no_grad():
            sample = batch(0, min(args.batch_size, len(ids)))
            value = diag(*sample)
            torch.testing.assert_close(value[:, :1], model(*sample), rtol=1e-6, atol=1e-7)
            runner = GraphRunner(diag, sample) if device.type == 'cuda' else lambda x: diag(*x)
            torch.testing.assert_close(runner(sample), value, rtol=1e-5, atol=1e-6)
            for start in range(0, len(ids), args.batch_size):
                stop = min(start+args.batch_size, len(ids))
                value = runner(batch(start, stop)).cpu().numpy()[:stop-start]
                predicted[start:stop] = value[:, 0]*meta['q_ref_mm_day']
                f, base = value[:, 1:1+h].astype(float), value[:, 1+h:1+2*h].astype(float)
                shift = f-base
                daily[start:stop] = np.column_stack((shift.mean(1), np.abs(shift).mean(1),
                    (shift > 0).mean(1), base.mean(1), f.mean(1), np.median(shift, axis=1)))
        np.testing.assert_allclose(predicted, reference, rtol=1e-4, atol=1e-5)
        np.savez_compressed(args.output/f'M1_seed_{seed}_daily.npz', basin_ids=ids, dates=dates,
                            metric_names=np.array(METRICS), values=daily)
        checks.append(dict(seed=seed, checkpoint_sha256=digest(args.runs/f'M1/seed_{seed}/best.pt'),
            best_epoch=result['best_epoch'], prediction_max_difference=float(np.abs(predicted-reference).max())))
        print(json.dumps(checks[-1]), flush=True)
        del runner, diag, model
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    save_json(args.output/'diagnostic_manifest.json', dict(seeds=args.seeds, checks=checks,
        definition='sigmoid(u_f+delta)-sigmoid(u_f) at M1 actual hidden state; not an adapter-off rollout.',
        cache_sha256=digest(args.cache/'formal.npz')))


if __name__ == '__main__':
    main()
