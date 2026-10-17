"""CUDA graphs for the unchanged recurrent model; state_dict keys stay unchanged."""
import torch
from torch import nn
from training import rng_state, restore_rng


class _TrainView(nn.Module):
    def __init__(self, model, raw):
        super().__init__()
        self.target = model
        self.raw = raw

    def forward(self, dynamic, static, land):
        return self.raw(dynamic, static, land)


class GraphBackend:
    def __init__(self, model, train_samples, eval_samples):
        self.model, self.raw = model, model.forward
        self.train_graphs, self.eval_graphs = {}, {}
        saved_rng, mode = rng_state(), model.training
        try:
            for sample in train_samples:
                model.train()
                view = _TrainView(model, self.raw)
                self.train_graphs[sample[0].shape[0]] = torch.cuda.make_graphed_callables(view, sample, num_warmup_iters=3)
            for sample in eval_samples:
                model.eval()
                buffers = tuple(x.detach().clone() for x in sample)
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream), torch.no_grad():
                    for _ in range(3):
                        self.raw(*buffers)
                torch.cuda.current_stream().wait_stream(stream)
                graph = torch.cuda.CUDAGraph()
                with torch.no_grad(), torch.cuda.graph(graph):
                    output = self.raw(*buffers)
                self.eval_graphs[sample[0].shape[0]] = (graph, buffers, output)
        finally:
            model.zero_grad(set_to_none=True)
            model.train(mode)
            restore_rng(saved_rng)

    def forward(self, dynamic, static, land=None, *, return_diagnostics=False):
        args = (dynamic, static, land)
        size = dynamic.shape[0]
        if return_diagnostics:
            return self.raw(*args, return_diagnostics=True)
        if self.model.training and torch.is_grad_enabled() and size in self.train_graphs:
            return self.train_graphs[size](*args)
        if not self.model.training and not torch.is_grad_enabled() and size in self.eval_graphs:
            graph, buffers, output = self.eval_graphs[size]
            for dst, src in zip(buffers, args):
                dst.copy_(src)
            graph.replay()
            return output.clone()
        # Preserve semantics for diagnostic calls or shapes not prepared in advance.
        return self.raw(*args)


def attach_graphs(model, train_batches, val_batches, train_count, val_count, size):
    if model.adapter_mode == "none":
        return model
    import numpy as np
    def samples(batches, count):
        sizes = sorted({min(size, count), count % size} - {0})
        result = []
        for n in sizes:
            batch = batches(np.arange(n))
            result.append(tuple(batch[k] for k in ("x_dynamic", "x_static", "x_land")))
        return result
    runner = GraphBackend(model, samples(train_batches, train_count), samples(val_batches, val_count))
    # A plain Python object avoids adding wrapper prefixes to checkpoint parameter names.
    model._graph_backend = runner
    model.forward = runner.forward
    return model
