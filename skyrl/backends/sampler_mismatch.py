"""Per-datum summary of trainer vs sampler token logprobs.

The importance-sampling loss weights each sampled token by
exp(train_lp - sampler_lp), so any numeric gap between the inference engine
that sampled a token and the trainer's forward pass enters the gradient. The
backend returns this fixed-size summary per datum (even with
TUNIX_MINIMAL_FB_OUTPUT, which drops the per-token arrays) and the client
aggregates it per inference farm.

Positions count when the sampler logprob is nonzero and not vLLM's NaN
sentinel: prompt tokens and injected (masked) tokens carry an exact 0.0. A
sampled token whose sampler logprob is exactly 0.0 (p == 1 in float32) is
excluded too; the backend never receives the client's loss mask.

Layout (float32): [VERSION, n, sum_d, sum_abs, sum_sq, max_abs, min_d, max_d,
sum_k3, *histogram of |d| over ABS_EDGES], with d = train_lp - sampler_lp and
k3 = exp(d) - 1 - d (an estimate of KL(sampler || trainer) per token).
"""
import numpy as np

VERSION = 1.0
ABS_EDGES = (1e-4, 1e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0, 3.0)
FIELDS = ('version', 'n', 'sum_d', 'sum_abs', 'sum_sq', 'max_abs', 'min_d', 'max_d', 'sum_k3')
LENGTH = len(FIELDS) + len(ABS_EDGES) + 1
SENTINEL = -9.0e3


def summarize(train_logprobs, sampler_logprobs):
    """Return the fixed-size summary list for one datum (zeros when nothing counts)."""
    out = np.zeros(LENGTH, dtype=np.float64)
    out[0] = VERSION
    if train_logprobs is None or sampler_logprobs is None:
        return out.astype(np.float32).tolist()
    train = np.asarray(train_logprobs, dtype=np.float64).reshape(-1)
    sampler = np.asarray(sampler_logprobs, dtype=np.float64).reshape(-1)
    n = min(train.shape[0], sampler.shape[0])
    train, sampler = train[:n], sampler[:n]
    keep = (sampler != 0.0) & (sampler > SENTINEL) & np.isfinite(sampler) & np.isfinite(train)
    if not keep.any():
        return out.astype(np.float32).tolist()
    d = train[keep] - sampler[keep]
    a = np.abs(d)
    out[1:9] = (d.size, d.sum(), a.sum(), np.square(d).sum(), a.max(), d.min(), d.max(),
                (np.expm1(np.clip(d, -50.0, 50.0)) - d).sum())
    out[len(FIELDS):] = np.bincount(np.searchsorted(ABS_EDGES, a, side='right'),
                                    minlength=len(ABS_EDGES) + 1)
    return out.astype(np.float32).tolist()
