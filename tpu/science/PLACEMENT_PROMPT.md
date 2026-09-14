Design a CPU algorithm for chip macro placement. Return one complete fenced
Python module implementing:

```python
def place(problem: dict, seed: int, *, time_budget_s: float) -> dict:
    # Return all hard-macro and soft-cluster centers, in input order.
    return {"positions": positions}  # numeric array [M, 2], x then y
```

Optimize the Partcl/TILOS Tier 1 proxy:

    cost = wirelength + 0.5 * density + 0.5 * congestion

The evaluator computes these normalized components with its pinned TILOS
implementation. Lower cost is better. Across the supplied evaluation cases,
reward = max(1e-6, 1 / (1 + mean(cost))). Any invalid case makes the whole
submission's reward zero. A cost of 1 gives reward 0.5; cost 2 gives 1/3.
No baseline division is applied. Feedback includes each component, legality
errors, case runtimes, and the suite's mean proxy cost.

All coordinates and sizes are in microns. Positions are continuous rectangle
CENTERS, not grid-cell IDs. The metric grid does not restrict legal centers.
The first H = problem["num_hard"] rows are hard macros; the remaining M-H rows
are soft standard-cell clusters. You may move either kind when not fixed.
There is no automatic soft-cluster completion after your algorithm returns.

Hard constraints:
- Return exactly M finite (x, y) centers in the original order.
- Keep the entire rectangle of EVERY macro and cluster inside the canvas.
- Preserve fixed positions exactly. Ports are fixed and are not returned.
- Hard macros must have zero overlap with every other hard macro, including
  fixed hard macros. Soft-cluster overlap is permitted and affects the proxy.
- Dimensions and the supplied pin offsets are fixed. This interface optimizes
  positions only; do not return flips, rotations, or resized soft clusters.
- Initial positions are a starting point and are not guaranteed legal.
  Leave a small positive margin to avoid float32 rounding violations.

Input schema (all arrays are NumPy arrays; all indices are zero based):
- canvas: float[2], width and height.
- initial_positions: float[M,2]; sizes: float[M,2], width and height.
- fixed: bool[M]; num_hard: integer H.
- port_positions: float[P,2].
- net_offsets: int[K+1], compressed offsets into the endpoint arrays below.
- pin_owner: int[L]; pin_offset: float[L,2]; net_weights: float[K].
  Net k contains endpoints [net_offsets[k]:net_offsets[k+1]], driver first,
  followed by sinks. Preserve this order for routing calculations. An owner
  below M refers to a macro/cluster; owner M+j refers to port j. A pin's
  location is its owner's center plus its offset. Multi-pin nets, repeated
  owners, and non-unit net weights are preserved.
- wirelength_normalizer: positive float used by the native weighted HPWL.
- grid_shape: int[2], rows and columns used for density and congestion.
- routes_per_micron: float[2], horizontal and vertical routing capacities.
- macro_routing_allocation: float[2]; congestion_smoothing_range: integer.
- schema_version: "partcl_cpu_positions_v1".

Allowed imports: numpy, scipy, networkx, and these standard-library modules:
math, random, time, itertools, functools, collections, heapq, bisect, array,
statistics, dataclasses, typing, enum, copy, operator. Seed your random choices
from the supplied seed. Use time.monotonic() to manage the supplied budget.
No PyTorch, JAX, GPU/TPU, external placer, compilation, filesystem, network,
external process, dynamic code execution, or instance-name lookup is allowed.

Resource profile for the initial CPU pilot: 4 CPU cores, 16 GiB total job
memory, 180 seconds per case including candidate import and initialization.
NumPy/SciPy BLAS and OpenMP pools are capped at four threads, within the same
four-core allocation. Memory is a shared process-tree allowance, not 16 GiB
per library. Respect time_budget_s; reserve time to return your best legal
placement. Final independent grading has a separate allowance.

This initial scaffold provides no in-process proxy helper. You can implement
cheap wirelength, density, or congestion estimates from the supplied data;
the independent evaluator determines the actual reward. Optimize generally
across problems, without embedded solutions or benchmark-specific branches.

You may treat congestion and density as spatial fields and apply image-processing
methods, including Gaussian smoothing, convolutions, multiscale analysis, and
spatial gradients to guide placement. Use the allowed numerical libraries for
these operations. All preprocessing and optimization count against the same
runtime and memory limits. Final placements must satisfy every legality
constraint and are scored by the trusted evaluator.
