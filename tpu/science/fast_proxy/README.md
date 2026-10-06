# CPU circuit proxy helper

Opt-in CPU circuit component, enabled with `SCIENCE_PLACEMENT_HELPER=fast_proxy_v1`.
Helper selection remains explicit. Circuit training now requires the IBM17 suite;
old prompt files and historical result artifacts are retained.

A NumPy/ctypes interface to AbuPlace's C incremental proxy evaluator. No Torch,
JAX, GPU, network, subprocess or compiler is required at evaluation time.
Concrete NumPy and JAX CPU arrays are accepted through NumPy conversion. Calls
are host-side, stateful and non-differentiable: do not place them inside jax.jit,
jax.grad or jax.vmap. JAX can generate proposals, then pass them to this helper.

## Build and use

Trusted preparation, before candidate execution:

```sh
python tpu/science/fast_proxy/build.py
```

```python
from tpu.science.fast_proxy import Evaluator

with Evaluator(problem, problem['initial_positions']) as ev:
    best = ev.score()['proxy_cost']
    trial = ev.apply(block_id, proposed_xy)
    if trial['proxy_cost'] < best:
        ev.commit()
    else:
        ev.revert()

    # Independent alternatives, each measured against the current state.
    scores = ev.evaluate_moves(block_ids, proposed_positions)
    # Complete alternative layouts (serial, bounded-memory CPU batch).
    scores = ev.evaluate_batch(layouts)
    checked = ev.rebuild()
```

All scores contain proxy_cost, wirelength_cost, density_cost, congestion_cost.
Lower is better: proxy = wirelength + 0.5*density + 0.5*congestion.
These are optimization scores, NOT legality certificates. Bounds of block
footprints, hard overlaps and independent final scoring remain the grader's
responsibility. The helper rejects nonfinite coordinates, malformed arrays,
changed fixed objects, out-of-canvas centers, and invalid block indices.

A state supports one pending move. Resolve it before another move/full scoring.
Each state is single-threaded and not safe to share across threads. Full-layout
calls restore the search state, refresh pin caches and rebuild all maps.
Automatic rebuilds occur every 128 commits or rejected probes. Independently
rebuild before selecting a final result. A caller must retain its own best
positions; the helper does not supply an optimizer or hidden labels.

The C runtime uses no OpenMP workers in our build and the Python batch creates
no workers. All helper wall time and native memory must count against candidate
limits (currently 4 CPUs, 8 GiB, 180 seconds). Those caps require the existing
worker isolation; this library alone is not a resource/security boundary.

## Verification

```sh
.science/venv/bin/python -m unittest tpu.science.fast_proxy.test_api -v
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
  .science/venv/bin/python -m tpu.science.fast_proxy.verify
```

The latter uses the published 17 Xplace layouts and two perturbed layouts per
netlist, checking all components against the pinned independent scorer with
absolute tolerance 1e-5. It also checks 200 incremental apply/revert cycles
against full scoring (tolerance 1e-7), and reports timings. It requires local
benchmark assets and the grading environment. Timings exclude imports/loading
and are local measurements, not TPU-host throughput or peak-RAM guarantees.

The adapter specifically matches the pinned NumPy 2 / float32 grader's pin
coordinate and grid-index arithmetic. It is not certified for other scorer
versions. Candidate problem geometry is already serialized at finite precision;
component comparisons are numerical, not claimed bit-for-bit identity.

## Provenance and changes

C source: https://github.com/rishivg/AbuPlace
Commit: a24087c45588f1873eb2dc18293e407e5477041d
Path: abuplace/extensions/congestion.c
Original SHA256: aea0eca9085f78293de4326662d9c709f1b825c46f2f8d4b4ee252b2297f6b40
Apache-2.0: LICENSE-AbuPlace accompanies the source.

Local changes: added cong_state_set_positions with pin-cache refresh and full
rebuild; matched pin/grid rounding to the pinned grader. Python adapter/build/
verification are new. Only scoring/state functions are bound; upstream search
routines present in the source are not exposed by the Python helper.

## Deployment

Trusted host preparation compiles the C library and records source, adapter and
binary SHA256 hashes. The worker verifies these hashes and mounts only the
adapter and shared library read-only. The candidate runner injects `Evaluator`
into the program namespace; generated programs need no helper import.
The current `placement-fast-proxy-cpu-ibm17-v2.txt` prompt documents the full contract.
The final scorer remains independent. Helper work counts against the same
4 CPU, 8 GiB and 180-second limits. Startup reference checks require the helper
identity and CPU device on every host and case before generation starts.

## Local results (2026-09-18)

Seven API tests and concrete JAX CPU input check passed. Twelve layouts across
all four netlists passed scorer comparison; maximum component error was
5.02e-7. Full evaluate-and-restore took approximately 1.3–5.7 ms; sampled
single-block apply-and-revert took 0.04–0.08 ms. See verification-results.json.
These are warm local timings, not guaranteed per-move timings for all blocks.

## IBM17 preparation (2026-09-19)

All 17 published starting layouts passed independent legality checks. Helper
components on those same layouts matched the trusted scorer within 1.83e-6
(maximum absolute component error; tolerance 1e-5). This extends baseline-layout
coverage; the previously recorded changed-layout and incremental-cycle tests
covered four cases. No claim of new TPU-host throughput measurements is made.
See `tpu/science/results/circuit-ibm17-v5p-20260919/helper-parity.json`.
