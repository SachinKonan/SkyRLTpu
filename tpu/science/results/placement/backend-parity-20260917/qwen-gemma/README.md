# Qwen and Gemma placement backend investigation

Measured 2026-09-17. Scope: the best previously recorded placement program from
each model, on ibm01, ibm04, ibm08 and ibm18. This is not a re-ranking of entire
model seed pools. Original candidates, seed records and production evaluator are
unchanged. No model training was launched.

## Full-suite results

All 24 new case evaluations passed legality checks: two models, four cases,
three executions (original CPU, original TPU v4, barrier-modified TPU v4).
The original v4 reruns reproduce both archived rewards exactly.

| Candidate | Original v4 reward | Original CPU reward | Barrier v4 reward |
| --- | ---: | ---: | ---: |
| Qwen | 0.440497617006 | 0.437608327175 | 0.437608389954 |
| Gemma | 0.439773217871 | 0.439773220753 | 0.439773217871 |

Reward is `1 / (1 + mean(proxy_cost))`, across the four cases. The unchanged
Xplace baseline is 0.438879877453. On the validated executions of these two
programs, Gemma remains above Xplace; Qwen falls below it. See the parent directory
for Muse: original TPU 0.440627467286, CPU 0.364481449964, barrier TPU 0.364473040943.
These results invalidate using the old ordering alone to choose a model.

## Direct gradient checks

Probes use the actual candidate objective code and the ibm01 initial placement.
An independent NumPy reference differentiates weighted bounding-box wirelength,
averages tied extrema and scatter-adds pin contributions to owning cells.

| Wirelength gradient | CPU max absolute error | Original v4 error | Barrier v4 error |
| --- | ---: | ---: | ---: |
| Qwen (physical coordinates) | 2.44e-11 | 2.0914552823 | 3.94e-11 |
| Gemma (normalized coordinates) | 1.70e-10 | 1.70e-10 | 1.70e-10 |

Qwen's forward wirelength agrees at about 0.068460792. Its original v4 gradient
is entirely nonnegative, ranging from 0 to 2.0915 instead of approximately
-0.000107687 to +0.000107687. Wrapping the gathered pin coordinates in
`jax.lax.optimization_barrier` restores the derivative. This is the same class
of composed gather/segment-reduction compilation failure observed in Muse.
The exact internal compiler pass and affected version range remain unidentified.

Gemma applies segment min/max to x and y separately; Qwen applies them to a
two-column pin array. Rewriting Qwen's probe to use separate x/y reductions,
without a barrier, also restores the reference gradient (max error 3.94e-11).
This establishes a working alternative formulation on this runtime, not a claim
that all two-dimensional segment reductions fail or all one-dimensional ones pass.

## Why the reward changes

Qwen executes 1,501 iterations on both backends. On ibm01, original v4 selects
checkpoint 600, with approximate objective 0.06845356. CPU and barrier v4 select
checkpoint 1,500, with objective about 0.06514046. Thus the discrepancy is not
caused by one execution exhausting its time budget sooner.

The corrected optimization reduces Qwen's approximate wirelength-heavy objective,
but worsens the trusted proxy: ibm01 proxy cost rises from 0.92381275 (original
v4) to 0.96615481 (CPU). Optimizing the candidate surrogate is not equivalent to
optimizing the real wirelength/density/congestion metric. The original reward
correctly scored its produced layout; it did not establish correct gradients.

Gemma's numerical agreement does not mean its search is effective. The ibm01 CPU
trace completes 1,000 iterations but retains checkpoint zero. The density weight
increases during optimization; its best-state comparison compares loss values
computed with different weights. It also uses the pre-update loss to select a
subsequently updated and legalized layout, without rescoring that layout. These
are candidate search/selection issues, separate from the TPU defect. The trace
shows first-checkpoint selection on CPU; an instrumented Gemma TPU trajectory
was not run. Its full-suite output scores agree across all three executions.

## Controls, resources and evidence

- Identical original sources and Xplace inputs; source/input SHA256 hashes are
  in CPU report files, with input hashes checked against the Xplace manifest.
- Pinned NumPy 2.4.6, JAX/JAXlib 0.10.1, Optax 0.2.8; TPU libtpu 0.0.41.
- TPU reruns use v4, matching the original Qwen/Gemma grading generation.
- Four CPU cores and 16 GiB per candidate; 180-second external candidate cap
  with 170 seconds supplied to the candidate. TPU allocation is one chip per
  case. Local CPU memory uses RLIMIT_AS; remote TPU workers use MemoryMax and
  CPUQuota/AllowedCPUs cgroups. These are not identical memory accounting schemes.
- Both use the trusted CPU scorer and same legality checks. Grading time is
  recorded separately from candidate time. No tests timed out.
- `comparison.json`: original candidate IDs, archived metrics, all rerun scores,
  per-component costs and runtimes. `*-remote.json`: complete remote results.
- `*-gradient*.py` and CPU/TPU logs: independent reference checks and workaround
  variants. `*-trace*.py` and logs: optimizer selection evidence.
- `*-original.py` preserves the original best sources; `*-barrier.py` contains
  diagnostic-only modifications. Runner scripts preserve the execution procedure
  and refer to scratch inputs and the immutable deployment bundle.
- `cleanup-audit.json` verifies no remaining TPU device owners or diagnostic
  systemd units on the diagnostic host after all runs.

Before interpreting model rankings or resuming placement training, choose a
validated execution policy and regrade the retained seeds under it. Applying a
workaround to one source is not an automatic fix for arbitrary generated code.
This investigation does not assess the numerical correctness of LLM training.
