# Muse placement CPU / TPU reward discrepancy

Investigated 2026-09-17. Original candidate and production evaluator are unchanged.

## Confirmed result

Same code and hashed Xplace inputs, matching NumPy/JAX/JAXlib/Optax versions.
Both CPU and original TPU optimizer execute 800 loop iterations. All outputs pass
the same four-netlist legality checks. The original TPU ibm01 score reproduces.

- Original TPU suite reward: 0.440627467286
- Original CPU suite reward: 0.364481449964
- TPU with one optimization barrier: 0.364473040943

## Localization

For ibm01, the forward wirelength agrees at about 0.06846079. The gradient with
respect to pin coordinates also agrees. A separate NumPy scatter-add of those
pin gradients back to their cell owners is the chain-rule reference.

CPU full-chain gradient matches that reference exactly (max absolute error 0).
TPU full-chain gradient has max absolute error 47.9989013671875: its cell
gradients range from 0 to 48 instead of approximately -0.002471 to +0.002481.
Density gradients agree closely. Small synthetic gather and segment tests pass;
the failure is in this composed compiled calculation, not every gather/reduction.

Replacing only `pins = all_c[owners] + offsets` with
`pins = jax.lax.optimization_barrier(all_c[owners] + offsets)` restores the
chain-rule gradient (max absolute error 0) on the pinned TPU runtime. An explicit
gather VJP with a cotangent barrier independently restores it. This strongly
localizes a TPU compilation/optimization defect; the exact internal XLA pass
has not been identified. It is not attributed to a known upstream issue.

## Why the faulty TPU run scored higher

The original TPU surrogate objective rises from about 0.19838 to 0.31706; the
program's best-state logic retains an early layout. CPU's objective falls to
about 0.18448 and it selects a much later layout. The candidate's approximate
wirelength/density objective does not match the trusted placement proxy. For
ibm01 CPU optimization increases the actual density cost from 0.52594 (Xplace)
to 1.66639, despite reducing wirelength. Thus correct descent on this surrogate
can worsen the scientific score.

The original reward is a correct score of its actual layout, but is not evidence
of successful gradient-based optimization. Existing model rankings need a common
validated execution backend before interpretation. This investigation does not
establish the behavior of other TPU generations, other candidates, or LLM training.

## Evidence

`comparison.json` records full-suite scores. `wire-probe.py` compares the full
chain with a separate pin-gradient accumulation using a supplied problem dict.
`barrier-probe.py` adds the barrier; `barrier-only.py` is the one-line modified
candidate. CPU and TPU logs preserve gradients and optimizer trajectories.
All probes ran in isolation with four CPU cores; TPU probes used one verified
free chip per case and the existing 16 GiB memory cgroup. No training was launched.
