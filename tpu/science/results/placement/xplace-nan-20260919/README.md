# ibm14 Xplace NaN diagnosis and start recovery

Confirmed 2026-09-19. This concerns precomputed Xplace starts, not generated
model programs, the fast proxy helper, JAX gradients, or the trusted scorer.

## Causal evidence

1. The original ibm14 trajectory is finite through initial global placement.
   At route-stage iteration 853, the optimizer is reset. By iteration 900,
   wirelength, objective and density weight are NaN.
2. An unchanged replay on A40 GPU 1 reproduced the same iteration-953 termination
   with NaN metrics and **exit code 0**. The original failed run used GPU 0.
   `original-repeat.log` preserves this second reproduction.
3. Minimal instrumentation stops at the first nonfinite step-size estimate:

   `step=0 backtrack=0 numerator=0.0 denominator=0.0 old_alpha=0.0 old_grad_finite=True new_grad_finite=True`

   See `minimal-instrumentation.log`. The first route-stage step already has
   zero alpha. The Nesterov adaptive update divides the norm of the position
   difference by the norm of the gradient difference; both are zero, producing
   NaN. The gradients themselves are still finite at that point.
4. Finite exported coordinates do not prove successful optimization. The failed
   ibm14 export had 2143 macros but only 868 unique centers. Export/conversion
   yielded finite numbers despite the internal nonfinite state. We reject this
   original trajectory, rather than legalizing it and calling it a good start.

## Recovery

`nesterov-positive-step.patch` is applied ONLY to a private diagnostic/recovery
Xplace copy at `.science/xplace-nan-audit-20260919/xplace`.
The human comparison repositories and their shared Xplace source are unchanged.

The patch uses the configured positive finite learning rate when the initial
step estimate is zero/nonfinite. A later degenerate estimate retains the current
positive step, but only if trial positions, gradients and objective remain finite.
It refuses to export a nonfinite trial. It does not replace NaN coordinates or
change objective weights, seed, route settings or candidate programs.

The full guarded ibm14 replay completed in 1072 iterations, with exact HPWL
911103 and overflow 0.1034. The guard triggered during initialization and
iterations 0–3. `guarded.log` preserves the complete run.
This is Xplace internal wirelength/overflow, NOT the challenge proxy score.
Independent final legality/proxy scoring follows legalization.

`check_degenerate_step.py` exercises a finite linear objective whose tiny step
is lost to float32 coordinate resolution: the original optimizer produces
nonfinite positions, while the guarded optimizer stays finite for three steps.
Run with `NUMBA_DISABLE_JIT=1 .science/venv-cuda/bin/python <script>` to avoid
unrelated import-name coupling in the existing Numba cache.

The Xplace-start wrapper now rejects logged nonfinite objective/wirelength/
density-weight metrics before legalization, even if output coordinates are finite.
Its legalization budget is configurable; the default for existing callers is
unchanged. Recovery uses a 7200-second preparation allowance, as authorized.

## In-flight recovery (at documentation time)

- ibm17: original finite raw export, same deterministic legalizer with 7200 s.
- ibm14: guarded Xplace export, same deterministic legalizer with 7200 s.
- Each must pass the independent unchanged challenge scorer before publication.
- Successful starts are written to the full17 audit's `inputs` directory, with
  explicit recovery provenance. The frozen Gemma/Qwen/Muse programs then run
  automatically using their original 180-second external candidate limit.
- Original failed reports remain preserved. No model failure is fabricated for
  a case blocked by preparation, and no failed case is omitted from full-suite
  eligibility. The supervisor will incorporate later reports into its summary.

Operational logs and PIDs are in `.science/xplace-nan-audit-20260919/`:
`recover-ibm14.log`, `recover-ibm17.log`, and their `*-launch.json` files.
The 17-case comparison continues independently.

The start-wrapper source hash differs from the initial full17 launch manifest
following this fix. Its human-method branch is unchanged. This supplemental
record documents the intervention rather than rewriting the initial provenance.
