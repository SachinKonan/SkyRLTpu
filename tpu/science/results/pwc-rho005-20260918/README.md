# Archived, unlaunched rho=0.05 proposal

Superseded by the approved v4 GRPO/PWC comparison with rho=0.5.
The profile is retained here as `unlaunched-profile.json` for provenance only;
it is not a launch profile. The repeatability measurements below are retained.

# Qwen Q20 adaptive PWC, rho 0.05

Prepared locally; no job submitted and no active run changed.

Profile: `tpu/swarm/ray_train/profiles/science-qubit-q20-v6e-qwen-pwc-rho005-001.json`.
Control: `science-qubit-q20-v6e-qwen-seeded-001.json`.
The experiment preserves the existing control's v6e-32 hardware; this is not a v4 fleet allocation.

The only training intervention is:
- `TTD_ADV_ESTIMATOR=piecewise_valid_entropic_centered_adaptive`
- `TTD_ADV_PIECEWISE_RHO=0.05`
- `TTD_ADV_PIECEWISE_INVALID_REWARD=0`

The learning rate remains 4e-5, with importance_sampling, 16 groups of 32,
the same native thinking/answer budgets, trainer TP8/FSDP2, inference TP4,
and 16 grading slots per host. The estimator implementation is unchanged.
The launcher now permits this explicit estimator for Q20 science qubit;
full-suite qubit and circuit PWC are not newly enabled.

Initialization must use the control's original verified seed pool:
`706e4cea5a2e324ba85221089dc61ac0427f76a1e30af4c8ba53c402d5ef58d6`.
Start from the same initial LoRA seed, not a newer best pool or trained checkpoint.
The existing checkpoint-resume option supports recovery under the new run ID;
it does not import the control's training state. Output roots, health markers,
and trainer/inference compile-cache destinations are separate.
The seed-pool contents still need staging with the normal training package before launch.

Control profile SHA256:
`c02a57e160016852d6135f6e5acd22556c291becd96021a9b6ab654926f61442`.

## Reward precision

Q20 uses 24 equally weighted cases and integer validated SWAP counts.
The common case weight cancels, yielding R=22714/(22714+total_SWAPS).
At 16464 SWAPs, one additional SWAP changes reward by approximately 1.4798e-5.
Equal totals tie exactly; no fuzzy tolerance or reward rounding was introduced.

The completed local circuit repeatability sample contains nine fixed layouts
(Xplace, AbuPlace and ArchGen on ibm01, ibm04 and ibm08), each scored three times
with fresh grader objects. All wirelength, density, congestion and proxy costs
were identical across repeats. See repeatability.json for measured values.
The ibm18 repeatability checks exceeded the bounded local audit runtime;
one objective-only pass took about 80 seconds, so no repeated ibm18 result
is claimed. This is a three-netlist repeatability sample, not a four-netlist certification.
This measures the deterministic scorer for fixed input on one local host,
not candidate time-budget sensitivity, TPU-host variation or helper/grader parity.
It does not establish a positive noise floor, so no numerical tie threshold
has been added. Circuit calibration remains separate from this Q20 trial.

## Validation

- 36 existing advantage-estimator tests passed.
- 47 profile and launcher-command tests passed, including three new tests
  covering the experiment/control delta, overlay inclusion and Q20-only scope.
- Both profiles validate, and output/compile-cache destinations differ.
- Existing circuit-helper work in this worktree was preserved.
