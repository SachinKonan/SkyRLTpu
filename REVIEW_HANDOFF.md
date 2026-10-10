# FAIR Universe cosmology: review handoff

As of October 10, 2026. This is a checkpoint of implementation and research, not a claim of a completed MOMO campaign or competitive leaderboard performance.

## Worktree and review scope

- **Directory:** `/scratch/gpfs/ZHUANGL/sk7524/SkyRLTpu-fair-universe`
- **Branch:** `agent/fair-universe-momo`
- **Base commit:** `d5088ad21561ce9afd0bb1ac073639960e7f3927`
- **Task:** build a JAX/TPU environment for the [FAIR Universe cosmology challenge](https://www.codabench.org/competitions/10902/), investigate normal-data anomaly detectors, and export prediction archives for the user to submit.
- **Research status:** paused by the user. The current request is to commit progress and prepare this handoff. It does not restart experiments or authorize a submission.

Review the changes against that base, rather than against an unrelated SkyRLTpu worktree. After the checkpoint commit, `git log -1` identifies it and the following commands show its scope:

```bash
cd /scratch/gpfs/ZHUANGL/sk7524/SkyRLTpu-fair-universe
git status --short
git log -1 --oneline
git diff --stat d5088ad21561ce9afd0bb1ac073639960e7f3927 HEAD
git diff d5088ad21561ce9afd0bb1ac073639960e7f3927 HEAD -- tpu/swarm/ray_train
```

Start with [research/PROGRESS.json](research/PROGRESS.json), then the relevant round report and implementation. Older reports are dated snapshots: their statements such as "uncommitted" or "no official score" describe their creation time. The progress record and this handoff provide the later status.

## Official result and interpretation

The user reported this Codabench response for **cosmology corrected spatial**, associated with local candidate **09**:

```json
{"Score": 0.525304}
```

The user also reported a best leaderboard score of **0.7835**, leaving an absolute gap of **0.258196**. Neither score was independently checked against a Codabench submission ID in this checkpoint. The association with the local candidate name comes from the conversation; the hash of the file actually uploaded has not been verified.

This result is substantially below the reported leader. Improvements on synthetic probes did not establish competitive performance on the hidden anomalies. We have no reported official score for candidate 08, so the current result cannot isolate whether candidate 09's added regional detector helped or hurt.

| Frozen candidate | Method | Reported official score |
|---|---|---:|
| 08-cosmology-corrected-hybrid | Cosmology-corrected summary/image hybrid | Unknown |
| 09-cosmology-corrected-spatial | Hybrid combined with a regional residual detector | 0.525304, user-reported |

The two ZIPs are in `.fair/deliverables/fair-r4-2026-10-09/`. Each was rechecked locally during this checkpoint: exactly one `result.json`, containing 10,000 finite `ood_scores`. Their SHA-256 hashes are recorded in `research/PROGRESS.json`. These checks validate archive structure and identity, not detection quality or upload identity.

## Data and evaluation contract

The competition provides normal training maps and cosmological/physical labels, without labeled examples of the competition's actual anomalies. The public test is for inference only. It must not collectively fit, normalize, cluster, calibrate, rank, adapt, or select the model. The intended score is a fixed per-map function; larger means more anomalous.

- Use the independent-realization training data, mask, and supplied labels. Maps have spatial dimensions 1424 by 176.
- The public test filename is `WIDE12H_bin2_2arcmin_kappa_test_phase2_new_v2.npy`; older test versions are not the submission input.
- The fixed seed-2026 split assigns 160/32/32/32 realizations per cosmology to fitting, normal validation, calibration/development, and normal audit. That is 16,160 fitting maps and 3,232 in each remaining partition. Round-specific subsets and their use are documented in the reports.
- Add the prescribed observational noise to clean training maps. The implemented pixel noise standard deviation is `0.4 / sqrt(240)`.
- Internal anomaly scores use explicitly synthetic perturbations of held-out training maps. These are diagnostic positives, not organizer-provided OoD examples or labels inferred from the public test.
- The scoring functional averages TPR at 100 logarithmically spaced FPRs between 0.001 and 0.05. Internal uses of this functional must remain labeled as synthetic proxy results.

Normal-only checkpoint and density fitting does not make the full research process blind: synthetic diagnostic results influenced later designs. Some probe families have been inspected repeatedly and no longer constitute an untouched holdout. At FPR 0.001, 3,232 calibration maps provide only about three tail observations; apparent improvements can be unstable.

## Implementation included

| Area | Entry points | What changed |
|---|---|---|
| Candidate contract and input preparation | `tpu/science/fair_universe/contract.py`, `data.py` | Versioned dataset/split contract, source/config/data identities, finite score checks, per-map independence checks. |
| Candidate execution | `child.py`, `evaluate.py`, `ray_grade.py`, `qualify.py` | Separate fit/load/predict phases, fixed prediction batch shape, isolated candidate process, explicit TPU device assignment and trial resource limits. |
| Farm integration | `tpu/swarm/ray_train/{config,bootstrap,controller,host,build,grading_service,serving}.py` | Opt-in `fair_universe` grading family, explicit grading host ranks, resource advertisement, preparation, source bundling, and dispatch through farm grading. |
| Export | `tpu/science/fair_universe_export.py` | Submission ZIP writer enforcing exactly 10,000 finite scores in `result.json`. |
| Supplied starting package | `tpu/science/fair_universe/vendor/` | Preserved user-provided package, including its original reports and limitations. Corrected candidates live separately under `candidates/`. |
| Research | `research/fair_ae/`, `research/fair_r2/` through `research/fair_r8/` | Training, scoring, numerical audits, export/reproduction helpers, experiment plans and reports. |
| Regression tests | `tests/tpu_swarm/test_fair_universe.py` | Host assignment, resource/config rejection, independence failures, corrected spatial features, cache identity and export contract. |

The intended v5p-32 farm layout reserves distinct hosts for inference and TPU grading. Config validation requires an inference-only v5p-32 profile, explicit disjoint grading/inference ranks, no all-host bootstrap, absolute runtime/data paths, and at most four grading slots per host. Each candidate trial reserves one TPU chip, four CPUs and 16 GiB; the isolated process has matching systemd limits. This is an opt-in extension, not a rollout to the production fleet.

The isolated candidate interface is `fit(train, val, config, artifact_dir)`, `load_model(artifact_dir, config)`, and `predict(model, maps, mask, config)`. Prediction receives maps and the mask, not judge labels. Constant padding to a batch of 32 controls numerical differences caused by XLA shapes; singleton, permutation and repeat checks still test dependence on companion maps or call state.

The original supplied hybrid had a spatial feature-axis bug that mixed examples. Corrected copies use spatial axes. The package also contained neural machinery that did not automatically participate in its default final score. See [PACKAGE_REVIEW.md](research/fair_ae/PACKAGE_REVIEW.md) before interpreting the supplied package's own results.

## Research progression

| Stage | What we learned | Read next |
|---|---|---|
| Infrastructure pilot | Recorded evidence includes 30 baseline matrix trials, a completed HTTP grade, Qwen generation, and cancellation cleanup. Proxy reward selection for MOMO remains unresolved. | `.fair/live-evidence.json`; `tpu/science/fair_universe/` |
| R1 | Nine reconstruction-only, parameter-prediction and joint configurations were studied. Better representation or reconstruction loss did not reliably beat physical-summary clusters on anomaly probes. | [REPORT.md](research/fair_ae/REPORT.md) |
| R2 | Conditional summary density models helped internally. Raw image likelihood was weak; conditioning its normal baseline on power statistics made it more useful in a hybrid. | [R2 report](research/fair_r2/RESEARCH.json) |
| R3 | Noise augmentation and longer summary training helped some diagnostics. A regional residual specialist improved sensitivity to spatial modulation, with tradeoffs on other probes. | [R3 report](research/fair_r3/RESEARCH.json) |
| R4 | Continuous cosmology profiling and normal score correction produced frozen candidates 08 and 09. | [R4 report](research/fair_r4/RESEARCH.json) |
| R5–R7 | Sparse latent coordinates, terminal-noise diagnostics and joint likelihood-component modeling did not establish improvements. No new candidate was promoted. | [R5](research/fair_r5/RESEARCH.json), [R6](research/fair_r6/RESEARCH.json), [R7](research/fair_r7/RESEARCH.json) |
| R8 | Four longer-trained field models completed. The wider pair improved normal modeling loss; the final matched anomaly comparison has not been verified or analyzed. | [R8 report](research/fair_r8/RESEARCH.json) |

For R8, widths 16 and 32 were each trained with seeds 0 and 1 for 16,000 updates. On the separate 1,024-map normal audit, width 32 reduced mean MSE by about 0.25% relative to the original pair. All four passed the recorded normal-data numerical checks. This is a small normal-modeling improvement, not evidence of better OoD detection. The longer budget also changes the cosine learning-rate schedule, so it is not a pure duration ablation.

The last verified R8 likelihood snapshot was **2026-10-10 00:43:02 UTC**: 16 shards running, 16,640 of 53,760 model-observation evaluations recorded, and no completed shards. This is historical, not live status. A subsequent observation attempt failed locally because an SSH config path was missing; it did not establish a remote failure. No R8 model has been promoted into a new submission.

## Artifacts and reproducibility

Large data, environments, fitted weights, arrays, prediction archives and detailed runtime evidence stay in **ignored `.fair/`**. They are available in this local worktree but will not accompany a normal Git clone. The source commit alone cannot regenerate the frozen predictions without their fitted artifacts and data.

Useful local locations:

- `.fair/live-evidence.json`: historical TPU grading, inference and cancellation records.
- `.fair/deliverables/`: frozen prediction ZIPs and reproduction bundles from completed rounds.
- `.fair/r8-evidence/training-001.tar.gz`: verified archive of completed R8 training.
- `.fair/r8-evidence/normal-study-001.tar.gz`: verified archive of R8 normal/numerical audits.
- `.fair/r8-evidence/status-009.json`: last durable R8 status snapshot.
- `.fair/commit-checks/`: current local checks, logs and artifact hash verification.

Training and normal-study archive hashes were rechecked during this checkpoint. The final R8 likelihood archive and the normal-noise diagnostic report have not been verified as durably downloaded. Do not delete remote RAM artifacts based on the presence of the earlier archives.

The recorded TPU research environment used Python 3.12, JAX/JAXlib 0.10.1, libtpu 0.0.41, Flax 0.12.4, Optax 0.2.6 and NumPy 2.3.5. Preserve each delivered bundle's exact source and checkpoint identities for reproduction; mixing later scripts with an earlier fitted detector is not a verified reproduction.

## Validation at the checkpoint

Python source, JSON and shell syntax checks passed for 162 Python files, 24 JSON files and six shell scripts in the change set. The default local `python` was 3.9, below the repository's Python 3.11 minimum; use a modern interpreter. No source syntax was changed to accommodate Python 3.9.

The staged whitespace check reports one trailing space in the preserved `vendor/README.md:304`; authored files pass. That supplied file was kept unchanged for provenance.

The targeted run completed with **69 passed and 2 failed** in 31.80 seconds, using an isolated Python 3.12 environment under `.fair/commit-checks/venv`. Both failures are in the unchanged `test_ray_train_inference_ranks.py` and reflect inconsistencies already present in the base:

- `test_lora_slot_ablation_changes_only_the_requested_cli_argument` expects eight adapter slots, but the unchanged profile sets `max_loras=1`.
- `test_population_serving_does_not_require_a_backward_output_mode` accesses `Config.sequential_probe`, which the base config does not define.

The base config was loaded separately to verify both conditions; the test file, command builder and affected profiles were confirmed byte-identical to the base. These unrelated assertions were not edited as part of this checkpoint. All selected FAIR, grading config/service, inference-only, grading transport and dedup tests passed. The selected suite as a whole is **not fully green**, and the full repository suite was not run.

An initial install using unpinned FastAPI 0.143.0 failed collection because Ray could not serialize FastAPI's telemetry lock. The completed test run uses FastAPI 0.136.3, Starlette 1.3.1 and Pydantic 2.13.4 from the repository lockfile, with Ray 2.51.1. Logs, a full package freeze, and the base comparison are in `.fair/commit-checks/`. This CPU environment is distinct from the historical TPU runtime and does not reproduce TPU performance. Dependency selection during fresh runtime installation is a review concern.

```bash
PYTHONPATH=. OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 JAX_PLATFORMS=cpu \
  .fair/commit-checks/venv/bin/python -m pytest -q \
  tests/tpu_swarm/test_fair_universe.py \
  tests/tpu_swarm/test_farm_grading_config.py \
  tests/tpu_swarm/test_farm_grading_service.py \
  tests/tpu_swarm/test_ray_train_inference_only.py \
  tests/tpu_swarm/test_ray_train_inference_ranks.py \
  tests/tpu_swarm/test_grading_transport.py \
  tests/tpu_swarm/test_grading_dedup.py
```

These checks do not establish a fully automated MOMO loop, production farm recovery under every failure, or exact equivalence between the short grading contract and the longer research experiments. The grader's baseline contract uses fixed 250/1,000-step trials; later research rounds use separate explicit training/evaluation scripts.

## Review priorities and next decisions

1. Trace every fit, calibration, checkpoint-selection and export input. Verify test-set independence, parameter/noise conventions, score direction, and absence of cross-map feature operations.
2. Review the shared farm changes for isolation and regressions: host roles, TPU resource accounting, grading readiness, cancellation, process cleanup, and error classification. Recorded smoke tests are evidence for those executions, not exhaustive failure coverage.
3. Audit research comparison validity: normal/synthetic split reuse, seed matching, cosine schedule changes, numerical integration, tail uncertainty and repeated selection on synthetic probes.
4. Verify candidate 09's submitted identity and obtain candidate 08's official result before assigning the observed deficit to the regional component. Do not tune using unlabeled test aggregates.
5. If the user resumes research, inspect the existing R8 work before launching replacements, retrieve completed outputs with hash verification, then run the matched analysis. Preserve negative findings.
6. Decide the search reward and candidate budget before calling this a working MOMO science environment. Reconstruction/parameter loss and synthetic probe reward each encode assumptions; neither guarantees hidden-anomaly performance.

## Authorized lab boundaries

The historical grants cover only `tpuswarm-v5p32-east5a-erdos-3191` and `tpuswarm-v5p32-east5a-erdos-3242`. They were held by jobs 2317 and 2306. These are handoff identifiers, not a fresh claim that the machines or holders are still current.

Original access and restrictions are in `/scratch/gpfs/ZHUANGL/sk7524/rq4-local/lab/CODEX_HANDOVER.md` and `CODEX_HANDOVER_2.md`. Do not touch unrelated clusters, farms, SkyPilot jobs or della services. Read cloud data only from the allowed us-east5 bucket, do not write production paths, and keep large remote files in RAM-backed folders. The allocating session releases the machines; this checkpoint does not cancel their holding jobs.

R8's historical remote root is `/dev/shm/fair-r8-20261009` on erdos-3242, with collector unit `fair-r8-finish-evaluation`. Revalidate access and ownership before any resumed remote operation. Review itself can proceed entirely from this worktree and its local evidence.
