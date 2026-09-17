# Criticism of `RAY_RUNTIME_PARITY_REVIEW.md` and the ray-runtime-parity patch

Audit date: 2026-09-17. Branch `agent/ray-runtime-parity`, reviewed against
`86f476a7`. Legacy reference: `/scratch/gpfs/ZHUANGL/sk7524/SkyRLTpu-multihost`
at `b48b5b61`, plus the v22 base bundle (sha256 `515ed9a5…`).

## Verdict

The Ray executor does **not yet meet** the four-item contract, and the review
doc itself says so. The patch is a sound step and mostly matches what the doc
describes. The doc's claim that all 59 science profiles validate unchanged
checks out (in fact all 222 profiles validate), and the patch introduces no
regressions I could find. The doc's validation claims do not all hold
(section 4).

- **Blocker:** one defect the doc does not know about. Backward warmup would
  hang or crash the TPU slice on every multi-host trainer.
- **Unlisted parity gaps:** several differences from legacy in serving
  settings, client packages and thread caps (sections 2 and 3).
- **Validation claims:** the "broad CPU suite" is narrower than it sounds.

### How this was checked

- Read the full diff against `86f476a7` and every new file.
- Ran all of `tests/tpu_swarm/` on a Slurm CPU node.
- Loaded all 222 profiles through the new validation.
- Wrote a CPU-only reproduction of the warmup bug.
- Three sub-audits compared trainer/client, serving and native thinking
  against the legacy checkout and the v22 base bundle.
- No TPU jobs were run. No repo files were changed other than this one.

## 1. Scorecard against the four contract items

| # | Requirement | Verdict |
|---|---|---|
| 1 | Trainer and serving environments match legacy | **Not met.** The inventory tool works, but no baseline exists, so nothing is compared to legacy. Real differences remain (sections 2 and 3). |
| 2 | Native thinking keeps two-phase semantics | **Substantially met.** Tokens, masks, logprobs and total budget match legacy, and mismatches fail closed. A few edge cases are untested, and the new contract check has holes. |
| 3 | Same effective settings reach every process | **Partly met.** Stripping ambient env vars and rejecting conflicting overrides both work. The launch records are incomplete, and the multi-host engine path is asymmetric. |
| 4 | Owned, repeatable startup and shutdown | **Partly met.** Shutdown and final-writeback errors now fail the job, which is correct. Gates 5–8 in the doc (completion certificate, recovery, storage reclamation, idempotence) are still open, as it says. |

## 2. Defects in the patch

### Blocker: backward warmup breaks multi-host trainers

- **Mechanism:**
  - `backward_warmup.run()` calls `backend.forward_backward(...)` from inside
    `create_model` (`skyrl/backends/backward_warmup.py:47`).
  - On process 0 the backend is `DistributedTunixBackend`, so that call
    broadcasts a second RPC (`skyrl/backends/tunix_backend.py:3626`).
  - The broadcast is a device collective
    (`multihost_utils.broadcast_one_to_all`, `skyrl/backends/rpc.py:51-56`),
    plus a nested acknowledgement barrier.
- **Why it desyncs:**
  - Worker ranks run plain `TunixBackend` (`skyrl/backends/rpc.py:195-204`)
    with the same `TUNIX_BACKWARD_WARMUP=1`. They are already inside the
    forward-backward program locally and never join that broadcast.
  - Process 0 and the workers therefore launch different TPU programs, which
    the comment at `skyrl/backends/rpc.py:109-113` says poisons the slice.
  - The acknowledgement sequence numbers (`_RPC_ACK_SEQUENCE`) also diverge.
- **Reproduction (CPU only, stubs for the device work):** process 0 issues two
  extra `forward_backward` broadcasts that the workers never issue.

  ```
  == DistributedTunixBackend (process 0)
      ('P0 BROADCAST (device collective)', 'forward_backward')
      ('P0 ack-barrier', 'forward_backward')
      ('DistributedTunixBackend local fb program', 2)
      ('P0 BROADCAST (device collective)', 'forward_backward')
      ('P0 ack-barrier', 'forward_backward')
      ('DistributedTunixBackend local fb program', 2)
  == TunixBackend (worker ranks 1..3)
      ('TunixBackend local fb program', 2)
      ('TunixBackend local fb program', 2)
  ```

- **Exposure:**
  - Every science training profile uses four trainer hosts.
  - The feature is off by default, so nothing is broken today.
  - `test_backward_warmup_preserves_accumulator_and_optimizer` uses a fake
    single-process backend and cannot catch this.
- **Fix:** call the local implementation directly,
  `TunixBackend.forward_backward(backend, batch)`, because `create_model` has
  already been broadcast to every rank. Add a test that uses the distributed
  backend class.

### Medium

- **Contract check can pass when it should fail**
  (`tpu/swarm/ray_train/thinking_budget/contract.py`):
  - It assumes the Gemma and Muse completer classes inherit from the Qwen one,
    and never reads the class bases.
  - A re-parented class, an annotated override such as
    `ANSWER_CUE: str = ...`, a tuple-target or post-class assignment, or a
    changed `_native_group` transition all pass.
  - The open/end markers in `FORMATS` are never checked.
  - A parse failure surfaces as a raw `KeyError`, not a contract error.
- **Contract check covers all three models regardless of the configured one:**
  - A Muse-only drift blocks Qwen and Gemma builds.
  - A base bundle that lacks the Muse class does the same.
- **Wrong file at build time:** `build.py:22-24` reads this worktree's
  `completers.py`, not the one in the base bundle, so the doc's "actual
  packaged legacy client at build" is only true at host install.
- **New hard dependency for inference-only hosts:**
  - Inference-only and benchmark hosts that use native thinking now need
    `completers.py` present and passing the check (`host.py:177-179`).
  - The current v22 bundle passes (verified).
  - A slimmer or older bundle would fail at source setup.
- **Launch records are incomplete and slightly wrong**
  (`tpu/swarm/ray_train/launch_contract.py:43-45`):
  - Only variables with the stripped prefixes are recorded.
    `HF_HUB_OFFLINE`, the OMP/OpenBLAS thread caps, `PYTHONPATH`, `HF_HOME`,
    and `GROUP_SIZE`, `LEARNING_RATE` and the other bare-named client settings
    are all missing.
  - No record is written for the client at all (`host.py` `start_client`).
  - URL cleaning turns `sqlite:////cache/run/tinker.db` into
    `sqlite://cache/run/tinker.db`.
- **Managed-flag rejection can be bypassed** (`config.py:630-636`):
  - `-tp`, `-pp` and underscore spellings such as `--max_model_len=` get
    through validation and are appended last, so they win.
  - `--no-enable-prefix-caching` and `--no-enable-chunked-prefill` are now
    rejected, and setting the field to false only omits the flag. A profile
    can no longer force either feature off.
  - Several emitted flags are not managed: `--enable-lora`, `--download-dir`,
    `--limit-mm-per-prompt`, `--distributed-executor-backend`,
    `--data-parallel-size`.
- **Runtime baselines leave out the client:**
  - They cover only trainer and serving.
  - The client does tokenization and rendering, and the doc itself flags its
    Python 3.11 vs 3.12 difference.
  - It gets no inventory.
- **`engine_env` guard is narrow:** `VLLM_XLA_CACHE_PATH`,
  `JAX_COMPILATION_CACHE_DIR`, `VLLM_LORA_RESOLVER_CACHE_DIR`, `VLLM_PLUGINS`
  and `HF_HUB_OFFLINE` can still be overridden (`config.py:734-750`).

### Low

- **Warmup shape check is stricter than the real packer:**
  `warmup_contract.shapes` rejects profiles whose token budget is below one
  sharded row. The five `muse-grpo` profiles would be refused if warmup were
  turned on, although the backend itself pads such rows and runs.
- **Warmup repeats on every model creation:** it runs on each `create_model`
  call, including re-creation on resume.
- **Warmup helps only through the on-disk compile cache:** each
  forward-backward call begins with `jax.clear_caches()`
  (`tunix_backend.py:1610-1611`), which evicts the in-memory programs.
- **One-time re-setup on native hosts:** adding `contract.py` to the
  thinking-budget assets changes the source identity. Every native host will
  re-download and re-patch its source once, and the old source directories are
  left behind.
- **Inventory gaps:** only `.py` files are hashed;
  `tpu/thinking_budget/server.py` is not in the `sources` list and is covered
  only by `source_identity`.

## 3. Differences from legacy the doc does not list

### Serving

- **Every profile in use overrides the legacy-matching presets.**

  | flag | legacy (`cell_worker.sh`) | Ray preset | Ray profiles actually used |
  |---|---|---|---|
  | `--max-num-seqs` | qwen 128, gemma 32, muse 64 | same | 16 everywhere |
  | `--max-num-batched-tokens` | 8192 | 8192 | 1024 or 4096 |
  | `--gpu-memory-utilization` | 0.90 | 0.9 | 0.65 to 0.8 |

  - These may be deliberate, but they are not on the doc's intentional list.
  - Chunk tokens and KV pool size go into compiled shapes, so legacy XLA
    caches cannot be reused.
- **The Muse serving recipe differs.**
  - Ray pins `tokenizers==0.22.2` for all models and has no way to install
    transformers from main (`host.py:226-235`).
  - Legacy force-installed `transformers@main` plus `tokenizers>=0.23.1`.
  - The doc acknowledges this; it is still unresolved.
- **`ragged_conv1d:true` on the v6e Qwen CPU-placement profile has no stated
  justification.**
- **The overlaid `vllm_tpu_server.py` changes behavior relative to legacy:**
  the load/unload order and when the last-upload marker is used.
- **Thread caps and offline mode:** `OMP_NUM_THREADS=1`,
  `OPENBLAS_NUM_THREADS=1` and `HF_HUB_OFFLINE=1` are set in Ray and not in
  legacy.

### Trainer

- **Thread caps:** `OMP_NUM_THREADS=1` and `OPENBLAS_NUM_THREADS=1` are set in
  Ray (`commands.py:60`) and were never set by legacy.
- **`HF_HUB_OFFLINE`:** 1 in Ray, 0 in legacy.
- **`TPU_VISIBLE_CHIPS`:** Ray sets `0,1,2,3`; legacy unsets it explicitly.
- **Compile-cache thresholds:** Ray adds zero-threshold JAX persistent-cache
  settings (`commands.py:41-45`).
- **Single-host mesh:** Ray always passes explicit mesh kwargs; legacy only
  did so for multi-host. The native-v5p Gemma profile uses TP4 × FSDP1, which
  no single-host legacy run ever did.
- **Latent bug:** at TP8, Gemma would receive `base_num_kv_heads`
  (`commands.py:76-78`) where legacy uses `global_num_kv_heads`. No current
  profile triggers it.
- **Timeouts and routing:** the profiles use timeouts and watchdogs far above
  legacy (28800 s vs 300 or 1800 s) and `routing: ingress` where legacy used
  direct round-robin.
- **Cold first step:**
  - The profiles set `TTD_WARMUP_FB=0` and leave the new warmup off, so
    neither warmup runs.
  - The first real training step compiles cold.
  - Legacy defaulted the client warmup to on.
- **MaxText revision:** legacy qwen/gemma track the moving branch
  `skyrl/qwen35-dense`; Ray pins `0fd40993…`. They are equal today; legacy can
  drift.

### Client

- **Python and lock:** Python 3.12 with a separate frozen lock, against legacy
  3.11 with an unfrozen sync.
- **Package versions:**

  | Package | Legacy | Ray |
  |---|---|---|
  | tinker | 0.22.7 | 0.22.7 |
  | transformers | **5.13.0** | **5.8.0** |
  | tokenizers | 0.22.2 | 0.22.2 |
  | ray | 2.56.0 | 2.58.0 |
  | torch | 2.12.1 | 2.10.0+cpu |

  transformers is the package that renders chat templates, so it needs
  validating.
- **`TTD_KL_MEASURE_EVERY`:** defaults to 0 for every environment in Ray;
  legacy used 1 for non-Erdős environments.
- **Grader threads:**
  - The thread cap of 1 propagates into grader sandboxes.
  - The profiles use `NUM_CPUS_PER_TASK` of 2 or 4.
  - Candidate programs therefore get one OMP thread where the sandbox would
    otherwise give 2 or 4. Legacy used 1 CPU per task, so the cap had no
    effect there.

### Multi-host engine pair (gpt-oss only, not v5p)

The second host's vLLM worker is not passed through env stripping. It also
does not receive the bucket or serialize settings, so a non-default value
would apply on the first host only.

### Verified correct

- The doc's claim that "Defaults match the corresponding legacy exports" is
  right: legacy exports `CUSTOM_NUM_TOKENS_BUCKETS=""` and
  `SERIALIZE_MODEL_AND_SAMPLING=0`, and tpu-inference treats an empty string
  as unset.
- Env stripping loses nothing the trainer, engine or client needs.
- The inventory does hash the thinking-budget-patched runner, because
  tpu-inference is installed non-editable.
- The shutdown-status change behaves as described.
- All 222 profiles pass validation.

## 4. The doc's validation claims

- **The review's 11 files pass:** 229 passed, 1 skipped.
- **The whole directory does not:** 710 passed, 13 failed, 1 collection error,
  3 skipped.
  - All failures are in code the patch does not touch:
    - stale fixtures (`test_gradient_conflict.py`,
      `test_ray_train_client_dependencies.py`);
    - missing `sky` and `tpuswarm` modules, missing
      `third_party/TPUSwarm/pyproject.toml`;
    - one bash orbax test (`test_cache_setup_recovery.py`).
  - Two of them are `Host.install_client` tests for this executor. They have
    the same stale-fixture problem the patch fixed in the lifecycle tests, and
    they were left out of the doc's "broad suite".
- The live Qwen-tokenizer test remains skipped, as the doc says.

## 5. Native thinking

- **Equivalence:**
  - Native forces the transition exactly when legacy would.
  - It forces the same tokens.
  - It keeps the same total ceiling of `context_window - prompt - 50`.
  - It uses weight 0 and logprob 0 on injected tokens.
  - Any mismatch raises `NativeCompletionError` and aborts rather than
    mis-masking.
- **One real deviation:**
  - A stop or EOS sampled exactly at the cap.
  - Legacy keeps sampling or injects after EOS; native stops.
  - Native is arguably more correct, but the case is untested.
- **Other untested cases:**
  - the close marker completing exactly on the cap token in detector mode;
  - a natural finish with `len < cap`;
  - live-tokenizer decode parity.

## 6. Recommended order of work

1. Fix the warmup multi-host bug and add a test that uses the distributed
   backend class.
2. Put the serving profile overrides and the thread caps on the intentional
   list, or revert them, so they stop being silent drift.
3. Capture real legacy inventories for trainer, serving and client, and
   populate `runtime_baselines/`. Resolve the Muse tokenizers/transformers
   recipe.
4. Make the launch records complete: record the whole non-secret env, add a
   client record, and fix URL cleaning.
5. Harden `contract.py`: read class bases, handle annotated assignments, check
   only the configured model, and raise clear errors. Normalize flag spellings
   in the managed-flag check.
6. Refresh the two stale `install_client` tests and add the uncovered
   native-thinking edge-case tests.
7. Run TPU validation: a warmup smoke test on four hosts, then the doc's
   numerical-parity and completion-certificate gates.

Items 1, 4, 5 and 6 are small and testable on CPU.

---

# Second review (2026-09-17, after the agent's fix pass)

Re-reviewed the working tree after the fix pass. Re-ran the whole
`tests/tpu_swarm/` directory on a Slurm CPU node, re-validated all 222
profiles, and ran the contract check under Python 3.9 to 3.13.

## Verdict

Most findings from the first review are fixed correctly. **One fix introduced
a new blocker:** 219 of 222 profiles now launch vLLM with
`--no-enable-chunked-prefill`, which legacy never passed. One fix also rewrote
client behavior inside the Discover submodule, and that change is uncommitted.

## New blocker: `--no-enable-chunked-prefill` is now emitted for 219 profiles

- **What changed:** `commands.py:210` now appends `--no-enable-chunked-prefill`
  whenever `inference.chunked_prefill` is false. Before, false meant "pass no
  flag". (`--no-enable-prefix-caching` got the same treatment at
  `commands.py:202`, but `prefix_caching` defaults to true, so it is not hit.)
- **Why it is wrong:**
  - `chunked_prefill` defaults to false (`config.py:153`), and the comment
    above it says legacy "never" passes the flag. `OPERATIONS.md:203-204`
    states the legacy command line has "no chunked-prefill flag".
  - So false was the encoding of "leave vLLM's default alone", and vLLM's
    default has chunked prefill on. The fix silently turned it into "force
    off".
- **Exposure:** measured on every profile. 219 emit
  `--no-enable-chunked-prefill`, and in all 219 `--max-num-batched-tokens`
  (1024, 4096 or 8192) is below `--max-model-len` (22528 or 16384).
- **Expected effect:** vLLM refuses to start when chunked prefill is disabled
  and `max_num_batched_tokens < max_model_len`. If the installed version
  tolerates it, prefill scheduling still differs from every legacy run. I
  could not confirm against vLLM 0.23 source; none is installed locally.
  Either way it is a departure from legacy on every engine.
- **Tests do not catch it:** `test_runtime_parity.py:194-198` asserts the new
  flag is present, so the suite now locks the regression in.
- **Origin:** my first review noted that a profile "can no longer force either
  feature off". That was a low-value observation (legacy cannot do it either)
  and the fix over-corrected.
- **Fix:** restore omit-when-false. If an explicit off switch is wanted, make
  the field tri-state (`null` = no flag, `true`, `false`) with `null` as the
  default.

## New concern: Discover client behavior changed, uncommitted, in the submodule

`third_party/discover/ttt_discover/tinker_utils/completers.py` has working-tree
edits on top of `1d662eb`. The review doc says so honestly (lines 108-110),
but the consequences need stating.

- **Changes:**
  - The legacy algorithm was split out as `_two_phase`.
  - Native insufficient headroom now falls back to `_two_phase` per sample
    instead of raising. Not a criticism: config validation guarantees about
    6k tokens of answer room after the cap on every profile, so this path
    cannot trigger.
  - A native result with `len(tokens) == cap` now gets a legacy-style phase-2
    continuation, even after EOS.
- **Parity:** the continuation matches legacy arithmetic (same prefill rule,
  `answer_max = total - len - len(prefill)`, zero logprob and zero mask on the
  prefill). It is faithful. It also deliberately copies the legacy quirk of
  sampling past an EOS and training on those tokens with weight 1. That is a
  defensible reading of "preserve two-phase semantics", but it should be a
  decision the user makes, not a side effect.
- **Reproducibility risk:** the submodule gitlink is not advanced and the edit
  is not committed. `contract.py` pins AST hashes of these exact methods
  (`METHODS`), and the overlay ships the working-tree file. A fresh clone, or a
  `git submodule update`, produces a tree whose completers fail the contract
  check at host setup for every native training profile. Commit the submodule
  change and advance the gitlink together with `contract.py`.

## New low-severity issues

- **Contract hashes depend on the Python minor version.** `ast.dump` omits
  empty fields from 3.13 on. Measured: the check passes on 3.9, 3.10, 3.11 and
  3.12 and fails on 3.13.7 with "unreviewed two-phase method
  QwenTwoPhaseTokenCompleter._sample". The executor pins 3.12 today, so this
  is latent. It fails closed, but the message blames the client source. Pass
  `show_empty=True` on 3.13+ or hash a normalized form.
- **Method pinning makes every Discover edit to those five methods a two-repo
  change.** That is the intent; it should be written down next to `METHODS`
  with the command that regenerates the hashes.
- **Multi-host engine env hook is unverified on a cluster**
  (`tpu/vllm_tpu_server.py`):
  - `worker_process_setup_hook` is a `functools.partial` of a module function.
    When the native `server.py` imports `vllm_tpu_server` as a module, the hook
    pickles by reference and host 2's worker must be able to import it.
  - The hook deletes every `TPU_*` variable not in the driver's list. That is
    fine only if Ray assigns `TPU_VISIBLE_CHIPS` after the hook runs.
  - All `VLLM_*` driver variables, including host-1 paths, are now copied to
    host 2.
  - This affects gpt-oss v6e pair profiles only, not v5p. CPU tests use fakes.
- **Host re-setup cost:** the client venv identity changed
  (`client-frozen-v2-inventory`), and `hosts_per_engine > 1` now requires the
  source overlay. Every host rebuilds its client venv once; pair profiles get
  a new source directory. Old directories are still not reclaimed.
- **Warmup over-budget case runs redundant tiles.** When one sharded row
  exceeds the token budget, `shapes` returns `row_shard` rows, and the packer
  splits them into `row_shard` singleton tiles of the same shape. Correct, just
  repeated work.

## Fixes verified correct

- **Warmup multi-host desync:** fixed. `backward_warmup.py:51` calls
  `backend._model_pass(...)`, which only `TunixBackend` defines, so every rank
  runs locally inside the already-broadcast `create_model`. `ErrorResponse`
  results now raise. A new test builds a real `DistributedTunixBackend` for
  rank 0 and asserts no nested RPC.
- **Warmup shape check:** `shapes` now returns at least `row_shard` rows, which
  matches the packer. All profiles accept warmup, including the five
  `muse-grpo` ones.
- **Baseline role name bug (I missed this in the first review):**
  `install_role` is called with `"inference"` while baselines are keyed
  `"serving"`, so a serving baseline was never applied. `verify_runtime` now
  maps the name.
- **Client inventory:** the client venv is now inventoried on install and
  reuse, and `runtime_baselines` accepts `client`.
- **Inventory coverage:** `ttt-discover`, `transformers` and `tokenizers`
  sources are now hashed, and `tpu/thinking_budget/server.py` is in `sources`.
- **Contract check:** reads class bases, handles annotated and tuple
  assignments, rejects post-class mutation, checks open/end markers, checks
  only the configured model, wraps parse errors, and no longer requires
  Discover for inference-only sources. The build-time check against the wrong
  file was removed; the check now also runs when a source directory is reused.
- **Managed flags:** `-tp`, `-pp`, `-dp` and underscore spellings are
  normalized; the managed set now covers the other emitted flags.
- **`engine_env` guard:** cache, plugin, HF and routing keys are rejected;
  `VLLM_PLUGINS` moved to an `inference.plugins` field.
- **Gemma TP8 KV-head key:** now `global_num_kv_heads`, matching legacy.
- **Launch records:** a client record is written; thread caps, HF settings,
  `PYTHONPATH`, client knobs and profile-supplied keys are recorded. I did not
  re-check the `sqlite:////` URL mangling.
- **Pair-host env:** bucket and serialize settings now reach host 2, subject to
  the cluster caveat above.
- **Stale `install_client` tests:** refreshed and passing.

## Test results

- Whole directory: **768 passed, 11 failed, 1 collection error, 3 skipped**
  (first review: 710 passed, 13 failed). The doc reports 757 passed; tests have
  been added since.
- The 11 failures and the collection error are all unrelated to this patch and
  match the doc's description: `training_failed` fixtures (3), missing
  TPUSwarm checkout files (6), the bash orbax test (1), missing `sky` (1),
  missing `tpuswarm` (collection).
- All 222 profiles validate.

## Still open from the first review

- No legacy inventory has been captured, so contract item 1 is still unmet.
- The Muse tokenizers/transformers serving recipe is unresolved.
- The unlisted legacy differences stand: serving `max-num-seqs`, chunk tokens
  and memory utilization in every profile; thread caps; `HF_HUB_OFFLINE`;
  client transformers 5.8.0 vs 5.13.0; `TTD_KL_MEASURE_EVERY`; timeouts and
  ingress routing. The doc should list them as intentional or fix them.
- No TPU validation of any kind has run.

## Required before deployment

1. Revert the chunked-prefill (and prefix-caching) flag emission to
   omit-when-false, and fix the test that asserts the new flag.
2. Commit the Discover change in the submodule and advance the gitlink in the
   same commit as `contract.py`. Get an explicit decision on copying the
   sample-past-EOS quirk.
3. Make the contract hashes independent of the Python minor version.
