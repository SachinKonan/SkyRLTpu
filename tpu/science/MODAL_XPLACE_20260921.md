# A10 Xplace preprocessing handoff — 2026-09-21

Implementation: `modal_xplace.py`, `modal_xplace_import.py`.
Working tree: SkyRLTpu-circuit-300s-v5p32.

User authorization: run the A10 pilot, then all 51 case/variant tasks if validated;
maximum $100 total usage including credits. Central training remains dependent on
verified published starts. No TPU training was submitted by this port.

## Current evidence

The CUDA 12.6 / Python 3.11 / Torch 2.5.1+cu124 image built successfully. Xplace's
120 build steps compiled and linked for SM86, with Bison/Flex added after the
first build exposed a missing parser-generator dependency. Pinned AbuPlace GP
Python is unchanged; generated native extensions replace SM80 binaries.

Final image: im-uSarGlR07sfDeBjyvcl6gp.
Before any GPU function could start, Modal rejected the run:
`Please add a payment method to use A10G GPU functions.`
This requires the account owner to configure payment in Modal. Credits alone
have not unlocked A10 for this account. No case results, GPU memory measurements,
or canonical scores exist yet. Successful compilation is not execution proof.

## Resume

Local credentials: `SkyRLTpu-hybrid-inference-migration/.modal.toml`.
Do not copy this file into an image or print its contents.
CLI: `.science/venv-modal/bin/modal` (isolated, working installation).
Set MODAL_CONFIG_PATH to the credential path and run:

```
.science/venv-modal/bin/modal run tpu/science/modal_xplace.py
```

Pilot: ibm01 off/rudy/rudy_hv, ibm18 rudy_hv, ibm10 rudy_hv. IBM10 has the
largest macro count (2,768). Once all five reports are valid, inspect GPU memory,
canonical scores, runtime, source/native hashes, billing and Slurm inventory
before running the same command with `--full`.

One A10 / eight capped physical CPU cores / 16 GiB capped RAM per task; maximum
four concurrent containers, zero configured retries, 1,980-second function cap,
1,800-second candidate cap, 120-second independent scorer. No credential-bearing
secrets are passed to candidate containers. This container path is for pinned
public GP code; it is not the sandbox for generated training programs.

Live rates: A10 $1.10/hour, CPU $0.0473/core-hour, RAM $0.008/GiB-hour. Requested
resources total $1.6064/hour; 51 tasks all reaching the function limit would cost
$45.06 in task compute, excluding build/startup/idle. Five pilot tasks are reused.
The $100 limit is operational, not a newly installed account-wide hard budget.
Check billing and leave a conservative reserve before additional attempts.

Operational files under `.science/modal-xplace`: payload/hash manifest, billing
snapshots, rates, build logs, eventual downloaded results. Source tar is curated
and contains no symlinks or credentials. Local submission markers use exclusive
creation, so uncertain submissions require inspection rather than blind retry.

The Slurm pilot and dependent array/publisher/bundle jobs remain queued and were
not cancelled. Check them again before full Modal execution to avoid duplicates.
Download each task's raw/legalized coordinates, logs, GPU-memory series, GP
metadata and score. `modal_xplace_import.py` verifies all 51 before writing a
separate queue for the existing portfolio publisher; do not race the Slurm queue.

## Update after billing enabled

Billing now permits A10. Corrected local-root inference for Modal's /root script
mount (the first GPU attempt failed during module import; stopped explicitly).
Pilot app ap-rjOV7nVk5LCKeRZuUQ03nL completed successfully: all five cases/variants
passed canonical scoring with zero overlaps and matching native/source hashes.
IBm01 variants: 28.1/33.1/36.1 seconds total; ibm18 rudy_hv: 93.2 seconds; ibm10
rudy_hv: 242.3 seconds. Maximum sampled GPU memory across pilots: 518 MiB.
Billing snapshot before expansion: $0.39 metered usage, $0 billed after credits.

Full suite invocation started, reusing five successful pilots and submitting
46 remaining tasks, at most four concurrent A10 functions. Log full-01.log.
Slurm GPU jobs 14241324 and 14241497 held with scontrol hold to prevent duplicate
execution. Publisher/bundle jobs still depend on them; repoint/resubmit those
only once the Modal portfolio is verified. No Central training submitted yet.

## Final legalization and revised destination

User revised the next training destination to v5p-32 for Qwen, Gemma, Muse, with
inference farms, superseding Central v6e-32. Keep 10 steps. Existing v5p profiles
have external_pool_updates enabled. No training submissions yet. The held
Central bundle job must not be released as the current launch workflow.

All 51 layouts now validate. ibm17/rudy_hv reused its saved raw GPU output and
completed unchanged CPU legalization in 264.35 seconds with a 600-second budget.
Canonical scoring passed in 151.09 seconds: proxy 1.5941317081, reward 0.3854854389,
zero overlaps. Slurm CPU job 14247012. Original failed report and the retry's
provenance are preserved under results/ibm17-rudy_hv. Publication/bundling remain.
