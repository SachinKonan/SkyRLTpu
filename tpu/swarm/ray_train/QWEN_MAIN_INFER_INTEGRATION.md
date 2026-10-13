# Qwen training integration onto Main-infer — October 13, 2026

Base: `1b8f4158150fb133b30740780625d26694c84b8d`, the current
`SkyRL-Main-infer` worktree (`agent/rq4-notrain`) at integration time.
Branch: `agent/qwen-training-on-main-infer-20261013`.

The original comparison tree remains untouched. Its tracked and untracked
non-ignored files were preserved in commit
`b2c3e2670c5b02ceb5a1649107a119f5a3a5de17` on
`backup/qwen-comparison-before-main-infer-20261013`. Its dirty discover patch
and ignored operational scripts were separately archived under the original
worktree's `.science/main-infer-integration-20261013/`.

## Integration decisions

- Main-infer is authoritative for inference/grading transport. Keep its
  per-farm HTTP semaphore, connection-wait policy, lease-health handling,
  load-aware engine routing, dashboard, renderer and no-train/eager features.
- Grading transport receives only the additional placement infrastructure-result
  rejection. The comparison tree's competing split-control/result HTTP client
  implementation and its implementation-specific tests were not transplanted.
- Carry forward hybrid multi-farm support and verified advertised engine sizes,
  per-engine request caps, assigned one/two-host v5p trainer partitions,
  topology checks, direct discovery, regional asset preflight, cache/disk
  retention, checkpoint recovery and startup preparation.
- Shared bootstrap is earlier campaign functionality, not a new October 13
  feature. Preserve the 1,024-draft / at-most-512-valid pool and cross-seed reuse
  checks, plus operational migration/continuation validation.
- Keep Main-infer's discover base and add failure guards: failed backward
  requests cannot produce optimizer/checkpoint advancement, and strict sampler
  export errors propagate. No eager sampling is enabled for these RL profiles.
- Retain the optional packing diagnostic; it is not enabled by this integration.

## Runtime boundary

This integration does not restart, redeploy, cancel or submit any cloud job.
Existing trainers, farms and discovery/latency monitors still use the previous
frozen bundles and original worktree's state directories. New source changes
are not automatically live fixes.

Tracked scripts in `.science/qwen-comparison-20261006/` preserve the operational
implementation. They depend on private, untracked deployment registries,
checkpoint proofs, cloud credentials and archives. Do not launch them without
revalidating those inputs. In particular, `od_prepare_twohost.py` is the
historical scoped modification of an existing frozen bundle; it is not a
full-source builder for this integration. Future deployment bundles must be
built and inspected from this branch, with a matching source-overlay manifest.
Historical profile URLs, host assignments and run IDs are not a live inventory.

The grading-health separation and queued-work rebalancing discussed October 13
are not implemented here. Preserve Main-infer's behavior pending a separately
reviewed change. A transient inference pause must not be interpreted as proof
that the CPU grader itself failed.

Submodule commits are local until published; publishing the superproject alone
is insufficient. Publish the discover commit first when a remote release is
requested.
