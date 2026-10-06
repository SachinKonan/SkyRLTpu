# AC2 Gemma/Muse farm opt-in — September 25, 2026

The user requested inference farms for the leading AC2 Gemma and Muse runs.
Jobs 1756 and 1757 were deployed with external_pool_updates=false,
external_pool_scheduler=false, and phase scope during the earlier v4 return.
Consequently their ingress had no Borrower and the clients had no borrowing
sampling hook. Discovery updates alone cannot enable these live processes.

Two frozen replacement bundles enable run-scoped borrowing and the hybrid
scheduler. All other source files are byte-identical (Gemma634, Muse700);
training parameters, adapters, optimizer state, search state, output namespace,
and target20 updates are preserved. All artifact reads/writes and eligible
farms are in Central2. Normal lease acquisition applies; no farm job is changed
or an existing owner's lease forcibly taken.

The supervised cutover waits for the next durable update: Gemma14, Muse18. It
requires the client checkpoint index, matching training archive and PUCT pool,
and a database snapshot at least as recent as the archive. A recovering source
can instead be replaced from its current verified checkpoint. It cancels only
1756/1757, waits for cancellation and all eight hosts to have no active jobs or
TPU owners, then submits the same run with the corrected frozen bundle. A run
already at20 is not restarted. Submission intents fail closed on ambiguity.
The monitor never deletes artifacts, releases workers, or provisions capacity.

The supervisor is `ac2-farm-enable-20260925.service` on della9. It updates only
the two replacement trainers on port31800, discovers compatible healthy farms
from the existing Central2 v4-32 pool, and preserves an owned lease's URL ahead
of available candidates. Engine/model compatibility hashes match:

- Gemma: `4970783063bc6b1fff3e541e56ec0d7a0572a27ee357280eadeb00494339eab0`.
- Muse: `b85b17feece06d34ab0496fa83374bd45f8b0de7ea1bfdf4f06807cab7d99a1d`.

Initial eligible services were Gemma1566/1567 and Muse1568/1569/1570. They are
shared: health or a supplied URL does not establish a lease or remote generation.
Actual reservation/readiness is recorded after restart in `farm-status.json`.

Evidence and executable scripts: `.science/ac2-farm-enable-20260925/`.
`latest.json` reports the cutover stage; per-model `submitted.json` gives the
replacement job ID once created. The service was verified active and seven
guard checks passed without remote mutations. At arming time both original
jobs still had13/17 updates and neither had an external farm lease.

Qwen's PWC continuation1734 is explicitly INACTIVE/held in the earlier backlog,
with9 updates and best0.9366030969355593. The better original GRPO run has6
updates and best0.9407202739465522. These are distinct lineages. The user has
been asked which Qwen lineage to resume; this farm change does not choose one.
