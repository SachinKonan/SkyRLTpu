# Muse qubit relocation to available central capacity

The third central v6e worker, replica 154, became READY and idle while the
Muse successor 1361 remained unassigned in east. The east entry was cancelled
and verified CANCELLED with no cluster before submitting the central replacement.
No running training jobs were cancelled.

The new run is `next-v6e-muse-qubit-central-10step-20260921`, with 10 steps,
priority 100, the same 48 original bootstrap seeds, fresh adapter and optimizer,
and optional farm borrowing. `profile-diff.json` records the complete difference:
zone, run ID/root, sick-marker path, and independent compilation-cache writes.
Sampling, loss, context, optimizer settings and cache-read seeds are unchanged.

The existing campaign prepare and submission functions were reused with this
artifact directory. Build ran on Slurm 14218136. Submission checks identity,
provider read access, bundle and seed SHA256 values, uploads and readback, and
active duplicate runs. `submissions.json` records the resulting job ID.

AC2 remains higher priority at 120. The AC2 shared-best second round remains
gated on all three original runs finishing 10 steps.
