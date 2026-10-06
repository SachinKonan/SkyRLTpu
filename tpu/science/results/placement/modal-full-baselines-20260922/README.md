# Full ArchGen and AbuPlace on Modal A10

User authorized full public-baseline evaluations on Modal, retaining the earlier
$100 total spending cap. This run is separate from the three Xplace seed variants.

- 34 evaluations: both pinned public placers across all 17 IBM cases, including IBM09.
- ArchGen commit 6b3661bad55d81c00fad151d81ec1e74ef0271e0; AbuPlace commit a24087c45588f1873eb2dc18293e407e5477041d.
- One A10, 8 physical CPU cores, 16 GiB RAM per task; maximum four concurrent tasks.
- Existing baseline environment: 3150s target search, 3300s candidate timeout;
  180s independent CPU scoring; 3540s function envelope. No automatic retries.
- Standard benchmark input positions; full public placement algorithms, including
  their own Xplace stages. Do not substitute the three cached training seeds.
- Canonical scorer checks shape, finiteness, unchanged fixed macros, legality,
  overlap, and proxy metrics. No post hoc legalization of baseline output.
- Same reward transformation as training: 1/(1+mean case proxy cost). Publish a
  full-suite aggregate only after all 17 cases are valid. Preserve failed reports.
- GPU speed can change the search reached under a fixed deadline. These are A10
  reproductions, not equal-compute comparisons to CPU candidates or older A100 runs.

Launch: `tpu/science/modal_baselines.py`.
App: https://modal.com/apps/sk7524/main/ap-o4xUEBsxcP2U1DjzYlmy8F
Managed local collector: `circuit-modal-baselines-v2-20260922.service` (user systemd).
Raw results and logs: `.science/modal-baselines/`.
The existing Xplace image is reused; the overlay contains only pinned ArchGen
source and trusted runners, with setuptools pinned for AbuPlace's GPU stages.
No credential files are shipped into the image.

Budget snapshot: $2.74 metered before launch, up to $53.71 for task envelopes,
plus $15 setup/billing-lag reserve, total reserved $71.45 against the $100 cap.
Billing can lag. Submitted markers prevent blind duplicate paid attempts.
At document creation the app was building its supplemental image; no completed
baseline results were yet established.

Initial app ap-tudYOIZF9fUo5q2Y86nzju stopped after container import failure,
before candidate execution. Its logs and reservations are archived under
`.science/modal-baselines/import-failure-attempt/`. The remote import now avoids
local image-builder imports; a simulated remote import passed. Billing after
that failure was $2.79. Replacement app above reuses the built image.
