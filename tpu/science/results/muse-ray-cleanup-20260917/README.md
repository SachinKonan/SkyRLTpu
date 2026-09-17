# Muse circuit Ray cleanup, 2026-09-17

Job 977 failed because a private Ray head from cancelled Qwen job 974
survived on worker 385. Its root was the old Qwen recovery directory and its
private session directory `.rtr-e818e3b7f3`. New Muse workers joined that stale
head. Current-root stop_ray could not retire another root's cluster.
Cancelled 977, inventoried exact private-session processes, verified their UID
and executable paths against the two retired roots, then stopped their process
trees on all eight hosts. A second audit found zero old-session processes.
No model files, checkpoints, shared caches or SkyPilot control Ray were removed.

Muse resubmitted as 984 using the EXACT original 977 immutable package, preserving
the common runtime. Gemma qubit job 981 acquired worker 385 after cleanup; Muse
must await another available slice. Do not interpret submission as training.

Strengthened clean_host_audit.py for future packages to reject existing private
executor Ray heads/raylets, in addition to model processes and TPU owners.
It remains read-only and does not infer that arbitrary Ray processes are safe
to terminate. Already submitted packages have not been changed.

Qwen circuit and v6e Gemma compile publication logs show create-only GCS HTTP 412
conflicts between concurrent uploads. These background errors do not stop engine
initialization; v6e Gemma engine logs show successful sequential bucket compilation.
The existing checksum gate can still reject nonmatching contents for a raced key;
no overwrite or relaxation of checksum checks was made.
