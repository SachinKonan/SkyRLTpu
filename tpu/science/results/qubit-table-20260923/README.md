# Qubit routing table — September 23, 2026

One best overall program per model, selected by the training reward. All counts are added SWAPs from saved valid 72-case grades; this export did not independently replay our programs. Steps use completed-update numbering; state timestep15 corresponds to saved step16.

| Method | Model / configuration | Q20 | Heron | Willow | Total |
|---|---|---:|---:|---:|---:|
| Qwen RLVR | Qwen-3.5-27B (Q) | 13,239 (s16) | 42,871 (s16) | 32,061 (s16) | 88,171 (s16) |
| Gemma RLVR | Gemma4-31B (G) | 13,706 (s16) | 41,810 (s16) | 31,763 (s16) | 87,279 (s16) |
| Muse RLVR | Muse-Glimmer-30B (M) | 16,156 (s16) | 45,024 (s16) | 36,859 (s16) | 98,039 (s16) |
| MOMO | Configuration next to result | Not run (Q→M→Q) | Not run (G→Q) | Not run (G→Q) | Not run |

MOMO cross-model continuations have not been run. Muse uses the configured checkpoint meta-models/Muse-Glimmer-30B; the user-provided 31B label was corrected to match it.
