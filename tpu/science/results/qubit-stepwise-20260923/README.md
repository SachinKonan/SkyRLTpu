# Qubit best reward by completed step — September 23, 2026

Best-so-far search-pool reward (`pool/best_value`), higher is better. Step 0 is the imported starting pool; later rows come from completed-update metrics. These are search results, not held-out policy scores. Different starting pools and hardware/time costs mean this is not a compute-matched comparison.

| Completed steps | Qwen | Gemma | Muse |
|---:|---:|---:|---:|
| 0 | 0.520291 | 0.532940 | 0.520719 |
| 1 | 0.522295 | 0.539501 | 0.520719 |
| 2 | 0.523038 | 0.542464 | 0.521129 |
| 3 | 0.527950 | 0.543218 | 0.521129 |
| 4 | 0.533522 | 0.543682 | 0.522211 |
| 5 | 0.533522 | 0.544866 | 0.522386 |
| 6 | 0.533609 | 0.549139 | 0.523001 |
| 7 | 0.542514 | 0.551566 | 0.523450 |
| 8 | 0.545345 | 0.552501 | 0.525075 |
| 9 | 0.546168 | 0.553127 | 0.525139 |
| 10 | 0.547742 | 0.553127 | 0.525306 |
| 11 | 0.548587 | 0.553519 | 0.525492 |
| 12 | 0.548976 | 0.553519 | 0.525492 |
| 13 | 0.549645 | 0.553519 | 0.525517 |
| 14 | 0.549902 | 0.553588 | 0.525517 |
| 15 | 0.549902 | 0.553588 | 0.525517 |
| 16 | 0.550548 | 0.553984 | 0.526162 |

All three checkpoint indices include batch 16. Qwen step16 was already present in its stored metrics at this check; earlier status reporting based on the cached step15 comparison was stale. Muse completed step16 during this inspection. No cancellation or replacement launch was performed.

Initial pool sizes: Qwen 46, Gemma 111, Muse 82. See `comparison.json` for full precision, generation-pinned source provenance, and checkpoint indices.
