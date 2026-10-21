# momo_proxy: clean-room CPU proxy evaluator

Drop-in `Evaluator` for placement candidates (same API as `fast_proxy`: `score`, `apply`,
`commit`, `revert`, `rebuild`, `evaluate`, `evaluate_batch`, `evaluate_moves`).

`proxy.c` is written only from the official Python scorer
(`external/MacroPlacement/CodeElements/Plc_client/plc_client_os.py`, with the grid-cell clamp from
`macro_place/objective.py`). It does not derive from any challenge entry; `fast_proxy` wraps
AbuPlace's C evaluator and is kept for comparison only.

Moving one block updates only that block's nets and grid cells: HPWL per net, the net's routing
(2-pin, 3-pin L/T, star split for larger nets) applied directly to the smoothed V/H arrays (smoothing
is linear), density area, and hard-macro routing allocation. Each score recomputes the density top 10%
and congestion top 5% with a linear-time selection.

```sh
python tpu/science/momo_proxy/build.py
python tpu/science/momo_proxy/verify_parity.py --problems DIR --probe CHALLENGE_PROBE [--compare fast_proxy]
```
