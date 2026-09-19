"""Macro placement guided by the injected Evaluator."""
import time
import numpy as np


def place(problem: dict, seed: int, *, time_budget_s: float) -> dict:
    rng = np.random.default_rng(seed)
    pos = np.array(problem['initial_positions'], dtype=np.float64, copy=True)
    sizes = np.asarray(problem['sizes'], dtype=np.float64)
    canvas = np.asarray(problem['canvas'], dtype=np.float64)
    fixed = np.asarray(problem['fixed'], dtype=bool)
    num_hard = int(problem['num_hard'])
    M = pos.shape[0]

    margin = 1e-7
    max_dim = float(np.max(canvas))

    hard_movable = np.where((~fixed) & (np.arange(M) < num_hard))[0]
    soft_movable = np.where((~fixed) & (np.arange(M) >= num_hard))[0]
    movable = np.concatenate([hard_movable, soft_movable])

    if movable.size == 0:
        return {"positions": pos.astype(np.float32)}

    best_pos = pos.copy()
    deadline = time.monotonic() + max(0.0, time_budget_s - 1.5)
    start = time.monotonic()
    phase1_end = start + max(0.0, time_budget_s - 1.5) * 0.7

    step_hard = max_dim * 0.02
    step_soft = max_dim * 0.03
    T = 1.0
    T_min = 1e-4

    half_sizes = sizes * 0.5

    with Evaluator(problem, pos) as ev:
        cur_cost = float(ev.score()['proxy_cost'])
        best_cost = cur_cost
        it = 0

        while time.monotonic() < deadline:
            it += 1
            now = time.monotonic()
            explore = now < phase1_end

            if explore:
                T = max(T_min, T * 0.9995)
                sh = step_hard
                ss = step_soft
                n_h = min(40, hard_movable.size)
                n_s = min(96, soft_movable.size)
                cand_per = 3
            else:
                T = 1e-6
                sh = max_dim * 0.002
                ss = max_dim * 0.0035
                n_h = min(24, hard_movable.size)
                n_s = min(64, soft_movable.size)
                cand_per = 2

            ids_all = []
            props_all = []

            if n_h > 0:
                ids_h = rng.choice(hard_movable, size=n_h, replace=False)
                ids_h_rep = np.repeat(ids_h, cand_per)
                noise = rng.normal(scale=sh, size=(n_h * cand_per, 2))
                mix = rng.random(n_h * cand_per) < 0.2
                if np.any(mix):
                    noise[mix] = rng.uniform(-3 * sh, 3 * sh, size=(int(mix.sum()), 2))
                prop_h = pos[ids_h_rep] + noise
                ids_all.append(ids_h_rep)
                props_all.append(prop_h)

            if n_s > 0:
                ids_s = rng.choice(soft_movable, size=n_s, replace=False)
                ids_s_rep = np.repeat(ids_s, cand_per)
                noise = rng.normal(scale=ss, size=(n_s * cand_per, 2))
                mix = rng.random(n_s * cand_per) < 0.15
                if np.any(mix):
                    noise[mix] = rng.uniform(-3 * ss, 3 * ss, size=(int(mix.sum()), 2))
                prop_s = pos[ids_s_rep] + noise
                ids_all.append(ids_s_rep)
                props_all.append(prop_s)

            if not ids_all:
                break

            ids_all = np.concatenate(ids_all)
            props_all = np.concatenate(props_all)

            # bounds
            hs = sizes[ids_all] * 0.5
            min_xy = props_all - hs
            max_xy = props_all + hs
            bounds_ok = np.all(min_xy >= margin, axis=1) & np.all(max_xy <= canvas - margin, axis=1)

            legal_mask = bounds_ok.copy()
            hard_mask = ids_all < num_hard
            if np.any(hard_mask):
                idx = np.where(hard_mask)[0]
                ids_hcand = ids_all[idx]
                prop_hcand = props_all[idx]
                cand_sizes = sizes[ids_hcand]
                hard_pos = pos[:num_hard]
                hard_sizes = sizes[:num_hard]

                dx = np.abs(prop_hcand[:, None, 0] - hard_pos[None, :, 0])
                dy = np.abs(prop_hcand[:, None, 1] - hard_pos[None, :, 1])
                w_sum = (cand_sizes[:, 0, None] + hard_sizes[None, :, 0]) * 0.5
                h_sum = (cand_sizes[:, 1, None] + hard_sizes[None, :, 1]) * 0.5
                overlap = (dx < w_sum) & (dy < h_sum)
                rows = np.arange(len(ids_hcand))
                overlap[rows, ids_hcand] = False
                overlap_any = np.any(overlap, axis=1)
                legal_mask[idx] = bounds_ok[idx] & (~overlap_any)

            legal_ids = ids_all[legal_mask]
            legal_pos = props_all[legal_mask]
            if legal_ids.size == 0:
                step_hard *= 0.98
                step_soft *= 0.98
                continue

            scores = ev.evaluate_moves(legal_ids, legal_pos)
            proxy_costs = np.array([s['proxy_cost'] for s in scores], dtype=np.float64)
            best_idx = int(np.argmin(proxy_costs))
            best_batch_cost = float(proxy_costs[best_idx])
            delta = best_batch_cost - cur_cost

            accept = False
            if delta < -1e-12:
                accept = True
            elif T > 1e-12 and rng.random() < np.exp(-max(delta, 0.0) / (T + 1e-12)):
                accept = True

            if accept:
                chosen_id = int(legal_ids[best_idx])
                chosen_xy = legal_pos[best_idx]
                trial = ev.apply(chosen_id, chosen_xy)
                ev.commit()
                pos[chosen_id] = chosen_xy
                cur_cost = float(trial['proxy_cost'])
                step_hard *= 1.005
                step_soft *= 1.005
                if cur_cost < best_cost:
                    best_cost = cur_cost
                    best_pos = pos.copy()
            else:
                step_hard *= 0.997
                step_soft *= 0.997

            step_hard = float(min(max_dim * 0.06, max(max_dim * 1e-4, step_hard)))
            step_soft = float(min(max_dim * 0.06, max(max_dim * 1e-4, step_soft)))

            if it % 150 == 0:
                ev.rebuild()
                cur_cost = float(ev.score()['proxy_cost'])
                if cur_cost < best_cost:
                    best_cost = cur_cost
                    best_pos = pos.copy()

        ev.rebuild()

    # final legality guard
    best_pos = best_pos.copy()
    best_pos[fixed] = np.asarray(problem['initial_positions'], dtype=np.float64)[fixed]
    half = sizes * 0.5
    mins = best_pos - half
    maxs = best_pos + half
    legal = np.all(mins >= -1e-6) and np.all(maxs <= canvas + 1e-6)
    if legal:
        # hard-hard overlap check
        hard_pos = best_pos[:num_hard]
        hard_sizes = sizes[:num_hard]
        overlap_found = False
        for i in range(num_hard):
            for j in range(i + 1, num_hard):
                dx = abs(hard_pos[i, 0] - hard_pos[j, 0])
                dy = abs(hard_pos[i, 1] - hard_pos[j, 1])
                if dx < (hard_sizes[i, 0] + hard_sizes[j, 0]) * 0.5 - 1e-9 and dy < (hard_sizes[i, 1] + hard_sizes[j, 1]) * 0.5 - 1e-9:
                    overlap_found = True
                    break
            if overlap_found:
                break
        if overlap_found:
            legal = False

    if not legal or not np.isfinite(best_pos).all():
        best_pos = np.array(problem['initial_positions'], dtype=np.float64, copy=True)

    return {"positions": best_pos.astype(np.float32)}
