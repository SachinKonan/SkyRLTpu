"""Replay: how often would starting step k+1 before step k's late grades land change step k+1's parents?

For each saved step k of a finished or running no-train run, this loads the PUCT tree before step k
(puct_sampler_step_k.json), re-applies step k's rollouts (trajectories step k) with the same bookkeeping
the ensemble uses, but leaves out a "late" subset, and runs the real PUCTSampler.sample_states on the
result. The picks are compared with the picks from the real tree after step k (puct_sampler_step_k+1).

Bookkeeping per step (ttt_discover rl/ensemble.py balanced_commit, sampler.py):
  failed rollout          -> n[parent and ancestors] += 1, T += 1
  per parent with >=1 success: top-k children by value become states (one update_states call):
                             m[parent] = max(m, best child), n[parent and ancestors] += 1, T += 1
  every other success     -> counted as a failed rollout (n, T += 1)
  then flush: keep top-k children per parent over the whole tree, cap the buffer.

Late sets: 'timeout' (rollouts whose grading hit a deadline), 'random p' (uncorrelated with quality),
'best p' (the slowest rollouts are the best ones; worst case).
Usage: python3 eager_replay.py <run_dir>...   (run_dir = .../client/tinker_log/<run>)
"""
import copy
import gzip
import json
import os
import random
import re
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, '/scratch/gpfs/ZHUANGL/sk7524/SkyRL-Main-infer/third_party/discover')
from ttt_discover.tinker_utils.sampler import PUCTSampler  # noqa: E402

TOPK, ELITE, PUCT_C, BUFFER, GROUPS = 2, 2, 1.0, 1000, 16
TIMEOUT = re.compile(r'time ?out|timed out|deadline', re.I)


def state(d):
    return SimpleNamespace(id=d['id'], value=d.get('value'), code=d.get('code') or '', parents=d.get('parents') or [],
                           construction=None, timestep=d.get('timestep'), parent_values=d.get('parent_values') or [])


def sampler(snapshot):
    s = PUCTSampler.__new__(PUCTSampler)
    s.env_type = SimpleNamespace(construction_length_limits=None, max_construction_len=None)
    s.problem_type, s.max_buffer_size, s.batch_size = '', BUFFER, 1
    s.topk_children, s.puct_c, s.elite_slots = TOPK, PUCT_C, ELITE
    s._states = [state(d) for d in snapshot['states']]
    s._initial_states = [state(d) for d in snapshot.get('initial_states', [])]
    s._n = dict(snapshot.get('puct_n') or {})
    s._m = dict(snapshot.get('puct_m') or {})
    s._T = int(snapshot.get('puct_T') or 0)
    s._last_sampled_states, s._last_sampled_indices, s._last_puct_stats = [], [], []
    s._lock = threading.Lock()
    s._current_step = snapshot.get('step', 0)
    s._save = lambda step: None
    return s


def picks(s):
    return [p.id for p in s.sample_states(GROUPS)]


def apply_step(before, rollouts, late):
    """Tree after step k from the tree before it, leaving out rollouts whose index is in `late`."""
    s = sampler(before)
    by_id = {x.id: x for x in s._states}
    by_parent = {}
    for i, r in enumerate(rollouts):
        if i in late:
            continue
        parent = by_id.get(r['parent_id'])
        if parent is None:
            continue
        ok = r.get('correctness') == 1 and r.get('reward') is not None
        if not ok:
            s.record_failed_rollout(parent)
            continue
        child = SimpleNamespace(id=f'replay-{i}', value=float(r['reward']), code=r.get('parsed_code') or 'x',
                                parents=[], construction=None, parent_values=[], timestep=None)
        by_parent.setdefault(parent.id, (parent, []))[1].append(child)
    for pid, (parent, children) in by_parent.items():
        children.sort(key=lambda c: c.value, reverse=True)
        chosen = children[:TOPK]
        s.update_states(chosen, [parent] * len(chosen), save=False)
        for _ in children[TOPK:]:
            s.record_failed_rollout(parent)
    s.flush(step=None)
    return s


def analyze(run_dir, seeds=30):
    run_dir = Path(run_dir)
    traj = sorted(run_dir.glob('member_*/trajectories/*_step_*.jsonl.gz'))
    out = []
    for f in traj:
        k = int(f.name.split('_step_')[1].split('.')[0])
        before, after = (run_dir / f'puct_sampler_step_{k:06d}.json', run_dir / f'puct_sampler_step_{k + 1:06d}.json')
        if not (before.exists() and after.exists()):
            continue
        b, a = json.loads(before.read_text()), json.loads(after.read_text())
        rollouts = [json.loads(line) for line in gzip.open(f, 'rt')]
        real = picks(sampler(a))
        full = apply_step(b, rollouts, set())
        valid = (full._T == a['puct_T'])
        # Replay children have new ids, so compare picks by (id if an old state, else parent+value).
        def key(s, ids):
            x = next(z for z in s._states if z.id == ids)
            return x.id if not x.id.startswith('replay-') else ('child', x.parents[0]['id'], round(x.value, 9))
        old_ids = {d['id'] for d in b['states']}
        akeys = {}
        for d in a['states']:
            akeys[d['id']] = d['id'] if d['id'] in old_ids else ('child', d['parents'][0]['id'], round(d['value'], 9))
        real_k = [akeys.get(i, i) for i in real]
        full_k = [key(full, i) for i in picks(full)]
        row = dict(step=k, rollouts=len(rollouts), T_match=valid, replay_matches_real=sorted(map(str, full_k)) == sorted(map(str, real_k)))

        def changed(late):
            s = apply_step(b, rollouts, late)
            got = [key(s, i) for i in picks(s)]
            return len(set(map(str, real_k)) - set(map(str, got))), int(got[:ELITE] != real_k[:ELITE])
        timeouts = {i for i, r in enumerate(rollouts) if TIMEOUT.search(r.get('message') or '')}
        row['timeout_n'] = len(timeouts)
        row['timeout_changed'], row['timeout_elite_changed'] = changed(timeouts) if timeouts else (0, 0)
        order = sorted(range(len(rollouts)), key=lambda i: -(rollouts[i].get('reward') or float('-inf'))
                       if rollouts[i].get('correctness') == 1 else float('inf'))
        for p in (0.05, 0.10, 0.20):
            n = int(round(p * len(rollouts)))
            rng = random.Random(k)
            res = [changed(set(rng.sample(range(len(rollouts)), n))) for _ in range(seeds)]
            row[f'rand{int(p*100)}_changed_mean'] = round(sum(c for c, _ in res) / seeds, 2)
            row[f'rand{int(p*100)}_elite_changed_frac'] = round(sum(e for _, e in res) / seeds, 2)
            row[f'best{int(p*100)}_changed'], row[f'best{int(p*100)}_elite_changed'] = changed(set(order[:n]))
        out.append(row)
    return out


if __name__ == '__main__':
    for rd in sys.argv[1:]:
        name = Path(rd).name.replace('rq4-notrain-local-', '').replace('-20261007', '')
        for row in analyze(rd):
            print(name, json.dumps(row))
        sys.stdout.flush()
