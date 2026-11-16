"""Seed-schedule grid scheduler for Inkling-2-Small Erdős runs (stateless; safe to restart at any time).

Design (user, 2026-11-16): 90 model-steps per arm, arms nested on shared seed runs.
  A  3 seeds x 30   B  6 x 15   C  9 x 10
  prune-base: 9 seeds to step 5; the top 3 (best C5 at step 5) split 45 more steps by w
  clone-base: 9 seeds to step 5; the top 3 continue to 10; 6 clones (6*w copies of the top 3), each = the source's
              step-5 weights with a FRESH optimizer + the source's step-5 tree with PUCT visit counts reset, to step 10
  w = softmax(beta * value) over the top 3 (value = -C5) with the ONE-BIT beta (TTT-Discover's entropic rule, the
  same solver as the advantage): the smallest beta with KL(w || uniform) = log 2, beta capped at 1e6 (gaps below
  ~1e-6 C5 soften toward even). Integer splits by largest remainder; prune-base caps every seed at SEED_CAP (30)
  total steps and passes overflow down the ranking.
Mapping to runs: seed run <G>-s<i> (LoRA + start-tree seed i) runs to max over the arms that use it:
  s1-3 -> 30 (A), s4-6 -> 15 (B), s7-9 -> 10 (C); after the step-5 decision a top-3 seed -> max(target, 5 + n_i).
  Clones <G>-c<j>-s<src> start at step 5 from prepared run dirs and run to 10.
Every loop: read each run's global_step from disk; if below its target and its unit is not running, (re)start it
with run_service.sh (the client resumes from its last checkpoint). decision.json is written once and never changed.

  grid.py status            show the plan, progress and unique steps used
  grid.py loop              reconcile every LOOP_S seconds (run as the systemd unit tinker-grid)
  grid.py once [--dry]      one reconcile pass (--dry: print what it would do)
"""
import json, math, os, re, shutil, subprocess, sys, time
from pathlib import Path

G = os.environ.get('GRID', 'g1')
RUNS = Path(os.environ.get('GRID_RUNS', '/scratch/gpfs/ZHUANGL/sk7524/tinker-inkling-runs'))
OPS = Path('/scratch/gpfs/ZHUANGL/sk7524/tinker-inkling/ops')
GRID_DIR = RUNS / f'grid-{G}'
DECISION = GRID_DIR / 'decision.json'
LOG = GRID_DIR / 'grid.log'
SEED_CAP = int(os.environ.get('GRID_SEED_CAP', '30'))   # prune-base: no seed beyond this step
FORK = int(os.environ.get('GRID_FORK', '5'))
PRUNE_EXTRA = int(os.environ.get('GRID_PRUNE_EXTRA', '45'))
CLONES = int(os.environ.get('GRID_CLONES', '6'))
CLONE_TO = int(os.environ.get('GRID_CLONE_TO', '10'))
SEED_TARGET = ({int(a): int(b) for a, b in (kv.split(':') for kv in os.environ['GRID_TARGETS'].split(','))}
               if os.environ.get('GRID_TARGETS') else
               {**{i: 30 for i in (1, 2, 3)}, **{i: 15 for i in (4, 5, 6)}, **{i: 10 for i in (7, 8, 9)}})
LOOP_S = int(os.environ.get('GRID_LOOP_S', '120'))
# Launch gate: every run stops at HOLD_AFTER steps until the user creates <grid dir>/RELEASE (0 disables the gate).
HOLD_AFTER = int(os.environ.get('GRID_HOLD_AFTER', '1'))
RELEASE = GRID_DIR / 'RELEASE'
# Stops written by the watchdog: RUNS/COST_STOP (spend hard stop: start nothing) and RUNS/<run>/BLOCKED (crash loop:
# do not restart that run). Delete the file to allow starts again.
COST_STOP = RUNS / 'COST_STOP'


def blocked(name):
    """A crash-loop pause written by the watchdog: '<until epoch> <reason>'. Expired pauses are removed."""
    f = RUNS / name / 'BLOCKED'
    try:
        until = float(f.read_text().split()[0])
    except FileNotFoundError:
        return False
    except (OSError, ValueError, IndexError):
        return True          # unreadable: stay blocked (the user deletes it)
    if time.time() < until:
        return True
    f.unlink(missing_ok=True)
    log(f'{name}: crash-loop pause expired; may start again')
    return False
EXTRA_ENV = [kv for kv in os.environ.get('GRID_RUN_ENV', '').split() if kv]   # e.g. smoke-test overrides


def log(msg):
    GRID_DIR.mkdir(parents=True, exist_ok=True)
    line = f'{time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())} {msg}'
    print(line, flush=True)
    with open(LOG, 'a') as f:
        f.write(line + '\n')


def seed_name(i):
    return f'{G}-s{i}'


def logdir(name):
    return RUNS / name / 'tinker_log' / name


def ledger(name):
    """Checkpoint rows of a run (what a restart resumes from)."""
    try:
        return [json.loads(l) for l in open(logdir(name) / 'member_inkling' / 'checkpoints.jsonl') if l.strip()]
    except (OSError, ValueError):
        return []


def gstep(name):
    """Committed steps = highest checkpoint batch in the ledger (global_step.json can lag it by one after a crash
    between the checkpoint row and the counter update)."""
    return max((int(r.get('batch', 0)) for r in ledger(name)), default=0)


def has_checkpoint(name, batch):
    return any(int(r.get('batch', -1)) == batch for r in ledger(name))


def is_clone(name):
    return re.fullmatch(rf'{re.escape(G)}-c\d+-s\d+', name) is not None


def snapshot(name, step):
    p = logdir(name) / f'puct_sampler_step_{step:06d}.json'
    return json.loads(p.read_text()) if p.exists() else None


def best_c5(tree):
    vals = [s['value'] for s in tree['states'] if s.get('value') is not None]
    return -max(vals) if vals else math.inf


def unit_running(name):
    """True unless systemd says explicitly 'inactive' or 'failed'. Anything else (active, activating, deactivating,
    reloading, empty output from a bus timeout) is treated as running, so the grid never stops a unit it cannot see."""
    st = subprocess.run(['systemctl', '--user', 'is-active', f'tinker-run-{name}.service'],
                        capture_output=True, text=True).stdout.strip()
    return st not in ('inactive', 'failed')


def largest_remainder(total, weights):
    raw = [total * w for w in weights]
    out = [int(math.floor(x)) for x in raw]
    for i in sorted(range(len(raw)), key=lambda i: raw[i] - out[i], reverse=True)[:total - sum(out)]:
        out[i] += 1
    return out


def one_bit_beta(r, delta=math.log(2), beta_max=1e6, iters=60):
    """Smallest beta with KL(softmax(beta * r) || uniform) >= delta (one bit); beta_max when never reached.
    Same bisection as ttt_discover.rl.train._solve_one_bit_beta."""
    k = len(r)
    if k < 2:
        return 0.0
    log_k, m = math.log(k), max(r)

    def kl(b):
        lg = [b * (x - m) for x in r]
        z = math.log(sum(math.exp(v) for v in lg))
        q = [math.exp(v - z) for v in lg]
        return sum(qi * (math.log(qi) + log_k) for qi in q if qi > 0)

    lo, hi = 0.0, 1.0
    if kl(hi) < delta:
        while hi < beta_max and kl(hi) < delta:
            hi *= 2.0
        if kl(hi) < delta:
            return hi
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if kl(mid) < delta:
            lo = mid
        else:
            hi = mid
    return hi


def capped_split(total, weights, cap):
    """Largest-remainder split of total by weights (ranked order), each share <= cap, overflow passed down."""
    alloc = largest_remainder(total, weights)
    overflow = 0
    for i in range(len(alloc)):
        alloc[i] += overflow
        overflow = max(0, alloc[i] - cap)
        alloc[i] = min(alloc[i], cap)
    if overflow:
        raise ValueError(f'cannot place {total} steps under cap {cap} x {len(alloc)}')
    return alloc


def decide():
    """At step 5 of all 9 seeds: rank, weights, prune steps and clone copies. Written once."""
    if DECISION.exists():
        return json.loads(DECISION.read_text())
    # Decide only on COMMITTED step-5 state: the step-5 tree is written before step 5's training and checkpoint, and
    # a crash in between would redo step 5 with a different tree.
    if not all(has_checkpoint(seed_name(i), FORK) for i in range(1, 10)):
        return None
    trees = {i: snapshot(seed_name(i), FORK) for i in range(1, 10)}
    if any(t is None for t in trees.values()):
        return None
    c5 = {i: best_c5(t) for i, t in trees.items()}
    top = sorted(c5, key=c5.get)[:3]
    best = c5[top[0]]
    values = [-c5[i] for i in top]
    beta = one_bit_beta(values)
    raw = [math.exp(beta * (v - max(values))) for v in values]
    w = [x / sum(raw) for x in raw]
    d = dict(grid=G, time=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), fork_step=FORK,
             weights_rule='one_bit_entropic', beta=beta, seed_cap=SEED_CAP,
             c5_at_fork={str(i): c5[i] for i in range(1, 10)}, top3=top, weights=w,
             prune_extra_steps=dict(zip(map(str, top), capped_split(PRUNE_EXTRA, w, SEED_CAP - FORK))),
             clone_copies=dict(zip(map(str, top), largest_remainder(CLONES, w))))
    GRID_DIR.mkdir(parents=True, exist_ok=True)
    tmp = DECISION.with_suffix('.tmp')
    tmp.write_text(json.dumps(d, indent=1))
    os.replace(tmp, DECISION)
    log(f'DECISION: top3 {top} c5 {[c5[i] for i in top]} w {[round(x, 3) for x in w]} '
        f'prune {d["prune_extra_steps"]} clones {d["clone_copies"]}')
    return d


def clone_names(d):
    names, j = [], 0
    for src in d['top3']:
        for _ in range(d['clone_copies'][str(src)]):
            j += 1
            names.append((f'{G}-c{j}-s{src}', src))
    return names


def prepare_clone(name, src):
    """Run dir for a clone: ledger row = source's step-5 checkpoint, tree = source's step-5 tree with visit counts
    reset, global_step 5. Idempotent; never touches an existing clone dir."""
    dst = logdir(name)
    if (dst / 'global_step.json').exists():
        return
    s = seed_name(src)
    rows = [r for r in ledger(s) if r.get('batch') == FORK and r.get('name') != 'final']
    if not rows:
        raise RuntimeError(f'{s} has no step-{FORK} checkpoint row')
    row = rows[0]
    tree = snapshot(s, FORK)
    tree['puct_n'], tree['puct_T'] = {}, 0          # visit counts reset; states, values and puct_m kept
    tmpdir = dst.with_name(dst.name + '.tmp')
    shutil.rmtree(tmpdir, ignore_errors=True)
    (tmpdir / 'member_inkling').mkdir(parents=True)
    (tmpdir / 'member_inkling' / 'checkpoints.jsonl').write_text(json.dumps(row) + '\n')
    (tmpdir / f'puct_sampler_step_{FORK:06d}.json').write_text(json.dumps(tree))
    (tmpdir / 'life_state.json').write_text(json.dumps(dict(gains=[], best=-best_c5(tree), restarts=0)))
    (tmpdir / 'clone_setup.json').write_text(json.dumps(dict(source=s, source_batch=FORK, state_path=row['state_path'],
                                                             visit_counts='reset', optimizer='fresh'), indent=1))
    (tmpdir / 'global_step.json').write_text(json.dumps(dict(global_step=FORK)))
    dst.parent.mkdir(parents=True, exist_ok=True)
    os.replace(tmpdir, dst)
    log(f'prepared clone {name} from {s} step {FORK} ({row["state_path"]})')


def plan():
    """{run name: (target step, seed, extra env)} for everything the grid needs so far."""
    p = {seed_name(i): (t, i, []) for i, t in SEED_TARGET.items()}
    d = decide()
    if d:
        for src, n in d['prune_extra_steps'].items():
            name = seed_name(int(src))
            p[name] = (max(p[name][0], FORK + n), int(src), [])
        for name, src in clone_names(d):
            p[name] = (CLONE_TO, src, [f'TTD_FRESH_OPTIMIZER_AT_BATCH={FORK}'])
    return p, d


def held():
    return HOLD_AFTER > 0 and not RELEASE.exists()


def reconcile(dry=False):
    if COST_STOP.exists():
        return
    p, d = plan()
    if held():
        p = {n: (min(t, HOLD_AFTER), seed, env) for n, (t, seed, env) in p.items()}
    for name, (target, seed, env) in sorted(p.items()):
        try:   # one failing run never blocks the others; errors go to grid.log for the watchdog
            if is_clone(name) and d:
                src = int(name.rsplit('-s', 1)[1])
                if not dry:
                    prepare_clone(name, src)
            g = gstep(name)
            if g >= target or blocked(name) or unit_running(name):
                continue
            cmd = ['bash', str(OPS / 'run_service.sh'), 'run', name, str(target), str(seed), *env, *EXTRA_ENV]
            log(f'{"DRY " if dry else ""}start {name} at step {g} -> target {target}: {" ".join(cmd[2:])}')
            if not dry:
                subprocess.run(cmd, check=False)
        except Exception as e:
            import traceback
            log(f'GRID ERROR {name}: {e!r}\n' + traceback.format_exc())


def status():
    p, d = plan()
    used = 0
    print(f'grid {G}: decision {"made" if d else "pending (needs all 9 seeds at step 5)"}'
          + (f'; HELD at step {HOLD_AFTER} until {RELEASE} exists' if held() else '; released')
          + (f'; COST STOP ({COST_STOP} exists): nothing starts' if COST_STOP.exists() else ''))
    for name, (target, seed, env) in sorted(p.items()):
        g = gstep(name)
        start = FORK if is_clone(name) else 0
        used += max(0, g - start)
        print(f'  {name:14s} step {g:2d} / {target:2d}  {"running" if unit_running(name) else "idle"}{"  BLOCKED" if blocked(name) else ""}')
    print(f'unique model-steps done: {used}')
    if d:
        print(f'decision: top3 {d["top3"]} weights {[round(x, 3) for x in d["weights"]]} '
              f'prune {d["prune_extra_steps"]} clones {d["clone_copies"]}')


if __name__ == '__main__':
    what = sys.argv[1] if len(sys.argv) > 1 else 'status'
    if what == 'status':
        status()
    elif what == 'once':
        reconcile(dry='--dry' in sys.argv)
    elif what == 'loop':
        log(f'grid {G} loop start (every {LOOP_S}s, extra env {EXTRA_ENV})')
        while True:
            try:
                reconcile()
            except Exception as e:  # keep looping; the next pass retries
                import traceback
                log(f'reconcile error {e!r} (retried next pass)\n' + traceback.format_exc())
            time.sleep(LOOP_S)
    else:
        sys.exit(f'unknown command {what}')
