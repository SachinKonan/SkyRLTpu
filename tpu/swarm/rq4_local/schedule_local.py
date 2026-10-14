"""Run RQ4 local no-train runs to TARGET steps per model as farm capacity allows (2 farms per run).

A run is done only when its client's global step reaches TARGET. A stopped run below TARGET is
rebuilt and relaunched (it resumes its bootstrap or its frozen step) before any new run starts;
after MAX_LAUNCHES launches it is left for a human.
"""
import json
import os
from pathlib import Path
import subprocess
import time

L = Path('/scratch/gpfs/ZHUANGL/sk7524/rq4-local')
T = Path('/scratch/gpfs/ZHUANGL/sk7524/rq4-local/scripts')
TARGET = int(os.environ.get('NUM_EPOCHS', '25'))
MAX_LAUNCHES = 40
# Groups in flight per leased farm (ingress-only override). Qwen3.5 v5p farms serve 64 seqs/engine;
# the lab measured that a farm needs the run's whole step (16 groups) in flight, not engines*seqs/32.
# Gemma farms run 16 seqs/engine with KV already 64-70% used, so raise groups in flight, not seqs: one group
# per engine (4 per farm) instead of 2, which left half of each farm's engines idle.
GROUP_CAP = {'qwen35': '16', 'gemma': '4'}
# Farms per run are no longer fixed here: a run leases farms only while it generates, as many as its
# outstanding groups need (multi_borrowing.MultiRunBorrower), and grades on any farm (open grading).
MODELS = {'qwen35': 'Qwen/Qwen3.5-27B', 'gemma': 'google/gemma-4-31B-it', 'muse': 'meta-models/Muse-Glimmer-30B'}
# Priority when farms are scarce (user, 2026-10-13): finish qubit and circuit first.
ORDER = [(p, s) for p in ('qubit', 'circuit', 'ac1', 'erdos') for s in (1, 2)]
QUEUE = {m: [f'rq4-notrain-local-{m}-{p}-s{s}-20261007' for p, s in ORDER] for m in MODELS}
# Muse qubit s2's first bootstrap died without a farm and could not resume (contract changed): new run ID.
QUEUE['muse'] = [r.replace('muse-qubit-s2-20261007', 'muse-qubit-s2-20261013') for r in QUEUE['muse']]
# Every problem runs to TARGET (25); the user restored circuit to 25 on 2026-10-13 after a brief 15 cap.
PROBLEM_TARGET = {}
# Muse ac1 is paused until every Muse qubit/circuit run reaches its target (user, 2026-10-13): restarting it
# whenever each qubit/circuit run held a farm let it take the next farm a circuit run lost.
HOLD_UNTIL_FIRST_DONE = {'muse'}


def target(run):
    return next((t for p, t in PROBLEM_TARGET.items() if f'-{p}-' in run), TARGET)
STOP = L / 'scheduler.stop'


def alive(run):
    pid = L / 'runs' / run / 'runner.pid'
    try:
        os.kill(int(pid.read_text()), 0)
        return True
    except (OSError, ValueError):
        return False


def step(run):
    path = L / 'runs' / run / 'runs' / run / 'client/tinker_log' / run / 'global_step.json'
    try:
        return int(json.loads(path.read_text())['global_step'])
    except (OSError, ValueError, KeyError):
        return 0


def launches():
    counts = {}
    try:
        for line in (L / 'logs/scheduler.jsonl').read_text().splitlines():
            event = json.loads(line)
            if event.get('event') == 'started':
                counts[event['run']] = counts.get(event['run'], 0) + 1
    except OSError:
        pass
    return counts


def log(**fields):
    with open(L / 'logs/scheduler.jsonl', 'a') as out:
        out.write(json.dumps(dict(time=time.time(), **fields)) + '\n')


while not STOP.exists():
    try:
        farms = json.loads((L / 'farms.json').read_text()).get('farms', [])
    except (OSError, ValueError):
        farms = []
    counts = launches()
    for tag, model in MODELS.items():
        ours = set(QUEUE[tag])
        usable = [f for f in farms if f.get('ok') and f.get('tunnel_ok') and model in (f.get('models') or [])
                  and (f.get('state') in ('unleased', 'expired') or (f.get('owner_run') or '').split(':')[0] in ours)]
        running = [r for r in QUEUE[tag] if alive(r)]
        # A run needs one usable farm to start; farms are shared by generation-scoped leases.
        slots = max(1, len(usable))
        resume = [r for r in QUEUE[tag] if counts.get(r) and not alive(r) and step(r) < target(r)
                  and counts[r] < MAX_LAUNCHES]
        fresh = [r for r in QUEUE[tag] if not counts.get(r)]
        # Qubit and circuit come first (user, 2026-10-13): while any of this model's qubit/circuit runs waits
        # to (re)start, its ac1/erdos runs do not take a slot.
        owners = {(f.get('owner_run') or '').split(':')[0] for f in farms if f.get('ok')}
        waiting_first = [r for r in resume + fresh if '-qubit-' in r or '-circuit-' in r] + [
            r for r in running if ('-qubit-' in r or '-circuit-' in r) and r not in owners and step(r) < target(r)]
        if tag in HOLD_UNTIL_FIRST_DONE:
            waiting_first += [r for r in QUEUE[tag] if ('-qubit-' in r or '-circuit-' in r) and step(r) < target(r)]
        for run in resume + fresh:
            if waiting_first and not ('-qubit-' in run or '-circuit-' in run):
                continue
            # A resumed run needs only one farm to make progress.
            if len(running) >= (max(1, len(usable)) if run in resume else slots):
                break
            prep = subprocess.run([str(T / 'prep_local.sh'), run], capture_output=True, text=True)
            if prep.returncode:
                log(event='prep_failed', model=tag, run=run, output=(prep.stdout + prep.stderr)[-300:])
                continue
            out = subprocess.run([str(T / 'run_local.sh'), run], capture_output=True, text=True,
                                 env=dict(os.environ, NUM_EPOCHS=str(target(run)), FARM_GROUP_CAP=GROUP_CAP.get(tag, '')))
            log(event='started', model=tag, run=run, resumed_at_step=step(run), usable_farms=len(usable),
                slots=slots, output=(out.stdout + out.stderr).strip()[-300:])
            running.append(run)
    time.sleep(60)
log(event='scheduler_stopped')
