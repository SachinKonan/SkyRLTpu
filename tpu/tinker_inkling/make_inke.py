"""Inkling MOMO Erdős grading list (user 2026-11-27 02:15Z, relayed by train-faster: "expand its pool to the broader
cluster"): the tvg v6e math farms from train-faster's ink-erdos-farms-plus.json, WITHOUT our 8 ink-gfarms. Those run the
full grader hash (2bb4ee34), the tvg math farms the slim one (57cc0755), and one client can grade under only one hash. So
the MOMO Erdős shim uses the slim tvg farms (the same farms the TPU models' MOMO children grade on), and our arms C/D keep
the ink-gfarms. Writes ink-erdos-slim.json only when the farm set changes; prints a line when it does.
usage: make_inke.py OUT.json"""
import json, os, sys, time
SRC = '/scratch/gpfs/ZHUANGL/sk7524/rq4-local/tvg/inkling/ink-erdos-farms-plus.json'
out = sys.argv[1]
d = json.load(open(SRC)); farms = d.get('farms', d) if isinstance(d, dict) else d
slim = sorted((f for f in farms if not f['name'].startswith('ink-gfarm')), key=lambda f: f['name'])
try:
    cur = json.load(open(out)).get('farms')
except (OSError, ValueError):
    cur = None
if cur != slim:
    tmp = f'{out}.tmp{os.getpid()}'
    json.dump(dict(time=time.time(), farms=slim), open(tmp, 'w'), indent=1); os.replace(tmp, out)
    print(f'{time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())} wrote {out}: {len(slim)} slim math farms (was {len(cur) if cur else 0})')
