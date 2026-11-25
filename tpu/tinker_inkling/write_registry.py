"""Write a client grading registry from a farm list, with the same port mapping as tunnel_ink.sh (farms sorted by name,
local port BASE + i). Only rewrites when the farm set differs, so running clients see a stable file.
usage: write_registry.py FARM_LIST.json REGISTRY_OUT.json PORT_BASE"""
import json, os, sys, time
src, out, base = sys.argv[1], sys.argv[2], int(sys.argv[3])
d = json.load(open(src)); farms = d.get('farms', d) if isinstance(d, dict) else d
farms = [dict(name=f['name'], url=f'http://127.0.0.1:{base + i}', ok=True, tunnel_ok=True, health='healthy')
         for i, f in enumerate(sorted(farms, key=lambda f: f['name']), start=1)]
try:
    cur = json.load(open(out)).get('farms', [])
except (OSError, ValueError):
    cur = None
if cur != farms:
    tmp = out + '.tmp%d' % os.getpid()
    json.dump(dict(time=time.time(), farms=farms), open(tmp, 'w')); os.replace(tmp, out)
    print(f'{time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())} wrote {out}: {len(farms)} farms (was {len(cur) if cur else 0})')
