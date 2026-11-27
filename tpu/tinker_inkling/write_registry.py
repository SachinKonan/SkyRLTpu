"""Write a client grading registry from a farm list, with the same port mapping as tunnel_ink.sh: the sticky name->port map
portmap-<PORT_BASE>.json next to REGISTRY_OUT (portmap.py; seeded from the old name-sorted layout, so existing ports never move). Only
rewrites when the farm set or a port differs, so running clients see a stable file.
usage: write_registry.py FARM_LIST.json REGISTRY_OUT.json PORT_BASE"""
import json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from portmap import load_farms, ports_for
src, out, base = sys.argv[1], sys.argv[2], int(sys.argv[3])
src_farms = load_farms(src)
ports = ports_for(src_farms, os.path.join(os.path.dirname(os.path.abspath(out)), f'portmap-{base}.json'), base)
farms = [dict(name=n, url=f'http://127.0.0.1:{p}', ok=True, tunnel_ok=True, health='healthy')
         for n, p in sorted(ports.items(), key=lambda kv: kv[1])]
try:
    cur = json.load(open(out)).get('farms', [])
except (OSError, ValueError):
    cur = None
if cur != farms:
    tmp = out + '.tmp%d' % os.getpid()
    json.dump(dict(time=time.time(), farms=farms), open(tmp, 'w')); os.replace(tmp, out)
    print(f'{time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())} wrote {out}: {len(farms)} farms (was {len(cur) if cur else 0})')
