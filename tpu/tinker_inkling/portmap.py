"""Sticky farm-name -> local-port map shared by write_registry.py and tunnel_ink.sh (2026-11-27).

Ports used to be BASE + position in the name-sorted farm list, so adding farms whose names sort into the middle moved
existing farms to new ports: clients then dropped those pools and regraded their in-flight cases. Now each list has a
portmap file (registry path + '.portmap.json', or PORTMAP env for the tunnel): a farm keeps its port for as long as the map
exists, new farms get the next free port, removed farms' ports are never reused. First use seeds the map from the old
sorted order, so nothing moves when switching over.
  python3 portmap.py FARM_LIST.json PORTMAP.json BASE   -> prints JSON {name: port} for the farms in the list (updates map)
"""
import json, os, sys


def load_farms(src):
    d = json.load(open(src))
    return d.get('farms', d) if isinstance(d, dict) else d


def ports_for(farms, map_path, base):
    try:
        m = json.load(open(map_path))
    except (OSError, ValueError):
        m = {}
    if not m:   # seed from the legacy sorted layout so existing tunnels/registries stay valid
        m = {f['name']: base + i for i, f in enumerate(sorted(farms, key=lambda f: f['name']), start=1)}
    nxt = max(list(m.values()) + [base]) + 1
    for f in sorted(farms, key=lambda f: f['name']):
        if f['name'] not in m:
            m[f['name']] = nxt; nxt += 1
    tmp = f'{map_path}.tmp{os.getpid()}'
    json.dump(m, open(tmp, 'w'), indent=0, sort_keys=True); os.replace(tmp, map_path)
    return {f['name']: m[f['name']] for f in farms}


if __name__ == '__main__':
    src, mp, base = sys.argv[1], sys.argv[2], int(sys.argv[3])
    print(json.dumps(ports_for(load_farms(src), mp, base)))
