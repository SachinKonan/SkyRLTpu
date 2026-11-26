"""Checkpoint storage across all 5 Tinker keys: total per key (incl. other people's checkpoints on shared keys) and OUR
part (training runs that appear in tinker-inkling-runs ledgers/logs). Prints JSON; with --alert USD emails when our part
costs more than USD per month ($0.10/GB-month). usage: storage_check.py [--alert 10]"""
import glob, json, os, re, subprocess, sys
R = '/scratch/gpfs/ZHUANGL/sk7524/tinker-inkling-runs'
RATE = 0.10
ours = set()
for f in glob.glob(f'{R}/*/tinker_log/*/member_*/checkpoints.jsonl') + glob.glob(f'{R}/*/tinker_log/*/logs.log'):
    try:
        ours |= set(re.findall(r'tinker://([0-9a-f-]{36}:train:\d+)', open(f, errors='ignore').read()))
    except OSError:
        pass


def key_env(k):
    var = 'TINKER_API_KEY' if k == '1' else f'TINKER_API_KEY_{k}'
    return subprocess.run(['bash', '-ic', f'printf %s "${var}"'], capture_output=True, text=True).stdout


import tinker
out = {}
for k in ('1', '2', '3', 'TAI', 'HARISH'):
    os.environ['TINKER_API_KEY'] = key_env(k)
    rc = tinker.ServiceClient().create_rest_client()
    tot = mine = 0; off = 0
    while True:
        page = rc.list_user_checkpoints(limit=100, offset=off).result().checkpoints
        for c in page:
            b = c.size_bytes or 0
            tot += b
            if c.tinker_path.split('tinker://')[-1].split('/')[0] in ours:
                mine += b
        if len(page) < 100:
            break
        off += 100
    out[k] = dict(total_gb=round(tot / 1e9, 1), ours_gb=round(mine / 1e9, 1), ours_usd_month=round(mine / 1e9 * RATE, 2))
ours_usd = round(sum(v['ours_usd_month'] for v in out.values()), 2)
print(json.dumps(dict(keys=out, ours_usd_month=ours_usd)))
if '--alert' in sys.argv:
    # Per key (user 2026-11-26): email when OUR checkpoints on a key first cost more than LIM $/month, then again at every
    # further LIM step (20, 30, ...). Last alerted level per key kept in storage_alert_state.json so it never repeats.
    lim = float(sys.argv[sys.argv.index('--alert') + 1])
    st_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'storage_alert_state.json')
    try:
        state = json.load(open(st_path))
    except (OSError, ValueError):
        state = {}
    hits = []
    for k, v in out.items():
        level = int(v['ours_usd_month'] // lim)
        if level >= 1 and level > state.get(k, 0):
            hits.append(f"key {k}: our checkpoints {v['ours_gb']} GB = ${v['ours_usd_month']:.2f}/month (> ${level * lim:.0f})")
            state[k] = level
    if hits:
        msg = (f'To: sk7524@princeton.edu\nSubject: [tinker-inkling] checkpoint storage over ${lim:.0f}/month on '
               f'{len(hits)} key(s)\n\n' + '\n'.join(hits) + '\n\nall keys:\n' + json.dumps(out, indent=1) + '\n')
        subprocess.run(['/usr/sbin/sendmail', '-t'], input=msg, text=True)
        print('ALERT:', '; '.join(hits))
    json.dump(state, open(st_path, 'w'))
