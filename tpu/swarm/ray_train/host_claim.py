"""Claim private TPU hosts (the shared v5p-64) before a launch, refusing if anyone else holds them.

Pool workers run one job at a time; the private v5p-64 is shared with other users, so a launch there
first checks every host it will use. A host is held when another user's process has a TPU device open
(/dev/accel*, /dev/vfio/*), when a process cannot be inspected (fail closed), or when a claim by
another run is still live (its systemd unit is active). Claims are a small JSON file per host; a
claim whose unit has stopped is stale and may be replaced.

`run(host, script, sudo)` executes a Python script on a host and returns its stdout; the launcher
supplies SSH. Nothing here launches or kills anything.
"""
import json

CLAIM = '/var/tmp/skyrl-host-claim.json'

PROBE = r'''
import json, os, pwd, subprocess
me = PARAMS['user']
holders, uninspectable = [], []
for pid in filter(str.isdigit, os.listdir('/proc')):
    try:
        uid = os.stat('/proc/' + pid).st_uid
        user = pwd.getpwuid(uid).pw_name
    except (OSError, KeyError):
        continue
    try:
        targets = [os.readlink('/proc/%s/fd/%s' % (pid, fd)) for fd in os.listdir('/proc/%s/fd' % pid)]
    except PermissionError:
        if user != me and uid != 0:
            uninspectable.append(dict(pid=int(pid), user=user))
        continue
    except OSError:
        continue
    if any(t.startswith(('/dev/accel', '/dev/vfio/')) for t in targets) and user != me:
        holders.append(dict(pid=int(pid), user=user))
claim = None
try:
    with open(PARAMS['claim']) as f:
        claim = json.load(f)
    unit = claim.get('unit') or ''
    claim['live'] = bool(unit) and subprocess.run(['systemctl', 'is-active', '--quiet', unit]).returncode == 0
except FileNotFoundError:
    pass
except (OSError, ValueError, AttributeError):
    claim = dict(invalid=True, live=True)
print(json.dumps(dict(holders=holders, uninspectable=uninspectable, claim=claim)))
'''

WRITE = r'''
import json, os
path, record = PARAMS['claim'], PARAMS['record']
tmp = path + '.partial'
with open(tmp, 'w') as f:
    json.dump(record, f)
os.replace(tmp, path)
print(json.dumps(dict(ok=True)))
'''


def verdict(probe, run_id):
    """Why a host is not free for run_id, or None when it is."""
    if probe.get('holders'):
        return 'other users hold the TPU: ' + ', '.join(sorted({h['user'] for h in probe['holders']}))
    if probe.get('uninspectable'):
        return 'processes of other users cannot be inspected'
    claim = probe.get('claim')
    if claim and claim.get('live') and claim.get('run_id') != run_id:
        return 'claimed by %s (%s)' % (claim.get('run_id') or 'an unreadable claim', claim.get('unit') or '-')
    return None


def check_hosts(hosts, run, run_id, user, sudo=True):
    """Probe every host; returns {host: reason or None}. Any probe failure counts as held."""
    result = {}
    for host in hosts:
        try:
            probe = json.loads(run(host, 'PARAMS = %r\n' % dict(user=user, claim=CLAIM) + PROBE, sudo))
            result[host] = verdict(probe, run_id)
        except Exception as exc:
            result[host] = 'probe failed: %s' % type(exc).__name__
    return result


def claim_hosts(hosts, run, run_id, unit, user, now, sudo=True):
    """Claim all hosts or none: refuses (returns the reasons) if any host is held."""
    reasons = {h: r for h, r in check_hosts(hosts, run, run_id, user, sudo).items() if r}
    if reasons:
        return dict(claimed=False, reasons=reasons)
    record = dict(run_id=run_id, unit=unit, user=user, claimed_at=now, hosts=list(hosts))
    for host in hosts:
        run(host, 'PARAMS = %r\n' % dict(claim=CLAIM, record=record) + WRITE, sudo)
    return dict(claimed=True, reasons={})
