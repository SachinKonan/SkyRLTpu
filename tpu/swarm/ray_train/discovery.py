"""One discovery service for pool farms, pool trainers and private hosts (the shared v5p-64).

It replaces borrowing_supervisor (pushes candidate farms to trainers), local_farms (farms.json and
SSH tunnels for della clients) and the v5p-64's od_watch (direct-SSH trainers and dedicated farms).

- Sources: SkyPilot managed jobs (farms by job name, trainers in listed pools) and a checked-in
  private-hosts file (farms and trainers reached by direct SSH to an external IP).
- Directory: every farm keeps first_seen/last_seen, consecutive probe misses, a health state
  (healthy, suspect after 1-2 misses, unhealthy after 3) and its incarnation (the ingress's
  per-process instance ID from /status). A farm that answers with a new incarnation came back:
  it is reported as `returned` and starts over (its old leases and grading entries are void).
  A farm whose job is no longer RUNNING is `gone` and leaves the published list.
- Publishes the same farms.json document local_notrain reads (each farm: models, url (tunnel),
  internal_url, owner_run, state, ok, tunnel_ok, ...), plus the directory fields above.
- Output contract (READY_ONLY): `farms` lists only farms that are fully ready (every expected engine up,
  not updating; the probe's ok), attested (they report a compatibility hash) and healthy (no missed
  probe). Loading, suspect and unreachable farms go to `unready` for diagnostics, never to `farms`.
  `updated` is the publish time; clients and the monitor treat a registry older than a few intervals as
  stale (registry_is_fresh), and the systemd unit's watchdog restarts a hung service (sd_notify).
- Pushes candidate lists to trainers only with --push. Without it (shadow mode) it computes what it
  would push, so it can run next to the old services, writing its own registry, and be diffed.
It never acquires leases or cancels jobs; farms arbitrate.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import time

from .borrowing_supervisor import REMOTE, inventory, rpc, tick as pool_tick
from .direct_discovery import candidates, tick as direct_tick
from .local_farms import Tunnel, local_port, write

MISSES_UNHEALTHY = 3
READY_ONLY = 'ready-only-v1'
PRIVATE_ID_BASE = 10_000_000  # synthetic job IDs for private farms: sortable and tunnel-port-stable


def ssh_argv(entry):
    """Direct SSH to a private host's external IP (the v5p-64 is not a SkyPilot cluster)."""
    return ['ssh', '-i', str(Path(entry['ssh_key']).expanduser()), '-o', 'BatchMode=yes',
            '-o', 'ConnectTimeout=8', '-o', 'StrictHostKeyChecking=accept-new',
            '%s@%s' % (entry.get('ssh_user', 'gcpuser'), entry['external_ip'])]


def private_rpc(entry, action, port, body=None, group=''):
    params = dict(action=action, port=port)
    if body is not None:
        params['body'] = body
    if group and action == 'target':
        params['query'] = '?group=' + group
    try:
        proc = subprocess.run(ssh_argv(entry) + ['python3 -'], input='PARAMS = %r\n' % params + REMOTE,
                              text=True, capture_output=True, timeout=50)
        if proc.returncode:
            return {'ok': False, 'error': 'ssh_failed', 'exit_code': proc.returncode}
        return json.loads(proc.stdout)
    except (OSError, subprocess.TimeoutExpired, ValueError) as exc:
        return {'ok': False, 'error': type(exc).__name__}


def local_rpc(url):
    """The same farm probe, run here against a farm reachable from this host (della workers: no SSH needed)."""
    from urllib.parse import urlsplit
    parts = urlsplit(url)
    params = dict(action='farm', port=parts.port, host=parts.hostname)
    try:
        proc = subprocess.run(['python3', '-'], input='PARAMS = %r\n' % params + REMOTE, text=True,
                              capture_output=True, timeout=50)
        if proc.returncode:
            return {'ok': False, 'error': 'probe_failed', 'exit_code': proc.returncode}
        return json.loads(proc.stdout)
    except (OSError, subprocess.TimeoutExpired, ValueError) as exc:
        return {'ok': False, 'error': type(exc).__name__}


def slurm_nodes(run=subprocess.run):
    """{Slurm job name: node} for this user's running jobs, from one squeue call; None if squeue failed."""
    try:
        proc = run(['squeue', '-h', '--me', '-t', 'R', '-o', '%j %N'], capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if proc.returncode:
        return None
    return dict(line.split(' ', 1) for line in proc.stdout.splitlines() if ' ' in line)


def slurm_farm_url(job_id, cluster, nodes):
    """A farm job on a Slurm-backed pool: its node (the SkyPilot cluster is the Slurm job of that name) and the
    HTTP port placed by its managed job id (config.farm_ports), or None while the node is unknown."""
    from .config import farm_ports
    # SkyPilot names the Slurm job "<cluster>-<user hash>", so match that suffix as well as the bare name.
    node = [n for j, n in nodes.items() if n and (j == cluster or j.rsplit('-', 1)[0] == cluster)]
    return f'http://{node[0]}:{farm_ports(job_id)[0]}' if len(node) == 1 else None


def load_private(path):
    """{'farms': [{name, external_ip, ssh_key, ...}], 'trainers': [direct registrations]}."""
    if not path:
        return dict(farms=[], trainers=[])
    document = json.loads(Path(path).read_text())
    for farm in document.get('farms', []):
        missing = {'name', 'external_ip', 'ssh_key'} - set(farm)
        if missing:
            raise ValueError('private farm %r lacks %s' % (farm.get('name'), sorted(missing)))
    for trainer in document.get('trainers', []):
        missing = {'run_id', 'model', 'group', 'external_ip', 'ssh_key', 'stable_urls'} - set(trainer)
        if missing:
            raise ValueError('private trainer %r lacks %s' % (trainer.get('run_id'), sorted(missing)))
    return dict(farms=document.get('farms', []), trainers=document.get('trainers', []))


def fair_share_cap(ready_farms, descriptors):
    """Farms each generating trainer may be offered: ceil(ready / generating trainers), or None (no cap).

    A trainer is generating when its borrower reports a phase or holds a lease. Trainers that are training or
    grading do not count, so the one generating then gets every farm."""
    generating = sum(1 for d in descriptors if (d.get('borrowing') or {}).get('phase')
                     or (d.get('borrowing') or {}).get('held'))
    if generating < 2:
        return None
    return max(1, -(-ready_farms // generating))


def trim_to_share(target, urls, cap):
    """Keep every farm the trainer already holds (never revoke a lease), then fill up to cap."""
    if cap is None:
        return urls
    held = set((target.get('borrowing') or {}).get('held') or [])
    kept = [u for u in urls if u in held]
    return kept + [u for u in urls if u not in held][:max(0, cap - len(kept))]


class Directory:
    """Health and incarnation per farm, keyed by job ID (pool) or name (private)."""

    def __init__(self):
        self.entries = {}

    def observe(self, key, probe, now):
        """Record one probe. Returns 'returned' when a known farm answers with a new incarnation."""
        entry = self.entries.setdefault(key, dict(first_seen=now, last_seen=None, misses=0,
                                                  incarnation=None, returns=0))
        event = None
        if probe.get('ok'):
            incarnation = (probe.get('result') or {}).get('instance')
            if entry['incarnation'] and incarnation and incarnation != entry['incarnation']:
                event = 'returned'
                entry.update(first_seen=now, returns=entry['returns'] + 1)
            entry.update(last_seen=now, misses=0, incarnation=incarnation or entry['incarnation'])
        else:
            entry['misses'] += 1
        entry['health'] = ('healthy' if entry['misses'] == 0 else
                           'unhealthy' if entry['misses'] >= MISSES_UNHEALTHY else 'suspect')
        return event

    def retire(self, live_keys):
        """Keys no longer RUNNING are gone; returns them."""
        gone = [k for k in self.entries if k not in live_keys]
        for key in gone:
            del self.entries[key]
        return gone


class Service:
    def __init__(self, args, private, now=time.time):
        self.args, self.private, self.now = args, private, now
        self.directory, self.tunnels = Directory(), {}
        # One squeue per tick for every Slurm farm; a failed call keeps the last answer (della's controller is slow).
        self.slurm_nodes = {}
        # Della grading farms for GCP trainers (grader_gateway): one endpoint here, one reverse tunnel per trainer host.
        self.gateway = None
        if getattr(args, 'grader_gateway', False):
            from .grader_gateway import GraderGateway
            self.gateway = GraderGateway(args, private['trainers'], ssh_argv, emit=self.emit)

    def farm_sources(self, rows):
        pool = [dict(key=str(r['job_id']), job_id=r['job_id'], name=r['run_id'], cluster=r['cluster'],
                     pool=r.get('pool'), source='pool')
                for r in rows if r['status'] == 'RUNNING' and r.get('cluster')
                and any(s.casefold() in (r.get('run_id') or '').casefold() for s in self.args.name_contains)]
        private = [dict(key='private:' + f['name'], job_id=PRIVATE_ID_BASE + i, name=f['name'], cluster=None,
                        pool=None, source='private', entry=f) for i, f in enumerate(self.private['farms'])]
        for farm in pool:
            if farm['pool'] in (getattr(self.args, 'slurm_pool', None) or ()):
                farm['source'] = 'slurm'  # reached directly at its node and job-placed port, no SSH
        return pool + private

    def probe(self, farm):
        if farm['source'] == 'slurm':
            url = slurm_farm_url(farm['job_id'], farm['cluster'], self.slurm_nodes)
            return local_rpc(url) if url else {'ok': False, 'error': 'node_unknown'}
        if farm['source'] == 'private':
            return private_rpc(farm['entry'], 'farm', farm['entry'].get('port', self.args.farm_port))
        return rpc(self.args.ssh_config_dir, farm['cluster'], 'farm', self.args.farm_port)

    def tunnel(self, farm):
        if not self.args.tunnels:
            return None
        tunnel = self.tunnels.get(farm['key'])
        if tunnel is None or tunnel.cluster != (farm['cluster'] or farm['name']):
            if tunnel is not None:
                tunnel.close()
            port = local_port(farm['job_id'], self.args.port_base)
            tunnel = Tunnel(self.args.ssh_config_dir, farm['cluster'] or farm['name'], port, self.args.farm_port)
            if farm['source'] == 'private':
                remote = farm['entry'].get('port', self.args.farm_port)
                tunnel.args = (ssh_argv(farm['entry'])[:-1] + ['-o', 'ExitOnForwardFailure=yes',
                               '-o', 'ServerAliveInterval=15', '-o', 'ServerAliveCountMax=4', '-N',
                               '-L', '127.0.0.1:%d:127.0.0.1:%d' % (port, remote), ssh_argv(farm['entry'])[-1]])
            self.tunnels[farm['key']] = tunnel
        tunnel.ensure()
        return tunnel

    def tick(self, rows):
        now = self.now()
        sources = self.farm_sources(rows)
        if any(f['source'] == 'slurm' for f in sources):
            nodes = slurm_nodes()
            if nodes is not None:
                self.slurm_nodes = nodes
        for key in self.directory.retire({f['key'] for f in sources}):
            if key in self.tunnels:
                self.tunnels.pop(key).close()
            self.emit(event='farm_gone', key=key)

        def observe(farm):
            probe = self.probe(farm)
            # A farm without cloud metadata (della) names its own address and is reached directly, no tunnel.
            direct = bool((probe.get('result') or {}).get('direct'))
            return farm, probe, None if direct else self.tunnel(farm)
        with ThreadPoolExecutor(max_workers=8) as workers:
            observed = list(workers.map(observe, sources))
        published, probes = [], {}
        for farm, probe, tunnel in observed:
            if self.directory.observe(farm['key'], probe, now) == 'returned':
                self.emit(event='farm_returned', key=farm['key'], name=farm['name'],
                          incarnation=probe['result'].get('instance'))
            entry = self.directory.entries[farm['key']]
            record = dict(probe.get('result') or {})
            record['internal_url'] = record.pop('url', None)
            record.update(job_id=str(farm['job_id']), name=farm['name'], cluster=farm['cluster'],
                          source=farm['source'], pool=farm['pool'],
                          url=tunnel.url if tunnel else record['internal_url'],
                          # --private-only (v6e-128): trainers reach farms at their internal URL, no tunnel, so a
                          # healthy probe is enough; open grading (grading_transport.listable) requires tunnel_ok.
                          tunnel_ok=(tunnel.healthy() if tunnel else
                                     bool(record.get('direct')) or (getattr(self.args, 'private_only', False)
                                                                    and bool(probe.get('ok')))),
                          tunnel_starts=tunnel.starts if tunnel else 0,
                          ok=bool(probe.get('ok')), error=None if probe.get('ok') else probe.get('error'),
                          **{k: entry[k] for k in ('health', 'misses', 'incarnation', 'first_seen',
                                                   'last_seen', 'returns')})
            published.append(record)
            if farm['cluster']:
                probes[farm['cluster']] = probe
            elif farm['source'] == 'private':
                probes[farm['key']] = probe
        trainers = self.push(rows, probes, [f for f, p, _ in observed if p.get('ok')])
        ready = [f for f in published if is_ready(f)]
        unready = [dict(name=f['name'], job_id=f['job_id'], health=f['health'], state=f.get('state'),
                        error=f.get('error')) for f in published if not is_ready(f)]
        document = dict(time=now, updated=now, contract=READY_ONLY, farms=ready, unready=unready,
                        trainers=trainers, shadow=not self.args.push)
        if self.gateway is not None:
            self.gateway_inputs = (rows, ready)
            document['gateway'] = self.gateway.tick(rows, ready)
        write(self.args.registry, document)
        return document

    def keep_gateway(self):
        """After a failed tick (e.g. the SkyPilot API is down): keep the gateway's tunnels and trainer registries
        alive with the last good trainer rows and farms; its farm probes and grades never depended on the tick."""
        if self.gateway is None or getattr(self, 'gateway_inputs', None) is None:
            return None
        return self.gateway.tick(*self.gateway_inputs)

    def push(self, rows, probes, healthy):
        """Candidate lists for pool trainers (borrowing_supervisor rules) and private trainers (direct)."""
        if getattr(self.args, 'gateway_only', False):
            # Next to the generation pusher: trainers are only tunnel targets here, their lists are not touched.
            return dict(pool=[], direct=[])
        pool = {}
        trim = None
        if getattr(self.args, 'fair_share', False):
            cap = fair_share_cap(sum(1 for f in healthy if f['source'] == 'pool'), self.trainer_descriptors(rows))
            self.emit(event='fair_share', cap=cap)
            trim = lambda target, urls: trim_to_share(target, urls, cap)
        if self.args.trainer_pool or self.args.trainer_job_id:
            def call(cluster, action, body=None):
                if action == 'farm':
                    return probes.get(cluster, {'ok': False, 'error': 'not_probed'})
                return rpc(self.args.ssh_config_dir, cluster, action, self.args.farm_port, body,
                           group=self.args.discovery_group)
            pool = pool_tick(rows, None, self.args.trainer_job_id, call, dry_run=not self.args.push,
                             trainer_pools=self.args.trainer_pool, run_scoped_only=self.args.run_scoped_only,
                             farm_name_contains=self.args.name_contains[0], trim=trim).get('targets', [])
        farms = [dict(probes.get(f['cluster'], {}).get('result') or {}, source_pool=f['pool'])
                 for f in healthy if f['source'] == 'pool']
        # Private farms (fixed hosts such as the v6e-128) are offered to private trainers that list the
        # farm's 'pool' label in their farm_pools; they are reached at their internal URL.
        farms += [dict(probes.get(f['key'], {}).get('result') or {}, source_pool=f['entry'].get('pool', 'private'))
                  for f in healthy if f['source'] == 'private']

        def direct_call(row, body=None):
            return private_rpc(row, 'target', row.get('port', self.args.farm_port), body, group=row['group'])
        direct = []
        model_caps = self.model_fair_caps(farms, direct_call) if getattr(self.args, 'model_fair_share', False) else {}
        for row in self.private['trainers']:
            eligible = [f for f in farms if f['source_pool'] in row.get('farm_pools', [])]
            if self.args.push:
                row_trim = trim
                if row['model'] in model_caps:
                    cap = model_caps[row['model']]
                    row_trim = lambda target, urls, cap=cap: trim_to_share(target, urls, cap)
                direct += direct_tick([row], eligible, direct_call, trim=row_trim)
                continue
            descriptor = direct_call(row)
            if not descriptor.get('ok'):
                direct.append(dict(run_id=row['run_id'], state='unreachable'))
                continue
            direct.append(dict(run_id=row['run_id'], state='would_update',
                               urls=candidates(descriptor['result'], row, eligible)))
        return dict(pool=pool, direct=direct)

    def model_fair_caps(self, farms, direct_call):
        """{model: cap} for private trainers: ready farms serving that model divided by that model's trainers that
        are generating (a borrowing phase or held leases), as fair_share_cap but per model, so a trainer that is
        grading or training leaves its farms to the others. No cap for a model with fewer than two generating."""
        import collections
        ready = collections.Counter(m for f in farms for m in (f.get('models') or [])[:1])
        generating = collections.Counter()
        for row in self.private['trainers']:
            d = direct_call(row)
            b = ((d.get('result') or {}).get('borrowing') or {}) if d.get('ok') else {}
            if b.get('phase') or b.get('held'):
                generating[row['model']] += 1
        caps = {m: max(1, -(-ready[m] // n)) for m, n in generating.items() if n >= 2}
        self.emit(event='model_fair_share', caps=caps, generating=dict(generating), ready=dict(ready))
        return caps

    def trainer_descriptors(self, rows):
        """Current target descriptors of every trainer this discovery pushes to (pool jobs and private)."""
        out = []
        running = {r['job_id']: r for r in rows if r['status'] == 'RUNNING' and r.get('cluster')}
        jobs = list(dict.fromkeys([*self.args.trainer_job_id,
                                   *[r['job_id'] for r in rows if r.get('pool') in self.args.trainer_pool]]))
        for job in jobs:
            if job in running:
                d = rpc(self.args.ssh_config_dir, running[job]['cluster'], 'target', self.args.farm_port,
                        group=self.args.discovery_group)
                if d.get('ok'):
                    out.append(d['result'])
        for row in self.private['trainers']:
            d = private_rpc(row, 'target', row.get('port', self.args.farm_port), group=row['group'])
            if d.get('ok'):
                out.append(d['result'])
        return out

    def emit(self, **fields):
        print(json.dumps(dict(time=self.now(), **fields)), flush=True)

    def close(self):
        for tunnel in self.tunnels.values():
            tunnel.close()
        if self.gateway is not None:
            self.gateway.close()


def is_ready(record):
    """Fully ready (probe ok: every expected engine up, not updating), attested and healthy.

    A farm without engines serves no model, so it has no serving attestation (grades carry the grader hash)."""
    capabilities = record.get('capabilities') or {}
    return bool(record.get('ok') and record.get('health') == 'healthy'
                and (capabilities.get('compatibility_sha256') or (record.get('engines') == 0 and record.get('grading'))))


def registry_is_fresh(path, max_age, now=time.time):
    """(fresh, age_seconds): a registry that stopped updating must not be trusted as the current fleet."""
    try:
        document = json.loads(Path(path).read_text())
        age = now() - float(document.get('updated', document.get('time')))
    except (OSError, ValueError, TypeError):
        return False, None
    return age <= max_age, round(age, 1)


def sd_notify(message):
    """Tell systemd we are alive (Type=notify + WatchdogSec); a no-op outside a notify unit."""
    import os
    import socket
    address = os.environ.get('NOTIFY_SOCKET')
    if not address:
        return False
    if address.startswith('@'):
        address = '\0' + address[1:]
    with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as sock:
        sock.connect(address)
        sock.sendall(message.encode())
    return True


def unit_argv(name, args, cwd, log, interval):
    """The discovery service as a user unit that systemd restarts when it exits or stops reporting."""
    from .lifecycle import user_unit_argv
    watchdog = max(3 * interval, 60)
    return user_unit_argv(name, args, cwd, log, restart='always',
                          properties=('Type=notify', 'NotifyAccess=main', 'WatchdogSec=%d' % watchdog,
                                      'RestartSec=10'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registry', type=Path, required=True, help='farms.json to publish')
    parser.add_argument('--ssh-config-dir', type=Path, required=True)
    parser.add_argument('--private-hosts', type=Path, help='checked-in private farms/trainers (v5p-64)')
    parser.add_argument('--slurm-pool', action='append', default=[],
                        help='a Slurm-backed pool (della): its farms are probed directly at node:farm_ports(job id)')
    parser.add_argument('--name-contains', action='append', help='farm job-name substring (default inference-farm)')
    parser.add_argument('--trainer-pool', action='append', default=[])
    parser.add_argument('--trainer-job-id', type=int, action='append', default=[])
    parser.add_argument('--discovery-group', default='')
    parser.add_argument('--run-scoped-only', action='store_true')
    parser.add_argument('--farm-port', type=int, default=24800)
    parser.add_argument('--tunnels', action='store_true', help='keep della SSH tunnels (local_farms)')
    parser.add_argument('--port-base', type=int, default=29000, help='tunnel ports; local_farms uses 27000')
    parser.add_argument('--interval', type=int, default=30)
    parser.add_argument('--grader-gateway', action='store_true',
                        help='grade for every trainer rank-0 host (trainer pools, job ids, private trainers) on the '
                             'ready della grading farms through one endpoint here and one reverse SSH tunnel per '
                             'host (grader_gateway); acts only with --push')
    parser.add_argument('--gateway-port', type=int, default=28650,
                        help='gateway listen port on this host (127.0.0.1)')
    parser.add_argument('--gateway-remote-port', type=int, default=26000,
                        help='gateway port on the trainer hosts (127.0.0.1); +1000 after two refused forwards')
    parser.add_argument('--gateway-priority', action='append', default=[], metavar='PREFIX=N',
                        help='grader_gateway: runs whose id starts with PREFIX get priority N (lower first; others 1); '
                             'repeatable, longest prefix wins')
    parser.add_argument('--gateway-queue-deadline', action='append', default=[], metavar='PREFIX=SECONDS',
                        help='grader_gateway: a queued grade of a run starting with PREFIX is handed back as an '
                             'infrastructure failure after SECONDS if NO farm can serve it, so a client with local '
                             'grading (a TPU trainer) retries there instead of waiting; runs matching no prefix wait '
                             'forever, which is what clients without a local fallback need. Repeatable.')
    parser.add_argument('--gateway-reserve', type=float, default=None,
                        help='grader_gateway: fraction of slots kept free from runs below the best priority (default 0.25)')
    parser.add_argument('--gateway-registry-path', default=None,
                        help='registry path on the trainer hosts (default %s); the profile sets the same path as '
                             'SKYRL_GRADING_REGISTRY' % '/home/gcpuser/.cache/skyrl/della-graders.json')
    parser.add_argument('--gateway-only', action='store_true',
                        help='run only the grader gateway: never read or push trainers\' candidate lists, so it '
                             'runs next to the generation pusher (own lock, default ~/.cache/skyrl/grader-gateway.lock)')
    parser.add_argument('--push', action='store_true', help='push candidate lists (off = shadow mode)')
    parser.add_argument('--fair-share', action='store_true',
                        help='while several trainers generate, offer each at most ceil(ready farms / generating '
                             'trainers) farms; leases already held are always kept')
    parser.add_argument('--once', action='store_true')
    parser.add_argument('--model-fair-share', action='store_true',
                        help='cap private trainers per model: ready farms of the model / its generating trainers')
    parser.add_argument('--private-only', action='store_true',
                        help='private-hosts farms and trainers only, never query SkyPilot (v6e-128 rebuttal runs)')
    parser.add_argument('--lock-file', type=Path, help='exclusive pusher lock (default: one per login)')
    parser.add_argument('--check-fresh', type=float, metavar='MAX_AGE',
                        help='exit 0 if the registry was published within MAX_AGE seconds, else alert and exit 1')
    args = parser.parse_args()
    if args.check_fresh is not None:
        fresh, age = registry_is_fresh(args.registry, args.check_fresh)
        print(json.dumps(dict(event='discovery_registry_fresh' if fresh else 'discovery_registry_stale',
                              registry=str(args.registry), age_seconds=age, max_age=args.check_fresh)))
        raise SystemExit(0 if fresh else 1)
    args.name_contains = args.name_contains or ['inference-farm']
    if args.gateway_only and not args.grader_gateway:
        parser.error('--gateway-only needs --grader-gateway')
    if args.gateway_priority:
        from .grader_gateway import parse_priorities
        try:
            parse_priorities(args.gateway_priority)
        except ValueError as exc:
            parser.error(str(exc))
    if args.gateway_queue_deadline:
        from .grader_gateway import parse_queue_deadlines
        try:
            parse_queue_deadlines(args.gateway_queue_deadline)
        except ValueError as exc:
            parser.error(str(exc))
    if args.gateway_reserve is not None and not 0 <= args.gateway_reserve < 1:
        parser.error('--gateway-reserve must be in [0, 1)')
    if args.interval < 10:
        parser.error('interval must be >= 10 seconds')
    if args.push:
        # Exclusive with the old supervisor: two pushers would fight over trainers' candidate lists.
        import fcntl
        lock_path = args.lock_file or Path.home() / ('.cache/skyrl/grader-gateway.lock' if args.gateway_only
                                                     else '.cache/skyrl/borrowing-supervisor.lock')
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        lock = lock_path.open('a')
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error('another %s is running under this login'
                         % ('grader gateway' if args.gateway_only else 'discovery pusher (borrowing_supervisor)'))
    service = Service(args, load_private(args.private_hosts))
    try:
        while True:
            try:
                document = service.tick([] if args.private_only else inventory())
                service.emit(farms=len(document['farms']), unready=len(document['unready']),
                             usable=sum(f['ok'] and (f['tunnel_ok'] or not args.tunnels) for f in document['farms']))
                # Only a completed publish feeds the watchdog: a hung or failing loop gets restarted.
                sd_notify('READY=1\nWATCHDOG=1')
            except Exception as exc:
                # An inventory outage must not look like an empty farm list: keep the last registry.
                service.emit(error=type(exc).__name__ + ': ' + str(exc)[:300])
                try:
                    if service.keep_gateway() is not None:
                        service.emit(event='gateway_kept_after_failed_tick')
                except Exception as gateway_exc:
                    service.emit(error='gateway: %s: %s' % (type(gateway_exc).__name__, str(gateway_exc)[:300]))
            if args.once:
                break
            time.sleep(args.interval)
    finally:
        service.close()


if __name__ == '__main__':
    main()
