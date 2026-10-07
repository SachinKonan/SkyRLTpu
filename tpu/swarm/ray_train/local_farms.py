"""Farm registry and SSH tunnels for trainer-less runs on a controller host.

Farm ingresses listen on private TPU addresses that this host cannot reach;
its SkyPilot SSH access can. For every RUNNING farm job whose name matches a
selector, this keeps one `ssh -L 127.0.0.1:<port>:127.0.0.1:<farm port>`
tunnel up and publishes the farm (model, lease state, owner, local URL) to a
JSON registry that local_notrain.py reads. It never acquires leases: the
runs' borrowers do, and the farm's atomic acquire arbitrates between them.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import time

import httpx

from .borrowing_supervisor import inventory, rpc


class Tunnel:
    def __init__(self, ssh_dir, cluster, local_port, farm_port):
        self.args = ['ssh', '-F', str(Path(ssh_dir) / cluster), '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10',
                     '-o', 'ExitOnForwardFailure=yes', '-o', 'ServerAliveInterval=15', '-o', 'ServerAliveCountMax=4',
                     '-N', '-L', f'127.0.0.1:{local_port}:127.0.0.1:{farm_port}', cluster]
        self.cluster = cluster
        self.url = f'http://127.0.0.1:{local_port}'
        self.process = None
        self.starts = 0

    def ensure(self):
        if self.process is None and self.starts == 0 and self.healthy():
            # Adopt a tunnel a previous daemon left serving this port, so a
            # daemon restart never cuts the runs' in-flight requests.
            return
        if self.process is None or self.process.poll() is not None:
            self.starts += 1
            self.process = subprocess.Popen(self.args, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                            stderr=subprocess.DEVNULL, start_new_session=True)

    def healthy(self):
        try:
            return httpx.get(self.url + '/health', timeout=5).status_code == 200
        except httpx.HTTPError:
            return False

    def close(self):
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(10)
            except subprocess.TimeoutExpired:
                self.process.kill()


def local_port(job_id, base):
    return base + int(job_id) % 2000


def write(path, document):
    path = Path(path)
    tmp = path.with_suffix('.partial')
    tmp.write_text(json.dumps(document, indent=1, sort_keys=True))
    tmp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registry', type=Path, required=True)
    parser.add_argument('--ssh-config-dir', type=Path, required=True)
    parser.add_argument('--name-contains', action='append',
                        help='Farm job-name substring (repeatable); default: every inference-farm job. '
                             'Runs pick farms by served model, never by name.')
    parser.add_argument('--farm-port', type=int, default=24800)
    parser.add_argument('--port-base', type=int, default=27000)
    parser.add_argument('--interval', type=int, default=20)
    args = parser.parse_args()
    args.name_contains = args.name_contains or ['inference-farm']
    tunnels = {}
    try:
        while True:
            try:
                rows = [r for r in inventory() if r['status'] == 'RUNNING' and r.get('cluster')
                        and any(s in (r.get('run_id') or '') for s in args.name_contains)]
            except Exception as exc:
                # A queue outage must not look like an empty farm list.
                print(json.dumps(dict(time=time.time(), error=type(exc).__name__)), flush=True)
                time.sleep(args.interval)
                continue
            live = {str(r['job_id']) for r in rows}
            for job in [j for j in tunnels if j not in live]:
                tunnels.pop(job).close()

            def probe(row):
                job = str(row['job_id'])
                tunnel = tunnels.get(job)
                if tunnel is None or tunnel.cluster != row['cluster']:
                    if tunnel is not None:
                        tunnel.close()
                    tunnel = tunnels[job] = Tunnel(args.ssh_config_dir, row['cluster'],
                                                   local_port(job, args.port_base), args.farm_port)
                tunnel.ensure()
                status = rpc(args.ssh_config_dir, row['cluster'], 'farm', args.farm_port)
                farm = dict(status.get('result') or {})
                # The farm reports its private origin; runs use the tunnel.
                farm['internal_url'] = farm.pop('url', None)
                farm.update(job_id=job, name=row['run_id'], cluster=row['cluster'], url=tunnel.url,
                            tunnel_ok=tunnel.healthy(), tunnel_starts=tunnel.starts,
                            ok=bool(status.get('ok')), error=None if status.get('ok') else status.get('error'))
                return farm
            with ThreadPoolExecutor(max_workers=8) as workers:
                farms = list(workers.map(probe, rows))
            write(args.registry, dict(time=time.time(), farms=farms))
            print(json.dumps(dict(time=time.time(), farms=len(farms),
                                  usable=sum(f['ok'] and f['tunnel_ok'] for f in farms))), flush=True)
            time.sleep(args.interval)
    finally:
        for tunnel in tunnels.values():
            tunnel.close()


if __name__ == '__main__':
    main()
