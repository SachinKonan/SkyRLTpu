"""Trainer-less no-train (RQ4) run on a controller host without TPUs.

A frozen policy that is the base model (no LoRA) needs no trainer: the only
TPU work is farm generation. This process stands in for a remote-only
trainer slice. It runs the same control-only Ray Serve ingress the trainer
head runs (serving.Ingress, no engines) on 127.0.0.1, feeds it the farms that
local_farms.py tunnels, holds the run reservation like controller.py, then
runs the seeded bootstrap and the frozen client against it. The client
samples the base model straight from the ingress (TTD_FROZEN_BASE_URL) and
grades every candidate on the leased farms (no local grading pool).
"""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

import httpx

from .commands import client_environment
from .config import Config
from .events import emit


def candidates(registry, config, limit):
    """Farms serving this model: the ones this run holds first, then free ones."""
    own, free = [], []
    for farm in registry.get('farms', []):
        if not (farm.get('ok') and farm.get('tunnel_ok') and config.model in (farm.get('models') or [])):
            continue
        owner = (farm.get('owner_run') or '').split(':', 1)[0]
        if owner == config.run_id:
            own.append(farm['url'])
        elif farm.get('state') in ('unleased', 'expired'):
            free.append(farm['url'])
    return sorted(set(own)) + sorted(set(free) - set(own))[:max(0, limit - len(set(own)))]


def child_environment(config, root, args, gcs):
    env = client_environment(config, root, '127.0.0.1')
    url = f'http://127.0.0.1:{config.ports.inference}'
    env.update(RAY_AUTH_MODE='token', RAY_AUTH_TOKEN=os.environ['RAY_AUTH_TOKEN'])
    env.update(HF_HOME=str(args.hf_home), HF_HUB_OFFLINE='1', RAY_ADDRESS=gcs,
               TTD_FROZEN_BASE_URL=url, SKYRL_BORROWING_BASE_MODEL='1',
               # No systemd grading pool on this host: every grade goes to a farm.
               SKYRL_GRADING_LOCAL_SLOTS='0', SKYRL_GRADING_LOCAL_HOSTS='0',
               PYTHONPATH=f'{args.package}:{args.source}:{args.source / "tpu"}:{args.source / "third_party/discover"}')
    env.pop('TINKER_BASE_URL', None)
    if args.num_epochs:
        # Client-only: the frozen client resumes at its global step, and the
        # bootstrap contract (profile config) is unaffected.
        env['NUM_EPOCHS'] = str(args.num_epochs)
    return env


class Run:
    def __init__(self, args):
        self.args = args
        self.config = Config.load(args.config)
        if not (self.config.inference.remote_only and self.config.client_env.get('TTD_FROZEN_POLICY') == '1'
                and float(self.config.client_learning_rate) == 0.0):
            raise SystemExit('local no-train runs require a remote-only frozen profile with learning rate 0')
        self.root = Path(self.config.root).expanduser()
        self.run = self.root / 'runs' / self.config.run_id
        self.run.mkdir(parents=True, exist_ok=True)
        self.log = self.run / 'local.jsonl'
        self.url = f'http://127.0.0.1:{self.config.ports.inference}'
        self.instance = None
        self.stopping = threading.Event()
        self.child = None

    def report(self, event, **fields):
        emit(self.log, event, run_id=self.config.run_id, **fields)

    def start_ingress(self):
        # This host is shared: the run's Ray cluster accepts only its own token,
        # which the bootstrap and client inherit through the environment.
        import secrets
        os.environ.setdefault('RAY_AUTH_TOKEN', secrets.token_hex(32))
        os.environ['RAY_AUTH_MODE'] = 'token'
        import ray
        from ray import serve
        from .serving import Catalog, Ingress
        ray.init(num_cpus=4, include_dashboard=False, namespace=self.config.run_id,
                 _node_ip_address='127.0.0.1',
                 _temp_dir=f'/tmp/rq4-{os.getuid()}-{self.config.ports.inference}')
        self.gcs = ray.get_runtime_context().gcs_address
        catalog = Catalog.options(name='inference-catalog').remote([], self.config.inference.restart_limit)
        serve.start(proxy_location='HeadOnly', http_options={'host': '127.0.0.1', 'port': self.config.ports.inference})
        serve.run_many([serve.RunTarget(target=Ingress.bind(self.config.to_dict(), [], catalog, []),
                                        name='skyrl-training')],
                       wait_for_ingress_deployment_creation=False, wait_for_applications_running=False)
        deadline = time.monotonic() + 300
        while time.monotonic() < deadline:
            try:
                status = httpx.get(self.url + '/status', timeout=10).json()
                if status.get('remote_only') and httpx.get(self.url + '/health', timeout=10).status_code == 200:
                    self.instance = status['instance']
                    self.report('local_control_plane', instance=self.instance, gcs=self.gcs)
                    return
            except (httpx.HTTPError, ValueError, KeyError):
                pass
            time.sleep(2)
        raise TimeoutError('local ingress did not become ready')

    def push_farms(self):
        last = None
        while not self.stopping.is_set():
            try:
                registry = json.loads(Path(self.args.registry).read_text())
                urls = candidates(registry, self.config, self.config.inference.external_pool_candidate_limit)
                if urls != last:
                    body = dict(model=self.config.model, run_id=self.config.run_id, instance=self.instance, urls=urls,
                                group=self.config.inference.external_pool_discovery_group)
                    httpx.post(self.url + '/skyrl/v1/borrowing/services', json=body, timeout=10).raise_for_status()
                    self.report('local_farms_offered', urls=urls)
                    last = urls
            except (OSError, ValueError, httpx.HTTPError) as exc:
                self.report('local_farms_offer_failed', reason=f'{type(exc).__name__}: {exc}'[:300])
            self.stopping.wait(15)

    def reservation(self, action, timeout=10):
        response = httpx.post(self.url + '/skyrl/v1/borrowing/reservation', timeout=timeout,
                              json=dict(run_id=self.config.run_id, instance=self.instance, action=action))
        response.raise_for_status()
        return response.json()

    def admit(self):
        """Wait without a deadline for the first farm (controller.wait_farm_admission)."""
        started, last_alert = time.monotonic(), 0.0
        while not self.stopping.is_set():
            try:
                status = self.reservation('acquire', self.config.inference.external_pool_prepare_timeout + 5)
                if status.get('reserved'):
                    self.report('farm_admitted', **status)
                    return
                if time.monotonic() - last_alert >= self.config.inference.external_pool_queue_alert_seconds:
                    last_alert = time.monotonic()
                    self.report('farm_admission_waiting', waited_seconds=round(last_alert - started, 1),
                                candidates=status.get('candidates', []), target=status.get('target'))
            except (httpx.HTTPError, ValueError) as exc:
                self.report('farm_admission_pending', reason=type(exc).__name__)
            self.stopping.wait(5)
        raise RuntimeError('farm admission interrupted')

    def heartbeat(self):
        last_report = 0.0
        while not self.stopping.wait(self.config.inference.external_pool_heartbeat_seconds):
            try:
                self.reservation('heartbeat')
                if time.monotonic() - last_report >= 300:
                    last_report = time.monotonic()
                    leases = httpx.get(self.url + '/status', timeout=10).json().get('leases', [])
                    self.report('remote_leases', count=len(leases), leases=leases)
            except (httpx.HTTPError, ValueError) as exc:
                self.report('reservation_heartbeat_failed', reason=type(exc).__name__)

    def execute(self, name, command, cwd):
        env = child_environment(self.config, self.root, self.args, self.gcs)
        with open(self.run / f'{name}.log', 'a') as output:
            self.report('child_started', child=name, command=command)
            self.child = subprocess.Popen(command, cwd=cwd, env=env, stdout=output, stderr=subprocess.STDOUT,
                                          start_new_session=True)
            code = self.child.wait()
        self.child = None
        self.report('child_finished', child=name, returncode=code)
        if code:
            raise RuntimeError(f'{name} exited with {code}')

    def bootstrap(self):
        client = self.run / 'client'
        if (client / 'bootstrap/complete.json').exists() or not self.config.bootstrap_max_drafts:
            return
        self.execute('bootstrap', [str(self.args.python), '-m', 'tpu.swarm.ray_train.seed_bootstrap',
                                   '--config-json', json.dumps(self.config.to_dict()),
                                   '--snapshot', str(self.args.snapshot), '--head', '127.0.0.1'],
                     self.args.package)
        summary = json.loads((client / 'bootstrap/complete.json').read_text())
        pool = client / 'tinker_log' / self.config.run_id / 'puct_sampler_step_000000.json'
        if summary.get('retained', 0) < 1 or not pool.exists():
            raise RuntimeError('bootstrap finished without a promoted seed pool')
        self.report('bootstrap_verified', **summary)

    def stop(self, *_):
        self.stopping.set()
        child = self.child
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)

    def main(self):
        signal.signal(signal.SIGTERM, self.stop)
        signal.signal(signal.SIGINT, self.stop)
        code = 1
        try:
            self.start_ingress()
            threading.Thread(target=self.push_farms, name='farms', daemon=True).start()
            self.admit()
            threading.Thread(target=self.heartbeat, name='heartbeat', daemon=True).start()
            self.bootstrap()
            if not self.stopping.is_set():
                self.execute('client', [str(self.args.python), str(self.args.source / 'tpu/run_ttd_ensemble.py')],
                             self.args.source)
            code = 0
            self.report('run_finished')
        except Exception as exc:
            self.report('run_failed', reason=f'{type(exc).__name__}: {exc}'[:500])
        finally:
            self.stopping.set()
            if self.instance:
                try:
                    self.reservation('close', self.config.inference.external_pool_release_timeout + 5)
                    self.report('reservation_closed')
                except (httpx.HTTPError, ValueError) as exc:
                    self.report('reservation_close_failed', reason=type(exc).__name__)
            try:
                from ray import serve
                serve.shutdown()
                import ray
                ray.shutdown()
            except Exception:
                pass
        return code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--registry', type=Path, required=True)
    parser.add_argument('--package', type=Path, required=True, help='Extracted ray-training package root')
    parser.add_argument('--source', type=Path, required=True, help='Base bundle with overlays applied')
    parser.add_argument('--python', type=Path, required=True, help='Client environment interpreter')
    parser.add_argument('--hf-home', type=Path, required=True)
    parser.add_argument('--snapshot', type=Path, required=True, help='Tokenizer snapshot of the served model')
    parser.add_argument('--num-epochs', type=int, help='Total client steps (overrides the profile NUM_EPOCHS)')
    sys.exit(Run(parser.parse_args()).main())


if __name__ == '__main__':
    main()
