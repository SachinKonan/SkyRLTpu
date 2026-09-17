"""Opt-in systemd ownership for a complete per-host runtime.

The ordinary SkyPilot process is only a launcher. PID/start-time monitoring and
slice heartbeats run in a separate systemd service; KillMode=control-group owns
all descendants, including double-forked Ray daemons. No network namespace.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
from pathlib import Path
import resource
import signal
import socket
import socketserver
import subprocess
import sys
import threading
import time
import uuid


def process_start(pid):
    try:
        fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        return None if fields[0] == 'Z' else fields[19]
    except (FileNotFoundError, ProcessLookupError):
        return None


def save(path, obj):
    path = Path(path)
    tmp = path.with_suffix('.partial')
    tmp.write_text(json.dumps(obj))
    tmp.replace(path)


def inherited_limit_properties():
    """Avoid replacing SkyPilot's resource limits with systemd defaults."""
    result = ['--property=TasksMax=infinity']
    for name in ('CPU', 'FSIZE', 'DATA', 'STACK', 'CORE', 'RSS', 'NOFILE',
                 'AS', 'NPROC', 'MEMLOCK', 'LOCKS', 'SIGPENDING', 'MSGQUEUE',
                 'NICE', 'RTPRIO', 'RTTIME'):
        key = getattr(resource, 'RLIMIT_' + name, None)
        if key is None:
            continue
        soft, hard = resource.getrlimit(key)
        values = ['infinity' if v == resource.RLIM_INFINITY else str(v) for v in (soft, hard)]
        result.append('--property=Limit' + name + '=' + ':'.join(values))
    return result


class SliceState:
    def __init__(self, hosts, token, timeout, now=time.monotonic):
        self.hosts, self.token, self.timeout, self.now = hosts, token, timeout, now
        self.started = now()
        self.seen, self.phases = {}, {}
        self.released = False
        self.failure = None
        self.lock = threading.Lock()

    def update(self, message):
        with self.lock:
            rank = message.get('rank')
            if (message.get('token') != self.token or type(rank) is not int
                    or not 0 <= rank < self.hosts
                    or message.get('phase') not in ('boot', 'ready', 'running', 'done', 'finished', 'failed')):
                return {'error': 'wrong attempt or rank'}
            now = self.now()
            # Check liveness before refreshing; a late peer cannot revive a
            # failed attempt. Never authorize Ray startup after a timeout.
            if any(now - last > self.timeout and self.phases[r] != 'finished' for r, last in self.seen.items()):
                self.failure = self.failure or 'peer heartbeat expired'
            self.seen[rank] = now
            self.phases[rank] = message['phase']
            if message['phase'] == 'failed':
                self.failure = self.failure or f'host {rank} failed'
            if (len(self.phases) == self.hosts
                    and all(p in ('ready', 'running', 'done') for p in self.phases.values())
                    and not self.failure):
                self.released = True
            return {'release': self.released and not self.failure,
                    'failure': self.failure,
                    'done': len(self.phases) == self.hosts and all(p in ('done', 'finished') for p in self.phases.values()),
                    'finished': all(self.phases.get(r) == 'finished' for r in range(1, self.hosts))}


def start_server(address, state):
    class Handler(socketserver.StreamRequestHandler):
        def handle(self):
            self.request.settimeout(3)
            try:
                raw = self.rfile.readline(8193)
                if len(raw) > 8192:
                    return
                answer = state.update(json.loads(raw))
                self.wfile.write(json.dumps(answer).encode() + b'\n')
            except (ValueError, OSError):
                return
    class Server(socketserver.ThreadingTCPServer):
        allow_reuse_address = True
        daemon_threads = True
    server = Server(address, Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


def exchange(address, message):
    with socket.create_connection(address, timeout=2) as sock:
        sock.settimeout(3)
        sock.sendall(json.dumps(message).encode() + b'\n')
        with sock.makefile('rb') as stream:
            return json.loads(stream.readline(8192))


def wait_preflight(config, stopped):
    """Called by bootstrap only after that host's full local port checks."""
    ready = os.environ.get('SKYRL_PREFLIGHT_READY')
    if not config.systemd_runtime:
        return
    if not ready or os.environ.get('SKYRL_SYSTEMD_OWNED') != '1':
        raise RuntimeError('systemd runtime requires its service supervisor')
    release = Path(os.environ['SKYRL_PREFLIGHT_RELEASE'])
    Path(ready).write_text('ready\n')
    deadline = time.monotonic() + config.setup_timeout
    while not release.exists():
        if stopped() or time.monotonic() >= deadline:
            raise RuntimeError('slice preflight aborted or timed out')
        time.sleep(.1)


def supervise(payload_path):
    payload_path = Path(payload_path)
    p = json.loads(payload_path.read_text())
    payload_path.unlink()  # Environment may contain credentials; never log it.
    run = Path(p['directory'])
    stopping = threading.Event()
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: stopping.set())
    # Prevent two private executor services from owning this host concurrently.
    lease = open(Path.home() / '.skyrl-systemd-runtime.lock', 'a')
    fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
    server = None
    child = None
    result = 1
    try:
        if process_start(p['parent_pid']) != p['parent_start']:
            return 143
        if p['rank'] == 0:
            server = start_server((p['ips'][0], p['port']),
                                  SliceState(len(p['ips']), p['token'], p['peer_timeout']))
        ready, release = run / 'ready', run / 'release'
        env = p['env'] | {'SKYRL_SYSTEMD_OWNED': '1', 'SKYRL_PREFLIGHT_READY': str(ready),
                          'SKYRL_PREFLIGHT_RELEASE': str(release)}
        child = subprocess.Popen(p['command'], env=env, cwd=p['cwd'])
        save(run / 'supervisor.json', {'pid': os.getpid(), 'child_pid': child.pid,
                                      'parent_pid': p['parent_pid'], 'rank': p['rank']})
        started, contacted = time.monotonic(), None
        while True:
            now = time.monotonic()
            dead_parent = process_start(p['parent_pid']) != p['parent_start']
            failed = (stopping.is_set() or dead_parent or child.poll() not in (None, 0)
                      or child.poll() == 0 and not release.exists())
            phase = ('failed' if failed else 'done' if child.poll() == 0 else
                     'running' if release.exists() else 'ready' if ready.exists() else 'boot')
            try:
                response = exchange((p['ips'][0], p['port']),
                                    {'token': p['token'], 'rank': p['rank'], 'phase': phase})
                if 'error' in response:
                    raise RuntimeError(response['error'])
                contacted = now
                if response.get('failure') or failed:
                    result = 143 if dead_parent or stopping.is_set() else 1
                    break
                if response.get('release') and not release.exists():
                    release.write_text('all hosts passed\n')
                if response.get('done'):
                    if p['rank'] != 0:
                        exchange((p['ips'][0], p['port']),
                                 {'token': p['token'], 'rank': p['rank'], 'phase': 'finished'})
                        result = 0
                        break
                    if response.get('finished'):
                        result = 0
                        break
            except (OSError, ValueError, RuntimeError) as exc:
                if failed or (contacted is not None and now - contacted > p['peer_timeout']):
                    save(run / 'peer-error.json', {'error': str(exc)})
                    break
            if not release.exists() and now - started > p['setup_timeout']:
                break
            time.sleep(p.get('poll_seconds', 1))
    finally:
        if child is not None and child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=p['shutdown_grace'])
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait(timeout=5)
        if server is not None:
            server.shutdown()
            server.server_close()
        save(run / 'result.json', {'exit_code': result})
        lease.close()
        # When this main process exits, systemd terminates all remaining
        # cgroup members, even descendants that detached or changed parents.
    return result


def launch(command, directory, ips, rank, token, *, port=24900,
           peer_timeout=30, shutdown_grace=90, setup_timeout=1800):
    if not token:
        raise ValueError('a shared SkyPilot task/attempt identity is required')
    attempt = Path(directory) / ('systemd-' + uuid.uuid4().hex)
    attempt.mkdir(parents=True, mode=0o700)
    unit = 'skyrl-runtime-' + hashlib.sha256(str(attempt).encode()).hexdigest()[:20]
    payload = dict(command=list(command), cwd=os.getcwd(), env=dict(os.environ),
                   directory=str(attempt), parent_pid=os.getpid(), parent_start=process_start(os.getpid()),
                   ips=ips, rank=rank, token=token, port=port, peer_timeout=peer_timeout,
                   shutdown_grace=shutdown_grace, setup_timeout=setup_timeout)
    path = attempt / 'payload.json'
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, 'w') as stream:
        json.dump(payload, stream)
    log = attempt / 'service.log'
    log.touch(mode=0o600)
    save(attempt / 'unit.json', {'unit': unit, 'parent_pid': os.getpid(), 'rank': rank})
    args = ['sudo', '-n', 'systemd-run', '--quiet', '--wait', '--unit=' + unit,
            '--uid=' + str(os.getuid()), '--gid=' + str(os.getgid()),
            *inherited_limit_properties(),
            '--property=Type=exec', '--property=KillMode=control-group',
            '--property=SendSIGKILL=yes', '--property=TimeoutStopSec=' + str(shutdown_grace + 10),
            '--property=StandardOutput=append:' + str(log),
            '--property=StandardError=append:' + str(log),
            '--working-directory=' + os.getcwd(), sys.executable, str(Path(__file__).resolve()),
            'supervise', str(path)]
    stopped = threading.Event()
    previous = {sig: signal.signal(sig, lambda *_: stopped.set()) for sig in (signal.SIGTERM, signal.SIGINT)}
    process = None
    stop_sent = False
    try:
        process = subprocess.Popen(args)
        while process.poll() is None:
            if stopped.is_set() and not stop_sent:
                subprocess.run(['sudo', '-n', 'systemctl', 'stop', '--no-block', unit], check=True, timeout=15)
                stop_sent = True
            time.sleep(.2)
        # systemd-run --wait returns after the unit, not merely the main PID.
        state = subprocess.check_output(['sudo', '-n', 'systemctl', 'show', unit,
                                         '-p', 'ActiveState', '-p', 'ControlGroup'], text=True, timeout=15)
        if 'ActiveState=active' in state or 'ActiveState=deactivating' in state:
            raise RuntimeError('runtime unit did not finish cleanup')
        return process.returncode
    finally:
        path.unlink(missing_ok=True)
        if process is not None and process.poll() is None:
            subprocess.run(['sudo', '-n', 'systemctl', 'stop', '--no-block', unit], timeout=15)
        for sig, handler in previous.items():
            signal.signal(sig, handler)


if __name__ == '__main__':
    if len(sys.argv) != 3 or sys.argv[1] != 'supervise':
        raise SystemExit('internal supervisor entrypoint')
    raise SystemExit(supervise(sys.argv[2]))
