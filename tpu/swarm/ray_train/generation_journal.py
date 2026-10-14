"""Client-side journal of generation groups sent to farms (acknowledged, polled protocol).

Each group is keyed by a content hash of its request (``group_key``), so the same
group asked for again after a client restart has the same key. Every transition
is appended to ``generation-journal.jsonl`` in the run directory:
``submitted`` -> ``acknowledged`` -> ``done`` | ``lost`` (rerouted) | ``dropped``,
plus ``reclaimed`` / ``reclaim_missing`` / ``duplicate_dropped``.

On restart the ingress re-polls every group the previous incarnation left
acknowledged-but-unfinished, using the run identity (``run`` + a token persisted
next to the journal) instead of the lost lease. A result that finished on the farm
meanwhile is cached and served when the client asks for that group again, so
only the missing groups are regenerated.

Exactly once: a request id's result is taken at most once (the farm hands it over
once, and the client never polls an abandoned id again). Two groups of one step
may legitimately send identical requests (the same parent chosen twice), so
reclaimed results queue per key and each ask takes one; reclaimed results nobody
asked for by the end of the phase are dropped and logged.
"""
import hashlib
import json
import os
from pathlib import Path
import secrets
import time

TERMINAL_EVENTS = ('done', 'lost', 'dropped', 'reclaimed', 'reclaim_missing', 'abandoned')


def group_key(payload):
    """Stable id of one generation group's request (model, prompt, n, sampling parameters)."""
    body = json.dumps(payload, sort_keys=True, separators=(',', ':'), default=str)
    return hashlib.sha256(body.encode()).hexdigest()[:32]


class GenerationJournal:
    def __init__(self, directory, run_id, report=lambda *a, **k: None, clock=time.time):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = self.directory / 'generation-journal.jsonl'
        self.run_id = run_id
        self.report = report
        self.clock = clock
        self.token = self._identity_token()
        self.cache = {}  # key -> [results reclaimed from a previous incarnation]
        self.open = self._load_open()

    def _identity_token(self):
        """Run identity for reclaiming groups across restarts; created once, private to the user."""
        path = self.directory / 'generation-identity.json'
        try:
            data = json.loads(path.read_text())
            if data.get('run') == self.run_id and isinstance(data.get('token'), str):
                return data['token']
        except (OSError, ValueError):
            pass
        token = secrets.token_hex(16)
        partial = path.with_suffix('.partial')
        fd = os.open(partial, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, 'w') as stream:
            json.dump({'run': self.run_id, 'token': token}, stream)
        os.replace(partial, path)
        return token

    @property
    def identity(self):
        return dict(run=self.run_id, token=self.token)

    @property
    def headers(self):
        return {'X-Generation-Run': self.run_id, 'X-Generation-Token': self.token}

    def _load_open(self):
        """Groups acknowledged by a farm but never finished, per the existing journal."""
        open_groups = {}
        try:
            lines = self.path.read_text().splitlines()
        except OSError:
            return open_groups
        for line in lines:
            try:
                row = json.loads(line)
            except ValueError:
                continue
            key = row.get('key')
            if not key:
                continue
            if row.get('event') == 'acknowledged':
                open_groups[key] = dict(url=row.get('url'), request_id=row.get('request_id'))
            elif row.get('event') in TERMINAL_EVENTS:
                open_groups.pop(key, None)
        return open_groups

    def record(self, event, key, **fields):
        row = dict(time=round(self.clock(), 3), event=event, key=key, **fields)
        with self.path.open('a') as stream:
            stream.write(json.dumps(row, default=str) + '\n')
        if event == 'acknowledged':
            self.open[key] = dict(url=fields.get('url'), request_id=fields.get('request_id'))
        elif event in TERMINAL_EVENTS:
            self.open.pop(key, None)
        try:
            self.report('generation_' + event, key=key, **fields)
        except Exception:
            pass

    def take_cached(self, key):
        """One result reclaimed from the previous incarnation for this request, handed over once."""
        results = self.cache.get(key)
        if not results:
            return None
        result = results.pop(0)
        if not results:
            del self.cache[key]
        self.record('reclaim_used', key)
        return result

    def deliver(self, key, **fields):
        self.record('done', key, **fields)

    def reclaimed(self, key, result, **fields):
        self.cache.setdefault(key, []).append(result)
        self.record('reclaimed', key, **fields)

    def discard_unused(self, reason='phase ended'):
        """Reclaimed results the client never asked for again are dropped, never kept for later steps."""
        for key, results in list(self.cache.items()):
            for _ in results:
                self.record('duplicate_dropped', key, reason=reason)
        self.cache.clear()


def registry_path_check(path, max_age=120, clock=time.time):
    """path_check(url) from the discovery registry: is a farm we cannot reach still healthy to discovery?

    'farm_ok': discovery reached it recently (our path, e.g. an SSH tunnel, is the suspect);
    'farm_down': discovery lists it unhealthy or no longer lists it; 'unknown': no fresh registry.
    """
    path = Path(path)

    def check(url):
        try:
            if clock() - path.stat().st_mtime > max_age:
                return 'unknown'
            farms = json.loads(path.read_text()).get('farms', [])
        except (OSError, ValueError, AttributeError):
            return 'unknown'
        row = next((f for f in farms if (f.get('url') or '').rstrip('/') == url.rstrip('/')), None)
        if row is None or not row.get('ok'):
            return 'farm_down'
        return 'farm_ok' if row.get('tunnel_ok', True) else 'farm_down'
    return check
