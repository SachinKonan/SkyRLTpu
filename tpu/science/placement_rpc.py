"""Bounded client for Circuit Training's native PlacementCost JSON protocol.

Protocol follows circuit_training/environment/plc_client.py (Apache-2.0,
Copyright 2021 The Circuit Training Team Authors). Adds explicit deadlines,
EOF handling and deterministic cleanup of process, sockets and socket path.
"""
import json
from pathlib import Path
import socket
import subprocess
import tempfile
import time


class PlacementCost:
    def __init__(self, binary, netlist, *, timeout=30):
        self.deadline=time.monotonic()+timeout
        self.temp = tempfile.TemporaryDirectory(prefix='science-plc-')
        self.listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.listener.settimeout(timeout)
        self.conn = self.process = None
        self.log = open(Path(self.temp.name) / 'native.log', 'w+b')
        address = str(Path(self.temp.name) / 'socket')
        try:
            self.listener.bind(address)
            self.listener.listen(1)
            self.process = subprocess.Popen([
                str(Path(binary).resolve()), '--uid=', '--gid=',
                '--pipe_address=' + address, '--netlist_file=' + str(Path(netlist).resolve()),
                '--macro_macro_x_spacing=0', '--macro_macro_y_spacing=0',
            ], stdout=self.log, stderr=self.log)
            self.conn, _ = self.listener.accept()
            self.conn.settimeout(timeout)
        except BaseException:
            self.close()
            raise

    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError(name)
        method = name.replace('_', ' ').title().replace(' ', '')

        def call(*args):
            remaining=self.deadline-time.monotonic()
            if remaining<=0:raise TimeoutError('native PlacementCost deadline exhausted')
            self.conn.settimeout(remaining)
            self.conn.sendall(json.dumps(dict(name=method, args=args), allow_nan=False).encode())
            data = b''
            while True:
                remaining=self.deadline-time.monotonic()
                if remaining<=0:raise TimeoutError('native PlacementCost deadline exhausted')
                self.conn.settimeout(remaining)
                part = self.conn.recv(1024 * 1024)
                if not part:
                    raise RuntimeError(f'PlacementCost exited during {method}')
                data += part
                if len(data) > 64 * 1024 * 1024:
                    raise ValueError('oversized native PlacementCost reply')
                try:
                    result = json.loads(data)
                    break
                except (json.JSONDecodeError, UnicodeDecodeError):
                    continue
            if isinstance(result, dict):
                if result.get('ok') is False:
                    raise ValueError(f'{method}: {result.get("message")}')
                if '__tuple__' in result:
                    return tuple(result['items'])
            if isinstance(result, list):
                return [tuple(x['items']) if isinstance(x, dict) and '__tuple__' in x else x for x in result]
            return result
        return call

    def close(self):
        if self.conn is not None:
            self.conn.close()
        if self.process is not None:
            if self.process.poll() is None:
                self.process.kill()
            self.process.wait(timeout=5)
        self.listener.close()
        self.log.close()
        self.temp.cleanup()
