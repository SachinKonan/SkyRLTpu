"""Out-of-band command transport for multi-host accelerator backends.

The command plane deliberately uses TCP rather than JAX collectives. Inference
engines such as eSurge run accelerator collectives from a background thread;
using a second JAX collective to wait for the next SkyRL command can reorder
launches across hosts and halt the TPU program.
"""

from __future__ import annotations

import socket
import time
import traceback
from multiprocessing.connection import Client, Connection, Listener
from typing import Any

from pydantic import BaseModel


class RpcPayload(BaseModel):
    """Backend method name and JSON-serializable keyword arguments."""

    method: str
    kwargs: dict[str, Any]


def command_address(coordinator_address: str) -> tuple[str, int]:
    """Derive the CPU command-plane endpoint from JAX's coordinator address."""
    host, separator, port = coordinator_address.rpartition(":")
    if not separator or not host:
        raise ValueError(f"Expected coordinator address HOST:PORT, got {coordinator_address!r}")
    return host, int(port) + 1


def _authkey(coordinator_address: str) -> bytes:
    return f"skyrl-easydel:{coordinator_address}".encode()


class CommandServer:
    """Process-zero command fanout with explicit worker acknowledgements."""

    def __init__(self, coordinator_address: str, num_processes: int):
        host, port = command_address(coordinator_address)
        del host  # Listen on every interface; workers connect through the coordinator hostname.
        self._listener = Listener(("0.0.0.0", port), authkey=_authkey(coordinator_address))
        self._connections: dict[int, Connection] = {}
        while len(self._connections) < num_processes - 1:
            connection = self._listener.accept()
            process_id = int(connection.recv())
            if process_id <= 0 or process_id >= num_processes or process_id in self._connections:
                connection.close()
                raise RuntimeError(f"Invalid or duplicate EasyDeL worker process id {process_id}")
            self._connections[process_id] = connection

    def send(self, payload: RpcPayload) -> None:
        message = payload.model_dump(mode="json")
        for connection in self._connections.values():
            connection.send(message)

    def wait(self) -> None:
        errors = []
        for process_id, connection in self._connections.items():
            response = connection.recv()
            if not response.get("ok", False):
                errors.append(f"worker {process_id}: {response.get('error', 'unknown error')}")
        if errors:
            raise RuntimeError("EasyDeL worker command failed:\n" + "\n".join(errors))

    def close(self) -> None:
        for connection in self._connections.values():
            connection.close()
        self._listener.close()


class CommandClient:
    """Worker-side half of the out-of-band command channel."""

    def __init__(
        self,
        coordinator_address: str,
        process_id: int,
        *,
        connect_timeout_s: float = 300.0,
    ):
        host, port = command_address(coordinator_address)
        deadline = time.monotonic() + connect_timeout_s
        last_error: OSError | None = None
        while time.monotonic() < deadline:
            try:
                self._connection = Client((host, port), authkey=_authkey(coordinator_address))
                break
            except OSError as exc:
                last_error = exc
                time.sleep(0.25)
        else:
            raise TimeoutError(f"Could not connect to EasyDeL command server at {host}:{port}") from last_error
        self._connection.send(process_id)

    def receive(self) -> RpcPayload:
        return RpcPayload.model_validate(self._connection.recv())

    def acknowledge(self, error: BaseException | None = None) -> None:
        if error is None:
            self._connection.send({"ok": True})
            return
        self._connection.send(
            {
                "ok": False,
                "error": "".join(traceback.format_exception(type(error), error, error.__traceback__)),
            }
        )

    def close(self) -> None:
        self._connection.close()


def local_ipv4_address() -> str:
    """Resolve the current worker's advertised IPv4 address."""
    return socket.gethostbyname(socket.gethostname())
