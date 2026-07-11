"""Regression tests for the EasyDeL CPU command plane."""

from __future__ import annotations

import socket
import threading

import pytest

from skyrl.backends.multihost import CommandClient, CommandServer, RpcPayload


def _coordinator_address() -> str:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        command_port = probe.getsockname()[1]
    return f"127.0.0.1:{command_port - 1}"


def _connected_pair() -> tuple[CommandServer, CommandClient]:
    coordinator = _coordinator_address()
    holder: list[CommandServer] = []
    server_thread = threading.Thread(target=lambda: holder.append(CommandServer(coordinator, 2)))
    server_thread.start()
    client = CommandClient(coordinator, 1, connect_timeout_s=5.0)
    server_thread.join(timeout=5.0)
    assert not server_thread.is_alive()
    return holder[0], client


def test_command_channel_roundtrip():
    server, client = _connected_pair()
    try:
        payload = RpcPayload(method="forward", kwargs={"step": 3})
        server.send(payload)
        assert client.receive() == payload
        client.acknowledge()
        server.wait()
    finally:
        client.close()
        server.close()


def test_command_channel_propagates_worker_traceback():
    server, client = _connected_pair()
    try:
        server.send(RpcPayload(method="forward", kwargs={}))
        client.receive()
        try:
            raise ValueError("worker exploded")
        except ValueError as exc:
            client.acknowledge(exc)
        with pytest.raises(RuntimeError, match="worker exploded"):
            server.wait()
    finally:
        client.close()
        server.close()
