"""Dependency-light tests of the controller boot guard, without cloud access."""

import ast
import os
from pathlib import Path
import socket
import time
import unittest
from unittest import mock


SOURCE = (Path(__file__).resolve().parents[2] / 'third_party/TPUSwarm/'
          'third_party/skypilot/sky/serve/service.py')


class ControllerBootTimeoutTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tree = ast.parse(SOURCE.read_text())
        names = {'_controller_boot_timeout_seconds', '_wait_for_controller_ready'}
        module = ast.Module(body=[node for node in cls.tree.body
                                  if isinstance(node, ast.FunctionDef)
                                  and node.name in names], type_ignores=[])
        cls.namespace = {'os': os, 'logger': mock.Mock(), 'socket': socket,
                         'time': time}
        exec(compile(module, str(SOURCE), 'exec'), cls.namespace)

    def test_default_and_override(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(self.namespace['_controller_boot_timeout_seconds'](),
                             1800)
        with mock.patch.dict(os.environ,
                             SKYPILOT_CONTROLLER_BOOT_TIMEOUT_SECONDS='2400'):
            self.assertEqual(self.namespace['_controller_boot_timeout_seconds'](),
                             2400)

    def test_invalid_values_keep_bounded_safe_default(self):
        for value in ('0', '-1', 'infinite', 'nan', '1.5', ''):
            with self.subTest(value=value), mock.patch.dict(
                    os.environ, SKYPILOT_CONTROLLER_BOOT_TIMEOUT_SECONDS=value):
                self.assertEqual(
                    self.namespace['_controller_boot_timeout_seconds'](), 1800)

    def test_delayed_listener_survives_old_sixty_second_deadline(self):
        clock = mock.Mock()
        clock.monotonic.side_effect = [0, 0, 61, 120]
        network = mock.Mock()
        network.create_connection.side_effect = [
            ConnectionRefusedError(), ConnectionRefusedError(), mock.MagicMock()]
        with mock.patch.dict(self.namespace, time=clock, socket=network):
            self.namespace['_wait_for_controller_ready']('0.0.0.0', 20003, 1800)
        self.assertEqual(network.create_connection.call_count, 3)
        network.create_connection.assert_called_with(('127.0.0.1', 20003),
                                                      timeout=0.5)

    def test_deadline_remains_bounded(self):
        clock = mock.Mock()
        clock.monotonic.side_effect = [0, 0, 1801]
        network = mock.Mock()
        network.create_connection.side_effect = ConnectionRefusedError()
        with mock.patch.dict(self.namespace, time=clock, socket=network):
            with self.assertRaisesRegex(RuntimeError, 'within 1800s'):
                self.namespace['_wait_for_controller_ready'](
                    '127.0.0.1', 20003, 1800)

    def test_start_passes_same_deadline_to_wait_and_preserving_bailout(self):
        start = next(node for node in self.tree.body
                     if isinstance(node, ast.FunctionDef) and node.name == '_start')
        calls = {node.func.id: node for node in ast.walk(start)
                 if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)}
        wait = calls['_wait_for_controller_ready']
        deadline = next(kw.value for kw in wait.keywords if kw.arg == 'timeout')
        self.assertEqual(deadline.id, 'boot_timeout')
        self.assertEqual(calls['_bail_on_boot_failure'].args[2].id, 'boot_timeout')


if __name__ == '__main__':
    unittest.main()
