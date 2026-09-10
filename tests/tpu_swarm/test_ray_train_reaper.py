"""The detached reaper stops only this run's pre-existing Ray daemons (jobs 612/614 -> 613)."""
from types import SimpleNamespace

from tpu.swarm.ray_train.reaper import matches, select_victims

TMP = "/home/gcpuser/.rtr-6579e689d3"


def proc(pid, cmdline, created, children=()):
    p = SimpleNamespace(pid=pid, cmdline=lambda: cmdline, create_time=lambda: created,
                        children=lambda recursive=True: list(children))
    return p


def test_matches_only_this_temp_dir():
    assert matches(["gcs_server", f"--log_dir={TMP}/session_x/logs"], TMP)
    assert matches(["ray", "start", f"--temp-dir={TMP}"], TMP) is False  # value is a separate arg form
    assert matches(["ray", "start", "--temp-dir", TMP], TMP)
    assert not matches(["gcs_server", "--log_dir=/home/gcpuser/skypilot-runtime/ray/session/logs"], TMP)


def test_select_victims_skips_newer_processes_and_self():
    old_gcs = proc(100, ["gcs_server", f"--log_dir={TMP}/s/logs"], 1000.0,
                   children=[proc(101, ["log_monitor"], 1001.0), proc(102, ["ray::Host"], 2001.0)])
    new_gcs = proc(200, ["gcs_server", f"--log_dir={TMP}/s2/logs"], 2000.0)
    sky_ray = proc(300, ["gcs_server", "--log_dir=/home/gcpuser/skypilot-runtime/ray/logs"], 900.0)
    me = proc(400, ["python", "-m", "tpu.swarm.ray_train.reaper", TMP, "5"], 999.0)
    victims = select_victims([old_gcs, new_gcs, sky_ray, me], TMP, before=1500.0, self_pid=400)
    assert sorted(victims) == [100, 101]
