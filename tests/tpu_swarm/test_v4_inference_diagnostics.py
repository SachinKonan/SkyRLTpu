from tpu.diagnostics.v4_inference.collect import classify, excerpt
from tpu.diagnostics.v4_inference.summarize import summarize
from tpu.diagnostics.v4_inference.long_decode import counters, require_idle
import pytest


def test_fatal_device_error_is_distinct_from_its_engine_exit_symptom():
    assert classify("EngineDeadError", "Received Error Interrupt! fatal: true\n"
                    "Terminating process because the task is disconnected") == [
                        "engine_exit", "tpu_fatal", "runtime_terminated_process"]
    assert classify("EngineDeadError", "") == ["engine_exit"]


def test_mapping_warning_is_not_proof_of_fork_or_fatal_error():
    assert classify("", "Mapping VMA does not have VM_DONTCOPY set") == []


def test_separate_failure_classes():
    assert classify("RESOURCE_EXHAUSTED", "") == ["allocation_error"]
    assert classify("NotImplementedError", "") == ["unsupported_code_path"]
    assert classify("timed out", "") == []


def test_bounded_excerpt_keeps_start_and_end(tmp_path):
    path = tmp_path / "engine.log"
    path.write_bytes(b"begin" + b"x" * 100000 + b"end")
    text = excerpt(path, limit=16000)
    assert text.startswith("begin") and text.endswith("end")
    assert len(text) < 16100 and "middle omitted" in text
    path.write_bytes(b"small")
    assert excerpt(path) == "small"


def test_summary_retains_run_pid_and_provenance():
    doc = dict(host="host", environments={"serving": {"jax": "test"}}, runs={
        "run": {"engines": [dict(file="/root/engine-abc.log", pids=["123"],
                                  classification=["engine_exit"], driver_errors={"driver.123": "error"})]}})
    row = summarize([doc])[0]
    assert row["run"] == "run" and row["pids"] == ["123"]
    assert row["driver_files"] == ["driver.123"]
    assert row["serving_versions"] == {"jax": "test"}


def test_metrics_parser_supports_labels_and_unlabelled_counters():
    raw = '# HELP header\nvllm:num_requests_running{model="qwen"} 0\nvllm:num_requests_waiting 0\n'
    values = counters(raw)
    assert values == {"vllm:num_requests_running": 0, "vllm:num_requests_waiting": 0}
    require_idle(dict(active=0, updating=False, version=None, committed=None,
                      exhausted=[], replicas=[dict(ip="host")]), ["host"], {"host": values})


@pytest.mark.parametrize("field,value", [("active", 1), ("updating", True), ("version", "adapter"),
                                        ("exhausted", ["host"]), ("replicas", [])])
def test_control_refuses_busy_or_wrong_endpoint(field, value):
    status = dict(active=0, updating=False, version=None, committed=None,
                  exhausted=[], replicas=[dict(ip="host")])
    status[field] = value
    with pytest.raises(RuntimeError, match="refusing"):
        require_idle(status, ["host"], {})


def test_control_refuses_missing_engine_counters():
    with pytest.raises(RuntimeError, match="unknown requests"):
        require_idle(dict(active=0, updating=False, version=None, committed=None,
                          exhausted=[], replicas=[dict(ip="host")]), ["host"], {"host": {}})
