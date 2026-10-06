"""Read-only host evidence; never imports JAX or opens TPU devices."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import re
import subprocess
import time


def excerpt(path, limit=65536):
    size = path.stat().st_size
    with path.open("rb") as stream:
        head = stream.read(min(8192, limit))
        stream.seek(max(len(head), size - max(0, limit - len(head))))
        tail = stream.read(max(0, limit - len(head)))
    return (head + (b"\n[... middle omitted ...]\n" if size > limit else b"") + tail).decode(errors="replace")


def classify(engine, driver):
    facts = []
    if "EngineDeadError" in engine or "engine core exited unexpectedly" in engine:
        facts.append("engine_exit")
    if "Received Error Interrupt! fatal: true" in driver or "fatal_error: 1" in driver:
        facts.append("tpu_fatal")
    if "Terminating process because the task is disconnected" in driver:
        facts.append("runtime_terminated_process")
    if re.search(r"RESOURCE_EXHAUSTED|out of memory|OutOfMemoryError", engine, re.I):
        facts.append("allocation_error")
    if "NotImplementedError" in engine:
        facts.append("unsupported_code_path")
    return facts


def collect(since, run_names):
    root = Path.home() / ".cache/skyrl-ray"
    result = dict(time=time.time(), host=platform.node(), kernel=platform.release(),
                  boot_id=Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
                  meminfo=Path("/proc/meminfo").read_text(), environments={}, runs={})
    packages = {"jax", "jaxlib", "libtpu", "vllm", "vllm-tpu", "tpu-inference",
                "torch", "torchax", "transformers", "ray", "tokamax"}
    for env in ("serving", "trainer", "controller"):
        sites = list((root / "envs" / env / "lib").glob("python*/site-packages"))
        result["environments"][env] = {
            d.metadata["Name"]: d.version for d in importlib.metadata.distributions(path=[str(s) for s in sites])
            if d.metadata["Name"].lower().replace("_", "-") in packages}
        if env == "serving":
            digest = hashlib.sha256()
            count = 0
            for site in sites:
                for path in sorted((site / "tpu_inference").rglob("*.py")):
                    digest.update(str(path.relative_to(site)).encode() + b"\0" + path.read_bytes())
                    count += 1
            result["installed_tpu_inference"] = dict(python_files=count, sha256=digest.hexdigest())
    result["serving_processes"] = []
    env_keys = ("USE_BATCHED_RPA_KERNEL", "USE_JAX_RAGGED_CONV1D", "TPU_BACKEND_TYPE",
                "VLLM_WORKER_MULTIPROC_METHOD", "TPU_PROCESS_BOUNDS", "TPU_CHIPS_PER_PROCESS_BOUNDS",
                "TPU_PROCESS_ADDRESSES", "TPU_VISIBLE_CHIPS", "RAY_NAMESPACE")
    for proc in Path("/proc").glob("[0-9]*"):
        try:
            args = (proc / "cmdline").read_bytes().decode(errors="replace").split("\0")
            if not any(arg.endswith("/vllm_tpu_server.py") for arg in args):
                continue
            env = dict(item.split("=", 1) for item in
                       (proc / "environ").read_bytes().decode(errors="replace").split("\0") if "=" in item)
            result["serving_processes"].append(dict(pid=int(proc.name),
                environment={k: env[k] for k in env_keys if k in env},
                flags={flag: args[args.index(flag)+1] for flag in (
                    "--tensor-parallel-size", "--max-model-len", "--max-num-seqs",
                    "--max-num-batched-tokens", "--gpu-memory-utilization") if flag in args}))
        except (OSError, IndexError):
            continue
    result["source_identities"] = [p.name for p in (root / "sources").glob("*") if p.is_dir()]
    driver_root = Path("/tmp/tpu_logs")
    for run in sorted((root / "runs").glob("*")):
        if not run.is_dir() or run_names and run.name not in run_names:
            continue
        engines = []
        for path in sorted(run.glob("engine-*.log")):
            text = excerpt(path)
            pids = sorted(set(re.findall(r"EngineCore pid=(\d+)", text)))
            drivers = {}
            for pid in pids:
                for driver in sorted(driver_root.glob("*ERROR.*." + pid)):
                    if not driver.is_symlink():
                        drivers[driver.name] = excerpt(driver)
            engines.append(dict(file=str(path), size=path.stat().st_size, mtime=path.stat().st_mtime,
                                pids=pids, log_excerpt=text, driver_errors=drivers,
                                classification=classify(text, "\n".join(drivers.values()))))
        if engines:
            result["runs"][run.name] = dict(engines=engines)
            for name in ("inference-events.jsonl", "controller.jsonl"):
                path = run / name
                if path.is_file():
                    result["runs"][run.name][name] = excerpt(path)
    kernel = subprocess.run(["sudo", "-n", "journalctl", "-k", "--since", since,
                             "--no-pager", "-o", "short-iso"], capture_output=True, text=True, timeout=45)
    lines = kernel.stdout.splitlines()
    warnings = [line for line in lines if "VM_DONTCOPY" in line]
    important = [line for line in lines if re.search(
        r"oom|out of memory|killed process|segfault|error_status|chip specific status|gasket_open|gasket_close", line, re.I)]
    result["kernel_log"] = dict(returncode=kernel.returncode, stderr=kernel.stderr,
                               since=since, mapping_warning_count=len(warnings),
                               mapping_warning_examples=warnings[:2], events=important[-160:])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--since", default="2026-09-07 00:00:00")
    parser.add_argument("--run", action="append", default=[])
    args = parser.parse_args()
    print(json.dumps(collect(args.since, args.run)))


if __name__ == "__main__":
    main()
