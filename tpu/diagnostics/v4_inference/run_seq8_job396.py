"""Run one base check, one adapter check, then eight realistic RL samples."""
import http.client
import json
from pathlib import Path
import time

from seq8_trial import check_idle, fetch, file_sha256, payload_from_fixture, sample, trial


def main():
    root = Path.home() / ".cache/skyrl-ray"
    output = root / "runs/qwen-ray-v4-64-concurrency-001/validation/seq8-20260907"
    output.mkdir(parents=True, exist_ok=False)
    code = Path(__file__).parent
    fixture = json.loads((code / "job401-sampling-request4.json").read_text())
    archive = code / "job401-adapter.tar"
    if archive.stat().st_size != 641812480:
        raise RuntimeError("incomplete diagnostic adapter transfer")
    digest = file_sha256(archive)
    if digest != "43682c201e6346d54905b4277f9a00152ba9feeaf644ef469946badee4fa73da":
        raise RuntimeError("adapter differs from the source job401 archive")
    (output / "metadata.json").write_text(json.dumps(dict(
        time=time.time(), source_job=401, source_request=4, adapter_sha256=digest,
        tp=4, max_sequences=8, max_model_length=22528, memory_utilization=0.8,
        max_loras=1, samples=8, prompt_tokens=628, long_max_tokens=13196,
        caveat="Eight samples of the real first-phase request; no grader, trainer, or second phase."), indent=2))
    deadline = time.monotonic() + 3600
    with (output / "startup.jsonl").open("w") as log:
        while True:
            row = sample()
            log.write(json.dumps(row) + "\n")
            log.flush()
            try:
                fetch("http://127.0.0.1:19800/health")
                check_idle(row, "10.130.0.105", None)
                break
            except Exception as exc:
                if time.monotonic() > deadline:
                    raise TimeoutError("seq8 readiness deadline exceeded") from exc
                time.sleep(10)
    commands = []
    for proc in Path("/proc").glob("[0-9]*"):
        try:
            args = (proc / "cmdline").read_bytes().decode(errors="replace").split("\0")
            if any(a.endswith("/vllm_tpu_server.py") for a in args):
                commands.append(args)
        except OSError:
            continue
    if len(commands) != 1:
        raise RuntimeError("expected exactly one serving API process")
    args = commands[0]
    for flag, expected in (("--max-num-seqs", "8"), ("--tensor-parallel-size", "4"),
                           ("--max-model-len", "22528"), ("--max-loras", "1")):
        if args[args.index(flag) + 1] != expected:
            raise RuntimeError("unexpected live engine config: " + flag)
    base = "Qwen/Qwen3.5-27B"
    if not trial(payload_from_fixture(fixture, base, 64), output / "base-short", None, "10.130.0.105", 1800):
        return 1
    check_idle(sample(), "10.130.0.105", None)
    version = "model_0a14a571_ss0_seq1"
    connection = http.client.HTTPConnection("127.0.0.1", 19800, timeout=600)
    try:
        connection.putrequest("POST", "/skyrl/v1/upload_lora_adapter?lora_name=" + version)
        connection.putheader("Content-Length", str(archive.stat().st_size))
        connection.putheader("Content-Type", "application/octet-stream")
        connection.endheaders()
        with archive.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                connection.send(chunk)
        response = connection.getresponse()
        body = response.read().decode()
        (output / "adapter-upload.json").write_text(body)
        if response.status != 200:
            raise RuntimeError("adapter upload failed: " + body)
    finally:
        connection.close()
    if not trial(payload_from_fixture(fixture, version, 64), output / "adapter-short", version, "10.130.0.105", 1800):
        return 1
    return 0 if trial(payload_from_fixture(fixture, version, 13196), output / "adapter-long", version,
                      "10.130.0.105", 7200) else 1


if __name__ == "__main__":
    raise SystemExit(main())
