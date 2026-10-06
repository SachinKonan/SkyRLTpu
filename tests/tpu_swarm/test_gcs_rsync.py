import os
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "tpu" / "gcs_rsync.sh"


def _write_cli(path: Path, body: str) -> None:
    path.write_text(f"#!/usr/bin/env bash\nset -eu\n{body}\n", encoding="utf-8")
    path.chmod(0o755)


def _run(
    tmp_path: Path,
    gcloud_body: str,
    gsutil_body: str,
    *extra_args: str,
) -> list[str]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    log = tmp_path / "calls.log"
    _write_cli(bin_dir / "gcloud", gcloud_body)
    _write_cli(bin_dir / "gsutil", gsutil_body)
    env = os.environ.copy()
    env["PATH"] = f"{bin_dir}:{env['PATH']}"
    env["CALL_LOG"] = str(log)
    subprocess.run(
        [
            str(SCRIPT),
            "-r",
            *extra_args,
            "gs://bucket/source",
            "/tmp/destination",
        ],
        check=True,
        env=env,
    )
    return log.read_text(encoding="utf-8").splitlines()


def test_uses_gcloud_storage_rsync_when_supported(tmp_path: Path) -> None:
    calls = _run(
        tmp_path,
        """
if [[ "$*" == "storage rsync --help" ]]; then exit 0; fi
printf 'gcloud:%s\n' "$*" >>"$CALL_LOG"
""",
        "printf 'gsutil:%s\n' \"$*\" >>\"$CALL_LOG\"",
    )
    assert calls == [
        "gcloud:storage rsync -r gs://bucket/source /tmp/destination"
    ]


def test_falls_back_to_parallel_gsutil_on_old_gcloud(tmp_path: Path) -> None:
    calls = _run(
        tmp_path,
        """
if [[ "$*" == "storage rsync --help" ]]; then exit 2; fi
printf 'gcloud:%s\n' "$*" >>"$CALL_LOG"
""",
        "printf 'gsutil:%s\n' \"$*\" >>\"$CALL_LOG\"",
    )
    assert calls == [
        "gsutil:-m rsync -r gs://bucket/source /tmp/destination"
    ]


def test_translates_exclude_for_gsutil_fallback(tmp_path: Path) -> None:
    calls = _run(
        tmp_path,
        """
if [[ "$*" == "storage rsync --help" ]]; then exit 2; fi
printf 'gcloud:%s\n' "$*" >>"$CALL_LOG"
""",
        "printf 'gsutil:%s\n' \"$*\" >>\"$CALL_LOG\"",
        r"--exclude=.*\.tmp$|.*\.partial$",
    )
    assert calls == [
        r"gsutil:-m rsync -r -x .*\.tmp$|.*\.partial$ "
        "gs://bucket/source /tmp/destination"
    ]
