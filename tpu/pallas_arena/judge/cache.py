"""hash->reward cache: repeat kernels get instant, byte-identical rewards.

Backends:
  * local directory (used by the CPU test battery and as the judge's warm
    local layer);
  * GCS prefix under gs://sk7524-pallas-arena-us-east5/reward-cache/ via the
    `gcloud storage` CLI (best-effort: every GCS error degrades to a miss —
    the cache must never take grading down with it).

The key is computed in grader.cache_key: sha256 over problem name, problem
version, grading mode and whitespace-normalized code, so a re-provisioned
(stateless) judge returns byte-identical rewards for repeat kernels.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path


class RewardCache:
    def __init__(self, root: str, *, gcloud_timeout_s: float = 30.0):
        self.root = root.rstrip("/")
        self.is_gcs = self.root.startswith("gs://")
        self.gcloud_timeout_s = gcloud_timeout_s
        if not self.is_gcs:
            Path(self.root).mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> str:
        return f"{self.root}/{key[:2]}/{key}.json"

    # ------------------------------------------------------------------ local
    def _local_get(self, key: str):
        p = Path(self._path(key))
        if not p.exists():
            return None
        try:
            return json.loads(p.read_text())
        except Exception:
            return None

    def _local_put(self, key: str, value: dict) -> None:
        p = Path(self._path(key))
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(value))
        os.replace(tmp, p)

    # ------------------------------------------------------------------- gcs
    def _gcs_get(self, key: str):
        with tempfile.NamedTemporaryFile(suffix=".json") as tf:
            try:
                r = subprocess.run(
                    ["gcloud", "storage", "cp", self._path(key), tf.name],
                    capture_output=True,
                    timeout=self.gcloud_timeout_s,
                )
                if r.returncode != 0:
                    return None
                with open(tf.name) as f:
                    return json.load(f)
            except Exception:
                return None

    def _gcs_put(self, key: str, value: dict) -> None:
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tf:
            json.dump(value, tf)
            tmp = tf.name
        try:
            subprocess.run(
                ["gcloud", "storage", "cp", tmp, self._path(key)], capture_output=True, timeout=self.gcloud_timeout_s
            )
        except Exception:
            pass
        finally:
            try:
                os.unlink(tmp)
            except OSError:
                pass

    # ------------------------------------------------------------------- api
    def get(self, key: str):
        return self._gcs_get(key) if self.is_gcs else self._local_get(key)

    def put(self, key: str, value: dict) -> None:
        if self.is_gcs:
            self._gcs_put(key, value)
        else:
            self._local_put(key, value)
