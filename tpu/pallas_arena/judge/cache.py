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
    # Two GCS backends. The python client is ~30 ms per object; shelling out
    # to `gcloud storage cp` is 1-3 s, almost entirely CLI startup, which on
    # a ~6 s grading lane would be a 30-50% throughput tax and would make the
    # fleet's measured candidates/min a measurement of the gcloud CLI. The
    # CLI stays as the no-dependency fallback. Every GCS error degrades to a
    # cache miss: the cache must never take grading down with it.
    _client_state = None

    def _bucket(self):
        if RewardCache._client_state is False:
            return None
        try:
            from google.cloud import storage  # type: ignore

            if RewardCache._client_state is None:
                RewardCache._client_state = storage.Client()
        except Exception:
            RewardCache._client_state = False
            return None
        name = self.root[len("gs://") :].split("/", 1)[0]
        return RewardCache._client_state.bucket(name)

    def _blob_name(self, key: str) -> str:
        rest = self.root[len("gs://") :].split("/", 1)
        prefix = rest[1] if len(rest) > 1 else ""
        return f"{prefix.rstrip('/')}/{key[:2]}/{key}.json".lstrip("/")

    def _api_get(self, key: str):
        b = self._bucket()
        if b is None:
            return None
        try:
            blob = b.blob(self._blob_name(key))
            if not blob.exists():
                return None
            return json.loads(blob.download_as_bytes())
        except Exception:
            return None

    def _api_put(self, key: str, value: dict) -> bool:
        b = self._bucket()
        if b is None:
            return False
        try:
            b.blob(self._blob_name(key)).upload_from_string(json.dumps(value), content_type="application/json")
            return True
        except Exception:
            return False

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
        if not self.is_gcs:
            return self._local_get(key)
        hit = self._api_get(key)
        return hit if hit is not None or RewardCache._client_state else self._gcs_get(key)

    def put(self, key: str, value: dict) -> None:
        if not self.is_gcs:
            self._local_put(key, value)
        elif not self._api_put(key, value):
            self._gcs_put(key, value)
