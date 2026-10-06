"""Device-free HTTP client for the Arena pull queue."""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request

class ArenaQueueClient:
    """Client-side helper: submit + wait with automatic resubmission.

    The client keeps every payload it submitted; if the queue restarts (it
    is memoryless by design) /result returns 404 and the item is silently
    resubmitted under a fresh work_id. Safe because grading is idempotent."""

    def __init__(self, base_url: str, timeout_s: float = 30.0):
        self.base = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self._payloads: dict[str, dict] = {}

    def _post(self, path: str, payload: dict) -> dict:
        req = urllib.request.Request(
            self.base + path,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
            return json.loads(resp.read())

    def _get(self, path: str) -> dict:
        with urllib.request.urlopen(self.base + path, timeout=self.timeout_s) as resp:
            return json.loads(resp.read())

    def submit(self, problem: str, code: str, **kw) -> str:
        payload = {"problem": problem, "code": code, **kw}
        work_id = self._post("/submit", payload)["work_id"]
        self._payloads[work_id] = payload
        return work_id

    def status(self) -> dict:
        return self._get("/status")

    def leases(self) -> list[dict]:
        return self._get("/leases")["leased"]

    def poll_bulk(self, work_ids: list[str], batch: int = 400) -> dict[str, dict]:
        """One request per <=batch outstanding ids; returns only the done
        ones, with their queue-side accounting (attempts, terminal, which
        judge held the lease)."""
        out: dict[str, dict] = {}
        for i in range(0, len(work_ids), batch):
            chunk = work_ids[i : i + batch]
            try:
                got = self._post("/results", {"work_ids": chunk})["results"]
            except Exception:
                continue
            for wid, rec in got.items():
                if rec.get("done"):
                    out[wid] = rec
        return out

    def wait(self, work_ids: list[str], timeout_s: float = 600.0, poll_s: float = 1.0) -> dict[str, dict]:
        """Wait for all ids; returns {original_work_id: result}. Resubmits
        through queue restarts and maps the fresh id back to the original."""
        alias = {wid: wid for wid in work_ids}  # original -> current
        results: dict[str, dict] = {}
        deadline = time.time() + timeout_s
        while time.time() < deadline and len(results) < len(work_ids):
            for orig in work_ids:
                if orig in results:
                    continue
                cur = alias[orig]
                try:
                    r = self._get(f"/result/{cur}")
                except urllib.error.HTTPError as e:
                    if e.code == 404 and cur in self._payloads:
                        # queue lost the item (restart): resubmit
                        new_id = self._post("/submit", self._payloads[cur])["work_id"]
                        self._payloads[new_id] = self._payloads[cur]
                        alias[orig] = new_id
                    continue
                except Exception:
                    continue  # queue briefly down: retry next poll
                if r.get("done"):
                    results[orig] = r["result"]
            if len(results) < len(work_ids):
                time.sleep(poll_s)
        return results

