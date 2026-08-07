# /// script
# requires-python = ">=3.10"
# dependencies = ["httpx>=0.27"]
# ///
"""llama-farm endpoint registry: the single source of truth for "where can I sample right now".

TPU workers hold PUBLIC IPs and the project firewall `strategist-vllm-ingress`
(128.112.0.0/16 -> tcp:8001-8012) already lets neuronic reach them, so clients talk to
`http://<worker-ip>:8001` directly -- no SSH tunnels. Two facts make a registry mandatory
rather than a convenience:

  * IPs change every time a spot slice is preempted and re-provisioned, so nothing may
    hardcode them;
  * a slice reporting READY does NOT mean its workers serve. Measured on a live v5p-64:
    2 of 8 workers did not answer /v1/models while the slice was READY. A client that sends
    to a dead worker stalls an entire step, so entries are health-GATED, not assumed.

The supervisor writes; every client reads. Writes are atomic (tmp + rename) because clients
poll this file between steps.

CLI:
  uv run registry.py show   --path <registry.json>
  uv run registry.py probe  --path <registry.json>            # re-health-check, rewrite
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import httpx

PORT = 8001
PROBE_TIMEOUT = 8.0


def probe(ip: str, port: int = PORT, api_key: str = "EMPTY", timeout: float = PROBE_TIMEOUT):
    """-> (healthy, served_model_id|None). Cheap GET; used for every entry before it is
    advertised."""
    try:
        r = httpx.get(f"http://{ip}:{port}/v1/models",
                      headers={"Authorization": f"Bearer {api_key}"}, timeout=timeout)
        if r.status_code != 200:
            return False, None
        data = r.json().get("data") or []
        # vLLM lists the base model first, then any LoRA adapters
        return True, (data[0].get("id") if data else None)
    except Exception:
        return False, None


def write(path, entries):
    """Atomic replace so a client never reads a half-written registry."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {"updated": round(time.time(), 1),
               "updated_str": time.strftime("%F %T"),
               "entries": entries}
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    os.replace(tmp, p)
    return payload


def read(path):
    p = Path(path)
    if not p.exists():
        return {"updated": 0, "entries": []}
    try:
        return json.loads(p.read_text())
    except Exception:
        return {"updated": 0, "entries": []}


def urls(path, model: str | None = None, healthy_only: bool = True):
    """Base URLs for sampling, e.g. ['http://34.1.2.3:8001', ...].

    `model` filters by the registry's model KEY (qwen35 / gemma4), which is what the
    composition factor is expressed in -- never by the served model id string."""
    out = []
    for e in read(path).get("entries", []):
        if healthy_only and not e.get("healthy"):
            continue
        if model and e.get("model") != model:
            continue
        out.append(e["url"])
    return out


def reprobe(path, api_key: str = "EMPTY"):
    """Re-health-check every known entry and rewrite. Returns (n_healthy, n_total)."""
    reg = read(path)
    entries = reg.get("entries", [])
    for e in entries:
        ok, served = probe(e["ip"], e.get("port", PORT), api_key)
        e["healthy"] = ok
        e["served_model"] = served
        e["checked"] = round(time.time(), 1)
        if ok:
            e["last_ok"] = e["checked"]
    write(path, entries)
    return sum(1 for e in entries if e["healthy"]), len(entries)


def make_entry(slice_name, worker, model, ip, port=PORT):
    return {"slice": slice_name, "worker": worker, "model": model, "ip": ip, "port": port,
            "url": f"http://{ip}:{port}", "healthy": False, "served_model": None,
            "last_ok": None, "checked": None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["show", "probe"])
    ap.add_argument("--path", required=True)
    ap.add_argument("--api-key", default="EMPTY")
    args = ap.parse_args()

    if args.cmd == "probe":
        h, n = reprobe(args.path, args.api_key)
        print(f"[registry] {h}/{n} healthy")

    reg = read(args.path)
    print(f"[registry] updated {reg.get('updated_str', 'never')}")
    by = {}
    for e in reg.get("entries", []):
        by.setdefault(e["model"], []).append(e)
    for model, es in sorted(by.items()):
        ok = [e for e in es if e.get("healthy")]
        print(f"  {model:8s}: {len(ok)}/{len(es)} healthy")
        for e in es:
            flag = "ok  " if e.get("healthy") else "DOWN"
            print(f"     {flag} {e['slice']}/w{e['worker']:<2} {e['url']:<28} {e.get('served_model') or ''}")


if __name__ == "__main__":
    main()
