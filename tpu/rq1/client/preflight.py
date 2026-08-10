# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""Site preflight for the RQ1 collector. PORTABLE, stdlib-only.

Probes the host with a few short `codex exec` calls and local checks, then prints a site-profile
recommendation. The workarounds it decides between were all earned the hard way on neuronic:

  landlock      neuronic sets max_user_namespaces=0, so codex's default bubblewrap sandbox dies
                ("bwrap: ... No space left on device") and agents silently lose ALL shell access.
                `[features] use_legacy_landlock=true` sandboxes without user namespaces.
  openai-long   some nodes drop the model stream mid-reasoning ("stream disconnected before
                completion: idle timeout"); a custom provider with stream_idle_timeout_ms=1800000
                fixes it. Harmless elsewhere, so any stream hiccup in the probe turns it on.

Usage:  uv run preflight.py [--farm-url URL --farm-key KEY] [--json]
Also importable: collect_t1/orchestrate_t3 call recommend() for --site auto.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
import time
from pathlib import Path


def write_codex_home(root: Path, landlock: bool, long_provider: bool, multi_agent: int = 0,
                     mcp_url: str | None = None) -> Path:
    """Isolated CODEX_HOME. Feature keys MUST be in config.toml: `-c` overrides are not
    validated and silently ignored (bitten twice)."""
    ch = Path(root) / "codex_home"
    ch.mkdir(parents=True, exist_ok=True)
    src = Path.home() / ".codex" / "auth.json"
    if src.exists():
        (ch / "auth.json").write_text(src.read_text())
    parts = []
    if long_provider:
        parts.append('model_provider = "openai-long"\n')
    feats = []
    if landlock:
        feats.append("use_legacy_landlock = true")
    if feats:
        parts.append("[features]\n" + "\n".join(feats) + "\n")
    if multi_agent:
        # V2 defaults to 4 threads INCLUDING root (3 concurrent children) and sol hides
        # spawn_agent's model args (openai/codex#31814) -- both must be overridden here.
        parts.append("[features.multi_agent_v2]\n"
                     "hide_spawn_agent_metadata = false\n"
                     'tool_namespace = "agents"\n'
                     "expose_spawn_agent_model_overrides = true\n"
                     f"max_concurrent_threads_per_session = {multi_agent}\n")
    if long_provider:
        parts.append("[model_providers.openai-long]\n"
                     'name = "OpenAI ChatGPT (long idle timeout)"\n'
                     'base_url = "https://chatgpt.com/backend-api/codex"\n'
                     'wire_api = "responses"\n'
                     "requires_openai_auth = true\n"
                     "stream_idle_timeout_ms = 1800000\n"
                     "stream_max_retries = 20\n"
                     "request_max_retries = 10\n")
    if mcp_url:
        parts.append("[mcp_servers.capture]\n"
                     f'url = "{mcp_url}"\n'
                     "tool_timeout_sec = 1800\n"
                     "startup_timeout_sec = 60\n"
                     'default_tools_approval_mode = "approve"\n')
    (ch / "config.toml").write_text("\n".join(parts) if parts else "# defaults\n")
    return ch


def _codex_probe(prompt: str, ch: Path, timeout: int = 300, model: str = "gpt-5.6-sol"):
    env = dict(**__import__("os").environ, CODEX_HOME=str(ch))
    npm = Path.home() / ".npm-global" / "bin"          # neuronic's codex install location
    if npm.exists():
        env["PATH"] = f"{npm}:{env['PATH']}"
    with tempfile.TemporaryDirectory() as wd:
        try:
            r = subprocess.run(
                ["codex", "exec", "--strict-config", "-m", model,
                 "-c", "model_reasoning_effort=low", "-s", "workspace-write",
                 "-c", "approval_policy=never", "--json", "-C", wd],
                input=prompt, capture_output=True, text=True, timeout=timeout, env=env)
            return (r.stdout or "") + (r.stderr or "")
        except subprocess.TimeoutExpired as e:
            return (e.stdout or b"").decode(errors="replace") if e.stdout else "PROBE-TIMEOUT"
        except FileNotFoundError:
            return "NO-CODEX-BINARY"


def recommend(scratch: Path, farm_url: str = None, farm_key: str = "EMPTY") -> dict:
    """Run all probes; return {landlock, long_provider, gxx, codex, stream_ok, shell_ok, ...}."""
    rep = {"codex": shutil.which("codex") is not None or
                    (Path.home() / ".npm-global/bin/codex").exists(),
           "uv": shutil.which("uv") is not None,
           "gxx": None, "python3": shutil.which("python3") is not None,
           "shell_ok": None, "landlock": False, "long_provider": False, "farm_ok": None}
    g = shutil.which("g++")
    if g:
        try:
            v = subprocess.run(["g++", "--version"], capture_output=True, text=True, timeout=20)
            rep["gxx"] = (v.stdout or "").splitlines()[0]
        except Exception:
            rep["gxx"] = "g++ present, --version failed"
    if not rep["codex"]:
        rep["error"] = "codex CLI not found on PATH"
        return rep

    # probe 1: default sandbox shell
    ch = write_codex_home(scratch / "pf_default", landlock=False, long_provider=False)
    out = _codex_probe("Run this exact shell command and show its output: echo SBX_OK_31337", ch)
    rep["shell_ok"] = "SBX_OK_31337" in out
    bwrap = ("bwrap" in out and "namespace" in out) or "Creating new namespace failed" in out
    if bwrap or not rep["shell_ok"]:
        # probe 1b: does landlock fix it?
        ch2 = write_codex_home(scratch / "pf_landlock", landlock=True, long_provider=False)
        out2 = _codex_probe("Run this exact shell command and show its output: echo SBX_OK_31337", ch2)
        if "SBX_OK_31337" in out2:
            rep["landlock"] = True
            rep["shell_ok"] = True
    # probe 2: stream stability signal (any disconnect in either probe -> long provider)
    if "stream disconnected" in out or "idle timeout" in out:
        rep["long_provider"] = True
    if farm_url:
        try:
            import urllib.request
            req = urllib.request.Request(f"{farm_url.rstrip('/')}/v1/models",
                                         headers={"Authorization": f"Bearer {farm_key}"})
            with urllib.request.urlopen(req, timeout=20) as f:
                rep["farm_ok"] = f.status == 200
        except Exception as e:
            rep["farm_ok"] = f"unreachable: {e}"
    return rep


def resolve_site(site: str, scratch: Path) -> dict:
    """Map --site to {landlock, long_provider}. `auto` runs the probes (2 short codex calls)."""
    if site == "neuronic":
        return {"landlock": True, "long_provider": True}
    if site == "default":
        return {"landlock": False, "long_provider": False}
    rep = recommend(scratch)
    if rep.get("error"):
        raise SystemExit(f"preflight: {rep['error']}")
    print(f"[preflight] {json.dumps(rep)}", flush=True)
    return {"landlock": bool(rep["landlock"]), "long_provider": bool(rep["long_provider"])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--farm-url", default=None)
    ap.add_argument("--farm-key", default="EMPTY")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    scratch = Path(tempfile.mkdtemp(prefix="rq1_preflight_"))
    t0 = time.time()
    rep = recommend(scratch, args.farm_url, args.farm_key)
    rep["secs"] = round(time.time() - t0, 1)
    if args.json:
        print(json.dumps(rep, indent=2))
    else:
        print(f"codex CLI:        {'OK' if rep['codex'] else 'MISSING'}")
        print(f"uv:               {'OK' if rep['uv'] else 'MISSING'}")
        print(f"g++:              {rep['gxx'] or 'MISSING (needed for C++ problems)'}")
        print(f"agent shell:      {'OK' if rep.get('shell_ok') else 'BROKEN'}"
              + (" (via legacy landlock)" if rep.get("landlock") else ""))
        print(f"stream:           {'needs openai-long provider' if rep.get('long_provider') else 'OK'}")
        if rep.get("farm_ok") is not None:
            print(f"farm:             {rep['farm_ok']}")
        prof = "neuronic" if (rep.get("landlock") or rep.get("long_provider")) else "default"
        print(f"\nrecommended --site {prof}")
    return rep


if __name__ == "__main__":
    main()
