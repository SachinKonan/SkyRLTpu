"""Reload the last uploaded adapter after an in-place vLLM process restart.

Run on the inference host using only the standard library. The upload endpoint
reuses the extracted adapter when given an empty body; it also applies MoE
sidecars, unlike the generic vLLM load endpoint.
"""

import argparse
import json
import os
from pathlib import Path
from urllib.error import URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


def restore_adapters(lora_dir: Path, port: int, engines: int, timeout: float = 1200) -> None:
    marker = lora_dir / ".skyrl-latest-lora"
    if not marker.exists():
        print("No previously uploaded adapter to restore", flush=True)
        return
    name = marker.read_text().strip()
    if not name or name.startswith(".") or "/" in name or "\\" in name:
        raise ValueError("Invalid last-uploaded adapter marker")
    if not (lora_dir / name / "adapter_config.json").is_file():
        raise FileNotFoundError(f"Last-uploaded adapter is incomplete: {name}")

    headers = {"Authorization": f"Bearer {os.environ.get('VLLM_API_KEY', 'EMPTY')}"}

    def listed(base_url: str) -> bool:
        with urlopen(Request(f"{base_url}/v1/models", headers=headers), timeout=10) as response:
            models = json.load(response)
        return any(model.get("id") == name for model in models.get("data", []))

    for engine in range(engines):
        base_url = f"http://127.0.0.1:{port + engine}"
        if not listed(base_url):
            print(f"Restoring adapter {name} on port {port + engine}", flush=True)
            request = Request(
                f"{base_url}/skyrl/v1/upload_lora_adapter?{urlencode({'lora_name': name})}",
                data=b"",
                headers=headers,
                method="POST",
            )
            try:
                with urlopen(request, timeout=timeout) as response:
                    response.read()
            except (URLError, TimeoutError):
                # A compile can finish after the HTTP caller loses its ACK.
                if not listed(base_url):
                    raise
            if not listed(base_url):
                raise RuntimeError(f"Adapter {name} is still absent on port {port + engine}")
        print(f"Adapter ready: {name} on port {port + engine}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--engines", type=int, default=1)
    parser.add_argument("--lora-dir", default="")
    args = parser.parse_args()
    if args.engines < 1:
        parser.error("--engines must be positive")
    restore_adapters(
        Path(args.lora_dir) if args.lora_dir else Path.home() / "skyrl-local-loras",
        args.port,
        args.engines,
    )
