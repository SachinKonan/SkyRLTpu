# Multi-LoRA inference farm

`profiles/qwen35-v432-native-multi-lora-inference-20260919.json` runs the existing
Ray v2 executor in inference-only mode. Ranks 0, 1, 2 and 3 each run one TP4
engine; no TPU is reserved for training. Ray Serve ingress listens on port
24800 and engines on 24801. The profile uses systemd ownership, native thinking,
one adapter slot, and a private compilation-cache destination seeded from the
previous native multi-LoRA run. No alternate engine launcher is involved.

Prefix caching is explicitly disabled with `--no-enable-prefix-caching`.
The cached greedy test on this Qwen runtime diverged even with the base model
selected; fresh-cache repetitions matched exactly. `prefix_caching: false`
now emits the explicit negative flag instead of relying on vLLM's default.
Model-download and compilation caches
remain enabled. See `tpu/results/multi-lora-farm-v432-20260919/README.md`.

Clients on a reachable private network send requests to the head's port 24800.
The profile does not create a public authenticated endpoint or a stable DNS
name. Each separately launched farm has its own endpoint and adapter registry.

Upload a PEFT tar body to
`POST /skyrl/v1/upload_lora_adapter?lora_name=NAME`. This is not a pathname API:
the sender can stream bytes from a local archive or memory. The ingress stores
the archive, drains active generation, publishes to every engine, then commits
the version. Use `previous_lora_name=OLD` when replacing an adapter at capacity.
Generation uses `/v1/completions` with `model=NAME`. The native token IDs, masks,
log probabilities and thinking audit pass through the shared ingress.

Each September 19 production farm is limited to one active adapter, replicated
to its four engines, pending investigation of concurrency-dependent logprobs.
A new version replaces the previous adapter; it does not add a second policy.
The two-adapter acceptance procedure below requires a separate capacity-two
test profile and is not enabled on these farms.

## Hardware acceptance

`validate_lora_farm.py` starts no engines. It exercises the deployed executor:

1. On the head engine, load real exported weights through the standard local
   `lora_path` API. Save deterministic native completions and token logprobs,
   then unload those reference adapters. Also save base-only completions.
2. From another host, upload the same checksummed tar archives to ingress.
   Require acknowledgements from all four engines.
3. Repeat the identical token-ID requests directly on every engine, with both
   adapters concurrently on each engine, and through ingress with alternating
   concurrent adapter requests. Require identical
   token IDs, finish reasons and masks, finite logprobs, and a maximum absolute
   sampled-token logprob difference of 1e-4. Forced positions must agree with
   the loss mask. The wire API retains their model logprobs; the existing
   training client normalizes those masked positions to zero.
4. Replace the first adapter at capacity using `previous_lora_name`, then
   recheck its probabilities and the untouched peer through ingress.
5. Require each adapter to differ from the base-only output, the two expected
   versions to remain committed, no engine restart, and no requests left active.

Reference and farm modes take the same JSON manifest with `base_model`,
`requests`, `engines`, and `adapters`. Each adapter supplies `name`, `local_path`,
`archive`, and `sha256`. The manifest's requests use the production prompt
renderer and native payload builder, with greedy decoding and a bounded native
budget for reproducibility. This verifies serving parity, not RL reward quality
or trainer-versus-vLLM numerical parity. Two versions of one trained policy are
sufficient to verify adapter selection; they are not independent training seeds.

```
python validate_lora_farm.py reference --manifest manifest.json \
  --endpoint http://127.0.0.1:24801 --output reference
python validate_lora_farm.py farm --manifest manifest.json \
  --endpoint http://HEAD_PRIVATE_IP:24800 --reference reference --output farm
```

The second command must run after copying the reference artifacts to its host.
Raw responses and measured differences are saved; only a successful
`farm/complete.json` is an acceptance result. Starting the service is not a pass.

## Exclusive leases

The six September 19 farm profiles enable `inference.require_lease`. This is
opt-in and does not change the API contract of existing training jobs.

1. `POST /acquire_lease` with `{"owner_run":"my-run","ttl_seconds":300}`.
   Save the returned `lease_id`. The default TTL is 300 seconds; allowed values
   are integer seconds from 30 to 86,400. A conflicting claim returns 409.
2. Upload the PEFT tar using `X-Lease-ID: ID`. Optionally send
   `X-Adapter-SHA256: HEX` to verify the client-side archive identity too.
3. `GET /status` reports `lease_id`, `owner_run`, `adapter_name`,
   `adapter_sha256`, `ready_engines`, `expected_engines`, `expires_at` (Unix
   seconds), and `state`. `ready` requires every engine's health and loaded
   adapter archive hash to match. This hashes the exact tar bytes, not the
   in-memory tensors. `/v1/models` remains the standard names-only API.
4. Send `X-Lease-ID` with `/v1/completions` and `/tokenize`. A new owner cannot
   generate using its predecessor's adapter until it uploads/verifies an
   adapter for its own lease. Base-model requests are permitted under a lease.
5. Renew using `/acquire_lease` with the same `owner_run` and `lease_id` before
   expiry. Renew during long generations. An expired token cannot be renewed;
   omit it to acquire a fresh lease after the prior requests drain.
6. `POST /release_lease` with `{"lease_id":"ID"}`. Release closes admission and
   waits for in-flight requests before freeing the farm. Expiry rejects new
   requests; a successor claim also drains the old requests. HTTP disconnects
   do not remove unfinished engine work from this drain accounting.

Example lifecycle (use a reachable ingress address):

```python
import hashlib
import requests

url = "http://FARM_IP:24800"
lease = requests.post(url + "/acquire_lease", json={
    "owner_run": "experiment-42", "ttl_seconds": 3600,
}).json()
headers = {"X-Lease-ID": lease["lease_id"]}
try:
    with open("adapter.tar", "rb") as f:
        digest = hashlib.file_digest(f, "sha256").hexdigest()
        f.seek(0)
        uploaded = requests.post(url + "/skyrl/v1/upload_lora_adapter",
            params={"lora_name": "experiment-42-step-1"}, data=f,
            headers={**headers, "X-Adapter-SHA256": digest})
        uploaded.raise_for_status()
    status = requests.get(url + "/status").json()
    assert status["state"] == "ready" and status["adapter_sha256"] == digest
    response = requests.post(url + "/v1/completions", headers=headers,
        json={"model": "experiment-42-step-1", "prompt": "Hello", "max_tokens": 32})
    response.raise_for_status()
finally:
    requests.post(url + "/release_lease", json={"lease_id": lease["lease_id"]}).raise_for_status()
```

Lease state belongs to the ingress process. Restart invalidates its leases;
clients must reacquire and republish. Release does not unload the physical
adapter immediately: admission quarantines it until the next owner verifies
its own upload. A failed fanout keeps generation closed, including across
lease handoff, until a full upload succeeds. Never reuse a version name for
different bytes. The engine independently rejects an immutable-name mismatch.

Leases coordinate trusted clients; they are not an authentication boundary.
Use the ingress on port 24800: direct backend ports bypass lease admission and
must not be used by lease clients. No public authentication or firewall change
is included in this deployment.
