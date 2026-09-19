# Multi-LoRA inference farm

`profiles/qwen35-v432-native-multi-lora-inference-20260919.json` runs the existing
Ray v2 executor in inference-only mode. Ranks 0, 1, 2 and 3 each run one TP4
engine; no TPU is reserved for training. Ray Serve ingress listens on port
24800 and engines on 24801. The profile uses systemd ownership, native thinking,
two adapter slots, and a private compilation-cache destination seeded from the
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
