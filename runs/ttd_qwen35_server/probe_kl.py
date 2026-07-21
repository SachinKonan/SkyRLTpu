#!/usr/bin/env python3
"""Validate trainer-chip compute_logprobs + untouched external sampling.

1. compute_logprobs on a real-tokenized ~L-token prompt against the BASE model:
   expect len == prompt length, leading 0.0, finite negative values.
2. A plain 16-token sample (external vLLM path): expect tokens back.

Usage: TINKER_BASE_URL=http://127.0.0.1:18045 python probe_kl.py [L]
"""
import os
import sys

import tinker
from tinker import types


def main() -> None:
    L = int(sys.argv[1]) if len(sys.argv) > 1 else 8192
    sc = tinker.ServiceClient(
        base_url=os.environ["TINKER_BASE_URL"], api_key=os.environ.get("TINKER_API_KEY", "tml-dummy")
    )
    samp = sc.create_sampling_client(base_model="Qwen/Qwen3.5-27B")

    tokens = [872, 3837, 15235, 1467] * (L // 4)
    lps = samp.compute_logprobs(types.ModelInput.from_ints(tokens)).result()
    finite = [x for x in lps[1:] if x is not None]
    print(f"compute_logprobs: len={len(lps)} (prompt {len(tokens)}), head={lps[:4]}")
    print(f"  finite={len(finite)}/{len(lps)-1}, mean={sum(finite)/len(finite):.4f}, "
          f"min={min(finite):.4f}, max={max(finite):.4f}")
    assert len(lps) == len(tokens), "length mismatch"
    assert all(x <= 0.0 for x in finite), "positive logprobs?"

    res = samp.sample(
        prompt=types.ModelInput.from_ints(tokens[:32]),
        num_samples=1,
        sampling_params=types.SamplingParams(max_tokens=16, temperature=1.0),
    ).result()
    print(f"plain sample: {len(res.sequences[0].tokens)} tokens, stop={res.sequences[0].stop_reason}")
    print("KL PROBE OK")


if __name__ == "__main__":
    main()
