#!/usr/bin/env python3
"""Max trainable sequence length probe, parameterized by model.

Generalizes runs/ttd_qwen35_server/probe_train_len.py (which hardcoded
Qwen3.5-27B) so the same methodology can size ANY model's fb graph. The qwen
number it produced -- 24576 fresh off boot, but only 12288 once same-boot
churn had eaten ~28G of HBM -- is why this version probes each length TWICE:
a length that passes cold and fails warm is not a usable ceiling.

The server pins ONE compiled shape (TUNIX_UNIFORM_SEQ_LEN), so a given boot
can only answer for its own uniform length; the caller re-boots per candidate.
Lengths below the uniform are padded up to it, which is exactly what the real
trainer does, so probing at the uniform length is the honest test.

Advantages are all-zero and lr=0, so the probe cannot move weights.

Usage:
  TINKER_BASE_URL=http://127.0.0.1:8000 PROBE_BASE_MODEL=google/gemma-4-31B-it \\
    python probe_train_len_model.py 16384
"""
import os
import sys
import time

import torch
import tinker
from tinker import types


def one_fb(tc, length: int, tag: str) -> tuple[bool, str]:
    tok = 872  # arbitrary mid-vocab token id
    tokens = [tok] * (length + 1)
    n = length
    datum = tinker.Datum(
        model_input=types.ModelInput.from_ints(tokens[:-1]),
        loss_fn_inputs={
            "target_tokens": types.TensorData.from_torch(torch.tensor(tokens[1:], dtype=torch.int64)),
            "logprobs": types.TensorData.from_torch(torch.zeros(n)),
            "advantages": types.TensorData.from_torch(torch.zeros(n)),
            # NO "mask": the production path builds the datum with a mask and
            # then STRIPS it before forward_backward (ttt_discover
            # rl/train.py:remove_mask). Sending it makes the server fail with
            # 400 "'NoneType' object has no attribute 'shape'" -- which reads
            # like a model fault but is purely a request-shape error.
        },
    )
    t0 = time.time()
    # Connection errors are NOT datapoints: the first fb at a new shape
    # compiles for many minutes, and a tunnel blip mid-poll surfaces as
    # tinker.APIConnectionError -- candidate 16384:65536 was mis-recorded as
    # empty exactly this way. Retry those; everything else is a verdict.
    for attempt in range(4):
        try:
            fb = tc.forward_backward([datum], loss_fn="importance_sampling")
            op = tc.optim_step(types.AdamParams(learning_rate=0.0))
            fb.result()
            op.result()
            return True, f"OK ({time.time() - t0:.0f}s)"
        except Exception as e:  # noqa: BLE001
            name = type(e).__name__
            if "Connection" in name or "Connection error" in str(e):
                print(f"  [retry {attempt+1}/4] {name}: connection blip; waiting 60s", flush=True)
                time.sleep(60)
                continue
            return False, f"FAIL: {name}: {str(e)[:300]}"
    return False, f"FAIL: connection errors through all retries ({time.time()-t0:.0f}s)"


def main() -> None:
    base_url = os.environ["TINKER_BASE_URL"]
    model = os.environ.get("PROBE_BASE_MODEL", "Qwen/Qwen3.5-27B")
    rank = int(os.environ.get("PROBE_LORA_RANK", "32"))
    lengths = [int(x) for x in sys.argv[1:]] or [16384]
    sc = tinker.ServiceClient(base_url=base_url, api_key=os.environ.get("TINKER_API_KEY", "tml-dummy"))
    tc = sc.create_lora_training_client(base_model=model, rank=rank)

    # create_model is ASYNC server-side (api.py: it enqueues a future and
    # returns), so the client hands back a handle before the LoRA state
    # exists. An immediate forward_backward then finds it None and the server
    # answers 400 "'NoneType' object has no attribute 'shape'" -- which looks
    # like a model/memory fault and is really a race. Production never trips
    # it because it samples before training; the qwen probe never tripped it
    # because it ran against a server a live job had already warmed.
    for attempt in range(60):
        try:
            tc.get_info()
            print(f"model materialized after {attempt * 5}s", flush=True)
            break
        except Exception as e:  # noqa: BLE001 -- still initializing
            if attempt == 0:
                print(f"waiting for model init ({type(e).__name__})", flush=True)
            time.sleep(5)
    time.sleep(10)  # settle after the handle appears
    print(f"probe ready: model={model} rank={rank} lengths={lengths}", flush=True)

    results = {}
    for L in lengths:
        # PASS 1 -- cold: the number a naive probe reports.
        ok1, why1 = one_fb(tc, L, "cold")
        print(f"L={L} cold : {why1}", flush=True)
        # PASS 2 -- warm: same length again with the fb program (and whatever
        # else the boot loaded) already resident. This is the pass that
        # demoted qwen from 24576 to 12288; a ceiling is only real if it
        # survives here.
        ok2, why2 = (False, "skipped (cold failed)")
        if ok1:
            ok2, why2 = one_fb(tc, L, "warm")
            print(f"L={L} warm : {why2}", flush=True)
        results[L] = {"cold": why1, "warm": why2, "usable": bool(ok1 and ok2)}
    print("PROBE-RESULTS-JSON:", __import__("json").dumps(results), flush=True)


if __name__ == "__main__":
    main()
