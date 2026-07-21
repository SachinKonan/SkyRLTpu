#!/usr/bin/env python3
"""Binary probe of the max trainable sequence length on the qwen35 tinker server.

Sends a synthetic importance_sampling forward_backward (+ lr=0 optim_step) of
descending lengths against a fresh throwaway LoRA model and reports which
lengths survive. Mirrors ttt-discover's datum layout exactly
(model_input = tokens[:-1], targets = tokens[1:], per-token logprobs /
advantages / mask). Advantages are all-zero so the probe cannot move weights,
and lr=0 belt-and-braces that.

Usage: TINKER_BASE_URL=http://127.0.0.1:18045 python probe_train_len.py [lengths...]
"""
import os
import sys

import torch
import tinker
from tinker import types


def main() -> None:
    base_url = os.environ["TINKER_BASE_URL"]
    lengths = [int(x) for x in sys.argv[1:]] or [28672, 24576, 20480, 16384, 12288]
    sc = tinker.ServiceClient(base_url=base_url, api_key=os.environ.get("TINKER_API_KEY", "tml-dummy"))
    tc = sc.create_lora_training_client(base_model="Qwen/Qwen3.5-27B", rank=32)
    print(f"probe model ready; lengths={lengths}", flush=True)

    tok = 872  # arbitrary mid-vocab token id
    results = {}
    for L in lengths:
        tokens = [tok] * (L + 1)
        n = L  # trained positions
        datum = tinker.Datum(
            model_input=types.ModelInput.from_ints(tokens[:-1]),
            loss_fn_inputs={
                "target_tokens": types.TensorData.from_torch(torch.tensor(tokens[1:], dtype=torch.int64)),
                "logprobs": types.TensorData.from_torch(torch.zeros(n)),
                "advantages": types.TensorData.from_torch(torch.zeros(n)),
                "mask": types.TensorData.from_torch(torch.ones(n)),
            },
        )
        try:
            fb = tc.forward_backward([datum], loss_fn="importance_sampling")
            op = tc.optim_step(types.AdamParams(learning_rate=0.0))
            fb.result()
            op.result()
            results[L] = "OK"
            print(f"L={L}: OK", flush=True)
        except Exception as e:
            results[L] = f"FAIL: {str(e)[:200]}"
            print(f"L={L}: FAIL: {str(e)[:200]}", flush=True)
    print("RESULTS:", results, flush=True)


if __name__ == "__main__":
    main()
