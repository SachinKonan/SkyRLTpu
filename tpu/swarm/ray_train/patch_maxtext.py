"""Apply the existing launcher's required FLCE return contract, failing closed.

FLCE (fused linear cross-entropy, num_vocab_tiling > 1) makes the decoder skip
the output head in training and return logits=None; the Transformer wrappers
must then surface the hidden state instead of that None. Every pinned MaxText
fork ends each wrapper's forward with the same two statements, whatever the
attention condition above them looks like:

    ...
      return hidden_state, kv_caches

    return logits

The Qwen/Muse pins spell the guard `attention == "vllm_rpa"`; the gpt-oss pin
(d388c547) spells it `attention in ("vllm_rpa", "vllm_batched_rpa")` and its
NNX wrapper adds an expert_indices return above the shared line. Anchoring on
the shared tail patches all of them and still fails closed on anything else.
"""
from pathlib import Path
import re
import sysconfig

TAIL = "      return hidden_state, kv_caches\n\n"
RETURN = "    return logits"
CONDITION = "    if self.config.num_vocab_tiling > 1 and model_mode == MODEL_MODE_TRAIN:\n      return hidden_state\n\n"
UNPATCHED = re.compile(re.escape(TAIL) + re.escape(RETURN))
PATCHED = re.compile(re.escape(TAIL) + re.escape(CONDITION) + re.escape(RETURN))


def patch(path):
    source = path.read_text()
    remaining, applied = len(UNPATCHED.findall(source)), len(PATCHED.findall(source))
    # The pinned package has both Linen and NNX wrappers. Older installations
    # may already have patched either wrapper through the shell launcher.
    if remaining + applied not in (1, 2):
        raise RuntimeError("pinned MaxText no longer matches the required FLCE patch contract")
    if not remaining:
        return False
    path.write_text(UNPATCHED.sub(lambda m: TAIL + CONDITION + RETURN, source))
    return True


if __name__ == "__main__":
    target = Path(sysconfig.get_path("purelib")) / "maxtext/models/models.py"
    print(f"FLCE contract verified: {target}; patched={patch(target)}", flush=True)
