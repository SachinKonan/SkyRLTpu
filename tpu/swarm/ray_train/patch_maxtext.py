"""Apply the existing launcher's required FLCE return contract, failing closed."""
from pathlib import Path
import sysconfig

OLD = ('    if self.config.attention == "vllm_rpa":\n'
       '      # In vLLM, logits are computed separately after updating the KV cache.\n'
       '      return hidden_state, kv_caches\n\n'
       '    return logits')
CONDITION = '    if self.config.num_vocab_tiling > 1 and model_mode == MODEL_MODE_TRAIN:\n      return hidden_state\n\n'
NEW = OLD.removesuffix('    return logits') + CONDITION + '    return logits'


def patch(path):
    source = path.read_text()
    remaining, applied = source.count(OLD), source.count(NEW)
    # The pinned package has both Linen and NNX wrappers. Older installations
    # may already have patched either wrapper through the shell launcher.
    if remaining + applied not in (1, 2):
        raise RuntimeError("pinned MaxText no longer matches the required FLCE patch contract")
    if not remaining:
        return False
    path.write_text(source.replace(OLD, NEW))
    return True


if __name__ == "__main__":
    target = Path(sysconfig.get_path("purelib")) / "maxtext/models/models.py"
    print(f"FLCE contract verified: {target}; patched={patch(target)}", flush=True)
