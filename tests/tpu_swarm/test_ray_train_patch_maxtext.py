"""The FLCE wrapper patch must cover every pinned MaxText fork (job 412)."""
import pytest

from tpu.swarm.ray_train import patch_maxtext

QWEN_WRAPPER = '''
    if self.config.attention == "vllm_rpa":
      # In vLLM, logits are computed separately after updating the KV cache.
      return hidden_state, kv_caches

    return logits
'''

GPTOSS_LINEN = '''
    if self.config.attention in ("vllm_rpa", "vllm_batched_rpa"):
      # In vLLM, logits are computed separately after updating the KV cache.
      return hidden_state, kv_caches

    return logits
'''

GPTOSS_NNX = '''
    if self.config.attention in ("vllm_rpa", "vllm_batched_rpa"):
      # In vLLM, logits are computed separately after updating the KV cache.
      if expert_indices is not None:
        return hidden_state, kv_caches, expert_indices
      return hidden_state, kv_caches

    return logits
'''

OUTPUT_HEAD = '''
  def apply_output_head(self, ...):
    logits = self.decoder.apply_output_head(...)
    return logits
'''


@pytest.mark.parametrize("body", [QWEN_WRAPPER * 2, GPTOSS_LINEN + GPTOSS_NNX, GPTOSS_LINEN])
def test_patch_inserts_hidden_state_return_in_every_wrapper(tmp_path, body):
    path = tmp_path / "models.py"
    path.write_text(OUTPUT_HEAD + body)
    assert patch_maxtext.patch(path) is True
    patched = path.read_text()
    assert patched.count(patch_maxtext.CONDITION) == body.count("return logits")
    # The output-head helper's bare `return logits` is not a wrapper tail.
    assert OUTPUT_HEAD in patched
    # Idempotent: a second run finds the contract satisfied and changes nothing.
    assert patch_maxtext.patch(path) is False
    assert path.read_text() == patched


def test_patch_fails_closed_on_unknown_layout(tmp_path):
    path = tmp_path / "models.py"
    path.write_text(OUTPUT_HEAD)
    with pytest.raises(RuntimeError, match="FLCE patch contract"):
        patch_maxtext.patch(path)
    path.write_text(GPTOSS_LINEN * 3)
    with pytest.raises(RuntimeError, match="FLCE patch contract"):
        patch_maxtext.patch(path)
