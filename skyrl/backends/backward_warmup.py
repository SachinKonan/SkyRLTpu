"""Warm production backward shapes without an optimizer update or fake example.

The caller must serialize this with training, as the normal backend does. Old
accumulators are never passed to a donating JIT. Failures propagate to startup.
"""
from contextlib import contextmanager
import os


from tpu.swarm.ray_train.warmup_contract import shapes

@contextmanager
def isolated_accumulator(slot):
    saved = slot.accum_grads, slot.accum_count
    slot.accum_grads, slot.accum_count = None, 0
    try:
        yield
    finally:
        slot.accum_grads, slot.accum_count = saved


def run(backend, model_id):
    import jax
    from skyrl.tinker import types

    slot = backend.models[model_id]
    if slot.mix is not None:
        raise ValueError("backward warmup is not validated for carried/fresh LoRA mixing")
    uniform = int(os.environ.get("TUNIX_UNIFORM_SEQ_LEN", "0"))
    buckets = [int(n) for n in os.environ.get("TUNIX_SEQ_BUCKETS", "").split(",") if n.strip()]
    maximum = int(os.environ["TUNIX_WARMUP_MAX_LENGTH"])
    cases = shapes(maximum, backend.config.train_token_budget, backend._row_shard(),
                   uniform=uniform, buckets=buckets)
    for rows, length in cases:
        batch = types.PreparedModelPassBatch(
            all_model_inputs=[types.ModelInput(chunks=[types.EncodedTextChunk(tokens=[2] * length)]) for _ in range(rows)],
            all_targets=[[2] * length for _ in range(rows)],
            all_token_weights=[[0.] * length for _ in range(rows)],
            all_sampling_logprobs=[[0.] * length for _ in range(rows)],
            all_advantages=[[0.] * length for _ in range(rows)],
            all_values=[[] for _ in range(rows)], all_returns=[[] for _ in range(rows)],
            all_model_ids=[model_id] * rows,
            all_loss_fns=[os.environ.get("TUNIX_WARMUP_LOSS_FN", "importance_sampling")] * rows,
            all_loss_fn_configs=[None] * rows,
            request_batch_slices=[("backward-warmup", model_id, 0, rows)])
        with isolated_accumulator(slot):
            # create_model is already executing on EVERY rank. Calling the
            # distributed forward_backward wrapper here would broadcast a
            # nested RPC on the leader while workers enter the model program.
            # _model_pass is the local implementation on both backend classes.
            results = backend._model_pass(batch, with_grads=True)
            if any(isinstance(result, types.ErrorResponse) for result in results.values()):
                raise RuntimeError(f"backward warmup failed: {results}")
            # Force asynchronous device errors to surface before clearing scratch
            # accumulators and allowing the client to start expensive rollouts.
            jax.block_until_ready(slot.accum_grads)
        print(f"BACKWARD_WARMUP_COMPLETE rows={rows} length={length} optimizer_unchanged=true", flush=True)
