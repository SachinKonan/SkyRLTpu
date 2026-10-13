"""Opt-in, single-host packing comparison; never applies an optimizer update."""
import gc
import json
import os
import time
from pathlib import Path

import jax
import numpy as np

from skyrl.backends.renderer import render_model_input


def compare_gradients(expected, actual):
    error2 = expected2 = actual2 = dot = 0.0
    for a, b in zip(expected, actual, strict=True):
        a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
        if a.shape != b.shape or not (np.isfinite(a).all() and np.isfinite(b).all()):
            raise RuntimeError("Packing probe gradients have incompatible shapes or nonfinite values")
        error2 += float(np.sum((a - b) ** 2))
        expected2 += float(np.sum(a ** 2))
        actual2 += float(np.sum(b ** 2))
        dot += float(np.sum(a * b))
    return dict(relative_error=float(np.sqrt(error2 / max(expected2, 1e-30))),
                baseline_norm=float(np.sqrt(expected2)), packed_norm=float(np.sqrt(actual2)),
                cosine=float(dot / max(np.sqrt(actual2 * expected2), 1e-30)))


def verify_once(backend, batch):
    """Replay four real examples at both budgets, restore accumulation, then proceed.

    The report is an audit artifact, not a cache: each new backend process must
    verify again. Failure restores the old budget for this process. No LoRA or
    optimizer updates occur during the diagnostic passes.
    """
    report_path = os.environ.get("TUNIX_PACKING_PROBE_REPORT")
    if not report_path or getattr(backend, "_packing_probe_done", False):
        return
    if getattr(backend, "_packing_probe_running", False):
        return
    if jax.process_count() != 1:
        raise RuntimeError("Packing probe is restricted to a single-process trainer")
    old_budget = int(os.environ["TUNIX_PACKING_PROBE_BASE_BUDGET"])
    new_budget = getattr(backend, "_packing_probe_target_budget", backend.config.train_token_budget)
    backend._packing_probe_target_budget = new_budget
    if old_budget <= 0 or new_budget <= old_budget:
        raise ValueError("Packing probe requires an increased positive token budget")
    inputs = render_model_input(batch.all_model_inputs)
    buckets = sorted(int(v) for v in os.environ.get("TUNIX_SEQ_BUCKETS", "").split(",") if v.strip())
    uniform = int(os.environ.get("TUNIX_UNIFORM_SEQ_LEN", "0"))
    groups = {}
    for i, item in enumerate(inputs):
        model_id = batch.all_model_ids[i]
        slot = backend.models[model_id]
        if slot.accum_count or slot.accum_grads is not None:
            continue  # Probe only at the start of an accumulation window.
        kind = backend.templates[slot.template_key].kind
        length = len(item.prompt_ids)
        padded = uniform or next((b for b in buckets if b >= length), length)
        key = (model_id, backend._round_seq_len(padded, kind))
        group = groups.setdefault(key, [])
        group.append(i)
        if len(group) == 4:
            break
    else:
        # A short first request may not contain four examples. Keep the proven
        # budget until a later, empty accumulation window can run the check.
        backend.config.train_token_budget = old_budget
        return
    model_id, padded = key
    slot = backend.models[model_id]
    if kind != "maxtext" or backend._row_shard() != 4:
        raise RuntimeError("Packing probe expects the reviewed FSDP4 MaxText layout")
    selected = {name: [getattr(batch, name)[i] for i in group]
                for name in type(batch).model_fields if name.startswith("all_")}
    selected["request_batch_slices"] = [("packing_probe", model_id, 0, 4)]
    fixture = batch.model_copy(update=selected)
    saved = slot.accum_grads, slot.accum_count
    report = dict(started=time.time(), model_id=model_id, examples=4, padded_length=padded,
                  baseline_budget=old_budget, packed_budget=new_budget, state="running", passes=[])
    path = Path(report_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def write():
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        temporary.replace(path)

    write()
    backend._packing_probe_running = True
    try:
        gradients = {}
        for name, budget in (("baseline", old_budget), ("packed", new_budget)):
            backend.config.train_token_budget = budget
            # Warm and measured passes use identical data and unchanged weights.
            for repeat in range(2):
                slot.accum_grads, slot.accum_count = None, 0
                gc.collect()
                start = time.monotonic()
                backend._model_pass(fixture, with_grads=True)
                leaves = [np.array(x, copy=True) for x in jax.device_get(jax.tree.leaves(slot.accum_grads))]
                if slot.accum_count != 4:
                    raise RuntimeError("Packing probe changed the real-example accumulation count")
                report["passes"].append(dict(mode=name, repeat=repeat, seconds=time.monotonic()-start,
                                             accum_count=slot.accum_count))
                if repeat == 1:
                    gradients[name] = leaves
                write()
        comparison = compare_gradients(gradients["baseline"], gradients["packed"])
        report.update(comparison)
        if comparison["baseline_norm"] == 0 or comparison["relative_error"] > 0.02 or comparison["cosine"] < 0.999:
            raise RuntimeError("Packing probe gradient comparison failed")
        report["state"] = "passed"
    except Exception as exc:
        report.update(state="failed", error=type(exc).__name__ + ": " + str(exc))
        raise
    finally:
        slot.accum_grads, slot.accum_count = saved
        backend.config.train_token_budget = new_budget if report["state"] == "passed" else old_budget
        backend._packing_probe_running = False
        backend._packing_probe_done = True
        report["finished"] = time.time()
        write()
