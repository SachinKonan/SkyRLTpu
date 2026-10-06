"""Host-side diagnostics for a frozen-parameter gradient experiment."""

import hashlib
import json
from pathlib import Path

import numpy as np


def fingerprint(arrays):
    digest = hashlib.sha256()
    for key in sorted(arrays):
        value = np.ascontiguousarray(arrays[key])
        digest.update(key.encode())
        digest.update(str((value.shape, str(value.dtype))).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def dot(left, right):
    if left.keys() != right.keys():
        raise ValueError("Gradient leaves differ")
    total = 0.0
    for key in left:
        if left[key].shape != right[key].shape:
            raise ValueError(f"Gradient shape differs: {key}")
        a = np.asarray(left[key], dtype=np.float64).reshape(-1)
        b = np.asarray(right[key], dtype=np.float64).reshape(-1)
        total += float(np.dot(a, b))
    if not np.isfinite(total):
        raise ValueError("Non-finite gradient dot product")
    return total


def compare(left, right, n_left, n_right):
    aa, bb, ab = dot(left, left), dot(right, right), dot(left, right)
    a, b = np.sqrt(aa), np.sqrt(bb)
    combined = np.sqrt(max(0.0, aa + bb + 2 * ab))
    return {
        "left_count": n_left, "right_count": n_right,
        "left_sum_norm": float(a), "right_sum_norm": float(b),
        "left_mean_norm": float(a / n_left) if n_left else None,
        "right_mean_norm": float(b / n_right) if n_right else None,
        "dot": ab, "cosine": float(ab / (a * b)) if a and b else None,
        "right_to_left_norm": float(b / a) if a else None,
        "combined_norm": float(combined),
        "cancellation_ratio": float(combined / (a + b)) if a + b else None,
        "combined_dot_left": aa + ab, "combined_dot_right": bb + ab,
    }


def export_probe(directory, model_id, index, gradients, parameters, count):
    if count <= 0:
        raise ValueError("Cannot export an empty accumulated gradient")
    norm = np.sqrt(dot(gradients, gradients))
    target = Path(directory) / model_id
    target.mkdir(parents=True, exist_ok=True)
    stem = target / f"{index:06d}"
    record = {
        "index": index, "count": count, "sum_norm": float(norm),
        "mean_norm": float(norm / count),
        "parameter_sha256": fingerprint(parameters),
        "gradient_sha256": fingerprint(gradients),
    }
    # The JSON is the completion marker, written after the arrays.
    np.savez(str(stem) + ".npz", **gradients)
    Path(str(stem) + ".json").write_text(json.dumps(record) + "\n")
    return record
