"""Keep a launcher's ambient model settings out of workload processes.

Authentication, system paths and the immutable bundle's Python import path are
preserved. Model/compiler/debug settings must come from the resolved profile.
This helper is also usable by a legacy launcher constructing a child env.
"""
import os

MODEL_PREFIXES = ("TUNIX_", "SKYRL_", "TTD_", "JAX_", "XLA_", "TPU_",
                  "CLOUD_TPU_", "VLLM_", "LIBTPU_", "PJRT_")
MODEL_KEYS = {"RAY_ADDRESS", "RAY_NAMESPACE", "CUSTOM_NUM_TOKENS_BUCKETS",
              "SERIALIZE_MODEL_AND_SAMPLING", "SKIP_JAX_PRECOMPILE",
              "USE_BATCHED_RPA_KERNEL", "USE_JAX_RAGGED_CONV1D",
              "MODEL_IMPL_TYPE"}


def workload_environment(parent=None):
    parent = os.environ if parent is None else parent
    return {k: v for k, v in parent.items()
            if not k.startswith(MODEL_PREFIXES) and k not in MODEL_KEYS}
