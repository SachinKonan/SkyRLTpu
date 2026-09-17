"""Secret-free effective launch records for comparing legacy and Ray processes."""
import json
from pathlib import Path
import re

from .environment import MODEL_KEYS, MODEL_PREFIXES
from .runtime_inventory import clean_url

SECRET = re.compile(r"API_?KEY|PASSWORD|SECRET|CREDENTIAL|ACCESS_TOKEN|AUTH_TOKEN|"
                    r"AUTHORIZATION|PRIVATE_KEY|(?:^|_)TOKEN$", re.I)
RUNTIME_PREFIXES = ("SCIENCE_", "ARENA_", "WANDB_")
RUNTIME_KEYS = {"HF_HOME", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE",
                "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "PYTHONPATH", "PATH", "LD_LIBRARY_PATH",
                "TINKER_BASE_URL", "EXPERIMENT_NAME", "NUM_CPUS_PER_TASK", "GROUPS_PER_BATCH",
                "GROUP_SIZE", "NUM_EPOCHS", "LEARNING_RATE", "LORA_RANK", "KL_PENALTY_COEF",
                "TEMPERATURE", "EVAL_TIMEOUT", "SAVE_EVERY"}


def _value(value):
    if isinstance(value, dict):
        return {k: "<redacted>" if SECRET.search(k.replace('-', '_')) else _value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_value(v) for v in value]
    if isinstance(value, str) and '://' in value:
        return clean_url(value)
    return value


def _command(command):
    result, hide_next = [], False
    for arg in command:
        if hide_next:
            result.append('<redacted>'); hide_next = False
        elif arg.startswith('--') and SECRET.search(arg.split('=', 1)[0].replace('-', '_')):
            result.append(arg.split('=', 1)[0] + ('=<redacted>' if '=' in arg else ''))
            hide_next = '=' not in arg
        elif arg.startswith('--') and '=' in arg:
            key, value = arg.split('=', 1)
            try:
                parsed = json.loads(value)
            except (ValueError, TypeError):
                result.append(key + '=' + _value(value))
            else:
                result.append(key + '=' + json.dumps(_value(parsed)))
        else:
            try:
                value = json.loads(arg)
            except (ValueError, TypeError):
                result.append(_value(arg))
            else:
                result.append(json.dumps(_value(value), sort_keys=True) if isinstance(value, (dict, list)) else arg)
    return result


def write_launch_contract(path, command, environment, *, extra_keys=()):
    # Capture role settings and known system inputs, never an unrestricted
    # inherited environment dump (unknown keys can contain credentials).
    keys = MODEL_KEYS | RUNTIME_KEYS | set(extra_keys)
    settings = {k: _value(v) for k, v in environment.items()
                if (k.startswith(MODEL_PREFIXES + RUNTIME_PREFIXES) or k in keys)
                and not SECRET.search(k)}
    path = Path(path)
    record = {"schema": 1, "command": _command(command), "environment": settings}
    temporary = path.with_suffix(".partial")
    temporary.write_text(json.dumps(record, sort_keys=True, indent=2) + "\n")
    temporary.replace(path)
