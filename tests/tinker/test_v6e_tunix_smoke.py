import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from skyrl.backends.tunix_backend import TunixBackend
from tpu.v6e_tunix_smoke import (
    load_gradient_diagnostics,
    logprob_vector,
    make_datum,
    sparse_expert_gradient_summary,
    vector_hash,
)

_WORKER_SCRIPT = Path(__file__).parents[2] / "tpu" / "jobman" / "v6e_tunix_smoke_worker.sh"
_COLOCATED_LAUNCHER = Path(__file__).parents[2] / "tpu" / "start_colocated_vllm_tinker.sh"
_GEMMA_TP4_CONFIG = Path(__file__).parents[2] / "tpu" / "jobman" / "configs" / "v6e_gemma_tp4_fsdp4_smoke.yaml"
_GEMMA_TP4_ASIA_CONFIG = (
    Path(__file__).parents[2] / "tpu" / "jobman" / "configs" / "v6e_gemma_tp4_fsdp4_smoke_asia_ne1.yaml"
)
_GPTOSS_SPARSE_CONFIG = (
    Path(__file__).parents[2]
    / "tpu"
    / "jobman"
    / "configs"
    / "v6e_gptoss20b_sparse_lora_tp8_fsdp2_smoke.yaml"
)
_GPTOSS_SPARSE_V6E8_CONFIG = (
    Path(__file__).parents[2]
    / "tpu"
    / "jobman"
    / "configs"
    / "v6e_gptoss20b_sparse_lora_tp8_smoke.yaml"
)


def _output(*rows):
    return SimpleNamespace(
        loss_fn_outputs=[
            {"logprobs": SimpleNamespace(data=list(row))}
            for row in rows
        ]
    )


def test_logprob_vector_preserves_row_order_and_hashes_bytes():
    values = logprob_vector(_output([1.0, 2.0], [3.5]))

    np.testing.assert_array_equal(values, np.asarray([1.0, 2.0, 3.5], dtype=np.float32))
    assert vector_hash(values) == vector_hash(values.copy())
    assert vector_hash(values) != vector_hash(values[::-1])


def test_logprob_vector_rejects_missing_or_empty_outputs():
    with pytest.raises(RuntimeError, match="no logprobs"):
        logprob_vector(SimpleNamespace(loss_fn_outputs=[{}]))
    with pytest.raises(RuntimeError, match="empty"):
        logprob_vector(SimpleNamespace(loss_fn_outputs=[]))
    with pytest.raises(RuntimeError, match="empty"):
        logprob_vector(_output([]))


def test_dense_datum_has_exactly_the_requested_number_of_valid_tokens():
    class Tokenizer:
        eos_token_id = 1

        @staticmethod
        def encode(text, add_special_tokens):
            return [2, 3, 4] if add_special_tokens else [5, 6]

    datum = make_datum(Tokenizer(), row=0, sequence_length=17)

    assert len(datum.model_input.to_ints()) == 17
    assert len(datum.loss_fn_inputs["target_tokens"].data) == 17
    assert len(datum.loss_fn_inputs["weights"].data) == 17
    assert sum(datum.loss_fn_inputs["weights"].data) == 15


def test_sparse_expert_gradient_gate_observes_zero_b_initialization(tmp_path):
    def record(index, norms):
        return {
            "kind": "mean_gradient",
            "index": index,
            "leaves": {f"['GptOssMlp']['{name}']['value']": {"norm": norm} for name, norm in norms.items()},
        }

    initial = {name: 0.0 for name in (
        "wi_0_lora_a",
        "wi_1_lora_a",
        "wo_lora_a",
    )}
    initial.update({name: 1.0 for name in (
        "wi_0_lora_b",
        "wi_1_lora_b",
        "wo_lora_b",
    )})
    second = {name: 2.0 for name in (
        "wi_0_lora_a",
        "wi_0_lora_b",
        "wi_1_lora_a",
        "wi_1_lora_b",
        "wo_lora_a",
        "wo_lora_b",
    )}
    path = tmp_path / "gradients.jsonl"
    path.write_text("\n".join(json.dumps(value) for value in (record(1, second), record(0, initial))) + "\n")

    records = load_gradient_diagnostics(path)
    summary = sparse_expert_gradient_summary(records)

    assert [value["index"] for value in records] == [0, 1]
    assert summary["pass"] is True
    assert summary["initial"]["wi_0_lora_a"] == 0.0
    assert summary["after_first_update"]["wi_0_lora_a"] == 2.0


def test_sparse_expert_gradient_gate_rejects_missing_second_record():
    summary = sparse_expert_gradient_summary([])

    assert summary["pass"] is False
    assert "at least two" in summary["failures"][0]


def test_gradient_diagnostic_records_per_leaf_hashes(tmp_path, monkeypatch):
    output = tmp_path / "gradients.jsonl"
    monkeypatch.setenv("TUNIX_REPLAY_DIAGNOSTICS", "1")
    monkeypatch.setenv("TUNIX_REPLAY_DIAGNOSTICS_PATH", str(output))
    backend = object.__new__(TunixBackend)
    backend.models = {"model": SimpleNamespace(diagnostic_grad_index=0, lora_state={})}

    backend._record_gradient_diagnostics(
        "model",
        {"attention": np.asarray([3.0, 4.0], dtype=np.float32)},
        5.0,
    )

    record = json.loads(output.read_text())
    assert record["index"] == 0
    assert record["global_norm"] == 5.0
    assert record["leaves"]["['attention']"]["norm"] == 5.0
    assert len(record["leaves"]["['attention']"]["sha256"]) == 64
    assert record["model_state_layouts"] == {}
    assert record["gradient_layouts"]["['attention']"]["sharding"] is None
    assert record["gradient_layouts"]["['attention']"]["committed"] is False


def test_host_global_norm_uses_gathered_gradient_values():
    flat_grads = {
        "a": np.asarray([3.0, 4.0], dtype=np.float32),
        "b": np.asarray([0.0, 12.0], dtype=np.float32),
    }

    assert TunixBackend._host_global_norm(flat_grads) == pytest.approx(13.0)


def test_gradient_diagnostic_reuses_pre_gathered_gradients(tmp_path, monkeypatch):
    output = tmp_path / "gradients.jsonl"
    monkeypatch.setenv("TUNIX_REPLAY_DIAGNOSTICS", "1")
    monkeypatch.setenv("TUNIX_REPLAY_DIAGNOSTICS_PATH", str(output))
    backend = object.__new__(TunixBackend)
    backend.models = {"model": SimpleNamespace(diagnostic_grad_index=0, lora_state={})}

    def fail_if_gathered_again(_state):
        raise AssertionError("gradient arrays were gathered twice")

    monkeypatch.setattr(backend, "_flat_numpy", fail_if_gathered_again)
    flat_grads = {"['attention']": np.asarray([3.0, 4.0], dtype=np.float32)}
    backend._record_gradient_diagnostics(
        "model",
        {"attention": np.asarray([3.0, 4.0], dtype=np.float32)},
        5.0,
        flat_grads=flat_grads,
    )

    record = json.loads(output.read_text())
    assert record["leaves"]["['attention']"]["norm"] == 5.0


def test_worker_uses_rank_32_and_dense_uniform_sequences_by_default():
    script = _WORKER_SCRIPT.read_text()

    assert 'LORA_RANK="${TUNIX_LORA_RANK:-32}"' in script
    assert 'TRAIN_TP_SIZE="${TRAIN_TP_SIZE:-8}"' in script
    assert 'TRAIN_FSDP_SIZE="${TRAIN_FSDP_SIZE:-2}"' in script
    assert 'TRAIN_WORKERS="$TRAIN_WORKERS_SPEC"' in script
    assert 'TP_SIZE="$TRAIN_TP_SIZE"' in script
    assert 'FSDP_SIZE="$TRAIN_FSDP_SIZE"' in script
    assert '--rank "$LORA_RANK"' in script
    assert '--rows "$SMOKE_ROWS"' in script
    assert '--replays "$SMOKE_REPLAYS"' in script
    assert '--extra-updates "$SMOKE_EXTRA_UPDATES"' in script
    assert '--sequence-length "$UNIFORM"' in script
    assert 'TUNIX_REPLAY_DIAGNOSTICS="${TUNIX_REPLAY_DIAGNOSTICS:-0}"' in script
    assert 'TUNIX_MINIMAL_FB_OUTPUT="${TUNIX_MINIMAL_FB_OUTPUT:-0}"' in script
    assert "TUNIX_SMOKE_DIAGNOSTIC_ACCEPT_FAILURE" in script
    assert "TUNIX_REQUIRE_SPARSE_EXPERT_GRADIENTS" in script


def test_worker_has_pinned_native_sparse_gptoss_profile():
    script = _WORKER_SCRIPT.read_text()
    profile = script.split("gpt-oss-20b)", 1)[1].split(";;", 1)[0]

    assert "b77f9f358a1dd9b223fcc16792b7d5c2530d7044" in profile
    assert '"sparse_matmul":true' in profile
    assert '"megablox":true' in profile


def test_gptoss_sparse_smoke_uses_two_updates_and_unique_durable_state():
    config = yaml.safe_load(_GPTOSS_SPARSE_CONFIG.read_text())
    spec = config["resumable"]["run_spec"]
    env = config["resumable"]["env"]

    assert config["tpu"]["accelerator"] == "v6e-16"
    assert spec["topology"] == "tp8-fsdp2"
    assert spec["sparse_matmul"] is True
    assert spec["megablox"] is True
    assert env["MODEL_NAME"] == "openai/gpt-oss-20b"
    assert env["TRAIN_TP_SIZE"] == "8"
    assert env["TRAIN_FSDP_SIZE"] == "2"
    assert env["TUNIX_SMOKE_EXTRA_UPDATES"] == "1"
    assert env["TUNIX_REQUIRE_SPARSE_EXPERT_GRADIENTS"] == "1"
    assert "b77f9f358a1dd9b223fcc16792b7d5c2530d7044" in env["TUNIX_MAXTEXT_PIP_SPEC"]
    assert json.loads(env["TUNIX_MAXTEXT_KWARGS"])["sparse_matmul"] is True
    assert config["tpu"]["zone"] == "us-south1-a"
    assert "asia-northeast1" in env["TUNIX_MAXTEXT_CKPT_CACHE_GCS"]
    assert "gptoss20b-sparse-lora" in env["SMOKE_RESULT_GCS"]
    assert "gptoss20b-sparse-lora-" in env["SKYRLTPU_BUNDLE_URL"]


def test_gptoss_sparse_v6e8_race_uses_all_eight_local_chips():
    config = yaml.safe_load(_GPTOSS_SPARSE_V6E8_CONFIG.read_text())
    env = config["resumable"]["env"]

    assert config["tpu"]["accelerator"] == "v6e-8"
    assert config["tpu"]["zone"] == "asia-northeast1-b"
    assert env["TRAIN_WORKERS"] == "0"
    assert env["TRAIN_TP_SIZE"] == "8"
    assert env["TRAIN_FSDP_SIZE"] == "1"
    assert env["TRAIN_TPU_PROCESS_BOUNDS"] == "1,1,1"
    assert env["TRAIN_TPU_CHIPS_PER_PROCESS_BOUNDS"] == "2,2,2"
    assert env["TUNIX_SMOKE_EXTRA_UPDATES"] == "1"
    assert env["TUNIX_REQUIRE_SPARSE_EXPERT_GRADIENTS"] == "1"


def test_replay_diagnostics_reach_every_multihost_controller():
    script = _COLOCATED_LAUNCHER.read_text()

    # One export is in process 0's API script and one is in the worker-script
    # template used by every nonzero JAX process. All controllers must take the
    # same diagnostic branch or they submit different TPU programs.
    assert script.count(
        'export TUNIX_REPLAY_DIAGNOSTICS="${TUNIX_REPLAY_DIAGNOSTICS}"'
    ) == 2


def test_qwen_tp8_profile_pads_four_kv_heads_via_maxtext_alignment():
    script = _WORKER_SCRIPT.read_text()

    assert '"override_model_config":true' in script
    assert '"base_num_kv_heads":8' in script
    assert '"attention":"autoselected"' in script
    assert '"use_tokamax_splash":true' in script
    assert "[h0,h0,h1,h1,h2,h2,h3,h3]" in script


def test_gemma_tp8_profile_pads_only_four_global_kv_heads():
    script = _WORKER_SCRIPT.read_text()

    gemma_profile = script.split("gemma4-31b)", 1)[1].split(";;", 1)[0]
    assert '"override_model_config":true' in gemma_profile
    assert '"global_num_kv_heads":8' in gemma_profile
    assert '"attention":"autoselected"' in gemma_profile
    assert '"use_tokamax_splash":true' in gemma_profile
    assert '"base_num_kv_heads"' not in gemma_profile
    assert 'if [ "$TRAIN_TP_SIZE" -eq 4 ]; then' in gemma_profile
    native_tp4_profile = gemma_profile.split('if [ "$TRAIN_TP_SIZE" -eq 4 ]; then', 1)[1].split("else", 1)[0]
    assert '"global_num_kv_heads"' not in native_tp4_profile
    assert '"override_model_config"' not in native_tp4_profile


def test_muse_tp8_profile_pads_two_kv_heads_and_uses_pinned_fork():
    script = _WORKER_SCRIPT.read_text()

    muse_profile = script.split("muse-glimmer-30b)", 1)[1].split(";;", 1)[0]
    assert "4f65ba50963bc975e7ad90ebaa1e752d8a9d8c82" in muse_profile
    assert '"override_model_config":true' in muse_profile
    assert '"base_num_kv_heads":8' in muse_profile
    assert '"attention":"autoselected"' in muse_profile
    assert '"use_tokamax_splash":true' in muse_profile
    assert "[h0,h0,h0,h0,h1,h1,h1,h1]" in muse_profile


@pytest.mark.parametrize(
    ("config_path", "expected_rows"),
    [(_GEMMA_TP4_CONFIG, 1), (_GEMMA_TP4_ASIA_CONFIG, 4)],
)
def test_gemma_long_context_diagnostics_use_tp8_and_three_replays(config_path, expected_rows):
    config = yaml.safe_load(config_path.read_text())
    spec = config["resumable"]["run_spec"]
    env = config["resumable"]["env"]

    assert spec["topology"] == "tp8-fsdp2"
    assert spec["global_batch"] == expected_rows
    assert spec["uniform_sequence_length"] == 22528
    assert spec["lora_rank"] == 32
    assert spec["dense_sequence"] is True
    assert env["TRAIN_TP_SIZE"] == "8"
    assert env["TRAIN_FSDP_SIZE"] == "2"
    assert env["TUNIX_ROW_SHARD"] == "2"
    assert env["TUNIX_SMOKE_ROWS"] == str(expected_rows)
    assert env["TUNIX_SMOKE_REPLAYS"] == "3"
    assert env["TUNIX_REPLAY_DIAGNOSTICS"] == "1"
    assert env["TUNIX_SMOKE_DIAGNOSTIC_ACCEPT_FAILURE"] == "1"
    assert "grad-diag" in env["SMOKE_RESULT_GCS"]
