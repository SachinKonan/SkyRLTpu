import json
import tarfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from cloudpathlib import AnyPath

from skyrl.backends.vllm_sampling import VllmSamplingClient
from skyrl.tinker import types


class _VllmHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers.get("content-length", "0"))
        payload = json.loads(self.rfile.read(length))
        self.server.requests.append((self.path, payload, dict(self.headers)))

        if self.path == "/v1/load_lora_adapter":
            if getattr(self.server, "fail_loads", 0) > 0:
                self.server.fail_loads -= 1
                self.send_response(503)
                self.send_header("content-type", "text/plain")
                self.end_headers()
                self.wfile.write(b"try again")
                return
            self.send_response(200)
            self.send_header("content-type", "text/plain")
            self.end_headers()
            self.wfile.write(b"ok")
            return

        if self.path == "/v1/unload_lora_adapter":
            self.send_response(200)
            self.send_header("content-type", "text/plain")
            self.end_headers()
            self.wfile.write(b"ok")
            return

        if self.path == "/v1/completions":
            response = {
                "choices": [
                    {
                        "token_ids": [101, 102],
                        "finish_reason": "length",
                        "logprobs": {"token_logprobs": [-0.1, -0.2]},
                    }
                ]
            }
            if payload.get("prompt_logprobs") is not None:
                response["prompt_logprobs"] = [
                    None,
                    {"2": {"logprob": -0.02}},
                    {"3": {"logprob": -0.03}},
                ]
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(response).encode())
            return

        self.send_response(404)
        self.end_headers()

    def log_message(self, *args):
        pass


def _serve():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _VllmHandler)
    server.requests = []
    server.fail_loads = 0
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


def test_vllm_sampling_client_loads_lora_checkpoint(tmp_path):
    server = _serve()
    try:
        checkpoint = tmp_path / "ckpt_1.tar.gz"
        src_dir = tmp_path / "src"
        src_dir.mkdir()
        (src_dir / "adapter_config.json").write_text("{}")
        (src_dir / "adapter_model.safetensors").write_bytes(b"weights")
        with tarfile.open(checkpoint, "w:gz") as tar:
            tar.add(src_dir, arcname="")

        client = VllmSamplingClient(
            base_url=f"http://127.0.0.1:{server.server_port}",
            model_name="Qwen/Qwen3-4B",
            lora_base_dir=tmp_path / "loras",
        )
        model_name = client.ensure_lora_loaded("model_a", AnyPath(checkpoint), checkpoint_id="ckpt_1")

        assert model_name == "model_a_ckpt_1"
        assert (tmp_path / "loras" / "model_a_ckpt_1" / "adapter_config.json").exists()
        assert server.requests[0][0] == "/v1/load_lora_adapter"
        assert server.requests[0][1]["lora_name"] == "model_a_ckpt_1"
    finally:
        server.shutdown()


def test_vllm_sampling_client_unloads_previous_lora_checkpoint(tmp_path):
    server = _serve()
    try:
        for checkpoint_id in ("ckpt_1", "ckpt_2"):
            checkpoint = tmp_path / f"{checkpoint_id}.tar.gz"
            src_dir = tmp_path / checkpoint_id
            src_dir.mkdir()
            (src_dir / "adapter_config.json").write_text("{}")
            (src_dir / "adapter_model.safetensors").write_bytes(checkpoint_id.encode())
            with tarfile.open(checkpoint, "w:gz") as tar:
                tar.add(src_dir, arcname="")

        client = VllmSamplingClient(
            base_url=f"http://127.0.0.1:{server.server_port}",
            model_name="Qwen/Qwen3-4B",
            lora_base_dir=tmp_path / "loras",
        )
        assert client.ensure_lora_loaded("model_a", AnyPath(tmp_path / "ckpt_1.tar.gz"), checkpoint_id="ckpt_1")
        assert client.ensure_lora_loaded("model_a", AnyPath(tmp_path / "ckpt_2.tar.gz"), checkpoint_id="ckpt_2")

        assert [request[0] for request in server.requests] == [
            "/v1/load_lora_adapter",
            "/v1/unload_lora_adapter",
            "/v1/load_lora_adapter",
        ]
        assert server.requests[1][1]["lora_name"] == "model_a_ckpt_1"
        assert server.requests[2][1]["lora_name"] == "model_a_ckpt_2"
    finally:
        server.shutdown()


def test_vllm_sampling_client_retries_lora_load(tmp_path):
    server = _serve()
    server.fail_loads = 1
    try:
        checkpoint = tmp_path / "ckpt_retry.tar.gz"
        src_dir = tmp_path / "src_retry"
        src_dir.mkdir()
        (src_dir / "adapter_config.json").write_text("{}")
        (src_dir / "adapter_model.safetensors").write_bytes(b"weights")
        with tarfile.open(checkpoint, "w:gz") as tar:
            tar.add(src_dir, arcname="")

        client = VllmSamplingClient(
            base_url=f"http://127.0.0.1:{server.server_port}",
            model_name="Qwen/Qwen3-4B",
            lora_base_dir=tmp_path / "loras",
            lora_load_retries=2,
            lora_load_retry_sleep_sec=0,
        )
        model_name = client.ensure_lora_loaded("model_a", AnyPath(checkpoint), checkpoint_id="ckpt_retry")

        assert model_name == "model_a_ckpt_retry"
        assert [request[0] for request in server.requests] == [
            "/v1/load_lora_adapter",
            "/v1/load_lora_adapter",
        ]
    finally:
        server.shutdown()


def test_vllm_sampling_client_maps_completion_response(tmp_path):
    server = _serve()
    try:
        client = VllmSamplingClient(
            base_url=f"http://127.0.0.1:{server.server_port}/v1",
            model_name="Qwen/Qwen3-4B",
            lora_base_dir=tmp_path / "loras",
        )
        params = types.SamplingParams(
            temperature=0.0,
            max_tokens=2,
            seed=7,
            top_k=-1,
            top_p=1.0,
        )

        sequences, prompt_logprobs = client.sample_many(
            [[1, 2, 3]],
            [params],
            ["Qwen/Qwen3-4B"],
            ["session:0"],
        )

        assert sequences == [
            types.GeneratedSequence(tokens=[101, 102], logprobs=[-0.1, -0.2], stop_reason="length")
        ]
        assert prompt_logprobs == [None]
        assert server.requests[0][0] == "/v1/completions"
        assert server.requests[0][1]["prompt"] == [1, 2, 3]
        assert server.requests[0][1]["return_token_ids"] is True
        headers = {k.lower(): v for k, v in server.requests[0][2].items()}
        assert headers["x-session-id"] == "session:0"
    finally:
        server.shutdown()


def test_vllm_sampling_client_maps_prompt_logprobs(tmp_path):
    server = _serve()
    try:
        client = VllmSamplingClient(
            base_url=f"http://127.0.0.1:{server.server_port}",
            model_name="Qwen/Qwen3-4B",
            lora_base_dir=tmp_path / "loras",
        )
        params = types.SamplingParams(
            temperature=0.0,
            max_tokens=2,
            seed=7,
            top_k=-1,
            top_p=1.0,
        )

        _, prompt_logprobs = client.sample_many(
            [[1, 2, 3]],
            [params],
            ["Qwen/Qwen3-4B"],
            [None],
            prompt_logprobs=True,
        )

        assert prompt_logprobs == [[0.0, -0.02, -0.03]]
        assert server.requests[0][1]["prompt_logprobs"] == 1
    finally:
        server.shutdown()
