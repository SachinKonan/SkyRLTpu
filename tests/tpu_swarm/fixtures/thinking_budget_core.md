The adjacent tar.gz holds three unmodified Python files from the serving bundle
used by RG-LRU run005, to verify focused patch application and execute the actual
patched sampler on CPU without importing TPU/vLLM platform dependencies.

Source: gs://sk7524-tinker-tpu-us-central2/code-bundles/tpuswarm-skyrl-v5p32-cells-v22.tar.gz
SHA256: 515ed9a5f2c3021b31ec78bd5eb9baaab983e2e20aa7c84fede9188f02d93a12

Files: tpu_runner.py, layers/jax/sample/sampling.py, layers/common/binary_search.py
under third_party/tpu-inference/tpu_inference. The original license headers remain.

`thinking_markers.json` records marker/cue tokens and tokenizer SHA256 hashes.
The completer compatibility fixtures additionally contain compressed finite-state
tables built from every vocabulary token in those same tokenizer.json files,
plus real decoded streams with missing headers, early answers, literal markers,
alternate tokenization and incomplete markers. The detector was built using
`build_text_detector` with special tokens retained and whitespace cleanup off.
The runtime build function is checked against decoded-substring matching; client parity
tests execute the actual single and grouped completer methods, not a duplicate
implementation of their branch logic.
