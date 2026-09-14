"""Execute a downloaded baseline inside the pilot's filesystem namespace."""
import importlib.util
import json
from pathlib import Path
import resource
import sys
import time

started = time.monotonic()
import numpy as np
import torch
from macro_place.loader import load_benchmark_from_dir

torch.set_num_threads(4)
torch.set_num_interop_threads(1)
name, source, class_name = sys.argv[1:]
benchmark, _ = load_benchmark_from_dir(
    '/eval/external/MacroPlacement/Testcases/ICCAD04/' + name)
loaded = time.monotonic()
spec = importlib.util.spec_from_file_location('baseline_submission', source)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
placer = getattr(module, class_name)()
positions = placer.place(benchmark)
if isinstance(positions, torch.Tensor):
    positions = positions.detach().cpu().numpy()
positions = np.asarray(positions, dtype=np.float32)
np.save('/output/positions.npy', positions, allow_pickle=False)
Path('/output/child.json').write_text(json.dumps({
    'import_and_load_seconds': loaded - started,
    'candidate_seconds': time.monotonic() - loaded,
    'peak_rss_mib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
    'cuda_available': torch.cuda.is_available(),
}))
