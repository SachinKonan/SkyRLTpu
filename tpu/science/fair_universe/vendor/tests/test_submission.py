import json
from pathlib import Path
import zipfile
import numpy as np
from fairdiscover.submission import load_test_batch


def test_compressed_input_test_order_and_mask():
    mask = np.array([[1, 0, 1], [0, 1, 0]], dtype=bool)
    compressed = np.array([[1,2,3],[4,5,6]], dtype=np.float16)
    x = load_test_batch(compressed, 0, 2, mask)
    assert x.shape == (2,2,3)
    assert x[0,mask].tolist() == [1,2,3]
    assert x[1,mask].tolist() == [4,5,6]
    assert np.all(x[:,~mask] == 0)
