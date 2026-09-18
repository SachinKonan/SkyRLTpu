"""CPU proxy helper. Host-side API for NumPy and concrete JAX arrays (not JIT).

Native scores are optimization feedback, not independent legality certificates.
Build libproxy.so ahead of candidate execution with build.py.
"""
import ctypes as ct
from pathlib import Path
import numpy as np

D = ct.POINTER(ct.c_double)
I = ct.POINTER(ct.c_int)


def _array(value, dtype, shape=None):
    a = np.array(value, dtype=dtype, order='C', copy=True)
    if shape is not None and a.shape != shape:
        raise ValueError(f'expected shape {shape}, got {a.shape}')
    if not np.isfinite(a).all():
        raise ValueError('nonfinite array')
    return a


def _ints(value):
    a = np.asarray(value)
    if a.dtype.kind not in 'iu' or (a < 0).any() or (a > np.iinfo(np.int32).max).any():
        raise ValueError('expected nonnegative int32-compatible indices')
    return _array(a, np.int32)


def _ptr(a):
    return a.ctypes.data_as(D if a.dtype == np.float64 else I)


def _library():
    lib = ct.CDLL(str(Path(__file__).with_name('libproxy.so')))
    problem_args = [D, D, I, ct.c_int, ct.c_int, ct.c_double, ct.c_double,
                    ct.c_int, ct.c_int, ct.c_int, ct.c_double, ct.c_double,
                    ct.c_double, ct.c_double, I, D, D, ct.c_int,
                    I, I, I, D, ct.c_int, ct.c_double, D]
    specs = {
        'create': (ct.c_void_p, problem_args),
        'destroy': (None, [ct.c_void_p]),
        'score_current': (ct.c_double, [ct.c_void_p, D, D, D]),
        'apply': (ct.c_double, [ct.c_void_p, ct.c_int, I, D, D, D, D, D]),
        'revert': (ct.c_double, [ct.c_void_p, D, D, D]),
        'commit': (None, [ct.c_void_p]),
        'rebuild': (None, [ct.c_void_p]),
        'set_positions': (None, [ct.c_void_p, D]),
    }
    for name, (restype, args) in specs.items():
        f = getattr(lib, 'cong_state_' + name)
        f.restype, f.argtypes = restype, args
    return lib


class Evaluator:
    """One mutable evaluator per search; not thread safe.

    apply(block_id, xy) opens one transaction; commit() or revert() closes it.
    evaluate/evaluate_batch do full rebuilds and leave the search state intact.
    Rebuild periodically to bound incremental floating-point drift.
    """
    def __init__(self, problem, positions=None):
        self._handle = None
        self._pending = None
        self._commits = 0
        self._probes = 0
        self._lib = _library()
        p = problem
        self._initial = _array(p['initial_positions'], np.float64)
        if self._initial.ndim != 2 or self._initial.shape[1] != 2 or len(self._initial) == 0:
            raise ValueError('invalid positions shape')
        n = len(self._initial)
        self._sizes = _array(p['sizes'], np.float64, (n, 2))
        self._canvas = _array(p['canvas'], np.float64, (2,))
        self._fixed = _array(p['fixed'], np.bool_, (n,))
        grid = _ints(p['grid_shape'])
        nh = int(p['num_hard'])
        smooth = int(p['congestion_smoothing_range'])
        if grid.shape != (2,) or (grid <= 0).any() or not 0 <= nh <= n or smooth < 0:
            raise ValueError('invalid grid/hard count/smoothing')
        if (self._sizes <= 0).any() or (self._canvas <= 0).any():
            raise ValueError('nonpositive geometry')
        routes = _array(p['routes_per_micron'], np.float64, (2,))
        allocation = _array(p['macro_routing_allocation'], np.float64, (2,))
        if (routes <= 0).any() or (allocation < 0).any():
            raise ValueError('invalid routing capacity')
        norm = float(p['wirelength_normalizer']) / self._canvas.sum()
        if not np.isfinite(norm) or norm <= 0:
            raise ValueError('invalid wirelength normalizer')
        owner = _ints(p['pin_owner'])
        offset = _array(p['pin_offset'], np.float64, (owner.size, 2))
        ports = _array(p['port_positions'], np.float64)
        if ports.ndim != 2 or ports.shape[1] != 2 or owner.ndim != 1 or (owner >= n + len(ports)).any():
            raise ValueError('invalid pin owners/ports')
        offsets = _ints(p['net_offsets'])
        weights = _array(p['net_weights'], np.float64)
        if (offsets.ndim != 1 or len(offsets) < 2 or offsets[0] != 0 or
            offsets[-1] != len(owner) or (np.diff(offsets.astype(np.int64)) < 1).any() or
            weights.shape != (len(offsets)-1,) or (weights < 0).any()):
            raise ValueError('invalid net CSR')
        pm = owner.copy()
        is_port = owner >= n
        offset[is_port] += ports[owner[is_port] - n]
        pm[is_port] = -1
        drivers = offsets[:-1].copy()
        sink_counts = np.diff(offsets) - 1
        sink_offsets = np.r_[0, np.cumsum(sink_counts)].astype(np.int32)
        sinks = np.concatenate([np.arange(a+1, b, dtype=np.int32) for a, b in zip(offsets[:-1], offsets[1:])])
        pos = self._validate(self._initial if positions is None else positions)
        self._positions = pos.copy()
        # C copies these buffers during creation; own them through the call.
        arrays = [pos, self._sizes, (~self._fixed).astype(np.int32), pm,
                  offset[:, 0].copy(), offset[:, 1].copy(), drivers,
                  sink_offsets, sinks, weights]
        a = arrays
        self._handle = self._lib.cong_state_create(
            _ptr(a[0]), _ptr(a[1]), _ptr(a[2]), n, nh, *self._canvas,
            int(grid[0]), int(grid[1]), smooth, *routes, *allocation,
            _ptr(a[3]), _ptr(a[4]), _ptr(a[5]), len(owner),
            _ptr(a[6]), _ptr(a[7]), _ptr(a[8]), _ptr(a[9]), len(weights), norm, None)
        if not self._handle:
            raise MemoryError('native evaluator allocation failed')

    def _validate(self, positions):
        pos = _array(positions, np.float64, self._initial.shape)
        if not np.array_equal(pos[self._fixed], self._initial[self._fixed]):
            raise ValueError('fixed blocks changed')
        # Hard-overlap legality remains the independent grader's responsibility.
        if (pos < 0).any() or (pos > self._canvas).any():
            raise ValueError('centers outside canvas')
        # Candidate output is cast to float32 by the independent runner.
        return pos.astype(np.float32).astype(np.float64)

    def _check(self, *, idle=False):
        if not self._handle:
            raise RuntimeError('evaluator is closed')
        if idle and self._pending is not None:
            raise RuntimeError('commit or revert the pending move first')

    def _score_call(self, name, *args):
        self._check()
        out = [ct.c_double() for _ in range(3)]
        cost = getattr(self._lib, 'cong_state_' + name)(
            self._handle, *args, *(ct.byref(x) for x in out))
        values = [cost, *(x.value for x in out)]
        if not np.isfinite(values).all():
            raise ArithmeticError('nonfinite native score')
        return dict(zip(('proxy_cost', 'wirelength_cost', 'density_cost', 'congestion_cost'), values))

    def score(self):
        return self._score_call('score_current')

    def apply(self, block_id, xy):
        self._check(idle=True)
        if not isinstance(block_id, (int, np.integer)) or not 0 <= block_id < len(self._positions):
            raise ValueError('invalid block id')
        if self._fixed[block_id]:
            raise ValueError('cannot move a fixed block')
        xy = _array(xy, np.float64, (2,)).astype(np.float32).astype(np.float64)
        if (xy < 0).any() or (xy > self._canvas).any():
            raise ValueError('center outside canvas')
        self._pending = (block_id, self._positions[block_id].copy())
        self._positions[block_id] = xy
        return self._score_call('apply', 1, _ptr(np.array([block_id], dtype=np.int32)),
                                _ptr(xy[:1].copy()), _ptr(xy[1:].copy()))

    def revert(self):
        self._check()
        if self._pending is None:
            raise RuntimeError('no pending move')
        i, old = self._pending
        result = self._score_call('revert')
        self._positions[i] = old
        self._pending = None
        self._probes += 1
        if self._probes % 128 == 0:
            result = self.rebuild()
        return result

    def commit(self):
        self._check()
        if self._pending is None:
            raise RuntimeError('no pending move')
        self._lib.cong_state_commit(self._handle)
        self._pending = None
        self._commits += 1
        if self._commits % 128 == 0:
            self.rebuild()

    def rebuild(self):
        self._check(idle=True)
        self._lib.cong_state_rebuild(self._handle)
        return self.score()

    def evaluate(self, positions):
        """Full score of an alternative; restores current search state."""
        self._check(idle=True)
        pos = self._validate(positions)
        self._lib.cong_state_set_positions(self._handle, _ptr(pos))
        try:
            return self.score()
        finally:
            self._lib.cong_state_set_positions(self._handle, _ptr(self._positions))

    def evaluate_batch(self, layouts):
        """Bounded-memory serial CPU batch; no parallel workers are spawned."""
        return [self.evaluate(layout) for layout in layouts]

    def evaluate_moves(self, block_ids, positions):
        """Score independent single-block alternatives against the current layout.

        This is a CPU batch API, not 96 simultaneous full-layout copies.
        """
        self._check(idle=True)
        ids = _ints(block_ids)
        if ids.ndim != 1:
            raise ValueError('block_ids must be one-dimensional')
        xy = _array(positions, np.float64, (len(ids), 2))
        rows = []
        for block_id, point in zip(ids, xy):
            try:
                rows.append(self.apply(int(block_id), point))
            finally:
                if self._pending is not None:
                    self.revert()
        return rows

    def close(self):
        if self._handle:
            self._lib.cong_state_destroy(self._handle)
            self._handle = None

    def __enter__(self):
        self._check()
        return self

    def __exit__(self, *args):
        self.close()

    def __del__(self):
        self.close()
