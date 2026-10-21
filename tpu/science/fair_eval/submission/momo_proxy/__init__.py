"""Clean-room CPU proxy evaluator for the Partcl/TILOS placement task.

Drop-in replacement for the ``Evaluator`` helper given to placement candidates.
The native code (proxy.c) is written from the official Python scorer only; it does
not derive from any challenge entry. Build libmomoproxy.so with build.py before use.

Scores are optimization feedback. The trusted grader remains the source of truth
for legality and the reported cost.
"""
import ctypes as ct
from pathlib import Path

import numpy as np

_D = ct.POINTER(ct.c_double)
_I = ct.POINTER(ct.c_int)
KEYS = ('proxy_cost', 'wirelength_cost', 'density_cost', 'congestion_cost')
REBUILD_EVERY = 128


def _library():
    lib = ct.CDLL(str(Path(__file__).with_name('libmomoproxy.so')))
    lib.mp_create.restype = ct.c_void_p
    lib.mp_create.argtypes = [ct.c_int, ct.c_int, _D, _D, ct.c_int, _I, _D, _D, ct.c_int, _I, _D,
                              ct.c_int, ct.c_int, ct.c_double, ct.c_double, ct.c_double, ct.c_double,
                              ct.c_double, ct.c_double, ct.c_int, ct.c_double]
    for name, args in [('mp_destroy', [ct.c_void_p]), ('mp_rebuild', [ct.c_void_p]),
                       ('mp_set_positions', [ct.c_void_p, _D]),
                       ('mp_move', [ct.c_void_p, ct.c_int, ct.c_double, ct.c_double]),
                       ('mp_score', [ct.c_void_p, _D])]:
        f = getattr(lib, name)
        f.restype, f.argtypes = None, args
    return lib


def _f64(value, shape=None):
    a = np.array(value, dtype=np.float64, order='C', copy=True)
    if shape is not None and a.shape != shape:
        raise ValueError(f'expected shape {shape}, got {a.shape}')
    if not np.isfinite(a).all():
        raise ValueError('nonfinite array')
    return a


def _i32(value):
    a = np.asarray(value)
    if a.dtype.kind not in 'iu' or (a < 0).any() or (a > np.iinfo(np.int32).max).any():
        raise ValueError('expected nonnegative int32-compatible indices')
    return np.array(a, dtype=np.int32, order='C', copy=True)


def _p(a):
    return a.ctypes.data_as(_D if a.dtype == np.float64 else _I)


class Evaluator:
    """One mutable evaluator per search; not thread safe.

    apply(block_id, xy) opens one transaction; commit() or revert() closes it.
    evaluate/evaluate_batch score whole alternative layouts and keep the search state.
    """

    def __init__(self, problem, positions=None):
        self._handle = None
        self._pending = None
        self._steps = 0
        self._lib = _library()
        p = problem
        self._initial = _f64(p['initial_positions'])
        if self._initial.ndim != 2 or self._initial.shape[1] != 2 or not len(self._initial):
            raise ValueError('invalid positions shape')
        n = len(self._initial)
        self._sizes = _f64(p['sizes'], (n, 2))
        self._canvas = _f64(p['canvas'], (2,))
        self._fixed = np.array(p['fixed'], dtype=bool).reshape(n)
        rows, cols = (int(v) for v in _i32(p['grid_shape']))
        nh = int(p['num_hard'])
        smooth = int(p['congestion_smoothing_range'])
        if rows <= 0 or cols <= 0 or not 0 <= nh <= n or smooth < 0:
            raise ValueError('invalid grid / hard count / smoothing')
        if (self._sizes <= 0).any() or (self._canvas <= 0).any():
            raise ValueError('nonpositive geometry')
        h_rpm, v_rpm = _f64(p['routes_per_micron'], (2,))
        h_alloc, v_alloc = _f64(p['macro_routing_allocation'], (2,))
        if h_rpm <= 0 or v_rpm <= 0 or h_alloc < 0 or v_alloc < 0:
            raise ValueError('invalid routing capacity')
        wl_norm = float(p['wirelength_normalizer'])
        if not np.isfinite(wl_norm) or wl_norm <= 0:
            raise ValueError('invalid wirelength normalizer')
        owner = np.asarray(p['pin_owner']).astype(np.int64)
        offset = _f64(p['pin_offset'], (len(owner), 2))
        ports = _f64(p['port_positions']).reshape(-1, 2)
        if (owner < 0).any() or (owner >= n + len(ports)).any():
            raise ValueError('invalid pin owners')
        start = _i32(p['net_offsets'])
        weight = _f64(p['net_weights'])
        if (start.ndim != 1 or len(start) < 2 or start[0] != 0 or start[-1] != len(owner)
                or (np.diff(start.astype(np.int64)) < 1).any() or weight.shape != (len(start) - 1,)
                or (weight < 0).any()):
            raise ValueError('invalid net CSR')
        is_port = owner >= n
        pin_owner = np.where(is_port, -1, owner).astype(np.int32)
        # A port pin carries its absolute position; a macro pin carries its offset.
        pin_x, pin_y = offset[:, 0].copy(), offset[:, 1].copy()
        pin_x[is_port] = ports[owner[is_port] - n, 0]
        pin_y[is_port] = ports[owner[is_port] - n, 1]
        self._positions = self._validate(self._initial if positions is None else positions)
        keep = [self._positions, self._sizes, pin_owner, pin_x, pin_y, start, weight]
        self._handle = self._lib.mp_create(
            n, nh, _p(keep[0]), _p(keep[1]), len(owner), _p(keep[2]), _p(keep[3]), _p(keep[4]),
            len(weight), _p(keep[5]), _p(keep[6]), rows, cols, float(self._canvas[0]),
            float(self._canvas[1]), float(h_rpm), float(v_rpm), float(h_alloc), float(v_alloc),
            smooth, wl_norm)
        if not self._handle:
            raise MemoryError('native evaluator allocation failed')

    def _validate(self, positions):
        pos = _f64(positions, self._initial.shape)
        if not np.array_equal(pos[self._fixed], self._initial[self._fixed]):
            raise ValueError('fixed blocks changed')
        if (pos < 0).any() or (pos > self._canvas).any():
            raise ValueError('centers outside canvas')
        # The independent runner casts candidate output to float32.
        return np.ascontiguousarray(pos.astype(np.float32).astype(np.float64))

    def _check(self, *, idle=False):
        if not self._handle:
            raise RuntimeError('evaluator is closed')
        if idle and self._pending is not None:
            raise RuntimeError('commit or revert the pending move first')

    def _score(self):
        out = np.zeros(4)
        self._lib.mp_score(self._handle, _p(out))
        if not np.isfinite(out).all():
            raise ArithmeticError('nonfinite native score')
        return dict(zip(KEYS, map(float, out)))

    def _tick(self):
        self._steps += 1
        if self._steps % REBUILD_EVERY == 0:
            self._lib.mp_rebuild(self._handle)

    def score(self):
        self._check()
        return self._score()

    def apply(self, block_id, xy):
        self._check(idle=True)
        if not isinstance(block_id, (int, np.integer)) or not 0 <= block_id < len(self._positions):
            raise ValueError('invalid block id')
        if self._fixed[block_id]:
            raise ValueError('cannot move a fixed block')
        xy = _f64(xy, (2,)).astype(np.float32).astype(np.float64)
        if (xy < 0).any() or (xy > self._canvas).any():
            raise ValueError('center outside canvas')
        i = int(block_id)
        self._pending = (i, self._positions[i].copy())
        self._positions[i] = xy
        self._lib.mp_move(self._handle, i, float(xy[0]), float(xy[1]))
        return self._score()

    def revert(self):
        self._check()
        if self._pending is None:
            raise RuntimeError('no pending move')
        i, old = self._pending
        self._positions[i] = old
        self._lib.mp_move(self._handle, i, float(old[0]), float(old[1]))
        self._pending = None
        self._tick()
        return self._score()

    def commit(self):
        self._check()
        if self._pending is None:
            raise RuntimeError('no pending move')
        self._pending = None
        self._tick()

    def rebuild(self):
        self._check(idle=True)
        self._lib.mp_rebuild(self._handle)
        return self._score()

    def evaluate(self, positions):
        """Full score of an alternative layout; restores the current search state."""
        self._check(idle=True)
        pos = self._validate(positions)
        self._lib.mp_set_positions(self._handle, _p(pos))
        try:
            return self._score()
        finally:
            self._lib.mp_set_positions(self._handle, _p(self._positions))

    def evaluate_batch(self, layouts):
        return [self.evaluate(layout) for layout in layouts]

    def evaluate_moves(self, block_ids, positions):
        """Score independent single-block alternatives against the current layout."""
        self._check(idle=True)
        ids = _i32(block_ids)
        if ids.ndim != 1:
            raise ValueError('block_ids must be one-dimensional')
        xy = _f64(positions, (len(ids), 2))
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
            self._lib.mp_destroy(self._handle)
            self._handle = None

    def __enter__(self):
        self._check()
        return self

    def __exit__(self, *args):
        self.close()

    def __del__(self):
        self.close()
