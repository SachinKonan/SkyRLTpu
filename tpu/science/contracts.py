"""Submission parsing and output validation; never execute source here."""
import ast
import numpy as np


def rust_literal(source):
    tree = ast.parse(source)
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.Assign):
        raise ValueError('expected exactly one literal RUST_CODE assignment')
    node = tree.body[0]
    if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name) or node.targets[0].id != 'RUST_CODE':
        raise ValueError('expected RUST_CODE assignment')
    if not isinstance(node.value, ast.Constant) or not isinstance(node.value.value, str) or not node.value.value.strip():
        raise ValueError('RUST_CODE must be a nonempty string literal')
    return node.value.value


def weights(value, assets):
    w = np.asarray(value, dtype=np.float64)
    if w.shape != (assets+1,) or not np.isfinite(w).all():
        raise ValueError('action must be finite [N+1] weights')
    if w.min() < -1e-7 or abs(w.sum()-1) > 1e-5:
        raise ValueError('action must be long-only and sum to one')
    # Repair only floating-point roundoff within the public tolerance.
    w = np.maximum(w,0)
    return w / w.sum()


def placement(problem, result):
    if not isinstance(result, dict) or set(result) != {'cell_ids','orientations'}:
        raise ValueError('expected cell_ids and orientations only')
    cells = np.asarray(result['cell_ids'])
    orientations = result['orientations']
    macros = problem['movable_macros']
    if cells.shape != (len(macros),) or cells.dtype.kind not in 'iu' or len(orientations) != len(macros):
        raise ValueError('one integer cell and orientation per movable macro required')
    cols, rows = problem['num_cols'], problem['num_rows']
    width, height = problem['width'], problem['height']
    rectangles = [tuple(o['bounds']) for o in problem['fixed_objects'] if 'bounds' in o]
    rectangles += [tuple(b[:4]) for b in problem['blockages']]
    for cell, orient, macro in zip(cells, orientations, macros):
        if not 0 <= cell < cols*rows or orient not in macro['allowed_orientations']:
            raise ValueError('illegal cell or orientation')
        x, y = (int(cell)%cols+.5)*width/cols, (int(cell)//cols+.5)*height/rows
        w,h = macro['width'], macro['height']
        if orient in ('E','W','FE','FW'): w,h = h,w
        rect = (x-w/2,y-h/2,x+w/2,y+h/2)
        if rect[0] < -1e-8 or rect[1] < -1e-8 or rect[2] > width+1e-8 or rect[3] > height+1e-8:
            raise ValueError('macro rectangle outside canvas')
        for other in rectangles:
            if min(rect[2],other[2])-max(rect[0],other[0]) > 1e-8 and min(rect[3],other[3])-max(rect[1],other[1]) > 1e-8:
                raise ValueError('macro overlap or blockage violation')
        rectangles.append(rect)
    return dict(cell_ids=cells.tolist(),orientations=list(orientations))
