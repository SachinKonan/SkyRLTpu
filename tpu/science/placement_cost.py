"""Trusted Circuit Training PlacementCost adapter. Never substitutes proxy engines."""
from contextlib import contextmanager
from pathlib import Path
import time
import numpy as np
from .contracts import placement as validate
from .placement_rpc import PlacementCost

COMPLETION = dict(num_steps=[100,100,100], move_distance_factors=[1.,1.,1.],
                  attract_factor=[100.,1e-3,1e-5], repel_factor=[0.,1e6,1e7],
                  use_current_loc=True, move_macros=False)


@contextmanager
def connection(repo,binary,netlist,initial,problem, *, timeout=30):
    # Imported lazily, so contract/data tests do not require the native binary.
    if not Path(binary).is_file():raise RuntimeError('the pinned PlacementCost binary is not installed')
    plc=PlacementCost(binary,netlist,timeout=timeout)
    try:
        plc.set_canvas_size(problem['width'],problem['height'])
        plc.set_placement_grid(problem['num_cols'],problem['num_rows'])
        plc.set_congestion_grid(problem['num_cols'],problem['num_rows'])
        capacity=problem['routing_capacity']
        plc.set_routes_per_micron(capacity['horizontal'],capacity['vertical'])
        plc.set_macro_routing_allocation(capacity['macro_horizontal'],capacity['macro_vertical'])
        plc.set_congestion_smooth_range(5);plc.set_overlap_threshold(0.)
        # Some upstream initial files have slightly out-of-bounds hard macros.
        # They are replaced by independently validated grid placements before scoring.
        plc.set_canvas_boundary_check(False)
        for blockage in problem['blockages']:plc.create_blockage(*blockage)
        plc.restore_placement(str(Path(initial).resolve()))
        plc.set_canvas_boundary_check(True)
        for node in problem['fixed_objects']:plc.fix_node_coord(node['id'])
        plc.make_soft_macros_square()
        yield plc
    finally:
        # Explicit lifecycle; upstream __del__ otherwise delays process termination.
        plc.close()


def score(repo,binary,netlist,initial,problem,placement, *, timeout=30):
    placement=validate(problem,placement)
    start=time.monotonic()
    with connection(repo,binary,netlist,initial,problem,timeout=timeout) as plc:
        for i in problem['movable_macro_ids']:
            plc.unfix_node_coord(i);plc.unplace_node(i)
        for i,cell,orientation in zip(problem['movable_macro_ids'],placement['cell_ids'],placement['orientations']):
            plc.update_macro_orientation(i,orientation)
            if not plc.can_place_node(i,cell):raise ValueError('native PlacementCost rejects macro location')
            plc.place_node(i,cell);plc.fix_node_coord(i)
        # Deterministic restart from the same initial cluster locations every call.
        # Hard macros are fixed; the schedule is part of our manifest.
        scale=max(problem['width'],problem['height'])
        fixed_before={n['id']:plc.get_node_location(n['id']) for n in problem['fixed_objects']}
        hard_before={i:plc.get_node_location(i) for i in problem['movable_macro_ids']}
        plc.optimize_stdcells(True,True,False,False,False,1.0,COMPLETION['num_steps'],
                              [scale/100]*3,COMPLETION['attract_factor'],COMPLETION['repel_factor'])
        for i,location in {**fixed_before,**hard_before}.items():
            if not np.allclose(plc.get_node_location(i),location,atol=1e-6,rtol=0):
                raise ValueError('completion moved a fixed object or hard macro')
        costs=[plc.get_wirelength(),plc.get_congestion_cost(),plc.get_density_cost()]
        if not np.isfinite(costs).all() or any(x<0 for x in costs):raise ValueError('nonfinite or negative PlacementCost output')
    return costs,time.monotonic()-start
