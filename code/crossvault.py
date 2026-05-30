import numpy as np
import compas_tna.envelope.parametricenvelope
original_apply_bounds = compas_tna.envelope.parametricenvelope.ParametricEnvelope.apply_bounds_to_formdiagram

def fixed_apply_bounds(self, formdiagram):
    xy = np.array(formdiagram.vertices_attributes("xy"))
    try:
        zub, zlb = self.compute_bounds(xy[:, 0], xy[:, 1])
    except TypeError:
        zub, zlb = self.compute_bounds(xy[:, 0], xy[:, 1], self.thickness)
        
    for i, key in enumerate(formdiagram.vertices()):
        ub_val = zub[i]
        lb_val = zlb[i]
        if hasattr(ub_val, "__len__"): ub_val = ub_val[0]
        if hasattr(lb_val, "__len__"): lb_val = lb_val[0]
        formdiagram.vertex_attribute(key, "ub", float(ub_val))
        formdiagram.vertex_attribute(key, "lb", float(lb_val))

compas_tna.envelope.parametricenvelope.ParametricEnvelope.apply_bounds_to_formdiagram = fixed_apply_bounds

from compas_tna.envelope import PointedVaultEnvelope
from compas_tna.diagrams import FormDiagram
from compas_tno.analysis import Analysis
from compas.data import json_dump 
import os
import math
from compas.geometry import Vector

# ----------------------------------------
# 1. Shape geometric definition
# ----------------------------------------
xy_span = [[0.0, 10.0], [0.0, 16.19]] 
thickness = 0.3
max_rise_at_crown = 2.0 
discretisation_level = [10, 10] 

vault = PointedVaultEnvelope(
    x_span=xy_span[0],
    y_span=xy_span[1],
    thickness=thickness,
    hc=max_rise_at_crown,
    n=discretisation_level[0]
)

# ----------------------------------------
# 2. Form diagram geometric definition (Quadrant)
# ----------------------------------------
discretisation = 10
cx, cy = 5.0, 8.095
form = FormDiagram.create_fan(x_span=[cx, 10.0], y_span=[cy, 16.19], n_fans=discretisation, n_hoops=discretisation)

# Set supports
for vkey in form.vertices():
    x, y = form.vertex_coordinates(vkey)[:2]
    if abs(x - 10.0) < 1e-6 or abs(y - 16.19) < 1e-6:
        form.vertex_attribute(vkey, 'is_fixed', True)

# ----------------------------------------
# 3. Structural Analysis
# ----------------------------------------

def get_spokes_indices(form, cx, cy):
    vkeys = list(form.vertices())
    center_vkey = min(vkeys, key=lambda v: (form.vertex_attribute(v, 'x') - cx)**2 + (form.vertex_attribute(v, 'y') - cy)**2)
    
    # Group vertices by angle from center
    from collections import defaultdict
    angle_groups = defaultdict(list)
    
    for vkey in vkeys:
        if vkey == center_vkey: continue
        x, y = form.vertex_coordinates(vkey)[:2]
        angle = math.atan2(y - cy, x - cx)
        # Round angle to handle floating point precision
        angle_rounded = round(angle, 4)
        angle_groups[angle_rounded].append(vkey)
        
    spokes_indices = []
    vkey_to_idx = {vkey: i for i, vkey in enumerate(vkeys)}
    
    for angle in sorted(angle_groups.keys()):
        vs = angle_groups[angle]
        # Sort by distance from center
        vs.sort(key=lambda v: (form.vertex_attribute(v, 'x') - cx)**2 + (form.vertex_attribute(v, 'y') - cy)**2)
        spoke = [center_vkey] + vs
        spokes_indices.append([vkey_to_idx[v] for v in spoke])
    
    print(f"Debug: Identified {len(spokes_indices)} spokes via angular grouping.")
    for i, s in enumerate(spokes_indices):
        print(f"  Spoke {i}: {len(s)} points")
    return spokes_indices

def export_geometry_to_obj_and_json(points, obj_filepath, json_filepath, edge_indices=None, corner=None, spokes=None):
    if not points: return False
    try:
        with open(obj_filepath, 'w') as f:
            for p in points:
                f.write(f"v {p[0]} {p[1]} {p[2]}\n")
            if edge_indices:
                for u, v in edge_indices:
                    f.write(f"l {u+1} {v+1}\n")
    except Exception: pass

    try:
        from compas.geometry import Point, Line
        compas_points = [Point(*p) for p in points]
        compas_lines = []
        if edge_indices:
            for u, v in edge_indices:
                compas_lines.append(Line(points[u], points[v]))
                
        with open(json_filepath, 'w') as f:
            json_dump({
                "points": compas_points,
                "lines": compas_lines,
                "edge_point_indices": edge_indices or [],
                "quadrant_corner": corner,
                "spokes": spokes or []
            }, f)
        print(f"Exported JSON to {json_filepath}")
        return True
    except Exception as e:
        print(f"Error JSON: {e}")
        return False

# Identify spokes BEFORE analysis while geometry is perfect
spokes = get_spokes_indices(form, cx, cy)

print("-" * 20)
print("Running Minimum Thrust Analysis")
print("-" * 20)
analysis_min = Analysis.create_minthrust_analysis(form, vault, printout=True)
analysis_min.apply_selfweight()
analysis_min.apply_envelope()
analysis_min.set_up_optimiser()
analysis_min.run()

points_min = [[analysis_min.formdiagram.vertex_attribute(v, 'x'), 
               analysis_min.formdiagram.vertex_attribute(v, 'y'), 
               analysis_min.formdiagram.vertex_attribute(v, 'z')] for v in analysis_min.formdiagram.vertices()]
vkey_to_idx = {vkey: i for i, vkey in enumerate(analysis_min.formdiagram.vertices())}
edges = [(vkey_to_idx[u], vkey_to_idx[v]) for u, v in analysis_min.formdiagram.edges()]
corner = [cx, cy, points_min[vkey_to_idx[list(analysis_min.formdiagram.vertices())[0]]][2]] # Corrected corner Z

export_geometry_to_obj_and_json(points_min, "./min_thrust_quadrant_geometry.obj", "./min_thrust_quadrant_geometry.json", edges, corner, spokes)

print("-" * 20)
print("Running Maximum Thrust Analysis")
print("-" * 20)
analysis_max = Analysis.create_maxthrust_analysis(form, vault, printout=True)
analysis_max.apply_selfweight()
analysis_max.apply_envelope()
analysis_max.set_up_optimiser()
analysis_max.run()

points_max = [[analysis_max.formdiagram.vertex_attribute(v, 'x'), 
               analysis_max.formdiagram.vertex_attribute(v, 'y'), 
               analysis_max.formdiagram.vertex_attribute(v, 'z')] for v in analysis_max.formdiagram.vertices()]

export_geometry_to_obj_and_json(points_max, "./max_thrust_quadrant_geometry.obj", "./max_thrust_quadrant_geometry.json", edges, corner, spokes)
