import numpy as np
import compas_tna.envelope.parametricenvelope
import compas_tna.envelope.pointedvault
import math

# ==============================================================================
# 0. Geometric Patch for PointedVaultEnvelope
# ==============================================================================
# The default PointedVaultEnvelope in compas_tna 0.7.0 assumes circle centers at z=0,
# which leads to non-monotonic "humped" shapes for shallow vaults (h < L/2).
# This patch ensures monotonic circular arcs by allowing circle centers to be at z < 0.

def _monotonic_find_r(h, length):
    return (h**2 + (length / 2)**2) / (2 * h)

def _sqrt(x):
    if x < 0: return 0.0
    return math.sqrt(x)

def monotonic_pointedvault_middle(x, y, min_lb, x_span, y_span, hc, he=None, hm=None, tol=1e-6):
    x0, x1 = x_span
    y0, y1 = y_span
    lx, ly = x1 - x0, y1 - y0
    middle = np.zeros((len(x), 1))
    for i in range(len(x)):
        xi, yi = x[i], y[i]
        x_mid, y_mid = (x0 + x1) / 2.0, (y0 + y1) / 2.0
        # Determine quadrant based on diagonals
        if abs(xi - x_mid) * ly > abs(yi - y_mid) * lx:
            # Side quadrants (arch spans in x)
            hi = hc
            ri = _monotonic_find_r(hi, lx)
            zc = hi - ri
            zi = _sqrt(ri**2 - (xi - x_mid)**2) + zc
        else:
            # Top/Bottom quadrants (arch spans in y)
            hi = hc
            ri = _monotonic_find_r(hi, ly)
            zc = hi - ri
            zi = _sqrt(ri**2 - (yi - y_mid)**2) + zc
        middle[i] = zi
    return middle

def monotonic_pointedvault_bounds(x, y, thk, min_lb, x_span, y_span, hc, he=None, hm=None, tol=1e-6):
    x0, x1 = x_span
    y0, y1 = y_span
    lx, ly = x1 - x0, y1 - y0
    ub = np.ones((len(x), 1))
    lb = np.ones((len(x), 1)) * -min_lb
    for i in range(len(x)):
        xi, yi = x[i], y[i]
        x_mid, y_mid = (x0 + x1) / 2.0, (y0 + y1) / 2.0
        if abs(xi - x_mid) * ly > abs(yi - y_mid) * lx:
            hi = hc
            ri = _monotonic_find_r(hi, lx)
            zc = hi - ri
            ub[i] = _sqrt((ri + thk/2)**2 - (xi - x_mid)**2) + zc
            lb[i] = _sqrt((ri - thk/2)**2 - (xi - x_mid)**2) + zc
        else:
            hi = hc
            ri = _monotonic_find_r(hi, ly)
            zc = hi - ri
            ub[i] = _sqrt((ri + thk/2)**2 - (yi - y_mid)**2) + zc
            lb[i] = _sqrt((ri - thk/2)**2 - (yi - y_mid)**2) + zc
    return ub, lb

# Apply geometric patches
compas_tna.envelope.pointedvault.pointedvault_middle = monotonic_pointedvault_middle
compas_tna.envelope.pointedvault.pointedvault_bounds = monotonic_pointedvault_bounds

# Original monkey-patch for bounds application
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
from compas.geometry import Vector

# ==============================================================================
# 1. Global Design Configuration
# ==============================================================================

# Span and geometric rise
X_SPAN = [0.0, 10.0]
Y_SPAN = [0.0, 15.0]
HC_RISE = 2.0
THICKNESS = 1.0

# Discretisation
N_DISCRETISATION = 2

# ==============================================================================
# 2. Geometry Setup
# ==============================================================================

# Create Vault Envelope
vault = PointedVaultEnvelope(
    x_span=X_SPAN,
    y_span=Y_SPAN,
    thickness=THICKNESS,
    hc=HC_RISE,
    n=N_DISCRETISATION
)

# Calculate Center (Crown)
cx = (X_SPAN[0] + X_SPAN[1]) / 2.0
cy = (Y_SPAN[0] + Y_SPAN[1]) / 2.0

# Create full fan vault diagram
form = FormDiagram.create_fan(
    x_span=X_SPAN, 
    y_span=Y_SPAN, 
    n_fans=4 * N_DISCRETISATION, 
    n_hoops=N_DISCRETISATION
)

# Set supports (All outer edges of the vault)
for vkey in form.vertices():
    x, y = form.vertex_coordinates(vkey)[:2]
    if (abs(x - X_SPAN[0]) < 1e-6 or abs(x - X_SPAN[1]) < 1e-6 or 
        abs(y - Y_SPAN[0]) < 1e-6 or abs(y - Y_SPAN[1]) < 1e-6):
        form.vertex_attribute(vkey, 'is_fixed', True)

# ==============================================================================
# 3. Utility Functions
# ==============================================================================

def get_spokes_indices(form, cx, cy):
    """Identifies radial spokes (crease lines) in the form diagram."""
    vkeys = list(form.vertices())
    center_vkey = min(vkeys, key=lambda v: (form.vertex_attribute(v, 'x') - cx)**2 + (form.vertex_attribute(v, 'y') - cy)**2)
    
    spokes_indices = []
    vkey_to_idx = {vkey: i for i, vkey in enumerate(vkeys)}
    
    # Trace from each neighbor of the center vertex
    for start_nbr in form.vertex_neighbors(center_vkey):
        spoke = [center_vkey, start_nbr]
        curr = start_nbr
        prev = center_vkey
        while True:
            candidates = []
            for nbr2 in form.vertex_neighbors(curr):
                if nbr2 == prev: continue
                p_prev = form.vertex_coordinates(prev)
                p_curr = form.vertex_coordinates(curr)
                p_next = form.vertex_coordinates(nbr2)
                v1 = Vector(p_curr[0]-p_prev[0], p_curr[1]-p_prev[1], 0)
                v2 = Vector(p_next[0]-p_curr[0], p_next[1]-p_curr[1], 0)
                if v1.length > 1e-8 and v2.length > 1e-8:
                    dot = v1.unitized().dot(v2.unitized())
                    if dot > 0.7:
                        candidates.append((dot, nbr2))
            
            if not candidates: break
            candidates.sort(key=lambda x: x[0], reverse=True)
            next_v = candidates[0][1]
            spoke.append(next_v)
            prev, curr = curr, next_v
        
        spokes_indices.append([vkey_to_idx[v] for v in spoke])
    
    def get_spoke_angle(s_idx):
        p0 = form.vertex_coordinates(vkeys[s_idx[0]])
        p1 = form.vertex_coordinates(vkeys[s_idx[1]])
        return math.atan2(p1[1] - p0[1], p1[0] - p0[0])
    
    spokes_indices.sort(key=get_spoke_angle)
    return spokes_indices

def export_vault_geometry(analysis, spokes, obj_path, json_path, cx, cy):
    """Exports optimized geometry to OBJ and JSON formats."""
    fdiagram = analysis.formdiagram
    vkey_to_idx = {vkey: i for i, vkey in enumerate(fdiagram.vertices())}
    
    points = [[fdiagram.vertex_attribute(v, 'x'), 
               fdiagram.vertex_attribute(v, 'y'), 
               fdiagram.vertex_attribute(v, 'z')] for v in fdiagram.vertices()]
    
    edges = [(vkey_to_idx[u], vkey_to_idx[v]) for u, v in fdiagram.edges()]
    
    with open(obj_path, 'w') as f:
        for p in points:
            f.write(f"v {p[0]} {p[1]} {p[2]}\n")
        for u, v in edges:
            f.write(f"l {u+1} {v+1}\n")

    from compas.geometry import Point, Line
    compas_points = [Point(*p) for p in points]
    compas_lines = [Line(points[u], points[v]) for u, v in edges]
    
    center_vkey = min(list(fdiagram.vertices()), key=lambda v: (fdiagram.vertex_attribute(v, 'x') - cx)**2 + (fdiagram.vertex_attribute(v, 'y') - cy)**2)
    corner = [cx, cy, fdiagram.vertex_attribute(center_vkey, 'z')]

    with open(json_path, 'w') as f:
        json_dump({
            "points": compas_points,
            "lines": compas_lines,
            "edge_point_indices": edges,
            "quadrant_corner": corner,
            "spokes": spokes
        }, f)
    print(f"Exported: {json_path}")

# ==============================================================================
# 4. Structural Analysis
# ==============================================================================

spokes = get_spokes_indices(form, cx, cy)

print("-" * 30)
print("MINIMUM THRUST ANALYSIS (Full Vault)")
print("-" * 30)
analysis_min = Analysis.create_minthrust_analysis(form, vault, printout=True)
analysis_min.optimiser.set_variables(['q']) # Optimize only force densities for stability
analysis_min.apply_selfweight()
analysis_min.apply_envelope()
analysis_min.set_up_optimiser()
analysis_min.run()
export_vault_geometry(analysis_min, spokes, "min_thrust_geometry.obj", "min_thrust_geometry.json", cx, cy)

print("-" * 30)
print("MAXIMUM THRUST ANALYSIS (Full Vault)")
print("-" * 30)
analysis_max = Analysis.create_maxthrust_analysis(form, vault, printout=True)
analysis_max.optimiser.set_variables(['q'])
analysis_max.optimiser.set_starting_point('current') # Use min_thrust solution as start
analysis_max.apply_selfweight()
analysis_max.apply_envelope()
analysis_max.set_up_optimiser()
analysis_max.run()
export_vault_geometry(analysis_max, spokes, "max_thrust_geometry.obj", "max_thrust_geometry.json", cx, cy)

# ==============================================================================
# 5. Corrugated Geometry Generation
# ==============================================================================

p_min = [[analysis_min.formdiagram.vertex_attribute(v, 'x'), 
          analysis_min.formdiagram.vertex_attribute(v, 'y'), 
          analysis_min.formdiagram.vertex_attribute(v, 'z')] for v in analysis_min.formdiagram.vertices()]

p_max = [[analysis_max.formdiagram.vertex_attribute(v, 'x'), 
          analysis_max.formdiagram.vertex_attribute(v, 'y'), 
          analysis_max.formdiagram.vertex_attribute(v, 'z')] for v in analysis_max.formdiagram.vertices()]

p_corrugated = [p[:] for p in p_max]
for i, spoke in enumerate(spokes):
    source = p_max if i % 2 == 0 else p_min
    for v_idx in spoke:
        p_corrugated[v_idx] = source[v_idx]

export_vault_geometry(analysis_max, spokes, "corrugated_geometry.obj", "corrugated_geometry.json", cx, cy)

from compas.data import json_load, json_dump
with open("corrugated_geometry.json", 'r') as f:
    data = json_load(f)
    from compas.geometry import Point
    data["points"] = [Point(*p) for p in p_corrugated]
with open("corrugated_geometry.json", 'w') as f:
    json_dump(data, f)

print("-" * 30)
print("Corrugated full vault geometry generated.")
print("-" * 30)
