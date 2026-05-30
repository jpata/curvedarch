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
X_SPAN = [0.0, 20.0]
Y_SPAN = [0.0, 32.38]
HC_RISE = 2.0
THICKNESS = 2

# Discretisation
N_DISCRETISATION = 8

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
    """Identifies radial spokes by tracing from boundary vertices back to the crown with geometric bias."""
    vkeys = list(form.vertices())
    center_vkey = min(vkeys, key=lambda v: (form.vertex_attribute(v, 'x') - cx)**2 + (form.vertex_attribute(v, 'y') - cy)**2)
    from collections import deque
    hop_dist = {center_vkey: 0}
    q = deque([center_vkey])
    while q:
        u = q.popleft()
        for v in form.vertex_neighbors(u):
            if v not in hop_dist:
                hop_dist[v] = hop_dist[u] + 1
                q.append(v)
    boundary_vs = form.vertices_on_boundary()
    spokes = []
    vkey_to_idx = {vkey: i for i, vkey in enumerate(vkeys)}
    for b_v in boundary_vs:
        path = [b_v]
        curr = b_v
        while curr != center_vkey:
            nbrs = form.vertex_neighbors(curr)
            def score(v):
                d = hop_dist.get(v, 999)
                p1 = form.vertex_coordinates(center_vkey)
                p2 = form.vertex_coordinates(b_v)
                p3 = form.vertex_coordinates(v)
                v1 = Vector.from_start_end(p1, p2)
                v2 = Vector.from_start_end(p1, p3)
                dist = v2.length * math.sin(v1.angle(v2)) if v1.length > 1e-8 and v2.length > 1e-8 else 0
                return (d, dist)
            best_nbr = min(nbrs, key=score)
            path.append(best_nbr)
            curr = best_nbr
        path.reverse()
        spokes.append([vkey_to_idx[v] for v in path])
    unique_spokes = []
    seen = set()
    for s in spokes:
        t = tuple(s)
        if t not in seen:
            unique_spokes.append(s)
            seen.add(t)
    unique_spokes.sort(key=lambda s: math.atan2(form.vertex_attribute(vkeys[s[-1]], 'y') - cy, form.vertex_attribute(vkeys[s[-1]], 'x') - cx))
    return unique_spokes

def export_vault_geometry(analysis, spokes, obj_path, json_path, cx, cy, custom_points=None):
    """Exports optimized geometry using Topological Corner Growth sectoring."""
    fdiagram = analysis.formdiagram
    vkey_to_idx = {vkey: i for i, vkey in enumerate(fdiagram.vertices())}
    
    if custom_points is not None:
        points = custom_points
    else:
        points = [[fdiagram.vertex_attribute(v, 'x'), fdiagram.vertex_attribute(v, 'y'), fdiagram.vertex_attribute(v, 'z')] for v in fdiagram.vertices()]
    
    edges = [(vkey_to_idx[u], vkey_to_idx[v]) for u, v in fdiagram.edges()]
    
    # 1. Identify Corners
    corners = {
        0: (X_SPAN[0], Y_SPAN[0]),
        1: (X_SPAN[1], Y_SPAN[0]),
        2: (X_SPAN[1], Y_SPAN[1]),
        3: (X_SPAN[0], Y_SPAN[1])
    }
    corner_vkeys = {}
    for v in fdiagram.vertices():
        x, y = fdiagram.vertex_coordinates(v)[:2]
        for q, (cx_c, cy_c) in corners.items():
            if abs(x - cx_c) < 1e-6 and abs(y - cy_c) < 1e-6:
                corner_vkeys[q] = v

    # 2. Extract root faces and assign sector IDs
    root_faces = []
    for q in range(4):
        c_vkey = corner_vkeys[q]
        c_faces = fdiagram.vertex_faces(c_vkey)
        num_sectors_per_quad = len(c_faces)
        cx_c, cy_c = corners[q]
        def face_angle(fkey):
            pts = [fdiagram.vertex_coordinates(v) for v in fdiagram.face_vertices(fkey)]
            mx = sum(p[0] for p in pts) / len(pts)
            my = sum(p[1] for p in pts) / len(pts)
            return math.atan2(my - cy_c, mx - cx_c)
        c_faces.sort(key=face_angle)
        
        for i, fkey in enumerate(c_faces):
            sector_id = q * len(c_faces) + i
            root_faces.append((fkey, sector_id))

    # 3. Propagate sector IDs topologically
    from collections import deque
    face_to_sector = {}
    q_bfs = deque()
    for fkey, sid in root_faces:
        face_to_sector[fkey] = sid
        q_bfs.append(fkey)
        
    while q_bfs:
        curr_f = q_bfs.popleft()
        sid = face_to_sector[curr_f]
        for nbr_f in fdiagram.face_neighbors(curr_f):
            if nbr_f not in face_to_sector:
                face_to_sector[nbr_f] = sid
                q_bfs.append(nbr_f)
                
    # 4. Create strip meshes
    from compas.datastructures import Mesh
    from collections import defaultdict
    sector_to_faces = defaultdict(list)
    for fkey, sid in face_to_sector.items():
        sector_to_faces[sid].append(fkey)
        
    meshes = []
    num_total_sectors = 4 * len(c_faces)
    for i in range(num_total_sectors):
        fkeys = sector_to_faces.get(i, [])
        strip_mesh = Mesh()
        v_map = {}
        for fk in fkeys:
            for v in fdiagram.face_vertices(fk):
                if v not in v_map:
                    p = points[vkey_to_idx[v]]
                    v_map[v] = strip_mesh.add_vertex(x=p[0], y=p[1], z=p[2])
            strip_mesh.add_face([v_map[v] for v in fdiagram.face_vertices(fk)])
        meshes.append(strip_mesh)
        
    with open(obj_path, 'w') as f:
        for p in points: f.write(f"v {p[0]} {p[1]} {p[2]}\n")
        for u, v in edges: f.write(f"l {u+1} {v+1}\n")

    from compas.geometry import Point, Line
    compas_points = [Point(*p) for p in points]
    compas_lines = [Line(points[u], points[v]) for u, v in edges]
    center_vkey = min(list(fdiagram.vertices()), key=lambda v: (fdiagram.vertex_attribute(v, 'x') - cx)**2 + (fdiagram.vertex_attribute(v, 'y') - cy)**2)
    corner = [cx, cy, fdiagram.vertex_attribute(center_vkey, 'z')]

    # Export full vault
    with open(json_path, 'w') as f:
        import json
        from compas.data import json_dump
        json_dump({
            "points": compas_points,
            "lines": compas_lines,
            "edge_point_indices": edges,
            "quadrant_corner": corner,
            "spokes": spokes,
            "meshes": meshes
        }, f)
    print(f"Exported: {json_path}")

    # NEW: Export quadrant-only JSON
    quad_path = json_path.replace("_geometry.json", "_quadrant_geometry.json")
    quad_meshes = meshes[:num_sectors_per_quad]
    
    # Collect points and edges for the quadrant
    quad_vkeys = set()
    for sid in range(num_sectors_per_quad):
        for fk in sector_to_faces[sid]:
            for v in fdiagram.face_vertices(fk):
                quad_vkeys.add(v)
    
    quad_points_list = []
    old_to_new_quad = {}
    for v in quad_vkeys:
        old_idx = vkey_to_idx[v]
        old_to_new_quad[old_idx] = len(quad_points_list)
        quad_points_list.append(points[old_idx])
        
    quad_edges = []
    for u, v in fdiagram.edges():
        if u in quad_vkeys and v in quad_vkeys:
            quad_edges.append((old_to_new_quad[vkey_to_idx[u]], old_to_new_quad[vkey_to_idx[v]]))
            
    quad_spokes = []
    for s in spokes:
        if any(idx in old_to_new_quad for idx in s):
            quad_spokes.append([old_to_new_quad[idx] for idx in s if idx in old_to_new_quad])

    with open(quad_path, 'w') as f:
        json_dump({
            "points": [Point(*p) for p in quad_points_list],
            "edge_point_indices": quad_edges,
            "quadrant_corner": corner,
            "spokes": quad_spokes,
            "meshes": quad_meshes
        }, f)
    print(f"Exported Quadrant: {quad_path}")


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

export_vault_geometry(analysis_max, spokes, "corrugated_geometry.obj", "corrugated_geometry.json", cx, cy, custom_points=p_corrugated)

print("-" * 30)
print("Corrugated full vault geometry generated.")
print("-" * 30)
