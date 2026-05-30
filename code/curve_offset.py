#LIBGL_ALWAYS_SOFTWARE=1 xvfb-run -a uv run python3 code/curve_offset.py min_thrust_quadrant_geometry.json
import math
import time
import argparse
import sys
import os
import json 
import functools 

from PySide6 import QtCore, QtWidgets
from PySide6.QtWidgets import QWidget, QVBoxLayout, QGroupBox, QLabel, QDoubleSpinBox, QPushButton, QScrollArea, QDockWidget, QHBoxLayout, QSpacerItem, QSizePolicy


from compas.geometry import Point, Polyline, Translation, Rotation, Vector, Line, Sphere
from compas.geometry import intersection_circle_circle_xy
from compas.colors import Color
from compas.datastructures import Mesh
from compas.data import json_load
from compas_viewer import Viewer

# --- Configuration ---
FLAT_Z_OFFSET = -15.0 

# --- Global variables ---
viewer = None 
instructions_data = [] 
loaded_json_points = [] 
loaded_quadrant_corner = None 
is_updating = False

def load_instructions_from_json(filepath):
    global instructions_data, loaded_json_points, loaded_quadrant_corner
    try:
        data_dict = json_load(filepath)
        compas_points = data_dict.get("points")
        if not isinstance(compas_points, list): return False

        loaded_json_points = [Point(*p) if isinstance(p, (list, tuple)) else p for p in compas_points]
        compas_points = loaded_json_points
        edge_indices_list = data_dict.get("edge_point_indices")
        if not isinstance(edge_indices_list, list): return False

        corner_data = data_dict.get("quadrant_corner")
        loaded_quadrant_corner = Point(*corner_data) if corner_data else compas_points[0]
            
        print(f"Info: Using corner {loaded_quadrant_corner}")

        # Use explicitly provided spokes if available, otherwise fallback to BFS extraction
        spokes_indices = data_dict.get("spokes")
        if spokes_indices:
            print(f"Info: Using {len(spokes_indices)} explicitly provided spokes.")
            polylines = [Polyline([compas_points[idx] for idx in spoke]) for spoke in spokes_indices]
        else:
            print("Info: No spokes provided in JSON, falling back to topological extraction.")
            from collections import defaultdict
            adj = defaultdict(list)
            for u, v in edge_indices_list:
                adj[u].append(v)
                adj[v].append(u)
                
            crown_v = min(range(len(compas_points)), key=lambda i: compas_points[i].distance_to_point(loaded_quadrant_corner))

            # BFS to find hop distance from crown to identify hoops
            hop_dist = {crown_v: 0}
            from collections import deque
            q = deque([crown_v])
            while q:
                u = q.popleft()
                for v in adj[u]:
                    if v not in hop_dist:
                        hop_dist[v] = hop_dist[u] + 1
                        q.append(v)
            
            # Trace spokes by following increasing hop distance
            polylines = []
            start_neighbors = adj[crown_v]
            for sn in start_neighbors:
                path = [crown_v, sn]
                curr = sn
                for d in range(2, 100): 
                    next_vs = [v for v in adj[curr] if hop_dist.get(v) == d]
                    if not next_vs: break
                    
                    if len(next_vs) > 1:
                        p_prev, p_curr = compas_points[path[-2]], compas_points[curr]
                        v_curr = Vector.from_start_end(p_prev, p_curr)
                        if v_curr.length < 1e-8: break
                        v_curr.unitize()
                        best_v, best_dot = None, -2.0
                        for v in next_vs:
                            v_next = Vector.from_start_end(p_curr, compas_points[v])
                            if v_next.length < 1e-8: continue
                            v_next.unitize()
                            dot = v_curr.dot(v_next)
                            if dot > best_dot: best_dot, best_v = dot, v
                        if best_v is not None and best_dot > 0.0: curr = best_v
                        else: break
                    else: curr = next_vs[0]
                    path.append(curr)
                polylines.append(Polyline([compas_points[idx] for idx in path]))

        # Sort by angle to ensure strips are generated in order
        def get_angle(poly):
            p0, p1 = poly.points[0], poly.points[1]
            return math.atan2(p1.y - p0.y, p1.x - p0.x)
        polylines.sort(key=get_angle)

        # Resample all polylines to the same number of points
        TARGET_SAMPLES = 20 # Increased for better accuracy
        resampled_polylines = []
        for poly in polylines:
            # Simple linear resampling
            new_points = []
            # Calculate total length
            segments = []
            total_length = 0
            for i in range(len(poly.points)-1):
                d = poly.points[i].distance_to_point(poly.points[i+1])
                segments.append(d)
                total_length += d
            
            if total_length < 1e-6: continue
            
            for i in range(TARGET_SAMPLES):
                target_d = (i / (TARGET_SAMPLES - 1)) * total_length
                # Find which segment this distance falls into
                curr_d = 0
                found = False
                for j, seg_len in enumerate(segments):
                    if curr_d + seg_len >= target_d - 1e-8:
                        # Interpolate in this segment
                        t = (target_d - curr_d) / seg_len if seg_len > 1e-9 else 0
                        p_start = poly.points[j]
                        p_end = poly.points[j+1]
                        new_pt = p_start + (p_end - p_start) * t
                        new_points.append(new_pt)
                        found = True
                        break
                    curr_d += seg_len
                if not found:
                    new_points.append(poly.points[-1])
            resampled_polylines.append(Polyline(new_points))

        print(f"Final extracted and resampled radial polylines: {len(resampled_polylines)}")
        for i, p in enumerate(resampled_polylines):
            print(f"  Polyline {i}: {len(p.points)} points, ends at {p.points[-1]}")

        instructions_data = resampled_polylines
        return True
    except Exception as e:
        print(f"Error loading JSON: {e}")
        return False

def develop_strip_to_plane(poly1_3d, poly2_3d, start_point_on_plane, initial_unroll_vec):
    N1 = len(poly1_3d.points)
    N2 = len(poly2_3d.points)
    N = min(N1, N2)
    if N < 2: return None, []
    z_plane = start_point_on_plane.z
    flat_vertices, flat_faces = [], []
    quad_distortions = [] 
    xy_plane_normal = Vector(0, 0, 1)
    
    # We'll develop by triangulating each quad: (P_j, Q_j, P_j+1) and (P_j+1, Q_j, Q_j+1)
    P_curr_flat = start_point_on_plane.copy()
    
    P0_3d, Q0_3d = poly1_3d.points[0], poly2_3d.points[0]
    dist_p0q0 = P0_3d.distance_to_point(Q0_3d)
    
    unroll_dir_xy = Vector(initial_unroll_vec.x, initial_unroll_vec.y, 0)
    if unroll_dir_xy.length < 1e-6: unroll_dir_xy = Vector(1,0,0)
    else: unroll_dir_xy.unitize()

    if dist_p0q0 < 1e-6:
        Q_curr_flat = P_curr_flat.copy()
    else:
        rung_dir_xy = Vector(-unroll_dir_xy.y, unroll_dir_xy.x, 0) 
        Q_curr_flat = P_curr_flat + rung_dir_xy * dist_p0q0
    
    Q_curr_flat.z = z_plane
    flat_vertices.extend([P_curr_flat, Q_curr_flat])
    
    for j in range(N - 1):
        P_curr_3d, P_next_3d = poly1_3d.points[j], poly1_3d.points[j+1]
        Q_curr_3d, Q_next_3d = poly2_3d.points[j], poly2_3d.points[j+1]
        
        # 1. Find P_next_flat using triangle (P_curr, Q_curr, P_next)
        d_pc_pn = P_curr_3d.distance_to_point(P_next_3d)
        d_qc_pn = Q_curr_3d.distance_to_point(P_next_3d)
        
        d_pq = P_curr_flat.distance_to_point(Q_curr_flat)
        if d_pq < 1e-6:
            P_next_flat = P_curr_flat + unroll_dir_xy * d_pc_pn
        else:
            try:
                intersections = intersection_circle_circle_xy(((P_curr_flat, xy_plane_normal), d_pc_pn), ((Q_curr_flat, xy_plane_normal), d_qc_pn))
            except ValueError:
                intersections = []
                
            if not intersections:
                # Handle near-miss or tangency: place P_next on the line P_curr-Q_curr or forward
                P_next_flat = P_curr_flat + unroll_dir_xy * d_pc_pn
            else:
                if len(intersections) > 1:
                    v0 = Vector.from_start_end(P_curr_flat, Point(*intersections[0]))
                    v1 = Vector.from_start_end(P_curr_flat, Point(*intersections[1]))
                    P_next_flat = Point(*intersections[0]) if v0.dot(unroll_dir_xy) >= v1.dot(unroll_dir_xy) else Point(*intersections[1])
                else:
                    P_next_flat = Point(*intersections[0])
        
        P_next_flat.z = z_plane
        
        # 2. Find Q_next_flat using triangle (P_next, Q_curr, Q_next)
        d_pn_qn = P_next_3d.distance_to_point(Q_next_3d)
        d_qc_qn = Q_curr_3d.distance_to_point(Q_next_3d)
        
        d_pnqc = P_next_flat.distance_to_point(Q_curr_flat)
        if d_pnqc < 1e-6:
            Q_next_flat = P_next_flat + unroll_dir_xy * d_qc_qn # Fallback
        else:
            try:
                intersections = intersection_circle_circle_xy(((P_next_flat, xy_plane_normal), d_pn_qn), ((Q_curr_flat, xy_plane_normal), d_qc_qn))
            except ValueError:
                intersections = []
                
            if not intersections:
                Q_next_flat = Q_curr_flat + unroll_dir_xy * d_qc_qn
            else:
                if len(intersections) > 1:
                    rung_vec = Vector.from_start_end(P_curr_flat, Q_curr_flat)
                    if rung_vec.length < 1e-8: rung_vec = xy_plane_normal.cross(unroll_dir_xy)
                    r0 = Vector.from_start_end(P_next_flat, Point(*intersections[0]))
                    r1 = Vector.from_start_end(P_next_flat, Point(*intersections[1]))
                    Q_next_flat = Point(*intersections[0]) if r0.dot(rung_vec) >= r1.dot(rung_vec) else Point(*intersections[1])
                else:
                    Q_next_flat = Point(*intersections[0])
                    
        Q_next_flat.z = z_plane
        
        # Distortion check
        diag_3d = P_curr_3d.distance_to_point(Q_next_3d)
        diag_flat = P_curr_flat.distance_to_point(Q_next_flat)
        quad_distortions.append(abs(diag_3d - diag_flat) / diag_3d if diag_3d > 1e-9 else 0.0)
        
        flat_vertices.extend([P_next_flat, Q_next_flat])
        idx_Pc, idx_Qc, idx_Pn, idx_Qn = 2*j, 2*j + 1, 2*j + 2, 2*j + 3
        flat_faces.append([idx_Pc, idx_Pn, idx_Qn, idx_Qc])
        
        P_curr_flat, Q_curr_flat = P_next_flat, Q_next_flat
        # Update unroll direction for next step
        unroll_dir_xy = Vector.from_start_end(flat_vertices[2*j], P_curr_flat).unitized()

    return Mesh.from_vertices_and_faces(flat_vertices, flat_faces), quad_distortions

def unfold_mesh_strip(mesh, start_point, unroll_dir):
    from compas.geometry import Point, Vector
    from compas.datastructures import Mesh
    import math
    
    mesh.quads_to_triangles()
    
    z_min = min(mesh.vertex_attribute(v, 'z') for v in mesh.vertices())
    boundary_edges = mesh.edges_on_boundary()
    start_edge = None
    for u, v in boundary_edges:
        if abs(mesh.vertex_attribute(u, 'z') - z_min) < 1e-2 and abs(mesh.vertex_attribute(v, 'z') - z_min) < 1e-2:
            start_edge = (u, v)
            break
    if start_edge is None: start_edge = boundary_edges[0]
        
    start_face = mesh.halfedge_face((start_edge[0], start_edge[1]))
    if start_face is None:
        start_edge = (start_edge[1], start_edge[0])
        start_face = mesh.halfedge_face((start_edge[0], start_edge[1]))
        
    flat_vertices = {}
    u, v = start_edge
    p_u = Point(*mesh.vertex_coordinates(u))
    p_v = Point(*mesh.vertex_coordinates(v))
    d_uv = p_u.distance_to_point(p_v)
    
    flat_vertices[u] = start_point.copy()
    rung_dir = Vector(-unroll_dir.y, unroll_dir.x, 0)
    if rung_dir.length < 1e-6: rung_dir = Vector(1,0,0)
    rung_dir.unitize()
    
    flat_vertices[v] = start_point + rung_dir * d_uv
    flat_vertices[u].z = start_point.z
    flat_vertices[v].z = start_point.z
    
    from collections import deque
    queue = deque([start_face])
    visited_faces = set([start_face])
    
    while queue:
        fkey = queue.popleft()
        f_vkeys = mesh.face_vertices(fkey)
        
        placed_idx = [i for i, vk in enumerate(f_vkeys) if vk in flat_vertices]
        
        if len(placed_idx) == 2:
            if (placed_idx[1] - placed_idx[0]) % 3 == 1:
                p1_key, p2_key = f_vkeys[placed_idx[0]], f_vkeys[placed_idx[1]]
                p3_key = f_vkeys[[i for i in range(3) if i not in placed_idx][0]]
            else:
                p1_key, p2_key = f_vkeys[placed_idx[1]], f_vkeys[placed_idx[0]]
                p3_key = f_vkeys[[i for i in range(3) if i not in placed_idx][0]]
                
            p1_flat, p2_flat = flat_vertices[p1_key], flat_vertices[p2_key]
            p1_3d = Point(*mesh.vertex_coordinates(p1_key))
            p2_3d = Point(*mesh.vertex_coordinates(p2_key))
            p3_3d = Point(*mesh.vertex_coordinates(p3_key))
            
            L13 = p1_3d.distance_to_point(p3_3d)
            v12_3d = Vector.from_start_end(p1_3d, p2_3d)
            v13_3d = Vector.from_start_end(p1_3d, p3_3d)
            
            angle = v12_3d.angle(v13_3d) if v12_3d.length > 1e-8 and v13_3d.length > 1e-8 else 0.0
            
            v12_flat = Vector.from_start_end(p1_flat, p2_flat)
            if v12_flat.length > 1e-8:
                v12_flat.unitize()
                alpha = angle
                ca, sa = math.cos(alpha), math.sin(alpha)
                vx_new = v12_flat.x * ca - v12_flat.y * sa
                vy_new = v12_flat.x * sa + v12_flat.y * ca
                p3_flat = p1_flat + Vector(vx_new, vy_new, 0) * L13
            else:
                p3_flat = p1_flat.copy()
                
            p3_flat.z = start_point.z
            flat_vertices[p3_key] = p3_flat
            
        for nbr in mesh.face_neighbors(fkey):
            if nbr not in visited_faces:
                visited_faces.add(nbr)
                queue.append(nbr)
                
    new_mesh = Mesh()
    v_map = {}
    for fk in visited_faces:
        f_vkeys = mesh.face_vertices(fk)
        if all(vk in flat_vertices for vk in f_vkeys):
            f_add = []
            for vk in f_vkeys:
                pt = flat_vertices[vk]
                if vk not in v_map:
                    v_map[vk] = new_mesh.add_vertex(x=pt.x, y=pt.y, z=pt.z)
                f_add.append(v_map[vk])
            new_mesh.add_face(f_add)
        
    return new_mesh, [0.0]

def generate_flat_patterns():
    global FLAT_Z_OFFSET
    from compas.data import json_load
    try:
        data = json_load('corrugated_geometry.json')
        source_meshes = data.get('meshes', [])
    except:
        source_meshes = []
        
    num_strips = len(source_meshes)
    flat_meshes, spacing_x, spacing_y, strips_per_row = [], 15.0, 15.0, 8
    
    for i in range(num_strips):
        mesh_3d = source_meshes[i]
        col, row = i % strips_per_row, i // strips_per_row
        from compas.geometry import Point
        flat_start_point = Point(col * spacing_x, row * spacing_y, FLAT_Z_OFFSET)
        
        z_min = min(mesh_3d.vertex_attribute(v, 'z') for v in mesh_3d.vertices())
        boundary_edges = mesh_3d.edges_on_boundary()
        base_edge = None
        for u, v in boundary_edges:
            if abs(mesh_3d.vertex_attribute(u, 'z') - z_min) < 1e-2 and abs(mesh_3d.vertex_attribute(v, 'z') - z_min) < 1e-2:
                base_edge = (u, v)
                break
        if not base_edge: base_edge = boundary_edges[0]
        
        from compas.geometry import Vector
        p_u = Point(*mesh_3d.vertex_coordinates(base_edge[0]))
        p_v = Point(*mesh_3d.vertex_coordinates(base_edge[1]))
        mid_base = p_u + (p_v - p_u) * 0.5
        
        z_max = max(mesh_3d.vertex_attribute(v, 'z') for v in mesh_3d.vertices())
        crown_v = [v for v in mesh_3d.vertices() if abs(mesh_3d.vertex_attribute(v, 'z') - z_max) < 1e-2][0]
        p_crown = Point(*mesh_3d.vertex_coordinates(crown_v))
        
        unroll_dir = Vector.from_start_end(mid_base, p_crown)
        unroll_dir.z = 0
        if unroll_dir.length < 1e-6: unroll_dir = Vector(0,1,0)
        unroll_dir.unitize()
        
        flat_mesh, distortions = unfold_mesh_strip(mesh_3d, flat_start_point, unroll_dir)
        if flat_mesh:
            flat_meshes.append(flat_mesh)
            
    return flat_meshes

def regenerate_all_geometry():
    global viewer, scene_objects_flat_strips, instructions_data, FLAT_Z_OFFSET, is_updating
    if is_updating or not viewer or not viewer.scene: return
    is_updating = True
    try:
        for obj in scene_objects_flat_strips: viewer.scene.remove(obj)
        scene_objects_flat_strips.clear()
        for i, mesh in enumerate(generate_flat_patterns()):
            if mesh: 
                import matplotlib.pyplot as plt
                cmap = plt.cm.tab20
                rgba = cmap(i % 20)
                from compas.colors import Color
                c_color = Color(rgba[0], rgba[1], rgba[2], 0.8)
                scene_objects_flat_strips.append(viewer.scene.add(mesh, name=f"FlatStrip_{i}", facecolor=c_color, show_edges=True))
        viewer.renderer.update() 
    finally: is_updating = False

from compas.data import json_dump
def save_flat_patterns_to_json(filepath):
    global loaded_quadrant_corner
    flat_meshes = [m for m in generate_flat_patterns() if m]
    try:
        with open(filepath, 'w') as f: 
            json_dump({
                "meshes": flat_meshes,
                "quadrant_corner": loaded_quadrant_corner
            }, f)
        return True
    except Exception as e:
        print(f"Error saving: {e}")
        return False

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("json_file")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--output")
    args = parser.parse_args()
    if not load_instructions_from_json(args.json_file): sys.exit(1)
    is_headless = args.headless or os.environ.get("QT_QPA_PLATFORM") == "offscreen" or os.environ.get("DISPLAY") is None
    if args.output:
        save_flat_patterns_to_json(args.output)
        if is_headless: sys.exit(0)
    if is_headless: sys.exit(0)
    viewer = Viewer(width=1600, height=900, show_grid=True)
    regenerate_all_geometry()
    viewer.show()
