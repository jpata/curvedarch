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

        from collections import defaultdict
        adj = defaultdict(list)
        for u, v in edge_indices_list:
            adj[u].append(v)
            adj[v].append(u)
            
        crown_v = min(range(len(compas_points)), key=lambda i: compas_points[i].distance_to_point(loaded_quadrant_corner))

        # Trace all 11 spokes from the crown
        polylines = []
        for start_neighbor in adj[crown_v]:
            path = [crown_v, start_neighbor]
            prev, curr = crown_v, start_neighbor
            for _ in range(50):
                next_v = None
                p_prev, p_curr = compas_points[prev], compas_points[curr]
                v_curr = Vector(p_curr.x - p_prev.x, p_curr.y - p_prev.y, 0)
                if v_curr.length < 1e-8: break
                v_curr.unitize()
                
                # Find best collinear neighbor (radial)
                best_dot = 0.5 
                for neighbor in adj[curr]:
                    if neighbor == prev: continue
                    p_next = compas_points[neighbor]
                    v_next = Vector(p_next.x - p_curr.x, p_next.y - p_curr.y, 0)
                    if v_next.length < 1e-8: continue
                    v_next.unitize()
                    dot = v_curr.dot(v_next)
                    if dot > best_dot:
                        best_dot, next_v = dot, neighbor
                
                if next_v is not None:
                    path.append(next_v)
                    prev, curr = curr, next_v
                else: break
            if len(path) > 1: polylines.append(Polyline([compas_points[i] for i in path]))

        # Sort and take exactly 11 longest (for 10 sections)
        polylines.sort(key=lambda p: len(p.points), reverse=True)
        polylines = polylines[:11]
        
        # Ensure all have the same number of points (truncate to most common or min)
        if polylines:
            counts = [len(p.points) for p in polylines]
            target_count = max(counts) # or min, but usually 11
            # If some are very long, they probably included hoops. 
            # In a quadrant fan with n_hoops=10, we expect 11 points.
            target_count = 11 
            
            for p in polylines:
                if len(p.points) > target_count:
                    p.points = p.points[:target_count]
                elif len(p.points) < target_count:
                    print(f"Warning: Polyline too short ({len(p.points)} < {target_count})")
        
        def get_angle(poly):
            p_end = poly.points[-1]
            return math.atan2(p_end.y - compas_points[crown_v].y, p_end.x - compas_points[crown_v].x)
        polylines.sort(key=get_angle)

        print(f"Final extracted radial polylines: {len(polylines)}")
        for i, p in enumerate(polylines):
            print(f"  Polyline {i}: {len(p.points)} points, ends at {p.points[-1]}")

        instructions_data = polylines
        return True
    except Exception as e:
        print(f"Error loading JSON: {e}")
        return False

def develop_strip_to_plane(poly1_3d, poly2_3d, start_point_on_plane, initial_unroll_vec):
    if len(poly1_3d.points) != len(poly2_3d.points) or len(poly1_3d.points) < 2: return None, []
    N = len(poly1_3d.points)
    z_plane = start_point_on_plane.z
    flat_vertices, flat_faces = [], []
    quad_distortions = [] 
    xy_plane_normal = Vector(0, 0, 1)
    
    P0_3d, Q0_3d = poly1_3d.points[0], poly2_3d.points[0]
    P_curr_flat = start_point_on_plane.copy()
    
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
    flat_vertices.append(P_curr_flat)
    flat_vertices.append(Q_curr_flat)
    
    P_prev_flat = None 

    for j in range(N - 1):
        P_curr_3d, P_next_3d = poly1_3d.points[j], poly1_3d.points[j+1]
        Q_curr_3d, Q_next_3d = poly2_3d.points[j], poly2_3d.points[j+1]

        d_pc_pn = P_curr_3d.distance_to_point(P_next_3d)
        d_qc_pn = Q_curr_3d.distance_to_point(P_next_3d)

        # Find P_next_flat
        if d_qc_pn < 1e-8:
            P_next_flat = Q_curr_flat.copy()
        elif d_pc_pn < 1e-8:
            P_next_flat = P_curr_flat.copy()
        elif P_curr_flat.distance_to_point(Q_curr_flat) < 1e-8:
             P_next_flat = P_curr_flat + unroll_dir_xy * d_pc_pn
        else:
            intersections_P = intersection_circle_circle_xy(((P_curr_flat, xy_plane_normal), d_pc_pn), ((Q_curr_flat, xy_plane_normal), d_qc_pn))
            if not intersections_P:
                # Precision near-miss handling
                dist_f = P_curr_flat.distance_to_point(Q_curr_flat)
                if abs(dist_f - (d_pc_pn + d_qc_pn)) < 1e-7 or dist_f > (d_pc_pn + d_qc_pn):
                    vec = Vector.from_start_end(P_curr_flat, Q_curr_flat).unitized()
                    P_next_flat = P_curr_flat + vec * d_pc_pn
                else:
                    print(f"FAILED intersection P at j={j}. Distances: {d_pc_pn}, {d_qc_pn}. P_curr_flat: {P_curr_flat}, Q_curr_flat: {Q_curr_flat}, Dist_flat: {dist_f}")
                    return None, []
            else:
                P_next_flat_chosen_xy = intersections_P[0]
                if len(intersections_P) > 1:
                    fwd_dir = unroll_dir_xy if P_prev_flat is None else Vector.from_start_end(P_prev_flat, P_curr_flat).unitized()
                    v0 = Vector(intersections_P[0][0]-P_curr_flat.x, intersections_P[0][1]-P_curr_flat.y, 0)
                    v1 = Vector(intersections_P[1][0]-P_curr_flat.x, intersections_P[1][1]-P_curr_flat.y, 0)
                    if v0.length > 1e-9: v0.unitize()
                    if v1.length > 1e-9: v1.unitize()
                    P_next_flat_chosen_xy = intersections_P[0] if v0.dot(fwd_dir) >= v1.dot(fwd_dir) else intersections_P[1]
                P_next_flat = Point(P_next_flat_chosen_xy[0], P_next_flat_chosen_xy[1], z_plane)

        d_qc_qn = Q_curr_3d.distance_to_point(Q_next_3d)
        d_pn_qn = P_next_3d.distance_to_point(Q_next_3d) 
        
        # Find Q_next_flat
        if d_pn_qn < 1e-8:
            Q_next_flat = P_next_flat.copy()
        elif d_qc_qn < 1e-8:
            Q_next_flat = Q_curr_flat.copy()
        elif Q_curr_flat.distance_to_point(P_next_flat) < 1e-8:
            fwd_dir = Vector.from_start_end(P_curr_flat, P_next_flat).unitized()
            Q_next_flat = P_next_flat + fwd_dir * d_qc_qn
        else:
            intersections_Q = intersection_circle_circle_xy(((Q_curr_flat, xy_plane_normal), d_qc_qn), ((P_next_flat, xy_plane_normal), d_pn_qn))
            if not intersections_Q:
                dist_f = Q_curr_flat.distance_to_point(P_next_flat)
                if abs(dist_f - (d_qc_qn + d_pn_qn)) < 1e-7 or dist_f > (d_qc_qn + d_pn_qn):
                    vec = Vector.from_start_end(Q_curr_flat, P_next_flat).unitized()
                    Q_next_flat = Q_curr_flat + vec * d_qc_qn
                else:
                    print(f"FAILED intersection Q at j={j}. Distances: {d_qc_qn}, {d_pn_qn}. Q_curr_flat: {Q_curr_flat}, P_next_flat: {P_next_flat}, Dist_flat: {dist_f}")
                    return None, [] 
            else:
                Q_next_flat_chosen_xy = intersections_Q[0]
                if len(intersections_Q) > 1:
                    rung_prev_dir = Vector.from_start_end(P_curr_flat, Q_curr_flat)
                    if rung_prev_dir.length < 1e-8:
                        rung_prev_dir = unroll_dir_xy.cross(xy_plane_normal)
                    rung_prev_dir.unitize()
                    r0 = Vector(intersections_Q[0][0]-P_next_flat.x, intersections_Q[0][1]-P_next_flat.y, 0)
                    r1 = Vector(intersections_Q[1][0]-P_next_flat.x, intersections_Q[1][1]-P_next_flat.y, 0)
                    if r0.length > 1e-9: r0.unitize()
                    if r1.length > 1e-9: r1.unitize()
                    Q_next_flat_chosen_xy = intersections_Q[0] if r0.dot(rung_prev_dir) >= r1.dot(rung_prev_dir) else intersections_Q[1]
                Q_next_flat = Point(Q_next_flat_chosen_xy[0], Q_next_flat_chosen_xy[1], z_plane)

        diag_3d_length = P_curr_3d.distance_to_point(Q_next_3d)
        diag_flat_length = P_curr_flat.distance_to_point(Q_next_flat) 
        quad_distortions.append(abs(diag_3d_length - diag_flat_length) / diag_3d_length if diag_3d_length > 1e-9 else 0.0)
        flat_vertices.extend([P_next_flat, Q_next_flat])
        idx_Pc, idx_Qc, idx_Pn, idx_Qn = 2*j, 2*j + 1, len(flat_vertices)-2, len(flat_vertices)-1
        flat_faces.append([idx_Pc, idx_Pn, idx_Qn, idx_Qc]) 
        P_prev_flat, P_curr_flat, Q_curr_flat = P_curr_flat, P_next_flat, Q_next_flat
    return Mesh.from_vertices_and_faces(flat_vertices, flat_faces), quad_distortions

def generate_flat_patterns():
    global instructions_data, FLAT_Z_OFFSET
    num_expected_strips = max(0, len(instructions_data) - 1)
    flat_meshes, spacing_x, spacing_y, strips_per_row = [], 15.0, 15.0, 5
    for i in range(num_expected_strips):
        poly1_3d, poly2_3d = instructions_data[i], instructions_data[i+1]
        col, row = i % strips_per_row, i // strips_per_row
        flat_start_point = Point(col * spacing_x, row * spacing_y, FLAT_Z_OFFSET)
        p0, p1 = poly1_3d.points[0], poly1_3d.points[1]
        initial_unroll_direction = Vector(p1.x - p0.x, p1.y - p0.y, 0)
        if initial_unroll_direction.length < 1e-6: initial_unroll_direction = Vector(1,0,0)
        else: initial_unroll_direction.unitize()
        mesh, distortions = develop_strip_to_plane(poly1_3d, poly2_3d, flat_start_point, initial_unroll_direction)
        if mesh:
            max_d = max(distortions) if distortions else 0
            avg_d = sum(distortions)/len(distortions) if distortions else 0
            print(f"Strip {i}: Max distortion = {max_d:.6f}, Avg distortion = {avg_d:.6f}")
            flat_meshes.append(mesh)
    return flat_meshes

def regenerate_all_geometry():
    global viewer, scene_objects_flat_strips, instructions_data, FLAT_Z_OFFSET, is_updating
    if is_updating or not viewer or not viewer.scene: return
    is_updating = True
    try:
        for obj in scene_objects_flat_strips: viewer.scene.remove(obj)
        scene_objects_flat_strips.clear()
        for i, mesh in enumerate(generate_flat_patterns()):
            if mesh: scene_objects_flat_strips.append(viewer.scene.add(mesh, name=f"FlatStrip_{i}", facecolor=Color(0.7, 0.9, 0.6, 0.7), show_edges=True))
        viewer.renderer.update() 
    finally: is_updating = False

from compas.data import json_dump
def save_flat_patterns_to_json(filepath):
    flat_meshes = [m for m in generate_flat_patterns() if m]
    try:
        with open(filepath, 'w') as f: json_dump({"meshes": flat_meshes}, f)
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
