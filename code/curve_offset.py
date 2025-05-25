import math
import time
import argparse
import sys
import json # For loading JSON data
import functools # For partial function application in signal connections

from PySide6 import QtCore, QtWidgets
from PySide6.QtWidgets import QWidget, QVBoxLayout, QGroupBox, QLabel, QDoubleSpinBox, QPushButton, QScrollArea, QDockWidget, QHBoxLayout, QSpacerItem, QSizePolicy


from compas.geometry import Point, Polyline, Translation, Rotation, Vector, Line, Sphere
from compas.geometry import intersection_circle_circle_xy
from compas.colors import Color
from compas.datastructures import Mesh
from compas_viewer import Viewer

# --- Configuration ---
# NUM_CATENARY_POINTS = 2 # No longer used as polylines are directly from JSON
# DEFAULT_CATENARY_HEIGHT = 0.01 # No longer used as polylines are directly from JSON
FLAT_Z_OFFSET = -15.0 # Initial FLAT_Z_OFFSET, will be controllable by UI

# --- Global variables for UI interaction and scene management ---
viewer = None # Will be initialized later

instructions_data = [] # Will be populated from JSON: list of [Polyline_span, height]
loaded_json_points = [] # Will store Point objects loaded from JSON
loaded_quadrant_corner = None # Will store the calculated quadrant corner Point

generated_polylines_3d = [] # Will store the actual 3D polylines (now directly from instructions_data)
scene_objects_3d_surfaces = [] # Will store SceneObject instances
scene_objects_flat_strips = [] # Will store SceneObject instances

# Track update state to prevent recursive calls
is_updating = False

# --- JSON Loading Function ---
def load_instructions_from_json(filepath):
    """
    Loads catenary span definitions from a JSON file.
    The JSON should contain:
    - A "points" key: a list of 3D point coordinates (e.g., [[x1,y1,z1], [x2,y2,z2], ...]).
    - An "edge_point_indices" key: a list of pairs of indices, where each pair
      refers to indices in the "points" list. Each pair defines the start and end
      points of a Polyline representing a catenary span.
    """
    global instructions_data, loaded_json_points, loaded_quadrant_corner
    try:
        with open(filepath, 'r') as f:
            data_dict = json.load(f)

        # 1. Load and deserialize points
        points_coords_list = data_dict.get("points")
        if not isinstance(points_coords_list, list):
            print(f"Error: JSON file {filepath} must contain a 'points' key with a list of point coordinates.")
            return False

        compas_points = []
        for i, coords in enumerate(points_coords_list):
            try:
                if not (isinstance(coords, list) and len(coords) >= 3): # Allow for more than 3, but use first 3
                    raise ValueError(f"Point {i} coordinates are not a list of at least 3 numbers.")
                compas_points.append(Point(coords[0], coords[1], coords[2]))
            except Exception as e:
                print(f"Error: Point {i} in JSON (data: {coords}) could not be deserialized: {e}. Aborting load.")
                return False
        loaded_json_points = compas_points[:] # Store a copy of the loaded points

        # 2. Load edge_point_indices
        edge_indices_list = data_dict.get("edge_point_indices")
        if not isinstance(edge_indices_list, list):
            print(f"Error: JSON file {filepath} must contain an 'edge_point_indices' key with a list of index pairs.")
            return False

        # Define the "quadrant corner" based on all loaded points.
        # This will be used to filter edges that are not "radiating outwards" from this corner.
        # The corner is defined as the point with the minimum x, y, and z coordinates from the set of points.
        quadrant_corner = compas_points[0]
        if compas_points: # Only define if there are points
            max_x_coord = max(p.x for p in compas_points)
            max_y_coord = max(p.y for p in compas_points)
            min_z_coord = min(p.z for p in compas_points)
            quadrant_corner = Point(max_x_coord, max_y_coord, min_z_coord)
            print(f"Info: Using point {quadrant_corner.x}, {quadrant_corner.y}, {quadrant_corner.z} as the 'quadrant corner' for edge filtering.")
        loaded_quadrant_corner = quadrant_corner # Store the calculated quadrant corner
        # If compas_points is empty, quadrant_corner remains None.
        # Subsequent logic for edge processing will likely fail due to IndexError if edge_indices_list is not empty,
        # which is handled by the try-except block for each edge.

        temp_instructions = []
        for i, edge_pair in enumerate(edge_indices_list):
            try:
                if not (isinstance(edge_pair, list) and len(edge_pair) == 2):
                    raise ValueError(f"Edge {i} definition (data: {edge_pair}) is not a list of 2 indices.")

                idx_start, idx_end = edge_pair

                if not (isinstance(idx_start, int) and isinstance(idx_end, int)):
                    raise ValueError(f"Edge {i} indices (data: {edge_pair}) must be integers.")

                # This check also handles the case where compas_points might be empty.
                # If len(compas_points) is 0, any valid index (e.g., 0) will fail `0 < 0`.
                if not (0 <= idx_start < len(compas_points) and 0 <= idx_end < len(compas_points)):
                    raise IndexError(f"Edge {i} indices [{idx_start}, {idx_end}] are out of bounds for points list (size {len(compas_points)}).")

                p_start = compas_points[idx_start]
                p_end = compas_points[idx_end]

                if p_start == p_end:
                    print(f"Warning: Edge {i} (indices [{idx_start}, {idx_end}]) connects a point to itself. Skipping this edge.")
                    continue

                # Filter: Check if the edge is radiating outwards from the defined quadrant corner.
                # This filter is applied if a quadrant_corner was defined (i.e., compas_points was not empty).
                # If compas_points was non-empty, quadrant_corner is a Point object.
                if quadrant_corner:
                    # --- Original distance-based filter (commented out) ---
                    # dist_start_to_corner = quadrant_corner.distance_to_point(p_start)
                    # dist_end_to_corner = quadrant_corner.distance_to_point(p_end)
                    # # If the end point of the edge is not strictly further from the corner than the start point, skip it.
                    # if dist_end_to_corner <= dist_start_to_corner:
                    #     print(f"Info: Edge {i} (indices [{idx_start}, {idx_end}]) from point {idx_start} to {idx_end} "
                    #           f"is not strictly radiating outwards from quadrant corner {quadrant_corner.x}, {quadrant_corner.y}, {quadrant_corner.z}. "
                    #           f"(Distances: start={dist_start_to_corner:.2f}, end={dist_end_to_corner:.2f}). Skipping edge.")
                    #     continue

                    # --- New filter: XY distance from quadrant_corner's Z-axis to the infinite line of the edge ---
                    qc_xy_tuple = (quadrant_corner.x, quadrant_corner.y)
                    p_start_xy_tuple = (p_start.x, p_start.y)
                    p_end_xy_tuple = (p_end.x, p_end.y)

                    x0, y0 = qc_xy_tuple      # Point (quadrant_corner XY projection)
                    x1, y1 = p_start_xy_tuple # Line point 1 (edge start XY projection)
                    x2, y2 = p_end_xy_tuple   # Line point 2 (edge end XY projection)

                    dx_line = x2 - x1
                    dy_line = y2 - y1

                    line_length_sq = dx_line * dx_line + dy_line * dy_line
                    dist_xy_line_to_corner_axis = 0.0

                    # Check if the projected start and end points of the edge are coincident in XY
                    if line_length_sq < 1e-12: # Using a small tolerance for floating point comparison
                        # If coincident, distance is point-to-point from quadrant_corner_xy to p_start_xy
                        dist_xy_line_to_corner_axis = math.sqrt((x0 - x1)**2 + (y0 - y1)**2)
                    else:
                        # Distance from point (x0,y0) to the infinite line passing through (x1,y1) and (x2,y2)
                        # Formula: |(x2-x1)(y1-y0) - (x1-x0)(y2-y1)| / sqrt((x2-x1)^2 + (y2-y1)^2)
                        numerator = abs(dx_line * (y1 - y0) - (x1 - x0) * dy_line)
                        dist_xy_line_to_corner_axis = numerator / math.sqrt(line_length_sq)

                    # Define a threshold for this distance. Tune this value based on your model's scale.
                    ALIGNMENT_THRESHOLD_XY = 0.1 # Example: Max allowed XY distance to corner's Z-axis

                    if dist_xy_line_to_corner_axis > ALIGNMENT_THRESHOLD_XY:
                        print(f"Info: Edge {i} (indices [{idx_start}, {idx_end}]) from point {idx_start} to {idx_end} "
                              f"XY projection of infinite line (dist to corner axis: {dist_xy_line_to_corner_axis:.2f}) "
                              f"is too far from quadrant corner's Z-axis (threshold: {ALIGNMENT_THRESHOLD_XY:.2f}). Skipping edge.")
                        continue

                span_polyline = Polyline([p_start, p_end])
                temp_instructions.append(span_polyline) # Store the polyline directly
            except Exception as e:
                print(f"Warning: Edge {i} (data: {edge_pair}) in JSON could not be processed: {e}. Skipping edge.")
                continue

        instructions_data = temp_instructions
        return True
    except FileNotFoundError:
        print(f"Error: JSON file {filepath} not found.")
        return False
    except json.JSONDecodeError as e:
        print(f"Error: Could not decode JSON from file {filepath}: {e}")
        return False
    except Exception as e:
        print(f"An unexpected error occurred while loading or parsing JSON file {filepath}: {e}")
        return False

def develop_strip_to_plane(poly1_3d, poly2_3d, start_point_on_plane, initial_unroll_vec):
    """
    Develops a 3D strip (defined by two polylines) onto a 2D plane.
    Returns the flattened mesh and a list of distortion values for each quad.
    """
    if len(poly1_3d.points) != len(poly2_3d.points) or len(poly1_3d.points) < 2:
        return None, []
        
    N = len(poly1_3d.points)
    z_plane = start_point_on_plane.z
    flat_vertices, flat_faces = [], []
    quad_distortions = [] # To store distortion for each quad
    xy_plane_normal = Vector(0, 0, 1)

    P0_3d, Q0_3d = poly1_3d.points[0], poly2_3d.points[0]
    P_curr_flat = start_point_on_plane.copy()
    flat_vertices.append(P_curr_flat)

    dist_p0q0 = P0_3d.distance_to_point(Q0_3d)
    unroll_dir_xy = Vector(initial_unroll_vec.x, initial_unroll_vec.y, 0)
    if unroll_dir_xy.length < 1e-6: unroll_dir_xy = Vector(1,0,0)
    else: unroll_dir_xy.unitize()
    
    rung_dir_xy = Vector(-unroll_dir_xy.y, unroll_dir_xy.x, 0) 
    Q_curr_flat = P_curr_flat + rung_dir_xy * dist_p0q0
    Q_curr_flat.z = z_plane
    flat_vertices.append(Q_curr_flat)

    P_prev_flat = None 

    for j in range(N - 1):
        P_curr_3d, P_next_3d = poly1_3d.points[j], poly1_3d.points[j+1]
        Q_curr_3d, Q_next_3d = poly2_3d.points[j], poly2_3d.points[j+1]

        d_pc_pn = P_curr_3d.distance_to_point(P_next_3d) 
        d_qc_pn = Q_curr_3d.distance_to_point(P_next_3d) 

        #must use xy_plane_normal here
        intersections_P = intersection_circle_circle_xy(
            ((P_curr_flat, xy_plane_normal), d_pc_pn), 
            ((Q_curr_flat, xy_plane_normal), d_qc_pn)
        )
        if not intersections_P: return None, [] 
        
        P_next_flat_chosen_xy = intersections_P[0]
        if len(intersections_P) > 1:
            fwd_dir = unroll_dir_xy if P_prev_flat is None else Vector.from_start_end(P_prev_flat, P_curr_flat).unitized()
            v0 = Vector(intersections_P[0][0]-P_curr_flat.x, intersections_P[0][1]-P_curr_flat.y, 0).unitized()
            v1 = Vector(intersections_P[1][0]-P_curr_flat.x, intersections_P[1][1]-P_curr_flat.y, 0).unitized()
            P_next_flat_chosen_xy = intersections_P[0] if v0.dot(fwd_dir) >= v1.dot(fwd_dir) else intersections_P[1]
        P_next_flat = Point(P_next_flat_chosen_xy[0], P_next_flat_chosen_xy[1], z_plane)

        d_qc_qn = Q_curr_3d.distance_to_point(Q_next_3d) 
        d_pn_qn = P_next_3d.distance_to_point(Q_next_3d) 

        #must use xy_plane_normal here
        intersections_Q = intersection_circle_circle_xy(
            ((Q_curr_flat, xy_plane_normal), d_qc_qn), 
            ((P_next_flat, xy_plane_normal), d_pn_qn)
        )
        if not intersections_Q: return None, [] 

        Q_next_flat_chosen_xy = intersections_Q[0]
        if len(intersections_Q) > 1:
            rung_prev_dir = Vector.from_start_end(P_curr_flat, Q_curr_flat).unitized()
            r0 = Vector(intersections_Q[0][0]-P_next_flat.x, intersections_Q[0][1]-P_next_flat.y, 0).unitized()
            r1 = Vector(intersections_Q[1][0]-P_next_flat.x, intersections_Q[1][1]-P_next_flat.y, 0).unitized()
            Q_next_flat_chosen_xy = intersections_Q[0] if r0.dot(rung_prev_dir) >= r1.dot(rung_prev_dir) else intersections_Q[1]
        Q_next_flat = Point(Q_next_flat_chosen_xy[0], Q_next_flat_chosen_xy[1], z_plane)

        # --- Measure distortion for the current quad ---
        diag_3d_length = P_curr_3d.distance_to_point(Q_next_3d)
        diag_flat_length = P_curr_flat.distance_to_point(Q_next_flat) 

        current_quad_distortion = 0.0
        if diag_3d_length > 1e-9: # Avoid division by zero
            current_quad_distortion = abs(diag_3d_length - diag_flat_length) / diag_3d_length
        elif diag_flat_length > 1e-9: # 3D diagonal is zero, but flat is not (implies large distortion)
            current_quad_distortion = 10.0 # Assign a large distortion value (e.g., 1000%)
        # Else (both are zero or very small), distortion remains 0.0
        quad_distortions.append(current_quad_distortion)
        # --- End Measure distortion ---

        idx_Pc, idx_Qc = 2*j, 2*j + 1
        if idx_Pc >= len(flat_vertices) or idx_Qc >= len(flat_vertices) : return None, []

        flat_vertices.extend([P_next_flat, Q_next_flat])
        idx_Pn, idx_Qn = len(flat_vertices)-2, len(flat_vertices)-1

        flat_faces.append([idx_Pc, idx_Pn, idx_Qn, idx_Qc]) 

        P_prev_flat = P_curr_flat 
        P_curr_flat = P_next_flat
        Q_curr_flat = Q_next_flat
    
    if not flat_vertices or not flat_faces: return None, []

    # Optional: Print average distortion for the strip (can be commented out)
    # if quad_distortions:
    #     avg_dist = sum(quad_distortions) / len(quad_distortions)
    #     print(f"    Strip avg quad distortion: {avg_dist:.4f} (min: {min(quad_distortions):.4f}, max: {max(quad_distortions):.4f})")

    return Mesh.from_vertices_and_faces(flat_vertices, flat_faces), quad_distortions

# --- UI Update and Geometry Regeneration Functions ---
def update_flat_z_offset_param(control_widget, *args):
    """Updates the global FLAT_Z_OFFSET and regenerates geometry."""
    global FLAT_Z_OFFSET, is_updating
    
    if is_updating:
        return
        
    print(f"\n=== FLAT Z OFFSET UPDATE TRIGGERED ===")
    print(f"Control widget value: {control_widget.value()}")
    
    try:
        old_value = FLAT_Z_OFFSET
        FLAT_Z_OFFSET = float(control_widget.value())
        print(f"Changed FLAT_Z_OFFSET from {old_value} to {FLAT_Z_OFFSET}")
        
        QtCore.QTimer.singleShot(100, regenerate_all_geometry)
        
    except ValueError as e:
        print(f"ValueError in update_flat_z_offset_param: {e}")
    except Exception as e:
        print(f"Unexpected error in update_flat_z_offset_param: {e}")

def regenerate_all_geometry():
    """Calculates new geometry and updates existing meshes or creates new ones."""
    global viewer, generated_polylines_3d, scene_objects_3d_surfaces, scene_objects_flat_strips
    global instructions_data, FLAT_Z_OFFSET, is_updating

    if is_updating: return
    is_updating = True
    
    print_prefix = "  [RegenGeom] " # Define print_prefix for commented out debug lines
    # print(f"\n{print_prefix}{'='*50}\n{print_prefix}STARTING GEOMETRY REGENERATION\n{print_prefix}{'='*50}")

    try:
        if viewer is None or viewer.scene is None:
            # print(f"{print_prefix}ERROR: Viewer or Viewer.scene is None!")
            is_updating = False
            return

        generated_polylines_3d.clear()

        # 1. Generate 3D Polylines
        # print(f"{print_prefix}--- Generating 3D Polylines ---")
        # instructions_data is now a list of Polyline objects (the catenaries themselves)
        for catenary_polyline in instructions_data:
            if catenary_polyline and len(catenary_polyline.points) >= 2:
                # The polylines from JSON are already in world coordinates and represent the catenary
                generated_polylines_3d.append(catenary_polyline.copy()) # Use a copy
            else:
                # print(f"{print_prefix}  SKIPPED 3D Polyline: Invalid or too few points.")
                generated_polylines_3d.append(None)

        num_expected_strips = max(0, len(generated_polylines_3d) - 1)
        
        # --- 2. Update or Create 3D Surface Strips ---
        # print(f"\n{print_prefix}--- Updating/Creating 3D Surfaces ({num_expected_strips} expected) ---")
        new_scene_objs_3d_surfaces_temp = [None] * num_expected_strips 
        for i in range(2):
            poly1_3d = generated_polylines_3d[i]
            poly2_3d = generated_polylines_3d[i+1]

            if not poly1_3d or not poly2_3d or \
               len(poly1_3d.points) != len(poly2_3d.points) or len(poly1_3d.points) < 2:
                # print(f"{print_prefix}  SKIPPED 3D Surface {i}: Invalid polylines.")
                continue
            
            try:
                strip_vertices_3d = poly1_3d.points + poly2_3d.points # Combine points from both polylines
                strip_faces_3d = []
                offset_3d = len(poly1_3d.points) # Offset to index into poly2_3d points when combined
                for j_seg in range(len(poly1_3d.points) - 1):
                    # Create quad faces linking corresponding segments of the two polylines
                    strip_faces_3d.append([j_seg, j_seg + 1, offset_3d + j_seg + 1, offset_3d + j_seg])
                
                new_mesh_data_3d = Mesh.from_vertices_and_faces(strip_vertices_3d, strip_faces_3d)

                if i < len(scene_objects_3d_surfaces) and scene_objects_3d_surfaces[i] is not None:
                    scene_obj = scene_objects_3d_surfaces[i]
                    mesh_to_update = scene_obj.item # Get the existing Mesh object
                    
                    mesh_to_update.clear() # Clear its current geometry

                    # Repopulate mesh_to_update with geometry from new_mesh_data_3d
                    vkey_map = {} # To map vertex keys from new_mesh_data_3d to mesh_to_update
                    for old_vkey in new_mesh_data_3d.vertices():
                        x, y, z = new_mesh_data_3d.vertex_coordinates(old_vkey)
                        new_vkey = mesh_to_update.add_vertex(x=x, y=y, z=z)
                        vkey_map[old_vkey] = new_vkey
                    
                    for old_fkey in new_mesh_data_3d.faces():
                        face_vkeys_in_new_data = new_mesh_data_3d.face_vertices(old_fkey)
                        # Map old vertex keys to new vertex keys for the face
                        face_vkeys_in_mesh_to_update = [vkey_map[ovk] for ovk in face_vkeys_in_new_data]
                        mesh_to_update.add_face(face_vkeys_in_mesh_to_update)
                    
                    scene_obj.update(update_data=True) # Tell the scene object to re-render with updated item
                    new_scene_objs_3d_surfaces_temp[i] = scene_obj
                    scene_objects_3d_surfaces[i] = None # Mark as processed
                else:
                    scene_obj = viewer.scene.add(new_mesh_data_3d, name=f"3DSurface_{i}",
                                                 facecolor=Color(0.6, 0.7, 0.9, 0.7), show_edges=True,
                                                 linecolor=Color(0.1, 0.1, 0.1, 0.9), linewidth=1.0)
                    new_scene_objs_3d_surfaces_temp[i] = scene_obj
            except Exception as e:
                print(f"{print_prefix}  ERROR creating/updating 3D surface {i}: {e}")
                # import traceback; traceback.print_exc()


        for old_scene_obj in scene_objects_3d_surfaces: # Remove any old objects not processed
            if old_scene_obj is not None: viewer.scene.remove(old_scene_obj)
        scene_objects_3d_surfaces[:] = new_scene_objs_3d_surfaces_temp

        # --- 3. Update or Create Flattened Strips with Distortion Coloring ---
        # print(f"\n{print_prefix}--- Updating/Creating Flattened Strips ({num_expected_strips} expected) ---")
        new_scene_objs_flat_strips_temp = [None] * num_expected_strips
        for i in range(num_expected_strips):
            poly1_3d = generated_polylines_3d[i]
            poly2_3d = generated_polylines_3d[i+1]

            if not poly1_3d or not poly2_3d or \
               len(poly1_3d.points) != len(poly2_3d.points) or len(poly1_3d.points) < 2:
                # print(f"{print_prefix}  SKIPPED Flat Strip {i}: Invalid polylines.")
                continue
            
            try:
                # Determine the starting point for the flattened strip.
                # It will be placed directly underneath the start of the 3D strip (poly1_3d).
                if poly1_3d and poly1_3d.points:
                    start_x_3d = poly1_3d.points[0].x
                    start_y_3d = poly1_3d.points[0].y
                    flat_start_point = Point(start_x_3d, start_y_3d, FLAT_Z_OFFSET)
                else:
                    # Fallback if poly1_3d is invalid or empty, place with a simple offset based on index
                    # This case should ideally not be reached if poly1_3d was valid for 3D strip generation.
                    print(f"{print_prefix} Warning: poly1_3d invalid or empty for flat strip {i}. Using default placement.")
                    flat_start_point = Point(i * 15.0, 0, FLAT_Z_OFFSET) # Default fallback placement

                initial_unroll_direction = Vector(0,0,0)
                if len(poly1_3d.points) >= 2:
                    p0, p1 = poly1_3d.points[0], poly1_3d.points[1]
                    # Use the vector from the first segment of poly1_3d for initial unroll direction.
                    # develop_strip_to_plane will project this to XY.
                    initial_unroll_direction = Vector.from_start_end(p0,p1) 
                if initial_unroll_direction.length < 1e-6: 
                    initial_unroll_direction = Vector(1,0,0) # Default to X-axis if vector is zero length
                
                new_flattened_mesh_data, quad_distortions = develop_strip_to_plane(poly1_3d, poly2_3d, flat_start_point, initial_unroll_direction)
                
                if new_flattened_mesh_data:
                    face_colors = {}
                    default_face_color = Color(0.7, 0.9, 0.6, 0.7) 

                    if quad_distortions:
                        min_d = min(quad_distortions) if quad_distortions else 0
                        max_d = max(quad_distortions) if quad_distortions else 0
                        delta_d = max_d - min_d
                        
                        for f_idx, fkey in enumerate(new_flattened_mesh_data.faces()):
                            if f_idx < len(quad_distortions):
                                distortion = quad_distortions[f_idx]
                                norm_d = 0.0
                                if delta_d > 1e-9: # Avoid division by zero if all distortions are the same
                                    norm_d = (distortion - min_d) / delta_d
                                norm_d = distortion * 10
                                norm_d = max(0.0, min(1.0, norm_d)) # Clamp to [0, 1]

                                # Color gradient: Green (0.0) -> Yellow (0.5) -> Red (1.0)
                                r_col, g_col, b_col = 0.0, 0.0, 0.0
                                if norm_d <= 0.5:
                                    r_col = 2 * norm_d; g_col = 1.0
                                else:
                                    r_col = 1.0; g_col = 2 * (1.0 - norm_d)
                                b_col = 0.0 
                                face_colors[fkey] = Color(r_col, g_col, b_col, 0.8) 
                            else: 
                                face_colors[fkey] = default_face_color # Should not happen
                    else: 
                        for fkey in new_flattened_mesh_data.faces():
                            face_colors[fkey] = default_face_color

                    if i < len(scene_objects_flat_strips) and scene_objects_flat_strips[i] is not None:
                        scene_obj = scene_objects_flat_strips[i]
                        mesh_to_update = scene_obj.item # Get the existing Mesh object
                        
                        mesh_to_update.clear() # Clear its current geometry

                        # Repopulate mesh_to_update with geometry from new_flattened_mesh_data
                        vkey_map_flat = {}
                        for old_vkey in new_flattened_mesh_data.vertices():
                            x,y,z = new_flattened_mesh_data.vertex_coordinates(old_vkey)
                            new_vkey = mesh_to_update.add_vertex(x=x,y=y,z=z)
                            vkey_map_flat[old_vkey] = new_vkey
                        
                        for old_fkey in new_flattened_mesh_data.faces():
                            face_vkeys_in_new_data = new_flattened_mesh_data.face_vertices(old_fkey)
                            face_vkeys_in_mesh_to_update = [vkey_map_flat[ovk] for ovk in face_vkeys_in_new_data]
                            mesh_to_update.add_face(face_vkeys_in_mesh_to_update)

                        scene_obj.facecolor = face_colors 
                        scene_obj.update(update_data=True)
                        new_scene_objs_flat_strips_temp[i] = scene_obj
                        scene_objects_flat_strips[i] = None
                    else:
                        scene_obj = viewer.scene.add(new_flattened_mesh_data, name=f"FlatStrip_{i}",
                                                     facecolor=face_colors, 
                                                     show_edges=True, 
                                                     linecolor=Color(0.2, 0.2, 0.2, 0.9), linewidth=1.0)
                        new_scene_objs_flat_strips_temp[i] = scene_obj
                else:
                    # print(f"{print_prefix}  ERROR: develop_strip_to_plane returned None for flat strip {i}")
                    pass 
            except Exception as e:
                print(f"{print_prefix}  ERROR creating/updating flat strip {i}: {e}")
                # import traceback; traceback.print_exc()


        for old_scene_obj in scene_objects_flat_strips: # Remove any old objects not processed
            if old_scene_obj is not None: viewer.scene.remove(old_scene_obj)
        scene_objects_flat_strips[:] = new_scene_objs_flat_strips_temp
        
        # print(f"\n{print_prefix}--- Forcing viewer renderer update ---")
        viewer.renderer.update() 
        
        # print(f"{print_prefix}GEOMETRY REGENERATION COMPLETE\n{print_prefix}{'='*50}")
    except Exception as e:
        print(f"CRITICAL ERROR in regenerate_all_geometry: {e}")
        import traceback
        traceback.print_exc()
    finally:
        is_updating = False

def setup_ui():
    """Sets up the Qt UI panel for adjusting parameters."""
    global viewer, instructions_data, FLAT_Z_OFFSET

    print("\n=== SETTING UP UI ===")

    dock = QDockWidget("Catenary Parameters")
    dock.setFeatures(QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable) 
    
    scroll_area = QScrollArea()
    scroll_area.setWidgetResizable(True)
    
    main_widget = QWidget()
    main_layout = QVBoxLayout(main_widget)
    main_layout.setAlignment(QtCore.Qt.AlignTop)

    global_group = QGroupBox("Global Settings")
    global_layout = QVBoxLayout(global_group)
    
    flat_z_layout = QHBoxLayout()
    flat_z_label = QLabel("Flat Strips Z-Offset:")
    flat_z_spinbox = QDoubleSpinBox()
    flat_z_spinbox.setRange(-100.0, 100.0)
    flat_z_spinbox.setSingleStep(0.5)
    flat_z_spinbox.setValue(FLAT_Z_OFFSET)
    flat_z_spinbox.valueChanged.connect(functools.partial(update_flat_z_offset_param, flat_z_spinbox))
    flat_z_layout.addWidget(flat_z_label)
    flat_z_layout.addWidget(flat_z_spinbox)
    global_layout.addLayout(flat_z_layout)
    
    regenerate_button = QPushButton("Regenerate Geometry")
    regenerate_button.clicked.connect(regenerate_all_geometry) 
    global_layout.addWidget(regenerate_button)
    
    main_layout.addWidget(global_group)

    # Individual catenary controls (like height) are removed as per the change.
    # If you want to display information about the loaded polylines (e.g., number of polylines, lengths),
    # you could add QLabels or a non-interactive list here.
    # For example:
    num_polylines_label = QLabel(f"Number of loaded polylines: {len(instructions_data)}")
    main_layout.addWidget(num_polylines_label)

    spacer = QSpacerItem(20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)
    main_layout.addSpacerItem(spacer)
    
    scroll_area.setWidget(main_widget)
    dock.setWidget(scroll_area)
    
    viewer.ui.sidedock.add(dock)
    print("UI Setup Complete.")


# --- Main script execution ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Visualize catenary curves and strips, loading initial curves from a COMPAS JSON file.")
    parser.add_argument("json_file", help="Path to the COMPAS JSON file defining the initial catenary spans (list of Polylines).")
    args = parser.parse_args()

    print("="*60)
    print("STARTING CATENARY VIEWER APPLICATION")
    print("="*60)
    
    print(f"\nLoading instructions from JSON: {args.json_file}...")
    if not load_instructions_from_json(args.json_file):
        print("Failed to load instructions. Exiting.")
        sys.exit(1)
    print(f"Successfully loaded {len(instructions_data)} catenary definitions.")

    print("Creating viewer...")
    viewer = Viewer(width=1600, height=900, show_grid=True)
    print(f"Viewer created: {viewer}")
    print(f"Viewer scene: {viewer.scene}")

    # Visualize loaded points and quadrant corner
    if loaded_json_points:
        print(f"\nVisualizing {len(loaded_json_points)} loaded points from JSON...")
        for i, pt in enumerate(loaded_json_points):
            viewer.scene.add(pt, name=f"LoadedPoint_{i}", color=Color.red(), size=10)
    
    if loaded_quadrant_corner:
        print(f"Visualizing quadrant corner: {loaded_quadrant_corner}")
        quadrant_corner_sphere = Sphere(radius=0.1, point=loaded_quadrant_corner) # Create a sphere
        viewer.scene.add(quadrant_corner_sphere, name="QuadrantCornerSphere", facecolor=Color.blue(), linecolor=Color.blue().darkened(50), opacity=0.8)

    print("\nSetting up UI...")
    # setup_ui() # Call setup_ui after viewer is initialized
    
    print("\nGenerating initial geometry...")
    regenerate_all_geometry()

    # print("\n" + "="*60)
    # print("LAUNCHING COMPAS VIEWER")
    # print("Try changing values in the UI panel to test updates...")
    # print("Close the viewer window to end the script.")
    # print("="*60)
    # viewer.ui.sidedock.visible = False
    viewer.show()