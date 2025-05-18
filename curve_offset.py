import math
import time
import functools # For partial function application in signal connections

from PySide6 import QtCore, QtWidgets
from PySide6.QtWidgets import QWidget, QVBoxLayout, QGroupBox, QLabel, QDoubleSpinBox, QPushButton, QScrollArea, QDockWidget, QHBoxLayout, QSpacerItem, QSizePolicy


from compas.geometry import Point, Polyline, Translation, Rotation, Vector, Line
from compas.geometry import intersection_circle_circle_xy
from compas.colors import Color
from compas.datastructures import Mesh
from compas_viewer import Viewer

# --- Configuration ---
NUM_CATENARY_POINTS = 21
FLAT_Z_OFFSET = -15.0 # Initial FLAT_Z_OFFSET, will be controllable by UI

# --- Global variables for UI interaction and scene management ---
viewer = None # Will be initialized later
initial_instructions_data_tuples = [
    (10.0, 4.0, 0.0, 0.0),   
    (10.0, 6.0, 4.0, 5.0),  
    (10.0, 4.0, 8.0, 10.0), 
    (10.0, 6.0, 12.0, 15.0),
    (10.0, 4.0, 16.0, 20.0), 
    (10.0, 6.0, 20.0, 25.0), 
    (10.0, 4.0, 24.0, 30.0), 
    (10.0, 6.0, 28.0, 35.0), 
    (10.0, 4.0, 32.0, 40.0), 
]

# Convert to list of lists for mutable parameters
instructions_data = [list(item) for item in initial_instructions_data_tuples]

generated_polylines_3d = []
scene_objects_3d_surfaces = [] # Will store SceneObject instances
scene_objects_flat_strips = [] # Will store SceneObject instances

# Track update state to prevent recursive calls
is_updating = False

# --- Geometry Functions (solve_for_catenary_a, create_catenary_polyline, develop_strip_to_plane) ---
# These functions remain largely the same as your last provided version.
# Minor adjustments might be made for clarity or to use global NUM_CATENARY_POINTS.

def solve_for_catenary_a(span_radius, target_height, tol=1e-6, max_iter=100):
    if target_height <= 1e-9: return None
    if span_radius <= 1e-9: return None
    def func_to_solve(a_param):
        if abs(a_param) < 1e-9: return float('inf')
        ratio = span_radius / a_param
        if abs(ratio) > 700: return float('inf')
        try: return a_param * (math.cosh(ratio) - 1) - target_height
        except OverflowError: return float('inf')
    a_low = max(1e-9, span_radius / 690.0)
    a_high = span_radius**2 / (2 * target_height + 1e-12) + span_radius
    a_high = max(a_high, span_radius * 20)
    val_low = func_to_solve(a_low)
    val_high = func_to_solve(a_high)
    if val_low == float('inf'): return None
    if val_low <= 0:
        a_low_retry = span_radius / 700.0
        if a_low_retry < a_low :
            val_low_retry = func_to_solve(a_low_retry)
            if val_low_retry > 0: a_low, val_low = a_low_retry, val_low_retry
            else: return None
        else: return None
    if val_high >= 0:
        a_high_retry = (span_radius**2 / (2 * target_height + 1e-12)) * 1000
        val_high_retry = func_to_solve(a_high_retry)
        if val_high_retry >=0: return None
        else: a_high, val_high = a_high_retry, val_high_retry
    if val_low * val_high >= 0: return None
    for _ in range(max_iter):
        a_mid = (a_low + a_high) / 2
        if a_mid == a_low or a_mid == a_high: break
        val_mid = func_to_solve(a_mid)
        if abs(val_mid) < tol: return a_mid
        if val_mid == float('inf'): a_low = a_mid; continue
        if val_mid > 0: a_low = a_mid
        else: a_high = a_mid
    return (a_low + a_high) / 2

def create_catenary_polyline(span_radius, catenary_a_param, num_points=NUM_CATENARY_POINTS):
    if not (isinstance(span_radius, (int, float)) and span_radius > 1e-6): return None
    if not (isinstance(catenary_a_param, (int, float)) and abs(catenary_a_param) > 1e-9): return None
    if num_points < 2: num_points = 2
    points = []
    a = catenary_a_param
    try:
        ratio_at_radius = span_radius / a
        if abs(ratio_at_radius) > 700: return None
        cosh_val_at_radius = math.cosh(ratio_at_radius)
    except: return None
    for i in range(num_points):
        x_coeff = (i / (num_points - 1)) * 2.0 - 1.0 if num_points > 1 else 0.0
        x_local = x_coeff * span_radius
        try:
            current_ratio = x_local / a
            if abs(current_ratio) > 700:
                if i == 0 or i == num_points -1 : return None
                continue
            local_z_sag = a * (cosh_val_at_radius - math.cosh(current_ratio))
            points.append(Point(x_local, 0, local_z_sag))
        except: continue
    if len(points) < 2: return None
    return Polyline(points)

def develop_strip_to_plane(poly1_3d, poly2_3d, start_point_on_plane, initial_unroll_vec):
    if len(poly1_3d.points) != len(poly2_3d.points) or len(poly1_3d.points) < 2: return None
    N = len(poly1_3d.points)
    z_plane = start_point_on_plane.z
    flat_vertices, flat_faces = [], []
    xy_plane_normal = Vector(0, 0, 1)

    # Initialize distortion measurement
    total_distortion_measure = 0.0
    num_quads_for_distortion = 0

    P0_3d, Q0_3d = poly1_3d.points[0], poly2_3d.points[0]
    P_curr_flat = start_point_on_plane.copy()
    flat_vertices.append(P_curr_flat)

    dist_p0q0 = P0_3d.distance_to_point(Q0_3d)
    unroll_dir_xy = Vector(initial_unroll_vec.x, initial_unroll_vec.y, 0)
    if unroll_dir_xy.length < 1e-6: unroll_dir_xy = Vector(1,0,0) # Default unroll direction
    else: unroll_dir_xy.unitize()
    
    rung_dir_xy = Vector(-unroll_dir_xy.y, unroll_dir_xy.x, 0) # Perpendicular to unroll direction
    Q_curr_flat = P_curr_flat + rung_dir_xy * dist_p0q0
    Q_curr_flat.z = z_plane # Ensure it's on the plane
    flat_vertices.append(Q_curr_flat)

    P_prev_flat = None # Used for choosing intersection point

    for j in range(N - 1):
        P_curr_3d, P_next_3d = poly1_3d.points[j], poly1_3d.points[j+1]
        Q_curr_3d, Q_next_3d = poly2_3d.points[j], poly2_3d.points[j+1]

        d_pc_pn = P_curr_3d.distance_to_point(P_next_3d) # Length of current segment on poly1
        d_qc_pn = Q_curr_3d.distance_to_point(P_next_3d) # Diagonal length for first triangle

        intersections_P = intersection_circle_circle_xy(((P_curr_flat, xy_plane_normal), d_pc_pn), ((Q_curr_flat, xy_plane_normal), d_qc_pn))
        if not intersections_P: return None # Cannot triangulate
        
        P_next_flat_chosen_xy = intersections_P[0]
        if len(intersections_P) > 1:
            # Choose point that continues "forward"
            fwd_dir = unroll_dir_xy if P_prev_flat is None else Vector.from_start_end(P_prev_flat, P_curr_flat).unitized()
            v0 = Vector(intersections_P[0][0]-P_curr_flat.x, intersections_P[0][1]-P_curr_flat.y, 0).unitized()
            v1 = Vector(intersections_P[1][0]-P_curr_flat.x, intersections_P[1][1]-P_curr_flat.y, 0).unitized()
            P_next_flat_chosen_xy = intersections_P[0] if v0.dot(fwd_dir) >= v1.dot(fwd_dir) else intersections_P[1]
        P_next_flat = Point(P_next_flat_chosen_xy[0], P_next_flat_chosen_xy[1], z_plane)

        d_qc_qn = Q_curr_3d.distance_to_point(Q_next_3d) # Length of current segment on poly2
        d_pn_qn = P_next_3d.distance_to_point(Q_next_3d) # Length of the "rung" at the end of the segment

        intersections_Q = intersection_circle_circle_xy(((Q_curr_flat, xy_plane_normal), d_qc_qn), ((P_next_flat, xy_plane_normal), d_pn_qn))
        if not intersections_Q: return None # Cannot triangulate

        Q_next_flat_chosen_xy = intersections_Q[0]
        if len(intersections_Q) > 1:
            # Choose point that maintains similar orientation of the P-Q rung
            rung_prev_dir = Vector.from_start_end(P_curr_flat, Q_curr_flat).unitized()
            # We want Q_next_flat such that vector P_next_flat -> Q_next_flat is similar to P_curr_flat -> Q_curr_flat
            r0 = Vector(intersections_Q[0][0]-P_next_flat.x, intersections_Q[0][1]-P_next_flat.y, 0).unitized()
            r1 = Vector(intersections_Q[1][0]-P_next_flat.x, intersections_Q[1][1]-P_next_flat.y, 0).unitized()
            Q_next_flat_chosen_xy = intersections_Q[0] if r0.dot(rung_prev_dir) >= r1.dot(rung_prev_dir) else intersections_Q[1]
        Q_next_flat = Point(Q_next_flat_chosen_xy[0], Q_next_flat_chosen_xy[1], z_plane)

        # --- Measure distortion for the current quad ---
        # The quad is defined by (P_curr_3d, Q_curr_3d, Q_next_3d, P_next_3d)
        # and its flattened version (P_curr_flat, Q_curr_flat, Q_next_flat, P_next_flat)
        # The "other" diagonal (not explicitly preserved by the triangulation method for both triangles)
        # is P_curr_3d to Q_next_3d.
        diag_3d_length = P_curr_3d.distance_to_point(Q_next_3d)
        diag_flat_length = P_curr_flat.distance_to_point(Q_next_flat) # P_curr_flat is from start of this quad segment

        current_quad_distortion = abs(diag_3d_length - diag_flat_length)/diag_3d_length
        total_distortion_measure += current_quad_distortion
        num_quads_for_distortion += 1
        # print(f"    Quad {j}: 3D Diag P_curr-Q_next: {diag_3d_length:.4f}, Flat Diag P_curr-Q_next: {diag_flat_length:.4f}, Diff: {current_quad_distortion:.6f}")
        # --- End Measure distortion ---

        idx_Pc, idx_Qc = 2*j, 2*j + 1 # Indices of P_curr_flat, Q_curr_flat in flat_vertices
        # These should already exist in flat_vertices
        if idx_Pc >= len(flat_vertices) or idx_Qc >= len(flat_vertices) : return None # Should not happen

        flat_vertices.extend([P_next_flat, Q_next_flat])
        idx_Pn, idx_Qn = len(flat_vertices)-2, len(flat_vertices)-1 # Indices of newly added points

        flat_faces.append([idx_Pc, idx_Pn, idx_Qn, idx_Qc]) # Define the quad face

        # Update for next iteration
        P_prev_flat = P_curr_flat # Store P_curr_flat before it's updated
        P_curr_flat = P_next_flat
        Q_curr_flat = Q_next_flat
    
    if not flat_vertices or not flat_faces: return None

    # Print distortion summary
    if num_quads_for_distortion > 0:
        average_distortion = total_distortion_measure / num_quads_for_distortion
        print(f"  Develop_strip_to_plane Distortion Report:")
        print(f"    Total diagonal length difference (sum over {num_quads_for_distortion} quads): {total_distortion_measure:.6f}")
        print(f"    Average diagonal length difference per quad: {average_distortion:.6f}")
    else:
        print("  Develop_strip_to_plane: No quads processed for distortion measure.")

    return Mesh.from_vertices_and_faces(flat_vertices, flat_faces)

# --- UI Update and Geometry Regeneration Functions ---
def update_catenary_param(catenary_idx, param_idx, control_widget, *args):
    """Updates a specific parameter in instructions_data and regenerates geometry."""
    global instructions_data, is_updating
    
    if is_updating:
        return
        
    print(f"\n=== UPDATE CALLBACK TRIGGERED ===")
    print(f"Catenary {catenary_idx+1}, Parameter {param_idx}")
    print(f"Control widget value: {control_widget.value()}")
    
    try:
        value = float(control_widget.value()) 
        old_value = instructions_data[catenary_idx][param_idx]
        instructions_data[catenary_idx][param_idx] = value
        print(f"Changed from {old_value} to {value}")
        
        QtCore.QTimer.singleShot(100, regenerate_all_geometry)
        
    except ValueError as e:
        print(f"ValueError in update_catenary_param: {e}")
    except Exception as e:
        print(f"Unexpected error in update_catenary_param: {e}")

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

    if is_updating:
        print("Regeneration already in progress, skipping...")
        return
        
    is_updating = True
    
    try:
        print("\n" + "="*60)
        print("STARTING GEOMETRY REGENERATION / UPDATE")
        print("="*60)

        if viewer is None or viewer.scene is None:
            print("ERROR: Viewer or Viewer.scene is None!")
            is_updating = False
            return

        print(f"Current instructions_data: {instructions_data}")
        print(f"Current FLAT_Z_OFFSET: {FLAT_Z_OFFSET}")
        
        generated_polylines_3d.clear()

        # 1. Generate 3D Polylines
        print(f"\n--- GENERATING 3D POLYLINES ---")
        valid_polylines_count = 0
        for idx, params in enumerate(instructions_data):
            radius, height, y_offset, rotation_angle_deg_z = params
            # print(f"\nPolyline {idx+1}: r={radius}, h={height}, y_off={y_offset}, rot={rotation_angle_deg_z}")
            
            current_catenary_a = solve_for_catenary_a(radius, height)
            if current_catenary_a is None:
                # print(f"  ERROR: Could not solve catenary for params {params}")
                generated_polylines_3d.append(None)
                continue
            
            base_catenary = create_catenary_polyline(radius, current_catenary_a, num_points=NUM_CATENARY_POINTS)
            if base_catenary is None:
                # print(f"  ERROR: Could not create polyline for params {params}")
                generated_polylines_3d.append(None)
                continue
            
            transformed_catenary = base_catenary.copy()
            if abs(y_offset) > 1e-6: 
                transformed_catenary.transform(Translation.from_vector(Vector(0, y_offset, 0)))
            if abs(rotation_angle_deg_z) > 1e-6: 
                transformed_catenary.transform(Rotation.from_axis_and_angle(Vector(0, 0, 1), math.radians(rotation_angle_deg_z)))
            generated_polylines_3d.append(transformed_catenary)
            valid_polylines_count += 1
        print(f"Generated {valid_polylines_count} valid polylines out of {len(instructions_data)}")

        num_expected_strips = 0
        if len(generated_polylines_3d) >= 2:
            num_expected_strips = len(generated_polylines_3d) - 1
        
        # --- 2. Update or Create 3D Surface Strips ---
        print(f"\n--- UPDATING/CREATING 3D SURFACES ({num_expected_strips} expected) ---")
        new_scene_objs_3d_surfaces_temp = [None] * num_expected_strips 
        surfaces_processed = 0

        for i in range(num_expected_strips):
            poly1_3d = generated_polylines_3d[i]
            poly2_3d = generated_polylines_3d[i+1]

            # print(f"\nProcessing 3D Surface strip {i+1}-{i+2}: P1 valid: {poly1_3d is not None}, P2 valid: {poly2_3d is not None}")

            if not poly1_3d or not poly2_3d or \
               len(poly1_3d.points) != len(poly2_3d.points) or \
               len(poly1_3d.points) < 2:
                print(f"  SKIPPED: Invalid polylines for 3D Surface strip {i+1}-{i+2}")
                continue
            
            try:
                strip_vertices_3d, strip_faces_3d = [], []
                for pt in poly1_3d.points: strip_vertices_3d.append(pt)
                offset_3d = len(poly1_3d.points)
                for pt in poly2_3d.points: strip_vertices_3d.append(pt)
                for j_seg in range(len(poly1_3d.points) - 1):
                    strip_faces_3d.append([j_seg, j_seg + 1, offset_3d + j_seg + 1, offset_3d + j_seg])
                
                # This is the new mesh data we want to display
                new_mesh_data_3d = Mesh.from_vertices_and_faces(strip_vertices_3d, strip_faces_3d)
                # print(f"  Mesh data for 3D surface {i+1}-{i+2}: {len(strip_vertices_3d)}V, {len(strip_faces_3d)}F")

                if i < len(scene_objects_3d_surfaces) and scene_objects_3d_surfaces[i] is not None:
                    scene_obj = scene_objects_3d_surfaces[i]
                    mesh_to_update = scene_obj.item # Get the existing Mesh object
                    mesh_to_update.clear() # Clear its current geometry

                    # Repopulate mesh_to_update with geometry from new_mesh_data_3d
                    vkey_map = {}
                    for old_vkey in new_mesh_data_3d.vertices():
                        x, y, z = new_mesh_data_3d.vertex_coordinates(old_vkey)
                        new_vkey = mesh_to_update.add_vertex(x=x, y=y, z=z)
                        vkey_map[old_vkey] = new_vkey
                    
                    for old_fkey in new_mesh_data_3d.faces():
                        face_vkeys_in_new_data = new_mesh_data_3d.face_vertices(old_fkey)
                        face_vkeys_in_mesh_to_update = [vkey_map[ovk] for ovk in face_vkeys_in_new_data]
                        mesh_to_update.add_face(face_vkeys_in_mesh_to_update)
                    
                    scene_obj.update(update_data=True) # Tell the scene object to re-render with updated item
                    new_scene_objs_3d_surfaces_temp[i] = scene_obj
                    scene_objects_3d_surfaces[i] = None # Mark as processed
                    print(f"  SUCCESS: Updated existing 3D surface SceneObject for strip {i+1}-{i+2}")
                else:
                    scene_obj = viewer.scene.add(new_mesh_data_3d, name=f"3DSurface_{i+1}-{i+2}",
                                                 facecolor=Color(0.6, 0.7, 0.9, 0.7), show_edges=True,
                                                 linecolor=Color(0.1, 0.1, 0.1, 0.9), linewidth=1.0)
                    new_scene_objs_3d_surfaces_temp[i] = scene_obj
                    print(f"  SUCCESS: Added new 3D surface SceneObject for strip {i+1}-{i+2}")
                surfaces_processed += 1
            except Exception as e:
                print(f"  ERROR creating/updating 3D surface for strip {i+1}-{i+2}: {e}")

        for idx, old_scene_obj in enumerate(scene_objects_3d_surfaces):
            if old_scene_obj is not None: 
                print(f"  Removing outdated/superfluous 3D surface SceneObject: {old_scene_obj.name}")
                viewer.scene.remove(old_scene_obj)
        scene_objects_3d_surfaces[:] = new_scene_objs_3d_surfaces_temp

        # --- 3. Update or Create Flattened Strips ---
        print(f"\n--- UPDATING/CREATING FLATTENED STRIPS ({num_expected_strips} expected) ---")
        new_scene_objs_flat_strips_temp = [None] * num_expected_strips
        flats_processed = 0

        for i in range(num_expected_strips):
            poly1_3d = generated_polylines_3d[i]
            poly2_3d = generated_polylines_3d[i+1]

            # print(f"\nProcessing Flat Strip {i+1}-{i+2}: P1 valid: {poly1_3d is not None}, P2 valid: {poly2_3d is not None}")
            if not poly1_3d or not poly2_3d or \
               len(poly1_3d.points) != len(poly2_3d.points) or \
               len(poly1_3d.points) < 2:
                print(f"  SKIPPED: Invalid polylines for flat strip {i+1}-{i+2}")
                continue
            
            try:
                flat_start_point = Point(poly1_3d.points[0].x, poly1_3d.points[0].y, FLAT_Z_OFFSET)
                initial_unroll_direction = Vector(0,0,0)
                if len(poly1_3d.points) >= 2:
                    p0, p1 = poly1_3d.points[0], poly1_3d.points[1]
                    initial_unroll_direction = Vector(p1.x - p0.x, p1.y - p0.y, 0)
                if initial_unroll_direction.length < 1e-6: initial_unroll_direction = Vector(1,0,0)
                else: initial_unroll_direction.unitize()
                
                # This is the new mesh data we want to display
                new_flattened_mesh_data = develop_strip_to_plane(poly1_3d, poly2_3d, flat_start_point, initial_unroll_direction)
                
                if new_flattened_mesh_data:
                    # print(f"  Flat mesh data: {len(list(new_flattened_mesh_data.vertices()))}V, {len(list(new_flattened_mesh_data.faces()))}F")
                    if i < len(scene_objects_flat_strips) and scene_objects_flat_strips[i] is not None:
                        scene_obj = scene_objects_flat_strips[i]
                        mesh_to_update = scene_obj.item # Get existing Mesh
                        mesh_to_update.clear() # Clear its geometry

                        # Repopulate mesh_to_update with new_flattened_mesh_data
                        vkey_map_flat = {}
                        for old_vkey in new_flattened_mesh_data.vertices():
                            x,y,z = new_flattened_mesh_data.vertex_coordinates(old_vkey)
                            new_vkey = mesh_to_update.add_vertex(x=x,y=y,z=z)
                            vkey_map_flat[old_vkey] = new_vkey
                        
                        for old_fkey in new_flattened_mesh_data.faces():
                            face_vkeys_in_new_data = new_flattened_mesh_data.face_vertices(old_fkey)
                            face_vkeys_in_mesh_to_update = [vkey_map_flat[ovk] for ovk in face_vkeys_in_new_data]
                            mesh_to_update.add_face(face_vkeys_in_mesh_to_update)

                        scene_obj.update(update_data=True) # Tell scene object to re-render
                        new_scene_objs_flat_strips_temp[i] = scene_obj
                        scene_objects_flat_strips[i] = None # Mark as processed
                        print(f"  SUCCESS: Updated existing flat strip SceneObject for strip {i+1}-{i+2}")
                    else:
                        scene_obj = viewer.scene.add(new_flattened_mesh_data, name=f"FlatStrip_{i+1}-{i+2}",
                                                     facecolor=Color(0.7, 0.9, 0.6, 0.7), show_edges=True, 
                                                     linecolor=Color(0.2, 0.2, 0.2, 0.9), linewidth=1.0)
                        new_scene_objs_flat_strips_temp[i] = scene_obj
                        print(f"  SUCCESS: Added new flat strip SceneObject for strip {i+1}-{i+2}")
                    flats_processed += 1
                else:
                    print(f"  ERROR: develop_strip_to_plane returned None for strip {i+1}-{i+2}")
                    # If an old object existed here, it should be removed
                    if i < len(scene_objects_flat_strips) and scene_objects_flat_strips[i] is not None:
                         # Will be caught by the cleanup loop below
                         pass


            except Exception as e:
                raise e
                print(f"  ERROR creating/updating flat strip for strip {i+1}-{i+2}: {e}")
                # If an old object existed here, it should be removed
                if i < len(scene_objects_flat_strips) and scene_objects_flat_strips[i] is not None:
                    # Will be caught by the cleanup loop below
                    pass


        for idx, old_scene_obj in enumerate(scene_objects_flat_strips):
            if old_scene_obj is not None: 
                print(f"  Removing outdated/superfluous flat strip SceneObject: {old_scene_obj.name}")
                viewer.scene.remove(old_scene_obj)
        scene_objects_flat_strips[:] = new_scene_objs_flat_strips_temp
        
        print(f"\nSummary: Processed {surfaces_processed} 3D surfaces and {flats_processed} flat strips.")
        print(f"Scene objects after regeneration: {len(viewer.scene.objects)}")

        print(f"\n--- FORCING VIEWER RENDERER UPDATE ---")
        viewer.renderer.update() 
        
        print("GEOMETRY REGENERATION / UPDATE COMPLETE")
        print("="*60)
        
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

    param_names = ["Radius", "Height", "Y-Offset", "Z-Rotation (deg)"]
    param_ranges = [(0.1, 100.0), (0.1, 50.0), (-50.0, 50.0), (-360.0, 360.0)]
    param_steps = [0.5, 0.5, 0.5, 5.0]

    for idx, catenary_params in enumerate(instructions_data):
        group_box = QGroupBox(f"Catenary {idx + 1}")
        group_layout = QVBoxLayout(group_box)
        
        for param_idx, param_name in enumerate(param_names):
            param_layout = QHBoxLayout()
            label = QLabel(f"{param_name}:")
            spinbox = QDoubleSpinBox()
            spinbox.setRange(param_ranges[param_idx][0], param_ranges[param_idx][1])
            spinbox.setSingleStep(param_steps[param_idx])
            spinbox.setValue(catenary_params[param_idx])
            spinbox.valueChanged.connect(
                functools.partial(update_catenary_param, idx, param_idx, spinbox)
            )
            param_layout.addWidget(label)
            param_layout.addWidget(spinbox)
            group_layout.addLayout(param_layout)
        
        main_layout.addWidget(group_box)

    spacer = QSpacerItem(20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)
    main_layout.addSpacerItem(spacer)
    
    scroll_area.setWidget(main_widget)
    dock.setWidget(scroll_area)
    
    viewer.ui.sidedock.add(dock)
    print("UI Setup Complete.")


# --- Main script execution ---
if __name__ == '__main__':
    print("="*60)
    print("STARTING CATENARY VIEWER APPLICATION")
    print("="*60)
    
    print("Creating viewer...")
    viewer = Viewer(width=1600, height=900, show_grid=True)
    print(f"Viewer created: {viewer}")
    print(f"Viewer scene: {viewer.scene}")

    print("\nSetting up UI...")
    setup_ui() 
    
    print("\nGenerating initial geometry...")
    regenerate_all_geometry() 

    print("\n" + "="*60)
    print("LAUNCHING COMPAS VIEWER")
    print("Try changing values in the UI panel to test updates...")
    print("Close the viewer window to end the script.")
    print("="*60)
    viewer.ui.sidedock.show = True
    viewer.show()