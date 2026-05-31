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
from compas_tna.diagrams import FormDiagram

# --- Configuration ---
NUM_CATENARY_POINTS = 41
FLAT_Z_OFFSET = -15.0 # Initial FLAT_Z_OFFSET, will be controllable by UI
USE_TNA = False
TNA_CATENARIES = []

# --- Global variables for UI interaction and scene management ---
# ... (rest remains same)

# --- TNA Extraction Helpers ---
def _get_vector(v1, v2):
    return (v2['x'] - v1['x'], v2['y'] - v1['y'])

def _normalize(v):
    L = math.sqrt(v[0]**2 + v[1]**2)
    if L == 0: return (0, 0)
    return (v[0]/L, v[1]/L)

def _dot(v1, v2):
    return v1[0]*v2[0] + v1[1]*v2[1]

def extract_spokes(diagram, target_corner_coords=(0.0, 0.0)):
    target_corner = None
    for v in diagram.vertices():
        x, y = diagram.vertex_attributes(v, names=['x', 'y'])
        if abs(x - target_corner_coords[0]) < 1e-6 and abs(y - target_corner_coords[1]) < 1e-6:
            target_corner = v
            break
    if target_corner is None: return []

    spokes = []
    neighbors = diagram.vertex_neighbors(target_corner)
    for n in neighbors:
        spoke = [target_corner, n]
        curr, prev = n, target_corner
        while True:
            v_curr = diagram.vertex_attributes(curr)
            if abs(v_curr['x'] - 5.0) < 1e-6 or abs(v_curr['y'] - 5.0) < 1e-6:
                break
            next_neighbors = diagram.vertex_neighbors(curr)
            best_n, max_dot = None, -2.0
            vec_prev = _normalize(_get_vector(diagram.vertex_attributes(prev), v_curr))
            for nn in next_neighbors:
                if nn == prev: continue
                vec_next = _normalize(_get_vector(v_curr, diagram.vertex_attributes(nn)))
                d = _dot(vec_prev, vec_next)
                if d > max_dot:
                    max_dot, best_n = d, nn
            if best_n is None or max_dot < 0.5: break
            prev, curr = curr, best_n
            spoke.append(curr)
        spokes.append(spoke)
    return spokes

def load_tna_catenaries():
    global TNA_CATENARIES, USE_TNA
    try:
        form_min = FormDiagram.from_json('thrust_min.json')
        form_max = FormDiagram.from_json('thrust_max.json')
        
        spokes_min_ids = extract_spokes(form_min)
        spokes_max_ids = extract_spokes(form_max)
        
        TNA_CATENARIES = []
        for i in range(len(spokes_max_ids)):
            if i % 2 == 0:
                ids, diagram = spokes_max_ids[i], form_max
            else:
                ids, diagram = spokes_min_ids[i], form_min
            
            pts = [Point(*diagram.vertex_coordinates(v)) for v in ids]
            TNA_CATENARIES.append(Polyline(pts))
        
        USE_TNA = True
        print(f"Loaded {len(TNA_CATENARIES)} TNA catenaries.")
        regenerate_all_geometry()
    except Exception as e:
        print(f"Error loading TNA data: {e}")

# --- Geometry Functions ...
viewer = None # Will be initialized later
initial_instructions_data_tuples = [
    (10.0, 4.0, 0.0, 0.0),   
    (10.0, 6.0, 4.0, 5.0),  
    (10.0, 4.0, 8.0, 10.0), 
    (10.0, 6.0, 12.0, 5.0),
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
    global instructions_data, FLAT_Z_OFFSET, is_updating, USE_TNA, TNA_CATENARIES

    if is_updating: return
    is_updating = True
    
    print_prefix = "  [RegenGeom] "

    try:
        if viewer is None or viewer.scene is None:
            is_updating = False
            return

        generated_polylines_3d.clear()

        # 1. Generate 3D Polylines
        if USE_TNA:
            generated_polylines_3d = [c.copy() for c in TNA_CATENARIES]
        else:
            for idx, params in enumerate(instructions_data):
                radius, height, y_offset, rotation_angle_deg_z = params
                current_catenary_a = solve_for_catenary_a(radius, height)
                if current_catenary_a is None:
                    generated_polylines_3d.append(None); continue
                base_catenary = create_catenary_polyline(radius, current_catenary_a, num_points=NUM_CATENARY_POINTS)
                if base_catenary is None:
                    generated_polylines_3d.append(None); continue
                
                transformed_catenary = base_catenary.copy()
                if abs(y_offset) > 1e-6: 
                    transformed_catenary.transform(Translation.from_vector(Vector(0, y_offset, 0)))
                if abs(rotation_angle_deg_z) > 1e-6: 
                    rotation_origin = transformed_catenary.points[0] if transformed_catenary.points else Point(0,0,0)
                    transformed_catenary.transform(Rotation.from_axis_and_angle(Vector(0, 0, 1), math.radians(rotation_angle_deg_z), point=rotation_origin))
                generated_polylines_3d.append(transformed_catenary)

        num_expected_strips = max(0, len(generated_polylines_3d) - 1)
        
        # --- 2. Update or Create 3D Surface Strips ---
        new_scene_objs_3d_surfaces_temp = [None] * num_expected_strips 
        for i in range(num_expected_strips):
            poly1_3d = generated_polylines_3d[i]
            poly2_3d = generated_polylines_3d[i+1]

            if not poly1_3d or not poly2_3d or \
               len(poly1_3d.points) != len(poly2_3d.points) or len(poly1_3d.points) < 2:
                continue
            
            try:
                strip_vertices_3d = poly1_3d.points + poly2_3d.points
                strip_faces_3d = []
                offset_3d = len(poly1_3d.points)
                for j_seg in range(len(poly1_3d.points) - 1):
                    strip_faces_3d.append([j_seg, j_seg + 1, offset_3d + j_seg + 1, offset_3d + j_seg])
                
                new_mesh_data_3d = Mesh.from_vertices_and_faces(strip_vertices_3d, strip_faces_3d)

                if i < len(scene_objects_3d_surfaces) and scene_objects_3d_surfaces[i] is not None:
                    scene_obj = scene_objects_3d_surfaces[i]
                    mesh_to_update = scene_obj.item
                    mesh_to_update.clear()
                    vkey_map = {}
                    for old_vkey in new_mesh_data_3d.vertices():
                        x, y, z = new_mesh_data_3d.vertex_coordinates(old_vkey)
                        new_vkey = mesh_to_update.add_vertex(x=x, y=y, z=z)
                        vkey_map[old_vkey] = new_vkey
                    for old_fkey in new_mesh_data_3d.faces():
                        face_vkeys_in_new_data = new_mesh_data_3d.face_vertices(old_fkey)
                        face_vkeys_in_mesh_to_update = [vkey_map[ovk] for ovk in face_vkeys_in_new_data]
                        mesh_to_update.add_face(face_vkeys_in_mesh_to_update)
                    scene_obj.update(update_data=True)
                    new_scene_objs_3d_surfaces_temp[i] = scene_obj
                    scene_objects_3d_surfaces[i] = None
                else:
                    scene_obj = viewer.scene.add(new_mesh_data_3d, name=f"3DSurface_{i}",
                                                 facecolor=Color(0.6, 0.7, 0.9, 0.7), show_edges=True,
                                                 linecolor=Color(0.1, 0.1, 0.1, 0.9), linewidth=1.0)
                    new_scene_objs_3d_surfaces_temp[i] = scene_obj
            except Exception as e:
                print(f"{print_prefix}  ERROR creating/updating 3D surface {i}: {e}")

        for old_scene_obj in scene_objects_3d_surfaces:
            if old_scene_obj is not None: viewer.scene.remove(old_scene_obj)
        scene_objects_3d_surfaces[:] = new_scene_objs_3d_surfaces_temp

        # --- 3. Update or Create Flattened Strips with Distortion Coloring ---
        new_scene_objs_flat_strips_temp = [None] * num_expected_strips
        for i in range(num_expected_strips):
            poly1_3d = generated_polylines_3d[i]
            poly2_3d = generated_polylines_3d[i+1]

            if not poly1_3d or not poly2_3d or \
               len(poly1_3d.points) != len(poly2_3d.points) or len(poly1_3d.points) < 2:
                continue
            
            try:
                if poly1_3d and poly1_3d.points:
                    start_x_3d = poly1_3d.points[0].x
                    start_y_3d = poly1_3d.points[0].y
                    flat_start_point = Point(start_x_3d, start_y_3d, FLAT_Z_OFFSET)
                else:
                    flat_start_point = Point(i * 15.0, 0, FLAT_Z_OFFSET)

                initial_unroll_direction = Vector(0,0,0)
                if len(poly1_3d.points) >= 2:
                    p0, p1 = poly1_3d.points[0], poly1_3d.points[1]
                    initial_unroll_direction = Vector.from_start_end(p0,p1) 
                if initial_unroll_direction.length < 1e-6: 
                    initial_unroll_direction = Vector(1,0,0)
                
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
                                norm_d = distortion * 10
                                norm_d = max(0.0, min(1.0, norm_d))
                                r_col, g_col = (2 * norm_d, 1.0) if norm_d <= 0.5 else (1.0, 2 * (1.0 - norm_d))
                                face_colors[fkey] = Color(r_col, g_col, 0.0, 0.8) 
                            else: 
                                face_colors[fkey] = default_face_color
                    else: 
                        for fkey in new_flattened_mesh_data.faces():
                            face_colors[fkey] = default_face_color

                    if i < len(scene_objects_flat_strips) and scene_objects_flat_strips[i] is not None:
                        scene_obj = scene_objects_flat_strips[i]
                        mesh_to_update = scene_obj.item
                        mesh_to_update.clear()
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
                                                     facecolor=face_colors, show_edges=True, 
                                                     linecolor=Color(0.2, 0.2, 0.2, 0.9), linewidth=1.0)
                        new_scene_objs_flat_strips_temp[i] = scene_obj
            except Exception as e:
                print(f"{print_prefix}  ERROR creating/updating flat strip {i}: {e}")

        for old_scene_obj in scene_objects_flat_strips:
            if old_scene_obj is not None: viewer.scene.remove(old_scene_obj)
        scene_objects_flat_strips[:] = new_scene_objs_flat_strips_temp
        viewer.renderer.update() 
    except Exception as e:
        print(f"CRITICAL ERROR in regenerate_all_geometry: {e}")
    finally:
        is_updating = False

def setup_ui():
    """Sets up the Qt UI panel for adjusting parameters."""
    global viewer, instructions_data, FLAT_Z_OFFSET

    from compas_viewer.components import Component
    ui_component = Component()
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
    
    tna_button = QPushButton("Load TNA Corrugation")
    tna_button.clicked.connect(load_tna_catenaries)
    global_layout.addWidget(tna_button)

    regenerate_button = QPushButton("Regenerate Geometry")
    regenerate_button.clicked.connect(regenerate_all_geometry) 
    global_layout.addWidget(regenerate_button)
    main_layout.addWidget(global_group)

    param_names = ["Radius", "Height", "Y-Offset", "Z-Rotation (deg)"]
    param_ranges = [(0.1, 100.0), (0.0, 50.0), (-50.0, 50.0), (-360.0, 360.0)]
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
            spinbox.valueChanged.connect(functools.partial(update_catenary_param, idx, param_idx, spinbox))
            param_layout.addWidget(label)
            param_layout.addWidget(spinbox)
            group_layout.addLayout(param_layout)
        main_layout.addWidget(group_box)

    spacer = QSpacerItem(20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)
    main_layout.addSpacerItem(spacer)
    ui_component.widget = main_widget
    viewer.ui.sidedock.add(ui_component)


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
