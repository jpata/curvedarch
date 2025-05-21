import math
import time
import functools # For partial function application in signal connections

from PySide6 import QtCore, QtWidgets
from PySide6.QtWidgets import QWidget, QVBoxLayout, QGroupBox, QLabel, QDoubleSpinBox, QPushButton, QScrollArea, QDockWidget, QHBoxLayout, QSpacerItem, QSizePolicy

import compas
from compas.geometry import Point, Polyline, Translation, Rotation, Vector, Frame 
from compas.geometry import Line as CompasLine
from compas.geometry.intersections import intersection_line_segment, intersection_circle_circle_xy
from compas.colors import Color
from compas.datastructures import Mesh
from compas_viewer import Viewer

# --- Configuration ---
NUM_CATENARY_POINTS = 21 # Number of points for discretizing catenaries
FLAT_Z_OFFSET = -15.0
GEOMETRY_TOLERANCE = 1e-6 # Tolerance for comparing points, distances, etc.
PARALLEL_TOLERANCE = 1e-6 # Tolerance for checking vector parallelism via dot product

# --- Global variables for UI interaction and scene management ---
viewer = None

# --- Geometry Functions ---
def solve_for_catenary_a(span_radius, target_arch_height, tol=1e-6, max_iter=100):
    if target_arch_height <= 1e-9: return None
    if span_radius <= 1e-9: return None

    def func_to_solve(a_param):
        if abs(a_param) < 1e-9: return float('inf')
        ratio = span_radius / a_param
        if abs(ratio) > 700: return float('inf')
        try: return a_param * (math.cosh(ratio) - 1.0) - target_arch_height
        except OverflowError: return float('inf')

    a_test_small = max(1e-9, target_arch_height * 0.1) 
    a_test_large = max(span_radius * 100, target_arch_height * 100, a_test_small + GEOMETRY_TOLERANCE) 

    val_small_a = func_to_solve(a_test_small)
    val_large_a = func_to_solve(a_test_large)

    if val_small_a == float('inf') and val_large_a == float('inf'): return None 
    if val_small_a == float('inf'): 
        a_test_small = target_arch_height 
        val_small_a = func_to_solve(a_test_small)
        if val_small_a == float('inf'): return None

    if val_small_a > 0 and val_large_a < 0:
        a_low = a_test_small
        a_high = a_test_large
    elif val_small_a < 0 and val_large_a > 0: 
        return None 
    else: 
        return None

    for iter_count in range(max_iter):
        a_mid = (a_low + a_high) / 2.0
        if a_mid == a_low or a_mid == a_high: break 
        val_mid = func_to_solve(a_mid)
        if abs(val_mid) < tol: return abs(a_mid) 
        if val_mid == float('inf'): 
            a_low = a_mid 
            continue
        if val_mid > 0: 
            a_low = a_mid
        else: 
            a_high = a_mid
            
    final_a = abs((a_low + a_high) / 2.0)
    if abs(func_to_solve(final_a)) < tol * 10 : 
        return final_a
    return None


def create_catenary_polyline(span_radius, catenary_a_param, num_points=NUM_CATENARY_POINTS):
    if not (isinstance(span_radius, (int, float)) and span_radius > GEOMETRY_TOLERANCE): return None
    if not (isinstance(catenary_a_param, (int, float)) and abs(catenary_a_param) > GEOMETRY_TOLERANCE): return None
    if num_points < 2: num_points = 2
    points = []
    a = abs(catenary_a_param) 
    try:
        cosh_val_at_radius = math.cosh(span_radius / a)
    except OverflowError: return None
    except ZeroDivisionError: return None

    for i in range(num_points):
        x_coeff = (i / (num_points - 1)) * 2.0 - 1.0 if num_points > 1 else 0.0
        x_local = x_coeff * span_radius 
        try:
            local_z_value = a * (cosh_val_at_radius - math.cosh(x_local / a))
            points.append(Point(x_local, 0, local_z_value)) 
        except OverflowError:
            if i == 0 or i == num_points -1: return None
            continue
        except ZeroDivisionError: return None
    if len(points) < 2: return None
    return Polyline(points)

# --- Custom Intersection Function ---
def _custom_intersection_plane_segment(plane_point, plane_normal, segment_p1, segment_p2, tol=GEOMETRY_TOLERANCE):
    """
    Calculates the intersection point of a plane and a line segment.
    """
    segment_vector = Vector.from_start_end(segment_p1, segment_p2)
    denominator = plane_normal.dot(segment_vector)

    if abs(denominator) < tol:
        if abs(plane_normal.dot(Vector.from_start_end(plane_point, segment_p1))) < tol:
            return None 
        else:
            return None

    numerator = plane_normal.dot(Vector.from_start_end(segment_p1, plane_point))
    t = numerator / denominator

    if -tol <= t <= 1 + tol: 
        intersection_point = segment_p1 + segment_vector.scaled(t)
        return intersection_point
    else:
        return None


def _get_point_on_polyline_at_distance(polyline, distance_from_start, tol=GEOMETRY_TOLERANCE):
    """
    Returns a Point on the polyline at a specific distance from its start point.
    """
    if not polyline or len(polyline.points) < 2:
        return None
    if distance_from_start < -tol: # Negative distance
        return polyline.points[0] 
    if distance_from_start < tol: # Effectively zero distance
        return polyline.points[0]

    current_accumulated_length = 0.0
    for i in range(len(polyline.points) - 1):
        p1 = polyline.points[i]
        p2 = polyline.points[i+1]
        segment_vector = Vector.from_start_end(p1, p2)
        segment_length = segment_vector.length

        if segment_length < tol: # Skip zero-length segments
            continue

        if current_accumulated_length + segment_length >= distance_from_start - tol:
            # The target distance falls within this segment
            remaining_dist_on_segment = distance_from_start - current_accumulated_length
            # Clamp remaining_dist_on_segment to be within [0, segment_length]
            remaining_dist_on_segment = max(0, min(remaining_dist_on_segment, segment_length))
            
            return p1 + segment_vector.unitized() * remaining_dist_on_segment
        
        current_accumulated_length += segment_length

    # If distance is beyond the polyline total length, return the last point
    return polyline.points[-1]



# --- Helper function to generate a single 3D catenary polyline object ---
def _generate_single_catenary_polyline_object(p1_coords, p2_coords, arch_value, num_points_for_poly):
    """
    Generates a single 3D Polyline object for a catenary arching upwards.
    """
    P1 = Point(*p1_coords)
    P2 = Point(*p2_coords)
    chord_vector = P2 - P1
    chord_length = chord_vector.length

    if chord_length < GEOMETRY_TOLERANCE: 
        return Polyline([P1 + chord_vector * (i / (num_points_for_poly - 1)) for i in range(num_points_for_poly)]) if num_points_for_poly >=2 else None

    if arch_value <= GEOMETRY_TOLERANCE: 
        return Polyline([P1 + chord_vector * (i / (num_points_for_poly - 1)) for i in range(num_points_for_poly)])

    catenary_param_a = solve_for_catenary_a(chord_length / 2.0, arch_value)
    if catenary_param_a is None: 
        return Polyline([P1 + chord_vector * (i / (num_points_for_poly - 1)) for i in range(num_points_for_poly)]) 

    canonical_poly = create_catenary_polyline(chord_length / 2.0, catenary_param_a, num_points_for_poly)
    if canonical_poly is None: 
        return Polyline([P1 + chord_vector * (i / (num_points_for_poly - 1)) for i in range(num_points_for_poly)]) 


    target_xaxis = chord_vector.unitized()
    world_z_up = Vector(0, 0, 1)
    
    arch_direction_vector = None
    dot_product_abs = abs(target_xaxis.dot(world_z_up))
    if (1.0 - dot_product_abs) < PARALLEL_TOLERANCE: 
        arch_direction_vector = Vector(1, 0, 0) 
        if (1.0 - abs(target_xaxis.dot(arch_direction_vector.unitized()))) < PARALLEL_TOLERANCE:
            arch_direction_vector = Vector(0,1,0) 
    else:
        plane_normal_for_vertical_plane = target_xaxis.cross(world_z_up)
        arch_direction_vector = plane_normal_for_vertical_plane.cross(target_xaxis)
        arch_direction_vector.unitize()
        if arch_direction_vector.z < 0: 
            arch_direction_vector.scale(-1)
    
    transformed_points = [P1 + target_xaxis * (pt_local.x + chord_length/2.0) + arch_direction_vector * pt_local.z for pt_local in canonical_poly.points]
    if len(transformed_points) >= 2:
        return Polyline(transformed_points)
    return None


# --- Function to Generate Vault Data ---
def generate_star_vault_data(vault_width, vault_depth, corner_z, apex_z, 
                             rib_arch_value, edge_arch_value,
                             sec_diag_endpoint_dist_on_edge, 
                             secondary_diag_arch_value):
    """
    Generates catenary instructions for a star-shaped vault.
    Secondary diagonals span diagonally between points on opposite edge catenaries.
    """
    half_width = vault_width / 2.0
    half_depth = vault_depth / 2.0

    # Corner coordinates (ordered: SW, SE, NE, NW)
    p_sw_coords = (-half_width, -half_depth, corner_z) # Index 0
    p_se_coords = ( half_width, -half_depth, corner_z) # Index 1
    p_ne_coords = ( half_width,  half_depth, corner_z) # Index 2
    p_nw_coords = (-half_width,  half_depth, corner_z) # Index 3
    
    corners_coords_list = [p_sw_coords, p_se_coords, p_ne_coords, p_nw_coords]
    p_apex_coords = (0.0, 0.0, apex_z)

    catenaries_definitions = [] 
    edge_catenary_polylines = [] 

    # 1. Define and Generate Edge Catenaries (as Polylines)
    # Edge 0: SW-SE, Edge 1: SE-NE, Edge 2: NE-NW, Edge 3: NW-SW
    edge_coords_pairs = [
        (p_sw_coords, p_se_coords), (p_se_coords, p_ne_coords),
        (p_ne_coords, p_nw_coords), (p_nw_coords, p_sw_coords)
    ]
    for idx_edge, (p1c, p2c) in enumerate(edge_coords_pairs):
        catenaries_definitions.append((*p1c, *p2c, edge_arch_value))
        edge_poly = _generate_single_catenary_polyline_object(p1c, p2c, edge_arch_value, NUM_CATENARY_POINTS)
        if edge_poly:
            edge_catenary_polylines.append(edge_poly)
        else:
            edge_catenary_polylines.append(None) 
            print(f"Error generating edge catenary {idx_edge} between {p1c} and {p2c}")

    # 2. Define Main Diagonal Rib Catenaries (Corner to Apex)
    for corner_coord in corners_coords_list:
        catenaries_definitions.append((*corner_coord, *p_apex_coords, rib_arch_value))

    # 3. Define 4 Secondary Diagonal Catenaries (spanning between points on opposite edge catenaries)
    num_edges = len(edge_catenary_polylines)
    if num_edges == 4: # Ensure we have the 4 edge polylines
        for i in range(num_edges):
            # P1 is on edge_poly_i, at sec_diag_endpoint_dist_on_edge from its start (corners_coords_list[i])
            edge_poly_1 = edge_catenary_polylines[i]
            
            # P2 is on the opposite edge: edge_poly_((i+2)%num_edges)
            # at sec_diag_endpoint_dist_on_edge from its start (corners_coords_list[(i+2)%num_edges])
            opposite_edge_index = (i + 2) % num_edges
            edge_poly_2 = edge_catenary_polylines[opposite_edge_index]

            if edge_poly_1 and edge_poly_2:
                p1_sec_diag = _get_point_on_polyline_at_distance(edge_poly_1, sec_diag_endpoint_dist_on_edge)
                
                # For the opposite edge, the distance is also from its "natural" start corner.
                # The start corner of edge_poly_2 (opposite_edge_index) is corners_coords_list[opposite_edge_index]
                p2_sec_diag = _get_point_on_polyline_at_distance(edge_poly_2, sec_diag_endpoint_dist_on_edge)

                if p1_sec_diag and p2_sec_diag:
                    catenaries_definitions.append((
                        p1_sec_diag.x, p1_sec_diag.y, p1_sec_diag.z,
                        p2_sec_diag.x, p2_sec_diag.y, p2_sec_diag.z,
                        secondary_diag_arch_value
                    ))
                else:
                    print(f"Warning: Could not determine endpoints for secondary diagonal between edge {i} and edge {opposite_edge_index}.")
            else:
                print(f"Warning: Missing edge polylines for secondary diagonal generation (edge {i} or {opposite_edge_index}).")
    else:
        print("Warning: Not enough edge catenaries generated to create secondary diagonals.")
        
    return catenaries_definitions


# Populate initial_instructions_data_tuples by calling the new function
initial_instructions_data_tuples = generate_star_vault_data(
    vault_width=26.0,
    vault_depth=40.0,
    corner_z=0.0,      
    apex_z=8.0,        
    rib_arch_value=3.0, 
    edge_arch_value=1.0, 
    sec_diag_endpoint_dist_on_edge=5.0, # Distance from corner along edge for sec. diag. endpoints
    secondary_diag_arch_value=4.0      
)

instructions_data = [list(item) for item in initial_instructions_data_tuples]

generated_polylines_3d = [] 
scene_objects_3d_surfaces = []
scene_objects_flat_strips = []
is_updating = False


def develop_strip_to_plane(poly1_3d, poly2_3d, start_point_on_plane, initial_unroll_vec):
    if len(poly1_3d.points) != len(poly2_3d.points) or len(poly1_3d.points) < 2:
        return None, []
    N = len(poly1_3d.points)
    z_plane = start_point_on_plane.z
    flat_vertices, flat_faces = [], []
    quad_distortions = []
    
    P0_3d, Q0_3d = poly1_3d.points[0], poly2_3d.points[0]
    P_curr_flat = start_point_on_plane.copy()
    flat_vertices.append(P_curr_flat)

    dist_p0q0 = P0_3d.distance_to_point(Q0_3d)
    unroll_dir_xy = Vector(initial_unroll_vec.x, initial_unroll_vec.y, 0)
    if unroll_dir_xy.length < GEOMETRY_TOLERANCE: unroll_dir_xy = Vector(1,0,0)
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

        intersections_P = intersection_circle_circle_xy(
            (P_curr_flat.xy, d_pc_pn), 
            (Q_curr_flat.xy, d_qc_pn)
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

        intersections_Q = intersection_circle_circle_xy(
            (Q_curr_flat.xy, d_qc_qn), 
            (P_next_flat.xy, d_pn_qn)
        )
        if not intersections_Q: return None, []

        Q_next_flat_chosen_xy = intersections_Q[0]
        if len(intersections_Q) > 1:
            rung_prev_dir = Vector.from_start_end(P_curr_flat, Q_curr_flat).unitized()
            if rung_prev_dir.length < GEOMETRY_TOLERANCE:
                main_dir_P = Vector.from_start_end(P_curr_flat, P_next_flat).unitized()
                if main_dir_P.length < GEOMETRY_TOLERANCE: main_dir_P = unroll_dir_xy 
                vec_PN_Q0 = Vector.from_start_end(P_next_flat, Point(intersections_Q[0][0], intersections_Q[0][1], z_plane))
                vec_PN_Q1 = Vector.from_start_end(P_next_flat, Point(intersections_Q[1][0], intersections_Q[1][1], z_plane))
                ref_perp_dir = Vector(-main_dir_P.y, main_dir_P.x, 0)
                if abs(vec_PN_Q0.dot(ref_perp_dir)) >= abs(vec_PN_Q1.dot(ref_perp_dir)):
                    Q_next_flat_chosen_xy = intersections_Q[0]
                else:
                    Q_next_flat_chosen_xy = intersections_Q[1]
            else:
                r0 = Vector(intersections_Q[0][0]-P_next_flat.x, intersections_Q[0][1]-P_next_flat.y, 0).unitized()
                r1 = Vector(intersections_Q[1][0]-P_next_flat.x, intersections_Q[1][1]-P_next_flat.y, 0).unitized()
                Q_next_flat_chosen_xy = intersections_Q[0] if r0.dot(rung_prev_dir) >= r1.dot(rung_prev_dir) else intersections_Q[1]
        Q_next_flat = Point(Q_next_flat_chosen_xy[0], Q_next_flat_chosen_xy[1], z_plane)

        diag_3d_length = P_curr_3d.distance_to_point(Q_next_3d)
        diag_flat_length = P_curr_flat.distance_to_point(Q_next_flat)
        current_quad_distortion = 0.0
        if diag_3d_length > 1e-9:
            current_quad_distortion = abs(diag_3d_length - diag_flat_length) / diag_3d_length
        elif diag_flat_length > 1e-9:
            current_quad_distortion = 10.0
        quad_distortions.append(current_quad_distortion)

        idx_Pc, idx_Qc = 2*j, 2*j + 1
        if idx_Pc >= len(flat_vertices) or idx_Qc >= len(flat_vertices) : return None, []

        flat_vertices.extend([P_next_flat, Q_next_flat])
        idx_Pn, idx_Qn = len(flat_vertices)-2, len(flat_vertices)-1
        flat_faces.append([idx_Pc, idx_Pn, idx_Qn, idx_Qc])

        P_prev_flat = P_curr_flat
        P_curr_flat = P_next_flat
        Q_curr_flat = Q_next_flat

    if not flat_vertices or not flat_faces: return None, []
    return Mesh.from_vertices_and_faces(flat_vertices, flat_faces), quad_distortions

# --- SYNCLASTIC HELPER FUNCTIONS (Conceptual Placeholders) ---
def detect_intersections_and_segment_polylines(polylines):
    print(f"{' '*2}[DetectIntersections] Placeholder: No actual intersection detection implemented.")
    segmented_polylines = [poly.copy() for poly in polylines if poly]
    return segmented_polylines

def identify_surface_forming_pairs(segmented_polylines, connectivity_graph=None):
    print(f"{' '*2}[IdentifyPairs] Placeholder: Using simple sequential pairing for now.")
    surface_pairs = []
    num_segments = len(segmented_polylines)
    if num_segments >= 2:
        for i in range(num_segments - 1): 
            poly1 = segmented_polylines[i]
            poly2 = segmented_polylines[i+1]
            if poly1 and poly2 and len(poly1.points) == len(poly2.points) and len(poly1.points) >=2:
                surface_pairs.append((poly1, poly2, i))
            else:
                print(f"{' '*4}Skipping pair {i}, {i+1} due to invalid polylines or point mismatch for strip definition.")
    return surface_pairs

# --- UI Update and Geometry Regeneration Functions ---
def update_catenary_param(catenary_idx, param_idx, control_widget, *args):
    global instructions_data, is_updating
    if is_updating: return
    try:
        value = float(control_widget.value())
        instructions_data[catenary_idx][param_idx] = value
        QtCore.QTimer.singleShot(10, regenerate_all_geometry)
    except Exception as e: print(f"Error in update_catenary_param: {e}")

def update_flat_z_offset_param(control_widget, *args):
    global FLAT_Z_OFFSET, is_updating
    if is_updating: return
    try:
        FLAT_Z_OFFSET = float(control_widget.value())
        QtCore.QTimer.singleShot(10, regenerate_all_geometry)
    except Exception as e: print(f"Error in update_flat_z_offset_param: {e}")


def regenerate_all_geometry():
    global viewer, generated_polylines_3d, scene_objects_3d_surfaces, scene_objects_flat_strips
    global instructions_data, FLAT_Z_OFFSET, is_updating

    if viewer:
        if hasattr(viewer, 'scene') and hasattr(viewer.scene, 'clear_objects'):
            viewer.scene.clear_objects()
        elif hasattr(viewer, 'clear'):
             viewer.clear()
        else: 
            for scene_list_to_clear in [scene_objects_3d_surfaces, scene_objects_flat_strips]:
                for scene_obj_to_clear in scene_list_to_clear:
                    if scene_obj_to_clear and viewer and viewer.scene and scene_obj_to_clear in viewer.scene.objects:
                        try:
                            viewer.scene.remove(scene_obj_to_clear)
                        except Exception as e_remove:
                            print(f"Error removing object: {e_remove}")
                scene_list_to_clear.clear()
    scene_objects_3d_surfaces.clear() 
    scene_objects_flat_strips.clear()


    if is_updating: return
    is_updating = True
    print_prefix = "  [RegenGeom] "
    print(f"\n{print_prefix}{'-'*20} STARTING GEOMETRY REGENERATION {'-'*20}")

    try:
        if viewer is None or viewer.scene is None:
            print(f"{print_prefix}ERROR: Viewer or Viewer.scene is None!")
            is_updating = False; return

        print(f"{print_prefix}Step 1: Generating initial 3D Catenary Polylines...")
        generated_polylines_3d.clear() 
        
        for idx, params in enumerate(instructions_data): 
            p1_coords, p2_coords, arch_value = params[0:3], params[3:6], params[6]
            
            current_poly = _generate_single_catenary_polyline_object(p1_coords, p2_coords, arch_value, NUM_CATENARY_POINTS)
            
            if current_poly:
                generated_polylines_3d.append(current_poly)
                viewer.scene.add(current_poly, name=f"CatenaryCurve_{idx}", linescolor=Color.black(), linewidth=2)
            else:
                generated_polylines_3d.append(None) 
                print(f"{print_prefix}  Failed to generate polyline for instruction {idx}: P1={p1_coords}, P2={p2_coords}, Arch={arch_value}")

        print(f"{print_prefix}  Generated {sum(1 for p in generated_polylines_3d if p)} actual polylines.")

        valid_polylines_for_surfacing = [poly for poly in generated_polylines_3d if poly is not None]

        processed_segments = detect_intersections_and_segment_polylines(valid_polylines_for_surfacing)
        surface_strip_definitions = identify_surface_forming_pairs(processed_segments)
        print(f"{print_prefix}  Identified {len(surface_strip_definitions)} surface strip definitions.")

        print(f"\n{print_prefix}Step 2: Creating 3D Surfaces ({len(surface_strip_definitions)} expected)...")
        for i, strip_definition in enumerate(surface_strip_definitions):
            poly1_3d, poly2_3d, original_idx_hint = strip_definition
            if not poly1_3d or not poly2_3d or \
               len(poly1_3d.points) != len(poly2_3d.points) or len(poly1_3d.points) < 2:
                continue
            try:
                strip_vertices_3d = poly1_3d.points + poly2_3d.points
                strip_faces_3d = [ [j_seg, j_seg + 1, len(poly1_3d.points) + j_seg + 1, len(poly1_3d.points) + j_seg] for j_seg in range(len(poly1_3d.points) - 1) ]
                new_mesh_data_3d = Mesh.from_vertices_and_faces(strip_vertices_3d, strip_faces_3d)
                new_mesh_data_3d.weld(GEOMETRY_TOLERANCE) 
                scene_obj_name = f"3DSurface_{original_idx_hint}"
                scene_obj = viewer.scene.add(new_mesh_data_3d, name=scene_obj_name,
                                             facecolor=Color(0.6, 0.7, 0.9, 0.7), show_edges=True,
                                             linecolor=Color(0.1, 0.1, 0.1, 0.9), linewidth=1.0)
                scene_objects_3d_surfaces.append(scene_obj) 
            except Exception as e:
                print(f"{print_prefix}  ERROR creating 3D surface {i} (hint {original_idx_hint}): {e}")

        print(f"\n{print_prefix}Step 3: Creating Flattened Strips ({len(surface_strip_definitions)} expected)...")
        current_flat_strip_offset_x = 0.0 
        default_strip_width_estimate = 20.0 
        y_offset_for_flat_layout = -30.0  
        base_y_coord_for_flat_strips = (instructions_data[0][1] if instructions_data else 0) + y_offset_for_flat_layout

        for i, strip_definition in enumerate(surface_strip_definitions):
            poly1_3d, poly2_3d, original_idx_hint = strip_definition
            if not poly1_3d or not poly2_3d or \
               len(poly1_3d.points) != len(poly2_3d.points) or len(poly1_3d.points) < 2:
                continue
            try:
                flat_start_point = Point(current_flat_strip_offset_x, 
                                         base_y_coord_for_flat_strips, 
                                         FLAT_Z_OFFSET)
                initial_unroll_direction = Vector.from_start_end(poly1_3d.points[0], poly1_3d.points[1]) if len(poly1_3d.points) >= 2 else Vector(1,0,0)
                if initial_unroll_direction.length < GEOMETRY_TOLERANCE: initial_unroll_direction = Vector(1,0,0)

                new_flattened_mesh_data, quad_distortions = develop_strip_to_plane(poly1_3d, poly2_3d, flat_start_point, initial_unroll_direction)
                if new_flattened_mesh_data:
                    new_flattened_mesh_data.weld(GEOMETRY_TOLERANCE)
                    face_colors = {}
                    default_face_color = Color(0.7, 0.9, 0.6, 0.7)
                    if quad_distortions:
                        min_d, max_d = (min(quad_distortions), max(quad_distortions)) if quad_distortions else (0,0)
                        delta_d = max_d - min_d
                        for f_idx, fkey in enumerate(new_flattened_mesh_data.faces()):
                            if f_idx < len(quad_distortions):
                                distortion, norm_d = quad_distortions[f_idx], 0.0
                                if delta_d > 1e-9: norm_d = (distortion - min_d) / delta_d
                                norm_d = max(0.0, min(1.0, norm_d * 10)) 
                                r_col, g_col = (2 * norm_d, 1.0) if norm_d <= 0.5 else (1.0, 2 * (1.0 - norm_d))
                                face_colors[fkey] = Color(r_col, g_col, 0.0, 0.8)
                            else: face_colors[fkey] = default_face_color
                    else: face_colors = {fkey: default_face_color for fkey in new_flattened_mesh_data.faces()}
                    
                    scene_obj_name = f"FlatStrip_{original_idx_hint}"
                    scene_obj = viewer.scene.add(new_flattened_mesh_data, name=scene_obj_name,
                                                 facecolor=face_colors, show_edges=True,
                                                 linecolor=Color(0.2, 0.2, 0.2, 0.9), linewidth=1.0)
                    scene_objects_flat_strips.append(scene_obj) 
                    if scene_obj.item: 
                        try:
                            bb_xy = scene_obj.item.bounding_box_xy() 
                            strip_width = bb_xy[1][0] - bb_xy[0][0]
                            current_flat_strip_offset_x += strip_width + 5.0 
                        except Exception:
                            current_flat_strip_offset_x += default_strip_width_estimate 
                    else:
                        current_flat_strip_offset_x += default_strip_width_estimate
                else: print(f"{print_prefix}  ERROR: develop_strip_to_plane returned None for flat strip {i}.")
            except Exception as e:
                print(f"{print_prefix}  ERROR creating flat strip {i} (hint {original_idx_hint}): {e}")

        print(f"\n{print_prefix}Processing Qt Events & Forcing viewer renderer update...")
        if viewer.app: 
            viewer.app.processEvents() 
        if viewer.renderer:
            viewer.renderer.update() 
        if hasattr(viewer, 'view') and hasattr(viewer.view, 'update'):
            try:
                viewer.view.update()
            except Exception as e_view_update:
                print(f"{print_prefix}  Note: viewer.view.update() failed: {e_view_update}")
        
        print(f"{print_prefix}{'-'*20} GEOMETRY REGENERATION COMPLETE {'-'*20}")

    except Exception as e:
        print(f"CRITICAL ERROR in regenerate_all_geometry: {e}")
        import traceback; traceback.print_exc()
    finally:
        is_updating = False

def setup_ui():
    global viewer, instructions_data, FLAT_Z_OFFSET
    dock = QDockWidget("Catenary Parameters")
    dock.setFeatures(QDockWidget.DockWidgetFeature.DockWidgetMovable | QDockWidget.DockWidgetFeature.DockWidgetFloatable)
    scroll_area = QScrollArea(); scroll_area.setWidgetResizable(True)
    main_widget = QWidget(); main_layout = QVBoxLayout(main_widget); main_layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop)
    
    global_group = QGroupBox("Global Settings"); global_layout = QVBoxLayout(global_group)
    flat_z_layout = QHBoxLayout(); flat_z_label = QLabel("Flat Strips Z-Offset:")
    flat_z_spinbox = QDoubleSpinBox(); flat_z_spinbox.setRange(-100.0, 100.0); flat_z_spinbox.setSingleStep(0.5); flat_z_spinbox.setValue(FLAT_Z_OFFSET)
    flat_z_spinbox.valueChanged.connect(functools.partial(update_flat_z_offset_param, flat_z_spinbox))
    flat_z_layout.addWidget(flat_z_label); flat_z_layout.addWidget(flat_z_spinbox); global_layout.addLayout(flat_z_layout)
    regenerate_button = QPushButton("Regenerate Geometry"); regenerate_button.clicked.connect(regenerate_all_geometry)
    global_layout.addWidget(regenerate_button); main_layout.addWidget(global_group)

    param_labels = [
        "P1 X", "P1 Y", "P1 Z",
        "P2 X", "P2 Y", "P2 Z",
        "Arch/Rise" 
    ]
    param_config = [ 
        (-50.0, 50.0, 0.5), (-50.0, 50.0, 0.5), (-50.0, 50.0, 0.5), 
        (-50.0, 50.0, 0.5), (-50.0, 50.0, 0.5), (-50.0, 50.0, 0.5), 
        (0.1, 50.0, 0.1)                                            
    ]

    max_catenaries_in_ui = 12 
    display_instructions = instructions_data[:max_catenaries_in_ui]


    for idx, catenary_params_list in enumerate(display_instructions): 
        group_box = QGroupBox(f"Catenary {idx + 1}"); group_layout = QVBoxLayout(group_box)
        for param_idx, param_label_text in enumerate(param_labels):
            param_layout = QHBoxLayout(); label = QLabel(f"{param_label_text}:"); spinbox = QDoubleSpinBox()
            min_val, max_val, step_val = param_config[param_idx]
            spinbox.setRange(min_val, max_val); spinbox.setSingleStep(step_val)
            spinbox.setValue(catenary_params_list[param_idx]) 
            spinbox.valueChanged.connect(functools.partial(update_catenary_param, idx, param_idx, spinbox))
            param_layout.addWidget(label); param_layout.addWidget(spinbox); group_layout.addLayout(param_layout)
        main_layout.addWidget(group_box)

    if len(instructions_data) > max_catenaries_in_ui:
        main_layout.addWidget(QLabel(f"... and {len(instructions_data) - max_catenaries_in_ui} more catenaries defined programmatically."))


    main_layout.addSpacerItem(QSpacerItem(20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding))
    scroll_area.setWidget(main_widget); dock.setWidget(scroll_area)
    
    try:
        if hasattr(viewer, 'add_dock'): 
            viewer.add_dock(dock_widget=dock, name="Params", area='left', floating=False)
        elif hasattr(viewer.ui, 'sidedock') and isinstance(viewer.ui.sidedock, QtWidgets.QTabWidget): 
             viewer.ui.sidedock.addTab(dock, "Params")
        elif hasattr(viewer.ui, 'sidedock') and hasattr(viewer.ui.sidedock, 'add'): 
            viewer.ui.sidedock.add(dock) 
        else: 
            if hasattr(viewer.ui, 'sidedock'):
                 viewer.ui.sidedock = dock 
                 print(f"Note: Directly assigned dock to viewer.ui.sidedock.")
            else:
                 print("Could not automatically add dock widget to viewer UI.")
    except Exception as e:
        print(f"Error adding dock widget: {e}")


# --- Main script execution ---
if __name__ == '__main__':
    print("="*60 + "\nSTARTING CATENARY VIEWER APPLICATION\n" + "="*60)
    try:
        viewer = Viewer(width=1600, height=900, show_grid=True, drift_speed=0.0, rotation_speed=0.01)
    except TypeError: 
        viewer = Viewer(width=1600, height=900, show_grid=True)
        print("Note: DRIFT_SPEED and ROTATION_SPEED not available in this compas_viewer version.")

    print(f"Viewer created: {viewer}")
    setup_ui()
    print("\nGenerating initial geometry...")
    regenerate_all_geometry() 
    
    if viewer.app: viewer.app.processEvents()
    if viewer.renderer: viewer.renderer.update()
    if hasattr(viewer, 'view') and hasattr(viewer.view, 'update'): viewer.view.update()
            
    print("\n" + "="*60 + "\nLAUNCHING COMPAS VIEWER\n" + "="*60)
    
    viewer.ui.sidedock.show = True
    
    viewer.show()

