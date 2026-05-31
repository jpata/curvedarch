import math
import time
import functools

from PySide6 import QtCore, QtWidgets
from PySide6.QtWidgets import QWidget, QVBoxLayout, QGroupBox, QLabel, QDoubleSpinBox, QPushButton, QHBoxLayout, QSpacerItem, QSizePolicy

from compas.geometry import Point, Polyline, Translation, Rotation, Vector, Line
from compas.geometry import intersection_circle_circle_xy
from compas.colors import Color
from compas.datastructures import Mesh
from compas_viewer import Viewer
from compas_tna.diagrams import FormDiagram

# --- Configuration ---
FLAT_Z_OFFSET = -15.0 
CORNER_CUT_RADIUS = 0.5
TNA_CATENARIES = []
RAW_TNA_CATENARIES = []

# --- Global variables for scene management ---
viewer = None 
generated_polylines_3d = []
scene_objects_3d_surfaces = [] 
scene_objects_flat_strips = [] 
is_updating = False

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

def cut_polyline_at_radius(pts, center_pt, radius):
    new_pts = []
    cut_done = False
    cx, cy = center_pt.x, center_pt.y
    for i in range(len(pts) - 1):
        p1, p2 = pts[i], pts[i+1]
        if not cut_done:
            d1 = math.hypot(p1.x - cx, p1.y - cy)
            d2 = math.hypot(p2.x - cx, p2.y - cy)
            if d1 <= radius and d2 >= radius:
                dx, dy = p2.x - p1.x, p2.y - p1.y
                qx, qy = p1.x - cx, p1.y - cy
                A = dx**2 + dy**2
                B = 2 * (qx*dx + qy*dy)
                C = qx**2 + qy**2 - radius**2
                if A > 1e-9:
                    det = B**2 - 4*A*C
                    if det >= 0:
                        t1 = (-B + math.sqrt(det)) / (2*A)
                        t2 = (-B - math.sqrt(det)) / (2*A)
                        t = t1 if 0 <= t1 <= 1 else t2
                        if 0 <= t <= 1:
                            new_pts.append(Point(p1.x + t*dx, p1.y + t*dy, p1.z + t*(p2.z - p1.z)))
                            new_pts.append(p2)
                            cut_done = True
                            continue
        else:
            new_pts.append(p2)
    return new_pts if new_pts else pts

def load_tna_catenaries():
    global RAW_TNA_CATENARIES
    try:
        form_min = FormDiagram.from_json('thrust_min.json')
        form_max = FormDiagram.from_json('thrust_max.json')
        
        spokes_min_ids = extract_spokes(form_min)
        spokes_max_ids = extract_spokes(form_max)
        
        def get_spoke_angle(spoke, diagram):
            c_coords = diagram.vertex_attributes(spoke[0], names=['x', 'y'])
            n_coords = diagram.vertex_attributes(spoke[1], names=['x', 'y'])
            return math.atan2(n_coords[1] - c_coords[1], n_coords[0] - c_coords[0])

        spokes_min_ids.sort(key=lambda s: get_spoke_angle(s, form_min))
        spokes_max_ids.sort(key=lambda s: get_spoke_angle(s, form_max))
        
        RAW_TNA_CATENARIES = []
        n_spokes = min(len(spokes_min_ids), len(spokes_max_ids))
        for i in range(n_spokes):
            if i % 2 == 0:
                ids, diagram = spokes_max_ids[i], form_max
            else:
                ids, diagram = spokes_min_ids[i], form_min
            
            pts = [Point(*diagram.vertex_coordinates(v)) for v in ids]
            RAW_TNA_CATENARIES.append(pts)
        
        print(f"Loaded {len(RAW_TNA_CATENARIES)} TNA catenaries sorted by angle.")
        apply_catenary_cuts()
    except Exception as e:
        print(f"Error loading TNA data: {e}")

def apply_catenary_cuts():
    global TNA_CATENARIES, RAW_TNA_CATENARIES
    TNA_CATENARIES = []
    for pts in RAW_TNA_CATENARIES:
        if CORNER_CUT_RADIUS > 1e-6 and len(pts) > 1:
            cut_pts = cut_polyline_at_radius(pts, pts[0], CORNER_CUT_RADIUS)
            TNA_CATENARIES.append(Polyline(cut_pts))
        else:
            TNA_CATENARIES.append(Polyline(pts))
    
    regenerate_all_geometry()

# --- Unrolling Logic ---

def develop_strip_to_plane(poly1_3d, poly2_3d, start_point_on_plane, initial_unroll_vec):
    if len(poly1_3d.points) != len(poly2_3d.points) or len(poly1_3d.points) < 2:
        return None, []
        
    N = len(poly1_3d.points)
    z_plane = start_point_on_plane.z
    flat_vertices, flat_faces = [], []
    quad_distortions = []
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

        diag_3d_length = P_curr_3d.distance_to_point(Q_next_3d)
        diag_flat_length = P_curr_flat.distance_to_point(Q_next_flat) 

        current_quad_distortion = 0.0
        if diag_3d_length > 1e-9:
            current_quad_distortion = abs(diag_3d_length - diag_flat_length) / diag_3d_length
        elif diag_flat_length > 1e-9:
            current_quad_distortion = 10.0
        quad_distortions.append(current_quad_distortion)

        flat_vertices.extend([P_next_flat, Q_next_flat])
        idx_Pc, idx_Qc = 2*j, 2*j + 1
        idx_Pn, idx_Qn = len(flat_vertices)-2, len(flat_vertices)-1
        flat_faces.append([idx_Pc, idx_Pn, idx_Qn, idx_Qc]) 

        P_prev_flat = P_curr_flat 
        P_curr_flat = P_next_flat
        Q_curr_flat = Q_next_flat
    
    return Mesh.from_vertices_and_faces(flat_vertices, flat_faces), quad_distortions

# --- UI Update and Geometry Regeneration Functions ---

def update_flat_z_offset_param(control_widget, *args):
    global FLAT_Z_OFFSET, is_updating
    if is_updating: return
    try:
        FLAT_Z_OFFSET = float(control_widget.value())
        QtCore.QTimer.singleShot(100, regenerate_all_geometry)
    except: pass

def update_corner_cut_radius_param(control_widget, *args):
    global CORNER_CUT_RADIUS, is_updating
    if is_updating: return
    try:
        CORNER_CUT_RADIUS = float(control_widget.value())
        QtCore.QTimer.singleShot(100, apply_catenary_cuts)
    except: pass


def regenerate_all_geometry():
    global viewer, generated_polylines_3d, scene_objects_3d_surfaces, scene_objects_flat_strips
    global FLAT_Z_OFFSET, is_updating, TNA_CATENARIES

    if is_updating: return
    is_updating = True
    
    try:
        if viewer is None or viewer.scene is None:
            is_updating = False
            return

        generated_polylines_3d = [c.copy() for c in TNA_CATENARIES]
        num_expected_strips = max(0, len(generated_polylines_3d) - 1)
        
        # --- 2. Update or Create 3D Surface Strips ---
        new_scene_objs_3d_surfaces_temp = [None] * num_expected_strips 
        for i in range(num_expected_strips):
            poly1_3d, poly2_3d = generated_polylines_3d[i], generated_polylines_3d[i+1]
            if not poly1_3d or not poly2_3d: continue
            
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
                        face_vkeys = [vkey_map[ovk] for ovk in new_mesh_data_3d.face_vertices(old_fkey)]
                        mesh_to_update.add_face(face_vkeys)
                    scene_obj.update(update_data=True)
                    new_scene_objs_3d_surfaces_temp[i] = scene_obj
                    scene_objects_3d_surfaces[i] = None
                else:
                    scene_obj = viewer.scene.add(new_mesh_data_3d, name=f"3DSurface_{i}",
                                                 facecolor=Color(0.6, 0.7, 0.9, 0.7), show_edges=True)
                    new_scene_objs_3d_surfaces_temp[i] = scene_obj
            except Exception as e: print(f"Error 3D strip {i}: {e}")

        for old_scene_obj in scene_objects_3d_surfaces:
            if old_scene_obj is not None: viewer.scene.remove(old_scene_obj)
        scene_objects_3d_surfaces[:] = new_scene_objs_3d_surfaces_temp

        # --- 3. Update or Create Flattened Strips ---
        new_scene_objs_flat_strips_temp = [None] * num_expected_strips
        for i in range(num_expected_strips):
            poly1_3d, poly2_3d = generated_polylines_3d[i], generated_polylines_3d[i+1]
            if not poly1_3d or not poly2_3d: continue
            
            try:
                start_x_3d, start_y_3d = poly1_3d.points[0].x, poly1_3d.points[0].y
                flat_start_point = Point(start_x_3d, start_y_3d, FLAT_Z_OFFSET)
                unroll_vec = Vector.from_start_end(poly1_3d.points[0], poly1_3d.points[1]) if len(poly1_3d.points) >= 2 else Vector(1,0,0)
                
                new_flat_mesh, quad_distortions = develop_strip_to_plane(poly1_3d, poly2_3d, flat_start_point, unroll_vec)
                
                if new_flat_mesh:
                    face_colors = {}
                    if quad_distortions:
                        for f_idx, fkey in enumerate(new_flat_mesh.faces()):
                            norm_d = min(1.0, quad_distortions[f_idx] * 10)
                            r, g = (2 * norm_d, 1.0) if norm_d <= 0.5 else (1.0, 2 * (1.0 - norm_d))
                            face_colors[fkey] = Color(r, g, 0.0, 0.8) 
                    else:
                        for fkey in new_flat_mesh.faces(): face_colors[fkey] = Color(0.7, 0.9, 0.6, 0.7)

                    if i < len(scene_objects_flat_strips) and scene_objects_flat_strips[i] is not None:
                        scene_obj = scene_objects_flat_strips[i]
                        mesh_to_update = scene_obj.item
                        mesh_to_update.clear()
                        vkey_map_flat = {}
                        for old_vkey in new_flat_mesh.vertices():
                            x,y,z = new_flat_mesh.vertex_coordinates(old_vkey)
                            new_vkey = mesh_to_update.add_vertex(x=x,y=y,z=z)
                            vkey_map_flat[old_vkey] = new_vkey
                        for old_fkey in new_flat_mesh.faces():
                            face_vkeys = [vkey_map_flat[ovk] for ovk in new_flat_mesh.face_vertices(old_fkey)]
                            mesh_to_update.add_face(face_vkeys)
                        scene_obj.facecolor = face_colors 
                        scene_obj.update(update_data=True)
                        new_scene_objs_flat_strips_temp[i] = scene_obj
                        scene_objects_flat_strips[i] = None
                    else:
                        scene_obj = viewer.scene.add(new_flat_mesh, name=f"FlatStrip_{i}", facecolor=face_colors, show_edges=True)
                        new_scene_objs_flat_strips_temp[i] = scene_obj
            except Exception as e: print(f"Error flat strip {i}: {e}")

        for old_scene_obj in scene_objects_flat_strips:
            if old_scene_obj is not None: viewer.scene.remove(old_scene_obj)
        scene_objects_flat_strips[:] = new_scene_objs_flat_strips_temp
        viewer.renderer.update() 
    except Exception as e: print(f"Critical error: {e}")
    finally: is_updating = False

def setup_ui():
    global viewer, FLAT_Z_OFFSET, CORNER_CUT_RADIUS
    from compas_viewer.components import Component
    ui_component = Component()
    main_widget = QWidget()
    main_layout = QVBoxLayout(main_widget)
    main_layout.setAlignment(QtCore.Qt.AlignTop)

    group = QGroupBox("TNA Vault Unrolling")
    layout = QVBoxLayout(group)
    
    z_layout = QHBoxLayout()
    z_spinbox = QDoubleSpinBox()
    z_spinbox.setRange(-100.0, 100.0)
    z_spinbox.setValue(FLAT_Z_OFFSET)
    z_spinbox.valueChanged.connect(functools.partial(update_flat_z_offset_param, z_spinbox))
    z_layout.addWidget(QLabel("Flat Z-Offset:"))
    z_layout.addWidget(z_spinbox)
    layout.addLayout(z_layout)
    
    cut_layout = QHBoxLayout()
    cut_spinbox = QDoubleSpinBox()
    cut_spinbox.setRange(0.0, 5.0)
    cut_spinbox.setSingleStep(0.1)
    cut_spinbox.setValue(CORNER_CUT_RADIUS)
    cut_spinbox.valueChanged.connect(functools.partial(update_corner_cut_radius_param, cut_spinbox))
    cut_layout.addWidget(QLabel("Corner Cut Radius:"))
    cut_layout.addWidget(cut_spinbox)
    layout.addLayout(cut_layout)
    
    btn = QPushButton("Load TNA Data")
    btn.clicked.connect(load_tna_catenaries)
    layout.addWidget(btn)
    
    main_layout.addWidget(group)
    ui_component.widget = main_widget
    viewer.ui.sidedock.add(ui_component)

if __name__ == '__main__':
    viewer = Viewer(width=1600, height=900)
    setup_ui() 
    load_tna_catenaries() # Load automatically on start if possible
    viewer.ui.sidedock.show = True
    viewer.show()
