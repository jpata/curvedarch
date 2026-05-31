import math
from compas.geometry import Point, Polyline, Vector, intersection_circle_circle_xy
from compas.datastructures import Mesh
from compas_tna.diagrams import FormDiagram

def _get_vector(v1, v2):
    return (v2['x'] - v1['x'], v2['y'] - v1['y'])

def _normalize(v):
    L = math.sqrt(v[0]**2 + v[1]**2)
    if L == 0: return (0, 0)
    return (v[0]/L, v[1]/L)

def _dot(v1, v2):
    return v1[0]*v2[0] + v1[1]*v2[1]

def extract_spokes(diagram, target_corner_coords=(0.0, 0.0), center_coords=(5.0, 5.0)):
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
            if abs(v_curr['x'] - center_coords[0]) < 1e-6 or abs(v_curr['y'] - center_coords[1]) < 1e-6:
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

def get_alternating_catenaries(form_min, form_max, corner_cut_radius=0.5, center_coords=(5.0, 5.0)):
    if isinstance(form_min, str):
        form_min = FormDiagram.from_json(form_min)
    if isinstance(form_max, str):
        form_max = FormDiagram.from_json(form_max)
    
    spokes_min_ids = extract_spokes(form_min, center_coords=center_coords)
    spokes_max_ids = extract_spokes(form_max, center_coords=center_coords)
    
    def get_spoke_angle(spoke, diagram):
        c_coords = diagram.vertex_attributes(spoke[0], names=['x', 'y'])
        n_coords = diagram.vertex_attributes(spoke[1], names=['x', 'y'])
        return math.atan2(n_coords[1] - c_coords[1], n_coords[0] - c_coords[0])

    spokes_min_ids.sort(key=lambda s: get_spoke_angle(s, form_min))
    spokes_max_ids.sort(key=lambda s: get_spoke_angle(s, form_max))
    
    catenaries = []
    n_spokes = min(len(spokes_min_ids), len(spokes_max_ids))
    for i in range(n_spokes):
        # Add MAX (Ridge)
        ids_max, diagram_max = spokes_max_ids[i], form_max
        pts_max = [Point(*diagram_max.vertex_coordinates(v)) for v in ids_max]
        if corner_cut_radius > 1e-6 and len(pts_max) > 1:
            pts_max = cut_polyline_at_radius(pts_max, pts_max[0], corner_cut_radius)
        catenaries.append(Polyline(pts_max))

        # Add MIN (Valley)
        ids_min, diagram_min = spokes_min_ids[i], form_min
        pts_min = [Point(*diagram_min.vertex_coordinates(v)) for v in ids_min]
        if corner_cut_radius > 1e-6 and len(pts_min) > 1:
            pts_min = cut_polyline_at_radius(pts_min, pts_min[0], corner_cut_radius)
        catenaries.append(Polyline(pts_min))
            
    return catenaries

def generate_vault_meshes(catenaries, flat_z_offset=-15.0):
    three_d_meshes = []
    flat_meshes = []
    distortions_list = []
    
    num_strips = max(0, len(catenaries) - 1)
    for i in range(num_strips):
        poly1_3d, poly2_3d = catenaries[i], catenaries[i+1]
        
        # 3D Mesh
        verts_3d = poly1_3d.points + poly2_3d.points
        faces_3d = []
        offset = len(poly1_3d.points)
        for j in range(len(poly1_3d.points) - 1):
            faces_3d.append([j, j+1, offset + j + 1, offset + j])
        three_d_meshes.append(Mesh.from_vertices_and_faces(verts_3d, faces_3d))
        
        # Flat Mesh
        start_x, start_y = poly1_3d.points[0].x, poly1_3d.points[0].y
        flat_start = Point(start_x, start_y, flat_z_offset)
        unroll_vec = Vector.from_start_end(poly1_3d.points[0], poly1_3d.points[1])
        
        flat_mesh, quad_distortions = develop_strip_to_plane(poly1_3d, poly2_3d, flat_start, unroll_vec)
        flat_meshes.append(flat_mesh)
        distortions_list.append(quad_distortions)
        
    return three_d_meshes, flat_meshes, distortions_list
