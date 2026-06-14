import math
from compas.geometry import Point, Polyline, Vector, intersection_circle_circle_xy
from compas.datastructures import Mesh
from compas_tna.diagrams import FormDiagram
from code.vault_shared import fanvault_middle_hc, crossvault_middle_hc

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
        if i % 2 == 0:
            ids, diagram = spokes_max_ids[i], form_max
        else:
            ids, diagram = spokes_min_ids[i], form_min
        
        pts = [Point(*diagram.vertex_coordinates(v)) for v in ids]
        if corner_cut_radius > 1e-6 and len(pts) > 1:
            pts = cut_polyline_at_radius(pts, pts[0], corner_cut_radius)
        catenaries.append(Polyline(pts))
            
    return catenaries

def compute_max_safe_cut_radius(catenaries):
    """
    Computes the maximum radius that can be applied to all catenaries
    while ensuring they all retain the same number of nodes.
    """
    if not catenaries:
        return 0.0
    
    # For each catenary, find the distance to the furthest point that would
    # still leave at least 2 points (the minimum needed for a line segment).
    # However, to be safe and consistent, we want to find the radius
    # that doesn't "jump" a node on any catenary.
    
    # The limit is the distance to the 2nd node of the "shortest" spoke
    min_dist_to_second_node = float('inf')
    for cat in catenaries:
        if len(cat.points) < 2:
            continue
        p0 = cat.points[0]
        p1 = cat.points[1]
        dist = math.hypot(p1.x - p0.x, p1.y - p0.y)
        if dist < min_dist_to_second_node:
            min_dist_to_second_node = dist
            
    # We return slightly less than the absolute limit to avoid precision issues
    return max(0.0, min_dist_to_second_node - 0.01)

def generate_envelope_catenaries(config, n_spokes=10, n_points=20, corner_cut_radius=0.0):
    """
    Generates a fixed number of catenaries directly from the vault envelope
    instead of using the TNA thrust diagrams.
    """
    xy_span = config['xy_span']
    thickness = config['thickness']
    hc = config['max_rise']
    v_type = config['vault_type']
    
    x0, x1 = xy_span[0]
    y0, y1 = xy_span[1]
    xm, ym = (x0 + x1) / 2, (y0 + y1) / 2
    
    # We follow the existing logic of only using the primary corner (x0, y0)
    xc, yc = x0, y0
    dx = xm - xc
    dy = ym - yc
    
    # Split n_spokes between the two boundary edges of the quadrant
    # nx + ny = n_spokes - 1
    total_len = abs(dx) + abs(dy)
    nx = max(1, int(round((n_spokes - 1) * abs(dx) / total_len)))
    ny = (n_spokes - 1) - nx
    
    # Quadrant corners
    corners = [
        (x0, y0), (x1, y0), (x1, y1), (x0, y1)
    ]
    
    all_catenaries = []
    global_spoke_idx = 0
    
    for ci, (xc, yc) in enumerate(corners):
        # Directions towards center lines for this quadrant
        qdx = xm - xc
        qdy = ym - yc
        
        qpts = []
        # Edge 1: Along the boundary where x varies (at y=ym)
        for i in range(nx + 1):
            t = i / nx
            qpts.append((xc + t * qdx, ym))
        # Edge 2: Along the boundary where y varies (at x=xm)
        for i in range(1, ny + 1):
            t = i / ny
            qpts.append((xm, ym - t * qdy))
            
        quadrant_cats = []
        for si, (px, py) in enumerate(qpts):
            pts = []
            for j in range(n_points):
                u = j / (n_points - 1)
                x = xc + u * (px - xc)
                y = yc + u * (py - yc)
                
                # Z from envelope
                if v_type == 'fan':
                    z_mid = fanvault_middle_hc([x], [y], [x0, x1], [y0, y1], hc)[0]
                else:
                    z_mid = crossvault_middle_hc([x], [y], [x0, x1], [y0, y1], hc)[0]
                
                # Alternate Z based on intrados and extrados
                if si % 2 == 0:
                    z = z_mid + thickness / 2
                else:
                    z = z_mid - thickness / 2
                pts.append(Point(x, y, z))
                
            if corner_cut_radius > 1e-6:
                pts = cut_polyline_at_radius(pts, Point(xc, yc, pts[0].z), corner_cut_radius)
            quadrant_cats.append(Polyline(pts))
            global_spoke_idx += 1
        all_catenaries.append(quadrant_cats)
            
    return all_catenaries

def generate_support_beams(config, n_spokes=10, ply_thickness=0.012):
    """
    Generates meshes for cross-shaped support beams running along the centerlines.
    The top edge matches the zigzag pattern of the corrugation, 
    offset by ply_thickness to support the underside.
    The bottom edge is flat at z=0.
    """
    xy_span = config['xy_span']
    thickness = config['thickness']
    hc = config['max_rise']
    v_type = config['vault_type']
    
    x0, x1 = xy_span[0]
    y0, y1 = xy_span[1]
    xm, ym = (x0 + x1) / 2, (y0 + y1) / 2
    
    # Calculate how points are distributed, same logic as generate_envelope_catenaries
    dx = xm - x0
    dy = ym - y0
    total_len = abs(dx) + abs(dy)
    nx = max(1, int(round((n_spokes - 1) * abs(dx) / total_len)))
    ny = (n_spokes - 1) - nx
    
    # --- Generate X-Beam (along y=ym, from x0 to x1) ---
    x_beam_verts = []
    x_beam_faces = []
    
    # First half of X-Beam: x0 to xm (from Corner 0's spoke ends)
    pts_x_half1 = []
    for i in range(nx + 1):
        t = i / nx
        px = x0 + t * dx
        z_mid = fanvault_middle_hc([px], [ym], [x0, x1], [y0, y1], hc)[0] if v_type == 'fan' else crossvault_middle_hc([px], [ym], [x0, x1], [y0, y1], hc)[0]
        # Same alternating logic: si corresponds to the point index on this edge
        si = i
        z_top = z_mid + thickness / 2 if si % 2 == 0 else z_mid - thickness / 2
        z_top -= ply_thickness # Offset for plywood material thickness
        pts_x_half1.append((px, z_top))
        
    # Second half of X-Beam: xm to x1 (from Corner 1's spoke ends)
    # Important: Ensure the zigzag aligns at xm.
    # Corner 1 spokes on Edge 1 go from x1 to xm. We need to go from xm to x1, or construct it matching.
    pts_x_half2 = []
    dx_corner1 = xm - x1 # negative
    for i in range(1, nx + 1):
        t = i / nx
        px = x1 + (1.0 - t) * dx_corner1 # Going from xm towards x1
        z_mid = fanvault_middle_hc([px], [ym], [x0, x1], [y0, y1], hc)[0] if v_type == 'fan' else crossvault_middle_hc([px], [ym], [x0, x1], [y0, y1], hc)[0]
        # In Corner 1, the spoke index si for points on Edge 1 is also i.
        # But we are iterating backwards from xm (which was si=nx).
        # To match exactly, the point at px corresponds to si = nx - i.
        si = nx - i
        z_top = z_mid + thickness / 2 if si % 2 == 0 else z_mid - thickness / 2
        z_top -= ply_thickness # Offset for plywood material thickness
        pts_x_half2.append((px, z_top))
        
    full_x_profile = pts_x_half1 + pts_x_half2
    
    for i, (px, pz) in enumerate(full_x_profile):
        x_beam_verts.append([px, ym, pz]) # Top
        x_beam_verts.append([px, ym, 0.0]) # Bottom
        if i > 0:
            idx = i * 2
            x_beam_faces.append([idx-2, idx, idx+1, idx-1])

    x_beam_mesh = Mesh.from_vertices_and_faces(x_beam_verts, x_beam_faces)
    
    # --- Generate Y-Beam (along x=xm, from y0 to y1) ---
    y_beam_verts = []
    y_beam_faces = []
    
    # First half of Y-Beam: y0 to ym (from Corner 0's spoke ends on Edge 2)
    pts_y_half1 = []
    # In Corner 0, Edge 2 starts from ym and goes down to y0. But let's build from y0 to ym.
    for i in range(ny, -1, -1):
        t = i / ny
        py = ym - t * dy
        z_mid = fanvault_middle_hc([xm], [py], [x0, x1], [y0, y1], hc)[0] if v_type == 'fan' else crossvault_middle_hc([xm], [py], [x0, x1], [y0, y1], hc)[0]
        # For Corner 0, Edge 2, spoke index is nx + i (since i goes 1 to ny).
        # Wait, if i=0, it's the center point. 
        # In generate_envelope_catenaries: for i in range(1, ny+1): t = i/ny; py = ym - t*dy
        # So the center point (xm, ym) is actually handled by Edge 1 (si=nx).
        # To make it continuous, we can use si = nx + i
        si = nx + i
        z_top = z_mid + thickness / 2 if si % 2 == 0 else z_mid - thickness / 2
        z_top -= ply_thickness # Offset for plywood material thickness
        pts_y_half1.append((py, z_top))
        
    # Second half of Y-Beam: ym to y1 (from Corner 3 (x0, y1) or Corner 2 (x1, y1))
    pts_y_half2 = []
    # Corner 3: x0, y1. dy_c3 = ym - y1 (negative)
    # Edge 2 for Corner 3 would be on x=xm, varying y.
    dy_c3 = ym - y1
    for i in range(1, ny + 1):
        t = i / ny
        py = ym - t * dy_c3 # This goes from ym towards y1
        z_mid = fanvault_middle_hc([xm], [py], [x0, x1], [y0, y1], hc)[0] if v_type == 'fan' else crossvault_middle_hc([xm], [py], [x0, x1], [y0, y1], hc)[0]
        # In Corner 3, to match symmetry, it's si = nx + i
        si = nx + i
        z_top = z_mid + thickness / 2 if si % 2 == 0 else z_mid - thickness / 2
        z_top -= ply_thickness # Offset for plywood material thickness
        pts_y_half2.append((py, z_top))
        
    full_y_profile = pts_y_half1 + pts_y_half2
    
    for i, (py, pz) in enumerate(full_y_profile):
        y_beam_verts.append([xm, py, pz]) # Top
        y_beam_verts.append([xm, py, 0.0]) # Bottom
        if i > 0:
            idx = i * 2
            y_beam_faces.append([idx-2, idx, idx+1, idx-1])

    y_beam_mesh = Mesh.from_vertices_and_faces(y_beam_verts, y_beam_faces)
    
    return [x_beam_mesh, y_beam_mesh]

def generate_perimeter_beams(config, n_spokes=10, ply_thickness=0.012):
    """
    Generates meshes for perimeter support beams running along the 4 outer edges.
    The top edge matches the boundary spokes of the fan quadrants.
    The bottom edge is flat at z=0.
    """
    xy_span = config['xy_span']
    thickness = config['thickness']
    hc = config['max_rise']
    v_type = config['vault_type']
    
    x0, x1 = xy_span[0]
    y0, y1 = xy_span[1]
    xm, ym = (x0 + x1) / 2, (y0 + y1) / 2
    
    # Resolution for the beam curve (matching n_points in generate_envelope_catenaries)
    # n_points is passed as discr+1 in app.py. We'll use a reasonable default or pass it.
    n_points = 21 # Default resolution
    
    beams = []
    
    # Helper to generate a beam mesh along an edge
    def create_edge_beam(p_start, p_end, quadrant_indices, spoke_indices):
        verts = []
        faces = []
        
        # p_start to p_mid and p_mid to p_end
        # Each segment comes from one quadrant's boundary spoke
        for q_idx, s_idx in zip(quadrant_indices, spoke_indices):
            # Deterministic corners based on generate_envelope_catenaries
            corners = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
            xc, yc = corners[q_idx]
            
            # Reconstruct the boundary spoke points
            # Edge 1 (si=0) goes from xc,yc to xc,ym (or x1,ym etc)
            # Edge 2 (si=nx+ny) goes from xc,yc to xm,yc
            
            # We need to know if this segment goes from corner to middle or middle to corner
            # Let's simplify: we know the start and end of the segment.
            # Segment 1: p_start to p_mid. Segment 2: p_mid to p_end.
            if q_idx in [0, 1, 2, 3]: # Just a placeholder for logic
                pass
            
        # Simplified: Just sample the envelope along the edge and alternate z based on the global spoke index
        # But wait, the perimeter is ONLY the boundary spokes (si=0 or si=nx+ny).
        # So we just need the z of THAT specific spoke.
        pass

    # Actually, let's just do it manually for the 4 edges for clarity.
    dx = xm - x0
    dy = ym - y0
    total_len = abs(dx) + abs(dy)
    nx = max(1, int(round((n_spokes - 1) * abs(dx) / total_len)))
    ny = (n_spokes - 1) - nx

    # 1. Beam along y = y0 (x0 to x1)
    verts, faces = [], []
    si = nx + ny
    z_bottom = (thickness / 2 if si % 2 == 0 else -thickness / 2) - ply_thickness
    # Half 1: x0 to xm (Quad 0, si = nx+ny)
    for i in range(n_points):
        u = i / (n_points - 1)
        px = x0 + u * (xm - x0)
        z_mid = fanvault_middle_hc([px], [y0], [x0, x1], [y0, y1], hc)[0] if v_type == 'fan' else crossvault_middle_hc([px], [y0], [x0, x1], [y0, y1], hc)[0]
        z_top = z_mid + thickness / 2 if si % 2 == 0 else z_mid - thickness / 2
        z_top -= ply_thickness
        verts.extend([[px, y0, z_top], [px, y0, z_bottom]])
    # Half 2: xm to x1 (Quad 1, si = nx+ny)
    for i in range(1, n_points):
        u = i / (n_points - 1)
        px = xm + u * (x1 - xm)
        z_mid = fanvault_middle_hc([px], [y0], [x0, x1], [y0, y1], hc)[0] if v_type == 'fan' else crossvault_middle_hc([px], [y0], [x0, x1], [y0, y1], hc)[0]
        z_top = z_mid + thickness / 2 if si % 2 == 0 else z_mid - thickness / 2
        z_top -= ply_thickness
        verts.extend([[px, y0, z_top], [px, y0, z_bottom]])
    for i in range(len(verts)//2 - 1):
        idx = i * 2
        faces.append([idx, idx+2, idx+3, idx+1])
    beams.append(Mesh.from_vertices_and_faces(verts, faces))

    # 2. Beam along y = y1 (x0 to x1)
    verts, faces = [], []
    si = nx + ny
    z_bottom = (thickness / 2 if si % 2 == 0 else -thickness / 2) - ply_thickness
    # Half 1: x0 to xm (Quad 3, si = nx+ny)
    for i in range(n_points):
        u = i / (n_points - 1)
        px = x0 + u * (xm - x0)
        z_mid = fanvault_middle_hc([px], [y1], [x0, x1], [y0, y1], hc)[0] if v_type == 'fan' else crossvault_middle_hc([px], [y1], [x0, x1], [y0, y1], hc)[0]
        z_top = z_mid + thickness / 2 if si % 2 == 0 else z_mid - thickness / 2
        z_top -= ply_thickness
        verts.extend([[px, y1, z_top], [px, y1, z_bottom]])
    # Half 2: xm to x1 (Quad 2, si = nx+ny)
    for i in range(1, n_points):
        u = i / (n_points - 1)
        px = xm + u * (x1 - xm)
        z_mid = fanvault_middle_hc([px], [y1], [x0, x1], [y0, y1], hc)[0] if v_type == 'fan' else crossvault_middle_hc([px], [y1], [x0, x1], [y0, y1], hc)[0]
        z_top = z_mid + thickness / 2 if si % 2 == 0 else z_mid - thickness / 2
        z_top -= ply_thickness
        verts.extend([[px, y1, z_top], [px, y1, z_bottom]])
    for i in range(len(verts)//2 - 1):
        idx = i * 2
        faces.append([idx, idx+2, idx+3, idx+1])
    beams.append(Mesh.from_vertices_and_faces(verts, faces))

    # 3. Beam along x = x0 (y0 to y1)
    verts, faces = [], []
    si = 0
    z_bottom = (thickness / 2 if si % 2 == 0 else -thickness / 2) - ply_thickness
    # Half 1: y0 to ym (Quad 0, si = 0)
    for i in range(n_points):
        u = i / (n_points - 1)
        py = y0 + u * (ym - y0)
        z_mid = fanvault_middle_hc([x0], [py], [x0, x1], [y0, y1], hc)[0] if v_type == 'fan' else crossvault_middle_hc([x0], [py], [x0, x1], [y0, y1], hc)[0]
        z_top = z_mid + thickness / 2 if si % 2 == 0 else z_mid - thickness / 2
        z_top -= ply_thickness
        verts.extend([[x0, py, z_top], [x0, py, z_bottom]])
    # Half 2: ym to y1 (Quad 3, si = 0)
    for i in range(1, n_points):
        u = i / (n_points - 1)
        py = ym + u * (y1 - ym)
        z_mid = fanvault_middle_hc([x0], [py], [x0, x1], [y0, y1], hc)[0] if v_type == 'fan' else crossvault_middle_hc([x0], [py], [x0, x1], [y0, y1], hc)[0]
        z_top = z_mid + thickness / 2 if si % 2 == 0 else z_mid - thickness / 2
        z_top -= ply_thickness
        verts.extend([[x0, py, z_top], [x0, py, z_bottom]])
    for i in range(len(verts)//2 - 1):
        idx = i * 2
        faces.append([idx, idx+2, idx+3, idx+1])
    beams.append(Mesh.from_vertices_and_faces(verts, faces))

    # 4. Beam along x = x1 (y0 to y1)
    verts, faces = [], []
    si = 0
    z_bottom = (thickness / 2 if si % 2 == 0 else -thickness / 2) - ply_thickness
    # Half 1: y0 to ym (Quad 1, si = 0)
    for i in range(n_points):
        u = i / (n_points - 1)
        py = y0 + u * (ym - y0)
        z_mid = fanvault_middle_hc([x1], [py], [x0, x1], [y0, y1], hc)[0] if v_type == 'fan' else crossvault_middle_hc([x1], [py], [x0, x1], [y0, y1], hc)[0]
        z_top = z_mid + thickness / 2 if si % 2 == 0 else z_mid - thickness / 2
        z_top -= ply_thickness
        verts.extend([[x1, py, z_top], [x1, py, z_bottom]])
    # Half 2: ym to y1 (Quad 2, si = 0)
    for i in range(1, n_points):
        u = i / (n_points - 1)
        py = ym + u * (y1 - ym)
        z_mid = fanvault_middle_hc([x1], [py], [x0, x1], [y0, y1], hc)[0] if v_type == 'fan' else crossvault_middle_hc([x1], [py], [x0, x1], [y0, y1], hc)[0]
        z_top = z_mid + thickness / 2 if si % 2 == 0 else z_mid - thickness / 2
        z_top -= ply_thickness
        verts.extend([[x1, py, z_top], [x1, py, z_bottom]])
    for i in range(len(verts)//2 - 1):
        idx = i * 2
        faces.append([idx, idx+2, idx+3, idx+1])
    beams.append(Mesh.from_vertices_and_faces(verts, faces))


    return beams

def generate_vault_meshes(catenaries, flat_z_offset=-15.0):
    three_d_meshes = []
    flat_meshes = []
    distortions_list = []
    
    num_strips = max(0, len(catenaries) - 1)
    for i in range(num_strips):
        poly1_3d, poly2_3d = catenaries[i], catenaries[i+1]
        
        # Validation
        n1, n2 = len(poly1_3d.points), len(poly2_3d.points)
        if n1 != n2:
            raise ValueError(
                f"Catenary length mismatch at strip {i}: "
                f"Left has {n1} points, Right has {n2} points. "
                "This usually happens when 'Corner Cut Radius' is too large, "
                "causing different numbers of segments to be trimmed from adjacent spokes."
            )

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
