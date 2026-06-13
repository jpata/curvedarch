import math
import numpy as np
from compas.geometry import Point, Rotation, Translation, Vector, intersection_segment_segment_xy, is_point_in_polygon_xy

def get_mesh_2d_bbox(mesh):
    """Returns (min_x, min_y, max_x, max_y) for a flat mesh."""
    vertices = [mesh.vertex_coordinates(v) for v in mesh.vertices()]
    xs = [v[0] for v in vertices]
    ys = [v[1] for v in vertices]
    return min(xs), min(ys), max(xs), max(ys)

def get_mesh_outline(mesh):
    """
    Extracts the boundary polygon points for a quad strip mesh.
    Assumes vertices are ordered [P0, Q0, P1, Q1, ...]
    """
    vertices = [mesh.vertex_coordinates(v) for v in mesh.vertices()]
    # P side: 0, 2, 4...
    p_side = vertices[0::2]
    # Q side: 1, 3, 5... (reversed to close the loop)
    q_side = vertices[1::2][::-1]
    return p_side + q_side

def polygons_intersect(pts1, pts2, margin=0.0):
    """Checks if two polygons intersect using COMPAS primitives."""
    # 1. BBox check (fast)
    b1 = [min(p[0] for p in pts1), min(p[1] for p in pts1), max(p[0] for p in pts1), max(p[1] for p in pts1)]
    b2 = [min(p[0] for p in pts2), min(p[1] for p in pts2), max(p[0] for p in pts2), max(p[1] for p in pts2)]
    
    # Apply margin to b2 for safety
    b2 = [b2[0]-margin, b2[1]-margin, b2[2]+margin, b2[3]+margin]
    
    if b1[2] < b2[0] or b1[0] > b2[2] or b1[3] < b2[1] or b1[1] > b2[3]:
        return False
    
    # 2. Segment-Segment intersection
    for i in range(len(pts1)):
        s1 = (pts1[i], pts1[(i+1)%len(pts1)])
        for j in range(len(pts2)):
            s2 = (pts2[j], pts2[(j+1)%len(pts2)])
            res = intersection_segment_segment_xy(s1, s2)
            if res: return True
                
    # 3. Point in Polygon (for containment)
    if is_point_in_polygon_xy(pts1[0], pts2): return True
    if is_point_in_polygon_xy(pts2[0], pts1): return True
    
    return False

def get_profiles(pts, resolution=0.01):
    """
    Computes top and bottom profiles of a polygon at given x-resolution.
    Returns (xs, bottoms, tops) arrays.
    """
    min_x = min(p[0] for p in pts)
    max_x = max(p[0] for p in pts)
    
    # Use linspace for more predictable counts
    n_pts = int(round((max_x - min_x) / resolution)) + 1
    xs = np.linspace(min_x, max_x, n_pts)
    
    bottoms = np.full(len(xs), float('inf'))
    tops = np.full(len(xs), float('-inf'))
    
    for i in range(len(pts)):
        p1, p2 = pts[i], pts[(i+1)%len(pts)]
        x1, y1 = p1[0], p1[1]
        x2, y2 = p2[0], p2[1]
        
        if abs(x2 - x1) < 1e-9:
            # Vertical segment: update at one x-bin
            idx = int(round((x1 - min_x) / resolution))
            if 0 <= idx < len(xs):
                bottoms[idx] = min(bottoms[idx], y1, y2)
                tops[idx] = max(tops[idx], y1, y2)
            continue
            
        # Range of bins covered
        x_min, x_max = min(x1, x2), max(x1, x2)
        idx_start = int(math.ceil((x_min - min_x) / resolution))
        idx_end = int(math.floor((x_max - min_x) / resolution))
        
        for idx in range(max(0, idx_start), min(len(xs), idx_end + 1)):
            x = xs[idx]
            y = y1 + (y2 - y1) * (x - x1) / (x2 - x1)
            bottoms[idx] = min(bottoms[idx], y)
            tops[idx] = max(tops[idx], y)
            
    # Fill gaps by interpolation
    valid_mask = (bottoms != float('inf'))
    if not np.all(valid_mask):
        valid_indices = np.where(valid_mask)[0]
        if len(valid_indices) > 0:
            bottoms = np.interp(xs, xs[valid_mask], bottoms[valid_mask])
            tops = np.interp(xs, xs[valid_mask], tops[valid_mask])
    
    return xs, bottoms, tops

def _prepare_items(meshes, optimize_rotation, res):
    items = []
    for i, m in enumerate(meshes):
        m_oriented = m.copy()
        vertices = [m_oriented.vertex_coordinates(v) for v in m_oriented.vertices()]
        if len(vertices) >= 4:
            p_start = Point(*vertices[0])
            p_end = Point(*vertices[-2]) 
            vec = Vector.from_start_end(p_start, p_end)
            angle = math.atan2(vec.y, vec.x)
            R = Rotation.from_axis_and_angle([0, 0, 1], -angle, point=p_start)
            m_oriented.transform(R)
        
        variants = []
        possible_angles = [0, math.pi/2, math.pi, 3*math.pi/2] if optimize_rotation else [0]
        
        for angle in possible_angles:
            m_rot = m_oriented.copy()
            if angle != 0:
                R = Rotation.from_axis_and_angle([0, 0, 1], angle, point=[0,0,0])
                m_rot.transform(R)
            
            mxr, myr, _, _ = get_mesh_2d_bbox(m_rot)
            m_rot.transform(Translation.from_vector([-mxr, -myr, 0]))
            
            pts_r = get_mesh_outline(m_rot)
            xs_r, b_r, t_r = get_profiles(pts_r, resolution=res)
            
            variants.append({
                'mesh': m_rot, 
                'bottoms': b_r, 
                'tops': t_r, 
                'w': max(xs_r),
                'n_bins': len(xs_r)
            })
        items.append({'index': i, 'variants': variants})
    return items

def _pack_one_sheet(items, sheet_width, sheet_height, margin, res):
    """Internal helper to pack as many items as possible onto one sheet."""
    total_bins = int(math.ceil(sheet_width / res)) + 1
    skyline = np.zeros(total_bins)
    packed_meshes = []
    packed_indices = []
    
    for i, item in enumerate(items):
        best_h = float('inf')
        best_config = None 
        
        for variant in item['variants']:
            w = variant['w']
            n_bins = variant['n_bins']
            max_ox_bin = int(math.floor((sheet_width - 2*margin - w) / res))
            
            for ox_bin in range(max_ox_bin + 1):
                v = 0
                for j in range(n_bins):
                    needed = skyline[ox_bin + j] + margin - variant['bottoms'][j]
                    if needed > v: v = needed
                
                # Check if this item fits within sheet height
                peak_h = 0
                for j in range(n_bins):
                    h_at_j = v + variant['tops'][j]
                    if h_at_j > peak_h: peak_h = h_at_j
                
                if peak_h <= (sheet_height - 2*margin):
                    if peak_h < best_h:
                        best_h = peak_h
                        best_config = (variant, ox_bin, v)
        
        if best_config:
            variant, ox_bin, v = best_config
            ox = ox_bin * res
            placed_m = variant['mesh'].copy()
            placed_m.transform(Translation.from_vector([margin + ox, margin + v, 0]))
            packed_meshes.append(placed_m)
            packed_indices.append(i)
            
            for j in range(variant['n_bins']):
                skyline[ox_bin + j] = max(skyline[ox_bin + j], v + variant['tops'][j])

    max_h_used = np.max(skyline) if len(packed_meshes) > 0 else 0
    max_w_used = 0
    if packed_meshes:
        for m in packed_meshes:
            _, _, mxx, _ = get_mesh_2d_bbox(m)
            if mxx > max_w_used: max_w_used = mxx
            
    return packed_meshes, packed_indices, (max_w_used + margin, max_h_used + margin)

def pack_strips(meshes, sheet_width, sheet_height, margin=0.02, optimize_rotation=True):
    """Legacy wrapper for single sheet packing. If height is too small, it still packs everything but returns success=False."""
    res = 0.02
    items = _prepare_items(meshes, optimize_rotation, res)
    items.sort(key=lambda x: x['variants'][0]['w'], reverse=True)
    
    # In legacy mode, we force everything to pack even if it exceeds height, 
    # so we use a huge height for the internal call then check the result.
    packed_meshes, _, used_dims = _pack_one_sheet(items, sheet_width, 1e6, margin, res)
    
    success = used_dims[1] <= sheet_height
    return packed_meshes, success, used_dims

def pack_strips_multi(meshes, sheet_width, sheet_height, margin=0.02, optimize_rotation=True):
    """Packs meshes onto as many sheets of fixed size as needed."""
    res = 0.02
    remaining_items = _prepare_items(meshes, optimize_rotation, res)
    remaining_items.sort(key=lambda x: x['variants'][0]['w'], reverse=True)
    
    sheets = []
    while remaining_items:
        packed_meshes, packed_indices, used_dims = _pack_one_sheet(remaining_items, sheet_width, sheet_height, margin, res)
        
        if not packed_indices:
            # Could not fit even a single item on a fresh sheet
            # This happens if an item is wider than the sheet
            break
            
        sheets.append({
            'meshes': packed_meshes,
            'dims': used_dims
        })
        
        # Remove packed items
        remaining_items = [item for i, item in enumerate(remaining_items) if i not in packed_indices]
        
    return sheets

