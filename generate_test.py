from compas.datastructures import Mesh
from compas.geometry import Plane, Vector, Point, intersection_segment_plane, Polyline
import math

def intersection_mesh_plane_simple(mesh, plane, center_coords):
    pts = []
    cx, cy = center_coords
    for u, v in mesh.edges():
        p1 = mesh.vertex_coordinates(u)
        p2 = mesh.vertex_coordinates(v)
        pt = intersection_segment_plane([p1, p2], plane)
        if pt:
            pts.append(Point(*pt))
            
    unique_pts = []
    for p in pts:
        if not any(math.hypot(p.x - up.x, p.y - up.y) < 1e-4 and abs(p.z - up.z) < 1e-4 for up in unique_pts):
            unique_pts.append(p)
            
    unique_pts.sort(key=lambda p: math.hypot(p.x - cx, p.y - cy))
    if len(unique_pts) > 1:
        return Polyline(unique_pts)
    return None

def extract_planar_creases(target_mesh, radial_planes, center_coords):
    creases = []
    for plane in radial_planes:
        pline = intersection_mesh_plane_simple(target_mesh, plane, center_coords)
        if pline:
            creases.append(pline)
    return creases

