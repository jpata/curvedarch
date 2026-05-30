import sys
import os
import math
from compas.geometry import Point, Vector, Polyline
from code.curve_offset import develop_strip_to_plane
from code.geometry_primitives import (
    get_planar_quad, 
    get_tilted_straight_strip, 
    get_planar_curved_strip, 
    get_ruled_cylinder_strip,
    get_uneven_spokes_strip
)

def format_pt(pt):
    return f"({pt.x:.4f}, {pt.y:.4f}, {pt.z:.4f})"

def verify_strip(name, poly1, poly2):
    print(f"\n{'='*20}")
    print(f"Verifying: {name}")
    print(f"{'='*20}")
    
    start_pt = Point(0, 0, 0)
    p0, p1 = poly1.points[0], poly1.points[1]
    unroll_vec = Vector(p1.x - p0.x, p1.y - p0.y, 0)
    if unroll_vec.length < 1e-6: unroll_vec = Vector(1, 0, 0)
    else: unroll_vec.unitize()
    
    mesh, distortions = develop_strip_to_plane(poly1, poly2, start_pt, unroll_vec)
    
    if mesh is None:
        print("FAILED: Mesh generation returned None")
        return False
        
    verts = list(mesh.vertices_attributes('xyz'))
    print(f"Unfolding produced {len(verts)} vertices and {len(distortions)} segments.")
    
    N1 = len(poly1.points)
    N2 = len(poly2.points)
    N = min(N1, N2)
    
    for i in range(N - 1):
        # 3D distances
        d_p_3d = poly1.points[i].distance_to_point(poly1.points[i+1])
        d_q_3d = poly2.points[i].distance_to_point(poly2.points[i+1])
        d_diag_3d = poly1.points[i+1].distance_to_point(poly2.points[i])
        d_rung_3d = poly1.points[i].distance_to_point(poly2.points[i])
        
        # 2D distances
        idx_pc, idx_qc = 2*i, 2*i+1
        idx_pn, idx_qn = 2*i+2, 2*i+3
        v_pc, v_qc = Point(*verts[idx_pc]), Point(*verts[idx_qc])
        v_pn, v_qn = Point(*verts[idx_pn]), Point(*verts[idx_qn])
        
        d_p_2d = v_pc.distance_to_point(v_pn)
        d_q_2d = v_qc.distance_to_point(v_qn)
        d_diag_2d = v_pn.distance_to_point(v_qc)
        d_rung_2d = v_pc.distance_to_point(v_qc)
        
        print(f"Step {i}:")
        print(f"  P_3D: {format_pt(poly1.points[i])} -> {format_pt(poly1.points[i+1])} | L={d_p_3d:.4f}")
        print(f"  P_2D: {format_pt(v_pc)} -> {format_pt(v_pn)} | L={d_p_2d:.4f} | Diff={abs(d_p_3d - d_p_2d):.6f}")
        print(f"  Q_3D: {format_pt(poly2.points[i])} -> {format_pt(poly2.points[i+1])} | L={d_q_3d:.4f}")
        print(f"  Q_2D: {format_pt(v_qc)} -> {format_pt(v_qn)} | L={d_q_2d:.4f} | Diff={abs(d_q_3d - d_q_2d):.6f}")
        print(f"  Diag_3D: {d_diag_3d:.4f} | Diag_2D: {d_diag_2d:.4f} | Diff={abs(d_diag_3d - d_diag_2d):.6f}")
        print(f"  Rung_3D: {d_rung_3d:.4f} | Rung_2D: {d_rung_2d:.4f} | Diff={abs(d_rung_3d - d_rung_2d):.6f}")
        print(f"  Distortion: {distortions[i]:.6f}")
        
    max_distortion = max(distortions) if distortions else 0
    print(f"\nSummary for {name}: Max Distortion = {max_distortion:.6f}")
    return max_distortion < 1e-4

if __name__ == "__main__":
    tests = [
        ("Level 0: Planar Quad", *get_planar_quad()),
        ("Level 1: Tilted Straight", *get_tilted_straight_strip()),
        ("Level 2: Planar Curved", *get_planar_curved_strip()),
        ("Level 3: Ruled Cylinder", *get_ruled_cylinder_strip()),
        ("Level 4: Uneven Spokes", *get_uneven_spokes_strip()),
    ]
    
    all_passed = True
    for name, p1, p2 in tests:
        if not verify_strip(name, p1, p2):
            all_passed = False
            
    if not all_passed:
        print("\n!!! SOME TESTS FAILED !!!")
        sys.exit(1)
    else:
        print("\n>>> ALL PRIMITIVE TESTS PASSED <<<")
