import matplotlib.pyplot as plt
from compas.geometry import Point, Polyline, Vector
from code.curve_offset import develop_strip_to_plane
from code.geometry_primitives import (
    get_planar_quad, 
    get_tilted_straight_strip, 
    get_planar_curved_strip, 
    get_ruled_cylinder_strip
)
import os

def plot_strip(ax, mesh, title, color='blue'):
    verts = list(mesh.vertices_attributes('xyz'))
    for fkey in mesh.faces():
        face_vkeys = mesh.face_vertices(fkey)
        f_verts = [verts[vkey] for vkey in face_vkeys]
        f_verts.append(f_verts[0]) # close loop
        xs = [v[0] for v in f_verts]
        ys = [v[1] for v in f_verts]
        ax.plot(xs, ys, color=color, alpha=0.5)
        ax.fill(xs, ys, color=color, alpha=0.2)
    ax.set_title(title)
    ax.set_aspect('equal')

def verify_visually():
    tests = [
        ("Level 0: Planar Quad", *get_planar_quad()),
        ("Level 1: Tilted Straight", *get_tilted_straight_strip()),
        ("Level 2: Planar Curved", *get_planar_curved_strip()),
        ("Level 3: Ruled Cylinder", *get_ruled_cylinder_strip()),
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    os.makedirs("renders", exist_ok=True)
    
    for i, (name, poly1, poly2) in enumerate(tests):
        start_pt = Point(0, 0, 0)
        p0, p1 = poly1.points[0], poly1.points[1]
        unroll_vec = Vector(p1.x - p0.x, p1.y - p0.y, 0)
        if unroll_vec.length < 1e-6: unroll_vec = Vector(1, 0, 0)
        else: unroll_vec.unitize()
        
        mesh, _ = develop_strip_to_plane(poly1, poly2, start_pt, unroll_vec)
        if mesh:
            plot_strip(axes[i], mesh, name)
            
    plt.tight_layout()
    output_path = "renders/systematic_verification.png"
    plt.savefig(output_path)
    print(f"Visual verification saved to {output_path}")

if __name__ == "__main__":
    verify_visually()
