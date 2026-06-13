import math
import numpy as np
import matplotlib.pyplot as plt
from compas.datastructures import Mesh
from compas.geometry import Point, Rotation, Translation
from code.packing import pack_strips, get_mesh_outline

def create_curved_strip(width=1.0, height=0.1, curvature=0.5, n_segments=20):
    """Creates a curved quad mesh strip for testing."""
    vertices = []
    faces = []
    
    for i in range(n_segments + 1):
        t = i / n_segments
        angle = (t - 0.5) * curvature * math.pi
        r = 1.0 / (curvature + 1e-9)
        
        # Center of arc
        cx, cy = 0, r
        
        # Inner and outer points
        x1 = cx + (r - height/2) * math.sin(angle)
        y1 = cy - (r - height/2) * math.cos(angle)
        
        x2 = cx + (r + height/2) * math.sin(angle)
        y2 = cy - (r + height/2) * math.cos(angle)
        
        vertices.extend([[x1 * width, y1 * width, 0], [x2 * width, y2 * width, 0]])
        
        if i > 0:
            idx = 2 * i
            faces.append([idx-2, idx, idx+1, idx-1])
            
    return Mesh.from_vertices_and_faces(vertices, faces)

def test_standalone_packing():
    print("Generating test strips...")
    # Create a set of varied curved strips
    strips = []
    for i in range(12):
        # Smaller strips to fit 2m sheet horizontally
        w = 0.5 + (i % 3) * 0.1
        curv = 0.3 + (i // 3) * 0.2
        strip = create_curved_strip(width=w, height=0.1, curvature=curv)
        strips.append(strip)
        from code.packing import get_mesh_2d_bbox
        mnx, mny, mxx, mxy = get_mesh_2d_bbox(strip)
        print(f"Strip {i}: Width={mxx-mnx:.2f}, Height={mxy-mny:.2f}")
        
    print(f"Packing {len(strips)} strips...")
    sheet_w, sheet_h = 2.0, 2.0
    margin = 0.02
    
    packed, success, used_dims = pack_strips(
        strips, 
        sheet_w, sheet_h, 
        margin=margin, 
        optimize_rotation=True
    )
    
    print(f"Packing completed. Success: {success}")
    print(f"Used Dimensions: {used_dims[0]:.3f}m x {used_dims[1]:.3f}m")
    
    # Calculate area utilization
    total_area = used_dims[0] * used_dims[1]
    used_strip_area = 0
    for m in strips:
        # Approximate area by bbox of segments (simple)
        used_strip_area += 0.15 * 1.0 # fixed for now
        
    print(f"Visualizing results to 'test_packing_result.png'...")
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Draw Sheet
    ax.add_patch(plt.Rectangle((0, 0), used_dims[0], used_dims[1], 
                               edgecolor='blue', facecolor='burlywood', alpha=0.1, label='Final Sheet'))
    
    for i, m in enumerate(packed):
        pts = get_mesh_outline(m)
        poly_pts = np.array(pts)[:, :2]
        ax.add_patch(plt.Polygon(poly_pts, closed=True, fill=True, 
                                  facecolor='white', edgecolor='black', alpha=0.8, linewidth=0.5))
        
    ax.set_aspect('equal')
    ax.set_xlim(-0.1, used_dims[0] + 0.1)
    ax.set_ylim(-0.1, used_dims[1] + 0.1)
    ax.set_title(f"Standalone Packing Test - {len(packed)} strips\n"
                 f"Area: {used_dims[0]:.2f} x {used_dims[1]:.2f} m")
    
    plt.savefig("test_packing_result.png", dpi=150, bbox_inches='tight')
    print("Done.")

if __name__ == "__main__":
    test_standalone_packing()
