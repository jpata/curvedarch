#LIBGL_ALWAYS_SOFTWARE=1 xvfb-run -a uv run python3 code/render_vault.py
import os
import pathlib
import matplotlib.pyplot as plt
import numpy as np
from compas.data import json_load
from compas_viewer import Viewer

import math

def render_geometry_matplotlib(json_file, output_prefix, view='top'):
    """Fallback renderer using matplotlib for 2D and 3D projections."""
    if not os.path.exists(json_file):
        return False

    print(f"Using matplotlib {view} view for {json_file}...")
    try:
        data = json_load(json_file)
        meshes = data.get('meshes', [])
        lines = data.get('lines', [])
        points = data.get('points', [])
        edges = data.get('edge_point_indices', [])

        # If lines are missing but edges/points exist, reconstruct them
        if not lines and edges and points:
            from compas.geometry import Line
            for u, v in edges:
                lines.append(Line(points[u], points[v]))

        output_dir = pathlib.Path("renders")
        output_dir.mkdir(exist_ok=True)

        if view == 'perspective':
            from mpl_toolkits.mplot3d import Axes3D
            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(111, projection='3d')
        else:
            fig, ax = plt.subplots(figsize=(10, 10))
            if view == 'top':
                idx1, idx2 = 0, 1
            elif view == 'front':
                idx1, idx2 = 0, 2
            elif view == 'right':
                idx1, idx2 = 1, 2
            else:
                idx1, idx2 = 0, 1

        has_data = False
        
        # Consistent colormap for strips
        cmap = plt.cm.tab20
        is_flat = "flat" in json_file
        is_quadrant = "quadrant" in json_file
        num_strips = len(meshes)
        
        # Determine quadrant strips (assuming symmetry)
        if is_quadrant or is_flat:
            quadrant_strips = num_strips
        else:
            quadrant_strips = num_strips // 4 if num_strips >= 4 else num_strips
            
        print(f"File {json_file}: total={num_strips}, quadrant_highlight={quadrant_strips}")

                # Plot meshes with unique colors
        for i, mesh in enumerate(meshes):
            has_data = True
            if i < quadrant_strips:
                rgba = cmap(i % 20)
                # Use solid color for highlighting
                color = (rgba[0], rgba[1], rgba[2], 0.9)
                edge_color = 'black'
                edge_width = 0.5
            else:
                # Almost invisible gray for other quadrants
                color = (0.9, 0.9, 0.9, 0.05) 
                edge_color = (0.5, 0.5, 0.5, 0.1)
                edge_width = 0.2
                
            for fkey in mesh.faces():
                pts = [mesh.vertex_coordinates(vkey) for vkey in mesh.face_vertices(fkey)]
                if view == 'perspective':
                    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
                    x = [p[0] for p in pts]
                    y = [p[1] for p in pts]
                    z = [p[2] for p in pts]
                    verts = [list(zip(x, y, z))]
                    poly = Poly3DCollection(verts, facecolor=color, edgecolor=edge_color, linewidth=edge_width)
                    ax.add_collection3d(poly)
                else:
                    x = [p[idx1] for p in pts] + [pts[0][idx1]]
                    y = [p[idx2] for p in pts] + [pts[0][idx2]]
                    ax.fill(x, y, facecolor=color, edgecolor=edge_color, linewidth=edge_width)

        # Plot lines (wireframe/edges)
        for line in lines:
            has_data = True
            try:
                if hasattr(line, 'start'): p1, p2 = line.start, line.end
                else: p1, p2 = line[0], line[1]
                
                if view == 'perspective':
                    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], color='black', linewidth=0.3, alpha=0.5)
                else:
                    ax.plot([p1[idx1], p2[idx1]], [p1[idx2], p2[idx2]], color='black', linewidth=0.5, alpha=0.3)
            except Exception: continue

        if not has_data:
            print(f"Warning: No geometry data to render for {json_file}")
            if view != 'perspective':
                ax.text(0.5, 0.5, "No Geometry Data", transform=ax.transAxes, ha='center')

        if view == 'perspective':
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            # Equalize axes
            pts_all = []
            if points:
                pts_all = [[p[0], p[1], p[2]] for p in points]
            elif meshes:
                for m in meshes:
                    pts_all.extend([m.vertex_coordinates(v) for v in m.vertices()])
            
            if pts_all:
                pts_array = np.array(pts_all)
                X, Y, Z = pts_array[:, 0], pts_array[:, 1], pts_array[:, 2]
                max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max() / 2.0
                mid_x = (X.max()+X.min()) * 0.5
                mid_y = (Y.max()+Y.min()) * 0.5
                mid_z = (Z.max()+Z.min()) * 0.5
                ax.set_xlim(mid_x - max_range, mid_x + max_range)
                ax.set_ylim(mid_y - max_range, mid_y + max_range)
                ax.set_zlim(mid_z - max_range, mid_z + max_range)
            
            ax.view_init(elev=30, azim=45)
        else:
            ax.set_aspect('equal')

        filepath = output_dir / f"{output_prefix}_{view}_matplotlib.png"
        plt.savefig(filepath)
        plt.close(fig)
        print(f"Successfully saved to {filepath}")
        return True
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Matplotlib rendering failed: {e}")
        return False

def render_geometry(json_file, output_prefix):
    if not os.path.exists(json_file):
        print(f"Error: {json_file} not found.")
        return

    # Always generate matplotlib views first for consistency and specialized coloring
    for v in ['perspective', 'top', 'front', 'right']:
        render_geometry_matplotlib(json_file, output_prefix, view=v)

    # Then attempt OpenGL if available
    is_darwin_headless = os.uname().sysname == 'Darwin' and os.environ.get("QT_QPA_PLATFORM") == "offscreen"
    
    if not is_darwin_headless:
        try:
            print(f"Attempting OpenGL rendering for {json_file}...")
            data = json_load(json_file)
            lines = data.get('lines', [])
            meshes = data.get('meshes', [])

            # Initialize Viewer
            viewer = Viewer(show_grid=True, width=1200, height=900)
            
            for line in lines:
                viewer.scene.add(line, linecolor=(0, 0, 0), linewidth=2)
            
            num_strips = len(meshes)
            quadrant_strips = num_strips // 4 if num_strips >= 4 else num_strips
            
            import matplotlib.pyplot as plt
            cmap = plt.cm.tab20
            
            for i, mesh in enumerate(meshes):
                if i < quadrant_strips:
                    rgba = cmap(i % 20)
                    from compas.colors import Color
                    c_color = Color(rgba[0], rgba[1], rgba[2], 0.8)
                else:
                    from compas.colors import Color
                    c_color = Color(0.8, 0.8, 0.8, 0.2)
                viewer.scene.add(mesh, name=f"Mesh_{i}", facecolor=c_color, show_edges=True)
            
            views = ['perspective', 'top', 'front', 'right']
            output_dir = pathlib.Path("renders")
            output_dir.mkdir(exist_ok=True)

            for view_name in views:
                viewer.renderer.view = view_name
                from compas_viewer.commands import zoom_selected
                zoom_selected(viewer)
                viewer.renderer.repaint()
                qimage = viewer.renderer.grabFramebuffer()
                
                if not qimage.isNull():
                    filepath = output_dir / f"{output_prefix}_{view_name}.png"
                    if qimage.save(str(filepath), "PNG"):
                        print(f"Successfully saved to {filepath}")
            
        except Exception as e:
            print(f"OpenGL rendering encountered an error: {e}")

if __name__ == "__main__":
    if os.path.exists("min_thrust_geometry.json"):
        render_geometry("min_thrust_geometry.json", "min_thrust")
    
    if os.path.exists("max_thrust_geometry.json"):
        render_geometry("max_thrust_geometry.json", "max_thrust")

    if os.path.exists("corrugated_geometry.json"):
        render_geometry("corrugated_geometry.json", "corrugated")

    if os.path.exists("min_thrust_quadrant_geometry.json"):
        render_geometry("min_thrust_quadrant_geometry.json", "min_thrust_quadrant")
    
    if os.path.exists("max_thrust_quadrant_geometry.json"):
        render_geometry("max_thrust_quadrant_geometry.json", "max_thrust_quadrant")

    if os.path.exists("corrugated_quadrant_geometry.json"):
        render_geometry("corrugated_quadrant_geometry.json", "corrugated_quadrant")

    if os.path.exists("flat_patterns.json"):
        # Always use matplotlib for flat patterns to get a true 2D view
        render_geometry_matplotlib("flat_patterns.json", "flat_patterns", view='top')
