#LIBGL_ALWAYS_SOFTWARE=1 xvfb-run -a uv run python3 code/render_vault.py
import os
import pathlib
import matplotlib.pyplot as plt
from compas.data import json_load
from compas_viewer import Viewer

def render_geometry_matplotlib(json_file, output_prefix):
    """Fallback renderer using matplotlib for 2D geometries (like flat patterns)."""
    if not os.path.exists(json_file):
        return False

    print(f"Using matplotlib fallback for {json_file}...")
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

        fig, ax = plt.subplots(figsize=(10, 10))
        
        has_data = False
        # Plot meshes
        for mesh in meshes:
            has_data = True
            for fkey in mesh.faces():
                pts = [mesh.vertex_coordinates(vkey) for vkey in mesh.face_vertices(fkey)]
                x = [p[0] for p in pts] + [pts[0][0]]
                y = [p[1] for p in pts] + [pts[0][1]]
                ax.fill(x, y, facecolor='lightblue', edgecolor='blue', linewidth=0.5, alpha=0.5)
                ax.plot(x, y, color='blue', linewidth=0.5)
        
        # Plot lines
        for line in lines:
            has_data = True
            try:
                if hasattr(line, 'start'): p1, p2 = line.start, line.end
                else: p1, p2 = line[0], line[1]
                ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color='black', linewidth=1)
            except Exception: continue

        if not has_data:
            print(f"Warning: No geometry data to render for {json_file}")
            ax.text(0.5, 0.5, "No Geometry Data", transform=ax.transAxes, ha='center')

        ax.set_aspect('equal')
        filepath = output_dir / f"{output_prefix}_matplotlib.png"
        plt.savefig(filepath)
        plt.close(fig)
        print(f"Successfully saved to {filepath}")
        return True
    except Exception as e:
        print(f"Matplotlib rendering failed: {e}")
        return False

def render_geometry(json_file, output_prefix):
    if not os.path.exists(json_file):
        print(f"Error: {json_file} not found.")
        return

    success = False
    
    # Try OpenGL rendering first unless we are headlessly on Darwin
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
            for i, mesh in enumerate(meshes):
                viewer.scene.add(mesh, name=f"Mesh_{i}", facecolor=(200, 200, 200), show_edges=True)
            
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
                        success = True
            
        except Exception as e:
            print(f"OpenGL rendering encountered an error: {e}")

    if not success:
        render_geometry_matplotlib(json_file, output_prefix)

if __name__ == "__main__":
    if os.path.exists("min_thrust_quadrant_geometry.json"):
        render_geometry("min_thrust_quadrant_geometry.json", "min_thrust")
    
    if os.path.exists("max_thrust_quadrant_geometry.json"):
        render_geometry("max_thrust_quadrant_geometry.json", "max_thrust")

    if os.path.exists("flat_patterns.json"):
        # Always use matplotlib for flat patterns to get a true 2D view
        render_geometry_matplotlib("flat_patterns.json", "flat_patterns")
