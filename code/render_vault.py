#LIBGL_ALWAYS_SOFTWARE=1 xvfb-run -a uv run python3 code/render_vault.py
import os
import pathlib
from compas.data import json_load
from compas_viewer import Viewer

def render_geometry(json_file, output_prefix):
    if not os.path.exists(json_file):
        print(f"Error: {json_file} not found.")
        return

    print(f"Loading geometry from {json_file}...")
    data = json_load(json_file)
    lines = data.get('lines', [])

    # Initialize Viewer with larger window for better resolution
    viewer = Viewer(show_grid=True, width=1200, height=900)
    
    # Add geometry to scene
    for line in lines:
        viewer.scene.add(line, linecolor=(0, 0, 0), linewidth=2)
    
    # Define views to render
    views = ['perspective', 'top', 'front', 'right']
    
    output_dir = pathlib.Path("renders")
    output_dir.mkdir(exist_ok=True)

    for view_name in views:
        print(f"Rendering {view_name} view...")
        
        # Set the view
        viewer.renderer.view = view_name
        
        # Zoom to all objects
        from compas_viewer.commands import zoom_selected
        zoom_selected(viewer)
        
        # Force a paint
        viewer.renderer.repaint()
        
        # Capture the view
        qimage = viewer.renderer.grabFramebuffer()
        
        filepath = output_dir / f"{output_prefix}_{view_name}.png"
        if qimage.save(str(filepath), "PNG"):
            print(f"Successfully saved to {filepath}")
        else:
            print(f"Failed to save {filepath}")

if __name__ == "__main__":
    if os.path.exists("min_thrust_quadrant_geometry.json"):
        render_geometry("min_thrust_quadrant_geometry.json", "min_thrust")
    
    if os.path.exists("max_thrust_quadrant_geometry.json"):
        render_geometry("max_thrust_quadrant_geometry.json", "max_thrust")
