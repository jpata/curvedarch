import os
import math
import numpy as np
from compas_tna.diagrams import FormDiagram
from compas_viewer import Viewer
from compas.colors import Color
from vault_shared import crossvault_middle_hc, fanvault_middle_hc, CONFIG

# NOTE: Headless rendering with QOpenGLWidget is notoriously difficult in many environments.
# This script will attempt a standard capture if a display is present, 
# or can be run interactively.

def create_envelope_meshes(x_span, y_span, hc, thickness, vault_type='cross', n=50):
    from compas_tna.diagrams.diagram_rectangular import create_cross_mesh
    mesh = create_cross_mesh(x_span=x_span, y_span=y_span, n=n)
    intrados = mesh.copy()
    extrados = mesh.copy()
    
    for vertex in mesh.vertices():
        x, y = mesh.vertex_attributes(vertex, names=["x", "y"])
        if vault_type == 'fan':
            z_mid = fanvault_middle_hc([x], [y], x_span, y_span, hc)[0]
        else:
            z_mid = crossvault_middle_hc([x], [y], x_span, y_span, hc)[0]
            
        intrados.vertex_attribute(vertex, "z", z_mid - thickness/2)
        extrados.vertex_attribute(vertex, "z", z_mid + thickness/2)
    return intrados, extrados

# 1. Load Data
try:
    form_min = FormDiagram.from_json('thrust_min.json')
    form_max = FormDiagram.from_json('thrust_max.json')
except Exception as e:
    print(f"Error loading JSON files: {e}")
    exit(1)

vertices = list(form_min.vertices())
xs = [form_min.vertex_attribute(v, 'x') for v in vertices]
ys = [form_min.vertex_attribute(v, 'y') for v in vertices]
x_span = [min(xs), max(xs)]
y_span = [min(ys), max(ys)]

hc = CONFIG['max_rise']
thickness = CONFIG['thickness']
vault_type = CONFIG['vault_type']

intrados, extrados = create_envelope_meshes(x_span, y_span, hc, thickness, vault_type=vault_type)

# 2. Setup Viewer
viewer = Viewer(width=1600, height=1200, show_grid=False)

viewer.scene.add(intrados, name="Intrados", opacity=0.4, show_edges=False, facecolor=Color.grey())
viewer.scene.add(extrados, name="Extrados", opacity=0.4, show_edges=False, facecolor=Color.white())
viewer.scene.add(form_min, name="Thrust Network (Min)", show_faces=False, show_edges=True, edgecolor=Color.red(), linewidth=3)
viewer.scene.add(form_max, name="Thrust Network (Max)", show_faces=False, show_edges=True, edgecolor=Color.blue(), linewidth=3)

# Center camera
target = [(x_span[0]+x_span[1])/2, (y_span[0]+y_span[1])/2, hc/2]
viewer.renderer.camera.target = target
viewer.renderer.camera.position = [x_span[1]*2, y_span[1]*2, hc*5]

# 3. Capture Function
def capture_and_exit():
    # Attempt to grab the frame
    print("Capturing...")
    try:
        # Give a small delay to ensure rendering is complete
        import time
        QApplication.processEvents()
        
        # In some versions, viewer.renderer directly has a screenshot method or can be grabbed
        pixmap = viewer.ui.window.widget.grab()
        pixmap.save("vault_screenshot.png")
        print("Success: Screenshot saved to vault_screenshot.png")
    except Exception as e:
        print(f"Failed to capture screenshot: {e}")
    
    QApplication.quit()

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

# If 'CAPTURE' env var is set, try to auto-capture
if os.environ.get('CAPTURE_HEADLESS'):
    os.environ['QT_QPA_PLATFORM'] = 'offscreen'
    QTimer.singleShot(2000, capture_and_exit)
    print("Running in auto-capture mode...")
else:
    print("Running in interactive mode. Press 'C' to take a screenshot manually.")
    
    @viewer.on(interval=100)
    def check_keys(frame):
        # This is a placeholder for manual capture if needed
        pass

viewer.show()
