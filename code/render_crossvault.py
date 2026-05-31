import math
import numpy as np
from compas_tna.diagrams import FormDiagram
from compas_viewer import Viewer
from compas.colors import Color
from vault_shared import crossvault_middle_hc, CONFIG

def create_envelope_meshes(x_span, y_span, hc, thickness, n=50):
    from compas_tna.diagrams.diagram_rectangular import create_cross_mesh
    mesh = create_cross_mesh(x_span=x_span, y_span=y_span, n=n)
    intrados = mesh.copy()
    extrados = mesh.copy()
    for vertex in mesh.vertices():
        x, y = mesh.vertex_attributes(vertex, names=["x", "y"])
        z_mid = crossvault_middle_hc([x], [y], x_span, y_span, hc)[0]
        intrados.vertex_attribute(vertex, "z", z_mid - thickness/2)
        extrados.vertex_attribute(vertex, "z", z_mid + thickness/2)
    return intrados, extrados

# ----------------------------------------
# 1. Load Thrust Networks
# ----------------------------------------
try:
    form_min = FormDiagram.from_json('thrust_min.json')
    form_max = FormDiagram.from_json('thrust_max.json')
    print("Successfully loaded thrust networks.")
except Exception as e:
    print(f"Error loading JSON files: {e}")
    exit(1)

# ----------------------------------------
# 2. Get Geometry Parameters from Shared CONFIG
# ----------------------------------------
# Derive spans from vertex attributes to ensure alignment with loaded data
vertices = list(form_min.vertices())
xs = [form_min.vertex_attribute(v, 'x') for v in vertices]
ys = [form_min.vertex_attribute(v, 'y') for v in vertices]
x_span = [min(xs), max(xs)]
y_span = [min(ys), max(ys)]

hc = CONFIG['max_rise']
thickness = CONFIG['thickness']

print(f"Rendering for Span: {x_span}, {y_span}, HC: {hc}, Thk: {thickness}")

intrados, extrados = create_envelope_meshes(x_span, y_span, hc, thickness)

# ----------------------------------------
# 3. Render with compas_viewer
# ----------------------------------------
viewer = Viewer()

# Add Intrados and Extrados as transparent surfaces
viewer.scene.add(intrados, name="Intrados", opacity=0.3, show_edges=False, facecolor=Color.grey())
viewer.scene.add(extrados, name="Extrados", opacity=0.3, show_edges=False, facecolor=Color.white())

# Add Thrust Networks as wireframes
viewer.scene.add(form_min, name="Thrust Network (Min)", show_faces=False, show_edges=True, edgecolor=Color.red(), linewidth=3)
viewer.scene.add(form_max, name="Thrust Network (Max)", show_faces=False, show_edges=True, edgecolor=Color.blue(), linewidth=3)

print("Opening COMPAS Viewer...")
viewer.show()
