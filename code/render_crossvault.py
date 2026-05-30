import math
import numpy as np
from compas_tna.diagrams import FormDiagram
from compas_viewer import Viewer
from compas.colors import Color

# Re-implement the envelope logic to render it as context
def crossvault_middle_hc(x, y, x_span, y_span, hc, tol=1e-6):
    x0, x1 = x_span
    y0, y1 = y_span
    rx = (x1 - x0) / 2
    ry = (y1 - y0) / 2
    z = np.zeros(len(x))
    for i in range(len(x)):
        xi, yi = x[i], y[i]
        xi = max(x0, min(x1, xi))
        yi = max(y0, min(y1, yi))
        xd = x0 + (x1 - x0) / (y1 - y0) * (yi - y0)
        yd = y0 + (y1 - y0) / (x1 - x0) * (xi - x0)
        hxd = math.sqrt(abs(rx**2 - (xd - x0 - rx)**2))
        hyd = math.sqrt(abs(ry**2 - (yd - y0 - ry)**2))
        if yi <= y0 + (y1 - y0) / (x1 - x0) * (xi - x0) + tol and yi >= y1 - (y1 - y0) / (x1 - x0) * (xi - x0) - tol:  # Q1
            z[i] = hc * (hxd + math.sqrt(abs(ry**2 - (yi - y0 - ry)**2))) / (rx + ry)
        elif yi >= y0 + (y1 - y0) / (x1 - x0) * (xi - x0) - tol and yi >= y1 - (y1 - y0) / (x1 - x0) * (xi - x0) - tol:  # Q3
            z[i] = hc * (hyd + math.sqrt(abs(rx**2 - (xi - x0 - rx)**2))) / (rx + ry)
        elif yi >= y0 + (y1 - y0) / (x1 - x0) * (xi - x0) - tol and yi <= y1 - (y1 - y0) / (x1 - x0) * (xi - x0) + tol:  # Q2
            z[i] = hc * (hxd + math.sqrt(abs(ry**2 - (yi - y0 - ry)**2))) / (rx + ry)
        elif yi <= y0 + (y1 - y0) / (x1 - x0) * (xi - x0) + tol and yi <= y1 - (y1 - y0) / (x1 - x0) * (xi - x0) + tol:  # Q4
            z[i] = hc * (hyd + math.sqrt(abs(rx**2 - (xi - x0 - rx)**2))) / (rx + ry)
    return z

def create_envelope_meshes(x_span, y_span, hc, thickness, n=40):
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
# 2. Create Envelope for Context
# ----------------------------------------
xy_span = [[0.0, 10.0], [0.0, 16.19]]
hc = 2.0
thickness = 0.3
intrados, extrados = create_envelope_meshes(xy_span[0], xy_span[1], hc, thickness)

# ----------------------------------------
# 3. Render with compas_viewer
# ----------------------------------------
viewer = Viewer()

# Add Intrados and Extrados as transparent surfaces
viewer.scene.add(intrados, name="Intrados", opacity=0.3, show_edges=False, facecolor=Color.grey())
viewer.scene.add(extrados, name="Extrados", opacity=0.3, show_edges=False, facecolor=Color.white())

# Add Thrust Networks as wireframes
viewer.scene.add(form_min, name="Thrust Network (Min)", show_faces=False, show_edges=True, edgecolor=Color.red(), linewidth=2)
viewer.scene.add(form_max, name="Thrust Network (Max)", show_faces=False, show_edges=True, edgecolor=Color.blue(), linewidth=2)

print("Opening COMPAS Viewer...")
viewer.show()
