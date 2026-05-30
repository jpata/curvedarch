import math
import numpy as np
from compas_tna.diagrams import FormDiagram
from compas_tna.envelope import ParametricEnvelope
from compas_tno.analysis import Analysis

# ----------------------------------------
# 0. Shims for missing compas_tno modules
# ----------------------------------------

def crossvault_middle_hc(x, y, x_span, y_span, hc, tol=1e-6):
    x0, x1 = x_span
    y0, y1 = y_span
    rx = (x1 - x0) / 2
    ry = (y1 - y0) / 2
    z = np.zeros(len(x))
    for i in range(len(x)):
        xi, yi = x[i], y[i]
        # Ensure points are within span
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

class FlatterCrossVaultEnvelope(ParametricEnvelope):
    def __init__(self, x_span, y_span, thickness, hc, discretisation=20, **kwargs):
        super(FlatterCrossVaultEnvelope, self).__init__(thickness=thickness, is_parametric=True, **kwargs)
        self.x_span = x_span
        self.y_span = y_span
        self.hc = hc
        self.discretisation = discretisation
        self.middle = None
    
    def update_envelope(self):
        from compas_tna.diagrams.diagram_rectangular import create_cross_mesh
        from compas.datastructures import Mesh
        n = self.discretisation if isinstance(self.discretisation, int) else self.discretisation[0]
        self.middle = create_cross_mesh(x_span=self.x_span, y_span=self.y_span, n=n)
        for vertex in self.middle.vertices():
            x, y = self.middle.vertex_attributes(vertex, names=["x", "y"])
            z = crossvault_middle_hc([x], [y], self.x_span, self.y_span, self.hc)[0]
            self.middle.vertex_attribute(vertex, "z", z)

    def compute_middle(self, x, y):
        return crossvault_middle_hc(x, y, self.x_span, self.y_span, self.hc)
    
    def compute_bounds(self, x, y, thickness=None):
        t = thickness if thickness is not None else self.thickness
        z_mid = self.compute_middle(x, y)
        ub = z_mid + t/2
        lb = z_mid - t/2
        return ub, lb

class ShapeShim:
    @staticmethod
    def create_flatter_crossvault(xy_span, thk, desired_max_rise, discretisation):
        return FlatterCrossVaultEnvelope(xy_span[0], xy_span[1], thk, desired_max_rise, discretisation=discretisation)

def export_thrust_network_to_json(form, path, **kwargs):
    form.to_json(path)
    print(f"Exported thrust network to {path}")
    return True

# ----------------------------------------
# 1. Shape geometric definition
# ----------------------------------------
xy_span = [[0.0, 10.0], [0.0, 1.619*10.0]] # 10m x 10m vault
thickness = 0.3
max_rise_at_crown = 2.0 # Force it to be much flatter than a semi-circle (which would be 5.0m)
discretisation_level = [20, 20] # For smoother surfaces, increase this

# Create the flatter cross vault shape
vault = ShapeShim.create_flatter_crossvault(
    xy_span=xy_span,
    thk=thickness,
    desired_max_rise=max_rise_at_crown,
    discretisation=discretisation_level
)

# ----------------------------------------
# 2. Form diagram geometric definition
# ----------------------------------------
discretisation = 10
form = FormDiagram.create_fan(x_span=xy_span[0], y_span=xy_span[1], n_fans=discretisation, n_hoops=discretisation)

# --------------------------------------------
# 3. Minimum thrust solution and visualisation
# --------------------------------------------
analysis = Analysis.create_minthrust_analysis(form, vault, printout=True)
analysis.apply_selfweight()
analysis.apply_envelope()
analysis.set_up_optimiser()
analysis.run()

# Export the thrust network
success = export_thrust_network_to_json(form, "./thrust_min.json", indent=2)

# # --------------------------------------------
# # 4. Maximum thrust solution and visualisation
# # --------------------------------------------
analysis = Analysis.create_maxthrust_analysis(form, vault, printout=True)
analysis.apply_selfweight()
analysis.apply_envelope()
analysis.set_up_optimiser()
analysis.run()
success = export_thrust_network_to_json(form, "./thrust_max.json", indent=2)
