import math
import numpy as np
from compas_tna.diagrams import FormDiagram
from compas_tna.envelope import ParametricEnvelope
from compas_tno.analysis import Analysis
from vault_shared import crossvault_middle_hc, fanvault_middle_hc, CONFIG

# ----------------------------------------
# 0. Shims for missing compas_tno modules
# ----------------------------------------

class GeneralVaultEnvelope(ParametricEnvelope):
    def __init__(self, x_span, y_span, thickness, hc, vault_type='cross', discretisation=20, **kwargs):
        super(GeneralVaultEnvelope, self).__init__(thickness=thickness, is_parametric=True, **kwargs)
        self.x_span = x_span
        self.y_span = y_span
        self.hc = hc
        self.vault_type = vault_type
        self.discretisation = discretisation
        self.middle = None
    
    def update_envelope(self):
        from compas_tna.diagrams.diagram_rectangular import create_cross_mesh
        from compas.datastructures import Mesh
        n = self.discretisation if isinstance(self.discretisation, int) else self.discretisation[0]
        # We use a dense cross mesh for the geometric representation
        self.middle = create_cross_mesh(x_span=self.x_span, y_span=self.y_span, n=n)
        for vertex in self.middle.vertices():
            x, y = self.middle.vertex_attributes(vertex, names=["x", "y"])
            z = self.compute_middle([x], [y])[0]
            self.middle.vertex_attribute(vertex, "z", z)

    def compute_middle(self, x, y):
        if self.vault_type == 'fan':
            return fanvault_middle_hc(x, y, self.x_span, self.y_span, self.hc)
        return crossvault_middle_hc(x, y, self.x_span, self.y_span, self.hc)
    
    def compute_bounds(self, x, y, thickness=None):
        t = thickness if thickness is not None else self.thickness
        z_mid = self.compute_middle(x, y)
        ub = z_mid + t/2
        lb = z_mid - t/2
        return ub, lb

class ShapeShim:
    @staticmethod
    def create_vault(xy_span, thk, desired_max_rise, vault_type, discretisation):
        return GeneralVaultEnvelope(xy_span[0], xy_span[1], thk, desired_max_rise, vault_type=vault_type, discretisation=discretisation)

def export_thrust_network_to_json(form, path, **kwargs):
    form.to_json(path)
    print(f"Exported thrust network to {path}")
    return True

# ----------------------------------------
# 1. Shape geometric definition
# ----------------------------------------
xy_span = CONFIG['xy_span']
thickness = CONFIG['thickness']
max_rise_at_crown = CONFIG['max_rise']
discretisation_level = CONFIG['discretisation_level']
vault_type = CONFIG['vault_type']

# Create the vault shape
vault = ShapeShim.create_vault(
    xy_span=xy_span,
    thk=thickness,
    desired_max_rise=max_rise_at_crown,
    vault_type=vault_type,
    discretisation=discretisation_level
)

# ----------------------------------------
# 2. Form diagram geometric definition
# ----------------------------------------
n = CONFIG['form_discretisation']
if vault_type == 'fan':
    print(f"Creating fan form diagram with {n} fans and {n} hoops...")
    form = FormDiagram.create_fan(x_span=xy_span[0], y_span=xy_span[1], n_fans=n, n_hoops=n)
else:
    print(f"Creating cross form diagram with n={n}...")
    form = FormDiagram.create_cross(x_span=xy_span[0], y_span=xy_span[1], n=n)

# Configure supports
if CONFIG['support_type'] == 'corners':
    print("Configuring corner supports...")
    # Support only the four corners
    # Corners in a FormDiagram often have degree 2
    for vertex in form.vertices():
        if form.vertex_degree(vertex) <= 2 and form.is_vertex_on_boundary(vertex):
            form.vertex_attribute(vertex, 'is_support', True)
else:
    print("Configuring perimeter supports...")
    # Support entire perimeter (distributed walls)
    for vertex in form.vertices():
        if form.is_vertex_on_boundary(vertex):
            form.vertex_attribute(vertex, 'is_support', True)

# Set starting point to the middle of the envelope
for vertex in form.vertices():
    x, y = form.vertex_attributes(vertex, names=["x", "y"])
    z_mid = vault.compute_middle([x], [y])[0]
    form.vertex_attribute(vertex, "z", z_mid)

# --------------------------------------------
# 3. Minimum thrust solution
# --------------------------------------------
print(f"\n--- Solving Minimum Thrust for {vault_type} vault ---")
analysis = Analysis.create_minthrust_analysis(form, vault, printout=True, solver=CONFIG['solver'])
analysis.apply_selfweight()
analysis.apply_envelope()
analysis.set_up_optimiser()
analysis.run()

# Export the thrust network
success = export_thrust_network_to_json(form, "./thrust_min.json", indent=2)

# # --------------------------------------------
# # 4. Maximum thrust solution
# # --------------------------------------------
print(f"\n--- Solving Maximum Thrust for {vault_type} vault ---")
analysis = Analysis.create_maxthrust_analysis(form, vault, printout=True, solver=CONFIG['solver'])
analysis.apply_selfweight()
analysis.apply_envelope()
analysis.set_up_optimiser()
analysis.run()
success = export_thrust_network_to_json(form, "./thrust_max.json", indent=2)
