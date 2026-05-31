import json
import math
import numpy as np
from compas_tna.diagrams import FormDiagram
from compas_tna.envelope import ParametricEnvelope
from compas_tno.analysis import Analysis
from code.vault_shared import crossvault_middle_hc, fanvault_middle_hc, CONFIG

# ----------------------------------------
# 0. Shims and Helpers
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
        n = self.discretisation if isinstance(self.discretisation, int) else self.discretisation[0]
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

def mesh_to_data(mesh):
    return {
        'vertices': [mesh.vertex_coordinates(v) for v in mesh.vertices()],
        'faces': [mesh.face_vertices(f) for f in mesh.faces()]
    }

def diagram_to_wire_data(diagram):
    return {
        'vertices': [diagram.vertex_coordinates(v) for v in diagram.vertices()],
        'edges': [list(edge) for edge in diagram.edges()]
    }

# ----------------------------------------
# 1. Main Simulation Function
# ----------------------------------------

def run_tna_simulation(config=None):
    if config is None:
        config = CONFIG

    xy_span = config['xy_span']
    thickness = config['thickness']
    max_rise_at_crown = config['max_rise']
    vault_type = config['vault_type']

    vault = GeneralVaultEnvelope(xy_span[0], xy_span[1], thickness, max_rise_at_crown, 
                                 vault_type=vault_type, 
                                 discretisation=config['discretisation_level'])

    discretisation = config['form_discretisation']
    if vault_type == 'fan':
        form = FormDiagram.create_fan(x_span=xy_span[0], y_span=xy_span[1], n_fans=discretisation, n_hoops=discretisation)
    else:
        form = FormDiagram.create_cross(x_span=xy_span[0], y_span=xy_span[1], n=discretisation)

    # Support logic
    if config['support_type'] == 'corners':
        for vertex in form.vertices():
            if form.vertex_degree(vertex) <= 2 and form.is_vertex_on_boundary(vertex):
                form.vertex_attribute(vertex, 'is_support', True)
    else:
        for vertex in form.vertices():
            if form.is_vertex_on_boundary(vertex):
                form.vertex_attribute(vertex, 'is_support', True)

    # Set starting point
    for vertex in form.vertices():
        x, y = form.vertex_attributes(vertex, names=["x", "y"])
        z_mid = vault.compute_middle([x], [y])[0]
        form.vertex_attribute(vertex, "z", z_mid)

    # Solve Min Thrust
    print(f"Solving Min Thrust for {vault_type}...")
    form_min = form.copy()
    analysis_min = Analysis.create_minthrust_analysis(form_min, vault, printout=False, solver=config['solver'])
    analysis_min.apply_selfweight()
    analysis_min.apply_envelope()
    analysis_min.set_up_optimiser()
    analysis_min.run()

    # Solve Max Thrust
    print(f"Solving Max Thrust for {vault_type}...")
    form_max = form.copy()
    analysis_max = Analysis.create_maxthrust_analysis(form_max, vault, printout=False, solver=config['solver'])
    analysis_max.apply_selfweight()
    analysis_max.apply_envelope()
    analysis_max.set_up_optimiser()
    analysis_max.run()

    # Generate intrados/extrados
    vault.update_envelope()
    intrados = vault.middle.copy()
    extrados = vault.middle.copy()
    for v in intrados.vertices():
        z = intrados.vertex_attribute(v, 'z')
        intrados.vertex_attribute(v, 'z', z - thickness/2)
        extrados.vertex_attribute(v, 'z', z + thickness/2)

    data = {
        'intrados': mesh_to_data(intrados),
        'extrados': mesh_to_data(extrados),
        'thrust_min': diagram_to_wire_data(form_min),
        'thrust_max': diagram_to_wire_data(form_max),
        'form_min': form_min,
        'form_max': form_max
    }

    return data

if __name__ == '__main__':
    f_min, f_max = run_tna_simulation()
    
    # Optional: still export if run as script
    f_min.to_json('thrust_min.json')
    f_max.to_json('thrust_max.json')
    print("Simulated and exported thrust diagrams.")
