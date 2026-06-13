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

def create_custom_fan(x_span, y_span, nx, ny, n_hoops):
    from compas.datastructures import Mesh
    from compas_tna.diagrams import FormDiagram

    x0, x1 = x_span
    y0, y1 = y_span
    xm, ym = (x0 + x1) / 2, (y0 + y1) / 2
    
    mesh = Mesh()
    vertex_map = {}

    def get_vertex(mesh, x, y):
        key = (round(x, 6), round(y, 6))
        if key in vertex_map:
            return vertex_map[key]
        idx = mesh.add_vertex(x=x, y=y, z=0)
        vertex_map[key] = idx
        return idx

    # Quadrant configurations: (corner, nx, ny)
    quads = [
        ((x0, y0), nx, ny), # Q1: BL
        ((x1, y0), nx, ny), # Q2: BR
        ((x1, y1), nx, ny), # Q3: TR
        ((x0, y1), nx, ny)  # Q4: TL
    ]

    for c, qnx, qny in quads:
        xc, yc = c
        # Boundary points for this fan quadrant
        qpts = []
        
        # Directions towards center lines
        dx = xm - xc
        dy = ym - yc
        
        # Edge 1: Along the boundary where x varies (at y=ym)
        # For Q1: (x0, ym) to (xm, ym)
        for i in range(qnx + 1):
            t = i / qnx
            qpts.append((xc + t * dx, ym))
            
        # Edge 2: Along the boundary where y varies (at x=xm)
        # From (xm, ym) to (xm, yc)
        for i in range(1, qny + 1):
            t = i / qny
            qpts.append((xm, ym - t * dy))

        # Build fan layers (hoops)
        prev_hoop = [get_vertex(mesh, xc, yc)] * len(qpts)
        
        for j in range(1, n_hoops + 1):
            curr_hoop = []
            th = j / n_hoops
            for px, py in qpts:
                vx = xc + th * (px - xc)
                vy = yc + th * (py - yc)
                curr_hoop.append(get_vertex(mesh, vx, vy))
            
            # Add faces
            for i in range(len(qpts) - 1):
                if j == 1:
                    mesh.add_face([prev_hoop[i], curr_hoop[i+1], curr_hoop[i]])
                else:
                    mesh.add_face([prev_hoop[i], prev_hoop[i+1], curr_hoop[i+1], curr_hoop[i]])
            prev_hoop = curr_hoop

    return FormDiagram.from_mesh(mesh)

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

    discr_x = config.get('form_discretisation_x', config.get('form_discretisation', 10))
    discr_y = config.get('form_discretisation_y', config.get('form_discretisation', 10))
    n_hoops = config.get('form_discretisation', 10)

    if vault_type == 'fan':
        # form = FormDiagram.create_fan(x_span=xy_span[0], y_span=xy_span[1], n_fans=discretisation, n_hoops=discretisation)
        form = create_custom_fan(x_span=xy_span[0], y_span=xy_span[1], nx=discr_x, ny=discr_y, n_hoops=n_hoops)
    else:
        # For cross vault, we still use symmetric n for now unless specifically requested to refactor its topology
        n_cross = max(discr_x, discr_y)
        form = FormDiagram.create_cross(x_span=xy_span[0], y_span=xy_span[1], n=n_cross)

    # Support logic
    if config['support_type'] == 'corners':
        for vertex in form.vertices():
            x, y = form.vertex_attributes(vertex, names=['x', 'y'])
            is_corner = False
            for cx in xy_span[0]:
                for cy in xy_span[1]:
                    if abs(x - cx) < 1e-6 and abs(y - cy) < 1e-6:
                        is_corner = True
                        break
            if is_corner:
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
