from compas_tno.shapes import Shape
from compas_tno.diagrams import FormDiagram
from compas_tno.viewers import Viewer
from compas_tno.analysis import Analysis
from compas_tno.utilities import export_thrust_network_to_json

# ----------------------------------------
# 1. Shape geometric definition
# ----------------------------------------
xy_span = [[0.0, 10.0], [0.0, 1.619*10.0]] # 10m x 10m vault
thickness = 0.3
max_rise_at_crown = 2.0 # Force it to be much flatter than a semi-circle (which would be 5.0m)
discretisation_level = [20, 20] # For smoother surfaces, increase this

# Create the flatter cross vault shape
vault = Shape.create_flatter_crossvault(
    xy_span=xy_span,
    thk=thickness,
    desired_max_rise=max_rise_at_crown,
    discretisation=discretisation_level
)

# ----------------------------------------
# 2. Form diagram geometric definition
# ----------------------------------------
discretisation = 10
form = FormDiagram.create_fan_form(xy_span=xy_span, discretisation=discretisation)

# --------------------------------------------
# 3. Minimum thrust solution and visualisation
# --------------------------------------------
analysis = Analysis.create_minthrust_analysis(form, vault, printout=True)
analysis.apply_selfweight()
analysis.apply_envelope()
analysis.set_up_optimiser()
analysis.run()

view = Viewer(form)
view.settings['scale.reactions'] = 0.001
view.show_solution()
output_file = "./my_thrust_network_solution.json"

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

view = Viewer(form)
view.settings['scale.reactions'] = 0.001
view.show_solution()
