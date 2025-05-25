from compas_tno.shapes import Shape
from compas_tno.diagrams import FormDiagram
from compas_tno.viewers import Viewer
from compas_tno.analysis import Analysis
from compas_tno.utilities import export_thrust_network_to_json, export_z_pointcloud_to_json
import os

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

def export_geometry_to_obj(points, filepath, edge_indices=None):
    """
    Exports 3D points and optionally lines (edges) to a Wavefront OBJ file.

    Points are written as vertex lines (e.g., "v x y z").
    Lines are written as line elements (e.g., "l v1 v2"), using 1-based indexing.

    Parameters
    ----------
    points : list of list of float
        List of [x, y, z] coordinates for the point cloud.
    filepath : str
        The full path (including filename and .obj extension) where the
        OBJ file will be saved.
    edge_indices : list of tuple of int, optional
        List of (start_index, end_index) pairs for lines, where indices are
        0-based and refer to the `points` list. Default is None.

    Returns
    -------
    bool
        True if the export was successful, False otherwise.
    """
    if not points:
        print("Error: Points list is empty or None. Cannot export geometry to OBJ.")
        return False

    directory = os.path.dirname(filepath)
    if directory and not os.path.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            print(f"Error creating directory {directory} for OBJ: {e}")
            return False
    try:
        with open(filepath, 'w') as f_obj:
            for p in points:
                if isinstance(p, (list, tuple)) and len(p) == 3 and all(isinstance(coord, (int, float)) for coord in p):
                    f_obj.write(f"v {p[0]} {p[1]} {p[2]}\n")
            
            if edge_indices:
                for u_idx, v_idx in edge_indices:
                    # OBJ uses 1-based indexing for vertices in line definitions
                    f_obj.write(f"l {u_idx + 1} {v_idx + 1}\n")

        print(f"Geometry (points and lines) successfully exported to OBJ: {filepath}")
        return True
    except IOError as e:
        print(f"IOError writing OBJ to file {filepath}: {e}")
        return False
    except Exception as e:
        print(f"An unexpected error occurred while writing OBJ to {filepath}: {e}")
        return False

analysis = Analysis.create_minthrust_analysis(form, vault, printout=True)
analysis.apply_selfweight()
analysis.apply_envelope()
analysis.set_up_optimiser()
analysis.run()

# Export the thrust network to JSON
print("\nExporting Min Thrust Network to JSON...")
success_json_min = export_thrust_network_to_json(form, "./thrust_min.json", indent=2)
if success_json_min:
    print("Min Thrust Network JSON export successful.")

# Calculate center of the span for quadrant filtering
center_x = (xy_span[0][0] + xy_span[0][1]) / 2.0
center_y = (xy_span[1][0] + xy_span[1][1]) / 2.0

# Extract points for Z-coordinate point cloud export
print("\nExtracting points for Min Thrust Z-coordinate cloud...")
points_z_min = []
points_z_min_quadrant = []
quadrant_vkey_to_idx_min = {} # To map original vkey to new 0-based index in quadrant list

for vkey in form.vertices():
    x = form.vertex_attribute(vkey, 'x')
    y = form.vertex_attribute(vkey, 'y')
    z = form.vertex_attribute(vkey, 'z') # Optimized Z-coordinate
    if x is not None and y is not None and z is not None:
        # points_z_min.append([x, y, z]) # Keep all points if needed for other exports
        # Filter for the top-right quadrant
        if x >= center_x and y >= center_y:
            points_z_min_quadrant.append([x, y, z])
            quadrant_vkey_to_idx_min[vkey] = len(points_z_min_quadrant) - 1
    else:
        print(f"Warning: Vertex {vkey} has None for x, y, or z. x={x}, y={y}, z={z}. Skipping for Z-cloud export.")

# Extract edges for the quadrant
edges_min_quadrant = []
for u, v in form.edges():
    if u in quadrant_vkey_to_idx_min and v in quadrant_vkey_to_idx_min:
        # Get the new 0-based indices for the quadrant points list
        edges_min_quadrant.append((quadrant_vkey_to_idx_min[u], quadrant_vkey_to_idx_min[v]))

# Export Z-coordinate point cloud to JSON
print("\nExporting Min Thrust Z-coordinate cloud to JSON (one quadrant)...")
success_z_cloud_min = export_z_pointcloud_to_json(
    points=points_z_min_quadrant,
    filepath="./z_pointcloud_min_quadrant.json"
)
if success_z_cloud_min:
    print("Min Thrust Z-coordinate cloud JSON export successful.")
else:
    print("Min Thrust Z-coordinate cloud JSON export failed.")

# Export Z-coordinate point cloud to OBJ
print("\nExporting Min Thrust geometry (points and lines) to OBJ (one quadrant)...")
success_z_cloud_obj_min = export_geometry_to_obj(
    points=points_z_min_quadrant,
    filepath="./min_thrust_quadrant_geometry.obj",
    edge_indices=edges_min_quadrant
)
if not success_z_cloud_obj_min:
    print("Min Thrust geometry OBJ export (quadrant) failed.")

# view = Viewer(form) # You can still view it with COMPAS viewer if not in Blender
# view.settings['scale.reactions'] = 0.001
# view.show_solution()

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

# Export the thrust network to JSON
print("\nExporting Max Thrust Network to JSON...")
success_json_max = export_thrust_network_to_json(form, "./thrust_max.json", indent=2)
if success_json_max:
    print("Max Thrust Network JSON export successful.")

# Extract points for Z-coordinate point cloud export
print("\nExtracting points for Max Thrust Z-coordinate cloud...")
points_z_max = []
points_z_max_quadrant = []
quadrant_vkey_to_idx_max = {}

for vkey in form.vertices():
    x = form.vertex_attribute(vkey, 'x')
    y = form.vertex_attribute(vkey, 'y')
    z = form.vertex_attribute(vkey, 'z') # Optimized Z-coordinate
    if x is not None and y is not None and z is not None:
        # points_z_max.append([x, y, z]) # Keep all points if needed for other exports
        # Filter for the top-right quadrant
        if x >= center_x-1e-6 and y >= center_y-1e-6:
            points_z_max_quadrant.append([x, y, z])
            quadrant_vkey_to_idx_max[vkey] = len(points_z_max_quadrant) - 1
    else:
        print(f"Warning: Vertex {vkey} has None for x, y, or z. x={x}, y={y}, z={z}. Skipping for Z-cloud export.")

# Extract edges for the quadrant
edges_max_quadrant = []
for u, v in form.edges():
    if u in quadrant_vkey_to_idx_max and v in quadrant_vkey_to_idx_max:
        edges_max_quadrant.append((quadrant_vkey_to_idx_max[u], quadrant_vkey_to_idx_max[v]))

# Export Z-coordinate point cloud to JSON
print("\nExporting Max Thrust Z-coordinate cloud to JSON (one quadrant)...")
success_z_cloud_max = export_z_pointcloud_to_json(
    points=points_z_max_quadrant,
    filepath="./z_pointcloud_max_quadrant.json"
)
if success_z_cloud_max:
    print("Max Thrust Z-coordinate cloud JSON export successful.")
else:
    print("Max Thrust Z-coordinate cloud JSON export failed.")

# Export Z-coordinate point cloud to OBJ
print("\nExporting Max Thrust geometry (points and lines) to OBJ (one quadrant)...")
success_z_cloud_obj_max = export_geometry_to_obj(
    points=points_z_max_quadrant,
    filepath="./max_thrust_quadrant_geometry.obj",
    edge_indices=edges_max_quadrant
)
if not success_z_cloud_obj_max:
    print("Max Thrust geometry OBJ export (quadrant) failed.")
# view = Viewer(form) # You can still view it with COMPAS viewer if not in Blender
# view.settings['scale.reactions'] = 0.001
# view.show_solution()
