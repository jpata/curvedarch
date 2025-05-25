from compas_tno.shapes import Shape
from compas_tno.diagrams import FormDiagram
from compas_tno.viewers import Viewer
from compas_tno.analysis import Analysis
from compas.data import json_dump # For direct JSON writing
import os

# ----------------------------------------
# 1. Shape geometric definition
# ----------------------------------------
xy_span = [[0.0, 10.0], [0.0, 1.619*10.0]] # 10m x 10m vault
thickness = 0.3
max_rise_at_crown = 2.0 # Force it to be much flatter than a semi-circle (which would be 5.0m)
discretisation_level = [10, 10] # For smoother surfaces, increase this

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

def export_geometry_to_obj_and_json(points, obj_filepath, json_filepath, edge_indices=None):
    """
    Exports 3D points and optionally lines (edges) to both a Wavefront OBJ file
    and a COMPAS JSON file.

    Points are written as vertex lines (e.g., "v x y z").
    Lines are written as line elements (e.g., "l v1 v2"), using 1-based indexing.
    In JSON, points are stored as compas.geometry.Point and lines as compas.geometry.Line.

    Parameters
    ----------
    points : list of list of float
        List of [x, y, z] coordinates for the point cloud.
    obj_filepath : str
        The full path (including filename and .obj extension) where the
        OBJ file will be saved.
    json_filepath : str
        The full path (including filename and .json extension) where the
        COMPAS JSON file will be saved.
    edge_indices : list of tuple of int, optional
        List of (start_index, end_index) pairs for lines, where indices are
        0-based and refer to the `points` list. Default is None.

    Returns
    -------
    bool
        True if both OBJ and JSON exports were successful, False otherwise.
    """
    if not points:
        print("Error: Points list is empty or None. Cannot export geometry.")
        return False

    obj_export_successful = False
    json_export_successful = False

    # --- OBJ Export ---
    obj_directory = os.path.dirname(obj_filepath)
    if obj_directory and not os.path.exists(obj_directory):
        try:
            os.makedirs(obj_directory)
        except OSError as e:
            print(f"Error creating directory {obj_directory} for OBJ: {e}")
            return False
    try:
        with open(obj_filepath, 'w') as f_obj:
            for p in points:
                if isinstance(p, (list, tuple)) and len(p) == 3 and all(isinstance(coord, (int, float)) for coord in p):
                    f_obj.write(f"v {p[0]} {p[1]} {p[2]}\n")

            if edge_indices:
                for u_idx, v_idx in edge_indices:
                    # OBJ uses 1-based indexing for vertices in line definitions
                    f_obj.write(f"l {u_idx + 1} {v_idx + 1}\n")

        print(f"Geometry (points and lines) successfully exported to OBJ: {obj_filepath}")
        obj_export_successful = True
    except IOError as e:
        print(f"IOError writing OBJ to file {obj_filepath}: {e}")
    except Exception as e:
        print(f"An unexpected error occurred while writing OBJ to {obj_filepath}: {e}")

    # --- JSON Export ---
    try:
        from compas.geometry import Point, Line
    except ImportError:
        print("COMPAS geometry (Point, Line) not available for JSON export.")
        return obj_export_successful # Return status of OBJ export only

    compas_points_data = [Point(*p).to_data() for p in points]
    compas_lines_data = []
    if edge_indices:
        for u_idx, v_idx in edge_indices:
            try:
                line = Line(points[u_idx], points[v_idx])
                compas_lines_data.append(line.to_data())
            except IndexError:
                print(f"Error: Edge index out of bounds for JSON. u_idx={u_idx}, v_idx={v_idx}, num_points={len(points)}")
            except Exception as e:
                print(f"Error creating line for JSON from points[{u_idx}] and points[{v_idx}]: {e}")

    export_data = {
        "points": compas_points_data,
        "lines": compas_lines_data,
        "edge_point_indices": edge_indices if edge_indices is not None else []  # Add connectivity
    }

    json_directory = os.path.dirname(json_filepath)
    if json_directory and not os.path.exists(json_directory):
        try:
            os.makedirs(json_directory)
        except OSError as e:
            print(f"Error creating directory {json_directory} for JSON: {e}")
            return obj_export_successful
    try:
        with open(json_filepath, 'w') as f_json:
            json_dump(export_data, f_json)
        print(f"Geometry (points and lines) successfully exported to JSON: {json_filepath}")
        json_export_successful = True
    except IOError as e:
        print(f"IOError writing JSON to file {json_filepath}: {e}")
    except Exception as e:
        print(f"An unexpected error occurred while writing JSON to {json_filepath}: {e}")

    return obj_export_successful and json_export_successful

analysis = Analysis.create_minthrust_analysis(form, vault, printout=True)
analysis.apply_selfweight()
analysis.apply_envelope()
analysis.set_up_optimiser()
analysis.run()

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

# Export geometry (points and lines) for the quadrant to OBJ and JSON
print("\nExporting Min Thrust quadrant geometry to OBJ and JSON...")
success_export_min = export_geometry_to_obj_and_json(
    points=points_z_min_quadrant,
    obj_filepath="./min_thrust_quadrant_geometry.obj",
    json_filepath="./min_thrust_quadrant_geometry.json",
    edge_indices=edges_min_quadrant
)
if success_export_min:
    print("Min Thrust quadrant geometry export successful.")
else:
    print("Min Thrust quadrant geometry export failed for OBJ and/or JSON.")

# view = Viewer(form) # You can still view it with COMPAS viewer if not in Blender
# view.settings['scale.reactions'] = 0.001
# view.show_solution()

# # --------------------------------------------
# # 4. Maximum thrust solution and visualisation
# # --------------------------------------------
analysis = Analysis.create_maxthrust_analysis(form, vault, printout=True)
analysis.apply_selfweight()
analysis.apply_envelope()
analysis.set_up_optimiser()
analysis.run()

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

# Export geometry (points and lines) for the quadrant to OBJ and JSON
print("\nExporting Max Thrust quadrant geometry to OBJ and JSON...")
success_export_max = export_geometry_to_obj_and_json(
    points=points_z_max_quadrant,
    obj_filepath="./max_thrust_quadrant_geometry.obj",
    json_filepath="./max_thrust_quadrant_geometry.json",
    edge_indices=edges_max_quadrant
)
if success_export_max:
    print("Max Thrust quadrant geometry export successful.")
else:
    print("Max Thrust quadrant geometry export failed for OBJ and/or JSON.")
# view = Viewer(form) # You can still view it with COMPAS viewer if not in Blender
# view.settings['scale.reactions'] = 0.001
# view.show_solution()
