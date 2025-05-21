import math
import types # For monkey-patching methods
from compas.geometry import Point, Vector, Line, Polygon, Rotation, NurbsCurve, Polyline
from compas.geometry import bestfit_plane, project_points_plane, distance_point_plane
from compas.datastructures import Mesh
from compas.colors import Color
from compas_viewer import Viewer

# Attempt to import QtCore
QT_BINDING_AVAILABLE = False
QtCore = None
try:
    from PySide6 import QtCore
    QT_BINDING_AVAILABLE = True
except ImportError:
    print("PySide6 not found, some Qt specific features might not be available if used elsewhere.")

# --- Global variables for viewer instance and animation control ---
viewer_instance = None
current_time_step = 0

# --- Globals for geometry definition ---
plane_width = 0.0
num_crease_discretization_points = 0
curved_crease_nurbs = None
crease_parameters = []
crease_points_g = [] # Shared crease points

# Globals for Strip 1 (now dynamic)
outer_edge_points1_folded_g = []
strip1_mesh = None
strip1_mesh_obj = None
strip1_crease_vertex_keys = [] # Vertex keys in strip1_mesh for crease points
strip1_outer_edge_vertex_keys = [] # Vertex keys in strip1_mesh for its outer edge points
initial_outer_edge_length_s1 = 0.0

# Globals for Strip 2 (dynamic)
outer_edge_points2_folded_g = []
strip2_mesh = None
strip2_mesh_obj = None
strip2_crease_vertex_keys = [] # Vertex keys in strip2_mesh for crease points (distinct from strip1's keys)
strip2_outer_edge_vertex_keys = [] # Vertex keys in strip2_mesh for its outer edge points
initial_outer_edge_length_s2 = 0.0


# Constants for sinusoidal motion (from user's latest prompt)
BASE_REFLECTION_ANGLE_DEG = 25.0 # This is alpha, the angle one panel makes with the original plane
AMPLITUDE_DEG = 20.0
FREQUENCY = 0.01

# --- Helper function to calculate folded geometry points for a single strip ---
def calculate_single_strip_folded_points(
    ref_crease_points,
    nurbs_crease_curve,
    crease_params,
    strip_width,
    normal_side_multiplier, # +1 or -1 to define which side of the crease
    rotation_angle_radians  # Angle to rotate this strip's panel
):
    new_outer_edge_points = []
    num_pts = len(ref_crease_points)

    if not ref_crease_points or nurbs_crease_curve is None or not crease_params or num_pts == 0:
        print("Warning: Geometry globals not sufficiently initialized for calculate_single_strip_folded_points.")
        return []
    if len(crease_params) != num_pts:
        print("Warning: Mismatch in lengths of crease data arrays.")
        return []

    for i in range(num_pts):
        p_crease_anchor = ref_crease_points[i]
        param_current = crease_params[i]

        tangent_vector = nurbs_crease_curve.tangent_at(param_current)
        if tangent_vector is None:
            print(f"Warning: Failed to get tangent at param {param_current} for point {i}.")
            new_outer_edge_points.append(p_crease_anchor) # Simplistic fallback
            continue
        T_local = tangent_vector.unitized()

        # Assuming crease is planar in XY, B_local is global Z defining the "up" direction
        B_local = Vector(0, 0, 1)
        # N_local_direction is perpendicular to T_local and lies in the XY plane (if T_local is in XY plane)
        N_local_direction = B_local.cross(T_local)
        if N_local_direction.length < 1e-9: # Handle cases like T_local aligned with B_local (e.g. vertical tangent)
            # This fallback might need adjustment based on expected crease geometry.
            # For a planar XY crease, T_local won't be parallel to B_local unless it's a point.
            if T_local.is_parallel(Vector.Xaxis()):
                 N_local_direction = Vector.Yaxis().copy() # Ensure it's a copy
            else:
                 N_local_direction = Vector.Xaxis().cross(T_local)

        N_local_direction.unitize() # Ensure it's a unit vector for consistent scaling

        initial_outer_point_flat = p_crease_anchor + N_local_direction.scaled(strip_width * normal_side_multiplier)
        
        rotation_axis = T_local # Rotation axis is the crease tangent
        
        rotation = Rotation.from_axis_and_angle(rotation_axis, rotation_angle_radians, point=p_crease_anchor)
        rotated_outer_point = initial_outer_point_flat.transformed(rotation)
        new_outer_edge_points.append(rotated_outer_point)
        
    return new_outer_edge_points

# --- Main Script ---
if __name__ == "__main__":
    g_plane_width = 1.0
    g_num_crease_discretization_points = 20
    # g_num_points_across_width_g = 3 # Not directly used for mesh strips

    plane_width = g_plane_width
    num_crease_discretization_points = g_num_crease_discretization_points

    cp1 = Point(-2, 0, 0)
    cp2 = Point(-1, 1.5, 0)
    cp3 = Point(1, 1.5, 0)
    cp4 = Point(2, 0, 0)
    control_points = [cp1, cp2, cp3, cp4]
    curved_crease_nurbs = NurbsCurve.from_points(control_points, degree=3)

    domain_start, domain_end = curved_crease_nurbs.domain
    if domain_start is None or domain_end is None:
        raise ValueError("Curve domain is not defined.")

    crease_parameters = []
    if num_crease_discretization_points > 1:
        for i in range(num_crease_discretization_points):
            param = domain_start + i * (domain_end - domain_start) / (num_crease_discretization_points - 1)
            crease_parameters.append(param)
    elif num_crease_discretization_points == 1:
        crease_parameters.append(domain_start)

    crease_points_g = []
    for param in crease_parameters:
        pt = curved_crease_nurbs.point_at(param)
        if pt is None: raise ValueError(f"Failed to get point on NURBS curve at parameter {param}")
        crease_points_g.append(pt)

    crease_polyline_viz = Polyline(crease_points_g)

    # Calculate initial (flat) outer edge lengths for both strips
    flat_outer_edge_points_s1 = calculate_single_strip_folded_points(crease_points_g, curved_crease_nurbs, crease_parameters, plane_width, -1, 0.0)
    flat_outer_edge_points_s2 = calculate_single_strip_folded_points(crease_points_g, curved_crease_nurbs, crease_parameters, plane_width, +1, 0.0)

    if len(flat_outer_edge_points_s1) > 1:
        for k in range(len(flat_outer_edge_points_s1) - 1):
            initial_outer_edge_length_s1 += flat_outer_edge_points_s1[k].distance_to_point(flat_outer_edge_points_s1[k+1])
        print(f"Initial (flat) outer edge length for Strip 1: {initial_outer_edge_length_s1:.4f}")

    if len(flat_outer_edge_points_s2) > 1:
        for k in range(len(flat_outer_edge_points_s2) - 1):
            initial_outer_edge_length_s2 += flat_outer_edge_points_s2[k].distance_to_point(flat_outer_edge_points_s2[k+1])
        print(f"Initial (flat) outer edge length for Strip 2: {initial_outer_edge_length_s2:.4f}")


    # Initial calculation for visualization (e.g. at time_step 0)
    initial_alpha_rad = math.radians(BASE_REFLECTION_ANGLE_DEG + AMPLITUDE_DEG * math.sin(FREQUENCY * 0))
    outer_edge_points1_folded_g = calculate_single_strip_folded_points(crease_points_g, curved_crease_nurbs, crease_parameters, plane_width, -1, initial_alpha_rad)
    outer_edge_points2_folded_g = calculate_single_strip_folded_points(crease_points_g, curved_crease_nurbs, crease_parameters, plane_width, +1, initial_alpha_rad)

    # === Create Strip 1 Mesh Datastructure (Initial State) ===
    strip1_mesh = Mesh()
    strip1_crease_vertex_keys.clear()
    strip1_outer_edge_vertex_keys.clear()
    if crease_points_g:
        for pt in crease_points_g:
            strip1_crease_vertex_keys.append(strip1_mesh.add_vertex(x=pt.x, y=pt.y, z=pt.z))
    if outer_edge_points1_folded_g:
        for pt in outer_edge_points1_folded_g:
            strip1_outer_edge_vertex_keys.append(strip1_mesh.add_vertex(x=pt.x, y=pt.y, z=pt.z))
    if num_crease_discretization_points > 1 and \
       len(strip1_crease_vertex_keys) == num_crease_discretization_points and \
       len(strip1_outer_edge_vertex_keys) == num_crease_discretization_points:
        for i in range(num_crease_discretization_points - 1):
            v0 = strip1_crease_vertex_keys[i]
            v1 = strip1_crease_vertex_keys[i+1]
            v2 = strip1_outer_edge_vertex_keys[i+1] # Outer edge point corresponding to crease_points_g[i+1]
            v3 = strip1_outer_edge_vertex_keys[i]   # Outer edge point corresponding to crease_points_g[i]
            strip1_mesh.add_face([v0, v1, v2, v3]) # Check winding order for correct normals if needed

    # === Create Strip 2 Mesh Datastructure (Initial State) ===
    strip2_mesh = Mesh()
    strip2_crease_vertex_keys.clear() # Was global `crease_vertex_keys`
    strip2_outer_edge_vertex_keys.clear() # Was global `outer_edge_vertex_keys`
    if crease_points_g:
        for pt in crease_points_g:
            strip2_crease_vertex_keys.append(strip2_mesh.add_vertex(x=pt.x, y=pt.y, z=pt.z))
    if outer_edge_points2_folded_g:
        for pt in outer_edge_points2_folded_g:
            strip2_outer_edge_vertex_keys.append(strip2_mesh.add_vertex(x=pt.x, y=pt.y, z=pt.z))
    if num_crease_discretization_points > 1 and \
       len(strip2_crease_vertex_keys) == num_crease_discretization_points and \
       len(strip2_outer_edge_vertex_keys) == num_crease_discretization_points:
        for i in range(num_crease_discretization_points - 1):
            v0 = strip2_crease_vertex_keys[i]
            v1 = strip2_crease_vertex_keys[i+1]
            v2 = strip2_outer_edge_vertex_keys[i+1]
            v3 = strip2_outer_edge_vertex_keys[i]
            strip2_mesh.add_face([v0, v1, v2, v3])


    # === Initialize Viewer and Add Objects ===
    viewer_instance = Viewer(width=1600, height=900, show_grid=True, viewmode='shaded')
    if crease_points_g:
        viewer_instance.scene.add(crease_polyline_viz, name="CurvedCrease", linecolor=Color.black(), linewidth=3)

    # Add Strip 1 Mesh
    if strip1_mesh and strip1_mesh.number_of_vertices() > 0:
        s1_face_color = Color.green().lightened(70) # Different color for strip 1
        strip1_mesh_obj = viewer_instance.scene.add(
            strip1_mesh,
            facecolor=s1_face_color,
            linecolor=s1_face_color.darkened(25),
            linewidth=0.5,
            name="Strip1_Animated_Mesh"
        )
    else:
        print("Error: Strip 1 mesh is empty or invalid.")
        strip1_mesh_obj = None

    # Add Strip 2 Mesh
    if strip2_mesh and strip2_mesh.number_of_vertices() > 0:
        s2_face_color = Color.blue().lightened(70)
        strip2_mesh_obj = viewer_instance.scene.add(
            strip2_mesh,
            facecolor=s2_face_color,
            linecolor=s2_face_color.darkened(25),
            linewidth=0.5,
            name="Strip2_Animated_Mesh"
        )
    else:
        print("Error: Strip 2 mesh is empty or invalid.")
        strip2_mesh_obj = None

    # --- Animation Step Function ---
    @viewer_instance.on(interval=50) # ms
    def step_time_animation(frame):
        global current_time_step
        global outer_edge_points1_folded_g, outer_edge_points2_folded_g # Ensure these are updated
        global initial_outer_edge_length_s1, initial_outer_edge_length_s2

        current_time_step += 1
        # dynamic_reflection_angle_deg is alpha for each panel
        dynamic_reflection_angle_deg = BASE_REFLECTION_ANGLE_DEG + AMPLITUDE_DEG * math.sin(FREQUENCY * current_time_step)
        current_alpha_rad = math.radians(dynamic_reflection_angle_deg)

        # Update Strip 1
        if strip1_mesh_obj and strip1_mesh:
            outer_edge_points1_folded_g = calculate_single_strip_folded_points(
                crease_points_g, curved_crease_nurbs, crease_parameters, plane_width, -1, current_alpha_rad
            )
            if len(outer_edge_points1_folded_g) == len(strip1_outer_edge_vertex_keys):
                for i, vkey in enumerate(strip1_outer_edge_vertex_keys):
                    new_point = outer_edge_points1_folded_g[i]
                    strip1_mesh.vertex_attributes(vkey, 'xyz', [new_point.x, new_point.y, new_point.z])
                # Crease points for strip1_mesh do not change their geometry, only outer ones
                strip1_mesh_obj.init()
                strip1_mesh_obj.update(update_data=True)
            
            current_folded_length_s1 = 0
            if len(outer_edge_points1_folded_g) > 1:
                for k in range(len(outer_edge_points1_folded_g) - 1):
                    current_folded_length_s1 += outer_edge_points1_folded_g[k].distance_to_point(outer_edge_points1_folded_g[k+1])
            if current_time_step % 40 == 0 : # Print less frequently
                 print(f"S1 Angle: {dynamic_reflection_angle_deg:.2f} deg, S1 Folded Outer Len: {current_folded_length_s1:.4f} (Target: {initial_outer_edge_length_s1:.4f})")


        # Update Strip 2
        if strip2_mesh_obj and strip2_mesh:
            outer_edge_points2_folded_g = calculate_single_strip_folded_points(
                crease_points_g, curved_crease_nurbs, crease_parameters, plane_width, +1, -current_alpha_rad
            )
            if len(outer_edge_points2_folded_g) == len(strip2_outer_edge_vertex_keys):
                for i, vkey in enumerate(strip2_outer_edge_vertex_keys):
                    new_point = outer_edge_points2_folded_g[i]
                    strip2_mesh.vertex_attributes(vkey, 'xyz', [new_point.x, new_point.y, new_point.z])
                # Crease points for strip2_mesh do not change their geometry
                strip2_mesh_obj.init()
                strip2_mesh_obj.update(update_data=True)

            current_folded_length_s2 = 0
            if len(outer_edge_points2_folded_g) > 1:
                for k in range(len(outer_edge_points2_folded_g) - 1):
                    current_folded_length_s2 += outer_edge_points2_folded_g[k].distance_to_point(outer_edge_points2_folded_g[k+1])
            if current_time_step % 40 == 10: # Offset print
                 print(f"S2 Angle: {dynamic_reflection_angle_deg:.2f} deg, S2 Folded Outer Len: {current_folded_length_s2:.4f} (Target: {initial_outer_edge_length_s2:.4f})")


    print("Starting COMPAS Viewer. Close the viewer window to end the script.")
    viewer_instance.show()