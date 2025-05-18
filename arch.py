import math
import torch # PyTorch import
import torch.optim as optim # PyTorch optimizer
import threading
import queue
import time # For potential debugging or yielding

from compas.geometry import Point, Vector, Polygon, Rotation, NurbsCurve, Polyline
from compas.datastructures import Mesh
from compas.colors import Color
from compas_viewer import Viewer

# --- Global variables ---
viewer_instance = None
current_time_step = 0
plane_width = 0.0
num_crease_discretization_points = 0
crease_parameters = []
crease_points_g = [] # List of compas.Point, MODIFIED IN PLACE by main thread

outer_edge_points1_folded_g = [] # MODIFIED IN PLACE by main thread
strip1_mesh = None
strip1_mesh_obj = None
strip1_crease_vertex_keys = []
strip1_outer_edge_vertex_keys = []
initial_outer_edge_length_s1 = 0.0
target_outer_segment_lengths_s1 = []

outer_edge_points2_folded_g = [] # MODIFIED IN PLACE by main thread
strip2_mesh = None
strip2_mesh_obj = None
strip2_crease_vertex_keys = []
strip2_outer_edge_vertex_keys = []
initial_outer_edge_length_s2 = 0.0
target_outer_segment_lengths_s2 = []

target_crease_segment_lengths_list = []
initial_total_crease_length = 0.0

fixed_crease_indices_global = set()
crease_polyline_obj = None

# --- Threading and Queue for Optimization ---
optimization_input_queue = queue.Queue(maxsize=1) # Max size 1 to process only the latest state
optimization_output_queue = queue.Queue()
optimization_thread = None
stop_event = threading.Event()


# --- Animation Parameters ---
BASE_REFLECTION_ANGLE_DEG = 5.0 # Initial angle of ONE strip from its flat position
AMPLITUDE_DEG = 40.0 # Max additional angle ONE strip will fold (e.g. 44 makes total 45 for 90 deg between strips)
FREQUENCY = 0.01 # Speed of folding animation

# --- Optimization parameters (can be tuned) ---
OPTIMIZER_ITERATIONS = 100 # Reduced for potentially faster updates in threaded context
OPTIMIZER_LR_CREASE = 0.01 # Learning rate for crease points
OPTIMIZER_LR_OUTER = 0.01  # Learning rate for outer edge points
K_RULING = 1.0 # Stiffness for ruling lines (strip width)
K_OUTER_EDGE = 1.0 # Stiffness for outer edge segment lengths
K_CREASE_SEGMENT = 1.0 # Stiffness for individual crease segment lengths
K_FLATNESS = 1.0 # Stiffness for quad planarity (developability)
K_REFLECTION_ANGLE = 1.0 # Stiffness for achieving target angle between strips
K_INTERNAL_ANGLES = 1.0 # New: Stiffness for maintaining 90-degree internal quad angles

plane_width = 1.0
num_crease_discretization_points = 10 # Number of points along the crease

# --- Helper Functions ---
def get_polyline_tangent(polyline_points, index, epsilon=1e-7):
    """Calculates the tangent vector at a point on a polyline."""
    n = len(polyline_points)
    if n < 2: return Vector(1, 0, 0)
    p_current = polyline_points[index]
    vec = None
    if n == 2: vec = polyline_points[1] - polyline_points[0]
    elif index == 0: vec = polyline_points[1] - polyline_points[0]
    elif index == n - 1: vec = polyline_points[n - 1] - polyline_points[n - 2]
    else: vec = polyline_points[index + 1] - polyline_points[index - 1]
    if vec is None or vec.length < epsilon:
        if index < n - 1 and (polyline_points[index+1] - p_current).length > epsilon :
            vec = polyline_points[index+1] - p_current
        elif index > 0 and (p_current - polyline_points[index-1]).length > epsilon:
            vec = p_current - polyline_points[index-1]
        else: return Vector(1,0,0)
    return vec.unitized()

def calculate_single_strip_kinematic_guess(
    ref_crease_points, strip_width, normal_side_multiplier, rotation_angle_radians
):
    """Calculates initial outer edge points for a strip based on crease, width, and rotation."""
    new_outer_edge_points = []
    num_pts = len(ref_crease_points)
    if num_pts == 0: return []
    for i in range(num_pts):
        p_crease_anchor = ref_crease_points[i]
        tangent_vector = get_polyline_tangent(ref_crease_points, i)
        T_local = tangent_vector
        
        N_local_direction = Vector(0,0,0)
        if abs(T_local.x) < 1e-7 and abs(T_local.y) < 1e-7: # Tangent is vertical or near-vertical
            cross_check_vec = Vector(1.0,0.0,0.0)
            if T_local.is_parallel(cross_check_vec, tol=1e-5):
                cross_check_vec = Vector(0.0,1.0,0.0)
            temp_N = T_local.cross(cross_check_vec)
            if temp_N.length > 1e-7: N_local_direction = temp_N
            else:
                another_cross_check = Vector(0.0,1.0,0.0) if cross_check_vec.x == 1.0 else Vector(1.0,0.0,0.0)
                temp_N = T_local.cross(another_cross_check)
                if temp_N.length > 1e-7: N_local_direction = temp_N
                else: N_local_direction = Vector(1.0,0.0,0.0) # Fallback
        else: # Tangent is not vertical
            global_z_axis = Vector(0.0, 0.0, 1.0)
            N_local_direction = global_z_axis.cross(T_local)
            if N_local_direction.length < 1e-7: # Tangent is parallel to Z (should be caught above, but as fallback)
                temp_N_fallback = Vector(1.0,0.0,0.0)
                if T_local.is_parallel(temp_N_fallback, tol=1e-5):
                    temp_N_fallback = Vector(0.0,1.0,0.0)
                N_local_direction = T_local.cross(temp_N_fallback)

        if N_local_direction.length < 1e-7: N_local_direction = Vector(1.0,0.0,0.0) # Final fallback
        N_local_direction.unitize()
        
        initial_outer_point_flat_vec = N_local_direction.scaled(strip_width * normal_side_multiplier)
        initial_outer_point_flat = p_crease_anchor + initial_outer_point_flat_vec
        
        rotation = Rotation.from_axis_and_angle(T_local, rotation_angle_radians, point=p_crease_anchor)
        rotated_outer_point = initial_outer_point_flat.transformed(rotation)
        new_outer_edge_points.append(rotated_outer_point)
    return new_outer_edge_points

def optimize_fold_system_geometry_pytorch(
    initial_crease_coords_np, initial_outer_s1_coords_np, initial_outer_s2_coords_np, # NumPy arrays
    target_crease_seg_lengths_list_val,
    target_outer_s1_seg_lengths_list, target_outer_s2_seg_lengths_list,
    strip_width_val, fixed_crease_indices_set,
    current_reflection_angle_rad_val,
    iterations=OPTIMIZER_ITERATIONS,
    learning_rate_crease_val=OPTIMIZER_LR_CREASE,
    learning_rate_outer_val=OPTIMIZER_LR_OUTER,
    k_ruling=K_RULING, k_outer_edge=K_OUTER_EDGE,
    k_crease_segment=K_CREASE_SEGMENT,
    k_flatness=K_FLATNESS,
    k_reflection_angle=K_REFLECTION_ANGLE,
    k_internal_angles=K_INTERNAL_ANGLES,
    epsilon = 1e-7
):
    """
    Optimizes the geometry of the folding system using PyTorch.
    Accepts initial coordinates as NumPy arrays and returns optimized coordinates as NumPy arrays.
    """
    device = torch.device("cpu") # Or "cuda" if available and desired

    crease_coords = torch.tensor(initial_crease_coords_np, dtype=torch.float32, device=device, requires_grad=True)
    outer_s1_coords = torch.tensor(initial_outer_s1_coords_np, dtype=torch.float32, device=device, requires_grad=True)
    outer_s2_coords = torch.tensor(initial_outer_s2_coords_np, dtype=torch.float32, device=device, requires_grad=True)

    num_all_c_pts = crease_coords.shape[0]
    num_o1_pts = outer_s1_coords.shape[0]
    num_o2_pts = outer_s2_coords.shape[0]

    if not (num_all_c_pts == num_o1_pts == num_o2_pts and num_all_c_pts > 0):
        print("Warning: Mismatch in point counts or zero points in optimization. Returning initial coords.")
        return initial_crease_coords_np, initial_outer_s1_coords_np, initial_outer_s2_coords_np

    target_crease_seg_tensor = torch.tensor(target_crease_seg_lengths_list_val, dtype=torch.float32, device=device)
    target_oe_s1_tensor = torch.tensor(target_outer_s1_seg_lengths_list, dtype=torch.float32, device=device)
    target_oe_s2_tensor = torch.tensor(target_outer_s2_seg_lengths_list, dtype=torch.float32, device=device)

    optimizer = optim.SGD([
        {'params': crease_coords, 'lr': learning_rate_crease_val},
        {'params': outer_s1_coords, 'lr': learning_rate_outer_val},
        {'params': outer_s2_coords, 'lr': learning_rate_outer_val}
    ])

    torch_fixed_crease_indices = torch.tensor(list(fixed_crease_indices_set), dtype=torch.long, device=device)
    
    target_opening_angle_rad = torch.pi - 2.0 * current_reflection_angle_rad_val
    cos_target_opening_angle_tensor = torch.cos(torch.tensor(target_opening_angle_rad, dtype=torch.float32, device=device))

    for iteration in range(iterations):
        optimizer.zero_grad()
        total_loss = torch.tensor(0.0, dtype=torch.float32, device=device)
        
        # --- Strip 1 Losses ---
        if num_o1_pts > 0:
            diff_r_s1 = outer_s1_coords - crease_coords
            len_r_s1 = torch.norm(diff_r_s1, dim=1, p=2) + epsilon
            loss_r_s1 = torch.sum(k_ruling * (len_r_s1 - strip_width_val)**2)
            total_loss += loss_r_s1
            if num_o1_pts > 1 and len(target_oe_s1_tensor) == num_o1_pts - 1:
                diff_oe_s1 = outer_s1_coords[1:] - outer_s1_coords[:-1]
                len_oe_s1 = torch.norm(diff_oe_s1, dim=1, p=2) + epsilon
                loss_oe_s1 = torch.sum(k_outer_edge * (len_oe_s1 - target_oe_s1_tensor)**2)
                total_loss += loss_oe_s1

        # --- Strip 2 Losses ---
        if num_o2_pts > 0:
            diff_r_s2 = outer_s2_coords - crease_coords
            len_r_s2 = torch.norm(diff_r_s2, dim=1, p=2) + epsilon
            loss_r_s2 = torch.sum(k_ruling * (len_r_s2 - strip_width_val)**2)
            total_loss += loss_r_s2
            if num_o2_pts > 1 and len(target_oe_s2_tensor) == num_o2_pts - 1:
                diff_oe_s2 = outer_s2_coords[1:] - outer_s2_coords[:-1]
                len_oe_s2 = torch.norm(diff_oe_s2, dim=1, p=2) + epsilon
                loss_oe_s2 = torch.sum(k_outer_edge * (len_oe_s2 - target_oe_s2_tensor)**2)
                total_loss += loss_oe_s2

        # --- Individual Crease Segment Length Loss ---
        if num_all_c_pts > 1 and len(target_crease_seg_tensor) == num_all_c_pts -1 :
            crease_segments = crease_coords[1:] - crease_coords[:-1]
            len_crease_segments = torch.norm(crease_segments, dim=1, p=2) + epsilon
            loss_crease_segments = k_crease_segment * torch.sum((len_crease_segments - target_crease_seg_tensor)**2)
            total_loss += loss_crease_segments
        
        # --- Reflection Angle Loss ---
        if num_all_c_pts > 0:
            v1_rulings = outer_s1_coords - crease_coords
            v2_rulings = outer_s2_coords - crease_coords
            
            dot_product_v1_v2 = torch.sum(v1_rulings * v2_rulings, dim=1)
            norm_v1 = torch.norm(v1_rulings, dim=1, p=2) + epsilon
            norm_v2 = torch.norm(v2_rulings, dim=1, p=2) + epsilon
            
            cos_phi_rulings = dot_product_v1_v2 / (norm_v1 * norm_v2)
            cos_phi_rulings = torch.clamp(cos_phi_rulings, -1.0 + epsilon, 1.0 - epsilon)

            loss_reflection = k_reflection_angle * torch.sum((cos_phi_rulings - cos_target_opening_angle_tensor)**2)
            total_loss += loss_reflection
        
        # --- Gaussian Flatness Loss ---
        loss_flatness_s1 = torch.tensor(0.0, dtype=torch.float32, device=device)
        if num_o1_pts > 1:
            P0_s1_flat = crease_coords[:-1]    
            P1_s1_flat = crease_coords[1:]     
            P3_s1_flat = outer_s1_coords[:-1]  
            P2_s1_flat = outer_s1_coords[1:]   
            u_s1 = P1_s1_flat - P0_s1_flat  
            v_s1 = P3_s1_flat - P0_s1_flat  
            w_s1 = P2_s1_flat - P0_s1_flat  
            cross_vw_s1 = torch.cross(v_s1, w_s1, dim=1)
            scalar_triple_s1 = torch.sum(u_s1 * cross_vw_s1, dim=1)
            loss_flatness_s1 = torch.sum(scalar_triple_s1**2)
        total_loss += k_flatness * loss_flatness_s1

        loss_flatness_s2 = torch.tensor(0.0, dtype=torch.float32, device=device)
        if num_o2_pts > 1:
            P0_s2_flat = crease_coords[:-1]
            P1_s2_flat = crease_coords[1:]
            P3_s2_flat = outer_s2_coords[:-1]
            P2_s2_flat = outer_s2_coords[1:]
            u_s2 = P1_s2_flat - P0_s2_flat
            v_s2 = P3_s2_flat - P0_s2_flat
            w_s2 = P2_s2_flat - P0_s2_flat
            cross_vw_s2 = torch.cross(v_s2, w_s2, dim=1)
            scalar_triple_s2 = torch.sum(u_s2 * cross_vw_s2, dim=1)
            loss_flatness_s2 = torch.sum(scalar_triple_s2**2)
        total_loss += k_flatness * loss_flatness_s2
        
        # --- Internal Quad Angles Loss ---
        loss_internal_angles_val = torch.tensor(0.0, dtype=torch.float32, device=device)
        if num_all_c_pts > 1: 
            # Strip 1
            c_i_s1   = crease_coords[:-1]
            c_ip1_s1 = crease_coords[1:]
            o_i_s1   = outer_s1_coords[:-1]
            o_ip1_s1 = outer_s1_coords[1:]

            vec_ci_cip1_s1 = c_ip1_s1 - c_i_s1
            vec_ci_oi_s1   = o_i_s1   - c_i_s1
            vec_cip1_ci_s1   = c_i_s1   - c_ip1_s1
            vec_cip1_oip1_s1 = o_ip1_s1 - c_ip1_s1
            vec_oip1_cip1_s1 = c_ip1_s1 - o_ip1_s1
            vec_oip1_oi_s1   = o_i_s1   - o_ip1_s1
            vec_oi_ci_s1   = c_i_s1   - o_i_s1
            vec_oi_oip1_s1 = o_ip1_s1 - o_i_s1

            cos_angle_ci_s1   = torch.sum(torch.nn.functional.normalize(vec_ci_cip1_s1, dim=-1, p=2) * torch.nn.functional.normalize(vec_ci_oi_s1, dim=-1, p=2), dim=-1)
            cos_angle_cip1_s1 = torch.sum(torch.nn.functional.normalize(vec_cip1_ci_s1, dim=-1, p=2) * torch.nn.functional.normalize(vec_cip1_oip1_s1, dim=-1, p=2), dim=-1)
            cos_angle_oip1_s1 = torch.sum(torch.nn.functional.normalize(vec_oip1_cip1_s1, dim=-1, p=2) * torch.nn.functional.normalize(vec_oip1_oi_s1, dim=-1, p=2), dim=-1)
            cos_angle_oi_s1   = torch.sum(torch.nn.functional.normalize(vec_oi_ci_s1, dim=-1, p=2) * torch.nn.functional.normalize(vec_oi_oip1_s1, dim=-1, p=2), dim=-1)
            
            loss_internal_angles_val += torch.sum(cos_angle_ci_s1**2) + torch.sum(cos_angle_cip1_s1**2) + torch.sum(cos_angle_oip1_s1**2) + torch.sum(cos_angle_oi_s1**2)

            # Strip 2
            c_i_s2   = crease_coords[:-1]
            c_ip1_s2 = crease_coords[1:]
            o_i_s2   = outer_s2_coords[:-1]
            o_ip1_s2 = outer_s2_coords[1:]

            vec_ci_cip1_s2 = c_ip1_s2 - c_i_s2
            vec_ci_oi_s2   = o_i_s2   - c_i_s2
            vec_cip1_ci_s2   = c_i_s2   - c_ip1_s2
            vec_cip1_oip1_s2 = o_ip1_s2 - c_ip1_s2
            vec_oip1_cip1_s2 = c_ip1_s2 - o_ip1_s2
            vec_oip1_oi_s2   = o_i_s2   - o_ip1_s2
            vec_oi_ci_s2   = c_i_s2   - o_i_s2
            vec_oi_oip1_s2 = o_ip1_s2 - o_i_s2

            cos_angle_ci_s2   = torch.sum(torch.nn.functional.normalize(vec_ci_cip1_s2, dim=-1, p=2) * torch.nn.functional.normalize(vec_ci_oi_s2, dim=-1, p=2), dim=-1)
            cos_angle_cip1_s2 = torch.sum(torch.nn.functional.normalize(vec_cip1_ci_s2, dim=-1, p=2) * torch.nn.functional.normalize(vec_cip1_oip1_s2, dim=-1, p=2), dim=-1)
            cos_angle_oip1_s2 = torch.sum(torch.nn.functional.normalize(vec_oip1_cip1_s2, dim=-1, p=2) * torch.nn.functional.normalize(vec_oip1_oi_s2, dim=-1, p=2), dim=-1)
            cos_angle_oi_s2   = torch.sum(torch.nn.functional.normalize(vec_oi_ci_s2, dim=-1, p=2) * torch.nn.functional.normalize(vec_oi_oip1_s2, dim=-1, p=2), dim=-1)
            
            loss_internal_angles_val += torch.sum(cos_angle_ci_s2**2) + torch.sum(cos_angle_cip1_s2**2) + torch.sum(cos_angle_oip1_s2**2) + torch.sum(cos_angle_oi_s2**2)
        
        total_loss += k_internal_angles * loss_internal_angles_val

        total_loss.backward()
        if crease_coords.grad is not None and len(torch_fixed_crease_indices) > 0:
            valid_fixed_indices = torch_fixed_crease_indices[torch_fixed_crease_indices < num_all_c_pts]
            if len(valid_fixed_indices) > 0:
                crease_coords.grad.index_fill_(0, valid_fixed_indices, 0.0)
        optimizer.step()

    # Detach and convert to NumPy arrays to send back
    optimized_c_data = crease_coords.detach().cpu().numpy()
    optimized_o1_data = outer_s1_coords.detach().cpu().numpy()
    optimized_o2_data = outer_s2_coords.detach().cpu().numpy()

    return optimized_c_data, optimized_o1_data, optimized_o2_data


def optimization_worker():
    """Worker function to run optimizations in a separate thread."""
    print("Optimization thread started.")
    while not stop_event.is_set():
        try:
            # Wait for a task from the input queue (with a timeout to check stop_event)
            task_data = optimization_input_queue.get(timeout=0.1)
        except queue.Empty:
            continue # No task, loop again to check stop_event

        # Unpack task data
        crease_np = task_data["crease_np"]
        outer1_np = task_data["outer1_np"]
        outer2_np = task_data["outer2_np"]
        target_crease_seg = task_data["target_crease_seg"]
        target_outer1_seg = task_data["target_outer1_seg"]
        target_outer2_seg = task_data["target_outer2_seg"]
        strip_width = task_data["strip_width"]
        fixed_indices = task_data["fixed_indices"]
        current_alpha = task_data["current_alpha"]

        # Run the optimization
        opt_c, opt_o1, opt_o2 = optimize_fold_system_geometry_pytorch(
            crease_np, outer1_np, outer2_np,
            target_crease_seg,
            target_outer1_seg, target_outer2_seg,
            strip_width, fixed_indices,
            current_alpha
            # Other K values are taken from globals by default in the function
        )

        # Put the results into the output queue
        try:
            optimization_output_queue.put({
                "crease": opt_c,
                "outer1": opt_o1,
                "outer2": opt_o2
            }, block=False) # Non-blocking put
        except queue.Full:
            # Should not happen if main thread is consuming, but good to be aware
            print("Optimization output queue is full. Discarding result.")
            pass

        optimization_input_queue.task_done() # Signal that this task is complete

    print("Optimization thread stopped.")


# --- Main Script ---
if __name__ == "__main__":
    # Define initial crease curve
    cp1, cp2, cp3, cp4 = Point(-2,0,0), Point(-1,1.5,0), Point(1,1.5,0), Point(2,0,0)
    initial_curved_crease_nurbs = NurbsCurve.from_points([cp1,cp2,cp3,cp4], degree=3)
    domain_start, domain_end = initial_curved_crease_nurbs.domain
    if domain_start is None or domain_end is None : raise ValueError("Curve domain error.")

    crease_parameters = ([domain_start + i * (domain_end - domain_start) / (num_crease_discretization_points - 1)
                          for i in range(num_crease_discretization_points)]
                         if num_crease_discretization_points > 1 else [domain_start])
    crease_points_g = [initial_curved_crease_nurbs.point_at(param) for param in crease_parameters]
    if not crease_points_g: raise ValueError("Crease points generation failed.")
    
    target_crease_segment_lengths_list = ([crease_points_g[k].distance_to_point(crease_points_g[k+1]) for k in range(len(crease_points_g)-1)] if len(crease_points_g)>1 else [])
    initial_total_crease_length = sum(target_crease_segment_lengths_list)
    print(f"Initial total crease length (sum of target segments): {initial_total_crease_length:.4f}")
    
    fixed_crease_indices_global = {0, num_crease_discretization_points - 1} if num_crease_discretization_points > 1 else {0}

    flat_outer_s1 = calculate_single_strip_kinematic_guess(crease_points_g, plane_width, -1, 0.0)
    flat_outer_s2 = calculate_single_strip_kinematic_guess(crease_points_g, plane_width, +1, 0.0)
    if not flat_outer_s1 or not flat_outer_s2 : raise ValueError("Flat outer edge calculation failed")

    target_outer_segment_lengths_s1 = ([flat_outer_s1[k].distance_to_point(flat_outer_s1[k+1]) for k in range(len(flat_outer_s1)-1)] if len(flat_outer_s1)>1 else [])
    target_outer_segment_lengths_s2 = ([flat_outer_s2[k].distance_to_point(flat_outer_s2[k+1]) for k in range(len(flat_outer_s2)-1)] if len(flat_outer_s2)>1 else [])
    initial_outer_edge_length_s1 = sum(target_outer_segment_lengths_s1)
    initial_outer_edge_length_s2 = sum(target_outer_segment_lengths_s2)
    print(f"Target outer S1 total len: {initial_outer_edge_length_s1:.4f}, S2 total len: {initial_outer_edge_length_s2:.4f}")

    initial_alpha_rad_setup = math.radians(BASE_REFLECTION_ANGLE_DEG)
    outer_edge_points1_folded_g = calculate_single_strip_kinematic_guess(crease_points_g, plane_width, -1, initial_alpha_rad_setup)
    outer_edge_points2_folded_g = calculate_single_strip_kinematic_guess(crease_points_g, plane_width, +1, -initial_alpha_rad_setup)
    if not outer_edge_points1_folded_g or not outer_edge_points2_folded_g : raise ValueError("Initial folded outer edge calculation failed")

    # --- Create COMPAS Meshes for Visualization ---
    strip1_mesh = Mesh()
    if crease_points_g: strip1_crease_vertex_keys = [strip1_mesh.add_vertex(x=pt.x,y=pt.y,z=pt.z) for pt in crease_points_g]
    if outer_edge_points1_folded_g: strip1_outer_edge_vertex_keys = [strip1_mesh.add_vertex(x=pt.x,y=pt.y,z=pt.z) for pt in outer_edge_points1_folded_g]
    if num_crease_discretization_points > 1 and len(strip1_crease_vertex_keys) == num_crease_discretization_points and len(strip1_outer_edge_vertex_keys) == num_crease_discretization_points:
        for i in range(num_crease_discretization_points - 1):
            strip1_mesh.add_face([strip1_crease_vertex_keys[i], strip1_crease_vertex_keys[i+1], strip1_outer_edge_vertex_keys[i+1], strip1_outer_edge_vertex_keys[i]])

    strip2_mesh = Mesh()
    if crease_points_g: strip2_crease_vertex_keys = [strip2_mesh.add_vertex(x=pt.x,y=pt.y,z=pt.z) for pt in crease_points_g]
    if outer_edge_points2_folded_g: strip2_outer_edge_vertex_keys = [strip2_mesh.add_vertex(x=pt.x,y=pt.y,z=pt.z) for pt in outer_edge_points2_folded_g]
    if num_crease_discretization_points > 1 and len(strip2_crease_vertex_keys) == num_crease_discretization_points and len(strip2_outer_edge_vertex_keys) == num_crease_discretization_points:
        for i in range(num_crease_discretization_points - 1):
            strip2_mesh.add_face([strip2_crease_vertex_keys[i], strip2_crease_vertex_keys[i+1], strip2_outer_edge_vertex_keys[i+1], strip2_outer_edge_vertex_keys[i]])

    # --- COMPAS Viewer Setup ---
    viewer_instance = Viewer(width=1600, height=900, show_grid=True, viewmode='shaded')
    crease_polyline_obj = viewer_instance.scene.add(Polyline(crease_points_g), name="CreasePolyline", linecolor=Color.black(), linewidth=3)
    
    strip1_mesh_obj = None
    if strip1_mesh.number_of_vertices() > 0: strip1_mesh_obj = viewer_instance.scene.add(strip1_mesh, facecolor=Color.green().lightened(70), name="Strip1", linewidth=0.5, show_edges=True)
    strip2_mesh_obj = None
    if strip2_mesh.number_of_vertices() > 0: strip2_mesh_obj = viewer_instance.scene.add(strip2_mesh, facecolor=Color.blue().lightened(70), name="Strip2", linewidth=0.5, show_edges=True)

    # --- Animation Callback for Viewer ---
    @viewer_instance.on(interval=50) # Animation interval in ms
    def step_time_animation(frame, epsilon=1e-7):
        global current_time_step, crease_points_g, outer_edge_points1_folded_g, outer_edge_points2_folded_g
        global strip1_mesh, strip2_mesh, strip1_mesh_obj, strip2_mesh_obj, crease_polyline_obj
        # Target lengths and fixed indices are effectively constant after setup for optimization
        # plane_width and initial_total_crease_length are also constant

        current_time_step += 1
        dynamic_reflection_angle_deg = BASE_REFLECTION_ANGLE_DEG + AMPLITUDE_DEG * math.sin(FREQUENCY * current_time_step)
        current_alpha_rad = math.radians(dynamic_reflection_angle_deg)
        
        # --- Prepare data for optimization thread ---
        # Convert current COMPAS points to NumPy arrays for the optimization thread
        # These are copies, so the optimization thread works on detached data
        current_crease_np = [[p.x, p.y, p.z] for p in crease_points_g]
        current_outer1_np = [[p.x, p.y, p.z] for p in outer_edge_points1_folded_g]
        current_outer2_np = [[p.x, p.y, p.z] for p in outer_edge_points2_folded_g]

        optimization_task = {
            "crease_np": current_crease_np,
            "outer1_np": current_outer1_np,
            "outer2_np": current_outer2_np,
            "target_crease_seg": target_crease_segment_lengths_list,
            "target_outer1_seg": target_outer_segment_lengths_s1,
            "target_outer2_seg": target_outer_segment_lengths_s2,
            "strip_width": plane_width,
            "fixed_indices": fixed_crease_indices_global,
            "current_alpha": current_alpha_rad
        }

        # Try to put the task in the queue. If full, it means the optimizer is busy.
        # The old task in the queue (if any) will be processed.
        # A more advanced approach might clear the queue and put the new one,
        # but maxsize=1 handles this by ensuring only one pending task.
        try:
            # Clear the queue first to ensure the latest frame data is used if optimizer is lagging
            while not optimization_input_queue.empty():
                try:
                    optimization_input_queue.get_nowait()
                except queue.Empty:
                    break # Queue is empty
            optimization_input_queue.put_nowait(optimization_task)
        except queue.Full:
            # print("Optimizer busy, skipping frame for optimization input.")
            pass # Optimizer is busy, it will work on the previous task.

        # --- Check for results from optimization thread ---
        try:
            optimized_data = optimization_output_queue.get_nowait() # Non-blocking get
            
            opt_c_data = optimized_data["crease"]
            opt_o1_data = optimized_data["outer1"]
            opt_o2_data = optimized_data["outer2"]

            # Update global COMPAS Point lists with optimized data
            # This modification happens in the main thread
            if len(crease_points_g) == len(opt_c_data):
                for i in range(len(crease_points_g)):
                    crease_points_g[i].x, crease_points_g[i].y, crease_points_g[i].z = opt_c_data[i]
            
            if len(outer_edge_points1_folded_g) == len(opt_o1_data):
                for i in range(len(outer_edge_points1_folded_g)):
                    outer_edge_points1_folded_g[i].x, outer_edge_points1_folded_g[i].y, outer_edge_points1_folded_g[i].z = opt_o1_data[i]

            if len(outer_edge_points2_folded_g) == len(opt_o2_data):
                for i in range(len(outer_edge_points2_folded_g)):
                    outer_edge_points2_folded_g[i].x, outer_edge_points2_folded_g[i].y, outer_edge_points2_folded_g[i].z = opt_o2_data[i]
            
            optimization_output_queue.task_done()

        except queue.Empty:
            # No new optimized data available yet, viewer will use current (possibly stale) data
            pass

        # --- Update Mesh Data and Visuals in Viewer (always, with current or updated global points) ---
        if strip1_mesh_obj and strip1_mesh:
            # Ensure vertex keys match the current number of points
            if len(strip1_crease_vertex_keys) == len(crease_points_g) and \
               len(strip1_outer_edge_vertex_keys) == len(outer_edge_points1_folded_g):
                for i, vk_c in enumerate(strip1_crease_vertex_keys):
                    strip1_mesh.vertex_attributes(vk_c, 'xyz', crease_points_g[i])
                for i, vk_o in enumerate(strip1_outer_edge_vertex_keys):
                    strip1_mesh.vertex_attributes(vk_o, 'xyz', outer_edge_points1_folded_g[i])
                strip1_mesh_obj.init(); strip1_mesh_obj.update(update_data=True)

        if strip2_mesh_obj and strip2_mesh:
            if len(strip2_crease_vertex_keys) == len(crease_points_g) and \
               len(strip2_outer_edge_vertex_keys) == len(outer_edge_points2_folded_g):
                for i, vk_c in enumerate(strip2_crease_vertex_keys):
                    strip2_mesh.vertex_attributes(vk_c, 'xyz', crease_points_g[i])
                for i, vk_o in enumerate(strip2_outer_edge_vertex_keys):
                    strip2_mesh.vertex_attributes(vk_o, 'xyz', outer_edge_points2_folded_g[i])
                strip2_mesh_obj.init(); strip2_mesh_obj.update(update_data=True)

        if crease_polyline_obj:
            if len(crease_points_g) == len(crease_polyline_obj.geometry.points):
                for i in range(len(crease_points_g)):
                    crease_polyline_obj.geometry.points[i] = crease_points_g[i]
            else: # Recreate if number of points changed (should not happen in this setup post-init)
                viewer_instance.scene.remove(crease_polyline_obj)
                crease_polyline_obj = viewer_instance.scene.add(Polyline(crease_points_g), name="CreasePolyline", linecolor=Color.black(), linewidth=3)
            
            crease_polyline_obj.init()
            crease_polyline_obj.update(update_data=True)

        # --- Print Status Periodically ---
        if current_time_step % 20 == 0: # Print status less frequently
            len_s1 = sum(outer_edge_points1_folded_g[k].distance_to_point(outer_edge_points1_folded_g[k+1]) for k in range(len(outer_edge_points1_folded_g)-1)) if len(outer_edge_points1_folded_g)>1 else 0
            len_s2 = sum(outer_edge_points2_folded_g[k].distance_to_point(outer_edge_points2_folded_g[k+1]) for k in range(len(outer_edge_points2_folded_g)-1)) if len(outer_edge_points2_folded_g)>1 else 0
            current_total_c_len = sum(crease_points_g[k].distance_to_point(crease_points_g[k+1]) for k in range(len(crease_points_g)-1)) if len(crease_points_g)>1 else 0
            
            avg_cos_phi = 0.0
            if num_crease_discretization_points > 0 and len(crease_points_g) == len(outer_edge_points1_folded_g) == len(outer_edge_points2_folded_g):
                device_cpu = torch.device("cpu")
                c_coords_T = torch.tensor([[p.x, p.y, p.z] for p in crease_points_g], dtype=torch.float32, device=device_cpu)
                o1_coords_T = torch.tensor([[p.x, p.y, p.z] for p in outer_edge_points1_folded_g], dtype=torch.float32, device=device_cpu)
                o2_coords_T = torch.tensor([[p.x, p.y, p.z] for p in outer_edge_points2_folded_g], dtype=torch.float32, device=device_cpu)
                
                v1_r = o1_coords_T - c_coords_T
                v2_r = o2_coords_T - c_coords_T
                
                dp_v1_v2 = torch.sum(v1_r * v2_r, dim=1)
                n_v1 = torch.norm(v1_r, dim=1, p=2) + epsilon
                n_v2 = torch.norm(v2_r, dim=1, p=2) + epsilon
                cos_phi_r = dp_v1_v2 / (n_v1 * n_v2)
                cos_phi_r_clamped = torch.clamp(cos_phi_r, -1.0 + epsilon, 1.0 - epsilon)
                avg_cos_phi = torch.mean(cos_phi_r_clamped).item()
            
            actual_opening_angle_deg = math.degrees(math.acos(avg_cos_phi)) if abs(avg_cos_phi) <=1.0 else (0.0 if avg_cos_phi > 1.0 else 180.0)
            # Target opening angle is 2 * reflection angle (angle of one strip to flat plane)
            # No, target_opening_angle_rad = torch.pi - 2.0 * current_reflection_angle_rad_val
            # So, target_opening_angle_deg = 180 - 2 * dynamic_reflection_angle_deg
            target_opening_angle_deg = 180.0 - (2.0 * dynamic_reflection_angle_deg)


            print(f"T:{current_time_step} RefAngle(1):{dynamic_reflection_angle_deg:.1f}° OpenAngle(T):{target_opening_angle_deg:.1f}° OpenAngle(A):{actual_opening_angle_deg:.1f}°")
            print(f"L_S1:{len_s1:.3f}(T:{initial_outer_edge_length_s1:.3f}) L_S2:{len_s2:.3f}(T:{initial_outer_edge_length_s2:.3f}) L_C:{current_total_c_len:.3f}(T:{initial_total_crease_length:.3f})")

    # --- Start the Optimization Thread ---
    optimization_thread = threading.Thread(target=optimization_worker, daemon=True) # Daemon so it exits when main exits
    optimization_thread.start()

    # --- Start the Viewer ---
    print("Starting COMPAS Viewer with deforming crease (Optimizer in separate thread)...")
    try:
        viewer_instance.show()
    finally:
        print("Viewer closed or error occurred. Stopping optimization thread...")
        stop_event.set() # Signal the thread to stop
        if optimization_thread and optimization_thread.is_alive():
            optimization_thread.join(timeout=2) # Wait for the thread to finish
        if optimization_thread and optimization_thread.is_alive():
            print("Optimization thread did not terminate cleanly.")
        print("Exiting.")
