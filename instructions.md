System Context & Objective
You are an expert computational designer specializing in discrete differential geometry and structural engineering using the COMPAS framework in Python. Your task is to refactor a Python script (vault_logic.py) that generates the geometry for a Curved-Crease Unfolding (CCU) bending-active formwork for a fan vault.

The Problem with the Current Code
The current script erroneously uses a "lofting" approach. It extracts structural spokes from a minimum and maximum thrust network, treats these as edge catenaries, and tries to unfold the surface bridged between them. This approach is geometrically flawed for CCU:

    It forces a non-developable surface flat, resulting in large quad distortions.

    It violates the kinematic generator of CCU, which relies on folding a continuous flat sheet along a curved crease. Min/max thrust lines define a structural boundary (the kern), not the geometric generation of the creases.

The Solution: The Reflection Method
You must rewrite the geometric generation pipeline to use the Reflection Method. The new logic must decouple the structural limits from the geometric generation.

Required Implementation Steps:

1. Define a Single Target Funicular Mesh:
Instead of using form_min and form_max, the script should accept a single target_mesh representing the optimal global mid-surface of the fan vault.

2. Generate Radial Osculating Planes:
Write a function generate_radial_planes(center_coords, num_planes, angle_start, angle_end) that creates vertical (or slightly inclined) radial planes spreading outward from the column capital. These act as the reflection planes.

3. Extract Planar Creases:
Write a function extract_planar_creases(target_mesh, radial_planes) that uses intersection_mesh_plane to slice the target mesh. This guarantees that all 3D crease curves are strictly planar (torsion τ=0), which is a fundamental requirement for flat-foldability in this context.

4. Generate Plates by Reflection:
Write a function generate_plate_by_reflection(ruling_lines_prev, reflection_plane) that mirrors the discrete ruling lines (from the previous plate) across the osculating plane to perfectly generate the geometry of the adjacent plate. Because it uses pure reflection (mirror_points_plane), the resulting plate is mathematically guaranteed to be developable and flat-foldable.

5. Integrate the Kinematic Constraint:
Ensure there is a validation check or initialization step that respects the CCU kinematic constraint:
cosα=∣r2D​∣∣r3D​∣​
Where α is the reflection angle, r3D​ is the 3D crease radius, and r2D​ is the 2D flattened crease radius.

Testing & Validation Instructions
Please also generate a test suite (using pytest or a standard if __name__ == '__main__': execution block) to validate the new geometry pipeline. The tests must assert the following:

    Planarity Test: Iterate through all generated 3D crease curves. Check that all vertices of a given crease lie strictly on its corresponding osculating plane (within a 1e-6 tolerance).

    Developability/Distortion Test: Calculate the unrolled flat mesh for each generated plate. Assert that the quad_distortion (difference between 3D diagonal lengths and 2D diagonal lengths) is less than 1e-5. If it is higher, the surface is not developable.

    Kern Containment Test (Mock): Write a placeholder function that checks if the depth of the generated corrugations (the distance between ridge and valley vertices) successfully encapsulates a provided form_min and form_max height limit.

Please output the fully refactored Python code utilizing the compas library geometry modules.
