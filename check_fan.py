import math
from compas.geometry import Point, Vector, Plane, mirror_points_plane
from compas.datastructures import Mesh

def generate_radial_planes(center_coords, num_planes, angle_start, angle_end):
    planes = []
    cx, cy = center_coords
    for i in range(num_planes):
        t = i / (num_planes - 1) if num_planes > 1 else 0
        angle = angle_start + t * (angle_end - angle_start)
        nx = -math.sin(angle)
        ny = math.cos(angle)
        planes.append(Plane(Point(cx, cy, 0), Vector(nx, ny, 0)))
    return planes

center_coords = (1.5, 0.9)
num_planes = 13
angle_start = 0.0
angle_end = math.pi / 2.0

planes = generate_radial_planes(center_coords, num_planes, angle_start, angle_end)

# Start with a point on Plane 0
p0 = Point(3.0, 0.9, 0.4) # x=3.0, y=0.9 is on Plane 0 (Normal=(0,1,0))
print(f"p0: {p0}")

# Strip 1: Reflect p0 across Plane 1 to get p2 on Plane 2?
# No, if Strip 0 is between Plane 0 and Plane 1, 
# then its points are p_0 (on Plane 0) and p_1 (on Plane 1).
# Reflection across Plane 1 takes p_0 to p_2 (on Plane 2).

p_curr_prev = p0
for i in range(1, num_planes - 1):
    reflection_plane = planes[i]
    p_next = Point(*mirror_points_plane([p_curr_prev], reflection_plane)[0])
    print(f"Step {i}: Reflect {p_curr_prev} across Plane {i} -> {p_next}")
    # Check if p_next is on Plane i+1?
    p_target = planes[i+1]
    dist = abs((p_next.x - p_target.point.x)*p_target.normal.x + (p_next.y - p_target.point.y)*p_target.normal.y)
    print(f"  Distance to Plane {i+1}: {dist:.6f}")
    p_curr_prev = p0 # Reset to p0 for each reflection? NO, reflection propagates.
    # Wait, if we reflect p0 across Plane 1, we get p2.
    # Then we reflect p1 across Plane 2 to get p3?
    # Strip 0: p0, p1
    # Strip 1: p1, p2
    # Strip 2: p2, p3
    # ...
    # Reflection: Strip i = Mirror(Strip i-1, Plane i)
    # This means (p_i, p_{i+1}) = Mirror((p_{i-1}, p_i), Plane i)
    # p_i reflects to p_i (it is on the plane).
    # p_{i-1} reflects to p_{i+1}.
    
p_prev = p0
# We need an initial p1 on Plane 1 to start.
# Let's say we extracted p1 from target mesh on Plane 1.
angle1 = 1 / (num_planes - 1) * (math.pi/2)
# Distance from center to p0 is 1.5
r = 1.5
p1 = Point(center_coords[0] + r*math.cos(angle1), center_coords[1] + r*math.sin(angle1), 0.4)
print(f"\nPropagation Test:")
print(f"p0: {p0}")
print(f"p1: {p1}")

curr_p_prev = p0
curr_p_curr = p1

for i in range(1, num_planes - 1):
    reflection_plane = planes[i]
    p_next = Point(*mirror_points_plane([curr_p_prev], reflection_plane)[0])
    print(f"Strip {i}: p{i}={curr_p_curr} -> p{i+1}={p_next}")
    p_target = planes[i+1]
    dist = abs((p_next.x - p_target.point.x)*p_target.normal.x + (p_next.y - p_target.point.y)*p_target.normal.y)
    print(f"  Distance of p{i+1} to Plane {i+1}: {dist:.6f}")
    curr_p_prev = curr_p_curr
    curr_p_curr = p_next

