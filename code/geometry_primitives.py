import math
from compas.geometry import Point, Polyline, Vector

def get_planar_quad():
    """Level 0: Planar Quad - A single 4-point shape in a plane."""
    p1 = Polyline([Point(0, 0, 0), Point(1, 0, 0)])
    p2 = Polyline([Point(0, 1, 0), Point(1, 1, 0)])
    return p1, p2

def get_tilted_straight_strip():
    """Level 1: Tilted Straight Strip - Planar quads aligned but tilted."""
    # Length per segment = sqrt(2)
    p1 = Polyline([Point(0, 0, 0), Point(1, 0, 1), Point(2, 0, 2)])
    p2 = Polyline([Point(0, 1, 0), Point(1, 1, 1), Point(2, 1, 2)])
    return p1, p2

def get_planar_curved_strip(n=5, radius=5.0, width=1.0):
    """Level 2: Planar Curved Strip - An arc on a 2D plane."""
    pts1 = []
    pts2 = []
    angle_step = math.pi / (2 * n)
    for i in range(n + 1):
        angle = i * angle_step
        pts1.append(Point(radius * math.cos(angle), radius * math.sin(angle), 0))
        pts2.append(Point((radius + width) * math.cos(angle), (radius + width) * math.sin(angle), 0))
    return Polyline(pts1), Polyline(pts2)

def get_ruled_cylinder_strip(n=5, radius=5.0, height=5.0, arc_angle=math.pi/4):
    """Level 3: Ruled Cylinder Strip - Perfectly developable 3D strip."""
    pts1 = []
    pts2 = []
    angle_step = arc_angle / n
    for i in range(n + 1):
        angle = i * angle_step
        pts1.append(Point(radius * math.cos(angle), radius * math.sin(angle), 0))
        pts2.append(Point(radius * math.cos(angle), radius * math.sin(angle), height))
    return Polyline(pts1), Polyline(pts2)

def get_uneven_spokes_strip():
    """Level 4: Uneven Spokes - One spoke longer than the other."""
    p1 = Polyline([Point(0,0,0), Point(1,0,0), Point(2,0,0)])
    p2 = Polyline([Point(0,1,0), Point(1,1,0)])
    return p1, p2

if __name__ == "__main__":
    p1, p2 = get_planar_quad()
    print(f"Planar Quad P1: {p1.points}")
    print(f"Planar Quad P2: {p2.points}")
