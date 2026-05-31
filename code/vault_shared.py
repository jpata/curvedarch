import math
import numpy as np

# ----------------------------------------
# 1. Shared Configuration
# ----------------------------------------

CONFIG = {
    'xy_span': [[0.0, 5.0], [0.0, 5.0]],
    'thickness': 0.8,
    'max_rise': 2.5,
    'discretisation_level': 6, # For envelope meshes
    'form_discretisation': 6,   # For FormDiagram
    'solver': 'IPOPT',           # Preferred solver
    'support_type': 'corners'  # 'corners' or 'perimeter'
}

# ----------------------------------------
# 2. Shared Geometric Logic
# ----------------------------------------

def crossvault_middle_hc(x, y, x_span, y_span, hc, tol=1e-6):
    """
    Calculate the z-coordinate of the middle surface of a cross vault
    using a circular arc quadrant logic.
    """
    x0, x1 = x_span
    y0, y1 = y_span
    rx = (x1 - x0) / 2
    ry = (y1 - y0) / 2
    z = np.zeros(len(x))
    for i in range(len(x)):
        xi, yi = x[i], y[i]
        # Ensure points are within span
        xi = max(x0, min(x1, xi))
        yi = max(y0, min(y1, yi))
        
        # Quadrant logic for crossvault
        if yi <= y0 + (y1 - y0) / (x1 - x0) * (xi - x0) + tol and yi >= y1 - (y1 - y0) / (x1 - x0) * (xi - x0) - tol:  # Q1
            dx = abs(xi - (x0 + rx))
            z[i] = hc * math.sqrt(max(0, 1 - (dx/rx)**2))
        elif yi >= y0 + (y1 - y0) / (x1 - x0) * (xi - x0) - tol and yi >= y1 - (y1 - y0) / (x1 - x0) * (xi - x0) - tol:  # Q3
            dy = abs(yi - (y0 + ry))
            z[i] = hc * math.sqrt(max(0, 1 - (dy/ry)**2))
        elif yi >= y0 + (y1 - y0) / (x1 - x0) * (xi - x0) - tol and yi <= y1 - (y1 - y0) / (x1 - x0) * (xi - x0) + tol:  # Q2
            dx = abs(xi - (x0 + rx))
            z[i] = hc * math.sqrt(max(0, 1 - (dx/rx)**2))
        elif yi <= y0 + (y1 - y0) / (x1 - x0) * (xi - x0) + tol and yi <= y1 - (y1 - y0) / (x1 - x0) * (xi - x0) + tol:  # Q4
            dy = abs(yi - (y0 + ry))
            z[i] = hc * math.sqrt(max(0, 1 - (dy/ry)**2))
    return z
