import math
import numpy as np

# ----------------------------------------
# 1. Shared Configuration
# ----------------------------------------

CONFIG = {
    'xy_span': [[0.0, 10.0], [0.0, 10.0]], # Square span for fan vault as per paper
    'thickness': 0.5,
    'max_rise': 2.5,
    'discretisation_level': 40,  # For envelope meshes (higher for smoother curves)
    'form_discretisation': 10,   # For FormDiagram (n_fans, n_hoops)
    'solver': 'IPOPT',           # Preferred solver
    'support_type': 'corners',   # 'corners' or 'perimeter'
    'vault_type': 'fan'          # 'cross' or 'fan'
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
        xi = max(x0, min(x1, xi))
        yi = max(y0, min(y1, yi))
        
        # Quadrant logic for crossvault
        if yi <= y0 + (y1 - y0) / (x1 - x0) * (xi - x0) + tol and yi >= y1 - (y1 - y0) / (x1 - x0) * (xi - x0) - tol:
            dx = abs(xi - (x0 + rx))
            z[i] = hc * math.sqrt(max(0, 1 - (dx/rx)**2))
        elif yi >= y0 + (y1 - y0) / (x1 - x0) * (xi - x0) - tol and yi >= y1 - (y1 - y0) / (x1 - x0) * (xi - x0) - tol:
            dy = abs(yi - (y0 + ry))
            z[i] = hc * math.sqrt(max(0, 1 - (dy/ry)**2))
        elif yi >= y0 + (y1 - y0) / (x1 - x0) * (xi - x0) - tol and yi <= y1 - (y1 - y0) / (x1 - x0) * (xi - x0) + tol:
            dx = abs(xi - (x0 + rx))
            z[i] = hc * math.sqrt(max(0, 1 - (dx/rx)**2))
        elif yi <= y0 + (y1 - y0) / (x1 - x0) * (xi - x0) + tol and yi <= y1 - (y1 - y0) / (x1 - x0) * (xi - x0) + tol:
            dy = abs(yi - (y0 + ry))
            z[i] = hc * math.sqrt(max(0, 1 - (dy/ry)**2))
    return z

def fanvault_middle_hc(x, y, x_span, y_span, hc):
    """
    Calculate the z-coordinate of the middle surface of a fan vault.
    The surface is a union of four surfaces of revolution (one at each corner).
    """
    x0, x1 = x_span
    y0, y1 = y_span
    xm = (x0 + x1) / 2
    ym = (y0 + y1) / 2
    
    # Distance from corner to center
    rmax = math.sqrt((xm - x0)**2 + (ym - y0)**2)
    
    # Circular profile radius
    R = (rmax**2 + hc**2) / (2 * hc)
    
    z = np.zeros(len(x))
    for i in range(len(x)):
        xi, yi = x[i], y[i]
        # Find nearest corner
        if xi <= xm:
            xc = x0
        else:
            xc = x1
        if yi <= ym:
            yc = y0
        else:
            yc = y1
            
        r = math.sqrt((xi - xc)**2 + (yi - yc)**2)
        # Profile: reaching hc at rmax
        # We ensure it doesn't exceed hc by clipping r
        r_eff = min(r, rmax)
        
        # z(r) = sqrt(R^2 - (rmax - r)^2) - (R - hc)
        z[i] = math.sqrt(max(0, R**2 - (rmax - r_eff)**2)) - (R - hc)
        
    return z
