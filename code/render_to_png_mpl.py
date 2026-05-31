import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from vault_shared import CONFIG

def plot_view(data, elevation, azimuth, filename, title):
    print(f"Rendering {title} view to {filename}...")
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # 1. Plot Surfaces (Intrados and Extrados)
    def plot_surface(mesh_data, color, alpha):
        verts = np.array(mesh_data['vertices'])
        faces = mesh_data['faces']
        # For simplicity in MPL, we plot as a scatter or wireframe if too complex
        # But we can try TriSurface if it's triangulated
        # COMPAS cross mesh uses quads, we split them into triangles
        triangles = []
        for face in faces:
            if len(face) == 3:
                triangles.append(face)
            elif len(face) == 4:
                triangles.append([face[0], face[1], face[2]])
                triangles.append([face[0], face[2], face[3]])
        
        if triangles:
            ax.plot_trisurf(verts[:, 0], verts[:, 1], verts[:, 2], 
                           triangles=triangles, color=color, alpha=alpha, linewidth=0)

    plot_surface(data['intrados'], 'cyan', 0.15)
    plot_surface(data['extrados'], 'tan', 0.15)
    
    # 2. Plot Thrust Networks (as lines)
    def plot_network(net_data, color, label, lw=1.5, ls='-'):
        verts = np.array(net_data['vertices'])
        edges = net_data['edges']
        for i, edge in enumerate(edges):
            p1, p2 = verts[edge[0]], verts[edge[1]]
            # Plot only first edge with label
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], 
                    color=color, linewidth=lw, linestyle=ls, label=label if i == 0 else "")

    # Draw Max (blue) first, then Min (red) on top with different style
    plot_network(data['thrust_max'], 'blue', 'Max Thrust', lw=1.2, ls='-')
    plot_network(data['thrust_min'], 'red', 'Min Thrust', lw=1.8, ls='--')
    
    # Axis settings
    x_span = CONFIG['xy_span'][0]
    y_span = CONFIG['xy_span'][1]
    hc = CONFIG['max_rise']
    
    ax.set_xlim(x_span)
    ax.set_ylim(y_span)
    ax.set_zlim([-0.5, hc + 1.0])
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f"Vault Validation - {title}")
    ax.legend()
    
    # Set orientation
    ax.view_init(elev=elevation, azim=azimuth)
    
    plt.savefig(filename, dpi=150)
    plt.close(fig)
    print(f"  Done.")

# Load Data
print("Loading data...")
try:
    with open('vault_model_data.json', 'r') as f:
        data = json.load(f)
except:
    print("Error: Run code/crossvault.py first.")
    exit(1)

# Generate views
# elev=90, azim=-90 is Top
# elev=0, azim=-90 is Front
# elev=0, azim=0 is Right
plot_view(data, 90, -90, "vault_mpl_top.png", "Top")
plot_view(data, 10, -110, "vault_mpl_front.png", "Front-ish")
plot_view(data, 10, -20, "vault_mpl_right.png", "Right-ish")
plot_view(data, 30, -135, "vault_mpl_iso.png", "Isometric")

print("\nMatplotlib PNG renders completed successfully.")
