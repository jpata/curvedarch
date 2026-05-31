import matplotlib.pyplot as plt
import numpy as np

def create_structural_plot(data, config, elevation=30, azimuth=-135, title="Isometric"):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 1. Plot Surfaces (Intrados and Extrados)
    def plot_surface(mesh_data, color, alpha):
        verts = np.array(mesh_data['vertices'])
        faces = mesh_data['faces']
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
    
    # 2. Plot Thrust Networks
    def plot_network(net_data, color, label, lw=1.5, ls='-'):
        verts = np.array(net_data['vertices'])
        edges = net_data['edges']
        for i, edge in enumerate(edges):
            p1, p2 = verts[edge[0]], verts[edge[1]]
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], 
                    color=color, linewidth=lw, linestyle=ls, label=label if i == 0 else "")

    plot_network(data['thrust_max'], 'blue', 'Max Thrust', lw=1.2, ls='-')
    plot_network(data['thrust_min'], 'red', 'Min Thrust', lw=1.8, ls='--')
    
    # Axis settings
    x_span = config['xy_span'][0]
    y_span = config['xy_span'][1]
    hc = config['max_rise']
    
    ax.set_xlim(x_span)
    ax.set_ylim(y_span)
    ax.set_zlim([-0.5, hc + 1.0])

    # Ensure equal scaling
    dx = x_span[1] - x_span[0]
    dy = y_span[1] - y_span[0]
    dz = (hc + 1.0) - (-0.5)
    ax.set_box_aspect((dx, dy, dz))
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f"Vault Validation - {title}")
    ax.legend()
    
    ax.view_init(elev=elevation, azim=azimuth)
    
    return fig
