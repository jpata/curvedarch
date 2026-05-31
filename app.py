import streamlit as st
import plotly.graph_objects as go
import numpy as np
import os

from code.vault_logic import get_alternating_catenaries, generate_vault_meshes

def mesh_to_plotly_dict(mesh, color='lightblue', opacity=0.8, name='Mesh'):
    # Extract vertices and faces for Plotly Mesh3d
    vertices = np.array([mesh.vertex_coordinates(v) for v in mesh.vertices()])
    faces = [mesh.face_vertices(f) for f in mesh.faces()]
    
    # Triangulate quads
    tri_i, tri_j, tri_k = [], [], []
    for face in faces:
        if len(face) == 3:
            tri_i.append(face[0]); tri_j.append(face[1]); tri_k.append(face[2])
        elif len(face) == 4:
            # Triangle 1
            tri_i.append(face[0]); tri_j.append(face[1]); tri_k.append(face[2])
            # Triangle 2
            tri_i.append(face[0]); tri_j.append(face[2]); tri_k.append(face[3])
    
    return dict(
        type='mesh3d',
        x=vertices[:, 0],
        y=vertices[:, 1],
        z=vertices[:, 2],
        i=tri_i,
        j=tri_j,
        k=tri_k,
        color=color,
        opacity=opacity,
        name=name,
        showlegend=True
    )

def mesh_to_plotly_with_distortion(mesh, distortions, name='Flat Strip'):
    # distortions is a list of floats for each face
    vertices = np.array([mesh.vertex_coordinates(v) for v in mesh.vertices()])
    faces = [mesh.face_vertices(f) for f in mesh.faces()]
    
    # Map distortions to colors
    max_d = max(distortions) if distortions else 0.01
    if max_d < 1e-6: max_d = 0.01
    
    # Plotly Mesh3d doesn't support per-face colors easily.
    # We use intensity per vertex by averaging face distortions at vertices.
    v_intensity = np.zeros(len(vertices))
    v_count = np.zeros(len(vertices))
    for f_idx, face in enumerate(faces):
        for v_idx in face:
            v_intensity[v_idx] += distortions[f_idx]
            v_count[v_idx] += 1
    v_intensity = v_intensity / np.maximum(v_count, 1)

    # Triangulate quads
    tri_i, tri_j, tri_k = [], [], []
    for face in faces:
        if len(face) == 3:
            tri_i.append(face[0]); tri_j.append(face[1]); tri_k.append(face[2])
        elif len(face) == 4:
            tri_i.append(face[0]); tri_j.append(face[1]); tri_k.append(face[2])
            tri_i.append(face[0]); tri_j.append(face[2]); tri_k.append(face[3])

    return dict(
        type='mesh3d',
        x=vertices[:, 0],
        y=vertices[:, 1],
        z=vertices[:, 2],
        i=tri_i,
        j=tri_j,
        k=tri_k,
        intensity=v_intensity,
        colorscale='Viridis',
        opacity=0.9,
        name=name,
        showlegend=True
    )

def main():
    st.set_page_config(page_title="Corrugated Vault Unrolling", layout="wide")
    st.title("Corrugated Vault Design & Unrolling")
    
    st.sidebar.header("Parameters")
    
    # Check for data files
    if not os.path.exists('thrust_min.json') or not os.path.exists('thrust_max.json'):
        st.error("Error: 'thrust_min.json' or 'thrust_max.json' not found. Please run the TNA solver first.")
        st.info("You can run 'uv run python code/crossvault.py' to generate the data.")
        return

    flat_z = st.sidebar.slider("Flat Z-Offset", -30.0, 10.0, -15.0)
    corner_cut = st.sidebar.slider("Corner Cut Radius", 0.0, 3.0, 0.5, 0.1)
    
    show_3d = st.sidebar.checkbox("Show 3D Surface", value=True)
    show_flat = st.sidebar.checkbox("Show Flat Patterns", value=True)

    with st.spinner("Processing Geometry..."):
        # 1. Load and extract catenaries
        catenaries = get_alternating_catenaries('thrust_min.json', 'thrust_max.json', corner_cut)
        
        # 2. Generate meshes
        meshes_3d, meshes_flat, distortions = generate_vault_meshes(catenaries, flat_z)

    # 3. Create Plotly Figure
    fig = go.Figure()

    if show_3d:
        for i, m in enumerate(meshes_3d):
            color = 'rgb(100, 150, 240)' if i % 2 == 0 else 'rgb(240, 150, 100)'
            fig.add_trace(go.Mesh3d(**mesh_to_plotly_dict(m, color=color, name=f"Strip {i}")))

    if show_flat:
        for i, (m, d) in enumerate(zip(meshes_flat, distortions)):
            if m:
                fig.add_trace(go.Mesh3d(**mesh_to_plotly_with_distortion(m, d, name=f"Flat {i}")))

    fig.update_layout(
        scene=dict(
            aspectmode='data',
            xaxis=dict(title='X'),
            yaxis=dict(title='Y'),
            zaxis=dict(title='Z'),
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        height=800,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )

    st.plotly_chart(fig, width='stretch')
    
    # Display some stats
    st.subheader("Design Statistics")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Strips", len(meshes_3d))
    
    all_d = [item for sublist in distortions for item in sublist]
    if all_d:
        col2.metric("Max Distortion", f"{max(all_d):.4f}")
        col3.metric("Avg Distortion", f"{np.mean(all_d):.4f}")

if __name__ == "__main__":
    main()
