import streamlit as st
import plotly.graph_objects as go
import numpy as np
import os
import math

from code.vault_logic import get_alternating_catenaries, generate_vault_meshes
from code.crossvault import run_tna_simulation
from code.vault_shared import CONFIG
from code.vault_plots import create_structural_plot

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
            tri_i.append(face[0]); tri_j.append(face[1]); tri_k.append(face[2])
            tri_i.append(face[0]); tri_j.append(face[2]); tri_k.append(face[3])
    
    return dict(
        type='mesh3d',
        x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
        i=tri_i, j=tri_j, k=tri_k,
        color=color, opacity=opacity, name=name, showlegend=True
    )

def mesh_to_plotly_with_distortion(mesh, distortions, name='Flat Strip', cmin=0.0, cmax=0.01, showscale=False):
    vertices = np.array([mesh.vertex_coordinates(v) for v in mesh.vertices()])
    faces = [mesh.face_vertices(f) for f in mesh.faces()]
    
    v_intensity = np.zeros(len(vertices))
    v_count = np.zeros(len(vertices))
    for f_idx, face in enumerate(faces):
        for v_idx in face:
            v_intensity[v_idx] += distortions[f_idx]
            v_count[v_idx] += 1
    v_intensity = v_intensity / np.maximum(v_count, 1)

    tri_i, tri_j, tri_k = [], [], []
    for face in faces:
        if len(face) == 3:
            tri_i.append(face[0]); tri_j.append(face[1]); tri_k.append(face[2])
        elif len(face) == 4:
            tri_i.append(face[0]); tri_j.append(face[1]); tri_k.append(face[2])
            tri_i.append(face[0]); tri_j.append(face[2]); tri_k.append(face[3])

    return dict(
        type='mesh3d',
        x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
        i=tri_i, j=tri_j, k=tri_k,
        intensity=v_intensity, 
        colorscale='Viridis',
        cmin=cmin,
        cmax=cmax,
        showscale=showscale,
        colorbar=dict(title='Distortion', x=1.1) if showscale else None,
        opacity=0.9, name=name, showlegend=True
    )

def main():
    st.set_page_config(page_title="Corrugated Vault Design", layout="wide")
    st.title("Corrugated Vault: Parametric Design & TNA Solver")
    
    st.sidebar.header("1. Vault Geometry")
    v_type = st.sidebar.selectbox("Vault Type", ["fan", "cross"], index=0 if CONFIG['vault_type'] == 'fan' else 1)
    span_x = st.sidebar.slider("Span X", 2.0, 20.0, 10.0, step=0.1)
    span_y = st.sidebar.slider("Span Y", 2.0, 20.0, 10.0, step=0.1)
    rise = st.sidebar.slider("Max Rise", 0.5, 5.0, 1.5)
    thick = st.sidebar.slider("Thickness", 0.05, 1.0, 0.2)
    
    st.sidebar.header("2. TNA Parameters")
    discr = st.sidebar.number_input("Form Discretisation", 5, 30, 10)
    solver = st.sidebar.selectbox("Solver", ["IPOPT", "SLSQP"], index=0)
    
    st.sidebar.header("3. Unrolling Parameters")
    flat_z = st.sidebar.slider("Flat Z-Offset", -30.0, 10.0, -1.0)
    corner_cut = st.sidebar.slider("Corner Cut Radius", 0.0, 3.0, 0.5, 0.1)
    
    st.sidebar.header("4. Visibility")
    show_3d = st.sidebar.checkbox("Show 3D Surface", value=True)
    show_flat = st.sidebar.checkbox("Show Flat Patterns", value=True)
    show_cats = st.sidebar.checkbox("Show Catenary Lines", value=True)
    show_pts = st.sidebar.checkbox("Show Vertex Points", value=True)
    show_intrados = st.sidebar.checkbox("Show Intrados (Envelope)", value=True)
    show_extrados = st.sidebar.checkbox("Show Extrados (Envelope)", value=True)
    
    # Session state to store simulation results
    if 'sim_data' not in st.session_state:
        st.session_state.sim_data = None
    if 'center_coords' not in st.session_state:
        st.session_state.center_coords = (5.0, 5.0)
    if 'current_config' not in st.session_state:
        st.session_state.current_config = CONFIG.copy()

    if st.sidebar.button("Execute TNA Analysis"):
        center_x = span_x / 2.0
        center_y = span_y / 2.0
        st.session_state.center_coords = (center_x, center_y)
        
        st.session_state.current_config = {
            'xy_span': [[0.0, span_x], [0.0, span_y]],
            'thickness': thick,
            'max_rise': rise,
            'discretisation_level': 40,
            'form_discretisation': discr,
            'solver': solver,
            'support_type': 'corners',
            'vault_type': v_type
        }
        with st.spinner("Solving Thrust Networks..."):
            try:
                data = run_tna_simulation(st.session_state.current_config)
                st.session_state.sim_data = data
                st.success("TNA Solver converged!")
            except Exception as e:
                st.error(f"Solver failed: {e}")

    # Main Tabs
    if st.session_state.sim_data:
        tab1, tab2 = st.tabs(["Corrugated Geometry", "Structural Validation"])
        
        with tab1:
            with st.spinner("Generating Corrugated Geometry..."):
                catenaries = get_alternating_catenaries(
                    st.session_state.sim_data['form_min'], 
                    st.session_state.sim_data['form_max'], 
                    corner_cut,
                    center_coords=st.session_state.center_coords
                )
                meshes_3d, meshes_flat, distortions = generate_vault_meshes(catenaries, flat_z)

            fig = go.Figure()

            # Add Catenaries
            if show_cats:
                for i, cat in enumerate(catenaries):
                    pts = np.array(cat.points)
                    fig.add_trace(go.Scatter3d(
                        x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
                        mode='lines',
                        line=dict(color='black', width=3),
                        name=f"Catenary {i}",
                        showlegend=False
                    ))

            # Add Points
            if show_pts:
                for i, cat in enumerate(catenaries):
                    pts = np.array(cat.points)
                    fig.add_trace(go.Scatter3d(
                        x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
                        mode='markers',
                        marker=dict(size=3, color='red'),
                        name=f"Points {i}",
                        showlegend=False
                    ))

            # Add 3D Surfaces
            if show_3d:
                for i, m in enumerate(meshes_3d):
                    color = 'rgb(100, 150, 240)' if i % 4 < 2 else 'rgb(240, 150, 100)'
                    fig.add_trace(go.Mesh3d(**mesh_to_plotly_dict(m, color=color, name=f"Strip {i}")))

            # Add Flat Patterns
            if show_flat:
                # Calculate global max distortion for uniform scale
                all_distortions = [d for sublist in distortions for d in sublist] if distortions else []
                global_max_d = max(all_distortions) if all_distortions else 0.01
                if global_max_d < 1e-6: global_max_d = 0.01
                
                # Use a rounded "reasonable" maximum for the colorbar
                # E.g., if max is 0.0138, maybe 0.015 or 0.02
                colorbar_max = math.ceil(global_max_d * 100) / 100.0
                if colorbar_max == 0: colorbar_max = 0.01

                for i, (m, d) in enumerate(zip(meshes_flat, distortions)):
                    if m:
                        # Only show scale for the first valid mesh
                        show_colorbar = (i == 0)
                        fig.add_trace(go.Mesh3d(**mesh_to_plotly_with_distortion(
                            m, d, 
                            name=f"Flat {i}", 
                            cmin=0.0, 
                            cmax=colorbar_max, 
                            showscale=show_colorbar
                        )))

            # Add Envelope Surfaces
            if show_intrados:
                # Convert list-based data to COMPAS Mesh for plotly helper
                from compas.datastructures import Mesh
                i_data = st.session_state.sim_data['intrados']
                i_mesh = Mesh.from_vertices_and_faces(i_data['vertices'], i_data['faces'])
                fig.add_trace(go.Mesh3d(**mesh_to_plotly_dict(i_mesh, color='cyan', opacity=0.15, name='Intrados')))

            if show_extrados:
                from compas.datastructures import Mesh
                e_data = st.session_state.sim_data['extrados']
                e_mesh = Mesh.from_vertices_and_faces(e_data['vertices'], e_data['faces'])
                fig.add_trace(go.Mesh3d(**mesh_to_plotly_dict(e_mesh, color='tan', opacity=0.15, name='Extrados')))

            fig.update_layout(
                scene=dict(aspectmode='data', xaxis_title='X', yaxis_title='Y', zaxis_title='Z'),
                margin=dict(l=0, r=0, b=0, t=40), height=800
            )
            st.plotly_chart(fig, width='stretch')
            
            st.subheader("Design Statistics")
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Strips", len(meshes_3d))
            all_d = [item for sublist in distortions for item in sublist]
            if all_d:
                col2.metric("Max Distortion", f"{max(all_d):.4f}")
                col3.metric("Avg Distortion", f"{np.mean(all_d):.4f}")
        
        with tab2:
            st.subheader("Intrados, Extrados & Thrust Networks")
            view_col1, view_col2 = st.columns(2)
            
            with view_col1:
                st.write("**Isometric View**")
                fig_iso = create_structural_plot(st.session_state.sim_data, st.session_state.current_config, title="Isometric")
                st.pyplot(fig_iso)
                
                st.write("**Top View**")
                fig_top = create_structural_plot(st.session_state.sim_data, st.session_state.current_config, elevation=90, azimuth=-90, title="Top")
                st.pyplot(fig_top)

            with view_col2:
                st.write("**Front View**")
                fig_front = create_structural_plot(st.session_state.sim_data, st.session_state.current_config, elevation=0, azimuth=-90, title="Front")
                st.pyplot(fig_front)
                
                st.write("**Right View**")
                fig_right = create_structural_plot(st.session_state.sim_data, st.session_state.current_config, elevation=0, azimuth=0, title="Right")
                st.pyplot(fig_right)
    else:
        st.info("Click 'Execute TNA Analysis' to generate geometry.")

if __name__ == "__main__":
    main()
