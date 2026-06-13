import streamlit as st
import plotly.graph_objects as go
import numpy as np
import os
import math

from code.vault_logic import get_alternating_catenaries, generate_vault_meshes, compute_max_safe_cut_radius, generate_envelope_catenaries
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
    # v_type = st.sidebar.selectbox("Vault Type", ["fan", "cross"], index=0 if CONFIG['vault_type'] == 'fan' else 1)
    v_type = "fan"
    st.sidebar.info("Vault Type: Fan (Cross vault is currently disabled)")
    span_x = st.sidebar.slider("Span X", 1.0, 20.0, 3.0, step=0.1, help="Total dimension of the vault along the X-axis (m).")
    span_y = st.sidebar.slider("Span Y", 1.0, 20.0, 1.8, step=0.1, help="Total dimension of the vault along the Y-axis (m).")
    rise = st.sidebar.slider("Max Rise", 0.1, 5.0, 0.4, help="The maximum vertical height of the middle surface at the crown.")
    thick = st.sidebar.slider("Thickness", 0.01, 1.0, 0.1, help="The structural thickness of the vault, defining the distance between intrados and extrados.")
    
    st.sidebar.header("2. TNA Parameters")
    if v_type == "fan":
        discr_x = st.sidebar.slider("Ribs X", 2, 30, 8, help="Number of radial ribs along the X boundary for TNA discretization.")
        discr_y = st.sidebar.slider("Ribs Y", 2, 30, 6, help="Number of radial ribs along the Y boundary for TNA discretization.")
        discr = st.sidebar.number_input("Hoop Discretisation", 5, 40, 10, help="Number of concentric 'hoop' segments from support to crown.")
    else:
        discr = st.sidebar.number_input("Form Discretisation", 5, 40, 12, help="Grid resolution for the cross-vault TNA diagram.")
        discr_x, discr_y = discr, discr
        
    solver = st.sidebar.selectbox("Solver", ["IPOPT", "SLSQP"], index=0, help="Numerical optimization engine used to solve for horizontal thrust.")
    
    st.sidebar.header("3. Unrolling Parameters")
    flat_z = -1
    
    n_catenaries = st.sidebar.slider("Number of Catenaries", 2, 60, 14, help="Total number of corrugation spokes. Increasing this makes the corrugations denser.")
    
    corner_cut = st.sidebar.slider(
        "Corner Cut Radius", 
        0.0, 
        0.2, 
        0.05,
        help="Trims the tight convergence of spokes at the supports. Essential for avoiding geometric singularities."
    )
    
    st.sidebar.header("4. Visibility")
    show_3d = st.sidebar.checkbox("Show 3D Surface", value=True, help="Toggle visibility of the corrugated 3D strips.")
    show_flat = st.sidebar.checkbox("Show Flat Patterns", value=True, help="Toggle visibility of the 2D unrolled manufacturing patterns.")
    show_cats = st.sidebar.checkbox("Show Catenary Lines", value=True, help="Visualize the skeleton polylines used to build the corrugated surface.")
    show_pts = st.sidebar.checkbox("Show Vertex Points", value=True, help="Visualize the discrete vertices along the catenaries.")
    show_intrados = st.sidebar.checkbox("Show Intrados (Envelope)", value=True, help="Visualize the lower boundary surface of the vault.")
    show_extrados = st.sidebar.checkbox("Show Extrados (Envelope)", value=True, help="Visualize the upper boundary surface of the vault.")
    
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
            'form_discretisation_x': discr_x,
            'form_discretisation_y': discr_y,
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

    # Construct reactive config for geometry generation
    active_config = {
        'xy_span': [[0.0, span_x], [0.0, span_y]],
        'thickness': thick,
        'max_rise': rise,
        'vault_type': v_type
    }

    # Visualization
    tab1, tab2 = st.tabs(["Corrugated Geometry", "Structural Validation"])

    with tab1:
        try:
            with st.spinner("Generating Corrugated Geometry..."):
                quadrant_catenaries = generate_envelope_catenaries(
                    active_config,
                    n_spokes=n_catenaries,
                    n_points=discr + 1,
                    corner_cut_radius=corner_cut
                )
                
                meshes_3d, meshes_flat, distortions = [], [], []
                all_cats_flat = []
                for quad_cats in quadrant_catenaries:
                    m3d, mflat, dists = generate_vault_meshes(quad_cats, flat_z)
                    meshes_3d.extend(m3d)
                    meshes_flat.extend(mflat)
                    distortions.extend(dists)
                    all_cats_flat.extend(quad_cats)
                    
        except ValueError as ve:
            st.error(f"Geometry Generation Error: {ve}")
            st.info("Try reducing the **Corner Cut Radius** to ensure all spokes retain the same number of nodes.")
            return

        fig = go.Figure()

        # Add Catenaries
        if show_cats:
            for i, cat in enumerate(all_cats_flat):
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
            for i, cat in enumerate(all_cats_flat):
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
            all_distortions = [d for sublist in distortions for d in sublist] if distortions else []
            global_max_d = max(all_distortions) if all_distortions else 0.01
            if global_max_d < 1e-6: global_max_d = 0.01
            colorbar_max = math.ceil(global_max_d * 100) / 100.0
            if colorbar_max == 0: colorbar_max = 0.01

            # Only show strips for the first quadrant (n_catenaries - 1 strips)
            strips_per_quad = n_catenaries - 1
            for i, (m, d) in enumerate(zip(meshes_flat[:strips_per_quad], distortions[:strips_per_quad])):
                if m:
                    show_colorbar = (i == 0)
                    fig.add_trace(go.Mesh3d(**mesh_to_plotly_with_distortion(
                        m, d, 
                        name=f"Flat {i}", 
                        cmin=0.0, 
                        cmax=colorbar_max, 
                        showscale=show_colorbar
                    )))


        # Add Envelope Surfaces (Always compute them reactively)
        from compas.datastructures import Mesh
        from code.vault_shared import fanvault_middle_hc, crossvault_middle_hc
        
        # Simple reactive envelope generation
        def get_envelope_mesh(config, side='middle'):
            x_span = config['xy_span'][0]
            y_span = config['xy_span'][1]
            hc = config['max_rise']
            t = config['thickness']
            n = 30 # Resolution for envelope
            
            from compas_tna.diagrams.diagram_rectangular import create_cross_mesh
            m = create_cross_mesh(x_span=x_span, y_span=y_span, n=n)
            for v in m.vertices():
                x, y = m.vertex_attributes(v, names=["x", "y"])
                if config['vault_type'] == 'fan':
                    z = fanvault_middle_hc([x], [y], x_span, y_span, hc)[0]
                else:
                    z = crossvault_middle_hc([x], [y], x_span, y_span, hc)[0]
                
                if side == 'intrados': z -= t/2
                elif side == 'extrados': z += t/2
                m.vertex_attribute(v, "z", z)
            return m

        if show_intrados:
            i_mesh = get_envelope_mesh(active_config, side='intrados')
            fig.add_trace(go.Mesh3d(**mesh_to_plotly_dict(i_mesh, color='cyan', opacity=0.25, name='Intrados')))

        if show_extrados:
            e_mesh = get_envelope_mesh(active_config, side='extrados')
            fig.add_trace(go.Mesh3d(**mesh_to_plotly_dict(e_mesh, color='tan', opacity=0.25, name='Extrados')))

        fig.update_layout(
            scene=dict(aspectmode='data', xaxis_title='X', yaxis_title='Y', zaxis_title='Z'),
            margin=dict(l=0, r=0, b=0, t=40), height=800
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Design Statistics")
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Strips", len(meshes_3d))
        all_d = [item for sublist in distortions for item in sublist]
        if all_d:
            col2.metric("Max Distortion", f"{max(all_d):.4f}")
            col3.metric("Avg Distortion", f"{np.mean(all_d):.4f}")
    
    with tab2:
        if st.session_state.sim_data:
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
            st.info("Click 'Execute TNA Analysis' to generate structural validation data.")

if __name__ == "__main__":
    main()
