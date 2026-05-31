#!/usr/bin/env python3
import os
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from datetime import datetime
import logging
import traceback

from code.vault_logic import get_alternating_catenaries, generate_vault_meshes, compute_max_safe_cut_radius
from code.crossvault import run_tna_simulation
from code.vault_shared import CONFIG
from code.vault_plots import create_structural_plot

# -----------------------------------------------------------------------------
# Configuration & Test Cases
# -----------------------------------------------------------------------------

TEST_CASES = [
    {
        "name": "standard_10x10",
        "span_x": 10.0,
        "span_y": 10.0,
        "rise": 1.5,
        "thick": 0.2,
        "discr": 10,
        "corner_cut_ratio": 0.5  # ratio of max safe radius
    },
    {
        "name": "large_span_16x16",
        "span_x": 16.0,
        "span_y": 16.0,
        "rise": 2.5,
        "thick": 0.3,
        "discr": 16,
        "corner_cut_ratio": 0.3
    },
    {
        "name": "high_rise_narrow",
        "span_x": 8.0,
        "span_y": 12.0,
        "rise": 3.0,
        "thick": 0.2,
        "discr": 12,
        "corner_cut_ratio": 0.5
    },
    {
        "name": "thin_shell",
        "span_x": 10.0,
        "span_y": 10.0,
        "rise": 1.5,
        "thick": 0.05,
        "discr": 10,
        "corner_cut_ratio": 0.4
    }
]

# -----------------------------------------------------------------------------
# Plotting Helpers (Matplotlib)
# -----------------------------------------------------------------------------

def plot_geometry_matplotlib(meshes_3d, catenaries, config, output_path, title="Corrugated Geometry", elevation=30, azimuth=-135):
    """Render the 3D corrugated geometry using Matplotlib."""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 1. Plot Catenaries
    for cat in catenaries:
        pts = np.array(cat.points)
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], color='black', linewidth=0.8, alpha=0.4)
    
    # 2. Plot Strips (Meshes)
    for i, mesh in enumerate(meshes_3d):
        verts = np.array([mesh.vertex_coordinates(v) for v in mesh.vertices()])
        faces = [mesh.face_vertices(f) for f in mesh.faces()]
        triangles = []
        for face in faces:
            if len(face) == 4:
                triangles.append([face[0], face[1], face[2]])
                triangles.append([face[0], face[2], face[3]])
            else:
                triangles.append(face)
        
        # Alternating colors like in app.py
        color = 'lightblue' if i % 4 < 2 else 'orange'
        if triangles:
            ax.plot_trisurf(verts[:, 0], verts[:, 1], verts[:, 2], 
                           triangles=triangles, color=color, alpha=0.7, linewidth=0.1, edgecolor='black')

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
    ax.set_title(f"{title} - {elevation}/{azimuth}")
    
    ax.view_init(elev=elevation, azim=azimuth)
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

def plot_flat_patterns_matplotlib(meshes_flat, distortions, output_path, title="Flat Patterns"):
    """Render the unrolled flat strips in 2D."""
    fig, ax = plt.subplots(figsize=(12, 10))
    
    all_d = [d for sublist in distortions for d in sublist] if distortions else []
    max_d = max(all_d) if all_d else 0.01
    if max_d < 1e-6: max_d = 0.01

    for i, (mesh, dists) in enumerate(zip(meshes_flat, distortions)):
        if not mesh: continue
        verts = np.array([mesh.vertex_coordinates(v) for v in mesh.vertices()])
        faces = [mesh.face_vertices(f) for f in mesh.faces()]
        
        for f_idx, face in enumerate(faces):
            poly_verts = verts[face][:, :2] # Project to XY
            
            # Color by distortion relative to max_d
            d_val = dists[f_idx] if f_idx < len(dists) else 0
            # Use Viridis mapping
            color_intensity = min(1.0, d_val / max_d)
            facecolor = plt.cm.viridis(color_intensity)
            
            polygon = plt.Polygon(poly_verts, closed=True, fill=True, 
                                  facecolor=facecolor, edgecolor='black', linewidth=0.2, alpha=0.8)
            ax.add_patch(polygon)
            
    ax.set_aspect('equal')
    ax.autoscale_view()
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(f"{title} (Max Distortion: {max_d:.4f})")
    
    # Add a simple colorbar
    sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=mcolors.Normalize(vmin=0, vmax=max_d))
    plt.colorbar(sm, ax=ax, label='Distortion')

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


# -----------------------------------------------------------------------------
# Core Execution Logic
# -----------------------------------------------------------------------------

def run_headless_workflow():
    # Create main output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    root_output = os.path.join("outputs", f"run_{timestamp}")
    os.makedirs(root_output, exist_ok=True)
    
    # Global log
    log_file = os.path.join(root_output, "execution.log")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger()
    
    logger.info(f"Starting headless workflow. Output directory: {root_output}")
    
    results = []

    for case in TEST_CASES:
        name = case['name']
        logger.info(f"--- Processing Test Case: {name} ---")
        
        case_dir = os.path.join(root_output, name)
        os.makedirs(case_dir, exist_ok=True)
        
        # Prepare parameters
        span_x = case['span_x']
        span_y = case['span_y']
        config = {
            'xy_span': [[0.0, span_x], [0.0, span_y]],
            'thickness': case['thick'],
            'max_rise': case['rise'],
            'discretisation_level': 40,
            'form_discretisation': case['discr'],
            'solver': 'IPOPT',
            'support_type': 'corners',
            'vault_type': 'fan'
        }
        
        try:
            # 1. TNA Analysis
            logger.info(f"  Running TNA simulation...")
            sim_data = run_tna_simulation(config)
            
            # 2. Geometry Calculation
            logger.info(f"  Generating corrugated geometry...")
            center_coords = (span_x / 2.0, span_y / 2.0)
            
            # Determine safe cut radius
            full_cats = get_alternating_catenaries(
                sim_data['form_min'], 
                sim_data['form_max'], 
                corner_cut_radius=0.0,
                center_coords=center_coords
            )
            max_safe_radius = compute_max_safe_cut_radius(full_cats)
            corner_cut = max_safe_radius * case.get('corner_cut_ratio', 0.5)
            
            logger.info(f"  Max safe cut radius: {max_safe_radius:.4f}. Using: {corner_cut:.4f}")
            
            catenaries = get_alternating_catenaries(
                sim_data['form_min'], 
                sim_data['form_max'], 
                corner_cut,
                center_coords=center_coords
            )
            
            meshes_3d, meshes_flat, distortions = generate_vault_meshes(catenaries, flat_z_offset=-5.0)
            
            # 3. Statistics & Logging
            all_d = [item for sublist in distortions for item in sublist]
            avg_dist = np.mean(all_d) if all_d else 0
            max_dist = max(all_d) if all_d else 0
            
            stats = {
                "name": name,
                "converged": True,
                "num_strips": len(meshes_3d),
                "avg_distortion": avg_dist,
                "max_distortion": max_dist
            }
            results.append(stats)
            
            logger.info(f"  Results: Strips={stats['num_strips']}, AvgDist={avg_dist:.6f}, MaxDist={max_dist:.6f}")
            
            # Save stats to text file
            with open(os.path.join(case_dir, "stats.txt"), "w") as f:
                for k, v in case.items(): f.write(f"{k}: {v}\n")
                f.write("---\n")
                for k, v in stats.items(): f.write(f"{k}: {v}\n")

            # 4. Renderings
            logger.info(f"  Generating renderings...")
            
            # Structural Plots (4 views)
            views = [
                ("isometric", 30, -135),
                ("top", 90, -90),
                ("front", 0, -90),
                ("right", 0, 0)
            ]
            
            for v_name, elev, azim in views:
                # Structural
                fig_struct = create_structural_plot(sim_data, config, elevation=elev, azimuth=azim, title=v_name.capitalize())
                fig_struct.savefig(os.path.join(case_dir, f"structural_{v_name}.png"), dpi=150, bbox_inches='tight')
                plt.close(fig_struct)
                
                # Geometry
                plot_geometry_matplotlib(
                    meshes_3d, catenaries, config, 
                    os.path.join(case_dir, f"geometry_{v_name}.png"), 
                    title=f"Corrugated - {v_name.capitalize()}",
                    elevation=elev, azimuth=azim
                )
            
            # Flat Patterns
            plot_flat_patterns_matplotlib(
                meshes_flat, distortions, 
                os.path.join(case_dir, "flat_patterns.csv" if False else "flat_patterns.png")
            )
            
            logger.info(f"  Case {name} completed successfully.")

        except Exception as e:
            logger.error(f"  FAILED Case {name}: {str(e)}")
            logger.error(traceback.format_exc())
            results.append({"name": name, "converged": False, "error": str(e)})

    # Final Summary
    logger.info("--- WORKFLOW COMPLETE ---")
    summary_file = os.path.join(root_output, "summary.txt")
    with open(summary_file, "w") as f:
        f.write(f"Headless Run Summary - {timestamp}\n")
        f.write("="*40 + "\n")
        for res in results:
            status = "PASS" if res.get('converged') else "FAIL"
            f.write(f"[{status}] {res['name']}\n")
            if res.get('converged'):
                f.write(f"    Strips: {res['num_strips']}, Max Dist: {res['max_distortion']:.6f}\n")
            else:
                f.write(f"    Error: {res.get('error')}\n")
    
    logger.info(f"Summary saved to {summary_file}")

if __name__ == "__main__":
    run_headless_workflow()
