---
title: Curvedarch
emoji: 🏗️
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
---

# Corrugated Vault Design Pipeline

This document maps out the workflow, scripts, and data formats exchanged to generate and flatten the final corrugated structural vault geometry.

## Background: Unfold Form Construction System
This computational pipeline supports the research and structural validation of the **Unfold Form Construction System**, as detailed in *"Scalability of the Unfold Form Construction System" (Scheder, Van Mele, Block, IASS 2025)*. 

The Unfold Form system aims to reduce the embodied carbon of conventional concrete floor slabs by creating funicular, unreinforced concrete shells. It uses a reusable, flat-packed formwork made of bending-active plywood plates joined by curved textile hinges. This pipeline mathematically models the scaling of these double-curved, corrugated geometries up to architectural spans (e.g., 6.0 m), optimizing variables such as:
- **Corrugation Density:** Balancing *Coarse* corrugations (which provide deeper structural static height for stability) against *Dense* corrugations.
- **Plate Thickness Scaling:** Using cubic-root scaling laws ($\mu = \sqrt[3]{\lambda^2 \cdot \eta}$) to balance necessary structural stiffness against the manual actuation force required to deploy the formwork on-site.

## 1. Form Finding & Thrust Network Analysis
**Script:** `code/crossvault.py`

This is the starting point of the pipeline. It defines a parametric vault envelope (cross or fan vault) and discretizes it into a TNA `FormDiagram`. The script runs two structural analysis passes using `compas_tno`:
1. **Minimum Thrust:** Finds the shallowest funicular network that fits within the envelope.
2. **Maximum Thrust:** Finds the steepest funicular network that fits within the envelope.

## 2. Spoke Extraction & Catenary Generation
**Script:** `code/vault_logic.py` (specifically `get_alternating_catenaries()`)

Instead of using the full continuous network, the corrugated vault is built by turning the TNA grid into discrete radial ribs (spokes).
- **Process:** The script processes the `form_min` and `form_max` diagrams. Starting from the center, it walks the graph topology to extract continuous "spokes" radiating outward.
- **Data Exchange:** It builds a new list of catenaries (polylines) by strictly alternating between spokes from the `max` thrust diagram and the `min` thrust diagram. A user-defined `corner_cut_radius` can be applied to trim the tight convergence at the supports.

## 3. 3D Corrugated Surface Generation
**Script:** `code/vault_logic.py` (specifically `generate_vault_meshes()`)

With the alternating min/max polylines extracted, the script builds the 3D surface of the vault.
- **Process:** For every pair of adjacent polylines (one min, one max), it creates a continuous strip by bridging the corresponding segments.
- **Data Exchange:** The generated 3D strips are instantiated as `compas.datastructures.Mesh` objects and are ready for 3D visualization.

## 4. Unrolling & Flattening
**Script:** `code/vault_logic.py` (specifically handled within `generate_vault_meshes()`)

This core algorithm flattens the doubly-curved/corrugated 3D strips into 2D manufacturing patterns.
- **Process:** Rigidly unfolds the local 3D spatial quad geometry onto the XY plane (placed at a specific `FLAT_Z_OFFSET`).
- **Distortion Tracking:** Because the spatial quads between the min and max polylines might not be perfectly developable (planar), the script calculates the discrepancy (distortion metric) between the 3D diagonal length and the flattened 2D diagonal length.
- **Data Exchange:** Outputs flat 2D `compas.datastructures.Mesh` objects representing the cut patterns, alongside associated distortion data for constructability verification.

---
## Interactive Visualization
A standalone interactive application is available to explore the vault geometry and unrolling parameters in real-time.

**Run the Streamlit App:**
```bash
uv run streamlit run app.py
```

This app allows you to:
- Adjust Vault Geometry (Spans, Rise, Thickness) and TNA Parameters.
- Adjust the **Flat Z-Offset** and dynamically compute the maximum safe **Corner Cut Radius**.
- Toggle visibility of 3D corrugated surfaces, catenary lines, and 2D unrolled patterns.
- Inspect **distortion metrics** via color-coded heatmaps on the flat patterns using Plotly.
- View the calculated Thrust Networks (Min/Max) alongside the vault envelope from multiple perspectives.

---
## Automated Headless Batch Processing
**Script:** `run_headless.py`

For automated structural validation and scaling tests across multiple geometries, a headless workflow is provided. It executes a suite of predefined architectural test cases (e.g., `standard_10x10`, `large_span_16x16`, `high_rise_narrow`, `thin_shell`), simulating different scaling challenges mentioned in the IASS 2025 paper.

**Run the Headless Script:**
```bash
python run_headless.py
```

**Features & Outputs:**
- Calculates structural asymmetry and maximum developability distortions.
- Generates high-quality automated Matplotlib renderings of 3D geometry, structural top/front/isometric views (via `code/vault_plots.py`), and 2D flat patterns.
- Outputs are saved in timestamped directories under `outputs/run_{timestamp}/` alongside detailed execution logs (`execution.log`) and a final summary (`summary.txt`).

## Summary of Data Flow
1. **Mathematical Envelope** $\rightarrow$ `crossvault.py` $\rightarrow$ **COMPAS FormDiagrams** (`form_min`, `form_max`)
2. **FormDiagrams** $\rightarrow$ `vault_logic.py (get_alternating_catenaries)` $\rightarrow$ **Alternating Polylines** (Max/Min Catenaries)
3. **Alternating Polylines** $\rightarrow$ `vault_logic.py (generate_vault_meshes)` $\rightarrow$ **3D Corrugated Meshes**
4. **3D Corrugated Meshes** $\rightarrow$ `vault_logic.py (generate_vault_meshes)` $\rightarrow$ **2D Flat Meshes with Distortion Metrics**