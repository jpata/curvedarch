# Corrugated Vault Design Pipeline

This document maps out the workflow, scripts, and data formats exchanged to generate and flatten the final corrugated structural vault geometry. 

The pipeline relies on [COMPAS](https://compas.dev/) and [COMPAS TNA](https://blockresearchgroup.github.io/compas_tna/) (Thrust Network Analysis) to calculate the funicular limits of a vault envelope and then processes that network to create unrollable corrugated strips.

## 1. Form Finding & Thrust Network Analysis
**Script:** `code/crossvault.py`

This is the starting point of the pipeline. It defines a parametric vault envelope (cross or fan vault) and discretizes it into a TNA `FormDiagram`. The script runs two structural analysis passes using `compas_tno`:
1. **Minimum Thrust:** Finds the shallowest funicular network that fits within the envelope.
2. **Maximum Thrust:** Finds the steepest funicular network that fits within the envelope.

**Data Formats:**
- *Current Output:* `vault_model_data.json` - A unified JSON containing basic dictionary representations of `vertices` and `faces` for the intrados/extrados, and `vertices` and `edges` for the min/max wireframes.
- *Required Output for Next Steps:* `thrust_min.json` and `thrust_max.json` - These are native COMPAS serialized JSON files containing the complete `compas_tna.diagrams/FormDiagram` datastructures (including full vertex, edge, and face attributes). *Note: `code/curve_offset.py` strictly expects these serialized FormDiagrams, meaning `crossvault.py` historically exported them (or requires modification to re-export them).*

## 2. Spoke Extraction & Catenary Generation
**Script:** `code/curve_offset.py` (specifically `load_tna_catenaries()` and `extract_spokes()`)

Instead of using the full continuous network, the corrugated vault is built by turning the TNA grid into discrete radial ribs (spokes).
- **Process:** The script loads `thrust_min.json` and `thrust_max.json`. Starting from a targeted corner coordinate (e.g., `[0, 0]`), it walks the graph topology along the straightest possible paths to extract continuous "spokes" radiating outward.
- **Data Exchange:** It converts these connected vertex sequences into internal `compas.geometry.Polyline` objects. To create the corrugated fold pattern, it builds a new list of catenaries by strictly alternating between spokes from the `max` thrust diagram and the `min` thrust diagram.

## 3. 3D Corrugated Surface Generation
**Script:** `code/curve_offset.py` (specifically `regenerate_all_geometry()`)

With the alternating min/max polylines extracted, the script builds the 3D surface of the vault.
- **Process:** For every pair of adjacent polylines (one min, one max), it creates a continuous strip by bridging the corresponding segments. It also applies user-defined geometric modifications, such as trimming the tight convergence at the supports via a "Corner Cut Radius".
- **Data Exchange:** The generated 3D strips are instantiated as `compas.datastructures.Mesh` objects and added to the 3D viewing scene.

## 4. Unrolling & Flattening
**Script:** `code/curve_offset.py` (specifically `develop_strip_to_plane()`)

This is the core algorithm to flatten the doubly-curved/corrugated 3D strips into 2D manufacturing patterns.
- **Process:** The algorithm marches along the two 3D bounding polylines of a strip. For each segment quad, it uses 2D circle-circle intersections (`intersection_circle_circle_xy`) to rigidly unfold the local geometry onto the XY plane (placed at a specific `FLAT_Z_OFFSET`).
- **Distortion Tracking:** Because the spatial quads between the min and max polylines might not be perfectly developable (planar), the script calculates the discrepancy between the 3D diagonal length and the flattened 2D diagonal length.
- **Data Exchange:** The output is a flat 2D `compas.datastructures.Mesh` representing the cut pattern. Faces are color-coded in the UI viewer based on their accumulated quadrilateral distortion, helping the designer verify constructability.

---
## Interactive Visualization
A standalone interactive application is available to explore the vault geometry and unrolling parameters in real-time.

**Run the Streamlit App:**
```bash
uv run streamlit run app.py
```

This app allows you to:
- Adjust the **Flat Z-Offset** to position unrolled strips.
- Modify the **Corner Cut Radius** to trim convergence points.
- Toggle visibility of 3D surfaces and 2D patterns.
- Inspect **distortion metrics** via color-coded heatmaps on the flat patterns.
- **Structural Validation:** View the calculated Thrust Networks (Min/Max) alongside the vault envelope (Intrados/Extrados) from Top, Front, Right, and Isometric perspectives.

## Summary of Data Flow
1. **Mathematical Envelope** $\rightarrow$ `crossvault.py` $\rightarrow$ **COMPAS FormDiagrams** (`thrust_min.json`, `thrust_max.json`)
2. **FormDiagrams** $\rightarrow$ `curve_offset.py (extract_spokes)` $\rightarrow$ **Alternating Polylines** (Max/Min Catenaries)
3. **Alternating Polylines** $\rightarrow$ `curve_offset.py (regenerate)` $\rightarrow$ **3D Corrugated Meshes**
4. **3D Corrugated Meshes** $\rightarrow$ `curve_offset.py (unroll)` $\rightarrow$ **2D Flat Meshes with Distortion Metrics**