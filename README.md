# Curved Arch Design via Curved-Crease Unfolding (CCU)

This repository implements a parametric design and simulation workflow for **Unfold Form**, a construction system for corrugated concrete shells using reusable, flat-packed formwork. The system is based on the principles of **Curved-Crease Unfolding (CCU)** and **Thrust Network Analysis (TNA)**.

## Methodology

The project follows an integrative co-design workflow that bridges structural geometry, bending-active kinematics, and fabrication logic.

### 1. Funicular Form Finding
The design begins with the definition of a funicular target geometry—a vault shape that primarily carries loads in compression. Using **Thrust Network Analysis (TNA)**, the vault is optimized to ensure that the thrust lines remain within the structural envelope, enabling the construction of unreinforced concrete shells.

### 2. Curved-Crease Corrugation
The smooth funicular surface is translated into a corrugated geometry. This is achieved through the **reflection method**, where planar strips are joined along curved creases. The corrugations provide:
- **Shape Control:** The crease geometry dictates the final 3D curvature of the bending-active plates.
- **Geometric Stiffness:** The corrugations increase the structural depth of the formwork and result in stiffening ribs in the final concrete shell.

### 3. Bending-Active Kinematics & Developability
The formwork consists of thin plywood plates joined by flexible textile hinges.
- **Active Bending:** The plates are elastically deformed from their initial flat state into their 3D shape during deployment.
- **Optimization for Developability:** To ensure the 3D shape can be flattened into 2D strips without material distortion, the geometry is optimized using an energy minimization approach. This minimizes the difference between the 3D geodesic distances and the 2D planar distances.

### 4. Fabrication & Deployment
The system is designed for "flat-packed" transport. On-site, the assembly is unfolded into a self-supporting formwork, ready for in-situ concrete casting.

## Code Structure

The repository contains several Python scripts that handle different stages of the design and simulation process:

- **`code/crossvault.py`**: 
  - Performs the initial TNA form-finding using `compas_tna`.
  - Generates min/max thrust envelopes for a vault geometry.
  - Exports quadrant geometry to OBJ and JSON for further processing.

- **`code/curve_offset.py`**:
  - Implements the "unrolling" or development of 3D strips into 2D planar shapes.
  - Calculates and visualizes geometric distortion (strain) across the developed strips.
  - Provides a UI for adjusting the flat-state offsets.

- **`code/energy_minimization.py`**:
  - Uses **PyTorch** to perform gradient-based optimization of the system's geometry.
  - Minimizes a multi-objective loss function including: ruling line lengths (strip width), outer edge segment lengths, crease segment lengths, and quad planarity (flatness).
  - Ensures the 3D folded state is as close to a developable surface as possible.

- **`code/folding_sim.py`**:
  - A kinematic simulation of the folding and unfolding process.
  - Visualizes the transition from a flat-packed state to the final 3D vaulted configuration using elastically bent plates.

- **`code/render_vault.py`**:
  - Utility for high-quality rendering or visualization of the final vault geometry.

## Dependencies

- **COMPAS Framework:** `compas`, `compas_viewer`
- **Structural Analysis:** `compas_tna`, `compas_tno`
- **Optimization:** `torch` (PyTorch)
- **UI/Graphics:** `PySide6`, `OpenGL`

## Papers
For more technical details, refer to the papers in the `papers/` directory:
- *Scalability of the Unfold Form Construction System* (IASS 2025)
- *Unfold Form: A curved-crease unfoldable formwork for a corrugated fan-vaulted floor* (IASS 2024)
