import math
import time
import functools

from PySide6 import QtCore, QtWidgets
from PySide6.QtWidgets import QWidget, QVBoxLayout, QGroupBox, QLabel, QDoubleSpinBox, QPushButton, QHBoxLayout, QSpacerItem, QSizePolicy

from compas.geometry import Point, Polyline, Translation, Rotation, Vector, Line
from compas.geometry import intersection_circle_circle_xy
from compas.colors import Color
from compas.datastructures import Mesh
from compas_viewer import Viewer
from compas_tna.diagrams import FormDiagram

# --- Configuration ---
FLAT_Z_OFFSET = -15.0 
CORNER_CUT_RADIUS = 0.5
TNA_CATENARIES = []
RAW_TNA_CATENARIES = []

# --- Global variables for scene management ---
viewer = None 
generated_polylines_3d = []
scene_objects_3d_surfaces = [] 
scene_objects_flat_strips = [] 
is_updating = False

# --- TNA Extraction Helpers ---

from code.vault_logic import (
    extract_spokes, 
    cut_polyline_at_radius, 
    develop_strip_to_plane, 
    get_alternating_catenaries, 
    generate_vault_meshes
)

# --- Configuration ---
FLAT_Z_OFFSET = -15.0 
CORNER_CUT_RADIUS = 0.5
TNA_CATENARIES = []

# --- Global variables for scene management ---
viewer = None 
scene_objects_3d_surfaces = [] 
scene_objects_flat_strips = [] 
is_updating = False

def load_tna_catenaries():
    global TNA_CATENARIES
    try:
        TNA_CATENARIES = get_alternating_catenaries('thrust_min.json', 'thrust_max.json', CORNER_CUT_RADIUS)
        print(f"Loaded {len(TNA_CATENARIES)} TNA catenaries sorted by angle.")
        regenerate_all_geometry()
    except Exception as e:
        print(f"Error loading TNA data: {e}")

def regenerate_all_geometry():
    global viewer, scene_objects_3d_surfaces, scene_objects_flat_strips
    global FLAT_Z_OFFSET, is_updating, TNA_CATENARIES

    if is_updating or viewer is None or viewer.scene is None:
        return
    is_updating = True
    
    try:
        meshes_3d, meshes_flat, distortions = generate_vault_meshes(TNA_CATENARIES, FLAT_Z_OFFSET)
        num_expected_strips = len(meshes_3d)
        
        # --- Update or Create 3D Surface Strips ---
        new_scene_objs_3d = [None] * num_expected_strips 
        for i, mesh_3d in enumerate(meshes_3d):
            try:
                if i < len(scene_objects_3d_surfaces) and scene_objects_3d_surfaces[i] is not None:
                    scene_obj = scene_objects_3d_surfaces[i]
                    # Update existing mesh
                    scene_obj.item.clear()
                    vkey_map = {}
                    for v in mesh_3d.vertices():
                        vkey_map[v] = scene_obj.item.add_vertex(x=mesh_3d.vertex_attribute(v, 'x'), 
                                                               y=mesh_3d.vertex_attribute(v, 'y'), 
                                                               z=mesh_3d.vertex_attribute(v, 'z'))
                    for f in mesh_3d.faces():
                        scene_obj.item.add_face([vkey_map[v] for v in mesh_3d.face_vertices(f)])
                    scene_obj.update(update_data=True)
                    new_scene_objs_3d[i] = scene_obj
                    scene_objects_3d_surfaces[i] = None
                else:
                    new_scene_objs_3d[i] = viewer.scene.add(mesh_3d, name=f"3DSurface_{i}",
                                                           facecolor=Color(0.6, 0.7, 0.9, 0.7), show_edges=True)
            except Exception as e: print(f"Error 3D strip {i}: {e}")

        for old_obj in scene_objects_3d_surfaces:
            if old_obj: viewer.scene.remove(old_obj)
        scene_objects_3d_surfaces[:] = new_scene_objs_3d

        # --- Update or Create Flattened Strips ---
        new_scene_objs_flat = [None] * num_expected_strips
        for i, (mesh_flat, quad_distortions) in enumerate(zip(meshes_flat, distortions)):
            if not mesh_flat: continue
            try:
                face_colors = {}
                for f_idx, fkey in enumerate(mesh_flat.faces()):
                    norm_d = min(1.0, quad_distortions[f_idx] * 10)
                    r, g = (2 * norm_d, 1.0) if norm_d <= 0.5 else (1.0, 2 * (1.0 - norm_d))
                    face_colors[fkey] = Color(r, g, 0.0, 0.8)

                if i < len(scene_objects_flat_strips) and scene_objects_flat_strips[i] is not None:
                    scene_obj = scene_objects_flat_strips[i]
                    scene_obj.item.clear()
                    vkey_map = {}
                    for v in mesh_flat.vertices():
                        vkey_map[v] = scene_obj.item.add_vertex(x=mesh_flat.vertex_attribute(v, 'x'), 
                                                               y=mesh_flat.vertex_attribute(v, 'y'), 
                                                               z=mesh_flat.vertex_attribute(v, 'z'))
                    for f in mesh_flat.faces():
                        scene_obj.item.add_face([vkey_map[v] for v in mesh_flat.face_vertices(f)])
                    scene_obj.facecolor = face_colors
                    scene_obj.update(update_data=True)
                    new_scene_objs_flat[i] = scene_obj
                    scene_objects_flat_strips[i] = None
                else:
                    new_scene_objs_flat[i] = viewer.scene.add(mesh_flat, name=f"FlatStrip_{i}", 
                                                           facecolor=face_colors, show_edges=True)
            except Exception as e: print(f"Error flat strip {i}: {e}")

        for old_obj in scene_objects_flat_strips:
            if old_obj: viewer.scene.remove(old_obj)
        scene_objects_flat_strips[:] = new_scene_objs_flat
        
        viewer.renderer.update() 
    except Exception as e: print(f"Critical error: {e}")
    finally: is_updating = False


def setup_ui():
    global viewer, FLAT_Z_OFFSET, CORNER_CUT_RADIUS
    from compas_viewer.components import Component
    ui_component = Component()
    main_widget = QWidget()
    main_layout = QVBoxLayout(main_widget)
    main_layout.setAlignment(QtCore.Qt.AlignTop)

    group = QGroupBox("TNA Vault Unrolling")
    layout = QVBoxLayout(group)
    
    z_layout = QHBoxLayout()
    z_spinbox = QDoubleSpinBox()
    z_spinbox.setRange(-100.0, 100.0)
    z_spinbox.setValue(FLAT_Z_OFFSET)
    z_spinbox.valueChanged.connect(functools.partial(update_flat_z_offset_param, z_spinbox))
    z_layout.addWidget(QLabel("Flat Z-Offset:"))
    z_layout.addWidget(z_spinbox)
    layout.addLayout(z_layout)
    
    cut_layout = QHBoxLayout()
    cut_spinbox = QDoubleSpinBox()
    cut_spinbox.setRange(0.0, 5.0)
    cut_spinbox.setSingleStep(0.1)
    cut_spinbox.setValue(CORNER_CUT_RADIUS)
    cut_spinbox.valueChanged.connect(functools.partial(update_corner_cut_radius_param, cut_spinbox))
    cut_layout.addWidget(QLabel("Corner Cut Radius:"))
    cut_layout.addWidget(cut_spinbox)
    layout.addLayout(cut_layout)
    
    btn = QPushButton("Load TNA Data")
    btn.clicked.connect(load_tna_catenaries)
    layout.addWidget(btn)
    
    main_layout.addWidget(group)
    ui_component.widget = main_widget
    viewer.ui.sidedock.add(ui_component)

if __name__ == '__main__':
    viewer = Viewer(width=1600, height=900)
    setup_ui() 
    load_tna_catenaries() # Load automatically on start if possible
    viewer.ui.sidedock.show = True
    viewer.show()
