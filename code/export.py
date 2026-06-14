import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import io
import numpy as np

def export_plywood_layout_pdf(sheets_data, title="Cut Patterns", color='black'):
    """
    Exports the packed meshes from multiple sheets to a multi-page PDF.
    'sheets_data' is a list of dicts: [{'meshes': [...], 'w': ..., 'h': ...}, ...]
    """
    from matplotlib.backends.backend_pdf import PdfPages
    from code.packing import get_mesh_2d_bbox
    
    buf = io.BytesIO()
    with PdfPages(buf) as pdf:
        scale = 39.37 # Meters to Inches for 1:1 scale
        
        for sheet in sheets_data:
            meshes = sheet['meshes']
            sheet_w = sheet['w']
            sheet_h = sheet['h']
            
            fig, ax = plt.subplots(figsize=(sheet_w * scale, sheet_h * scale))
            
            # 1. Collect all edges
            all_lines = []
            
            # Add Sheet Boundary
            sheet_boundary = [
                (0, 0), (sheet_w, 0), (sheet_w, sheet_h), (0, sheet_h), (0, 0)
            ]
            for i in range(len(sheet_boundary) - 1):
                all_lines.append([sheet_boundary[i], sheet_boundary[i+1]])

            # Add Mesh Edges and Labels
            for mesh in meshes:
                for edge in mesh.edges():
                    p1, p2 = mesh.edge_coordinates(edge)
                    all_lines.append([(p1[0], p1[1]), (p2[0], p2[1])])
                
                # Add Label
                name = mesh.attributes.get('name', '')
                if name:
                    # Simple bbox center for label placement
                    min_x, min_y, max_x, max_y = get_mesh_2d_bbox(mesh)
                    ax.text((min_x + max_x)/2, (min_y + max_y)/2, name, 
                            color=color, ha='center', va='center', 
                            fontsize=8, fontweight='bold')
                    
            # 2. Draw lines
            lc = LineCollection(all_lines, colors=color, linewidths=4.0)
            ax.add_collection(lc)
            
            # 3. Setup axes
            ax.set_aspect('equal')
            ax.set_xlim(0, sheet_w)
            ax.set_ylim(0, sheet_h)
            ax.axis('off')
            
            plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
            plt.margins(0,0)
            ax.xaxis.set_major_locator(plt.NullLocator())
            ax.yaxis.set_major_locator(plt.NullLocator())

            pdf.savefig(fig, transparent=True, pad_inches=0)
            plt.close(fig)
            
    buf.seek(0)
    return buf
