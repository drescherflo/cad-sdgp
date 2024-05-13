# Code wurde mit Unterstützung von ChatGPT erzeugt
import os.path

import open3d as o3d
import numpy as np

def save_visualization_as_png(geometry, filename):
    # Visualisierungseinstellungen
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False)  # Fenster nicht anzeigen
    vis.add_geometry(geometry)

    ctr = vis.get_view_control()
    ctr.set_zoom(0.9)  # Reduziere den Zoom, um mehr von der Szene zu sehen
    #ctr.set_front([1, 0, 0])  # Richtung der Kameraansicht

    vis.poll_events()  # Verarbeite GUI-Events
    vis.update_renderer()  # Aktualisiere Renderer
    vis.capture_screen_image(filename)  # Bild speichern
    vis.destroy_window()  # Fenster schließen

def load_display_obj(file_paths):
    for file_path in file_paths:
        # Ein OBJ-File als Mesh laden
        mesh = o3d.io.read_triangle_mesh(file_path)
        mesh.compute_vertex_normals()
        mesh.scale(0.5, center=mesh.get_center())
        R = mesh.get_rotation_matrix_from_xyz((-np.pi/4, 0, -np.pi/4))
        mesh.rotate(R, center=(0, 0, 0))

        # PNGs für jede Geometrieart speichern
        save_visualization_as_png(mesh, "out/08_" + os.path.basename(file_path) + ".png")

# Dateipfad der OBJ-Datei
file_paths = [
    "../../CAD Models/OBJ/MA Large Object.obj",
    "../../CAD Models/OBJ/MA Round Object With Holes.obj",
    "../../CAD Models/OBJ/MA Simple Object.obj",
    "../../CAD Models/OBJ/MA Simple Object higher.obj",
    "../../CAD Models/OBJ/MA Small Object.obj",
]

# Funktion aufrufen
load_display_obj(file_paths)
