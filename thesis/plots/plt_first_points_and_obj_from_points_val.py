import os.path
import pickle

import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

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

def load_display_obj(file_path):
    # Ein OBJ-File als Mesh laden
    mesh = o3d.io.read_triangle_mesh(file_path)
    mesh.compute_vertex_normals()
    mesh.scale(0.5, center=mesh.get_center())
    R = mesh.get_rotation_matrix_from_xyz((np.deg2rad(-45), np.deg2rad(0), np.deg2rad(0)))
    mesh.rotate(R, center=(0, 0, 0))

    # PNG speichern
    save_visualization_as_png(mesh, os.path.join("out", "04_" + os.path.basename(file_path) + ".png"))

# Load points_val
with open("input/points_val.pkl", "rb") as f:
    data = pickle.load(f)

# Extract the points from the first object
first_object = data[0]
points = first_object['points']

# Create a 3D plot
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# Plot the points
ax.scatter(points[:, 0], points[:, 1], points[:, 2])

# Setting the aspect ratio to be equal
max_range = np.array([points[:, 0].max()-points[:, 0].min(), points[:, 1].max()-points[:, 1].min(), points[:, 2].max()-points[:, 2].min()]).max() / 2.0
mid_x = (points[:, 0].max()+points[:, 0].min()) * 0.5
mid_y = (points[:, 1].max()+points[:, 1].min()) * 0.5
mid_z = (points[:, 2].max()+points[:, 2].min()) * 0.5
ax.set_xlim(mid_x - max_range, mid_x + max_range)
ax.set_ylim(mid_y - max_range, mid_y + max_range)
ax.set_zlim(mid_z - max_range, mid_z + max_range)

# Setting labels for axes
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

plt.savefig(os.path.join("out", "04_points_val_first_obj.pdf"))

# Show the plot
plt.show()


# Dateipfade der OBJ-Dateien
file_path = "input/model_normalized.obj"

# Funktion aufrufen
load_display_obj(file_path)