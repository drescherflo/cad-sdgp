import open3d as o3d
import numpy as np

def capture_image(window):
    image = window.capture_screen_float_buffer(False)
    return image

def save_visualization_as_png(geometry, filename):
    # Visualisierungseinstellungen
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False)  # Fenster nicht anzeigen
    vis.add_geometry(geometry)
    vis.poll_events()  # Verarbeite GUI-Events
    vis.update_renderer()  # Aktualisiere Renderer
    vis.capture_screen_image(filename)  # Bild speichern
    vis.destroy_window()  # Fenster schließen

def load_display_obj(file_path):
    # Ein OBJ-File als Mesh laden
    mesh = o3d.io.read_triangle_mesh(file_path)
    mesh.scale(0.5, center=mesh.get_center())
    R = mesh.get_rotation_matrix_from_xyz((-np.pi / 2, 0, 0))
    mesh.rotate(R, center=(0, 0, 0))
    mesh.compute_vertex_normals()

    # PNGs für jede Geometrieart speichern
    save_visualization_as_png(mesh, "out/cone_mesh.png")
    pointcloud = mesh.sample_points_poisson_disk(2000)
    save_visualization_as_png(pointcloud, "out/cone_pointcloud.png")
    voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh(mesh, voxel_size=0.05)
    save_visualization_as_png(voxel_grid, "out/cone_voxelgrid.png")

# Dateipfad der OBJ-Datei
file_path = 'input/cone.obj'

# Funktion aufrufen
load_display_obj(file_path)
