# Code wurde mit Unterstützung von ChatGPT erzeugt

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def rodrigues_rotation_matrix(axis, theta):
    """
    Erstellt eine Rotationsmatrix, die einen Vektor um die gegebene Achse um den Winkel theta rotiert.
    Die Rotation folgt der Rodrigues-Rotationsformel.
    """
    theta = np.radians(theta)
    axis = axis / np.sqrt(np.dot(axis, axis))
    a = np.cos(theta / 2)
    b, c, d = axis * np.sin(theta / 2)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])


# Ursprüngliche Achsen (parent frame)
axes = np.identity(3)

# Rotationsachse und Winkel
rotation_axis = np.array([1/3, 2/3, 2/3])
rotation_angle = 30  # Grad

# Rotationsmatrix berechnen
rotation_matrix = rodrigues_rotation_matrix(rotation_axis, rotation_angle)

# Gedrehte Achsen (child frame)
rotated_axes = np.dot(rotation_matrix, axes)

# Plot vorbereiten
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Funktion, um eine Achse mit Beschriftung zu zeichnen
def draw_axis(ax, direction, color, label):
    ax.quiver(0, 0, 0, direction[0], direction[1], direction[2], color=color, length=1.0, arrow_length_ratio=0.1)
    ax.text(direction[0]*1.1, direction[1]*1.1, direction[2]*1.1, label, color=color)

# Ursprüngliche Achsen zeichnen mit Beschriftungen
draw_axis(ax, axes[0], 'b', 'X_parent')
draw_axis(ax, axes[1], 'b', 'Y_parent')
draw_axis(ax, axes[2], 'b', 'Z_parent')

# Gedrehte Achsen zeichnen mit Beschriftungen
draw_axis(ax, rotated_axes[0], 'g', 'X_child')
draw_axis(ax, rotated_axes[1], 'g', 'Y_child')
draw_axis(ax, rotated_axes[2], 'g', 'Z_child')

# Rotationsachse zeichnen
draw_axis(ax, rotation_axis, 'k', 'Rotation Axis')

# Plot anpassen
ax.set_xlim([-0.5, 1])
ax.set_ylim([-0.5, 1])
ax.set_zlim([-0.5, 1])
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.view_init(elev=20., azim=30)

# Plot anzeigen
plt.show()
