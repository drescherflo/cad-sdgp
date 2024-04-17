# Code wurde mit Unterstützung von ChatGPT erzeugt

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
# import tikzplotlib

#mpl.use("pgf")

# Schriftgrößen anpassen
#plt.rc('text', usetex=True)
#plt.rc('font', size=14)  # Grundlegende Schriftgröße
#plt.rc('axes', titlesize=24)     # Schriftgröße für Titel
#plt.rc('axes', labelsize=20)     # Schriftgröße für Achsenbeschriftungen
#plt.rc('xtick', labelsize=16)    # Schriftgröße für die X-Achsen-Tick-Markierungen
#plt.rc('ytick', labelsize=16)    # Schriftgröße für die Y-Achsen-Tick-Markierungen
#plt.rc('legend', fontsize=16)    # Schriftgröße für Legenden

# Definition der Aktivierungsfunktionen
def identity(x):
    return x

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

def leaky_relu(x, alpha=0.1):
    return np.where(x > 0, x, x * alpha)

# Erzeugen eines Arrays von Werten für die Visualisierung
x = np.linspace(-10, 10, 1000)

# Erstellen von Plot-Figuren für jede Aktivierungsfunktion
fig, axs = plt.subplots(2, 2, dpi=300, figsize=(13, 6))
fig.suptitle("Aktivierungsfunktionen")

# Identitätsfunktion
axs[0, 0].plot(x, identity(x), label=r"$f(x) = x$")
axs[0, 0].set_title("Identität")
axs[0, 0].spines['left'].set_position('zero')
axs[0, 0].spines['left'].set_color('gray')
axs[0, 0].spines['bottom'].set_position('zero')
axs[0, 0].spines['bottom'].set_color('gray')
axs[0, 0].spines['right'].set_color('none')
axs[0, 0].spines['top'].set_color('none')
axs[0, 0].xaxis.set_major_locator(MaxNLocator(integer=True))
axs[0, 0].yaxis.set_major_locator(MaxNLocator(integer=True))
axs[0, 0].legend()

# Sigmoidfunktion
axs[0, 1].plot(x, sigmoid(x), label=r"$f(x) = \frac{1}{1 + e^{-x}}$", color="orange")
axs[0, 1].set_title("Sigmoid")
axs[0, 1].spines['left'].set_position('zero')
axs[0, 1].spines['left'].set_color('gray')
axs[0, 1].spines['bottom'].set_position('zero')
axs[0, 1].spines['bottom'].set_color('gray')
axs[0, 1].spines['right'].set_color('none')
axs[0, 1].spines['top'].set_color('none')
axs[0, 1].xaxis.set_major_locator(MaxNLocator(integer=True))
axs[0, 1].yaxis.set_major_locator(MaxNLocator(integer=True))
axs[0, 1].legend()

# ReLU
axs[1, 0].plot(x, relu(x), label=r"$f(x) = \max(0, x)$", color="green")
axs[1, 0].set_title("ReLU")
axs[1, 0].spines['left'].set_position('zero')
axs[1, 0].spines['left'].set_color('gray')
axs[1, 0].spines['bottom'].set_position('zero')
axs[1, 0].spines['bottom'].set_color('gray')
axs[1, 0].spines['right'].set_color('none')
axs[1, 0].spines['top'].set_color('none')
axs[1, 0].xaxis.set_major_locator(MaxNLocator(integer=True))
axs[1, 0].yaxis.set_major_locator(MaxNLocator(integer=True))
axs[1, 0].legend()

# Leaky ReLU
axs[1, 1].plot(x, leaky_relu(x), label=r"$f(x) = \max(\alpha x, x)$", color="red")
axs[1, 1].set_title("Leaky ReLU")
axs[1, 1].spines['left'].set_position('zero')
axs[1, 1].spines['left'].set_color('gray')
axs[1, 1].spines['bottom'].set_position('zero')
axs[1, 1].spines['bottom'].set_color('gray')
axs[1, 1].spines['right'].set_color('none')
axs[1, 1].spines['top'].set_color('none')
axs[1, 1].xaxis.set_major_locator(MaxNLocator(integer=True))
axs[1, 1].yaxis.set_major_locator(MaxNLocator(integer=True))
axs[1, 1].legend()

# Anpassen des Layouts
plt.tight_layout()

# Speichern der Figur als PDF
plt.savefig("out/02_Aktivierungsfunktionen.pdf")
#plt.savefig("02_Aktivierungsfunktionen.pgf")

plt.show()
