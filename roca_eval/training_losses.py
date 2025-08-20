# Created with help by ChatGPT

import os
import re
import matplotlib.pyplot as plt

# Pfad zum übergeordneten Ordner
root_directory = '/Volumes/NAS/Benutzer/Flo/Masterarbeit/custom_dataset/ROCA_Outputs'

# Initialisierung für den Plot
plt.figure()# figsize=(10, 6))

# Muster für das Parsen der Daten
pattern = re.compile(r"iter:\s+(\d+)\s+total_loss:\s+(\d+\.\d+)")

# Sammeln aller Unterverzeichnisse mit 'terminal_log.txt'
subdirs_with_logs = []
for subdir, _, files in os.walk(root_directory):
    if 'terminal_log.txt' in files:
        subdirs_with_logs.append(subdir)

# Alphabetisches Sortieren der Verzeichnisse
subdirs_with_logs.sort()

# Durchlaufen der sortierten Unterverzeichnisse
for subdir in subdirs_with_logs:
    file_path = os.path.join(subdir, 'terminal_log.txt')
    iter_values = []
    total_loss_values = []

    # Auslesen der terminal_log.txt
    with open(file_path, 'r') as f:
        for line in f:
            match = pattern.search(line)
            if match:
                iter_values.append(int(match.group(1)))
                total_loss_values.append(float(match.group(2)))

    # Extrahieren des Verzeichnisnamens für das Label
    directory_label = os.path.basename(subdir)

    # Hinzufügen der Daten zum Plot
    plt.plot(iter_values, total_loss_values, label=directory_label.removesuffix("_Augmentation"))

# Plot formatieren
plt.xlabel('Iteration')
plt.ylabel('Fehler')
plt.ylim(top=10)
plt.legend()
plt.title('Fehlerverlauf von ROCA während des Trainings')
plt.grid(True)
plt.tight_layout()
plt.savefig("output/training_losses.pdf")
plt.show()
