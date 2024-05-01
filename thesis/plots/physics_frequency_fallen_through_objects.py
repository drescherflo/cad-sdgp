# Skript erstellt mit Hilfe von ChatGPT

import os
import re
import matplotlib.pyplot as plt

# Der Pfad zum Ordner, der die Textdateien enthält
folder_path = 'input/Physik-Frequenz-Auswahl'

# Regex, um die benötigten Daten zu extrahieren
filename_pattern = re.compile(r"phys_(\d+)hz_small_object_glitched_through_belt_terminal_log\.txt")
warning_pattern = re.compile(r"Warning! (\d+) are under the conveyor belt.")

# Listen für die Daten
frequencies = []
object_counts = []

# Durchlaufe alle Dateien im angegebenen Ordner
for filename in os.listdir(folder_path):
    if filename.endswith(".txt"):
        freq_match = filename_pattern.match(filename)
        if freq_match:
            frequency = int(freq_match.group(1))
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r') as file:
                for line in file:
                    warning_match = warning_pattern.search(line)
                    if warning_match:
                        object_count = int(warning_match.group(1))
                        frequencies.append(frequency)
                        object_counts.append(object_count)
                        break

# Nach Frequenz sortieren
frequencies_and_object_counts = list(zip(frequencies, object_counts))
frequencies_and_object_counts.sort()
frequencies, object_counts = zip(*frequencies_and_object_counts)

# Erstellen des Liniendiagramms
plt.figure()#(figsize=(10, 6))
plt.plot(frequencies, object_counts, marker='o', linestyle='-')
plt.title('Anzahl der Objekte unter dem Förderband nach Physik-Frequenz')
plt.xlabel('Frequenz [Hz]')
plt.ylabel('Anzahl Objekte')
plt.xticks(frequencies)#, labels=[f"{freq} Hz" for freq in frequencies])
plt.ylim(bottom=0)
plt.grid(True)
plt.tight_layout()
plt.savefig("out/04_physics_frequency_fallen_through_objects.pdf")
plt.show()
