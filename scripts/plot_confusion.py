"""Matriz de confusion sobre el conjunto de prueba completo de MNIST."""
import csv
import os

import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CM_PATH = os.path.join(ROOT, "results", "confusion_matrix.csv")
OUT_PATH = os.path.join(ROOT, "paper", "cnn_confusion.png")

cm = []
with open(CM_PATH, newline="") as f:
    for row in csv.reader(f):
        cm.append([int(v) for v in row if v != ""])
cm = np.array(cm)
total = cm.sum()
acc = np.trace(cm) / total

fig, ax = plt.subplots(figsize=(6.2, 5.5))
im = ax.imshow(cm, cmap="Blues")
ax.set_xticks(range(10))
ax.set_yticks(range(10))
ax.set_xlabel("Prediccion")
ax.set_ylabel("Etiqueta real")
ax.set_title(f"Matriz de confusion (N={total}, exactitud={acc*100:.2f}%)")

thresh = cm.max() / 2.0
for i in range(10):
    for j in range(10):
        val = cm[i, j]
        color = "white" if val > thresh else "black"
        ax.text(j, i, str(val), ha="center", va="center", color=color, fontsize=8)

fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
fig.tight_layout()
fig.savefig(OUT_PATH, dpi=200)
print("Guardado:", OUT_PATH)

# Reporta las confusiones fuera de diagonal mas frecuentes (para discusion en el paper).
offdiag = [(cm[i, j], i, j) for i in range(10) for j in range(10) if i != j and cm[i, j] > 0]
offdiag.sort(reverse=True)
print("Top confusiones (real -> prediccion : conteo):")
for count, i, j in offdiag[:8]:
    print(f"  {i} -> {j} : {count}")
