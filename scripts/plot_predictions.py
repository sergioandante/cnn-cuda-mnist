"""Ejemplos cualitativos de predicciones sobre el conjunto de prueba de MNIST."""
import csv
import os
import struct

import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
PAPER = os.path.join(ROOT, "paper")


def read_mnist_image(idx_path, index):
    with open(idx_path, "rb") as f:
        magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
        assert magic == 2051
        f.seek(16 + index * rows * cols)
        buf = f.read(rows * cols)
        return np.frombuffer(buf, dtype=np.uint8).reshape(rows, cols).astype(np.float32) / 255.0


rows_data = []
with open(os.path.join(RES, "predictions_sample.csv"), newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        rows_data.append(row)

img_path = os.path.join(ROOT, "data", "t10k-images-idx3-ubyte")
n = len(rows_data)
ncols = 6
nrows = (n + ncols - 1) // ncols
fig, axes = plt.subplots(nrows, ncols, figsize=(2.0 * ncols, 2.2 * nrows))

n_errors = 0
for k, row in enumerate(rows_data):
    idx = int(row["test_index"])
    true_l = int(row["true_label"])
    pred_l = int(row["pred_label"])
    conf = float(row["confidence"])
    img = read_mnist_image(img_path, idx)
    ax = axes.flat[k]
    ax.imshow(img, cmap="gray")
    ax.set_xticks([]); ax.set_yticks([])
    ok = true_l == pred_l
    n_errors += 0 if ok else 1
    color = "black" if ok else "red"
    ax.set_title(f"real={true_l} pred={pred_l}\nconf={conf*100:.1f}%", fontsize=8, color=color)

for k in range(n, nrows * ncols):
    axes.flat[k].axis("off")

fig.suptitle("Predicciones cualitativas de la CNN sobre el conjunto de prueba de MNIST")
fig.tight_layout()
fig.savefig(os.path.join(PAPER, "cnn_predictions.png"), dpi=200)
print("Guardado cnn_predictions.png")
print(f"Errores en la muestra de {n}: {n_errors}")
