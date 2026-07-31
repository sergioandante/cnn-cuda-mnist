"""Filtros de conv1 y mapas de activacion (conv1/conv2) para una imagen de muestra."""
import csv
import os
import struct

import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
PAPER = os.path.join(ROOT, "paper")


def read_floats_csv(path):
    with open(path, newline="") as f:
        row = next(csv.reader(f))
    return np.array([float(v) for v in row if v != ""], dtype=np.float32)


def read_mnist_image(idx_path, index):
    with open(idx_path, "rb") as f:
        magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
        assert magic == 2051
        f.seek(16 + index * rows * cols)
        buf = f.read(rows * cols)
        return np.frombuffer(buf, dtype=np.uint8).reshape(rows, cols).astype(np.float32) / 255.0


# --- Filtros de conv1 (16 filtros de 1x3x3) --------------------------------
filters = read_floats_csv(os.path.join(RES, "conv1_filters.csv")).reshape(16, 3, 3)
fig, axes = plt.subplots(4, 4, figsize=(6, 6))
vmax = np.abs(filters).max()
for i, ax in enumerate(axes.flat):
    ax.imshow(filters[i], cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"F{i}", fontsize=8)
fig.suptitle("Filtros aprendidos de la primera capa convolucional (conv1)")
fig.tight_layout()
fig.savefig(os.path.join(PAPER, "cnn_filters.png"), dpi=200)
print("Guardado cnn_filters.png")

# --- Mapas de activacion para una imagen de muestra ------------------------
with open(os.path.join(RES, "featuremap_sample_meta.csv")) as f:
    reader = csv.DictReader(f)
    meta = next(reader)
test_index = int(meta["test_index"])
true_label = int(meta["true_label"])

img = read_mnist_image(os.path.join(ROOT, "data", "t10k-images-idx3-ubyte"), test_index)
act1 = read_floats_csv(os.path.join(RES, "conv1_activations.csv")).reshape(16, 28, 28)
act2 = read_floats_csv(os.path.join(RES, "conv2_activations.csv")).reshape(32, 14, 14)

fig = plt.figure(figsize=(13, 7))
gs = fig.add_gridspec(4, 16, height_ratios=[1.2, 1, 1, 1])
ax_img = fig.add_subplot(gs[0, 0:2])
ax_img.imshow(img, cmap="gray")
ax_img.set_title(f"Entrada (digito {true_label})", fontsize=9)
ax_img.set_xticks([]); ax_img.set_yticks([])

for i in range(16):
    ax = fig.add_subplot(gs[1, i])
    ax.imshow(act1[i], cmap="viridis")
    ax.set_xticks([]); ax.set_yticks([])
fig.text(0.02, 0.52, "conv1 (16 canales, 28x28)", fontsize=9, va="center")

for i in range(32):
    r = 2 + i // 16
    c = i % 16
    ax = fig.add_subplot(gs[r, c])
    ax.imshow(act2[i], cmap="viridis")
    ax.set_xticks([]); ax.set_yticks([])
fig.text(0.02, 0.20, "conv2 (32 canales, 14x14)", fontsize=9, va="center")

fig.suptitle("Mapas de activacion tras conv1 y conv2 para una imagen de prueba")
fig.savefig(os.path.join(PAPER, "cnn_featuremaps.png"), dpi=200, bbox_inches="tight")
print("Guardado cnn_featuremaps.png")
