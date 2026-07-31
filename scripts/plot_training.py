"""Curvas de entrenamiento: perdida, exactitud y programacion de tasa de aprendizaje."""
import csv
import os

import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_PATH = os.path.join(ROOT, "results", "training_log.csv")
OUT_PATH = os.path.join(ROOT, "paper", "cnn_training.png")

epochs, train_loss, train_acc, test_loss, test_acc, lr = [], [], [], [], [], []
with open(LOG_PATH, newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        epochs.append(int(row["epoch"]))
        train_loss.append(float(row["train_loss"]))
        train_acc.append(float(row["train_acc"]) * 100.0)
        test_loss.append(float(row["test_loss"]))
        test_acc.append(float(row["test_acc"]) * 100.0)
        lr.append(float(row["lr"]))

plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.alpha": 0.3})

fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

ax = axes[0]
ax.plot(epochs, train_loss, label="Entrenamiento", color="#1f77b4", linewidth=2)
ax.plot(epochs, test_loss, label="Prueba", color="#d62728", linewidth=2)
ax.set_xlabel("Epoca")
ax.set_ylabel("Perdida (entropia cruzada)")
ax.set_title("Perdida")
ax.legend()

ax = axes[1]
ax.plot(epochs, train_acc, label="Entrenamiento", color="#1f77b4", linewidth=2)
ax.plot(epochs, test_acc, label="Prueba", color="#d62728", linewidth=2)
ax.set_xlabel("Epoca")
ax.set_ylabel("Exactitud (%)")
ax.set_title("Exactitud")
ax.legend(loc="lower right")

ax = axes[2]
ax.plot(epochs, lr, color="#2ca02c", linewidth=2)
ax.set_xlabel("Epoca")
ax.set_ylabel("Tasa de aprendizaje")
ax.set_title("Programacion de LR (warmup + coseno)")

fig.tight_layout()
fig.savefig(OUT_PATH, dpi=200)
print("Guardado:", OUT_PATH)
print("Mejor test_acc:", max(test_acc), "% en epoca", epochs[test_acc.index(max(test_acc))])
print("Ultima epoca: test_acc=%.4f%% test_loss=%.4f" % (test_acc[-1], test_loss[-1]))
