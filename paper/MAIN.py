import matplotlib.pyplot as plt
import numpy as np

# ==========================
# Data from the LaTeX figure
# ==========================

epochs = np.array([1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55])

# Loss
train_loss = np.array([
    0.54, 0.44, 0.32, 0.23, 0.16, 0.13,
    0.11, 0.10, 0.09, 0.09, 0.08, 0.08
])

val_loss = np.array([
    0.59, 0.47, 0.35, 0.26, 0.18, 0.15,
    0.13, 0.12, 0.11, 0.10, 0.10, 0.10
])

# Accuracy
train_acc = np.array([
    84.5, 88.5, 91.2, 92.9, 93.9, 94.6,
    95.1, 95.4, 95.6, 95.7, 95.8, 95.8
])

val_acc = np.array([
    83.8, 87.6, 90.8, 92.5, 93.7, 94.4,
    94.9, 95.2, 95.5, 95.6, 95.7, 95.7
])

best_val_acc = 95.87

# Learning Rate Schedule
lr_epochs = np.array([1, 8, 15, 22, 30, 38, 45, 52, 55])

lr_values = np.array([
    0.05, 1.5, 3.8, 5.0,
    4.2, 2.5, 0.8, 0.05, 0.05
])

# ==========================
# Colors
# ==========================

train_color = "#1f77b4"      # blue
val_color = "#ff7f0e"        # orange
best_color = "#d62728"       # red
lr_color = "#17becf"         # teal

# ==========================
# Plot
# ==========================

fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))

# --------------------------
# (a) Loss
# --------------------------
ax = axes[0]
ax.plot(epochs, train_loss, color=train_color, linewidth=2, label="Train")
ax.plot(epochs, val_loss, color=val_color, linewidth=2, label="Val")

ax.set_title("(a) Loss", fontweight="bold")
ax.set_xlabel("Epoch")
ax.set_ylabel("Focal Loss")
ax.set_xlim(0, 55)
ax.set_ylim(0.06, 0.65)
ax.grid(True, alpha=0.3)

# --------------------------
# (b) Accuracy
# --------------------------
ax = axes[1]
ax.plot(epochs, train_acc, color=train_color, linewidth=2)
ax.plot(epochs, val_acc, color=val_color, linewidth=2)
ax.axhline(
    best_val_acc,
    color=best_color,
    linestyle="--",
    linewidth=2,
    label="Best Val"
)

ax.set_title("(b) Accuracy", fontweight="bold")
ax.set_xlabel("Epoch")
ax.set_ylabel("Accuracy (%)")
ax.set_xlim(0, 55)
ax.set_ylim(83, 97)
ax.grid(True, alpha=0.3)

# --------------------------
# (c) Learning Rate
# --------------------------
ax = axes[2]
ax.plot(
    lr_epochs,
    lr_values,
    color=lr_color,
    linewidth=2
)

ax.set_title("(c) LR", fontweight="bold")
ax.set_xlabel("Epoch")
ax.set_ylabel(r"LR ($\times 10^{-4}$)")
ax.set_xlim(0, 55)
ax.set_ylim(0, 5.8)
ax.grid(True, alpha=0.3)

# --------------------------
# Shared Legend
# --------------------------
handles = [
    plt.Line2D([0], [0], color=train_color, lw=2, label="Train"),
    plt.Line2D([0], [0], color=val_color, lw=2, label="Val"),
    plt.Line2D([0], [0], color=best_color, lw=2, ls="--", label="Best Val")
]

fig.legend(
    handles=handles,
    loc="lower center",
    ncol=3,
    frameon=False,
    bbox_to_anchor=(0.5, -0.03)
)

plt.tight_layout(rect=[0, 0.08, 1, 1])
plt.show()