"""Task vector arithmetic

Combine parameter displacements measured from a common starting point.

Ilharco et al., Editing Models with Task Arithmetic (2022)
https://arxiv.org/abs/2212.04089"""

import numpy as np
import matplotlib.pyplot as plt


def main():
    base = np.array([0.5, -0.2])
    A = base + [1.5, 0]
    B = base + [0, -1.2]
    X = np.eye(2)
    truth = np.array([A[0], B[1]])
    models = np.stack([base, A, B, base + (A - base) + (B - base)])
    err = (models @ X.T - truth) ** 2
    print("squared errors for tasks A and B")
    print(err.round(3))
    assert np.allclose(err[-1], 0)
    same = np.array([1.0, 0.0])
    conflicting_targets = np.array([1.0, -1.0])
    best = conflicting_targets.mean()
    print(
        "best shared prediction for conflicting targets",
        best,
        "irreducible mean squared error",
        np.mean((best - conflicting_targets) ** 2),
    )
    fig, ax = plt.subplots(figsize=(5.8, 3.1), constrained_layout=True)
    im = ax.imshow(err, aspect="auto", cmap="Oranges")
    ax.set(
        xticks=[0, 1],
        xticklabels=["Task A", "Task B"],
        yticks=range(4),
        yticklabels=["Base", "A", "B", "Combined"],
    )
    fig.colorbar(im, ax=ax, label="Squared error")
    plt.show()


if __name__ == "__main__":
    main()
