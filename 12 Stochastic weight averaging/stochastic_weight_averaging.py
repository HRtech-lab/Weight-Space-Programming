"""Stochastic weight averaging

Average late SGD checkpoints and measure train and test error at the averaged point.

Izmailov et al., Averaging Weights Leads to Wider Optima (2018)
https://arxiv.org/abs/1803.05407"""

import numpy as np
import matplotlib.pyplot as plt


def main():
    rng = np.random.default_rng(73)
    X = rng.normal(size=(140, 2))
    wstar = np.array([1.1, -0.7])
    y = X @ wstar + rng.normal(0, 0.6, len(X))
    T = rng.normal(size=(400, 2))
    yt = T @ wstar + rng.normal(0, 0.6, len(T))
    w = np.zeros(2)
    path = []
    for epoch in range(48):
        for group in np.array_split(rng.permutation(len(X)), 10):
            w -= 0.12 * (X[group].T @ (X[group] @ w - y[group])) / len(group)
        path.append(w.copy())
    path = np.array(path)
    avg = path[20:].mean(0)
    scores = [np.mean((T @ p - yt) ** 2) for p in [w, avg]]
    print("late checkpoints", len(path[20:]), "test MSE final and averaged", np.round(scores, 4))
    gridx, gridy = np.meshgrid(np.linspace(0.6, 1.4, 65), np.linspace(-1.1, -0.35, 65))
    grid = np.stack((gridx, gridy), axis=-1)
    loss = np.mean((np.einsum("ij,abj->abi", X, grid) - y) ** 2, axis=-1)
    fig, ax = plt.subplots(figsize=(5.6, 3.8), constrained_layout=True)
    ax.contour(gridx, gridy, loss, levels=15, colors="0.75")
    ax.plot(*path[2:].T, ".-", ms=3, label="SGD epochs")
    ax.scatter(*avg, marker="*", s=95, label="Average")
    ax.set(xlabel="First weight", ylabel="Second weight")
    ax.legend()
    plt.show()


if __name__ == "__main__":
    main()
