"""Generating weights with a hypernetwork

Generate a family of two-dimensional linear maps from an angle.

Ha, Dai and Le, HyperNetworks (2016)
https://arxiv.org/abs/1609.09106"""

import numpy as np
import matplotlib.pyplot as plt


def main():
    generator = np.array([[1.0, 0.0], [0.0, -1.0], [0.0, 1.0], [1.0, 0.0]])

    def weights(angle):
        return (generator @ np.array([np.cos(angle), np.sin(angle)])).reshape(2, 2)

    angles = np.linspace(0, 2 * np.pi, 25)
    vec = np.array([1.0, 0.3])
    path = np.stack([weights(t) @ vec for t in angles])
    err = max(np.linalg.norm(weights(t).T @ weights(t) - np.eye(2)) for t in angles)
    print(f"largest orthogonality error {err:.2e}")
    assert np.allclose(weights(0.7) @ weights(0.2), weights(0.9))
    fig, ax = plt.subplots(figsize=(4.6, 3.7), constrained_layout=True)
    ax.plot(*path.T, "o-", ms=3)
    ax.quiver(
        [0], [0], [vec[0]], [vec[1]], angles="xy", scale_units="xy", scale=1, color="darkorange"
    )
    ax.set(
        aspect="equal",
        xlabel="Output x",
        ylabel="Output y",
        title="A vector under generated rotation weights",
    )
    plt.show()


if __name__ == "__main__":
    main()
