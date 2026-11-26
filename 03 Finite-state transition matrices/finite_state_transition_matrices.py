"""Finite-state transition matrices

Compile a parity recogniser into one transition matrix for each input symbol.

Stanford CS103, finite automata and regular languages
https://web.stanford.edu/class/cs103/
WeightCode, weight-space programming preprint (2026)
https://arxiv.org/abs/2611.25476"""

import numpy as np
import matplotlib.pyplot as plt
from itertools import product


def main():
    T = np.array([np.eye(2), [[0, 1], [1, 0]]])
    start = np.array([1, 0])

    def states(word):
        result = [start]
        for bit in word:
            result.append(result[-1] @ T[bit])
        return np.stack(result)

    words = [w for n in range(9) for w in product([0, 1], repeat=n)]
    assert all(states(w)[-1].argmax() == sum(w) % 2 for w in words)
    word = [1, 1, 0, 1, 0, 0, 1, 0]
    trace = states(word)
    print("words tested", len(words), "state path", trace.argmax(1))
    fig, ax = plt.subplots(figsize=(7, 2.6), constrained_layout=True)
    ax.imshow(trace.T, aspect="auto", cmap="Blues")
    ax.set(
        xticks=range(len(trace)),
        xticklabels=["start", *map(str, word)],
        yticks=[0, 1],
        yticklabels=["even", "odd"],
        xlabel="Input token",
        title="Active state after each token",
    )
    plt.show()


if __name__ == "__main__":
    main()
