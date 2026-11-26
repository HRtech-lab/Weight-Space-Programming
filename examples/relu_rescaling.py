"""Rescale ReLU hidden units while preserving the network's output."""
import numpy as np


def predict(x, incoming, bias, outgoing):
    return np.maximum(x @ incoming + bias, 0) @ outgoing


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    x = rng.normal(size=(32, 3))
    incoming = rng.normal(size=(3, 4))
    bias = rng.normal(size=4)
    outgoing = rng.normal(size=(4, 2))
    scales = np.array([0.25, 0.5, 2.0, 4.0])
    # ReLU(s*x) = s*ReLU(x) for positive s. Undo the scale at the output.
    reference = predict(x, incoming, bias, outgoing)
    rescaled = predict(x, incoming * scales, bias * scales, outgoing / scales[:, None])
    np.testing.assert_allclose(rescaled, reference, atol=1e-12)
    print(f"Maximum absolute difference: {np.max(np.abs(rescaled-reference)):.2e}")
