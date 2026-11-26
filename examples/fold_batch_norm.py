"""Fold inference-time batch normalisation into a preceding dense layer."""
import numpy as np


def fold_batch_norm(weight, bias, mean, variance, gamma, beta, epsilon=1e-5):
    # Dense weights are laid out as (output features, input features).
    scale = gamma / np.sqrt(variance + epsilon)
    return scale[:, None] * weight, scale * (bias - mean) + beta


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    x, weight = rng.normal(size=(16, 3)), rng.normal(size=(4, 3))
    bias, mean, gamma, beta = rng.normal(size=(4, 4))
    variance = np.array([0.2, 0.4, 1.5, 2.0])
    dense = x @ weight.T + bias
    reference = gamma * (dense - mean) / np.sqrt(variance + 1e-5) + beta
    new_weight, new_bias = fold_batch_norm(weight, bias, mean, variance, gamma, beta)
    folded = x @ new_weight.T + new_bias
    np.testing.assert_allclose(folded, reference, atol=1e-12)
    print(f"Maximum absolute difference: {np.max(np.abs(folded-reference)):.2e}")
