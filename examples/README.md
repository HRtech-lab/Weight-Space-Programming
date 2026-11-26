# Runnable examples

Run these commands from the repository root. The examples use NumPy.

## Fold batch norm

At inference, batch normalisation uses fixed means and variances. The example folds that fixed transformation into the dense layer’s weights and bias, then compares both versions on the same 16 inputs. The printed maximum absolute difference should be close to zero, up to floating-point rounding.

```sh
python examples/fold_batch_norm.py
```

## ReLU rescaling

Rescale ReLU hidden units while preserving the network's output.

```sh
python examples/relu_rescaling.py
```
