# Runnable examples

Run these commands from the repository root. The examples use NumPy.

## Fold batch norm

Fold inference-time batch normalisation into a preceding dense layer.

```sh
python examples/fold_batch_norm.py
```

## ReLU rescaling

Rescale ReLU hidden units while preserving the network's output.

```sh
python examples/relu_rescaling.py
```
