# Model soups with held-out selection

Average compatible checkpoints and compare uniform and greedy soups on unseen data.

Train ridge models with common features on several overlapping in-distribution samples and one shifted sample. Rank candidates by validation error. Accept an addition only when the running soup improves on validation. Evaluate the frozen selection on an independent test set.

[Worked notebook](model_soups_with_held_out_selection.ipynb) · [Python script](model_soups_with_held_out_selection.py)

## Construct the weights and test their behaviour

Inspect the arrays and the reference output. Each check describes a specific property of the constructed model.

## Recorded output

```text
greedy members [3] test MSE [0.367 0.426 0.367]
```

## Interpreting the result

In a linear model, averaging weights and averaging predictions coincide. For nonlinear networks this identity generally fails. Validation reuse can also overfit a small validation set.

![Model soups with held-out selection, worked result](figure.svg)

## Exercise

Repeat the experiment across seeds and compare the distribution of held-out differences.

## References

[Wortsman et al., Model soups (2022)](https://arxiv.org/abs/2203.05482)
