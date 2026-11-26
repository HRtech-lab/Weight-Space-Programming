# TIES merging

Trace trimming, sign election and disjoint averaging through a small merge.

Work with checkpoint differences relative to a common base. Keep the largest magnitudes in each task vector, elect the sign by the sum of retained values at each coordinate and average the entries agreeing with that sign. This example specifies a zero update for a tied vote.

[Worked notebook](ties_merging.ipynb) · [Python script](ties_merging.py)

## Construct the weights and test their behaviour

Inspect the arrays and the reference output. Each check describes a specific property of the constructed model.

## Recorded output

```text
elected signs [ 1  1  0 -1 -1] merged [ 1.6   0.95  0.   -1.05 -1.9 ]
```

## Interpreting the result

A high-magnitude conflicting value can affect the election even when it is outvoted by count. The rule uses a signed sum.

![TIES merging, worked result](figure.svg)

## Exercise

Change the trimming fraction and report which coordinates switch sign.

## References

[Yadav et al., TIES-Merging (2023)](https://arxiv.org/abs/2306.01708)
