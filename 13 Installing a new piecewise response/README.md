# Installing a new piecewise response

Add a specified response to an existing fitted checkpoint by writing one hidden unit and its outgoing weight.

Suppose the original model should keep its response up to x = 0.4 and then acquire an extra slope of 0.85. The added function is one hinge, 0.85 × max(0, x − 0.4). Append one weight, bias and readout coefficient. Copy the original units exactly. This small weight-space program has an explicit installation target and a region in which its effect must be zero.

[Worked notebook](installing_a_new_piecewise_response.ipynb) · [Python script](installing_a_new_piecewise_response.py)

## Fit a baseline, write the patch and reload both checkpoints

Only the baseline fit uses labelled data. The installation program reads the saved weights and writes the new checkpoint. Its only additional inputs are the requested threshold and slope.

## Measure installation error and off-region drift

Draw independent inputs on both sides and include the threshold itself. Compare the change with the piecewise specification, separately checking the unaffected region.

## Recorded output

```text
Hidden units before and after 4 5
New unit 1.0 -0.4 0.85
Change RMSE 9.57e-17 | maximum off-region drift 0
```

## Interpreting the result

The output displacement matches the specification to floating-point precision. Architectural expansion was allowed in this example. Installing the same change in a network of fixed width would require a spare unit or another representation. The construction is a small independent example of direct weight programming.

![Installing a new piecewise response, worked result](figure-2.svg)

![Installing a new piecewise response, worked result](figure.svg)

## Exercise

Install a bounded triangular response using three hinges. Verify equality before the first knot and after the last one.

## References

[WeightCode, installing capabilities through weight-space programming](https://arxiv.org/abs/2611.25476)

[Unser, a representer theorem for deep neural networks](https://www.jmlr.org/papers/v20/18-418.html)
