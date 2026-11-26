# Weight Space Programming

A weight matrix can be written down from a specification. For example, subtract the inputs in both directions, rectify the differences, then add the results. The resulting two-unit network computes XOR on binary inputs and absolute difference over the whole plane. A truth-table check covers the first claim. The second follows from the algebra.

This course starts with direct parameter construction, moves on to parameter generation and constrained edits, and compares established methods for aligning or merging checkpoints. The final workshop opens a fitted checkpoint, installs a specified new response and measures its effect on independent inputs. The latter methods are closely related tools for reasoning about parameter space. Each lecture includes runnable Python, an executed notebook, an exercise and a figure from the recorded run. [WeightCode preprint](https://arxiv.org/abs/2611.25476) · [Tracr](https://arxiv.org/abs/2301.05062)

## From a function to its parameters

Suppose a continuous one-dimensional function has known values at a sequence of knots, and is straight between them. An affine skip connection sets the initial line. At each internal knot a ReLU switches on, contributing the difference between the slopes on either side. The width of the hidden layer is the number of internal knots. Moving a knot changes a bias. Changing the slope to its right changes an outgoing weight. The compiled function can be tested on a dense grid because an independent linear interpolator gives a reference response.

![Compiling a piecewise linear function](02%20Compiling%20a%20piecewise%20linear%20function/figure.svg)

A causal attention head gives a different example. Zero query and key matrices make every allowed attention logit equal. A causal mask sets the future logits to negative infinity. Softmax therefore assigns weight 1/t to each of the t visible tokens. An identity value projection returns the feature-wise prefix mean. The class includes the mask, normalisation and projection in its test because the projection matrices alone do not specify the computation. [Attention Is All You Need](https://arxiv.org/abs/1706.03762) · [Thinking Like Transformers](https://proceedings.mlr.press/v139/weiss21a.html)

## Editing an existing layer

An update to a matrix acts on every input, even when it was derived for a single key. If a linear layer gives the wrong output at a particular vector, one rank-one update corrects it exactly. The displacement at another input is proportional to its projection onto that key. A useful evaluation therefore measures the requested change and its side effects. The rank-one lecture plots this displacement through a full circle of input directions.

A low-rank adapter has a different constraint. Its factors restrict the rank of the entire correction. The adapter can be folded into the original matrix for evaluation and tested by comparing outputs before and after merging. The exercise reports the factor count and singular values as well as the numeric discrepancy. [LoRA](https://arxiv.org/abs/2106.09685)

## Averaging requires a correspondence

Hidden units can exchange positions without changing a network's function when the outgoing weights follow the same permutation. Averaging two such checkpoints coordinate by coordinate can produce a bad intermediate model even though both endpoints agree everywhere. The assignment lecture constructs this failure, solves for the correspondence and repeats the interpolation.

![Permutation alignment before averaging](08%20Permutation%20alignment%20before%20averaging/figure.svg)

Task arithmetic and TIES operate on displacements from a common model. The arithmetic lesson gives both an exact composition and incompatible targets. The TIES lesson exposes which coordinates survive trimming and which sign wins. The model-soup lesson selects on validation data and scores the frozen choice on a separate test set. Stochastic weight averaging follows late iterates from a single training run. Those experiments ask different questions and report their own evaluation criteria.

## Worked lectures

| Lecture | Question or experiment |
|---|---|
| [01. Boolean circuits with fixed weights](01%20Boolean%20circuits%20with%20fixed%20weights/README.md) | Construct a two-unit network that computes exclusive OR on every binary input. |
| [02. Compiling a piecewise linear function](02%20Compiling%20a%20piecewise%20linear%20function/README.md) | Turn a table of knots and heights into the weights of a rectifier network. |
| [03. Finite-state transition matrices](03%20Finite-state%20transition%20matrices/README.md) | Compile a parity recogniser into one transition matrix for each input symbol. |
| [04. Programming a causal attention head](04%20Programming%20a%20causal%20attention%20head/README.md) | Choose query, key and value matrices so each token reports the running mean. |
| [05. Generating weights with a hypernetwork](05%20Generating%20weights%20with%20a%20hypernetwork/README.md) | Generate a family of two-dimensional linear maps from an angle. |
| [06. Low-rank adapters and weight merging](06%20Low-rank%20adapters%20and%20weight%20merging/README.md) | Express a low-rank update in factored form and merge it into a base matrix. |
| [07. A minimum-change rank-one edit](07%20A%20minimum-change%20rank-one%20edit/README.md) | Impose a new output at one key while measuring how the edit affects other inputs. |
| [08. Permutation alignment before averaging](08%20Permutation%20alignment%20before%20averaging/README.md) | Recover the hidden-unit correspondence between two equivalent networks. |
| [09. Task vector arithmetic](09%20Task%20vector%20arithmetic/README.md) | Combine parameter displacements measured from a common starting point. |
| [10. TIES merging](10%20TIES%20merging/README.md) | Trace trimming, sign election and disjoint averaging through a small merge. |
| [11. Model soups with held-out selection](11%20Model%20soups%20with%20held-out%20selection/README.md) | Average compatible checkpoints and compare uniform and greedy soups on unseen data. |
| [12. Stochastic weight averaging](12%20Stochastic%20weight%20averaging/README.md) | Average late SGD checkpoints and measure train and test error at the averaged point. |
| [13. Installing a new piecewise response](13%20Installing%20a%20new%20piecewise%20response/README.md) | Add a specified response to an existing fitted checkpoint by writing one hidden unit and its outgoing weight. |

## Running the course

Install the packages in [requirements.txt](requirements.txt). Open any worked notebook in its lecture directory, or run its Python script from the command line. The notebooks are saved with their outputs. The small constructed models use public methods and locally generated data. [MIT licence](../LICENSE)
