# BBO Bayesian Optimisation Model Card

This model card describes the optimisation approach used in `Submission Files/BO_main.ipynb`. It follows the Mini-lesson 21.2 model-card framework and draws on the course-note reflections about transparency, interpretability, strengths, and limitations.

## Model Description

**Name:** BBO Gaussian Process Bayesian Optimisation.

**Type:** Sequential black-box optimisation approach using a Gaussian Process surrogate model and acquisition-function search.

**Version:** Final capstone repository version, September 2026.

**Implementation:** Python notebook using NumPy, SciPy, scikit-learn, and scrambled Sobol candidate generation.

**Model architecture:** A scikit-learn `GaussianProcessRegressor` with a constant kernel multiplied by an automatic-relevance-determination Matern kernel, plus a learned `WhiteKernel` noise term.

```text
ConstantKernel * Matern(ARD length scales) + WhiteKernel
```

**Input:** Observed input matrix `x`, where each row is a point in `[0, 1]^d`, and observed output vector `y`. Dimensionality ranges from 2 to 8 across the eight functions.

**Output:** Recommended next input points rounded to six decimal places. The notebook prints UCB, EI, and PI suggestions, then records a final per-function recommendation mode.

## Intended Use

This approach is suitable for the BBO capstone setting, where objective functions are hidden, evaluations are limited, and each new query should be chosen carefully. It is designed to recommend candidate points, support reflection on exploration and exploitation, and make the optimisation process reproducible.

Appropriate uses:

- Selecting candidate inputs for the eight capstone black-box functions.
- Comparing acquisition strategies such as UCB, EI, PI, and posterior-mean exploitation.
- Exploring how Gaussian Process assumptions behave under sparse data.
- Producing transparent evidence for a portfolio-ready capstone repository.

Use cases to avoid:

- Business, policy, scientific, or safety-critical decision-making without independent validation.
- Optimisation problems with categorical, text, image, or other structured inputs without major redesign.
- Large datasets where Gaussian Process fitting becomes computationally impractical.
- Claims of guaranteed global optimality.
- Blind automation without reviewing diagnostics, fitted kernels, uncertainty, and duplicate checks.

## Details

The decision process is iterative. For each function, the notebook validates the observed data, transforms targets where needed, fits one or more Gaussian Process models, selects the best Matern smoothness option by log marginal likelihood, generates a large candidate pool, removes previously evaluated points, scores candidates, and returns rounded recommendations.

The main acquisition functions are:

- UCB: prioritises high predicted value plus uncertainty.
- EI: prioritises expected improvement over the best observed value.
- PI: prioritises probability of improving over the best observed value.
- Exploit: selects the largest posterior mean when uncertainty should not drive the final ranking.

The strategy evolved across the submitted rounds:

- Early rounds used broader exploration to compensate for sparse initial coverage. The course notes identify Function 2 as an example where the initial samples were clustered in one coordinate, so uncertainty-guided search was useful.
- As more outputs were returned, the approach moved toward exploitation with scale-aware EI. The tenth-round reflection notes that the strategy was increasingly focused on exploiting promising regions while keeping `xi` scaled to each function's observed output range.
- Function 1 became a special case because raw outputs varied over many orders of magnitude and contained sharp peaks or troughs. The final notebook uses a rank-transformed target, compares rougher Matern smoothness, and avoids neighbourhoods of poor observations.
- Function 5 was tested at the all-ones boundary after outputs suggested that higher coordinate values were associated with larger returns. This produced the current best observed Function 5 value.
- Functions 3, 4, and 6 use trust-region candidate pools in the final settings to refine around strong observations or recover from stalled global search.
- Functions 7 and 8 keep EI-based refinement with larger distance filters because recent points were strong but the high-dimensional spaces remain sparse.

The current implementation is more function-specific than the early one-size-fits-all approach. This directly addresses a limitation identified in the course reflections: using one kernel and one policy for all eight functions trades simplicity for reduced accuracy on individual functions.

## Performance

Performance is measured by observed optimisation progress because the true global maxima are hidden. The main metrics are best observed output, improvement over the initial best, and qualitative confidence based on dimensionality, noise evidence, and sampling density.

| Function | Initial best | Current best | Improvement | Current best input |
| --- | ---: | ---: | ---: | --- |
| 1 | 7.710875e-16 | 1.115019e-11 | 1.114942e-11 | `[0.715262, 0.720947]` |
| 2 | 0.611205 | 0.629629 | 0.018424 | `[0.690566, 0.997509]` |
| 3 | -0.034835 | -0.029348 | 0.005487 | `[0.987564, 0.501912, 0.078963]` |
| 4 | -4.025542 | 0.525735 | 4.551278 | `[0.401606, 0.419143, 0.393235, 0.413199]` |
| 5 | 1088.859618 | 8662.482500 | 7573.622882 | `[1.000000, 1.000000, 1.000000, 1.000000]` |
| 6 | -0.714265 | -0.224250 | 0.490015 | `[0.380179, 0.370163, 0.595055, 0.782124, 0.000000]` |
| 7 | 1.364968 | 2.237965 | 0.872997 | `[0.057896, 0.316107, 0.433405, 0.132816, 0.342848, 0.711037]` |
| 8 | 9.598482 | 9.949390 | 0.350908 | `[0.045526, 0.142949, 0.122326, 0.039481, 0.991262, 0.613700, 0.199489, 0.499233]` |

Additional diagnostics printed by the notebook include fitted kernels, log marginal likelihood, selected Matern smoothness, learned noise levels, ARD length scales, candidate counts after filtering, and the model-space posterior mean at each recommendation.

The strongest observed improvement is Function 5, where the model-supported boundary test produced a large gain. Function 4 also improved substantially after local refinement, and Function 6 now shows a meaningful recovery after the latest recorded local search. Function 3 remains the clearest underperformance case because the improvement is still small.

## Limitations

The approach assumes that nearby candidate points often have related output values and that each function can be approximated well enough by a Matern-kernel Gaussian Process to guide the next query. This assumption supports data-efficient optimisation, but it can fail for discontinuous, extremely sharp, highly noisy, or irregular functions.

Key limitations:

- Sparse data: each function has few observations relative to its search space.
- High dimensionality: Functions 6, 7, and 8 are difficult to cover with limited queries.
- Kernel dependence: a GP can mislead recommendations if the true function shape is not well represented by the selected kernel.
- Noise uncertainty: Function 2 shows evidence of noisy repeated outputs, but there are too few repeats to estimate noise robustly.
- Candidate-pool dependence: recommendations are selected from generated candidate pools rather than continuous analytic optima.
- Local refinement trade-off: trust regions can improve exploitation but may miss distant optima.
- Boundary bias: Function 5's all-ones result is strong, but it does not prove that all functions benefit from boundary chasing.

Strengths:

- Works well with small data.
- Provides uncertainty estimates, not just point predictions.
- Makes acquisition choices explicit.
- Uses duplicate filtering and rounded submission checks.
- Allows per-function settings rather than forcing one global policy.
- Produces diagnostics that make decisions easier to audit.

## Trade-offs

The main trade-off is between exploration and exploitation. UCB explores uncertain regions but can spend scarce evaluations on points that are informative rather than high-scoring. EI and PI focus more directly on improvement, but they can become too local when the current best region looks convincing. Posterior-mean exploitation is useful near the end of the query budget, but it depends heavily on the GP being well specified.

There is also a simplicity trade-off. A single global policy would be easier to explain, but it performed less well because the eight functions have different dimensions, scales, and apparent smoothness. The final notebook therefore uses per-function settings, which improves transparency and performance at the cost of more configuration.

Candidate-pool search is another practical compromise. Large Sobol pools are reproducible and robust, but they do not guarantee a continuous optimum. Trust-region candidates improve local density around strong observations, but they can miss distant optima if the surrogate becomes overconfident.

Adding more detail improves usefulness when it explains decisions, assumptions, and failure modes. The model card deliberately stays at a stakeholder-readable level and points to the notebook for executable detail. That structure is sufficient for the GitHub deliverable because the README, notebook, datasheet, and model card work together: the README orients the reader, the datasheet documents the data, the model card explains the approach, and the notebook contains implementation-level evidence.

## Ethical Considerations

The capstone data is synthetic and does not include personal information, so privacy and consent risks are low. The main ethical issue is responsible transparency: black-box optimisation can produce confident recommendations even when the objective function, constraints, and uncertainty are incompletely understood.

Transparency supports reproducibility by showing:

- where the data came from;
- which points were submitted;
- how updated arrays were rebuilt;
- which transformations and model choices were used;
- how recommendations were selected;
- where the model can fail.

This matters for real-world adaptation. In applied settings such as product development, marketing experiments, engineering design, or scientific experimentation, black-box optimisation could affect budgets, safety, users, or operational outcomes. Before adapting this approach outside the academic challenge, a practitioner should add stronger validation, domain constraints, repeat evaluations for noise, human review of recommendations, and monitoring for harms that are not captured by a single scalar output.

The model card is intentionally candid about limitations because a document that only advertises strengths is less useful for decision-makers. Honest documentation improves trust by making it easier for reviewers to reproduce the logic, challenge assumptions, and decide whether the approach is appropriate for their own context.
