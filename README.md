# Black Box Optimisation Capstone Project

## Project overview

This repository documents my Black Box Optimisation capstone project for the Imperial College London Machine Learning programme. The challenge is to maximise eight unknown functions using a strict budget of 13 evaluations per function. Their mathematical form and noise level are hidden, so every query must balance learning about the search space with exploiting promising regions.

Each function accepts a continuous vector

$$
x = [x_1, x_2, \ldots, x_d], \qquad x_i \in [0,1],
$$

where the dimensionality ranges from 2 to 8. Query points are submitted to six decimal places and produce a scalar response $y = f(x)$.

## Non-technical explanation

This project searches for the best settings for eight hidden scoring systems when each trial is limited and valuable. Instead of trying random settings, it builds a statistical picture of each system from previous results, then recommends the next setting that is either promising or informative. The model uses uncertainty to balance learning about unexplored areas with improving on the best result so far. After twelve rounds, the strongest gains came from functions where the search learned useful local patterns, especially Functions 4, 5, 6, 7, and 8. The repository shows the data, code, choices, limitations, and results needed to reproduce the work.

## Final deliverable materials

| Component | Location |
| --- | --- |
| Main optimisation notebook | [`Submission Files/BO_main.ipynb`](Submission%20Files/BO_main.ipynb) |
| Data update notebook | [`Submission Files/Data Saver.ipynb`](Submission%20Files/Data%20Saver.ipynb) |
| Diagnostic notebook | [`Submission Files/Data Explorer.ipynb`](Submission%20Files/Data%20Explorer.ipynb) |
| Data arrays | [`Submission Files/Data/`](Submission%20Files/Data/) |
| Datasheet | [`docs/Datasheet.md`](docs/Datasheet.md) |
| Model card | [`docs/Model Card.md`](docs/Model%20Card.md) |
| Python dependencies | [`requirements.txt`](requirements.txt) |

## Data

The data consists of course-provided initial samples for eight black-box functions and twelve recorded optimisation rounds generated during the capstone. Each function has `initial_inputs.npy`, `initial_outputs.npy`, `updated_inputs.npy`, and `updated_outputs.npy` files under `Submission Files/Data/function_1` to `Submission Files/Data/function_8`.

The arrays are small enough to keep directly in GitHub. There are no large external datasets in this project. The only external source is the capstone black-box evaluator/course materials, which provided the initial data and returned the submitted output values.

## Model

The optimisation loop uses Gaussian Process regression as a probabilistic surrogate. The current implementation includes:

- evidence-based selection between Matérn covariance functions, with optional rough Matérn 1/2 support;
- automatic relevance determination (one length scale per input dimension);
- a learned white-noise term for noisy or repeated observations;
- output normalisation and multiple optimiser restarts;
- scrambled Sobol candidate generation;
- Upper Confidence Bound (UCB), Expected Improvement (EI), and Probability of Improvement (PI);
- protection against zero-variance EI and PI calculations;
- removal of candidates already evaluated, including collisions after six-decimal rounding.

The acquisition functions are

$$
\operatorname{UCB}(x) = \mu(x) + \beta\sigma(x)
$$

and

$$
\operatorname{EI}(x) = (\mu(x)-y^+-\xi)\Phi(z)+\sigma(x)\phi(z),
\qquad z=\frac{\mu(x)-y^+-\xi}{\sigma(x)}.
$$

Probability of Improvement is

$$
\operatorname{PI}(x)=\Phi\left(\frac{\mu(x)-y^+-\xi}{\sigma(x)}\right).
$$

Here, $y^+$ is the best observed value, $\beta$ controls UCB exploration, and $\xi$ controls EI exploration.

Kernel smoothness can be controlled when calling `main`:

```python
# Default: compare moderately smooth and smooth kernels.
main(x, y, smoothness_options=(1.5, 2.5))

# Include the rough Matérn 1/2 kernel in model selection.
main(x, y, smoothness_options=(0.5, 1.5, 2.5))

# Force the rough kernel for a targeted experiment.
main(x, y, smoothness_options=(0.5,))
```

Rough-kernel support is optional because it can capture sharp changes but may overfit noise on smoother functions.

The final recommendation policy can also be selected explicitly:

```python
# Pure exploitation: choose the candidate with the largest GP posterior mean.
main(x, y, recommendation_mode="exploit")

# Balanced improvement or confidence-bound alternatives.
main(x, y, recommendation_mode="ei")
main(x, y, recommendation_mode="pi")
main(x, y, recommendation_mode="ucb")
```

`exploit` guarantees that predictive uncertainty is not part of the final ranking. UCB, EI, and PI are still calculated and printed for comparison. Pure exploitation is useful near the end of the query budget, but it is more dependent on the current GP being correctly specified.

The submission notebook now applies per-function settings rather than one global policy for all eight functions. Function 1 is fitted on ranked outputs to reduce the effect of extreme scale and outliers; stalled or sharply peaked functions use EI, rough-kernel model selection, trust-region candidate pools around strong observations, and larger distance filters where recent local refinement has become brittle.

## Hyperparameter optimisation

The Gaussian Process kernel hyperparameters are optimised by scikit-learn's marginal-likelihood optimiser with five restarts. These include the signal scale, one ARD length scale per input dimension, and the learned white-noise level. The notebook compares allowed Matérn smoothness values and keeps the fitted model with the highest log marginal likelihood.

The default smoothness comparison uses Matérn $\nu = 1.5$ and $\nu = 2.5$. Functions with rougher or stalled behaviour also compare $\nu = 0.5$. UCB uses $\beta = 3$. EI and PI use a scale-aware $\xi$, set to one percent of the transformed target standard deviation for each function.

## Results

The table below reports best observed values after twelve recorded optimisation rounds. These are observed improvements under a strict query budget, not certified global optima.

| Function | Initial best | Current best | Improvement | Best observed input |
| --- | ---: | ---: | ---: | --- |
| 1 | 7.710875e-16 | 1.115019e-11 | 1.114942e-11 | `[0.715262, 0.720947]` |
| 2 | 0.611205 | 0.629629 | 0.018424 | `[0.690566, 0.997509]` |
| 3 | -0.034835 | -0.029348 | 0.005487 | `[0.987564, 0.501912, 0.078963]` |
| 4 | -4.025542 | 0.525735 | 4.551278 | `[0.401606, 0.419143, 0.393235, 0.413199]` |
| 5 | 1088.859618 | 8662.482500 | 7573.622882 | `[1.000000, 1.000000, 1.000000, 1.000000]` |
| 6 | -0.714265 | -0.224250 | 0.490015 | `[0.380179, 0.370163, 0.595055, 0.782124, 0.000000]` |
| 7 | 1.364968 | 2.237965 | 0.872997 | `[0.057896, 0.316107, 0.433405, 0.132816, 0.342848, 0.711037]` |
| 8 | 9.598482 | 9.949390 | 0.350908 | `[0.045526, 0.142949, 0.122326, 0.039481, 0.991262, 0.613700, 0.199489, 0.499233]` |

## Repository structure

```text
Capstone-Project/
├── README.md
├── requirements.txt
├── docs/
│   ├── Datasheet.md
│   └── Model Card.md
└── Submission Files/
    ├── BO_main.ipynb          # GP fitting and candidate recommendation
    ├── Data Explorer.ipynb    # diagnostics and visual checks
    ├── Data Saver.ipynb       # validated construction of updated datasets
    └── Data/
        ├── function_1/ ... function_8/
        ├── Submission Data/   # submitted inputs and returned outputs
        └── Legacy Data/       # retained backup of original data
```

The legacy directory is intentionally retained as a backup. The optimisation notebooks use only the current files under `Submission Files/Data/function_1` to `Submission Files/Data/function_8`.

## Setup and execution

Create an isolated environment from the repository root:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m ipykernel install --user --name capstone-bo --display-name "Capstone BO"
jupyter lab
```

Run notebooks from either the repository root or the `Submission Files` directory. The notebooks resolve data paths through `Submission Files/Data`.

1. Update the weekly input and output records in `Data Saver.ipynb`.
2. Run all cells there to rebuild and validate the updated arrays.
3. Restart the kernel for `BO_main.ipynb` and run all cells from top to bottom.
4. Review convergence warnings, fitted kernels, and both recommendations.
5. Confirm the chosen point is new and formatted to six decimal places.

## Data integrity

The notebooks validate that inputs are finite, two-dimensional, inside the unit hypercube, and matched to finite outputs. They also check that every weekly record covers all eight functions, surface repeated locations, verify final input/output row counts, and prevent repeated recommendations after rounding.

`Data Saver.ipynb` deliberately rebuilds `updated_*.npy` from the original arrays plus the complete weekly history, including the latest recorded submission round. This makes reruns idempotent and prevents accidentally appending the same week twice.

## Evaluation and interpretation

Model quality should be assessed separately for every function because dimensionality, response scale, smoothness, and noise can differ substantially. Useful diagnostics include:

- log marginal likelihood and kernel-bound warnings;
- leave-one-out RMSE and predictive log likelihood;
- empirical coverage of predictive intervals;
- best observed value by week;
- comparison with random and pure Sobol baselines;
- stability of recommendations across optimiser seeds.

ARD length scales can suggest relatively influential dimensions, but they should be interpreted cautiously when there are few observations in a high-dimensional space.

## Current status and next experiments

The repository contains the initial datasets plus twelve recorded optimisation rounds. Current development focuses on reliable GP fitting, duplicate-safe recommendations, and performance in higher dimensions.

Planned experiments are:

1. extend the current evidence-based Matérn comparison with leave-one-out validation and an RBF baseline;
2. comparison of isotropic and ARD length scales;
3. sensitivity analysis for the learned noise level, $\beta$, and $\xi$;
4. local bounded refinement of the strongest Sobol acquisition candidates;
5. retrospective comparison of EI, UCB, and Sobol search;
6. posterior mean, uncertainty, and acquisition plots for the two-dimensional functions.

## Limitations

Gaussian Processes can be sensitive to kernel and noise assumptions when observations are sparse. ARD also introduces many hyperparameters for the higher-dimensional functions. Acquisition values are model-dependent and should be treated as decision support rather than guaranteed optima. This project addresses these limitations through explicit noise modelling, diagnostics, low-discrepancy global search, and planned retrospective validation.
