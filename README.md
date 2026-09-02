# Black Box Optimisation Capstone Project

## Project overview

This repository documents my Black Box Optimisation capstone project for the Imperial College London Machine Learning programme. The challenge is to maximise eight unknown functions using a strict budget of 13 evaluations per function. Their mathematical form and noise level are hidden, so every query must balance learning about the search space with exploiting promising regions.

Each function accepts a continuous vector

\[
x = [x_1, x_2, \ldots, x_d], \qquad x_i \in [0,1],
\]

where the dimensionality ranges from 2 to 8. Query points are submitted to six decimal places and produce a scalar response \(y=f(x)\).

## Approach

The optimisation loop uses Gaussian Process regression as a probabilistic surrogate. The current implementation includes:

- evidence-based selection between Matérn 3/2 and Matérn 5/2 covariance functions;
- automatic relevance determination (one length scale per input dimension);
- a learned white-noise term for noisy or repeated observations;
- output normalisation and multiple optimiser restarts;
- scrambled Sobol candidate generation;
- Upper Confidence Bound (UCB) and Expected Improvement (EI);
- protection against zero-variance EI calculations;
- removal of candidates already evaluated, including collisions after six-decimal rounding.

The acquisition functions are

\[
\operatorname{UCB}(x) = \mu(x) + \beta\sigma(x)
\]

and

\[
\operatorname{EI}(x) = (\mu(x)-y^+-\xi)\Phi(z)+\sigma(x)\phi(z),
\qquad z=\frac{\mu(x)-y^+-\xi}{\sigma(x)}.
\]

Here, \(y^+\) is the best observed value, \(\beta\) controls UCB exploration, and \(\xi\) controls EI exploration.

## Repository structure

```text
Capstone-Project/
├── README.md
├── requirements.txt
└── Submission/
    ├── BO_main.ipynb          # GP fitting and candidate recommendation
    ├── Data Saver.ipynb       # validated construction of updated datasets
    ├── function_1/ ... function_8/
    ├── Submission Data/       # submitted inputs and returned outputs
    └── Legacy Data/           # retained backup of original data
```

The legacy directory is intentionally retained as a backup. The optimisation notebooks use only the current files under `Submission/function_1` to `Submission/function_8`.

## Setup and execution

Create an isolated environment from the repository root:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m ipykernel install --user --name capstone-bo --display-name "Capstone BO"
jupyter lab
```

Run notebooks from the `Submission` directory so their relative data paths resolve correctly.

1. Update the weekly input and output records in `Data Saver.ipynb`.
2. Run all cells there to rebuild and validate the updated arrays.
3. Restart the kernel for `BO_main.ipynb` and run all cells from top to bottom.
4. Review convergence warnings, fitted kernels, and both recommendations.
5. Confirm the chosen point is new and formatted to six decimal places.

## Data integrity

The notebooks validate that inputs are finite, two-dimensional, inside the unit hypercube, and matched to finite outputs. They also check that every weekly record covers all eight functions, surface repeated locations, verify final input/output row counts, and prevent repeated recommendations after rounding.

`Data Saver.ipynb` deliberately rebuilds `updated_*.npy` from the original arrays plus the complete weekly history. This makes reruns idempotent and prevents accidentally appending the same week twice.

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

The repository contains the initial datasets plus seven recorded optimisation rounds. Current development focuses on reliable GP fitting, duplicate-safe recommendations, and performance in higher dimensions.

Planned experiments are:

1. extend the current evidence-based Matérn comparison with leave-one-out validation and an RBF baseline;
2. comparison of isotropic and ARD length scales;
3. sensitivity analysis for the learned noise level, \(\beta\), and \(\xi\);
4. local bounded refinement of the strongest Sobol acquisition candidates;
5. retrospective comparison of EI, UCB, and Sobol search;
6. posterior mean, uncertainty, and acquisition plots for the two-dimensional functions.

## Limitations

Gaussian Processes can be sensitive to kernel and noise assumptions when observations are sparse. ARD also introduces many hyperparameters for the higher-dimensional functions. Acquisition values are model-dependent and should be treated as decision support rather than guaranteed optima. This project addresses these limitations through explicit noise modelling, diagnostics, low-discrepancy global search, and planned retrospective validation.
