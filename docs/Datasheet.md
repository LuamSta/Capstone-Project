# BBO Capstone Dataset Datasheet

This datasheet documents the dataset used in the Black Box Optimisation capstone project. It follows the Mini-lesson 21.1 datasheet framework and uses the course notes as reflection material. The notes are treated as background context only; the repository data and notebooks are the source of truth for file locations, shapes, results, and implementation details.

## Motivation

The dataset was created to support the BBO capstone task: maximise eight hidden objective functions under a limited query budget. Each function returns a scalar output for a continuous input vector in the unit hypercube. The mathematical definitions, smoothness, and noise properties of the functions are intentionally hidden, so the dataset supports sequential decision-making under uncertainty rather than ordinary supervised learning.

The original data was provided by Imperial College London / Emeritus course materials for an academic optimisation challenge. The updated dataset was created during the capstone by appending submitted query points and returned outputs. Its purpose is to make the optimisation process transparent, reproducible, and reviewable by peers and facilitators.

This dataset is useful for demonstrating:

- Bayesian optimisation with very small sample sizes.
- Gaussian Process surrogate modelling under uncertainty.
- Exploration and exploitation trade-offs across functions with different dimensions and output scales.
- Documentation of data provenance, assumptions, gaps, and maintenance.

## Composition

The dataset contains eight separate black-box optimisation records, one for each objective function. Each function directory contains four NumPy files:

- `initial_inputs.npy`: the original course-provided input locations.
- `initial_outputs.npy`: the original course-provided output values.
- `updated_inputs.npy`: the initial inputs plus submitted capstone query points.
- `updated_outputs.npy`: the initial outputs plus returned capstone output values.

All input values are numeric, continuous, and bounded in `[0, 1]`. Outputs are scalar numeric scores. There is no personal, sensitive, or demographic data.

| Function | Input dimensionality | Initial rows | Current rows | Best observed output | Best observed input |
| --- | ---: | ---: | ---: | ---: | --- |
| 1 | 2 | 10 | 22 | 1.115019e-11 | `[0.715262, 0.720947]` |
| 2 | 2 | 10 | 22 | 0.629629 | `[0.690566, 0.997509]` |
| 3 | 3 | 15 | 27 | -0.029348 | `[0.987564, 0.501912, 0.078963]` |
| 4 | 4 | 30 | 42 | 0.525735 | `[0.401606, 0.419143, 0.393235, 0.413199]` |
| 5 | 4 | 20 | 32 | 8662.482500 | `[1.000000, 1.000000, 1.000000, 1.000000]` |
| 6 | 5 | 20 | 32 | -0.224250 | `[0.380179, 0.370163, 0.595055, 0.782124, 0.000000]` |
| 7 | 6 | 30 | 42 | 2.237965 | `[0.057896, 0.316107, 0.433405, 0.132816, 0.342848, 0.711037]` |
| 8 | 8 | 40 | 52 | 9.949390 | `[0.045526, 0.142949, 0.122326, 0.039481, 0.991262, 0.613700, 0.199489, 0.499233]` |

## Function-Specific Overview

The template asks for a real-world scenario for each function. The capstone does not reveal the real domains, so this datasheet does not invent application meanings. Each function is treated as a generic bounded tuning problem: choose a continuous input vector in `[0, 1]^d` to maximise a hidden scalar score.

| Function | Template-style description | Observed data behaviour | Strategy and confidence |
| --- | --- | --- | --- |
| 1 | Two-dimensional hidden score with 10 initial and 22 current rows. | Outputs span many orders of magnitude and are dominated by near-zero values plus sharp outliers. | Uses rank-transformed targets, rough-kernel comparison, and avoidance of poor regions. Confidence is low to moderate because the scale is difficult to model. |
| 2 | Two-dimensional hidden score with 10 initial and 22 current rows. | A repeated point returned noticeably different values, so noise is plausible. Initial coverage was also uneven. | Uses GP uncertainty and duplicate-aware recommendations. Confidence is low to moderate because noise is hard to estimate from few repeats. |
| 3 | Three-dimensional hidden score with 15 initial and 27 current rows. | Improvement has been small, suggesting a difficult or poorly sampled landscape. | Uses EI and local candidate pools around the strongest observations. Confidence is low because the best value improved only slightly. |
| 4 | Four-dimensional hidden score with 30 initial and 42 current rows. | The search found a much better local region after several poor responses, suggesting sharp structure. | Uses rough-kernel model selection and trust-region refinement. Confidence is moderate because improvement is strong but global coverage is still limited. |
| 5 | Four-dimensional hidden score with 20 initial and 32 current rows. | Larger coordinate values appeared promising, and the all-ones boundary point produced the best observed score. | Uses exploitation and boundary-aware reasoning. Confidence is moderate, with a risk that the apparent boundary trend is local or coincidental. |
| 6 | Five-dimensional hidden score with 20 initial and 32 current rows. | Recent local search produced a meaningful recovery after weaker boundary-focused points. | Uses EI, rough-kernel comparison, wider distance filters, and local refinement. Confidence is low to moderate because dimensionality is high. |
| 7 | Six-dimensional hidden score with 30 initial and 42 current rows. | One submitted point produced a strong best value, but the space remains sparse. | Uses EI refinement while keeping spacing from previous points. Confidence is low to moderate because high dimensionality leaves large unexplored regions. |
| 8 | Eight-dimensional hidden score with 40 initial and 52 current rows. | Best values are close together and several strong points sit near coordinate boundaries. | Uses EI with larger distance filters to avoid brittle repeats. Confidence is low to moderate because the eight-dimensional space is sparsely covered. |

Known gaps and limitations:

- The true objective functions are unknown, so the dataset cannot label global optima.
- The updated data is sparse, especially for Functions 6, 7, and 8, where dimensionality is high.
- Initial points are not guaranteed to cover each search space evenly. The course notes highlight Function 2 as an example where the first coordinate was initially clustered, leaving parts of the search space less explored.
- Noise cannot be fully estimated because exact repeats are rare. One repeated Function 2 location, `[0.702489, 0.926564]`, returned `0.495529` and `0.606690`, suggesting that noisy outputs are possible.
- Output scales vary substantially. Function 5 reaches thousands, while Function 1 has values clustered near zero with sharp negative outliers.

## Collection Process

The initial observations were supplied by the course as the starting point for the capstone challenge. Additional observations were collected by submitting one query per function in each optimisation round and recording the returned output.

Queries were generated mainly with Bayesian optimisation. The repository's `BO_main.ipynb` notebook fits a Gaussian Process surrogate to each function's observed input-output pairs, generates candidate points using scrambled Sobol sequences, scores those candidates with acquisition functions, and prints recommended points. The update process is recorded in `Data Saver.ipynb`, which appends the submitted query points and returned outputs by rebuilding `updated_*.npy` files from the initial data and the full submitted history.

The course reflection notes show that the strategy evolved over the rounds:

- Early rounds prioritised exploration, especially where the initial samples looked clustered or sparse.
- UCB was used to investigate uncertain regions of the search space.
- EI became more important later as the strategy shifted toward exploitation under the limited submission budget.
- Function 1 required special handling because its values varied over many orders of magnitude and were difficult to model directly.
- Function 5 was explicitly tested at `[1, 1, 1, 1]` after outputs suggested that values increased near the upper boundary.
- Functions 3, 4, 6, 7, and 8 received more local refinement once global recommendations stalled or recent observations looked strong.

The repository currently contains the initial observations plus twelve recorded optimisation rounds. The course notes include a tenth-round reflection; the dataset now includes later recorded updates as well.

## Weekly Iteration And Learning

The first rounds focused on broad exploration because the hidden functions had few observations and unknown smoothness. As the returned outputs accumulated, the strategy became more function-specific. Function 1 needed a target transform because raw values were numerically awkward. Function 5 benefited from testing a clear boundary hypothesis. Functions 3, 4, and 6 needed local candidate pools because global search was either stalled or too diffuse. Functions 7 and 8 kept more conservative spacing because high-dimensional local refinement can repeatedly choose nearly identical points after rounding.

If restarted, the main changes would be to introduce per-function settings earlier, deliberately schedule a small number of repeated evaluations for noise estimation, and compare the GP recommendations against simple Sobol and local-search baselines after every round.

## Preprocessing And Uses

Inputs are already scaled to `[0, 1]`, so no feature scaling is applied before modelling. The notebooks validate that inputs are finite, two-dimensional in storage, inside the unit hypercube, and matched to finite outputs. Candidate points are rounded to six decimal places for submission and filtered to avoid exact repeats at that precision.

Outputs are stored in their raw returned form. During modelling, the Gaussian Process uses `normalize_y=True`. Function 1 is additionally modelled with a rank-transformed target in the final notebook because its raw output scale is dominated by near-zero values and an extreme negative outlier. The raw outputs remain preserved in the dataset.

Intended uses:

- Reproduce the capstone optimisation process.
- Audit submitted inputs and returned outputs.
- Train and compare surrogate models for the eight black-box functions.
- Evaluate exploration-exploitation strategies under small-data constraints.
- Support the repository README, datasheet, model card, and final discussion-board submission.

Inappropriate uses:

- Claiming that the best observed outputs are proven global optima.
- Using the data as evidence for real-world business, scientific, medical, or policy decisions.
- Treating the hidden functions as representative of a real population or real-world process.
- Benchmarking unrelated optimisation methods without explaining the limited query budget, sparse coverage, and synthetic source.
- Deleting poor observations or outliers simply because they are inconvenient; they are part of the observed optimisation history.

## Distribution And Maintenance

The dataset is available in this GitHub repository under `Submission Files/Data/`. The README links to the data, notebooks, datasheet, and model card so reviewers can find the full project artefact from the repository landing page.

Terms of use are educational and repository-based. The initial data and evaluator outputs come from the capstone course context, so they should be used for course review, portfolio demonstration, and reproducibility of this project rather than redistributed as an official public benchmark. The data contains no personal information.

The repository owner maintains the dataset during the capstone submission period. `Data Saver.ipynb` is the maintenance mechanism for rebuilding the updated arrays from the original data and recorded submissions. There is no planned post-course maintenance unless further capstone feedback or portfolio refinement requires corrections.

Maintenance transparency matters because future reviewers need to know which files are original, which files were updated, how updates were made, and who is responsible for corrections. Clear maintenance notes also make the optimisation history easier to reproduce and reduce the risk of accidental duplicate submissions or hidden data changes.
