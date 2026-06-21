# Black Box Optimisation Capstone Project

## Project Overview

This repository documents my work on a Black Box Optimisation (BBO) challenge completed as part of the WorldQuant University Data Science programme.

The objective is to maximise the outputs of eight unknown functions using a limited query budget. The underlying mathematical form of each function is hidden, meaning that optimisation decisions must be made using only historical observations and machine learning models.

This type of problem is highly relevant in real-world machine learning and engineering applications where experiments are expensive, time-consuming or impossible to evaluate exhaustively. Examples include hyperparameter tuning, manufacturing optimisation, algorithm design and scientific experimentation.

For me, this project provides practical experience with Bayesian Optimisation, surrogate modelling and decision-making under uncertainty, all of which are valuable skills for future Data Science and Machine Learning roles.

---

## Inputs and Outputs

### Inputs

Each function accepts an input vector:

x = [x₁, x₂, ..., xₙ]

where:

* Number of dimensions ranges from 2 to 8
* Input values are continuous
* Query points must be submitted to six decimal places
* Only one new query can be submitted per function each week

Example:

Function 1 (2D)

x = [0.523412, 0.872341]

Function 8 (8D)

x = [0.123456, 0.234567, 0.345678, 0.456789, 0.567891, 0.678912, 0.789123, 0.891234]

### Outputs

Each query returns a single response value:

y = f(x)

The optimisation objective is to identify input combinations that produce the highest possible output values.

---

## Challenge Objectives

The primary goal is to maximise the value of each unknown function while operating under a strict query budget.

Key constraints include:

* Only 13 total submissions per function
* Unknown functional form
* Unknown level of noise
* Limited observations available early in the competition
* Increasing dimensionality across functions

Because queries are expensive, each submission must balance learning about the search space with exploiting promising regions already discovered.

---

## Technical Approach

### Surrogate Model

My current approach uses Gaussian Process Regression (GPR) as a surrogate model for the unknown functions.

The Gaussian Process provides:

* Mean prediction
* Prediction uncertainty
* A probabilistic estimate of unexplored regions

This makes it well suited for Bayesian Optimisation problems where evaluation budgets are limited.

### Acquisition Function

I currently use an Upper Confidence Bound (UCB) acquisition function:

UCB(x) = μ(x) + βσ(x)

where:

* μ(x) = predicted mean
* σ(x) = predicted uncertainty
* β = exploration parameter

### Exploration Strategy

My current optimisation strategy is deliberately exploration-focused.

Weeks 1-3:

* UCB with β = 5
* Prioritise uncertainty reduction
* Sample unexplored regions

Planned Weeks 4-6:

* Reduce β
* Transition towards balanced exploration and exploitation

Planned Weeks 7-13:

* Focus increasingly on exploitation
* Concentrate searches around high-performing regions

This staged approach is designed to avoid premature convergence while still allowing sufficient time to refine solutions near the end of the competition.

### Future Experiments

Potential future improvements include:

* Expected Improvement (EI) acquisition functions
* Feature importance analysis
* Dimensionality reduction techniques
* SVM-based classification of high-performing regions
* Alternative Gaussian Process kernels

---

## Current Status

The project is currently in Week 3 of 13.

Current focus:

* Building robust Gaussian Process models
* Improving performance in higher-dimensional functions
* Expanding coverage of the search space
* Evaluating the impact of exploration-heavy query selection

This repository will be updated throughout the competition as new observations are collected and the optimisation strategy evolves.
