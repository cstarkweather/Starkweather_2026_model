# Starkweather_2026_dynamical_systems

## Overview

This repository contains MATLAB code for the dynamical systems modeling described in:

**Starkweather et al., 2026, Nature Neuroscience (Revision 2)**

The model implements a two-node mutual inhibition dynamical system to explain decision-making behavior in an approach–avoidance task. It captures how reward and punishment signals are integrated over time to produce bistable neural dynamics, state transitions, and choice outcomes.

---

## Model Summary

Following Machens et al 2005, Science: The model consists of two interacting neural populations representing competing decision states (approach vs. avoid). Activity evolves according to:

dx/dt = (−x + F(−w*y + λ*E_x(t)) + η) / τ
dy/dt = (−y + F(−w*x + λ*E_y(t)) + η) / τ

where:

* `x`, `y`: neural activity of the two populations
* `w`: mutual inhibition strength
* `λ`: input gain
* `E_x(t)`, `E_y(t)`: external inputs (reward/punishment)
* `η`: Gaussian noise
* `τ`: time constant

The nonlinear transfer function is:
F(s) = a * tanh(s + b) + c

---

## Key Features

* Bistable dynamics with stable and unstable fixed points
* Noise-driven transitions between decision states
* Trial-by-trial simulation of behavior
* Bayesian decoding of latent decision states
* Fits to behavioral data (choice probability, RT, conflict)

---

## Repository Structure

* master_fitting.m --> essential 'start' for multi-start optimization code
* once have an optimized fit (or load optimization_run.mat containing the result of running these codes), then can run plot_trajectories to see nullcline plots, trajectories
* bistability_analysis_final and conflict_bistability_relationships to assess fit relative to whether bistable at particular offers

* Additional helper functions for simulation, decoding, and visualization

---

## Behavioral Task

The model reproduces behavior from an approach–avoidance task in which:

* Reward (R) and punishment (P) are independently varied
* Subjects choose to approach (risk reward/punishment) or avoid
* Conflict is highest when approach and avoidance probabilities are similar

Conflict is quantified using Shannon entropy:

Conflict = −p * log2(p) − (1 − p) * log2(1 − p)

---

## Parameterization

The model is parameterized by:

* `alphareward`, `alphapunish`: input scaling
* `noise`: stochastic noise magnitude
* `w_i`: mutual inhibition strength
* `lambda`: input gain
* `a`, `b`, `c`: nonlinear transfer parameters
* `time_stable`: decision stability window
* `thresh`: state preference threshold

---

## Running the Model

1. Set parameters (either manually or via fitting output)

2. Run simulations:

   ```matlab
   simulate_model_summaries_cached(...)
   ```

3. Visualize dynamics:

   ```matlab
   mfsim_fitting_plot_nullclines_fast(...)
   ```

---

## Reproducing Figures

Example mappings (adjust as needed):

* Behavioral fits → `fit_mutual_inhibition_direct*.m`
* Nullclines and fixed points → `mfsim_fitting_plot_nullclines_*`
* State transitions → `detectStatePreferences.m`
* Model summary plots → `simulate_model_summaries_cached.m`

---

## Notes on Model Behavior

* High conflict conditions produce multiple fixed points, including an unstable intermediate state
* Decision timing emerges from stabilization into a dominant attractor
* Reaction times scale inversely with value magnitude
* Noise modulates switching dynamics and variability

---

## Requirements

* MATLAB (tested on R2022+ recommended)
* No external toolboxes required beyond base MATLAB

---

## Citation

If you use this code, please cite:

Starkweather et al., Nature Neuroscience (2026)

---

## Contact

For questions, please contact:
Clara Starkweather, MD, PhD
University of California, San Francisco

---

## License

This repository is provided for academic use. Please contact the author for reuse or redistribution.
