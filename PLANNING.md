# PSI Pipeline — Consolidated Development Plan

This document outlines a prioritized plan to improve the correctness, performance, and reproducibility of the `psi_cloud_pipeline.py` script.

---

## A. Correctness & Reproducibility (Highest Priority)

### A1. Unify Prediction Post-Processing
**Problem**: The post-processing steps applied to the predicted transition state (TS) distance matrix are different between the training evaluation (`train_pipeline`) and the inference (`predict_transition_state`) code paths. The inference path includes extra steps like `apply_spectator_constraints` and `enforce_triangle_inequality` that are missing from the training evaluation.
**Impact**: This is a critical correctness bug. The `PhysicsEaCalculator` is trained on geometries generated from one distribution of post-processing, but at inference time, it sees geometries from a different distribution. This mismatch can silently degrade the accuracy of the final activation energy (Ea) predictions.
**Solution**: Create a single, shared helper function that contains the complete and consistent set of post-processing steps. This function will be called from both the training and inference paths to guarantee they are identical.

### A2. Ensure Reproducible Augmentation
**Problem**: The coordinate augmentation noise added in `ReactionDataset.__getitem__` uses an unseeded random number generator (`np.random.randn`).
**Impact**: This makes training runs non-reproducible. Even with the same train/validation split, the model will see slightly different data in each run, leading to variance in final performance and making it difficult to reliably compare experiments.
**Solution**: Add a seed to the random number generator within `__getitem__`. A simple `np.random.seed(idx)` is a good first step. For multi-worker data loading, this will need to be evolved to a per-worker seeding strategy, but it is a crucial first step for reproducibility.

### A3. Clarify Activation Energy Definition
**Problem**: The calculation of the raw activation energy in `build_reaction_samples` is `ts_e["energy"] - max(r_e["energy"], p_e["energy"])`. This means Ea is calculated relative to the higher-energy state (reactant or product). For endothermic reactions, this corresponds to the reverse barrier, not the forward one.
**Impact**: This can be a source of confusion and may not align with the standard definition of a forward activation barrier.
**Solution**: Change the calculation to be `ts_e["energy"] - r_e["energy"]` to consistently represent the forward activation energy. Add a comment to clarify this definition.

---

## B. Code Simplification & Performance

### B1. Streamline Feature Generation
**Problem**: The `build_energy_features` function calculates a 20-dimensional feature vector, but only one feature (`de_rxn_signed`) is ever used by the `PhysicsEaCalculator`. The expensive bond-angle calculations are dead weight.
**Impact**: This adds unnecessary computational overhead to the data loading hot path for every single sample.
**Solution**: Remove the call to `build_energy_features`. Instead, directly calculate and store the single required value (`de_rxn_raw`) in each sample dictionary. This will significantly speed up the `build_reaction_samples` step.

### B2. Optimize Results Assembly Lookup
**Problem**: During the final results assembly in `train_pipeline`, a linear scan (`next(s for s in samples if s["rxn_id"] == rxn_id)`) is performed inside a loop, resulting in O(n²) complexity.
**Impact**: This can be slow for large datasets.
**Solution**: Pre-build a dictionary that maps `rxn_id` to the sample object. This will change the lookup from an O(n) scan to an O(1) dictionary access, speeding up the final results assembly.

### B3. Remove Redundant Code
**Problem**: There are minor instances of dead code, such as the initialization of `train_X, train_y = [], []` in `train_pipeline`, which is immediately overwritten.
**Impact**: Reduces code clarity.
**Solution**: Remove the unnecessary initializations.

---

## C. Documentation

### C1. Update README Architecture
**Problem**: The `README.md` is out of date. It describes an "Energy Head" that was part of a previous architecture (`psi_full_pipeline.py`). The current `psi_cloud_pipeline.py` uses a `PhysicsEaCalculator` instead.
**Impact**: New users or collaborators will be confused about how the model works.
**Solution**: Update the "Architecture" section of the `README.md` to accurately describe the current two-step process:
1. A geometry-only neural network (`PSI`) predicts the 3D coordinates of the TS.
2. A separate, post-hoc `PhysicsEaCalculator` uses these coordinates and classical chemical principles (Marcus Theory, Hammond Postulate) to calculate the Ea via a simple linear model.
