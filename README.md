# PSI Transition State Prediction

PSI is a deep learning pipeline for predicting the 3D geometries and activation energies (Ea) of chemical reaction transition states from only the reactant and product geometries.

## Overview
This repository contains the full end-to-end pipeline:
1. **Data Extraction**: Extracts XYZ coordinates and energies from Gaussian `.log` files.
2. **Training Pipeline**: Trains a state-of-the-art hybrid deep learning model to predict transition state geometries and activation energies.
3. **Visualization**: Generates an interactive HTML dashboard using `plotly` to evaluate the model's predictions (activation energy parity plot, distance MAE histograms, etc.).

## Installation
Ensure you have Python 3.8+ and install the dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training
To train the model on your dataset of Gaussian `.log` files:
```bash
python psi_full_pipeline.py train --extract-limit 30000 --target-reactions 10000 --force-extract
```
This extracts the files, processes the dataset, splits it into training and validation sets, and trains the model. The best model weights will be saved as `psi_final.pt`.

### Prediction
To predict a transition state given a new reactant and product:
```bash
python psi_full_pipeline.py predict -r reactant.log -p product.log -o prediction.json --xyz predicted_ts.xyz
```
This requires a trained `psi_final.pt` model.

### Visualization
After training or prediction, you can generate a performance dashboard:
```bash
python psi_full_pipeline.py dashboard
```
This will read the `detailed_analysis.json` generated during training/evaluation and create an interactive `psi_results_dashboard.html` that can be opened in any web browser.

## Architecture (Current Version)
The pipeline is currently powered by a highly optimized, physics-informed Equivariant Graph Neural Network (EGNN) architecture designed specifically for the complex RGD1 Transition State dataset (40k reactions). The architecture is divided into three primary components:

### 1. SE(3) Equivariant Graph Backbone (Geometry Representation)
The core geometry predictor uses a stacked Equivariant Graph Neural Network.
- **Message Passing**: The network passes messages between atoms using invariant pairwise distances ($||x_i - x_j||^2$) and node-level features (atomic numbers, Pauling electronegativities, and covalent radii).
- **Coordinate Updates**: The 3D coordinates ($x_i$) are updated equivariantly by computing a weighted sum of the relative displacement vectors $(x_i - x_j)$ during each layer. This guarantees that rotating or translating the input reactant/product pair perfectly rotates/translates the predicted transition state.
- **Z-Matrix Invariance**: By focusing solely on invariant distances and equivariant displacements, the model bypasses the need for arbitrary Z-matrix alignments.

### 2. The Activation Energy (Ea) Cross-Attention Head
Instead of predicting Ea from a flattened geometry array, the network uses a Latent Cross-Attention mechanism.
- **Detached Features**: The final geometry node features ($h_{ts}$) are explicitly detached (`h.detach()`) before being passed into the Ea head. This is critical because geometry optimization and thermodynamic scalar optimization have conflicting gradient trajectories. Detaching them allows the Ea head to act as a highly specialized regressor without physically "melting" the underlying graph structure.
- **Global Context**: The Ea head cross-attends to the node features, learning which specific atomic clusters (reaction centers) dominate the energetic barrier.

### 3. Physics-Informed Loss Functions & Risk Penalties
The objective function is a highly engineered, multi-objective formulation:
- **Huber (Smooth L1) Geometry Loss**: Replaced Mean Squared Error (MSE) to prevent outlier reactions from causing infinite gradient spikes, and prevents the "variance collapse" typical of Gaussian NLL.
- **Inverse-Distance Weighting**: Standard graph networks suffer from "fragment melting" where they ignore local chemistry to minimize global distance errors. We apply a $1.0 / (D_{TS} + 1.0)$ multiplier to the loss, mathematically forcing the network to prioritize short-range chemical bonds over long-range spectator atoms.
- **Active-Site Risk Penalty**: The network dynamically generates a `risk_pair_mask` identifying actively changing bonds—specifically highly strained N-N, N-O, and O-O bonds. These active pairs receive a massive artificial gradient multiplier, forcing the network to memorize delicate quantum chemistry rather than cheating the global loss.

### 4. Generalization & Throughput Optimization
- **Dataset Pre-caching**: Distance matrices, feature embeddings, and reaction masks are pre-computed in CPU RAM and streamed to the GPU via `samples_cache_rgd1.pkl`, yielding a >11x throughput increase.
- **Cosine Annealing & SWA**: To combat generalization gaps on complex reactions, the learning rate is aggressively annealed, followed by Stochastic Weight Averaging (SWA) in the final phases of training to find flat, highly generalized minima in the loss landscape.
