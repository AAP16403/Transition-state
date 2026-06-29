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
python psi_full_pipeline.py train --extract-limit 16000
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

## Architecture
The core architecture (`PSICore`) uses:
- GRU layers to contextualize pair-wise distances.
- Transformer encoder layers to capture global structural features.
- A Geometry Head to predict the TS distance matrix.
- An Energy Head with cross-attention to predict the activation energy (Ea).
