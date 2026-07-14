# PSI Swarm MoE: Failure Mode & Bottleneck Analysis Report

This document outlines the core failure modes, bottleneck diagnostics, and dataset limitations identified during the Phase 2 training of the PSI Swarm Mixture-of-Experts (MoE) architecture.

---

## 1. The "Hyper-Confidence" Trap (Loss Function Collapse)
**Symptom:** During Swarm training, the Training Loss becomes massively negative, while the Validation Loss explodes. The model achieves an incredible Training Ea MAE (~5.25) but completely fails to generalize (Validation Ea MAE stuck > 8.0).

**Root Cause:** The Gaussian Negative Log-Likelihood (NLL) loss function is mathematically defined roughly as:
`Loss = (Error² / Variance) + log(Variance)`

Because the 5.5-million parameter Swarm MoE is highly expressive, it rapidly memorizes the exact answers for the 14,000 training reactions (driving `Error²` to near 0). The optimizer realizes it can force the loss into deep negative values by predicting an infinitely small Variance (near 100% certainty). When this hyper-confident model encounters an unseen validation reaction, its `Error²` is non-zero. Because it is dividing that error by a near-zero variance, the penalty mathematically explodes.

**Attempted Remediation Parameters:**
* `ea_log_var_min`: Increased from `-5.0` to `-1.5` to place a floor on uncertainty.
* `ea_detach_during_warmup`: Set to `True` to prevent the Ea head from destroying the EGNN geometric backbone during early-stage hyper-confidence.

---

## 2. The High-Barrier Activation Energy Cliff
**Symptom:** The model performs exceptionally well on standard reactions but catastrophically underpredicts rare, complex reactions.

**Root Cause:** The model learns the statistical mean of the dataset and aggressively regresses outliers back toward that mean. The model completely loses physical accuracy on reactions exceeding 80 kcal/mol.

**Baseline Validation Breakdown (B97D3 Dataset):**
| True Ea Bin (kcal/mol) | Validation Count | Mean Absolute Error (MAE) | Bias (Direction) | Underprediction Rate |
|---|---:|---:|---:|---:|
| `0 - 20` | 403 | **3.153** | +0.764 | 46.2% |
| `20 - 80` | 1,100 | **~5.100** | ~ -1.500 | ~64.0% |
| `80 - 100` | 95 | **6.889** | -3.231 | 77.9% |
| `100 - 120` | 27 | **17.942** | -15.724 | 81.5% |
| `120+` | 4 | **34.925** | -34.925 | 100.0% |

**Attempted Remediation Parameters:**
* `ea_tail_weighting_enabled`: `True`
* `ea_tail_weight_bins`: `[80, 100, 120]`
* `ea_tail_weight_values`: `[1.0, 1.5, 2.0, 2.5]`
* *Status:* Failed. Loss weighting causes gradient shocks that destabilize the EGNN rather than teaching it geometric nuances.

---

## 3. Dataset Imbalance & Capacity Constraints
**Symptom:** Piecewise loss weighting and MoE top-K routing fail to rescue the high-barrier predictions.

**Root Cause (Data Volume):** The `b97d3` dataset contains only 16,365 valid reactions. 
* A 5.5M parameter network is too large for 16k reactions, leading to unavoidable memorization (overfitting).
* The extreme high-barrier reactions (>120 kcal/mol) make up a microscopic **3.0%** of the dataset. The MoE Router starves because there are not enough examples to reliably train a dedicated "High-Barrier Expert".

**Dataset Analysis Comparison:**
* **B97D3 (Current):** Mean Ea = 75.24 kcal/mol. Only **37.4%** of the dataset is >80 kcal/mol.
* **WB97XD3 (New Alternative):** Mean Ea = 83.39 kcal/mol. Over **53.2%** of the dataset is >80 kcal/mol.

---

## 4. Proposed Architectural & Data Solutions
To fix these failures, the codebase must shift away from mathematical loss-hacking and towards robust data engineering:

1. **Migrate to WB97XD3 or RGD1:** Switch the data source to `wb97xd3` to achieve natural high-barrier oversampling, or scale up to the `RGD1` dataset (~177k reactions) to fully saturate the Swarm MoE's parameter capacity.
2. **Error-Tail Oversampling (Design 2C):** If remaining on `b97d3`, abandon piecewise loss multipliers and physically duplicate the >100 kcal/mol reactions within the PyTorch `DataLoader` to force geometric exposure.
3. **Hardcoded Expert Isolation:** Override the dynamic MoE router and explicitly hardcode Expert 5 to *only* receive gradients from reactions where `True Ea > 80`.
4. **Early Stochastic Weight Averaging (SWA):** The initial `swa_start` was set to Epoch 200, which is useless because the Gaussian NLL collapse triggers massive overfitting by Epoch 60. Moving `swa_start` to Epoch 60 will force the weights to average out the hyper-confidence spikes exactly when the model attempts to memorize the training data.

---

## 5. The Geometric "Hinge" Blind Spot (Global Rotation Failure)
**Symptom:** On complex, risky reactions, the predicted Transition State has accurate local covalent bonds, but the moving fragments are rotated at wildly incorrect global angles relative to the rigid backbone.

**Root Cause:** The geometry loss function relies on Inverse Distance Weighting (`dist_weights = 1.0 / (DTS * m2d + 1.0)`). This brilliantly solves the "fragment melting" issue by forcing the neural network to dedicate 99% of its gradient to atoms that are right next to each other, but completely ignores long-range distances.
However, global rotation and dihedral angles are dictated entirely by the long-range "cross-terms" (the distance between a moving atom on fragment A, and a rigid spectator atom on fragment B). Because the loss function is mathematically "blind" to these long-range hinge distances, the EGNN learns that it can freely rotate large fragments in 3D space without suffering any penalty, effectively destroying the steric accuracy required by the Ea Head.

**Attempted Remediation Parameters:**
* Engineered 8 new 28D global features (`_rc_angle_features`) explicitly tracking dihedrals and pyramidalization to give the Ea Head angle awareness.
* *Future Fix Required:* The `l_geom` loss must be updated to explicitly weight the cross-distances between the `active_mask` and the `spectator_mask`.

---

## 6. Gradient Pollution (Signal-to-Noise Drowning)
**Symptom:** The EGNN Swarm struggles to refine the exact reactive center of large molecules, wasting parameters on the unreacting backbone.

**Root Cause:** In a standard 30-atom reaction, only ~4 atoms are actively breaking or forming bonds. The other 26 atoms are rigid spectators. Because the EGNN computes a fully connected graph, the geometry loss function is flooded by the static distances of the 26 spectator atoms. The critical learning signal (the 4 active atoms) is completely drowned out by the noise of the stagnant backbone.
This severely exacerbates the MoE Swarm's overfitting, as the experts take the "lazy" route of memorizing the rigid backbones of the 16k training samples rather than learning the complex quantum mechanics of the reactive centers.

**Proposed Solution:**
* **Active-Atom Gradient Masking:** Implement a gradient mask inside `run_epoch` that severely mutes the loss weight on pure spectator-to-spectator distances. This will force the EGNN to spend 90% of its learning capacity strictly on refining the active reactive zone.
