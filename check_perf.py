import json

import numpy as np

d = json.load(open("detailed_analysis.json"))

print(f"Total evaluated reactions: {len(d)}")

ea_errors = [r["Ea_error"] for r in d]

dist_maes = [r["dist_MAE"] for r in d]

ea_errors = np.array(ea_errors)

dist_maes = np.array(dist_maes)

print(f"\nEnergy Prediction Errors (kcal/mol):")

print(f"  Mean: {ea_errors.mean():.4f}")

print(f"  Std:  {ea_errors.std():.4f}")

print(f"  Max:  {ea_errors.max():.4f}")

print(f"\nDistance MAE (Angstrom):")

print(f"  Mean: {dist_maes.mean():.4f}")

print(f"  Std:  {dist_maes.std():.4f}")

print(f"  Max:  {dist_maes.max():.4f}")

ea_trues = [r["Ea_true"] for r in d]

ea_preds = [r["Ea_pred"] for r in d]

corr = np.corrcoef(ea_trues, ea_preds)[0,1]

print(f"\nEa Correlation (true vs pred): {corr:.4f}")

worst = sorted(d, key=lambda x: x["dist_MAE"], reverse=True)[:5]

print("\nWorst 5 by distance MAE:")

for w in worst:

    print(f"  {w['rxn_id']}: dist_MAE={w['dist_MAE']:.4f}, Ea_err={w['Ea_error']:.2f}, n_atoms={w['n_atoms']}")
