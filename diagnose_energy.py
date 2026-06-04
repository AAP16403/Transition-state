import json

import numpy as np

from sklearn.linear_model import Ridge

from sklearn.model_selection import cross_val_predict

from sklearn.metrics import mean_absolute_error

d = json.load(open("extracted_dataset.json"))

reactions = {}

for entry in d:

    parts = entry["filename"].split("/")

    if len(parts) < 3: continue

    rxn_id = parts[1]

    prefix = parts[2].lower()

    role = "r" if prefix.startswith("r") else "p" if prefix.startswith("p") else "ts" if prefix.startswith("ts") else None

    if not role:

        raise ValueError(f"Could not classify role for entry: {entry['filename']}")

    if rxn_id not in reactions: reactions[rxn_id] = {}

    reactions[rxn_id][role] = entry

features = []

targets = []

for rxn_id, roles in sorted(reactions.items()):

    if not ("r" in roles and "p" in roles and "ts" in roles):

        continue

    r_e = roles["r"]; p_e = roles["p"]; ts_e = roles["ts"]

    n = len(ts_e["atoms"])

    if n > 30: continue

    ea = (ts_e["energy"] - max(r_e["energy"], p_e["energy"])) * 627.509

    def get_coords(e):

        return np.array([[a["x"], a["y"], a["z"]] for a in e["atoms"]])

    c_R = get_coords(r_e)

    c_P = get_coords(p_e)

    feat = [n]

    atom_types = [a["atom"] for a in ts_e["atoms"]]

    for at in ["C", "H", "N", "O"]:

        feat.append(atom_types.count(at))

    diff = c_R - c_P

    diff_norms = np.linalg.norm(diff, axis=1)

    feat.extend([diff_norms.mean(), diff_norms.std(), diff_norms.max(), diff_norms.min()])

    def dist_stats(coords):

        n = len(coords)

        dists = []

        for i in range(n):

            for j in range(i+1, n):

                dists.append(np.linalg.norm(coords[i] - coords[j]))

        dists = np.array(dists)

        return [dists.mean(), dists.std(), dists.min(), dists.max()]

    feat.extend(dist_stats(c_R))

    feat.extend(dist_stats(c_P))

    feat.append(r_e["energy"])

    feat.append(p_e["energy"])

    feat.append(abs(r_e["energy"] - p_e["energy"]) * 627.509)                   

    features.append(feat)

    targets.append(ea)

X = np.array(features)

y = np.array(targets)

print(f"Dataset: {len(X)} reactions, {X.shape[1]} features")

print(f"Target Ea range: [{y.min():.1f}, {y.max():.1f}] kcal/mol")

y_pred = cross_val_predict(Ridge(alpha=1.0), X, y, cv=5)

mae = mean_absolute_error(y, y_pred)

corr = np.corrcoef(y, y_pred)[0, 1]

print(f"\n--- Ridge Regression (5-fold CV) ---")

print(f"MAE:  {mae:.2f} kcal/mol")

print(f"Corr: {corr:.4f}")

X_no_energy = X[:, :-3]

y_pred_ne = cross_val_predict(Ridge(alpha=1.0), X_no_energy, y, cv=5)

mae_ne = mean_absolute_error(y, y_pred_ne)

corr_ne = np.corrcoef(y, y_pred_ne)[0, 1]

print(f"\n--- Ridge (NO R/P energies, just geometry) ---")

print(f"MAE:  {mae_ne:.2f} kcal/mol")

print(f"Corr: {corr_ne:.4f}")

X_only_energy = X[:, -3:]

y_pred_oe = cross_val_predict(Ridge(alpha=1.0), X_only_energy, y, cv=5)

mae_oe = mean_absolute_error(y, y_pred_oe)

corr_oe = np.corrcoef(y, y_pred_oe)[0, 1]

print(f"\n--- Ridge (ONLY R/P energies) ---")

print(f"MAE:  {mae_oe:.2f} kcal/mol")

print(f"Corr: {corr_oe:.4f}")

X_geom_comp = X[:, :13]  

y_pred_gc = cross_val_predict(Ridge(alpha=1.0), X_geom_comp, y, cv=5)

mae_gc = mean_absolute_error(y, y_pred_gc)

corr_gc = np.corrcoef(y, y_pred_gc)[0, 1]

print(f"\n--- Ridge (Geometry + Composition only) ---")

print(f"MAE:  {mae_gc:.2f} kcal/mol")

print(f"Corr: {corr_gc:.4f}")

print("\n--- Feature importance (absolute correlation with Ea) ---")

feat_names = ["n_atoms", "n_C", "n_H", "n_N", "n_O",

              "diff_mean", "diff_std", "diff_max", "diff_min",

              "R_dist_mean", "R_dist_std", "R_dist_min", "R_dist_max",

              "P_dist_mean", "P_dist_std", "P_dist_min", "P_dist_max",

              "E_R", "E_P", "dE_rxn"]

for i, name in enumerate(feat_names):

    c = abs(np.corrcoef(X[:, i], y)[0, 1])

    print(f"  {name:>15}: {c:.4f}")
