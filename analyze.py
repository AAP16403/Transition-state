import json

with open('detailed_analysis.json', 'r') as f:
    reactions = json.load(f)

# Filter for validation split only
val_rxns = [r for r in reactions if r.get('split') == 'val']

if not val_rxns:
    print("No validation reactions found!")
    exit()

# Overall Metrics
val_ea_errors = [abs(r['Ea_true'] - r['Ea_pred']) for r in val_rxns]
val_ea_biases = [r['Ea_pred'] - r['Ea_true'] for r in val_rxns]
val_dist_maes = [r['dist_MAE'] for r in val_rxns]

val_ea_mae = sum(val_ea_errors) / len(val_ea_errors)
val_ea_bias = sum(val_ea_biases) / len(val_ea_biases)
val_dist_mae = sum(val_dist_maes) / len(val_dist_maes)

print('--- OVERALL METRICS ---')
print(f"Validation Ea MAE:  {val_ea_mae:.3f} kcal/mol")
print(f"Validation Ea Bias: {val_ea_bias:.3f} kcal/mol")
print(f"Validation Geometry MAE: {val_dist_mae:.5f} A")

# Ea Bins
bins = [(0, 20), (20, 40), (40, 60), (60, 80), (80, 100), (100, 120), (120, float('inf'))]
print('\n--- EA BINS ---')
for b_min, b_max in bins:
    bin_rxns = [r for r in val_rxns if b_min <= r['Ea_true'] < b_max]
    if not bin_rxns: continue
    bin_mae = sum(abs(r['Ea_true'] - r['Ea_pred']) for r in bin_rxns) / len(bin_rxns)
    bin_bias = sum(r['Ea_pred'] - r['Ea_true'] for r in bin_rxns) / len(bin_rxns)
    b_name = f"{b_min}-{b_max}" if b_max != float('inf') else f"{b_min}+"
    print(f"[{b_name}] Count: {len(bin_rxns)}, MAE: {bin_mae:.3f}, Bias: {bin_bias:.3f}")

# Geometry Bins
geom_bins = [(0, 0.05), (0.05, 0.10), (0.10, 0.20), (0.20, 0.30), (0.30, float('inf'))]
print('\n--- GEOM BINS ---')
for b_min, b_max in geom_bins:
    bin_rxns = [r for r in val_rxns if b_min <= r['dist_MAE'] < b_max]
    if not bin_rxns: continue
    bin_ea_mae = sum(abs(r['Ea_true'] - r['Ea_pred']) for r in bin_rxns) / len(bin_rxns)
    b_name = f"{b_min}-{b_max}" if b_max != float('inf') else f"{b_min}+"
    print(f"[{b_name}] Count: {len(bin_rxns)}, Ea_MAE: {bin_ea_mae:.3f}")

# Severe Errors
print('\n--- SEVERE ERRORS ---')
under_20 = len([r for r in val_rxns if (r['Ea_pred'] - r['Ea_true']) <= -20.0])
over_20 = len([r for r in val_rxns if (r['Ea_pred'] - r['Ea_true']) >= 20.0])
print(f"Under <= -20: {under_20}")
print(f"Over  >= +20: {over_20}")
