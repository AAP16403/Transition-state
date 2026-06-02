import json

d = json.load(open("extracted_dataset.json"))
print(f"Total entries: {len(d)}")
print(f"Sample keys: {list(d[0].keys())}")
print(f"Sample filename: {d[0]['filename']}")
atoms_counts = [len(e['atoms']) for e in d]
print(f"Atom count range: {min(atoms_counts)}-{max(atoms_counts)}")
print("First 10 filenames:")
for e in d[:10]:
    print(f"  {e['filename']}")

# Check triplet formation
reactions = {}
for entry in d:
    parts = entry["filename"].split("/")
    if len(parts) < 3:
        continue
    rxn_id = parts[1]
    prefix = parts[2].lower()
    role = "r" if prefix.startswith("r") else "p" if prefix.startswith("p") else "ts" if prefix.startswith("ts") else None
    if not role:
        continue
    if rxn_id not in reactions:
        reactions[rxn_id] = {}
    reactions[rxn_id][role] = entry

complete = sum(1 for r in reactions.values() if "r" in r and "p" in r and "ts" in r)
partial = sum(1 for r in reactions.values() if not ("r" in r and "p" in r and "ts" in r))
print(f"\nTotal reaction IDs: {len(reactions)}")
print(f"Complete triplets (R+P+TS): {complete}")
print(f"Incomplete: {partial}")

# Check energy distribution
energies = []
for rxn_id, roles in reactions.items():
    if "r" in roles and "p" in roles and "ts" in roles:
        ea = (roles["ts"]["energy"] - max(roles["r"]["energy"], roles["p"]["energy"])) * 627.509
        energies.append(ea)

if energies:
    import numpy as np
    ea = np.array(energies)
    print(f"\nActivation energy stats (kcal/mol):")
    print(f"  Mean: {ea.mean():.2f}")
    print(f"  Std:  {ea.std():.2f}")
    print(f"  Min:  {ea.min():.2f}")
    print(f"  Max:  {ea.max():.2f}")
    print(f"  Negative Ea count: {(ea < 0).sum()}")
