import json
import pickle
import os
from collections import defaultdict, Counter

def convert_json_to_mace_cache(json_path, out_pkl_path, max_limit=0):
    print(f"Loading {json_path}...")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    print(f"Loaded {len(data)} log entries. Grouping by reaction...")
    rxns = defaultdict(dict)
    
    for entry in data:
        fname = entry.get('filename', '')
        if not fname:
            continue
            
        parts = fname.replace('\\', '/').split('/')
        if len(parts) < 2:
            continue
            
        folder = parts[-2]
        file_basename = parts[-1].lower()
        
        if file_basename.startswith('r'):
            rxns[folder]['r'] = entry
        elif file_basename.startswith('p'):
            rxns[folder]['p'] = entry
        elif file_basename.startswith('ts'):
            rxns[folder]['ts'] = entry
            
    print(f"Grouped into {len(rxns)} potential reactions. Formatting for MACE...")
    samples = []
    
    HARTREE_TO_KCAL = 627.509
    
    for folder, logs in rxns.items():
        if 'r' in logs and 'p' in logs and 'ts' in logs:
            r = logs['r']
            p = logs['p']
            ts = logs['ts']
            
            j1_atoms = r['atoms']
            j2_atoms = p['atoms']
            j3_atoms = ts['atoms']
            
            j1_energy = r['energy']
            j2_energy = p['energy']
            j3_energy = ts['energy']
            
            if len(j1_atoms) != len(j2_atoms) or len(j1_atoms) != len(j3_atoms):
                continue
                
            syms1 = [a['atom'] for a in j1_atoms]
            
            true_ea = (j3_energy - j1_energy) * HARTREE_TO_KCAL
            reaction_enthalpy = (j2_energy - j1_energy) * HARTREE_TO_KCAL
            
            samples.append({
                "folder_name": folder,
                "j1_atoms": j1_atoms,
                "j2_atoms": j2_atoms,
                "j3_atoms": j3_atoms,
                "j3_forces": [],
                "has_forces": False,
                "true_ea": float(true_ea),
                "reaction_enthalpy": float(reaction_enthalpy),
                "atom_counts": dict(Counter(syms1)),
                "atom_types": syms1
            })
            
            if max_limit > 0 and len(samples) >= max_limit:
                break
                
    print(f"Successfully formatted {len(samples)} valid reaction triplets for MACE.")
    print(f"Saving to {out_pkl_path}...")
    
    with open(out_pkl_path, 'wb') as f:
        pickle.dump(samples, f)
        
    print("Done!")

if __name__ == "__main__":
    convert_json_to_mace_cache("extracted_b97d3.json", "parsed_dataset_cache_nocsv_all.pkl")
