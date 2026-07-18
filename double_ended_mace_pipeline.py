import os
import torch
torch.manual_seed(42)
import numpy as np
np.random.seed(42)

import torch
try:
    torch.set_num_threads(4)         
    torch.set_num_interop_threads(4)
except RuntimeError:
    pass

import re
import json
import argparse
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

# PyTorch 2.6 compatibility patch for e3nn
try:
    import torch.serialization
    torch.serialization.add_safe_globals([slice])
except Exception:
    pass

from e3nn import o3
from mace.modules.models import MACE
from torch_geometric.nn import MessagePassing, global_mean_pool, global_add_pool
from mace.modules.blocks import RealAgnosticResidualInteractionBlock, RealAgnosticInteractionBlock

# ----------------------------------------------------
# 1. Physics Constants and Helper Functions
# ----------------------------------------------------
HARTREE_TO_KCAL = 627.509

def kabsch_rotation(P, Q):
    """Optimal PROPER rotation (det = +1) aligning centered P onto centered Q."""
    P_centered = P - P.mean(axis=0)
    Q_centered = Q - Q.mean(axis=0)
    C = P_centered.T @ Q_centered
    V, _, W = np.linalg.svd(C)
    d = np.sign(np.linalg.det(V @ W))
    if d == 0.0: d = 1.0
    D = np.diag([1.0, 1.0, d])
    return V @ D @ W

def kabsch_align(P, Q):
    P_centered = P - P.mean(axis=0)
    R = kabsch_rotation(P, Q)
    return P_centered @ R + Q.mean(axis=0)

def write_xyz(path, atom_types, coords, comment):
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"{len(atom_types)}\n")
        f.write(f"{comment}\n")
        for atom, (x, y, z) in zip(atom_types, coords):
            f.write(f"{atom:<2} {x: .8f} {y: .8f} {z: .8f}\n")

SYMBOL_TO_Z = {"H": 1, "He": 2, "Li": 3, "Be": 4, "B": 5, "C": 6, "N": 7, "O": 8, "F": 9, "Ne": 10, "Na": 11, "Mg": 12, "Al": 13, "Si": 14, "P": 15, "S": 16, "Cl": 17, "Ar": 18, "K": 19, "Ca": 20, "Sc": 21, "Ti": 22, "V": 23, "Cr": 24, "Mn": 25, "Fe": 26, "Co": 27, "Ni": 28, "Cu": 29, "Zn": 30, "Ga": 31, "Ge": 32, "As": 33, "Se": 34, "Br": 35, "Kr": 36, "Rb": 37, "Sr": 38, "Y": 39, "Zr": 40, "Nb": 41, "Mo": 42, "Tc": 43, "Ru": 44, "Rh": 45, "Pd": 46, "Ag": 47, "Cd": 48, "In": 49, "Sn": 50, "Sb": 51, "Te": 52, "I": 53, "Xe": 54, "Cs": 55, "Ba": 56, "La": 57, "Ce": 58, "Pr": 59, "Nd": 60, "Pm": 61, "Sm": 62, "Eu": 63, "Gd": 64, "Tb": 65, "Dy": 66, "Ho": 67, "Er": 68, "Tm": 69, "Yb": 70, "Lu": 71, "Hf": 72, "Ta": 73, "W": 74, "Re": 75, "Os": 76, "Ir": 77, "Pt": 78, "Au": 79, "Hg": 80, "Tl": 81, "Pb": 82, "Bi": 83, "Po": 84, "At": 85, "Rn": 86, "Fr": 87, "Ra": 88, "Ac": 89, "Th": 90, "Pa": 91, "U": 92, "Np": 93, "Pu": 94, "Am": 95, "Cm": 96, "Bk": 97, "Cf": 98, "Es": 99, "Fm": 100, "Md": 101, "No": 102, "Lr": 103, "Rf": 104, "Db": 105, "Sg": 106, "Bh": 107, "Hs": 108, "Mt": 109, "Ds": 110, "Rg": 111, "Cn": 112, "Nh": 113, "Fl": 114, "Mc": 115, "Lv": 116, "Ts": 117, "Og": 118}

PAULING_EN = {
    1: 2.20, 5: 2.04, 6: 2.55, 7: 3.04, 8: 3.44, 9: 3.98,
    14: 1.90, 15: 2.19, 16: 2.58, 17: 3.16, 35: 2.96, 53: 2.66
}

COVALENT_RADII = {
    1: 0.31, 5: 0.84, 6: 0.76, 7: 0.71, 8: 0.66, 9: 0.57,
    14: 1.11, 15: 1.07, 16: 1.05, 17: 1.02, 35: 1.20, 53: 1.39
}

def compute_reaction_features(j1_atoms, j2_atoms, delta_e_kcal):
    """
    9-dim feature vector describing the reaction path transformation.
    """
    c_R = np.array([[a["x"], a["y"], a["z"]] for a in j1_atoms], dtype=np.float64)
    c_P = np.array([[a["x"], a["y"], a["z"]] for a in j2_atoms], dtype=np.float64)
    c_R_c = c_R - c_R.mean(axis=0)
    c_P_aligned = kabsch_align(c_P, c_R)
    c_P_c = c_P_aligned - c_P_aligned.mean(axis=0)

    disp = np.linalg.norm(c_P_c - c_R_c, axis=1)
    max_d, mean_d, std_d = float(disp.max()), float(disp.mean()), float(disp.std())

    rg_R = float(np.sqrt(np.mean(np.sum(c_R_c ** 2, axis=1))))
    rg_P = float(np.sqrt(np.mean(np.sum(c_P_c ** 2, axis=1))))

    n = len(j1_atoms)
    if n > 1:
        dist_R = np.linalg.norm(c_R[:, None, :] - c_R[None, :, :], axis=-1)
        dist_P = np.linalg.norm(c_P[:, None, :] - c_P[None, :, :], axis=-1)
        
        atomic_numbers = [SYMBOL_TO_Z[a["atom"]] for a in j1_atoms]
        radii = np.array([COVALENT_RADII.get(z, 0.75) for z in atomic_numbers])
        radii_sum = radii[:, None] + radii[None, :]
        
        iu = np.triu_indices(n, k=1)
        bonded_R = dist_R[iu] < (1.3 * radii_sum[iu])
        bonded_P = dist_P[iu] < (1.3 * radii_sum[iu])
        bond_change_frac = float(np.mean(np.logical_xor(bonded_R, bonded_P)))
    else:
        bond_change_frac = 0.0

    return np.array([
        delta_e_kcal, max_d, mean_d, std_d,
        rg_R, rg_P, rg_P - rg_R,
        float(n), bond_change_frac
    ], dtype=np.float32)

# ----------------------------------------------------
# 2. Q-Chem Triplet Log Parsing
# ----------------------------------------------------
def parse_true_qchem_reaction(rxn_folder, stats=None):
    def reject(reason):
        if stats is not None: stats[reason] = stats.get(reason, 0) + 1
        return None

    folder_name = os.path.basename(rxn_folder)
    idx_match = re.search(r'rxn(\d+)', folder_name)
    if not idx_match: return reject("folder_not_rxn_format")
    
    csv_idx = int(idx_match.group(1))

    idx_str = f"{csv_idx:06d}"
    r_path = os.path.join(rxn_folder, f"r{idx_str}.log")
    p_path = os.path.join(rxn_folder, f"p{idx_str}.log")
    ts_path = os.path.join(rxn_folder, f"ts{idx_str}.log")

    if not (os.path.exists(r_path) and os.path.exists(p_path) and os.path.exists(ts_path)):
        return reject("missing_log_files")

    def read_file(path):
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()

    r_content = read_file(r_path)
    p_content = read_file(p_path)
    ts_content = read_file(ts_path)

    def parse_qchem_gradient(job_content):
        matches = list(re.finditer(r"Gradient of SCF Energy\n", job_content))
        if not matches: return None
        last_match = matches[-1]
        lines = job_content[last_match.end():].splitlines()
        atom_forces = {}
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if "Max gradient component" in line or "Geometry Optimization" in line: break
            parts = line.split()
            if len(parts) > 0 and all(p.isdigit() for p in parts):
                indices = [int(p) - 1 for p in parts]
                l1 = lines[i+1].strip().split()[1:]
                l2 = lines[i+2].strip().split()[1:]
                l3 = lines[i+3].strip().split()[1:]
                for col_idx, atom_idx in enumerate(indices):
                    if atom_idx not in atom_forces: atom_forces[atom_idx] = [0.0, 0.0, 0.0]
                    atom_forces[atom_idx][0] = -float(l1[col_idx])
                    atom_forces[atom_idx][1] = -float(l2[col_idx])
                    atom_forces[atom_idx][2] = -float(l3[col_idx])
                i += 3
            i += 1
        if not atom_forces: return None
        n_atoms = max(atom_forces.keys()) + 1
        force_array = np.zeros((n_atoms, 3))
        for idx, f in atom_forces.items(): force_array[idx] = f
        return force_array

    def parse_job_section(job_content):
        atoms, energy = [], None
        latest_atoms = []
        lines = job_content.splitlines()
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if 'Standard Nuclear Orientation' in line:
                current_atoms = []
                i += 3
                while i < len(lines) and not lines[i].strip().startswith('---'):
                    parts = lines[i].split()
                    if len(parts) == 5:
                        current_atoms.append({
                            'atom': parts[1],
                            'x': float(parts[2]), 'y': float(parts[3]), 'z': float(parts[4])
                        })
                    i += 1
                latest_atoms = current_atoms
            elif 'Total energy in the final basis set =' in line or 'SCF   energy in the final basis set =' in line:
                energy = float(line.split()[-1])
                atoms = latest_atoms
            i += 1
        return atoms, energy

    j1_atoms, j1_energy = parse_job_section(r_content)
    j2_atoms, j2_energy = parse_job_section(p_content)
    j3_atoms, j3_energy = parse_job_section(ts_content)
    j3_forces = parse_qchem_gradient(ts_content)
    has_forces = True
    if j3_forces is None: 
        j3_forces = np.zeros((len(j3_atoms), 3))
        has_forces = False

    if not j1_atoms or j1_energy is None or not j3_atoms or j3_energy is None:
        return reject("missing_reactant_or_ts_data")
    if not j2_atoms or j2_energy is None:
        return reject("missing_product_data")

    n1, n2, n3 = len(j1_atoms), len(j2_atoms), len(j3_atoms)
    if n1 != n2 or n1 != n3:
        return reject("atom_count_mismatch")
    syms1 = [a['atom'] for a in j1_atoms]
    if syms1 != [a['atom'] for a in j2_atoms] or syms1 != [a['atom'] for a in j3_atoms]:
        return reject("atom_order_mismatch")

    true_ea = (j3_energy - j1_energy) * HARTREE_TO_KCAL
    reaction_enthalpy = (j2_energy - j1_energy) * HARTREE_TO_KCAL

    return {
        "folder_name": folder_name,
        "j1_atoms": j1_atoms,
        "j2_atoms": j2_atoms,
        "j3_atoms": j3_atoms,
        "j3_forces": j3_forces,
        "has_forces": has_forces,
        "true_ea": float(true_ea),
        "reaction_enthalpy": float(reaction_enthalpy),
        "atom_counts": dict(Counter(syms1)),
        "atom_types": syms1
    }

# ----------------------------------------------------
# 3. Double-Ended Reaction Aware Dataset
# ----------------------------------------------------
def iterative_steric_clamp(coords, atomic_numbers, max_iters=25, lr=0.5):
    c = coords.copy()
    for _ in range(max_iters):
        dist_mid_orig = np.linalg.norm(c[:, None, :] - c[None, :, :], axis=-1)
        np.fill_diagonal(dist_mid_orig, 999.0)
        delta_c = np.zeros_like(c)
        has_clash = False
        
        i_idx, j_idx = np.where(np.triu(dist_mid_orig < 2.5, k=1)) # Fast pre-filter
        
        for i, j in zip(i_idx, j_idx):
            dist = dist_mid_orig[i, j]
            z_i, z_j = atomic_numbers[i], atomic_numbers[j]
            r_i, r_j = COVALENT_RADII.get(z_i, 0.75), COVALENT_RADII.get(z_j, 0.75)
            
            trigger = 0.55 * (r_i + r_j)
            target = 0.75 * (r_i + r_j)
            
            if dist < trigger and dist > 1e-4:
                has_clash = True
                vec = (c[i] - c[j]) / dist
                push = (target - dist) * 0.5 * vec
                delta_c[i] += push
                delta_c[j] -= push
                
        c += delta_c * lr
        if not has_clash:
            break
    return c

class MACEReactionAwareDataset(torch.utils.data.Dataset):
    def __init__(self, samples, z_table_elements, cutoff=5.0, scaler_mean=None, scaler_scale=None):
        self.samples = samples
        self.z_table = z_table_elements
        self.cutoff = cutoff
        self.scaler_mean = np.array(scaler_mean, dtype=np.float32) if scaler_mean is not None else None
        self.scaler_scale = np.array(scaler_scale, dtype=np.float32) if scaler_scale is not None else None

        for s in self.samples:
            c_R = np.array([[a["x"], a["y"], a["z"]] for a in s["j1_atoms"]], dtype=np.float32)
            c_P = np.array([[a["x"], a["y"], a["z"]] for a in s["j2_atoms"]], dtype=np.float32)
            c_TS = np.array([[a["x"], a["y"], a["z"]] for a in s["j3_atoms"]], dtype=np.float32)

            c_R_c = c_R - c_R.mean(axis=0)
            c_P_c = kabsch_align(c_P, c_R_c)
            c_P_c = c_P_c - c_P_c.mean(axis=0)
            
            c_midpoint = (c_R_c + c_P_c) / 2.0

            # Get atomic numbers for dynamic steric radii
            atom_types = [a["atom"] for a in s["j1_atoms"]]
            s["atom_types"] = atom_types
            atomic_numbers = [SYMBOL_TO_Z[a] for a in atom_types]
            
            # STERIC MIDPOINT CLAMPING (Permutation-Invariant, Iterative)
            c_midpoint = iterative_steric_clamp(c_midpoint, atomic_numbers)
                        
            # PHYSICS PRIOR: Calculate Pauling Electronegativity and Covalent Radii
            # PRECOMPUTE GRAPH CONNECTIVITY
            dist_mid = np.linalg.norm(c_midpoint[:, None, :] - c_midpoint[None, :, :], axis=-1)
            dist_R = np.linalg.norm(c_R_c[:, None, :] - c_R_c[None, :, :], axis=-1)
            dist_P = np.linalg.norm(c_P_c[:, None, :] - c_P_c[None, :, :], axis=-1)
            mask = (dist_mid <= self.cutoff) | (dist_R <= self.cutoff) | (dist_P <= self.cutoff)
            np.fill_diagonal(mask, False)
            edges = np.vstack(np.where(mask))
            s["edge_index"] = torch.tensor(edges, dtype=torch.long).contiguous() if edges.size > 0 else torch.empty((2, 0), dtype=torch.long)
            
            # SPECTATOR MASKING (Reaction Weights)
            atom_disp = np.linalg.norm(c_P_c - c_R_c, axis=-1)
            reaction_weight = np.clip(atom_disp / 0.5, 0.1, 1.0)
            s["reaction_weight"] = reaction_weight
            
            R_TS = kabsch_rotation(c_TS, c_midpoint)
            c_TS_aligned = (c_TS - c_TS.mean(axis=0)) @ R_TS + c_midpoint.mean(axis=0)
            c_TS_centered = c_TS_aligned - c_TS_aligned.mean(axis=0)
            
            forces = np.array(s.get("j3_forces", np.zeros((len(c_TS), 3))), dtype=np.float32)
            forces_aligned = (forces * 1185.82) @ R_TS
            
            s["c_midpoint"] = c_midpoint
            s["c_TS_centered"] = c_TS_centered
            s["forces_aligned"] = forces_aligned
            s["c_R_c"] = c_R_c
            s["c_P_c"] = c_P_c
            
            rxn_feats = compute_reaction_features(s["j1_atoms"], s["j2_atoms"], s["reaction_enthalpy"])
            if self.scaler_mean is not None and self.scaler_scale is not None:
                rxn_feats = (rxn_feats - self.scaler_mean) / self.scaler_scale
            s["rxn_feats"] = rxn_feats

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        n = len(s["j1_atoms"])

        c_midpoint = s["c_midpoint"]
        c_TS_centered = s["c_TS_centered"]
        forces_aligned = s["forces_aligned"]
        c_R_c = s["c_R_c"]
        c_P_c = s["c_P_c"]
        rxn_feats = s["rxn_feats"]
        edge_index = s["edge_index"]

        node_attrs = np.zeros((n, len(self.z_table) + 2), dtype=np.float32)
        atomic_numbers = []
        for i, a in enumerate(s["atom_types"]):
            z = SYMBOL_TO_Z[a]
            if z not in self.z_table:
                raise ValueError(f"Error: Element {a} (Z={z}) was not seen during training! Check your dataset.")
            atomic_numbers.append(z)
            node_attrs[i, self.z_table.index(z)] = 1.0
            node_attrs[i, -2] = PAULING_EN.get(z, 2.0)
            node_attrs[i, -1] = COVALENT_RADII.get(z, 1.0)

        return Data(
            positions=torch.tensor(c_midpoint, dtype=torch.float32),
            pos_R=torch.tensor(c_R_c, dtype=torch.float32),
            pos_P=torch.tensor(c_P_c, dtype=torch.float32),
            z=torch.tensor(atomic_numbers, dtype=torch.long),
            node_attrs=torch.tensor(node_attrs, dtype=torch.float32),
            edge_index=edge_index,
            shifts=torch.zeros((edge_index.shape[1], 3), dtype=torch.float32),
            unit_shifts=torch.zeros((edge_index.shape[1], 3), dtype=torch.float32),
            atomic_numbers=torch.tensor(atomic_numbers, dtype=torch.long),
            cell=torch.zeros(3, 3, dtype=torch.float32).unsqueeze(0),
            pbc=torch.tensor([False, False, False], dtype=torch.bool).unsqueeze(0),
            
            x_TS_true=torch.tensor(c_TS_centered, dtype=torch.float32),
            true_forces=torch.tensor(forces_aligned, dtype=torch.float32),
            reaction_feats=torch.tensor(rxn_feats, dtype=torch.float32).unsqueeze(0),
            reaction_weight=torch.tensor(s["reaction_weight"], dtype=torch.float32),
            target_energy=torch.tensor([s["true_ea"]], dtype=torch.float32),
            norm_energy=torch.tensor([s.get("norm_energy", 0.0)], dtype=torch.float32),
            reaction_enthalpy=torch.tensor([s["reaction_enthalpy"]], dtype=torch.float32),
            has_forces=torch.tensor([s.get("has_forces", False)], dtype=torch.bool)
        )

# ----------------------------------------------------
# 4. Reaction-Aware MACE TS Estimator Model
# ----------------------------------------------------
class MACEReactionAwareTSModel(nn.Module):
    def __init__(self, z_table_elements, rxn_feat_dim=9, dropout=0.0, avg_num_neighbors=8.0, e_mean=0.0, e_std=1.0, dh_mean=0.0, dh_std=1.0):
        super().__init__()
        padded_z_table = z_table_elements + [1001, 1002] # Append 2 unique dummy elements for Physics Priors
        
        # The Actor: Dedicated lightweight network (32x capacity, 2 layers) for predicting 3D displacements
        self.mace_geom = MACE(
            r_max=5.0, num_bessel=8, num_polynomial_cutoff=5, max_ell=3,
            interaction_cls_first=RealAgnosticInteractionBlock,
            interaction_cls=RealAgnosticResidualInteractionBlock,
            num_interactions=2, num_elements=len(padded_z_table),
            hidden_irreps=o3.Irreps("32x0e + 32x1o + 32x2e + 32x3o"), MLP_irreps=o3.Irreps("32x0e"),
            atomic_energies=np.zeros(len(padded_z_table), dtype=float), avg_num_neighbors=avg_num_neighbors,
            atomic_numbers=padded_z_table, correlation=3, gate=torch.nn.functional.silu,
        )
        self.coord_proj = o3.Linear(o3.Irreps("32x0e + 32x1o + 32x2e + 32x3o + 32x0e"), o3.Irreps("1x1o"))
        
        # The Critic: Dedicated heavyweight network (64x capacity, 2 layers) for evaluating energy & forces
        self.mace_ener = MACE(
            r_max=5.0, num_bessel=8, num_polynomial_cutoff=5, max_ell=3,
            interaction_cls_first=RealAgnosticInteractionBlock,
            interaction_cls=RealAgnosticResidualInteractionBlock,
            num_interactions=2, num_elements=len(padded_z_table),
            hidden_irreps=o3.Irreps("64x0e + 64x1o + 64x2e + 64x3o"), MLP_irreps=o3.Irreps("64x0e"),
            atomic_energies=np.zeros(len(padded_z_table), dtype=float), avg_num_neighbors=avg_num_neighbors,
            atomic_numbers=padded_z_table, correlation=3, gate=torch.nn.functional.silu,
        )
        
        # 1. Physics Baseline Parameters (Dynamic Marcus Theory Reorganization Energy)
        self.lambda_mlp = nn.Sequential(
            nn.Linear(9, 32),
            nn.GELU(),
            nn.Linear(32, 1)
        )
        self.register_buffer("e_mean", torch.tensor(e_mean, dtype=torch.float32))
        self.register_buffer("e_std", torch.tensor(e_std, dtype=torch.float32))
        self.register_buffer("dh_mean", torch.tensor(dh_mean, dtype=torch.float32))
        self.register_buffer("dh_std", torch.tensor(dh_std, dtype=torch.float32))
        
        # 2. Homoscedastic Uncertainty Parameters for dynamic loss weighting
        self.log_sigma_geom = nn.Parameter(torch.tensor(0.0))
        self.log_sigma_ener = nn.Parameter(torch.tensor(0.0))
        self.log_sigma_force = nn.Parameter(torch.tensor(0.0))
        
        # --- NEW: Auxiliary Enthalpy Predictor ---
        self.dh_mlp = nn.Sequential(
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 1)
        )
        
        # Final predictor: combines structural energy, 1D Marcus baseline, 3D Hammond Index, and reaction feats
        self.ea_mlp = nn.Sequential(
            nn.Linear(64 + 1 + 1 + 1 + 9, 128),
            nn.GELU(), nn.Dropout(dropout),
            nn.Linear(128, 64), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(64, 1), # Changed from 2 to 1 (Energy only)
        )

    def forward(self, batch_dict, reaction_feats, pos_R, pos_P, compute_forces=False):
        # Use standard PyTorch training flag
        is_training = self.training
        
        # Pass 1a: Predict intermediate displacement from midpoint graph using Actor
        mace_out_1 = self.mace_geom(batch_dict, training=is_training, compute_force=False)
        disp_1 = self.coord_proj(mace_out_1["node_feats"])
        x_int = batch_dict["positions"] + disp_1
        
        # Pass 1b: Predict final refinement from intermediate graph using Actor
        batch_dict_int = batch_dict.copy()
        batch_dict_int["positions"] = x_int
        mace_out_2 = self.mace_geom(batch_dict_int, training=is_training, compute_force=False)
        disp_2 = self.coord_proj(mace_out_2["node_feats"])
        x_pred = x_int + disp_2

        # ----------------------------------------------------------------------------------
        # THE GRADIENT WALL & TEACHER FORCING
        # ----------------------------------------------------------------------------------
        batch_dict_ts = batch_dict.copy()
        
        if is_training:
            if "x_TS_true" in batch_dict:
                # Graph-level Teacher Forcing (50% chance per molecule).
                # We use per-molecule instead of per-atom masking to prevent "chimeric geometries"
                # where half the molecule is at the true TS and half is at a wildly wrong predicted TS,
                # creating unphysical bonds that confuse the equivariant message passing.
                batch_size = batch_dict["ptr"].shape[0] - 1
                graph_mask = (torch.rand(batch_size, 1, device=x_pred.device) > 0.5)
                mask = graph_mask[batch_dict["batch"]]
                x_input = torch.where(mask, batch_dict["x_TS_true"], x_pred)
            else:
                x_input = x_pred
                
            x_input.requires_grad_(True)
            batch_dict_ts["positions"] = x_input
        else:
            x_input = x_pred
            # Ensure coordinates can receive gradients if forces are requested during inference
            if compute_forces:
                x_input.requires_grad_(True)
            batch_dict_ts["positions"] = x_input

        # Pass 2: Evaluate Energy & Forces of the predicted TS using Critic
        pred_forces = None
        # Since GPU VRAM is highly available (4.4/15.0 GB), compute forces 100% of the time for maximum accuracy
        if is_training and "true_forces" in batch_dict:
            mace_out_ts = self.mace_ener(batch_dict_ts, training=is_training, compute_force=True)
            pred_forces = mace_out_ts["forces"]
        else:
            mace_out_ts = self.mace_ener(batch_dict_ts, training=is_training, compute_force=False)
        energy = mace_out_ts["energy"]
        
        # --- NEW: Siamese Reactant and Product Passes for Enthalpy Prediction ---
        batch_dict_R = batch_dict.copy()
        batch_dict_R["positions"] = pos_R
        batch_dict_P = batch_dict.copy()
        batch_dict_P["positions"] = pos_P
        
        mace_out_R = self.mace_ener(batch_dict_R, training=is_training, compute_force=False)
        mace_out_P = self.mace_ener(batch_dict_P, training=is_training, compute_force=False)
        
        from torch_geometric.utils import scatter, softmax
        feats_R = scatter(mace_out_R["node_feats"][:, :64], batch_dict["batch"], dim=0, reduce="sum")
        feats_P = scatter(mace_out_P["node_feats"][:, :64], batch_dict["batch"], dim=0, reduce="sum")
        dh_pred = self.dh_mlp(feats_P - feats_R)
        
        # --- PHYSICS DELTA LEARNING ---
        if not is_training and ("reaction_enthalpy" not in batch_dict or torch.isnan(batch_dict["reaction_enthalpy"]).all()):
            de_rxn_raw = dh_pred
            # FIX: Substitute dh_pred back into reaction_feats so ea_mlp and lambda_mlp see a consistent normalized feature
            dh_pred_norm = (dh_pred - self.dh_mean) / (self.dh_std + 1e-8)
            reaction_feats = reaction_feats.clone()
            reaction_feats[:, 0] = dh_pred_norm.squeeze(-1)
        else:
            if "reaction_enthalpy" in batch_dict:
                de_rxn_raw = batch_dict["reaction_enthalpy"].unsqueeze(-1)
            else:
                de_rxn_raw = torch.zeros_like(reaction_feats[:, 0:1])
                
        # Project dynamic reorganization energy from reaction features (bias initialized around 40 kcal/mol)
        dynamic_lambda = 40.0 + self.lambda_mlp(reaction_feats)
        lam = torch.clamp(dynamic_lambda, min=0.1)
        thermo_ratio = torch.clamp(de_rxn_raw / lam, min=-1.0)
        
        physics_baseline_raw = (lam / 4.0) * (1.0 + thermo_ratio)**2
        physics_baseline_norm = (physics_baseline_raw - self.e_mean) / (self.e_std + 1e-8)
        
        from torch_geometric.utils import scatter, softmax
        x_target = batch_dict_ts["positions"]
        dist_R = torch.norm(x_target - pos_R, dim=-1)
        dist_P = torch.norm(x_target - pos_P, dim=-1)
        mol_dist_R = scatter(dist_R, batch_dict["batch"], dim=0, reduce="sum")
        mol_dist_P = scatter(dist_P, batch_dict["batch"], dim=0, reduce="sum")
        hammond_index = (mol_dist_R / (mol_dist_R + mol_dist_P + 1e-8)).unsqueeze(-1).detach()
        
        atom_activity = torch.norm(pos_R - pos_P, dim=-1, keepdim=True)
        active_site_attn = softmax(atom_activity, batch_dict_ts["batch"])
        
        scalar_node_feats = mace_out_ts["node_feats"][:, :64]
        global_feats = scatter(scalar_node_feats * active_site_attn, batch_dict_ts["batch"], dim=0, reduce="sum")
        
        combined_feats = torch.cat([global_feats, energy.unsqueeze(1), physics_baseline_norm, hammond_index, reaction_feats], dim=-1)
        
        ea_out = self.ea_mlp(combined_feats)
        energy_pred_norm = ea_out[:, 0] + physics_baseline_norm.squeeze(-1)
        log_var_ea = None
        
        forces_pred = None
        # FIX: Decouple forces from `is_training`
        if is_training or compute_forces:
            grad_outputs = torch.ones_like(energy_pred_norm)
            forces_pred = -torch.autograd.grad(
                outputs=energy_pred_norm,
                inputs=batch_dict_ts["positions"],
                grad_outputs=grad_outputs,
                create_graph=is_training, # ONLY build the Hessian graph during training
                retain_graph=is_training,
                allow_unused=True
            )[0]
            
        # Return dh_pred alongside standard outputs
        return x_pred, x_int, energy_pred_norm, log_var_ea, forces_pred, dh_pred

# ----------------------------------------------------
# 5. Pipeline Wrapper with Anti-Memorization Regularization
# ----------------------------------------------------
class MACEPredictorWrapper:
    def __init__(self, net, device, epochs=1200, lr=1e-3, noise_std=0.05, weight_decay=1e-2, accumulation_steps=8):
        self.net = net.to(device)
        self.device = device
        self.epochs, self.lr, self.noise_std = epochs, lr, noise_std
        self.weight_decay = weight_decay
        self.accumulation_steps = accumulation_steps
        self.energy_mean, self.energy_std = 0.0, 1.0

    def fit(self, train_loader, val_loader):
        ea_mlp_params = []
        no_decay_params = []
        base_params = []
        for name, param in self.net.named_parameters():
            if 'log_sigma' in name or 'lambda_mlp' in name:
                no_decay_params.append(param)
            elif 'ea_mlp' in name:
                ea_mlp_params.append(param)
            else:
                base_params.append(param)
                
        optimizer = torch.optim.AdamW([
            {'params': base_params},
            {'params': ea_mlp_params, 'lr': self.lr * 3.0},
            {'params': no_decay_params, 'weight_decay': 0.0}
        ], lr=self.lr, weight_decay=self.weight_decay)
        
        # (Removed vestigial switch_epoch)
        warmup_epochs = max(10, self.epochs // 10)
        scheduler1 = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs)
        scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs - warmup_epochs, eta_min=1e-5)
        scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[scheduler1, scheduler2], milestones=[warmup_epochs])
        
        start_epoch = 1
        best_val_loss, best_weights = float('inf'), None
        patience = max(50, self.epochs // 3)
        patience_counter = 0
        
        script_dir = os.path.dirname(os.path.abspath(__file__))
        latest_ckpt = os.path.join(script_dir, "mace_checkpoint_latest.pt")
        
        if os.path.exists(latest_ckpt):
            print(f"Resuming training from {latest_ckpt}...")
            checkpoint = torch.load(latest_ckpt, map_location=self.device, weights_only=True)
            
            # Backwards compatibility for old checkpoints that used 'net_state_dict'
            state_key = 'model_state_dict' if 'model_state_dict' in checkpoint else 'net_state_dict'
            self.net.load_state_dict(checkpoint[state_key])
            
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            # Defensive Scheduler Loading: only load if the target epochs haven't been altered
            if checkpoint.get('epochs', self.epochs) == self.epochs:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                start_epoch = checkpoint['epoch'] + 1
            else:
                start_epoch = checkpoint['epoch'] + 1
                print(f"Notice: Target epochs changed ({checkpoint.get('epochs')} -> {self.epochs}). Initializing fresh scheduler trajectory.")
                # Fast-forward the fresh scheduler to the correct epoch to maintain the LR trajectory
                for _ in range(start_epoch - 1):
                    scheduler.step()
                
            best_val_loss = checkpoint.get('best_val_loss', float('inf'))
            best_weights = checkpoint.get('best_weights', self.net.state_dict())
            patience_counter = checkpoint.get('patience_counter', 0)

        for epoch in range(start_epoch, self.epochs + 1):
            
            self.net.train()
            total_geom_loss, total_ener_loss, total_force_loss = 0.0, 0.0, 0.0
            
            for batch_idx, batch in enumerate(train_loader):
                batch = batch.to(self.device)
                if batch_idx % self.accumulation_steps == 0:
                    optimizer.zero_grad(set_to_none=True)
                batch_dict = batch.to_dict()
                
                if self.noise_std > 0:
                    noise = torch.randn_like(batch_dict["positions"]) * self.noise_std
                    
                    # Zero-mean the noise per-molecule to preserve translational invariance
                    from torch_geometric.utils import scatter
                    noise_mean = scatter(noise, batch_dict["batch"], dim=0, reduce="mean")
                    noise = noise - noise_mean[batch_dict["batch"]]
                    
                    batch_dict["positions"] += noise
                
                x_pred, x_int, pred_energy_norm, log_var_ea, pred_forces, dh_pred = self.net(batch_dict, batch.reaction_feats, batch.pos_R, batch.pos_P)
                
                # Hammond-Guided Coarse Supervision (LST Target)
                # Instead of matching the exact quantum TS, the intermediate graph predicts classical Linear Synchronous Transit (LST)
                x_lst_target = 0.5 * batch.pos_R + 0.5 * batch.pos_P
                geom_penalty = batch.reaction_weight.unsqueeze(-1)
                
                loss_geom_refined = torch.mean(geom_penalty * (x_pred - batch.x_TS_true)**2)
                loss_geom_coarse = torch.mean(geom_penalty * (x_int - x_lst_target)**2)
                loss_geom = loss_geom_refined + 0.5 * loss_geom_coarse
                
                # Advanced Physics Constraints: Vectorized over the generous edge_index
                row, col = batch.edge_index[0], batch.edge_index[1]
                # Process only upper triangle to avoid double counting and self-loops
                mask_triu = row < col
                row_t, col_t = row[mask_triu], col[mask_triu]
                
                if len(row_t) > 0:
                    dist_pred = torch.norm(x_pred[row_t] - x_pred[col_t], dim=-1)
                    dist_true = torch.norm(batch.x_TS_true[row_t] - batch.x_TS_true[col_t], dim=-1)
                    dist_R = torch.norm(batch.pos_R[row_t] - batch.pos_R[col_t], dim=-1)
                    dist_P = torch.norm(batch.pos_P[row_t] - batch.pos_P[col_t], dim=-1)
                    cov_r = batch["node_attrs"][:, -1]
                    ref_dist = cov_r[row_t] + cov_r[col_t]
                    
                    # 1. Steric Floor (Pauli Repulsion)
                    # Adaptive floor matches the 0.55x pre-clash trigger used in the iterative_steric_clamp
                    loss_steric = F.relu((0.55 * ref_dist) - dist_pred).mean()
                    
                    # 2. Pauling Bond Order Adaptive Weighting
                    c_param = 0.3
                    BO_true = torch.exp(torch.clamp((ref_dist - dist_true) / c_param, max=5.0))
                    dist_weight = BO_true + 0.05
                    dist_huber = F.huber_loss(dist_pred, dist_true, reduction='none', delta=0.5)
                    loss_bo_dist = (dist_weight * dist_huber).mean()
                    
                    # 3. Harmonic Spectator Strain (MM-style)
                    spectator_mask = torch.abs(dist_R - dist_P) < 0.15
                    if spectator_mask.any():
                        loss_spectator_strain = ((dist_pred[spectator_mask] - dist_R[spectator_mask])**2).mean()
                    else:
                        loss_spectator_strain = torch.tensor(0.0, device=self.device)
                        
                    # 4. Continuous Valency Conservation
                    from torch_geometric.utils import scatter
                    # Compute over all directed edges (excluding self-loops, which edge_index handles)
                    dist_pred_all = torch.norm(x_pred[row] - x_pred[col], dim=-1)
                    dist_R_all = torch.norm(batch.pos_R[row] - batch.pos_R[col], dim=-1)
                    dist_P_all = torch.norm(batch.pos_P[row] - batch.pos_P[col], dim=-1)
                    
                    ref_dist_all = cov_r[row] + cov_r[col]
                    
                    BO_pred_all = torch.exp(torch.clamp((ref_dist_all - dist_pred_all) / c_param, max=5.0))
                    BO_R_all = torch.exp(torch.clamp((ref_dist_all - dist_R_all) / c_param, max=5.0))
                    BO_P_all = torch.exp(torch.clamp((ref_dist_all - dist_P_all) / c_param, max=5.0))
                    
                    valency_pred = scatter(BO_pred_all, row, dim=0, dim_size=x_pred.shape[0], reduce='sum')
                    valency_R = scatter(BO_R_all, row, dim=0, dim_size=x_pred.shape[0], reduce='sum')
                    valency_P = scatter(BO_P_all, row, dim=0, dim_size=x_pred.shape[0], reduce='sum')
                    
                    valency_target = 0.5 * valency_R + 0.5 * valency_P
                    active_mask = batch.reaction_weight > 0.5
                    if active_mask.any():
                        loss_valency = F.huber_loss(valency_pred[active_mask], valency_target[active_mask], delta=2.0)
                    else:
                        loss_valency = torch.tensor(0.0, device=self.device)
                else:
                    loss_steric = loss_bo_dist = loss_spectator_strain = loss_valency = torch.tensor(0.0, device=self.device)

                # Softened Geometry Physics weights to prevent astronomical numbers on Epoch 1
                loss_geom = loss_geom + 0.1 * loss_steric + 0.5 * loss_bo_dist + 0.5 * loss_spectator_strain + 0.2 * loss_valency

                # Standard Huber Loss (Variance scaling is handled by Homoscedastic wrapper below)
                loss_ener = F.huber_loss(pred_energy_norm, batch.norm_energy, delta=1.0)
                
                # Thermodynamic Consistency Penalty (Ea >= max(0, dH))
                thermo_floor_raw = torch.clamp(batch.reaction_enthalpy.squeeze(-1), min=0.0)
                thermo_floor_norm = (thermo_floor_raw - self.energy_mean) / (self.energy_std + 1e-8)
                thermo_violation = F.relu(thermo_floor_norm - pred_energy_norm)
                loss_thermo = thermo_violation.mean() * 2.0 # Strict constraint
                loss_ener = loss_ener + loss_thermo
                
                # Force Scale Normalization & Missing Data Masking
                loss_forces = torch.tensor(0.0, device=self.device)
                if pred_forces is not None:
                    mask_forces_nodes = batch.has_forces[batch.batch]
                    if mask_forces_nodes.any():
                        true_forces_scaled = batch.true_forces[mask_forces_nodes] / (self.energy_std + 1e-8)
                        pred_forces_masked = pred_forces[mask_forces_nodes]
                        loss_forces = F.mse_loss(pred_forces_masked, true_forces_scaled)

                with torch.no_grad():
                    g, e, f = loss_geom.item(), loss_ener.item(), loss_forces.item()

                clamped_log_sigma_geom = torch.clamp(self.net.log_sigma_geom, min=-6.0, max=3.0)
                clamped_log_sigma_ener = torch.clamp(self.net.log_sigma_ener, min=-6.0, max=3.0)
                var_geom = torch.exp(2 * clamped_log_sigma_geom) + 1e-4
                var_ener = torch.exp(2 * clamped_log_sigma_ener) + 1e-4
                
                # --- NEW: Enthalpy Auxiliary Loss ---
                # Normalize both predictions and targets by energy_std to match loss_ener variance scale
                loss_dh = F.mse_loss(dh_pred / (self.energy_std + 1e-8), batch.reaction_enthalpy.unsqueeze(-1) / (self.energy_std + 1e-8))
                
                # Rely solely on Kendall-Gal Homoscedastic Uncertainty for dynamic weighting.
                # Note: loss_dh is deliberately excluded from uncertainty weighting because it is an auxiliary side-task.
                loss = (loss_geom / (2 * var_geom)) + (loss_ener / (2 * var_ener)) + loss_dh
                loss = loss + clamped_log_sigma_geom + clamped_log_sigma_ener
                
                has_any_forces = pred_forces is not None and batch.has_forces[batch.batch].any()
                if has_any_forces:
                    # Hard clamp for absolute safety against runaway variance
                    clamped_log_sigma_force = torch.clamp(self.net.log_sigma_force, min=-6.0, max=3.0)
                    var_force = torch.exp(2 * clamped_log_sigma_force)
                    loss = loss + (loss_forces / (2 * var_force)) + clamped_log_sigma_force
                
                loss = loss / self.accumulation_steps
                loss.backward()
                
                if (batch_idx + 1) % self.accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
                    torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=1.0)
                    optimizer.step()
                
                total_geom_loss += loss_geom.item()
                total_ener_loss += loss_ener.item()
                total_force_loss += loss_forces.item()
                
            scheduler.step()
            self.net.eval()
            val_geom_loss, val_ener_loss = 0.0, 0.0
            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(self.device)
                    xp, x_int_val, ep_norm, log_var_ea_val, _, _ = self.net(batch.to_dict(), batch.reaction_feats, batch.pos_R, batch.pos_P)
                    geom_penalty = batch.reaction_weight.unsqueeze(-1)
                    val_geom_loss += torch.mean(geom_penalty * (xp - batch.x_TS_true)**2).item()
                    
                    val_ener_loss += F.huber_loss(ep_norm, batch.norm_energy, delta=1.0).item()
                    
            mean_val_geom = val_geom_loss / max(1, len(val_loader))
            mean_val_ener = val_ener_loss / max(1, len(val_loader))

            mean_val = mean_val_geom + mean_val_ener
            
            improved = mean_val < best_val_loss
            if improved:
                best_val_loss = mean_val
                best_weights = {k: v.cpu().clone() for k, v in self.net.state_dict().items()}
                patience_counter = 0
            else: patience_counter += 1
                
            print(f"Epoch {epoch:3d} | Train Geom: {total_geom_loss/max(1, len(train_loader)):.4f} Ener: {total_ener_loss/max(1, len(train_loader)):.4f} Force: {total_force_loss/max(1, len(train_loader)):.4f} | Val Score: {mean_val:.4f}{' *' if improved else ''} | log_σ_geom: {self.net.log_sigma_geom.item():.2f}, log_σ_ener: {self.net.log_sigma_ener.item():.2f}, log_σ_f: {self.net.log_sigma_force.item():.2f}")
            
            # Save continuous checkpoint to Google Drive to protect against Colab timeouts
            torch.save({
                'epoch': epoch,
                'epochs': self.epochs,
                'model_state_dict': self.net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss': best_val_loss,
                'best_weights': best_weights,
                'patience_counter': patience_counter
            }, latest_ckpt)
                
            if patience_counter >= patience and epoch >= (self.epochs * 0.5):
                print(f"Patience exceeded at epoch {epoch}. Stopping early.")
                break
                
        if best_weights: self.net.load_state_dict({k: v.to(self.device) for k, v in best_weights.items()})

# ----------------------------------------------------
# 6. Standalone Inference Pipeline
# ----------------------------------------------------
def parse_xyz(path):
    atoms = []
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        if len(lines) < 3: return atoms
        n_atoms = int(lines[0].strip())
        for line in lines[2:2+n_atoms]:
            parts = line.split()
            if len(parts) >= 4:
                atoms.append({"atom": parts[0], "x": float(parts[1]), "y": float(parts[2]), "z": float(parts[3])})
    return atoms

def run_inference(reactant_xyz, product_xyz, dh=None, model_path="mace_double_ended.pt"):
    print(f"Loading Reactant from {reactant_xyz}")
    j1_atoms = parse_xyz(reactant_xyz)
    print(f"Loading Product from {product_xyz}")
    j2_atoms = parse_xyz(product_xyz)
    if len(j1_atoms) != len(j2_atoms) or len(j1_atoms) == 0:
        print("Atom count mismatch or empty file!")
        return
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading trained weights from {model_path}...")
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    meta = checkpoint["metadata"]
    z_table = meta["z_table_elements"]
    e_mean, e_std = meta["energy_mean"], meta["energy_std"]
    scaler_mean, scaler_scale = np.array(meta["scaler_mean"]), np.array(meta["scaler_scale"])
    
    net = MACEReactionAwareTSModel(z_table, avg_num_neighbors=meta.get("avg_num_neighbors", 8.0), e_mean=e_mean, e_std=e_std).to(device)
    net.load_state_dict(checkpoint["model_state_dict"])
    net.eval()
    
    # If dh is None, use a dummy value for feature scaling, we will rely on dh_pred
    dummy_dh = dh if dh is not None else 0.0
    rxn_feats = compute_reaction_features(j1_atoms, j2_atoms, dummy_dh)
    rxn_feats_norm = (rxn_feats - scaler_mean) / scaler_scale
    
    n = len(j1_atoms)
    c_R = np.array([[a["x"], a["y"], a["z"]] for a in j1_atoms], dtype=np.float32)
    c_P = np.array([[a["x"], a["y"], a["z"]] for a in j2_atoms], dtype=np.float32)
    c_R_c = c_R - c_R.mean(axis=0)
    c_P_aligned = kabsch_align(c_P, c_R)
    c_P_c = c_P_aligned - c_P_aligned.mean(axis=0)
    c_midpoint = (c_R_c + c_P_c) / 2.0
    
    # STERIC MIDPOINT CLAMPING (Iterative)
    atomic_numbers = [SYMBOL_TO_Z[x['atom']] for x in j1_atoms]
    c_midpoint = iterative_steric_clamp(c_midpoint, atomic_numbers)
                
    cutoff = 5.0
    dist_mid = np.linalg.norm(c_midpoint[:, None, :] - c_midpoint[None, :, :], axis=-1)
    dist_R = np.linalg.norm(c_R_c[:, None, :] - c_R_c[None, :, :], axis=-1)
    dist_P = np.linalg.norm(c_P_c[:, None, :] - c_P_c[None, :, :], axis=-1)
    
    mask = (dist_mid <= cutoff) | (dist_R <= cutoff) | (dist_P <= cutoff)
    np.fill_diagonal(mask, False)
    edges = np.vstack(np.where(mask))
    edge_index = torch.tensor(edges, dtype=torch.long).contiguous() if edges.shape[1] > 0 else torch.empty((2, 0), dtype=torch.long)
    
    node_attrs = np.zeros((n, len(z_table) + 2), dtype=np.float32)
    for i, z in enumerate(atomic_numbers):
        if z in z_table:
            node_attrs[i, z_table.index(z)] = 1.0
        else:
            raise ValueError(f"Error: Element Z={z} was not seen during training! Check your dataset.")
        node_attrs[i, -2] = PAULING_EN.get(z, 2.0)
        node_attrs[i, -1] = COVALENT_RADII.get(z, 1.0)
            
    batch = Data(
        positions=torch.tensor(c_midpoint, dtype=torch.float32), 
        node_attrs=torch.tensor(node_attrs, dtype=torch.float32),
        edge_index=edge_index,
        shifts=torch.zeros((edge_index.shape[1], 3), dtype=torch.float32),
        unit_shifts=torch.zeros((edge_index.shape[1], 3), dtype=torch.float32),
        atomic_numbers=torch.tensor(atomic_numbers, dtype=torch.long),
        cell=torch.zeros(3, 3, dtype=torch.float32).unsqueeze(0),
        pbc=torch.tensor([False, False, False], dtype=torch.bool).unsqueeze(0),
        reaction_enthalpy=torch.tensor([float('nan') if dh is None else dh], dtype=torch.float32)
    ).to_dict()
    
    for k, v in batch.items():
        if isinstance(v, torch.Tensor): batch[k] = v.to(device)
    
    batch["ptr"] = torch.tensor([0, n], dtype=torch.long).to(device)
    batch["batch"] = torch.zeros(n, dtype=torch.long).to(device)
    rxn_feats_t = torch.tensor(rxn_feats_norm, dtype=torch.float32).unsqueeze(0).to(device)
    pos_R_t = torch.tensor(c_R_c, dtype=torch.float32).to(device)
    pos_P_t = torch.tensor(c_P_c, dtype=torch.float32).to(device)
    
    print("Running Double-Ended MACE Inference...")
    
    compute_forces = True 
    context = torch.enable_grad() if compute_forces else torch.no_grad()
    
    with context:
        # Requires gradients on inputs to compute forces if compute_forces is True
        if compute_forces:
            batch["positions"].requires_grad_(True)
            
        x_pred, _, ea_pred_norm, _, forces_pred, dh_pred = net(
            batch, rxn_feats_t, pos_R_t, pos_P_t, compute_forces=compute_forces
        )
        ea_pred = ea_pred_norm * e_std + e_mean
        
        # Un-normalize or format dh_pred if using the internal auxiliary head
        internal_dh = dh_pred.item()
        
    ea_pred = ea_pred.item()
    print(f"\n======================================")
    if dh is None:
        print(f" PREDICTED ENTHALPY (dH):     {internal_dh:.4f} kcal/mol (Auxiliary Head)")
    else:
        print(f" PROVIDED ENTHALPY (dH):      {dh:.4f} kcal/mol")
    print(f" PREDICTED ACTIVATION ENERGY: {ea_pred:.4f} kcal/mol")
    if compute_forces and forces_pred is not None:
        max_force = forces_pred.abs().max().item()
        print(f" MAX FORCE COMPONENT:         {max_force:.4f} kcal/(mol·Å)")
    print(f"======================================\n")
    
    x_pred_np = x_pred.detach().cpu().numpy()
    out_xyz = "predicted_ts.xyz"
    write_xyz(out_xyz, [a['atom'] for a in j1_atoms], x_pred_np, f"Predicted TS | Ea={ea_pred:.4f} kcal/mol")
    print(f"Predicted Transition State geometry saved to -> {out_xyz}")

# ----------------------------------------------------
# 7. Main Execution
# ----------------------------------------------------
global_csv_dict = None
def init_worker():
    torch.set_num_threads(1)

def process_folder_global(folder):
    local_stats = {}
    try:
        p = parse_true_qchem_reaction(folder, stats=local_stats)
        return p, local_stats
    except Exception as e:
        return None, {f"parse_exception:{type(e).__name__}": 1}

def main():
    parser = argparse.ArgumentParser(description="Double-Ended MACE TS Pipeline")
    parser.add_argument("command", nargs="?", default="train")
    parser.add_argument("--epochs", type=int, default=1200)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--refresh_cache", action="store_true", help="Force re-parsing of dataset and ignore cache")
    parser.add_argument("--data_dir", nargs='+', default=["subset"], help="Path to the folder(s) containing rxn subfolders")
    parser.add_argument("--reactant", type=str, help="Path to reactant .xyz (for inference)")
    parser.add_argument("--product", type=str, help="Path to product .xyz (for inference)")
    parser.add_argument("--dh", type=float, help="Reaction enthalpy kcal/mol (for inference)")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate for EA MLP (regularization)")
    parser.add_argument("--noise_std", type=float, default=0.05, help="Coordinate noise std dev for data augmentation")
    parser.add_argument("--weight_decay", type=float, default=2e-2, help="L2 Regularization parameter")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size (keep tiny 1-4 for Hessian OOM prevention)")
    parser.add_argument("--accumulation_steps", type=int, default=8, help="Gradient accumulation steps to simulate larger batch size (e.g. batch_size=4 * steps=8 = 32)")
    args = parser.parse_args()
    
    if args.command == "infer":
        if not args.reactant or not args.product:
            print("Error: --reactant and --product required for inference.")
            return
        if args.dh is None:
            print("Notice: --dh not provided. The model will automatically predict the Reaction Enthalpy using the auxiliary head.")
        run_inference(args.reactant, args.product, args.dh, "mace_double_ended.pt")
        return
        
    if args.command != "train":
        print(f"Command '{args.command}' is not implemented yet. Defaulting to 'train' or exit.")
        return
        
    folder_paths = args.data_dir
    import glob
    rxn_folders = []
    for fp in folder_paths:
        rxn_folders.extend(glob.glob(os.path.join(fp, "rxn*")))
    print(f"Found {len(rxn_folders)} raw reaction folders. Parsing logs...")
    
    if args.limit > 0:
        rxn_folders = rxn_folders[:args.limit]
        print(f"Limiting to {args.limit} folders as requested.")

    cache_file = f"parsed_dataset_cache_nocsv_{args.limit if args.limit > 0 else 'all'}.pkl"
    import pickle
    if os.path.exists(cache_file) and not getattr(args, 'refresh_cache', False):
        print(f"Found parsed cache at {cache_file}. Loading instantly...")
        with open(cache_file, "rb") as f:
            samples = pickle.load(f)
        reject_stats = {}
    else:
        import concurrent.futures
        samples, reject_stats = [], {}
        print(f"Parsing logs concurrently using os.cpu_count() processes (bypasses Google Drive latency)...")
        
        max_workers = os.cpu_count() or 4
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers, initializer=init_worker) as executor:
            futures = [executor.submit(process_folder_global, f) for f in rxn_folders]
            for i, future in enumerate(concurrent.futures.as_completed(futures)):
                if i > 0 and i % 500 == 0: 
                    print(f"Parsed {i}/{len(rxn_folders)} folders...")
                p, local_stats = future.result()
                if p: samples.append(p)
                for k, v in local_stats.items():
                    reject_stats[k] = reject_stats.get(k, 0) + v
                    
        print(f"Successfully loaded {len(samples)} valid reaction triplets with True Ea.")
        if reject_stats:
            print("Rejected files by reason:")
            for reason, count in sorted(reject_stats.items(), key=lambda kv: -kv[1]):
                print(f"  {reason}: {count}")
                
        if len(samples) > 0:
            print(f"Saving parsed data to cache ({cache_file}) for instant loading next time...")
            with open(cache_file, "wb") as f:
                pickle.dump(samples, f)
    
    if len(samples) == 0: return

    y_energy = np.array([s["true_ea"] for s in samples])
    # 1. Stratified Element Split: Guarantee every element is seen in training
    all_elements = set()
    for s in samples:
        all_elements.update(s["atom_counts"].keys())
        
    z_table = sorted([SYMBOL_TO_Z[el] for el in all_elements])
    
    train_idx_set = set()
    elements_covered = set()
    
    import random
    shuffled_indices = list(range(len(samples)))
    random.Random(42).shuffle(shuffled_indices)
    
    for idx in shuffled_indices:
        sample_elements = set(samples[idx]["atom_counts"].keys())
        if not sample_elements.issubset(elements_covered):
            train_idx_set.add(idx)
            elements_covered.update(sample_elements)
        if len(elements_covered) == len(all_elements):
            break
            
    remaining_indices = [i for i in shuffled_indices if i not in train_idx_set]
    num_train_needed = int(0.8 * len(samples)) - len(train_idx_set)
    
    if num_train_needed > 0:
        train_idx_set.update(remaining_indices[:num_train_needed])
        test_idx_set = set(remaining_indices[num_train_needed:])
    else:
        test_idx_set = set(remaining_indices)
        
    train_idx = np.array(list(train_idx_set))
    test_idx = np.array(list(test_idx_set))

    e_mean = y_energy[train_idx].mean()
    e_std = y_energy[train_idx].std() if len(train_idx) > 1 else 1.0
    
    # Absolute safety check against zero-variance NaNs
    if e_std < 1e-4:
        e_std = 1.0
        
    norm_e = (y_energy - e_mean) / e_std
    for i, s in enumerate(samples): s["norm_energy"] = float(norm_e[i])
        
    rxn_feat_matrix = np.stack([
        compute_reaction_features(s["j1_atoms"], s["j2_atoms"], s["reaction_enthalpy"]) for s in samples
    ])
    scaler = StandardScaler().fit(rxn_feat_matrix[train_idx])
    
    train_dataset = MACEReactionAwareDataset([samples[i] for i in train_idx], z_table, scaler_mean=scaler.mean_.tolist(), scaler_scale=scaler.scale_.tolist())
    test_dataset = MACEReactionAwareDataset([samples[i] for i in test_idx], z_table, scaler_mean=scaler.mean_.tolist(), scaler_scale=scaler.scale_.tolist())

    n_probe = min(300, len(train_dataset))
    probe_idx = np.random.RandomState(42).choice(len(train_dataset), size=n_probe, replace=False)
    total_edges, total_nodes = 0, 0
    for idx in probe_idx:
        data = train_dataset[int(idx)]
        # Use the actual dataset graph (Midpoint + R + P union) to estimate neighbors
        total_edges += data.edge_index.shape[1]
        total_nodes += data.num_nodes
    avg_num_neighbors = float(total_edges / max(1, total_nodes))
    import multiprocessing
    num_workers = min(2, multiprocessing.cpu_count())
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Initializing Model...")
    net = MACEReactionAwareTSModel(
        z_table, 
        avg_num_neighbors=avg_num_neighbors,
        e_mean=e_mean,
        e_std=e_std,
        dh_mean=float(scaler.mean_[0]),
        dh_std=float(scaler.scale_[0]),
        dropout=args.dropout
    ).to(device)
    
    wrapper = MACEPredictorWrapper(net, device, epochs=args.epochs, noise_std=args.noise_std, weight_decay=args.weight_decay, accumulation_steps=args.accumulation_steps)
    wrapper.energy_mean, wrapper.energy_std = e_mean, e_std
    
    print("Training Double-Ended MACE (Hybrid Aleatoric/Homoscedastic Uncertainty)...")
    wrapper.fit(train_loader, val_loader)
    
    torch.save({
        "model_state_dict": wrapper.net.state_dict(),
        "metadata": {
            "z_table_elements": z_table, "energy_mean": float(e_mean), "energy_std": float(e_std), 
            "scaler_mean": scaler.mean_.tolist(), "scaler_scale": scaler.scale_.tolist(),
            "avg_num_neighbors": avg_num_neighbors
        }
    }, "mace_double_ended.pt")
    
    # --- EVALUATION AND METRIC COMPUTATION ---
    print("\nEvaluating trained model on the 20% test split...")
    checkpoint = torch.load("mace_double_ended.pt", map_location=wrapper.device, weights_only=True)
    wrapper.net.load_state_dict(checkpoint["model_state_dict"])
    wrapper.net.eval()
    
    test_e_preds, test_e_trues, test_d_rmsd, dashboard_samples = [], [], [], []
    z_to_symbol = {v: k for k, v in SYMBOL_TO_Z.items()}
    from sklearn.metrics import r2_score, mean_absolute_error
    
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(wrapper.device)
            with torch.no_grad():
                x_pred, _, ea_pred_norm, _, _, _ = wrapper.net(batch.to_dict(), batch.reaction_feats, batch.pos_R, batch.pos_P)
            
            ea_pred_kcal = ea_pred_norm * wrapper.energy_std + wrapper.energy_mean
            x_pred, x_TS_true, ea_pred_kcal = x_pred.cpu().numpy(), batch.x_TS_true.cpu().numpy(), ea_pred_kcal.cpu().numpy()
            ptr = batch.ptr.cpu().numpy()
            for i in range(len(ptr) - 1):
                start, end = ptr[i], ptr[i+1]
                
                pred_ea = ea_pred_kcal[i]
                true_ea = batch.target_energy[i].item()
                test_e_preds.append(pred_ea); test_e_trues.append(true_ea)
                
                # RMSD = sqrt(mean over atoms of squared 3D distance), NOT sqrt(mean over all
                # x/y/z components) — the latter under-reports by a factor of sqrt(3).
                x_pred_aligned = kabsch_align(x_pred[start:end], x_TS_true[start:end])
                c_R_aligned = kabsch_align(batch.pos_R[start:end].cpu().numpy(), x_pred_aligned)
                c_P_aligned = kabsch_align(batch.pos_P[start:end].cpu().numpy(), x_pred_aligned)

                atom_sq_dist = np.sum((x_pred_aligned - x_TS_true[start:end]) ** 2, axis=1)
                rmsd = float(np.sqrt(np.mean(atom_sq_dist)))
                test_d_rmsd.append(rmsd)
                
                z_array = batch.atomic_numbers[start:end].cpu().numpy()
                
                def make_xyz(coords, title):
                    lines = [f"{len(z_array)}", title]
                    for z, (x, y, z_coord) in zip(z_array, coords):
                        lines.append(f"{z_to_symbol.get(z, 'X'):<2} {x: .8f} {y: .8f} {z_coord: .8f}")
                    return "\n".join(lines)
                
                xyz_ts = make_xyz(x_pred_aligned, f"MACE Predicted TS, Ea={pred_ea:.4f} kcal/mol")
                xyz_r = make_xyz(c_R_aligned, "Aligned Reactant")
                xyz_p = make_xyz(c_P_aligned, "Aligned Product")
                
                dashboard_samples.append({
                    "rxn_id": f"rxn_test_{len(dashboard_samples)}", "true_ea": float(true_ea),
                    "pred_ea": float(pred_ea), "error": float(pred_ea - true_ea),
                    "rmsd": rmsd, "xyz_reactant": xyz_r, "xyz_ts": xyz_ts, "xyz_product": xyz_p
                })
                
    test_e_preds, test_e_trues, test_d_rmsd = np.array(test_e_preds), np.array(test_e_trues), np.array(test_d_rmsd)
    r2_ea = r2_score(test_e_trues, test_e_preds) if len(test_e_trues) > 1 else float('nan')
    mae_ea = mean_absolute_error(test_e_trues, test_e_preds) if len(test_e_trues) > 0 else float('nan')
    rmse_ea = np.sqrt(np.mean((test_e_preds - test_e_trues) ** 2)) if len(test_e_trues) > 0 else float('nan')
    
    test_c_preds = np.concatenate(test_c_preds).flatten() if test_c_preds else np.array([])
    test_c_trues = np.concatenate(test_c_trues).flatten() if test_c_trues else np.array([])
    
    print("\n" + "="*54)
    print("                 TEST SET METRICS                 ")
    print("="*54)
    print(f"Activation Energy Ea R^2 Score:         {r2_ea:.6f}")
    print(f"Activation Energy Ea MAE:              {mae_ea:.4f} kcal/mol")
    print(f"Activation Energy Ea RMSE:             {rmse_ea:.4f} kcal/mol")
    print(f"Transition State Geometry Mean RMSD:    {np.mean(test_d_rmsd):.4f} Angstrom")
    print("="*54)
    
    # Sanitize metrics for safe JSON Tkinter parsing
    safe_r2 = float(r2_ea) if not np.isnan(r2_ea) else 0.0
    safe_mae = float(mae_ea) if not np.isnan(mae_ea) else 0.0
    safe_rmse = float(rmse_ea) if not np.isnan(rmse_ea) else 0.0
    safe_rmsd = float(np.mean(test_d_rmsd)) if len(test_d_rmsd) > 0 else 0.0
    
    with open("mace_dashboard_data.json", "w", encoding="utf-8") as f:
        json.dump({
            "metrics": {
                "r2": safe_r2, 
                "mae": safe_mae, 
                "rmse": safe_rmse, 
                "rmsd": safe_rmsd
            }, 
            "samples": dashboard_samples
        }, f, indent=2)
    print("Dashboard data exported to 'mace_dashboard_data.json'")

if __name__ == "__main__":
    main()