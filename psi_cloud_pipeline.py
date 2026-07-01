import os
import sys
import subprocess

# Ensure dependencies are installed in the cloud environment
try:
    import numpy as np
    import torch
except ImportError:
    print("Installing required dependencies...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "numpy", "torch"])
    import numpy as np
    import torch

import json
import math
import tarfile
import argparse
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset

# =============================================================================
# Chemistry / geometry utilities (formerly psi_utils.py)
# =============================================================================

# Covalent radii table in Angstroms
COVALENT_RADII = {
    'H': 0.31, 'C': 0.76, 'N': 0.71, 'O': 0.66, 'F': 0.57,
    'S': 1.05, 'Cl': 1.02, 'Br': 1.20, 'I': 1.39, 'P': 1.07,
    'Si': 1.11, 'B': 0.84,
}

def covalent_radius(atom_type):
    return COVALENT_RADII.get(atom_type, 0.76)

# Pauling electronegativity (dimensionless)
PAULING_EN = {
    'H': 2.20, 'C': 2.55, 'N': 3.04, 'O': 3.44, 'F': 3.98,
    'S': 2.58, 'Cl': 3.16, 'Br': 2.96, 'I': 2.66, 'P': 2.19,
    'Si': 1.90, 'B': 2.04,
}
# Atomic number Z
ATOMIC_NUMBER = {
    'H': 1, 'C': 6, 'N': 7, 'O': 8, 'F': 9,
    'S': 16, 'Cl': 17, 'Br': 35, 'I': 53, 'P': 15,
    'Si': 14, 'B': 5,
}
# Standard atomic mass (u)
ATOMIC_MASS = {
    'H': 1.008, 'C': 12.011, 'N': 14.007, 'O': 15.999, 'F': 18.998,
    'S': 32.06, 'Cl': 35.45, 'Br': 79.904, 'I': 126.904, 'P': 30.974,
    'Si': 28.085, 'B': 10.811,
}

def electronegativity(atom_type):
    return PAULING_EN.get(atom_type, 2.55)

def atomic_number(atom_type):
    return ATOMIC_NUMBER.get(atom_type, 6)

def atomic_mass(atom_type):
    return ATOMIC_MASS.get(atom_type, 12.011)

def bond_adjacency_from_coords(coords, atom_types, n, bond_scale=1.45):
    """Bonded-neighbour lists for the first `n` atoms, by covalent-radius cutoff."""
    adjacency = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            cutoff = bond_scale * (covalent_radius(atom_types[i]) + covalent_radius(atom_types[j]))
            if np.linalg.norm(coords[i] - coords[j]) <= cutoff:
                adjacency[i].append(j)
                adjacency[j].append(i)
    return adjacency

def bond_angles_from_coords(coords, atom_types, n, bond_scale=1.45):
    """Map each bonded triplet (i, j, k) with central atom j and i<k to its angle (deg).

    The bond graph is derived from `coords`, so passing reactant vs product
    coordinates yields the respective angle sets for comparison.
    """
    adjacency = bond_adjacency_from_coords(coords, atom_types, n, bond_scale)
    angles = {}
    for j in range(n):
        nbrs = sorted(adjacency[j])
        for a in range(len(nbrs)):
            for b in range(a + 1, len(nbrs)):
                i, k = nbrs[a], nbrs[b]
                v1 = coords[i] - coords[j]
                v2 = coords[k] - coords[j]
                n1 = np.linalg.norm(v1)
                n2 = np.linalg.norm(v2)
                if n1 < 1e-9 or n2 < 1e-9:
                    continue
                cos = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
                angles[(i, j, k)] = float(np.degrees(np.arccos(cos)))
    return angles

def _stats4(values):
    """(mean, std, min, max) over a list/array, all 0.0 when empty."""
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return 0.0, 0.0, 0.0, 0.0
    return float(arr.mean()), float(arr.std()), float(arr.min()), float(arr.max())

def build_energy_features(atom_types, n, c_R_aligned, c_P, e_r, e_p, bond_scale=1.45):
    """Construct the energy-head input feature vector from reactant + product only.

    Shared by training (build_reaction_samples) and inference
    (predict_transition_state) so the two can never drift out of sync. All
    inputs are available before the TS is known. Returns float32 of fixed length.

    Feature groups (20D total):
      [0:10]  reaction energetics + composition
      [10:20] bond-angle statistics for reactant, product, and their change

    Note: per-atom atomic descriptors (EN, Z, Mass) have been moved out of this
    global vector and are now attached directly to each atom in PSICore via
    build_atom_physical_features(), preserving their spatial identity.
    """
    cR = np.asarray(c_R_aligned, dtype=np.float64)[:n]
    cP = np.asarray(c_P, dtype=np.float64)[:n]
    types = list(atom_types[:n])

    # --- reaction energetics + composition -------------------------------
    de_rxn = abs(e_r - e_p)
    de_rxn_signed = e_p - e_r  # signed reaction energy (Bell-Evans-Polanyi driver)
    diff_norms = np.linalg.norm(cR - cP, axis=1)
    c_count = sum(1 for t in types if t == 'C')
    h_count = sum(1 for t in types if t == 'H')
    n_count = sum(1 for t in types if t == 'N')
    o_count = sum(1 for t in types if t == 'O')

    # --- bond-angle statistics -------------------------------------------
    ang_R = bond_angles_from_coords(cR, types, n, bond_scale)
    ang_P = bond_angles_from_coords(cP, types, n, bond_scale)
    aR_mean, aR_std, aR_min, aR_max = _stats4(ang_R.values())
    aP_mean, aP_std, aP_min, aP_max = _stats4(ang_P.values())
    common = set(ang_R) & set(ang_P)
    if common:
        changes = np.array([abs(ang_R[t] - ang_P[t]) for t in common], dtype=np.float64)
        ang_change_mean = float(changes.mean())
        ang_change_max = float(changes.max())
    else:
        ang_change_mean = 0.0
        ang_change_max = 0.0

    feats = np.array([
        # reaction energetics + composition (10 features)
        de_rxn, de_rxn_signed, float(diff_norms.mean()), float(diff_norms.std()),
        float(diff_norms.max()), float(n),
        float(c_count), float(h_count), float(n_count), float(o_count),
        # bond-angle statistics (10 features)
        aR_mean, aR_std, aR_min, aR_max,
        aP_mean, aP_std, aP_min, aP_max,
        ang_change_mean, ang_change_max,
    ], dtype=np.float32)
    return feats

ATOM_PHYS_DIM = 3  # electronegativity, atomic number, mass
ENERGY_FEAT_DIM = 20  # reaction energetics + composition + bond-angle statistics

def build_atom_physical_features(atom_types, n, max_atoms):
    """Per-atom physical descriptors: [EN, Z, Mass] for each atom, zero-padded.

    These features are attached directly to each atom node in PSICore so the
    Transformer can reason about *which* atom has which property spatially,
    rather than receiving only global min/max/mean statistics.
    """
    feats = np.zeros((max_atoms, ATOM_PHYS_DIM), dtype=np.float32)
    for i in range(n):
        t = atom_types[i]
        feats[i] = [electronegativity(t), float(atomic_number(t)), atomic_mass(t)]
    return feats

def compute_distance_matrix(coords):
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=-1) + 1e-8)
    return dist.astype(np.float32)

def mds(D, dim=3):
    n = D.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * H @ (D ** 2) @ H
    evals, evecs = np.linalg.eigh(B)
    idx = np.argsort(evals)[::-1]
    evals = evals[idx]
    evecs = evecs[:, idx]
    n_dims = min(dim, n)
    X = evecs[:, :n_dims] @ np.diag(np.sqrt(np.maximum(evals[:n_dims], 0)))
    if n_dims < dim:
        X = np.pad(X, ((0, 0), (0, dim - n_dims)))
    return X

def kabsch(P, Q):
    """Align P onto Q using the Kabsch algorithm.

    Includes reflection correction without re-running SVD.
    """
    P_centered = P - P.mean(axis=0)
    Q_centered = Q - Q.mean(axis=0)
    C = P_centered.T @ Q_centered
    V, _, W = np.linalg.svd(C)
    if np.linalg.det(V @ W) < 0.0:
        V[:, -1] *= -1.0
    R = V @ W
    return P_centered @ R + Q.mean(axis=0)

def connected_components(adjacency):
    seen = set()
    fragments = []
    for start in range(len(adjacency)):
        if start in seen:
            continue
        stack = [start]
        seen.add(start)
        frag = []
        while stack:
            node = stack.pop()
            frag.append(node)
            for nbr in adjacency[node]:
                if nbr not in seen:
                    seen.add(nbr)
                    stack.append(nbr)
        fragments.append(sorted(frag))
    return sorted(fragments, key=lambda frag: (frag[0], len(frag)))

def find_fragments_from_coords(coords, atom_types, n, bond_scale=1.45):
    adjacency = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            cutoff = bond_scale * (covalent_radius(atom_types[i]) + covalent_radius(atom_types[j]))
            if np.linalg.norm(coords[i] - coords[j]) <= cutoff:
                adjacency[i].append(j)
                adjacency[j].append(i)
    return connected_components(adjacency)

def find_fragments_from_distances(D, atom_types, bond_scale=1.45):
    n = len(atom_types)
    adjacency = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            cutoff = bond_scale * (covalent_radius(atom_types[i]) + covalent_radius(atom_types[j]))
            if D[i, j] <= cutoff:
                adjacency[i].append(j)
                adjacency[j].append(i)
    return connected_components(adjacency)

def geometry_pair_mask_from_fragments(fragments, max_atoms, include_diagonal=False):
    geom_mask = np.zeros((max_atoms, max_atoms), dtype=np.float32)
    for frag in fragments:
        idx = np.array(frag, dtype=np.int64)
        geom_mask[np.ix_(idx, idx)] = 1.0
    if not include_diagonal:
        np.fill_diagonal(geom_mask, 0.0)
    return geom_mask

def choose_alignment_fragments(c_R, c_P, atom_types, n, bond_scale=1.45):
    frags_R = find_fragments_from_coords(c_R, atom_types, n, bond_scale)
    frags_P = find_fragments_from_coords(c_P, atom_types, n, bond_scale)
    if len(frags_P) > len(frags_R):
        return frags_P
    if len(frags_R) > len(frags_P):
        return frags_R
    return frags_R

def kabsch_align_reactant_fragments(c_R, c_P, atom_types, n, bond_scale=1.45):
    c_R_aligned = c_R.copy()
    fragments = choose_alignment_fragments(c_R, c_P, atom_types, n, bond_scale)
    for frag in fragments:
        idx = np.array(frag, dtype=np.int64)
        if len(idx) >= 2:
            c_R_aligned[idx] = kabsch(c_R[idx], c_P[idx])
        else:
            c_R_aligned[idx] = c_P[idx]
    return c_R_aligned

def mds_by_fragments(D, atom_types=None, fragments=None, reference_coords=None, dim=3, bond_scale=1.45):
    n = D.shape[0]
    if fragments is None:
        if atom_types is None:
            fragments = [list(range(n))]
        else:
            fragments = find_fragments_from_distances(D, atom_types, bond_scale)
    X = np.zeros((n, dim), dtype=np.float64)
    cursor = 0.0
    for frag in fragments:
        idx = np.array(frag, dtype=np.int64)
        if len(idx) >= 2:
            frag_coords = mds(D[np.ix_(idx, idx)], dim=dim)
        else:
            frag_coords = np.zeros((1, dim), dtype=np.float64)
        if reference_coords is not None:
            ref = reference_coords[idx]
            if len(idx) >= 2:
                frag_coords = kabsch(frag_coords, ref)
            else:
                frag_coords[0] = ref[0]
        else:
            frag_coords = frag_coords - frag_coords.mean(axis=0)
            span = np.ptp(frag_coords[:, 0]) if len(idx) > 1 else 0.0
            frag_coords[:, 0] += cursor - frag_coords[:, 0].min()
            cursor += max(span, 1.5) + 3.0
        X[idx] = frag_coords
    return X.astype(np.float32)

STERIC_FLOOR_FRAC = 0.75

def clamp_steric_collisions(pred_dist, atom_types, floor_frac=STERIC_FLOOR_FRAC):
    n = len(atom_types)
    for i in range(n):
        for j in range(i + 1, n):
            r_i = covalent_radius(atom_types[i])
            r_j = covalent_radius(atom_types[j])
            min_d = floor_frac * (r_i + r_j)
            if pred_dist[i, j] < min_d:
                pred_dist[i, j] = min_d
                pred_dist[j, i] = min_d
    return pred_dist

def classify_bonds(D_R, D_P, n, threshold=0.15):
    active, spectator = [], []
    for i in range(n):
        for j in range(i + 1, n):
            if abs(D_R[i, j] - D_P[i, j]) > threshold:
                active.append((i, j))
            else:
                spectator.append((i, j))
    return active, spectator

def apply_spectator_constraints(pred_dist, D_R, D_P, n, threshold=0.15, tol=0.05, pair_mask=None):
    _, spectator = classify_bonds(D_R, D_P, n, threshold)
    for (i, j) in spectator:
        if pair_mask is not None and pair_mask[i, j] <= 0:
            continue
        d_ref = (D_R[i, j] + D_P[i, j]) / 2.0
        lo = d_ref * (1.0 - tol)
        hi = d_ref * (1.0 + tol)
        clamped = float(np.clip(pred_dist[i, j], lo, hi))
        pred_dist[i, j] = clamped
        pred_dist[j, i] = clamped
    return pred_dist

def enforce_triangle_inequality(D, fragments=None, tol=0.05):
    D = D.copy()
    n = D.shape[0]
    if fragments is None:
        fragments = [list(range(n))]
    for frag in fragments:
        idx = np.array(frag, dtype=np.int64)
        if len(idx) < 3:
            continue
        sub = D[np.ix_(idx, idx)].copy()
        m = len(idx)
        for k in range(m):
            for i in range(m):
                for j in range(m):
                    shortcut = sub[i, k] + sub[k, j]
                    if sub[i, j] - shortcut > tol:
                        sub[i, j] = sub[j, i] = shortcut
        D[np.ix_(idx, idx)] = sub
    return D

def validate_ts_geometry(pred_dist, D_R, D_P, atom_types, n, spectator_threshold=0.15):
    issues = []
    for i in range(n):
        for j in range(i + 1, n):
            r_i = covalent_radius(atom_types[i])
            r_j = covalent_radius(atom_types[j])
            min_d = STERIC_FLOOR_FRAC * (r_i + r_j)
            if pred_dist[i, j] < min_d:
                issues.append(f"  STERIC   {atom_types[i]}{i}-{atom_types[j]}{j}: {pred_dist[i,j]:.3f} Å < floor {min_d:.3f} Å")
    active, _ = classify_bonds(D_R, D_P, n, spectator_threshold)
    for (i, j) in active:
        d_lo = min(D_R[i, j], D_P[i, j]) - 0.30
        d_hi = max(D_R[i, j], D_P[i, j]) + 0.30
        if not (d_lo <= pred_dist[i, j] <= d_hi):
            issues.append(f"  ACT_OOB  {atom_types[i]}{i}-{atom_types[j]}{j}: pred={pred_dist[i,j]:.3f} Å  (R={D_R[i,j]:.3f}, P={D_P[i,j]:.3f})")
    if issues:
        print(f"[TS Validation] {len(issues)} issue(s) detected:")
        for iss in issues:
            print(iss)
    else:
        print("[TS Validation] Geometry passed all physical checks.")
    return len(issues) == 0

def padded_coords(atoms, max_atoms):
    coords = np.zeros((max_atoms, 3), dtype=np.float32)
    for i, atom in enumerate(atoms):
        coords[i] = [atom["x"], atom["y"], atom["z"]]
    return coords

def parse_log_content(file_content):
    atoms = []
    energy = None
    lines = file_content.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if "Standard Nuclear Orientation" in line:
            current_atoms = []
            i += 3
            while i < len(lines) and not lines[i].strip().startswith("---"):
                parts = lines[i].split()
                if len(parts) == 5:
                    current_atoms.append({
                        "atom": parts[1],
                        "x": float(parts[2]), "y": float(parts[3]), "z": float(parts[4])
                    })
                i += 1
            atoms = current_atoms
        elif "Final energy is" in line:
            energy = float(line.split()[-1])
        elif "Total energy in the final basis set =" in line:
            energy = float(line.split()[-1])
        i += 1
    return {"energy": energy, "atoms": atoms}

def load_log_file(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        parsed = parse_log_content(f.read())
    if not parsed["atoms"]:
        raise ValueError(f"No atoms found in {path}")
    return parsed

def write_xyz(path, atom_types, coords, comment):
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"{len(atom_types)}\n")
        f.write(f"{comment}\n")
        for atom, (x, y, z) in zip(atom_types, coords):
            f.write(f"{atom:<2} {x: .8f} {y: .8f} {z: .8f}\n")

def masked_mae(A, B, mask=None):
    diff = np.abs(np.array(A) - np.array(B))
    if mask is None:
        return float(diff.mean())
    mask = np.array(mask, dtype=np.float64)
    return float((diff * mask).sum() / max(mask.sum(), 1.0))

def fragments_from_mask(mask):
    mask = np.array(mask)
    n = mask.shape[0]
    adjacency = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if mask[i, j] > 0:
                adjacency[i].append(j)
                adjacency[j].append(i)
    return connected_components(adjacency)

def get_bonds_from_distances(D, atom_types, fragments=None, bond_scale=1.45):
    bonds = []
    n = len(atom_types)
    allowed = np.zeros((n, n), dtype=bool)
    if fragments is None:
        allowed[:, :] = True
    else:
        for frag in fragments:
            idx = np.array(frag, dtype=np.int64)
            allowed[np.ix_(idx, idx)] = True
    for i in range(n):
        for j in range(i+1, n):
            if not allowed[i, j]:
                continue
            r_i = covalent_radius(atom_types[i])
            r_j = covalent_radius(atom_types[j])
            if D[i, j] < bond_scale * (r_i + r_j):
                bonds.append((i, j))
    return bonds

# =============================================================================
# Training / prediction pipeline (formerly psi_full_pipeline.py)
# =============================================================================

CONFIG = {
    "dataset_json": "extracted_dataset.json",
    "save_dir": ".",
    "target_reactions": 5000,
    "max_atoms": 30,
    "n_gaussians": 32,
    "gauss_start": 0.4,
    "gauss_stop": 6.0,
    "atom_embed_dim": 32,
    "gru_hidden": 128,
    "gru_layers": 2,
    "gru_dropout": 0.3,
    "attn_heads": 8,
    "attn_layers": 3,
    "ff_dim": 512,
    "dropout": 0.35,
    "delta_clamp": 3.0,
    # --- EGNN coordinate refiner -----------------------------------------
    # After the geometry head predicts a TS distance matrix, we embed it to 3D
    # (differentiable MDS) and refine the coordinates with an E(n)-equivariant
    # GNN that consumes a per-atom chemical-property vector + the TS coords.
    "egnn_enabled": True,
    "egnn_layers": 4,
    "egnn_hidden": 128,
    "egnn_coord_clamp": 2.0,  # max per-step coordinate displacement (Angstrom)
    "geom_coarse_weight": 0.5,  # weight on the pre-EGNN (coarse) distance aux loss
    # --- Learned activation-energy (Ea) head -----------------------------
    # A small head consumes the EGNN's refined per-atom features (h_ts) + the
    # signed reaction energy and regresses Ea. Trained jointly with geometry but
    # only after a warmup, so its gradient reshapes the EGNN ("learn physics")
    # once predicted TS geometries are good enough to learn from. PhysicsEa
    # (Marcus/Hammond/OLS) is kept as a side-by-side baseline.
    "ea_loss_weight": 1.0,     # weight on the Ea loss (computed on normalized Ea)
    "ea_warmup_epochs": 200,   # geometry-only until here, then the Ea loss turns on
    "ea_select_weight": 0.25,  # Ea contribution to checkpoint selection (post-warmup)
    "ea_head_dropout": 0.35,   # dropout inside the Ea head MLP
    "lr": 1.5e-4,
    "weight_decay": 1e-2,
    "warmup_epochs": 40,
    "grad_clip": 1.0,
    "batch_size": 32,
    "num_workers": 2,
    "pin_memory": True,
    "device": "auto",
    "require_cuda": False,
    "amp": True,
    "epochs": 1500,
    "print_every": 25,
    "val_split": 0.2,
    "split_seed": 42,
    "patience": 120,
    "coord_noise_std": 0.05,
    "spectator_threshold": 0.15,
    "spectator_tol": 0.05,
    "fragment_bond_scale": 1.45,
    "hartree_to_kcal": 627.509,
    "skip_negative_ea": True,
}

def resolve_device(config):
    requested = config["device"].lower()
    if requested == "auto":
        if config["require_cuda"]:
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is required but not available!")
            return torch.device("cuda")
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is requested but not available!")
    return device

def configure_torch_runtime(device):
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

def move_batch_to_device(batch, device):
    return (
        batch["D_R"].to(device, non_blocking=True),
        batch["D_I"].to(device, non_blocking=True),
        batch["D_P"].to(device, non_blocking=True),
        batch["D_TS"].to(device, non_blocking=True),
        batch["mask"].to(device, non_blocking=True),
        batch["geom_mask"].to(device, non_blocking=True),
        batch["atom_ids"].to(device, non_blocking=True),
        batch["atom_phys"].to(device, non_blocking=True),
        batch["Ea"].to(device, non_blocking=True),
        batch["de_rxn"].to(device, non_blocking=True),
        batch["energy_feats"].to(device, non_blocking=True),
    )

def extract_raw_data(config):
    if not os.path.exists(config["dataset_json"]):
        raise FileNotFoundError(f"Dataset file {config['dataset_json']} not found! Please upload it to the cloud environment.")
    print(f"Using pre-extracted dataset at {config['dataset_json']}.")

def build_atom_vocab(raw_data):
    atom_set = set()
    for entry in raw_data:
        for a in entry["atoms"]:
            atom_set.add(a["atom"])
    sorted_atoms = sorted(atom_set)
    vocab = {atom: i + 1 for i, atom in enumerate(sorted_atoms)}
    print(f"Atom vocabulary ({len(vocab)} types): {vocab}")
    return vocab

def build_reaction_samples(config):
    """Parse the dataset JSON once and build raw (un-normalized) samples.

    Targets that don't depend on coordinate augmentation -- the true TS distance
    matrix and the fragment geometry mask (both derived from the unaugmented
    TS coords) -- are precomputed here so __getitem__ stays cheap.
    """
    with open(config["dataset_json"], "r") as f:
        raw_data = json.load(f)
    atom_vocab = build_atom_vocab(raw_data)
    reactions = {}
    for entry in raw_data:
        parts = entry["filename"].split("/")
        if len(parts) < 3: continue
        rxn_id = parts[1]
        prefix = parts[2].lower()
        role = "r" if prefix.startswith("r") else "p" if prefix.startswith("p") else "ts" if prefix.startswith("ts") else None
        if not role:
            raise ValueError(f"Could not classify role for entry: {entry['filename']}")
        if rxn_id not in reactions: reactions[rxn_id] = {}
        reactions[rxn_id][role] = entry
    samples = []
    atom_types_map = {}
    target_reactions = config.get("target_reactions", float("inf"))
    for rxn_id, roles in sorted(reactions.items()):
        if len(samples) >= target_reactions:
            break
        if "r" in roles and "p" in roles and "ts" in roles:
            r_e = roles["r"]; p_e = roles["p"]; ts_e = roles["ts"]
            n = len(ts_e["atoms"])
            if n > config["max_atoms"]: continue
            c_R = padded_coords(r_e["atoms"], config["max_atoms"])
            c_P = padded_coords(p_e["atoms"], config["max_atoms"])
            c_TS = padded_coords(ts_e["atoms"], config["max_atoms"])
            atom_ids = np.zeros(config["max_atoms"], dtype=np.int64)
            for i, a in enumerate(ts_e["atoms"]):
                atom_ids[i] = atom_vocab[a["atom"]]
            mask = np.zeros(config["max_atoms"], dtype=np.float32)
            mask[:n] = 1.0
            ea = (ts_e["energy"] - max(r_e["energy"], p_e["energy"])) * config["hartree_to_kcal"]
            if config.get("skip_negative_ea", True) and ea < 0:
                continue
            e_r = r_e["energy"] * config["hartree_to_kcal"]
            e_p = p_e["energy"] * config["hartree_to_kcal"]
            atom_types = [a["atom"] for a in ts_e["atoms"]]
            c_R_aligned_init = kabsch_align_reactant_fragments(
                c_R, c_P, atom_types, n, config["fragment_bond_scale"]
            )
            energy_feats = build_energy_features(
                atom_types, n, c_R_aligned_init, c_P, e_r, e_p, config["fragment_bond_scale"]
            )
            atom_phys = build_atom_physical_features(
                atom_types, n, config["max_atoms"]
            )
            D_TS = compute_distance_matrix(c_TS)
            ts_fragments = find_fragments_from_coords(
                c_TS, atom_types, n, config["fragment_bond_scale"]
            )
            geom_mask = geometry_pair_mask_from_fragments(ts_fragments, config["max_atoms"])
            atom_types_map[rxn_id] = atom_types
            samples.append({
                "rxn_id": rxn_id, "n_atoms": n,
                "c_R": c_R, "c_P": c_P,
                "atom_types": atom_types,
                "atom_ids": torch.from_numpy(atom_ids),
                "mask": torch.from_numpy(mask),
                "Ea_raw": ea,
                "de_rxn_raw": float(e_p - e_r),  # signed reaction energy (kcal/mol), BEP driver
                "energy_feats_raw": energy_feats,
                "atom_phys_raw": atom_phys,
                "D_TS": torch.from_numpy(D_TS),
                "geom_mask": torch.from_numpy(geom_mask),
            })
    print(f"Loaded {len(samples)} complete reaction triplets.")
    return samples, atom_vocab, atom_types_map

def compute_normalization(samples, indices):
    """Compute atom-phys + Ea + de_rxn normalization stats over the given indices.

    Restricting to the training indices keeps validation reactions out of the
    normalization statistics.  Ea and de_rxn are normalized (z-scored) so the
    learned Ea head regresses a well-scaled target/input; the physics Ea
    baseline is unaffected (it reads raw kcal/mol values directly).
    """
    # Atom-physics normalization: collect all *valid* (non-padding) atom rows
    # across training samples and compute per-feature mean/std.
    all_aphys_rows = []
    for i in indices:
        s = samples[i]
        n = s["n_atoms"]
        all_aphys_rows.append(s["atom_phys_raw"][:n])  # (n, 3)
    all_aphys = np.concatenate(all_aphys_rows, axis=0)   # (total_atoms, 3)
    aphys_mean = all_aphys.mean(axis=0).astype(np.float32)
    aphys_std = all_aphys.std(axis=0).astype(np.float32)
    aphys_std[aphys_std < 1e-6] = 1.0
    # Ea z-score stats (target for the learned head).
    all_ea = np.array([samples[i]["Ea_raw"] for i in indices], dtype=np.float64)
    ea_mean = float(all_ea.mean())
    ea_std = float(all_ea.std())
    if ea_std < 1e-6:
        ea_std = 1.0
    # de_rxn z-score stats (input feature to the head).
    all_de = np.array([samples[i]["de_rxn_raw"] for i in indices], dtype=np.float64)
    de_rxn_mean = float(all_de.mean())
    de_rxn_std = float(all_de.std())
    if de_rxn_std < 1e-6:
        de_rxn_std = 1.0
    print(f"Ea stats (train split): mean={ea_mean:.2f}, std={ea_std:.2f} kcal/mol")
    print(f"Ea range (train split): [{all_ea.min():.2f}, {all_ea.max():.2f}] kcal/mol")
    print(f"de_rxn stats (train split): mean={de_rxn_mean:.2f}, std={de_rxn_std:.2f} kcal/mol")
    print(f"Atom-phys stats (train): mean={aphys_mean}, std={aphys_std}")
    # Energy-feature normalization: z-score the 20D reaction descriptor vector.
    all_efeats = np.array([samples[i]["energy_feats_raw"] for i in indices], dtype=np.float32)
    efeat_mean = all_efeats.mean(axis=0).astype(np.float32)
    efeat_std = all_efeats.std(axis=0).astype(np.float32)
    efeat_std[efeat_std < 1e-6] = 1.0
    print(f"Energy-feats stats (train): mean_range=[{efeat_mean.min():.2f}, {efeat_mean.max():.2f}]")
    return {
        "aphys_mean": aphys_mean,
        "aphys_std": aphys_std,
        "ea_mean": ea_mean,
        "ea_std": ea_std,
        "de_rxn_mean": de_rxn_mean,
        "de_rxn_std": de_rxn_std,
        "efeat_mean": efeat_mean,
        "efeat_std": efeat_std,
    }

class ReactionDataset(Dataset):
    """Thin view over a shared list of prebuilt samples.

    Multiple views (e.g. augmented train vs. clean eval) share the same sample
    list and normalization stats; only the `augment` flag differs.
    Returns the raw Ea target (kcal/mol) and the z-scored de_rxn feature for the
    learned Ea head; Ea is normalized inside the training loop using ea_mean/std.
    """
    def __init__(self, config, samples, atom_vocab, atom_types_map, stats, augment=False):
        self.config = config
        self.samples = samples
        self.atom_vocab = atom_vocab
        self.atom_types_map = atom_types_map
        self.augment = augment
        self.aphys_mean = stats["aphys_mean"]
        self.aphys_std = stats["aphys_std"]
        self.de_rxn_mean = stats["de_rxn_mean"]
        self.de_rxn_std = stats["de_rxn_std"]
        self.efeat_mean = stats["efeat_mean"]
        self.efeat_std = stats["efeat_std"]

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        n = s["n_atoms"]
        c_R = s["c_R"].copy()
        c_P = s["c_P"].copy()
        if self.augment:
            noise_std = self.config["coord_noise_std"]
            c_R[:n] += np.random.randn(n, 3).astype(np.float32) * noise_std
            c_P[:n] += np.random.randn(n, 3).astype(np.float32) * noise_std
        # Distance matrices are rotation/translation invariant, so no alignment
        # of the coordinates is needed before computing them.
        D_R = compute_distance_matrix(c_R)
        D_P = compute_distance_matrix(c_P)
        D_I = (D_R + D_P) / 2.0
        aphys_norm = (s["atom_phys_raw"] - self.aphys_mean) / self.aphys_std
        de_rxn_norm = (s["de_rxn_raw"] - self.de_rxn_mean) / self.de_rxn_std
        efeat_norm = (s["energy_feats_raw"] - self.efeat_mean) / self.efeat_std
        return {
            "rxn_id": s["rxn_id"],
            "n_atoms": n,
            "D_R": torch.from_numpy(D_R),
            "D_I": torch.from_numpy(D_I),
            "D_P": torch.from_numpy(D_P),
            "D_TS": s["D_TS"],
            "mask": s["mask"],
            "geom_mask": s["geom_mask"],
            "atom_ids": s["atom_ids"],
            "atom_phys": torch.from_numpy(aphys_norm.astype(np.float32)),
            "Ea": torch.tensor(s["Ea_raw"], dtype=torch.float32),
            "de_rxn": torch.tensor(de_rxn_norm, dtype=torch.float32),
            "energy_feats": torch.from_numpy(efeat_norm.astype(np.float32)),
        }

class GaussianEmbedding(nn.Module):
    def __init__(self, n_gaussians=50, start=0.4, stop=6.0):
        super().__init__()
        centers = torch.linspace(start, stop, n_gaussians)
        self.register_buffer("centers", centers)
        self.sigma = (stop - start) / (n_gaussians - 1) * 0.5

    def forward(self, D):
        return torch.exp(-0.5 * ((D.unsqueeze(-1) - self.centers) / self.sigma) ** 2)

class PreNormTransformerLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x, src_key_padding_mask=None):
        x2 = self.norm1(x)
        x2, _ = self.attn(x2, x2, x2, key_padding_mask=src_key_padding_mask)
        x = x + x2
        x2 = self.norm2(x)
        x = x + self.ff(x2)
        return x

class PSICore(nn.Module):
    def __init__(self, config, num_atom_types):
        super().__init__()
        N = config["max_atoms"]
        K = config["n_gaussians"]
        atom_dim = config["atom_embed_dim"]
        gru_hidden = config["gru_hidden"]
        d_model = gru_hidden * 2
        self.atom_embed = nn.Embedding(num_atom_types + 1, atom_dim, padding_idx=0)
        self.gaussian = GaussianEmbedding(K, config["gauss_start"], config["gauss_stop"])
        # Per-atom feature width = learnable embedding + physical descriptors (EN, Z, Mass)
        atom_feat_dim = atom_dim + ATOM_PHYS_DIM
        self.input_proj = nn.Sequential(
            nn.Linear(N * K + atom_feat_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(config["dropout"]),
        )
        self.gru = nn.GRU(
            input_size=d_model,
            hidden_size=gru_hidden,
            num_layers=config["gru_layers"],
            batch_first=True,
            bidirectional=True,
            dropout=config["gru_dropout"] if config["gru_layers"] > 1 else 0.0,
        )
        self.gru_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
        )
        self.transformer_layers = nn.ModuleList([
            PreNormTransformerLayer(d_model, config["attn_heads"], config["ff_dim"], config["dropout"])
            for _ in range(config["attn_layers"])
        ])
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, D_R, D_I, D_P, mask, atom_ids, atom_phys):
        B, N, _ = D_R.shape
        atom_emb = self.atom_embed(atom_ids)
        # Concatenate learnable embedding with explicit physical descriptors
        atom_feat = torch.cat([atom_emb, atom_phys], dim=-1)  # [B, N, atom_dim + 3]
        emb_R = self.gaussian(D_R).view(B, N, -1)
        emb_I = self.gaussian(D_I).view(B, N, -1)
        emb_P = self.gaussian(D_P).view(B, N, -1)
        emb_R = torch.cat([emb_R, atom_feat], dim=-1)
        emb_I = torch.cat([emb_I, atom_feat], dim=-1)
        emb_P = torch.cat([emb_P, atom_feat], dim=-1)
        emb_R = self.input_proj(emb_R)
        emb_I = self.input_proj(emb_I)
        emb_P = self.input_proj(emb_P)
        seq = torch.stack([emb_R, emb_I, emb_P], dim=2).view(B * N, 3, -1)
        out, _ = self.gru(seq)
        context = out[:, 1, :].view(B, N, -1)
        context = self.gru_proj(context)
        pad_mask = (mask == 0)
        x = context
        for layer in self.transformer_layers:
            x = layer(x, src_key_padding_mask=pad_mask)
        return self.final_norm(x)

class GeometryHead(nn.Module):
    def __init__(self, d_model, atom_embed_dim, atom_phys_dim=ATOM_PHYS_DIM, dropout=0.25, delta_clamp=3.0):
        super().__init__()
        self.delta_clamp = delta_clamp
        # Pairwise features: transformer features (i,j) + atom embeddings (i,j)
        # + physical descriptors (i,j) + raw distances (R,I,P)
        pair_dim = d_model * 2 + (atom_embed_dim + atom_phys_dim) * 2 + 3
        self.net = nn.Sequential(
            nn.Linear(pair_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 2),
        )
        # Initialize: alpha_logit=0 → sigmoid=0.5 (starts at midpoint), delta=0
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, features, atom_emb, atom_phys, D_R, D_I, D_P, mask):
        B, N, D = features.shape
        # Concatenate learnable atom embedding with physical descriptors
        atom_feat = torch.cat([atom_emb, atom_phys], dim=-1)  # [B, N, atom_dim + 3]
        atom_feat_dim = atom_feat.shape[-1]
        fi = features.unsqueeze(2).expand(B, N, N, D)
        fj = features.unsqueeze(1).expand(B, N, N, D)
        ai = atom_feat.unsqueeze(2).expand(B, N, N, atom_feat_dim)
        aj = atom_feat.unsqueeze(1).expand(B, N, N, atom_feat_dim)
        pair_dist = torch.stack([D_R, D_I, D_P], dim=-1)
        pair = torch.cat([fi, fj, ai, aj, pair_dist], dim=-1)
        out = self.net(pair)
        alpha = torch.sigmoid(out[..., 0])
        delta = torch.clamp(out[..., 1], min=-self.delta_clamp, max=self.delta_clamp)
        D_base = alpha * D_R + (1.0 - alpha) * D_P
        D_TS_pred = torch.clamp(D_base + delta, min=0.0)
        D_TS_pred = (D_TS_pred + D_TS_pred.transpose(1, 2)) / 2.0
        eye = torch.eye(N, device=D_TS_pred.device, dtype=D_TS_pred.dtype).unsqueeze(0)
        valid = mask.unsqueeze(-1) * mask.unsqueeze(-2)
        return D_TS_pred * (1.0 - eye) * valid

# =============================================================================
# Physics-based activation energy (replaces NN EnergyHead + EnergyRefiner)
# =============================================================================

# Approximate harmonic bond force constants in kcal/(mol·Å²).
# Derived from simplified Badger's rule and UFF-like parameters.
# These are rough but sufficient for estimating the reorganization energy λ.
BOND_FORCE_CONSTANTS = {
    ('C', 'C'): 600.0,  ('C', 'H'): 700.0,  ('C', 'N'): 650.0,
    ('C', 'O'): 750.0,  ('C', 'F'): 800.0,  ('C', 'S'): 400.0,
    ('C', 'Cl'): 450.0, ('C', 'Br'): 350.0, ('C', 'I'): 300.0,
    ('C', 'P'): 400.0,  ('C', 'Si'): 350.0, ('C', 'B'): 500.0,
    ('N', 'H'): 750.0,  ('N', 'N'): 600.0,  ('N', 'O'): 700.0,
    ('O', 'H'): 800.0,  ('O', 'O'): 600.0,  ('S', 'H'): 500.0,
    ('S', 'S'): 350.0,  ('S', 'O'): 550.0,  ('S', 'N'): 450.0,
    ('P', 'O'): 500.0,  ('P', 'H'): 400.0,  ('P', 'N'): 400.0,
    ('Si', 'H'): 400.0, ('Si', 'O'): 500.0, ('Si', 'N'): 400.0,
    ('B', 'H'): 500.0,  ('B', 'O'): 600.0,  ('B', 'N'): 550.0,
    ('H', 'H'): 750.0,  ('F', 'H'): 900.0,  ('Cl', 'H'): 550.0,
    ('Br', 'H'): 450.0, ('I', 'H'): 350.0,
}

def estimate_bond_force_constant(a1, a2):
    """Look up an approximate force constant k (kcal/(mol·Å²)) for a bond pair.

    Falls back to a geometric-mean estimate from covalent radii via Badger's
    rule when the pair isn't in the explicit table.
    """
    key = (a1, a2) if (a1, a2) in BOND_FORCE_CONSTANTS else (a2, a1)
    if key in BOND_FORCE_CONSTANTS:
        return BOND_FORCE_CONSTANTS[key]
    # Badger-like fallback: k ∝ 1 / (r_1 + r_2)^3, scaled to ~500 kcal/(mol·Å²)
    r_sum = covalent_radius(a1) + covalent_radius(a2)
    return 500.0 / max(r_sum, 0.5) ** 3


def compute_reorganization_energy(coords_R, coords_TS, coords_P, atom_types, n,
                                   bond_scale=1.45):
    """Geometric reorganization energy λ from bond-stretching displacements.

    For every bond in the TS geometry, accumulates:
        λ += 0.5 · k_ij · [(d_TS - d_R)² + (d_P - d_TS)²]

    This captures how much bond stretching/compression the molecule undergoes
    along the R → TS → P reaction coordinate.  Units: kcal/mol.
    """
    D_R = np.zeros((n, n), dtype=np.float64)
    D_TS = np.zeros((n, n), dtype=np.float64)
    D_P = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            dr = np.linalg.norm(coords_R[i] - coords_R[j])
            dts = np.linalg.norm(coords_TS[i] - coords_TS[j])
            dp = np.linalg.norm(coords_P[i] - coords_P[j])
            D_R[i, j] = D_R[j, i] = dr
            D_TS[i, j] = D_TS[j, i] = dts
            D_P[i, j] = D_P[j, i] = dp

    # Use bonds from the *union* of R, TS, and P topologies so we capture both
    # forming and breaking bonds.
    bonds_set = set()
    for D in (D_R, D_TS, D_P):
        for i in range(n):
            for j in range(i + 1, n):
                cutoff = bond_scale * (covalent_radius(atom_types[i]) + covalent_radius(atom_types[j]))
                if D[i, j] <= cutoff:
                    bonds_set.add((i, j))

    lam = 0.0
    for (i, j) in bonds_set:
        k_ij = estimate_bond_force_constant(atom_types[i], atom_types[j])
        dr_fwd = D_TS[i, j] - D_R[i, j]   # R → TS displacement
        dr_rev = D_P[i, j] - D_TS[i, j]   # TS → P displacement
        lam += 0.5 * k_ij * (dr_fwd ** 2 + dr_rev ** 2)
    return lam


def hammond_index(coords_R, coords_TS, coords_P, atom_types, n,
                  threshold=0.15, bond_scale=1.45):
    """Hammond postulate index η ∈ [0, 1] from active-bond displacements.

    η ≈ 0: TS resembles reactant (early, exothermic).
    η ≈ 1: TS resembles product  (late, endothermic).

    Computed over the active bonds (those whose distance changes by more than
    `threshold` Å between R and P).
    """
    sum_r, sum_p = 0.0, 0.0
    D_R = np.zeros((n, n), dtype=np.float64)
    D_TS = np.zeros((n, n), dtype=np.float64)
    D_P = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            D_R[i, j] = np.linalg.norm(coords_R[i] - coords_R[j])
            D_TS[i, j] = np.linalg.norm(coords_TS[i] - coords_TS[j])
            D_P[i, j] = np.linalg.norm(coords_P[i] - coords_P[j])

    active_count = 0
    for i in range(n):
        for j in range(i + 1, n):
            # Only look at bonds in the reactive region
            cutoff = bond_scale * (covalent_radius(atom_types[i]) + covalent_radius(atom_types[j]))
            is_bonded = (D_R[i, j] <= cutoff or D_TS[i, j] <= cutoff or D_P[i, j] <= cutoff)
            if not is_bonded:
                continue
            if abs(D_R[i, j] - D_P[i, j]) > threshold:
                sum_r += abs(D_TS[i, j] - D_R[i, j])
                sum_p += abs(D_TS[i, j] - D_P[i, j])
                active_count += 1

    if active_count == 0 or (sum_r + sum_p) < 1e-8:
        return 0.5  # no active bonds → ambiguous, return midpoint
    return sum_r / (sum_r + sum_p)


class PhysicsEaCalculator:
    """Physics-based activation energy from 3D TS geometry.

    Replaces the NN-based EnergyHead + EnergyRefiner (~80K parameters) with
    actual chemistry equations and only 4 OLS-fitted scaling parameters:

        Ea_pred = a * Ea_marcus + b * η_hammond + c * ΔE_rxn + d

    where:
        - Ea_marcus = (λ / 4) · (1 + ΔE / λ)² (Marcus theory)
        - λ = geometric reorganization energy from bond displacements
        - η = Hammond postulate index (early vs late TS)
        - ΔE_rxn = E_product - E_reactant (signed reaction energy)

    The model focuses 100% on predicting the best TS geometry; Ea follows
    from physics applied to those predicted 3D coordinates.
    """
    def __init__(self, bond_scale=1.45, spectator_threshold=0.15):
        self.bond_scale = bond_scale
        self.threshold = spectator_threshold
        # OLS coefficients: fit by .fit() on training data
        self.coeffs = None   # [a, b, c, d] for the 4 features
        self.fitted = False

    def _compute_features(self, coords_R, coords_TS, coords_P, atom_types, n,
                          de_rxn):
        """Compute the 4 physics-based features for one reaction."""
        lam = compute_reorganization_energy(
            coords_R, coords_TS, coords_P, atom_types, n, self.bond_scale
        )
        eta = hammond_index(
            coords_R, coords_TS, coords_P, atom_types, n,
            self.threshold, self.bond_scale
        )
        # Marcus theory: Ea = (λ/4)(1 + ΔE/λ)²
        if lam > 1e-6:
            ea_marcus = (lam / 4.0) * (1.0 + de_rxn / lam) ** 2
        else:
            # Zero reorganization → pure BEP-like
            ea_marcus = abs(de_rxn) * 0.5
        return np.array([ea_marcus, eta, de_rxn, 1.0], dtype=np.float64)

    def compute_features_batch(self, samples, coords_TS_list, config):
        """Compute physics features for a list of samples + predicted TS coords.

        Args:
            samples: list of sample dicts (with c_R, c_P, atom_types, n_atoms, Ea_raw)
            coords_TS_list: list of (n, 3) numpy arrays with predicted TS coords
            config: pipeline config dict
        Returns:
            X: (N, 4) feature matrix
            y: (N,) true Ea values
        """
        X, y = [], []
        hartree_to_kcal = config["hartree_to_kcal"]
        for s, coords_ts in zip(samples, coords_TS_list):
            n = s["n_atoms"]
            c_R = np.asarray(s["c_R"][:n], dtype=np.float64)
            c_P = np.asarray(s["c_P"][:n], dtype=np.float64)
            c_TS = np.asarray(coords_ts[:n], dtype=np.float64)
            atom_types = s["atom_types"]
            # de_rxn is the signed reaction energy in kcal/mol
            # It was computed as e_p - e_r at build time; reconstruct from energy_feats
            de_rxn = float(s["energy_feats_raw"][1])  # index 1 = de_rxn_signed
            feats = self._compute_features(c_R, c_TS, c_P, atom_types, n, de_rxn)
            X.append(feats)
            y.append(s["Ea_raw"])
        return np.array(X, dtype=np.float64), np.array(y, dtype=np.float64)

    def fit(self, X, y):
        """Fit the 4 OLS coefficients on training data.

        Uses np.linalg.lstsq (ordinary least squares) — no iterative optimizer,
        no learning rate, no epochs. Just a single closed-form solution.
        """
        self.coeffs, residuals, rank, sv = np.linalg.lstsq(X, y, rcond=None)
        self.fitted = True
        y_pred = X @ self.coeffs
        mae = float(np.mean(np.abs(y_pred - y)))
        rmse = float(np.sqrt(np.mean((y_pred - y) ** 2)))
        corr = float(np.corrcoef(y, y_pred)[0, 1]) if len(y) > 1 else 0.0
        print(f"\n[PhysicsEa] OLS fit on {len(y)} training reactions:")
        print(f"  Coefficients: a_marcus={self.coeffs[0]:.4f}, b_hammond={self.coeffs[1]:.4f}, "
              f"c_dErxn={self.coeffs[2]:.4f}, d_intercept={self.coeffs[3]:.4f}")
        print(f"  Train MAE:  {mae:.2f} kcal/mol")
        print(f"  Train RMSE: {rmse:.2f} kcal/mol")
        print(f"  Train R:    {corr:.4f}")
        return self.coeffs

    def predict(self, X):
        """Predict Ea for a feature matrix X (N, 4)."""
        if not self.fitted:
            raise RuntimeError("PhysicsEaCalculator not fitted yet. Call .fit() first.")
        return X @ self.coeffs

    def predict_single(self, coords_R, coords_TS, coords_P, atom_types, n,
                       de_rxn):
        """Predict Ea for a single reaction from its 3D coordinates."""
        feats = self._compute_features(coords_R, coords_TS, coords_P,
                                        atom_types, n, de_rxn)
        return float(feats @ self.coeffs)


def torch_mds_coords(D, mask, dim=3):
    """Differentiable-friendly classical MDS: distance matrix -> 3D coordinates.

    Embeds each molecule's predicted TS distance matrix into Cartesian space so
    an EGNN can refine the geometry. Double-centering is masked so padded atoms
    never contaminate the per-molecule centroid. The eigendecomposition is run
    in float64 for numerical stability; callers should pass a *detached* D (the
    geometry head is supervised directly on the coarse distances, while the EGNN
    learns the coordinate refinement on top), which keeps the unstable backward
    pass of eigh out of the graph.

    Args:
        D:    [B, N, N] pairwise distances (padded entries should be ~0).
        mask: [B, N] 1 for real atoms, 0 for padding.
    Returns:
        [B, N, dim] coordinates, zeroed on padded atoms.
    """
    B, N, _ = D.shape
    # Move to CPU for eigh: CPU LAPACK is far faster than GPU for small (30x30)
    # batched matrices, and this path is detached so no backward is needed.
    D_cpu = D.detach().cpu()
    mask_cpu = mask.detach().cpu()
    m = mask_cpu.to(torch.float64)                       # [B, N]
    pair = m.unsqueeze(-1) * m.unsqueeze(-2)         # [B, N, N] valid atom pairs
    cnt = m.sum(dim=1).clamp(min=1.0)                # [B] atoms per molecule
    S = (D_cpu.to(torch.float64) ** 2) * pair            # squared, masked
    # Masked double centering: subtract row/col/grand means over valid atoms.
    row_mean = S.sum(dim=2) / cnt.unsqueeze(-1)                      # [B, N]
    grand = S.sum(dim=(1, 2)) / (cnt ** 2)                          # [B]
    Bmat = -0.5 * (S - row_mean.unsqueeze(-1) - row_mean.unsqueeze(-2)
                   + grand.view(B, 1, 1))
    Bmat = Bmat * pair                               # keep padded rows/cols at 0
    # Symmetrize defensively before eigh.
    Bmat = 0.5 * (Bmat + Bmat.transpose(1, 2))
    # eigh fragility: with max_atoms padding, every molecule's Bmat carries many
    # exactly-repeated (zero) eigenvalues from the padded block, and a single
    # degenerate element can make the *batched* GPU solver fail to converge
    # ("too many repeated eigenvalues", LinAlgError). Lift the degeneracy with a
    # tiny diagonal jitter. Running on CPU avoids the GPU eigh convergence issue
    # entirely and is faster for small matrices.
    eyeN = torch.eye(N, dtype=Bmat.dtype).unsqueeze(0)
    Bmat = Bmat + 1e-6 * eyeN
    evals, evecs = torch.linalg.eigh(Bmat)       # ascending eigenvalues
    top_vals = evals[:, -dim:].flip(-1).clamp(min=0.0)              # [B, dim]
    top_vecs = evecs[:, :, -dim:].flip(-1)                          # [B, N, dim]
    coords = top_vecs * top_vals.clamp(min=0.0).sqrt().unsqueeze(1)
    coords = coords * m.unsqueeze(-1)                # zero padded atoms
    # eigh can silently return NaN eigenvectors on a degenerate Bmat (no
    # LinAlgError raised), which would poison the EGNN every forward pass. This
    # seed is detached, so replacing a bad embedding with zeros is harmless.
    coords = torch.nan_to_num(coords, nan=0.0, posinf=0.0, neginf=0.0)
    # Move result back to the original device (GPU) for the EGNN.
    return coords.to(D.dtype).to(D.device)


class EGCL(nn.Module):
    """One E(n)-equivariant graph convolution layer (Satorras et al., 2021).

    Operates on node features `h` (the chemical-property vector) and node
    coordinates `x` (the TS geometry). Messages depend only on squared
    interatomic distances, so node features stay E(3)-invariant while the
    coordinate update is E(3)-equivariant.
    """
    def __init__(self, hidden, coord_clamp=2.0, dropout=0.25):
        super().__init__()
        self.coord_clamp = coord_clamp
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden * 2 + 1, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
        )
        # Scalar coordinate weight per edge. Zero-initialized so the layer starts
        # as an identity map on coordinates (no displacement at init).
        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        nn.init.zeros_(self.coord_mlp[-1].weight)
        nn.init.zeros_(self.coord_mlp[-1].bias)

    def forward(self, h, x, mask):
        B, N, H = h.shape
        # Edge mask: valid atom pairs, excluding self-loops.
        pair = mask.unsqueeze(-1) * mask.unsqueeze(-2)                # [B, N, N]
        eye = torch.eye(N, device=h.device, dtype=h.dtype).unsqueeze(0)
        emask = (pair * (1.0 - eye)).unsqueeze(-1)                    # [B, N, N, 1]
        rel = x.unsqueeze(2) - x.unsqueeze(1)                         # [B, N, N, 3]
        dist2 = (rel ** 2).sum(dim=-1, keepdim=True)                  # [B, N, N, 1]
        hi = h.unsqueeze(2).expand(B, N, N, H)
        hj = h.unsqueeze(1).expand(B, N, N, H)
        edge_in = torch.cat([hi, hj, dist2], dim=-1)
        m_ij = self.edge_mlp(edge_in) * emask                        # masked messages
        # Equivariant coordinate update (normalized by neighbor count).
        trans = rel * torch.clamp(self.coord_mlp(m_ij),
                                  min=-self.coord_clamp, max=self.coord_clamp)
        trans = trans * emask
        deg = emask.sum(dim=2).clamp(min=1.0)                        # [B, N, 1]
        x = x + trans.sum(dim=2) / deg
        x = x * mask.unsqueeze(-1)
        # Invariant node update from aggregated messages.
        agg = m_ij.sum(dim=2)                                        # [B, N, H]
        h = h + self.node_mlp(torch.cat([h, agg], dim=-1))
        h = h * mask.unsqueeze(-1)
        return h, x


class EGNN(nn.Module):
    """E(n)-equivariant refiner: chemical-property vector + TS coords -> coords.

    The node features come entirely from chemistry (learned atom embedding +
    physical descriptors EN/Z/Mass); the coordinates come from the MDS embedding
    of the geometry head's predicted TS distance matrix. Stacked EGCL layers nudge
    the coordinates into a refined transition-state geometry.
    """
    def __init__(self, node_in_dim, hidden, n_layers, coord_clamp=2.0, dropout=0.25):
        super().__init__()
        self.embed_in = nn.Sequential(
            nn.Linear(node_in_dim, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
        )
        self.layers = nn.ModuleList([
            EGCL(hidden, coord_clamp, dropout) for _ in range(n_layers)
        ])

    def forward(self, node_feats, x, mask):
        h = self.embed_in(node_feats) * mask.unsqueeze(-1)
        for layer in self.layers:
            h, x = layer(h, x, mask)
        # Return the refined node features too: `h` now encodes each atom's
        # local 3D environment (angles, neighbour distances) after message
        # passing, which the energy refiner consumes directly instead of a
        # 7-scalar summary of the distance matrix.
        return h, x


class EaHead(nn.Module):
    """Learned activation-energy head on the EGNN's refined node features.

    After the EGNN message-passing, each atom's feature vector `h_ts` encodes its
    local 3D environment in the predicted TS (neighbour distances, angles). We
    masked-mean-pool those per-atom features into a molecule descriptor, append
    the signed reaction energy (z-scored de_rxn -- the Bell-Evans-Polanyi
    driver), and regress a *normalized* Ea. The gradient flows back into the
    EGNN, so the same message-passing that places the TS atoms also learns the
    structure->energy relationship ("learn physics using the EGNN").

    Output is the normalized Ea; callers denormalize with ea_mean/ea_std.
    """
    def __init__(self, node_dim, hidden, energy_feat_dim=0, dropout=0.25):
        super().__init__()
        # Input: pooled EGNN features + de_rxn scalar + energy descriptor vector
        in_dim = node_dim + 1 + energy_feat_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, h_ts, mask, de_rxn, energy_feats=None):
        # Masked mean-pool over real atoms -> [B, node_dim]
        m = mask.unsqueeze(-1)                                   # [B, N, 1]
        pooled = (h_ts * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)
        parts = [pooled, de_rxn.unsqueeze(-1)]                   # [B, node_dim], [B, 1]
        if energy_feats is not None:
            parts.append(energy_feats)                            # [B, 20]
        feat = torch.cat(parts, dim=-1)
        return self.net(feat).squeeze(-1)                        # [B] normalized Ea


class PSI(nn.Module):
    """Geometry + learned-Ea TS predictor.

    The model predicts a transition-state distance matrix and refines it to 3D
    coordinates via an EGNN.  Activation energy is regressed by a small EaHead
    on the EGNN's refined node features (trained jointly after a warmup). The
    physics-based PhysicsEaCalculator (Marcus + Bell-Evans-Polanyi + Hammond) is
    retained separately as a baseline for comparison, not used inside the model.
    """
    def __init__(self, config, num_atom_types):
        super().__init__()
        d_model = config["gru_hidden"] * 2
        atom_dim = config["atom_embed_dim"]
        drop = config["dropout"]
        delta_clamp = config["delta_clamp"]
        self.core = PSICore(config, num_atom_types)
        self.geom_head = GeometryHead(d_model, atom_dim, ATOM_PHYS_DIM, drop, delta_clamp)
        # EGNN coordinate refiner. Node features = chemical-property vector
        # (learned atom embedding + physical descriptors); coordinates come from
        # the MDS embedding of the geometry head's predicted TS distance matrix.
        self.egnn_enabled = config.get("egnn_enabled", True)
        if self.egnn_enabled:
            node_in_dim = atom_dim + ATOM_PHYS_DIM
            self.egnn = EGNN(
                node_in_dim,
                hidden=config["egnn_hidden"],
                n_layers=config["egnn_layers"],
                coord_clamp=config["egnn_coord_clamp"],
                dropout=drop,
            )
            # Learned Ea head on the EGNN's refined per-atom features (h_ts)
            # plus 20D energy descriptor (composition, bond-angle stats).
            self.ea_head = EaHead(
                node_dim=config["egnn_hidden"],
                hidden=config["egnn_hidden"],
                energy_feat_dim=ENERGY_FEAT_DIM,
                dropout=config.get("ea_head_dropout", drop),
            )

    @staticmethod
    def _coords_to_distance(x, mask):
        """Masked pairwise Euclidean distance matrix from coordinates."""
        N = x.shape[1]
        diff = x.unsqueeze(2) - x.unsqueeze(1)
        dist = torch.sqrt((diff ** 2).sum(dim=-1) + 1e-8)
        eye = torch.eye(N, device=x.device, dtype=x.dtype).unsqueeze(0)
        valid = mask.unsqueeze(-1) * mask.unsqueeze(-2)
        return dist * (1.0 - eye) * valid

    def forward(self, D_R, D_I, D_P, mask, atom_ids, atom_phys, de_rxn=None, energy_feats=None):
        """Predict TS distance matrix and (optionally) the learned Ea.

        Args:
            de_rxn: [B] z-scored signed reaction energy, fed to the Ea head.
                    May be None at geometry-only call sites; Ea is then None.
            energy_feats: [B, 20] z-scored molecular descriptor vector
                    (composition, bond-angle stats) for the Ea head.
        Returns:
            D_TS_pred:    [B, N, N] EGNN-refined TS distances (or coarse if EGNN off)
            D_TS_coarse:  [B, N, N] pre-EGNN coarse distances (for aux loss)
            ea_pred_norm: [B] normalized Ea from the EaHead, or None if the head
                          is absent (EGNN off) or de_rxn was not supplied.
        """
        f = self.core(D_R, D_I, D_P, mask, atom_ids, atom_phys)
        atom_emb = self.core.atom_embed(atom_ids)
        # Coarse TS distance matrix from the geometry head.
        D_TS_coarse = self.geom_head(f, atom_emb, atom_phys, D_R, D_I, D_P, mask)
        ea_pred_norm = None
        if self.egnn_enabled:
            # Chemical properties in one vector; predicted TS coordinates in the
            # other -- both fed to the EGNN. The MDS seed is detached so the
            # geometry head is trained directly by the coarse-distance aux loss
            # (keeping eigh's unstable backward out of the graph) while the EGNN
            # learns the coordinate refinement under the main geometry loss.
            node_feats = torch.cat([atom_emb, atom_phys], dim=-1)
            # Coordinate-space MDS is numerically delicate (eigh) and runs on
            # CPU for speed; only the eigh needs fp32/fp64.  The EGNN itself
            # stays under the outer AMP context for full GPU throughput.
            with torch.amp.autocast(device_type=D_R.device.type, enabled=False):
                x_init = torch_mds_coords(D_TS_coarse.detach().float(), mask)
            # h_ts is NOT detached: the Ea loss backpropagates through the
            # EGNN, so message passing learns the structure->energy relation.
            h_ts, x_ts = self.egnn(node_feats, x_init, mask)
            D_TS_pred = self._coords_to_distance(x_ts, mask)
            if de_rxn is not None:
                ef = energy_feats.float() if energy_feats is not None else None
                ea_pred_norm = self.ea_head(h_ts, mask, de_rxn.float(), ef)
        else:
            D_TS_pred = D_TS_coarse
        return D_TS_pred, D_TS_coarse, ea_pred_norm

class CosineAnnealingWarmup(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_epochs, total_epochs, min_lr=1e-6, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            scale = (self.last_epoch + 1) / max(1, self.warmup_epochs)
            return [base_lr * scale for base_lr in self.base_lrs]
        else:
            progress = (self.last_epoch - self.warmup_epochs) / max(1, self.total_epochs - self.warmup_epochs)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return [self.min_lr + (base_lr - self.min_lr) * cosine for base_lr in self.base_lrs]

def run_epoch(model, loader, optimizer, scaler, device, config, use_amp, epoch, stats, is_train=True):
    """Joint geometry + Ea training loop.

    Geometry is the backbone objective. After `ea_warmup_epochs`, the learned Ea
    loss is switched on so its gradient flows through the EGNN (the head trains
    on the *predicted* TS, matching inference). The physics Ea baseline is
    computed separately, post-training, and does not touch this loop.
    """
    if is_train:
        model.train()
    else:
        model.eval()
    total_loss, total_geom, total_ea_mae, total_ea_norm, n_batches = 0.0, 0.0, 0.0, 0.0, 0
    ea_mean, ea_std = stats["ea_mean"], stats["ea_std"]
    # Ea loss is gated by the geometry warmup so the head learns from sane TS.
    ea_active = epoch > config["ea_warmup_epochs"]
    ea_w = config["ea_loss_weight"] if ea_active else 0.0
    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for batch in loader:
            DR, DI, DP, DTS, mask, _geom_mask, atom_ids, atom_phys, Ea, de_rxn, energy_feats = move_batch_to_device(batch, device)
            if is_train:
                optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                p_DTS, p_DTS_coarse, ea_pred_norm = model(DR, DI, DP, mask, atom_ids, atom_phys, de_rxn, energy_feats)
                N = DR.shape[1]
                valid_mask = mask.unsqueeze(-1) * mask.unsqueeze(-2)
                eye = torch.eye(N, device=mask.device, dtype=mask.dtype).unsqueeze(0)
                m2d = valid_mask * (1.0 - eye)
                denom = m2d.sum().clamp(min=1)
                # Main geometry loss on the EGNN-refined distances.
                l_geom = F.huber_loss(p_DTS * m2d, DTS * m2d, reduction='sum', delta=0.5) / denom
                # Auxiliary loss on the coarse (pre-EGNN) distances trains the
                # geometry head directly, giving the EGNN a stable MDS seed.
                l_geom_coarse = F.huber_loss(p_DTS_coarse * m2d, DTS * m2d, reduction='sum', delta=0.5) / denom
                
                # --- PINN Matrix-wise Cross Check ---
                # Physics constraint 1: Spectator bonds (abs(DR - DP) < threshold) shouldn't change.
                # Target them towards the reactant/product midpoint (DI).
                spectator_mask = (torch.abs(DR - DP) < config["spectator_threshold"]).float() * m2d
                l_pinn_spectator = F.mse_loss(p_DTS * spectator_mask, DI * spectator_mask, reduction='sum') / denom
                
                # Physics constraint 2: Active bonds should generally be bounded by R and P.
                min_D = torch.minimum(DR, DP) - 0.2
                max_D = torch.maximum(DR, DP) + 0.2
                excess_high = F.relu(p_DTS - max_D) * m2d
                excess_low = F.relu(min_D - p_DTS) * m2d
                l_pinn_bounds = (excess_high**2 + excess_low**2).sum() / denom
                
                l_pinn = l_pinn_spectator + 0.5 * l_pinn_bounds
                
                loss = l_geom + config["geom_coarse_weight"] * l_geom_coarse + 0.2 * l_pinn

                # Learned Ea loss on the normalized target (computed every epoch
                # for monitoring; only added to `loss` once warmed up).
                if ea_pred_norm is not None:
                    ea_target_norm = (Ea - ea_mean) / ea_std
                    l_ea = F.smooth_l1_loss(ea_pred_norm, ea_target_norm)
                    if ea_w > 0.0:
                        loss = loss + ea_w * l_ea
                else:
                    l_ea = None
            if is_train:
                # Guard against a non-finite batch corrupting the weights: skip
                # the step entirely rather than relying solely on GradScaler.
                if not torch.isfinite(loss):
                    optimizer.zero_grad(set_to_none=True)
                    continue
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])
                scaler.step(optimizer)
                scaler.update()
            total_loss += loss.item()
            total_geom += l_geom.item()
            if l_ea is not None:
                total_ea_norm += l_ea.item()
                # Denormalized Ea MAE (kcal/mol) for human-readable tracking.
                total_ea_mae += (ea_pred_norm.detach() - ea_target_norm).abs().mean().item() * ea_std
            n_batches += 1
    nb = max(n_batches, 1)
    return {
        "loss": total_loss / nb,
        "geom": total_geom / nb,
        "ea_mae": total_ea_mae / nb,
        "ea_norm": total_ea_norm / nb,
    }

def train_pipeline(config):
    device = resolve_device(config)
    configure_torch_runtime(device)
    print("="*70); print(" PSI FULL PIPELINE (v2) "); print("="*70)
    extract_raw_data(config)
    samples, atom_vocab, atom_types_map = build_reaction_samples(config)
    if len(samples) == 0:
        print("Error: No complete reaction triplets found.")
        return
    n_total = len(samples)
    n_val = max(1, int(n_total * config["val_split"]))
    n_train = n_total - n_val
    rng = torch.Generator().manual_seed(config["split_seed"])
    indices = torch.randperm(n_total, generator=rng).tolist()
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]
    stats = compute_normalization(samples, train_indices)
    train_dataset = ReactionDataset(config, samples, atom_vocab, atom_types_map, stats, augment=True)
    eval_dataset = ReactionDataset(config, samples, atom_vocab, atom_types_map, stats, augment=False)
    train_subset = Subset(train_dataset, train_indices)
    val_subset = Subset(eval_dataset, val_indices)
    print(f"\nData split: {n_train} train, {n_val} validation")
    loader_kwargs = {
        "batch_size": config["batch_size"],
        "num_workers": config["num_workers"],
        "pin_memory": config["pin_memory"] and device.type == "cuda",
    }
    if config["num_workers"] > 0:
        loader_kwargs["persistent_workers"] = True
    train_loader = DataLoader(train_subset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_subset, shuffle=False, **loader_kwargs)
    eval_loader = DataLoader(Subset(eval_dataset, list(range(n_total))), shuffle=False, **loader_kwargs)
    num_atom_types = len(atom_vocab)
    model = PSI(config, num_atom_types).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")
    # Single param group: all parameters serve the geometry objective.
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"]
    )
    scheduler = CosineAnnealingWarmup(optimizer, warmup_epochs=config["warmup_epochs"], total_epochs=config["epochs"])
    use_amp = config["amp"] and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    metadata = {
        "atom_vocab": atom_vocab,
        "aphys_mean": stats["aphys_mean"].tolist(),
        "aphys_std": stats["aphys_std"].tolist(),
        "ea_mean": stats["ea_mean"],
        "ea_std": stats["ea_std"],
        "de_rxn_mean": stats["de_rxn_mean"],
        "de_rxn_std": stats["de_rxn_std"],
        "efeat_mean": stats["efeat_mean"].tolist(),
        "efeat_std": stats["efeat_std"].tolist(),
        "config_snapshot": {k: v for k, v in config.items() if isinstance(v, (int, float, str, bool))},
    }
    print(f"\nTraining for up to {config['epochs']} epochs (patience={config['patience']})...")
    print(f"  Joint geometry + learned Ea; Ea loss turns on after epoch {config['ea_warmup_epochs']}.")
    print(f"{'Epoch':>6} | {'Train Loss':>11} | {'Val Loss':>11} | {'T.Geom':>8} | {'V.Geom':>8} | {'V.EaMAE':>8} | {'LR':>10}")
    print("-" * 84)
    best_val_loss = float('inf')
    patience_counter = 0
    history = []
    best_model_path = os.path.join(config["save_dir"], "psi_best.pt")
    for epoch in range(1, config["epochs"] + 1):
        train_metrics = run_epoch(model, train_loader, optimizer, scaler, device, config, use_amp, epoch, stats, is_train=True)
        val_metrics = run_epoch(model, val_loader, None, scaler, device, config, use_amp, epoch, stats, is_train=False)
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        # Geometry is the primary selection signal; once the Ea loss is active,
        # blend in the (normalized) Ea error so early stopping does not cut off a
        # still-improving head. ea_select_weight keeps geometry dominant.
        ea_active = epoch > config["ea_warmup_epochs"]
        if epoch == config["ea_warmup_epochs"] + 1:
            best_val_loss = float('inf')
            patience_counter = 0
            print(f"\n--- Ea warmup ended. Resetting early stopping tracking ---")

        val_select = val_metrics["geom"]
        if ea_active:
            val_select = val_select + config["ea_select_weight"] * val_metrics["ea_norm"]
        history.append({
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "val_loss": val_metrics["loss"],
            "val_select": val_select,
            "train_geom": train_metrics["geom"],
            "val_geom": val_metrics["geom"],
            "train_ea_mae": train_metrics["ea_mae"],
            "val_ea_mae": val_metrics["ea_mae"],
            "val_ea_norm": val_metrics["ea_norm"],
            "lr": current_lr,
        })
        improved = val_select < best_val_loss
        if improved:
            best_val_loss = val_select
            torch.save({"model_state_dict": model.state_dict(), "metadata": metadata}, best_model_path)
        patience_counter = 0 if improved else patience_counter + 1
        if epoch % config["print_every"] == 0 or epoch == 1 or improved:
            marker = " *" if improved else ""
            print(f"{epoch:6d} | {train_metrics['loss']:11.4f} | {val_metrics['loss']:11.4f} | "
                  f"{train_metrics['geom']:8.5f} | {val_metrics['geom']:8.5f} | "
                  f"{val_metrics['ea_mae']:8.3f} | {current_lr:10.2e}{marker}")
        if patience_counter >= config["patience"]:
            print(f"\nEarly stopping at epoch {epoch} (no improvement for {config['patience']} epochs)")
            break
    history_path = os.path.join(config["save_dir"], "training_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"\nTraining history saved to {history_path}")
    print(f"\nLoading best model (best val_geom={best_val_loss:.4f})...")
    checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    # =========================================================================
    # Post-training: predict TS geometries, then compute Ea via physics
    # =========================================================================
    print("\n" + "="*70); print(" EVALUATION (geometry + learned Ea + physics baseline) "); print("="*70)
    model.eval()
    # Step 1: collect predicted TS distance matrices + learned Ea for all reactions.
    pred_dists_map = {}   # rxn_id -> (n, n) numpy pred dist
    ea_neural_map = {}    # rxn_id -> learned Ea (kcal/mol, denormalized)
    ea_mean, ea_std = stats["ea_mean"], stats["ea_std"]
    geom_results = []
    val_rxn_ids = {samples[vi]["rxn_id"] for vi in val_indices}
    with torch.no_grad():
        for batch in eval_loader:
            DR, DI, DP, DTS, mask, geom_mask, atom_ids, atom_phys, _Ea, de_rxn, energy_feats = move_batch_to_device(batch, device)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                p_DTS, _, ea_pred_norm = model(DR, DI, DP, mask, atom_ids, atom_phys, de_rxn, energy_feats)
            ea_pred_kcal = None
            if ea_pred_norm is not None:
                ea_pred_kcal = (ea_pred_norm.float().cpu().numpy() * ea_std + ea_mean)
            for i in range(len(batch["rxn_id"])):
                rxn_id = batch["rxn_id"][i]
                n = int(mask[i].sum().item())
                di = DI[i, :n, :n].cpu().numpy()
                dp = p_DTS[i, :n, :n].cpu().numpy()
                dt = DTS[i, :n, :n].cpu().numpy()
                gm = geom_mask[i, :n, :n].cpu().numpy()
                d_abs = np.abs(dp - dt)
                d_mae = (d_abs * gm).sum().item() / max(float(gm.sum()), 1.0)
                d_mae_all = d_abs.mean().item()
                split = "val" if rxn_id in val_rxn_ids else "train"
                pred_dists_map[rxn_id] = dp
                if ea_pred_kcal is not None:
                    ea_neural_map[rxn_id] = float(ea_pred_kcal[i])
                geom_results.append({
                    "rxn_id": rxn_id, "split": split, "n_atoms": n,
                    "dist_MAE": d_mae, "dist_MAE_all": d_mae_all,
                    "D_I": di.tolist(), "D_pred": dp.tolist(), "D_true": dt.tolist(),
                    "geom_mask": gm.tolist(),
                    "atom_types": atom_types_map[rxn_id],
                })
    # Step 2: recover 3D coords from predicted distance matrices and compute
    # physics-based Ea using Marcus theory + Hammond postulate.
    print("\n[PhysicsEa] Recovering 3D coordinates from predicted distance matrices...")
    ea_calculator = PhysicsEaCalculator(
        bond_scale=config["fragment_bond_scale"],
        spectator_threshold=config["spectator_threshold"],
    )
    # Build per-sample predicted TS coords + physics features (ordered by sample index)
    train_X, train_y = [], []
    all_coords_ts = {}  # rxn_id -> predicted TS coords (n, 3)
    for idx in range(n_total):
        s = samples[idx]
        rxn_id = s["rxn_id"]
        n = s["n_atoms"]
        pred_dist = pred_dists_map[rxn_id]
        atom_types = s["atom_types"]
        # Post-process the predicted distance matrix
        pred_dist = np.maximum((pred_dist + pred_dist.T) / 2.0, 0.0)
        np.fill_diagonal(pred_dist, 0.0)
        pred_dist = clamp_steric_collisions(pred_dist, atom_types)
        # Recover 3D coordinates via fragment-aware MDS
        c_R = np.asarray(s["c_R"][:n], dtype=np.float64)
        c_P = np.asarray(s["c_P"][:n], dtype=np.float64)
        align_frags = choose_alignment_fragments(c_R, c_P, atom_types, n, config["fragment_bond_scale"])
        c_I = (kabsch_align_reactant_fragments(c_R, c_P, atom_types, n, config["fragment_bond_scale"])[:n] + c_P[:n]) / 2.0
        pred_coords = mds_by_fragments(
            pred_dist, atom_types=atom_types, fragments=align_frags,
            reference_coords=c_I, bond_scale=config["fragment_bond_scale"],
        )
        all_coords_ts[rxn_id] = pred_coords
    # Compute physics features for train split, fit OLS
    train_samples_ordered = [samples[i] for i in train_indices]
    train_coords_ordered = [all_coords_ts[samples[i]["rxn_id"]] for i in train_indices]
    train_X, train_y = ea_calculator.compute_features_batch(train_samples_ordered, train_coords_ordered, config)
    ea_calculator.fit(train_X, train_y)
    # Step 3: assemble results. Primary Ea is the learned neural head; the
    # physics calculator (Marcus/Hammond/OLS) is reported alongside as a baseline.
    samples_by_id = {s["rxn_id"]: s for s in samples}
    results = []
    for gr in geom_results:
        rxn_id = gr["rxn_id"]
        s = samples_by_id[rxn_id]
        n = s["n_atoms"]
        c_R = np.asarray(s["c_R"][:n], dtype=np.float64)
        c_P = np.asarray(s["c_P"][:n], dtype=np.float64)
        c_TS_pred = all_coords_ts[rxn_id]
        de_rxn = float(s["energy_feats_raw"][1])
        ea_true = s["Ea_raw"]
        # Physics baseline.
        ea_physics = ea_calculator.predict_single(c_R, c_TS_pred, c_P, s["atom_types"], n, de_rxn)
        # Learned head (falls back to physics if EGNN/head disabled).
        ea_pred = ea_neural_map.get(rxn_id, ea_physics)
        results.append({
            **gr,
            "Ea_true": ea_true, "Ea_pred": ea_pred,
            "Ea_error": abs(ea_pred - ea_true),
            "Ea_pred_physics": ea_physics,
            "Ea_error_physics": abs(ea_physics - ea_true),
        })
    # Save PhysicsEa coefficients into metadata for inference
    metadata["physics_ea_coeffs"] = ea_calculator.coeffs.tolist()
    train_results = [r for r in results if r["split"] == "train"]
    val_results = [r for r in results if r["split"] == "val"]
    def print_stats(name, res_list):
        if not res_list: return
        d_maes = [r["dist_MAE"] for r in res_list]
        ea_trues = [r["Ea_true"] for r in res_list]
        ea_preds = [r["Ea_pred"] for r in res_list]
        ea_phys = [r["Ea_pred_physics"] for r in res_list]
        corr = np.corrcoef(ea_trues, ea_preds)[0, 1] if len(ea_trues) > 1 else 0.0
        r2 = _r2(ea_trues, ea_preds)
        mae_neural = float(np.mean([r["Ea_error"] for r in res_list]))
        r2_phys = _r2(ea_trues, ea_phys)
        mae_phys = float(np.mean([r["Ea_error_physics"] for r in res_list]))
        print(f"\n{name} ({len(res_list)} reactions):")
        print(f"  Ea MAE (neural):   {mae_neural:8.2f} kcal/mol   |  R²: {r2:7.4f}   r: {corr:7.4f}")
        print(f"  Ea MAE (physics):  {mae_phys:8.2f} kcal/mol   |  R²: {r2_phys:7.4f}   (baseline)")
        print(f"  Dist MAE:          {np.mean(d_maes):8.4f} Å      |  std: {np.std(d_maes):.4f} Å")
    print_stats("TRAIN SET", train_results)
    print_stats("VALIDATION SET", val_results)
    print_stats("ALL DATA", results)
    print(f"\n{'Reaction':<15} {'Split':<6} {'Ea True':>10} {'Ea Pred':>10} {'Ea Err':>10} {'Dist MAE':>10}")
    for r in sorted(results, key=lambda x: x["rxn_id"]):
        print(f"{r['rxn_id']:<15} {r['split']:<6} {r['Ea_true']:10.2f} {r['Ea_pred']:10.2f} {r['Ea_error']:10.2f} {r['dist_MAE']:10.4f}")
    output_path = os.path.join(config["save_dir"], "detailed_analysis.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    final_path = os.path.join(config["save_dir"], "psi_final.pt")
    torch.save({"model_state_dict": model.state_dict(), "metadata": metadata}, final_path)
    torch.save({"model_state_dict": model.state_dict(), "metadata": metadata}, best_model_path)
    print(f"\nModel saved to {final_path}")
    print(f"Predictions saved to {output_path}")
    create_dashboard(output_path, config["save_dir"])

def predict_transition_state(config, reactant_path, product_path, model_path, output_path, xyz_path=None):
    device = resolve_device(config)
    configure_torch_runtime(device)
    reactant = load_log_file(reactant_path)
    product = load_log_file(product_path)
    r_atoms = reactant["atoms"]
    p_atoms = product["atoms"]
    if len(r_atoms) != len(p_atoms):
        raise ValueError("Reactant and product must have the same number of atoms in the same order.")
    if len(r_atoms) > config["max_atoms"]:
        raise ValueError(f"Prediction has {len(r_atoms)} atoms, but max_atoms is {config['max_atoms']}.")
    r_types = [a["atom"] for a in r_atoms]
    p_types = [a["atom"] for a in p_atoms]
    if r_types != p_types:
        raise ValueError("Reactant and product atom ordering/types differ. Align atom order before prediction.")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    state_dict = checkpoint["model_state_dict"]
    meta = checkpoint["metadata"]
    atom_vocab = meta["atom_vocab"]
    num_atom_types = max(atom_vocab.values())
    aphys_mean = np.array(meta["aphys_mean"], dtype=np.float32)
    aphys_std = np.array(meta["aphys_std"], dtype=np.float32)
    # Ea / de_rxn normalization stats for the learned head (older checkpoints
    # without a learned head won't have these -> neural Ea is then skipped).
    ea_mean = meta.get("ea_mean")
    ea_std = meta.get("ea_std")
    de_rxn_mean = meta.get("de_rxn_mean", 0.0)
    de_rxn_std = meta.get("de_rxn_std", 1.0)
    efeat_mean = np.array(meta.get("efeat_mean", np.zeros(ENERGY_FEAT_DIM)), dtype=np.float32)
    efeat_std = np.array(meta.get("efeat_std", np.ones(ENERGY_FEAT_DIM)), dtype=np.float32)
    n = len(r_atoms)
    c_R = padded_coords(r_atoms, config["max_atoms"])
    c_P = padded_coords(p_atoms, config["max_atoms"])

    c_R_aligned = kabsch_align_reactant_fragments(
        c_R, c_P, r_types, n, config["fragment_bond_scale"]
    )
    c_I = np.zeros_like(c_R)
    c_I[:n] = (c_R_aligned[:n] + c_P[:n]) / 2.0

    D_R = compute_distance_matrix(c_R)
    D_P = compute_distance_matrix(c_P)
    D_I = (D_R + D_P) / 2.0
    align_fragments = choose_alignment_fragments(
        c_R, c_P, r_types, n, config["fragment_bond_scale"]
    )
    geom_mask = geometry_pair_mask_from_fragments(align_fragments, config["max_atoms"])

    mask = np.zeros(config["max_atoms"], dtype=np.float32)
    mask[:n] = 1.0
    atom_ids = np.zeros(config["max_atoms"], dtype=np.int64)
    for i, atom_type in enumerate(r_types):
        if atom_type not in atom_vocab:
            raise KeyError(f"Atom type '{atom_type}' not in training vocab {sorted(atom_vocab)}.")
        atom_ids[i] = atom_vocab[atom_type]
    atom_phys = build_atom_physical_features(r_types, n, config["max_atoms"])
    atom_phys_norm = (atom_phys - aphys_mean) / aphys_std
    # Signed reaction energy (kcal/mol) -> z-scored input to the learned Ea head.
    e_r = reactant["energy"] * config["hartree_to_kcal"]
    e_p = product["energy"] * config["hartree_to_kcal"]
    de_rxn = e_p - e_r
    de_rxn_norm = (de_rxn - de_rxn_mean) / de_rxn_std
    # 20D energy descriptor (composition, bond-angle stats) for the Ea head.
    energy_feats = build_energy_features(
        r_types, n, c_R_aligned, c_P, e_r, e_p, config["fragment_bond_scale"]
    )
    energy_feats_norm = (energy_feats - efeat_mean) / efeat_std
    model = PSI(config, num_atom_types).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    with torch.no_grad():
        t_DR = torch.from_numpy(D_R).unsqueeze(0).to(device)
        t_DI = torch.from_numpy(D_I).unsqueeze(0).to(device)
        t_DP = torch.from_numpy(D_P).unsqueeze(0).to(device)
        t_mask = torch.from_numpy(mask).unsqueeze(0).to(device)
        t_atom_ids = torch.from_numpy(atom_ids).unsqueeze(0).to(device)
        t_aphys = torch.from_numpy(atom_phys_norm).unsqueeze(0).to(device)
        t_de_rxn = torch.tensor([de_rxn_norm], dtype=torch.float32, device=device)
        t_efeats = torch.from_numpy(energy_feats_norm.astype(np.float32)).unsqueeze(0).to(device)
        p_DTS, _, ea_pred_norm = model(t_DR, t_DI, t_DP, t_mask, t_atom_ids, t_aphys, t_de_rxn, t_efeats)
    # Learned Ea (denormalized); None if the checkpoint predates the head.
    ea_neural = None
    if ea_pred_norm is not None and ea_mean is not None and ea_std is not None:
        ea_neural = float(ea_pred_norm.float().cpu().item() * ea_std + ea_mean)
    pred_dist = p_DTS[0, :n, :n].cpu().numpy()
    pred_dist = np.maximum((pred_dist + pred_dist.T) / 2.0, 0.0)
    np.fill_diagonal(pred_dist, 0.0)
    c_I_real = c_I[:n]
    pred_dist = clamp_steric_collisions(pred_dist, r_types[:n])
    pred_dist = apply_spectator_constraints(
        pred_dist,
        D_R[:n, :n], D_P[:n, :n], n,
        threshold=config["spectator_threshold"],
        tol=config["spectator_tol"],
        pair_mask=geom_mask[:n, :n],
    )
    pred_dist = enforce_triangle_inequality(
        pred_dist, fragments=[f for f in align_fragments if max(f) < n]
    )
    validate_ts_geometry(
        pred_dist, D_R[:n, :n], D_P[:n, :n], r_types[:n], n,
        spectator_threshold=config["spectator_threshold"],
    )
    pred_coords = mds_by_fragments(
        pred_dist,
        atom_types=r_types[:n],
        fragments=align_fragments,
        reference_coords=c_I_real,
        bond_scale=config["fragment_bond_scale"],
    )
    # Physics-based Ea baseline from the predicted 3D TS coordinates.
    ea_calculator = PhysicsEaCalculator(
        bond_scale=config["fragment_bond_scale"],
        spectator_threshold=config["spectator_threshold"],
    )
    ea_calculator.coeffs = np.array(meta["physics_ea_coeffs"], dtype=np.float64)
    ea_calculator.fitted = True
    ea_physics = ea_calculator.predict_single(
        c_R[:n], pred_coords, c_P[:n], r_types[:n], n, de_rxn
    )
    # Primary Ea is the learned head; fall back to physics for legacy checkpoints.
    energy_pred = ea_neural if ea_neural is not None else ea_physics
    ea_source = "neural" if ea_neural is not None else "physics (no learned head in checkpoint)"
    result = {
        "reactant_path": reactant_path,
        "product_path": product_path,
        "model_path": model_path,
        "n_atoms": n,
        "atom_types": r_types,
        "Ea_pred": energy_pred,
        "Ea_pred_physics": ea_physics,
        "Ea_source": ea_source,
        "D_I": D_I[:n, :n].tolist(),
        "D_pred": pred_dist.tolist(),
        "geom_mask": geom_mask[:n, :n].tolist(),
        "coords_pred": pred_coords.tolist(),
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    if xyz_path:
        write_xyz(xyz_path, r_types, pred_coords, f"PSI predicted TS, Ea={energy_pred:.4f} kcal/mol")
    print("\n" + "="*70)
    print(" PREDICTION RESULT ")
    print("="*70)
    print(f"Atoms: {n}")
    print(f"Predicted activation energy ({ea_source}): {energy_pred:.4f} kcal/mol")
    print(f"  Physics baseline: {ea_physics:.4f} kcal/mol")
    print(f"Prediction JSON saved to: {output_path}")
    if xyz_path:
        print(f"Predicted TS XYZ saved to: {xyz_path}")

# =============================================================================
# Results dashboard (formerly psi_visualize.py)
# =============================================================================

def _r2(true, pred):
    """Coefficient of determination R^2 = 1 - SS_res / SS_tot."""
    true = np.asarray(true, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    if len(true) < 2:
        return 0.0
    ss_res = float(np.sum((true - pred) ** 2))
    ss_tot = float(np.sum((true - true.mean()) ** 2))
    if ss_tot < 1e-12:
        return 0.0
    return 1.0 - ss_res / ss_tot


def energy_metrics(records, pred_key="Ea_pred"):
    """Regression metrics for an activation-energy prediction over `records`.

    `pred_key` selects which prediction column to score (the learned head's
    "Ea_pred" by default, or "Ea_pred_physics" for the baseline).
    """
    if not records or pred_key not in records[0]:
        return {"n": 0, "MAE": 0.0, "RMSE": 0.0, "R2": 0.0, "Pearson": 0.0, "MAPE": 0.0}
    true = np.array([r["Ea_true"] for r in records], dtype=np.float64)
    pred = np.array([r[pred_key] for r in records], dtype=np.float64)
    err = pred - true
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    pearson = float(np.corrcoef(true, pred)[0, 1]) if len(true) > 1 else 0.0
    denom = np.where(np.abs(true) < 1e-6, np.nan, np.abs(true))
    mape = float(np.nanmean(np.abs(err) / denom) * 100.0)
    return {"n": len(records), "MAE": mae, "RMSE": rmse, "R2": _r2(true, pred),
            "Pearson": pearson, "MAPE": mape}


def geometry_metrics(records):
    """Distance-prediction metrics, aggregated over all masked atom pairs.

    Compares AI-predicted distances (D_pred) against the true TS distances on
    the geometry-mask pairs, and reports the percentage improvement over the
    plain reactant/product interpolation guess (D_I).
    """
    if not records:
        return {"n": 0, "MAE": 0.0, "RMSE": 0.0, "R2": 0.0,
                "guess_MAE": 0.0, "improve_pct": 0.0}
    true_all, pred_all, guess_all = [], [], []
    for r in records:
        dt = np.array(r["D_true"], dtype=np.float64)
        dp = np.array(r["D_pred"], dtype=np.float64)
        di = np.array(r["D_I"], dtype=np.float64)
        mask = np.array(r.get("geom_mask"), dtype=np.float64) if r.get("geom_mask") is not None else np.ones_like(dt)
        # upper triangle of masked pairs only (matrices are symmetric)
        iu = np.triu_indices_from(dt, k=1)
        sel = mask[iu] > 0
        true_all.append(dt[iu][sel])
        pred_all.append(dp[iu][sel])
        guess_all.append(di[iu][sel])
    true = np.concatenate(true_all)
    pred = np.concatenate(pred_all)
    guess = np.concatenate(guess_all)
    mae = float(np.mean(np.abs(pred - true)))
    rmse = float(np.sqrt(np.mean((pred - true) ** 2)))
    guess_mae = float(np.mean(np.abs(guess - true)))
    improve = (1.0 - mae / guess_mae) * 100.0 if guess_mae > 1e-9 else 0.0
    return {"n": int(true.size), "MAE": mae, "RMSE": rmse, "R2": _r2(true, pred),
            "guess_MAE": guess_mae, "improve_pct": improve}


def create_dashboard(data_path, save_dir):
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found.")
        return

    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    data = sorted(data, key=lambda x: x["rxn_id"])

    ea_trues = np.array([r["Ea_true"] for r in data])
    ea_preds = np.array([r["Ea_pred"] for r in data])
    ea_errors = np.abs(ea_preds - ea_trues)
    dist_maes = np.array([r["dist_MAE"] for r in data])

    guess_maes = []
    for r in data:
        di = np.array(r["D_I"])
        dt = np.array(r["D_true"])
        guess_maes.append(masked_mae(di, dt, r.get("geom_mask")))
    guess_maes = np.array(guess_maes)

    ea_corr = np.corrcoef(ea_trues, ea_preds)[0, 1] if len(ea_trues) > 1 else 0.0

    train_data = [r for r in data if r["split"] == "train"]
    val_data = [r for r in data if r["split"] == "val"]

    # Full regression metric breakdown (Train / Val / All).
    ea_metrics = {"Train": energy_metrics(train_data), "Val": energy_metrics(val_data), "All": energy_metrics(data)}
    # Physics baseline (only present if the results carry "Ea_pred_physics").
    ea_phys_metrics = {
        "Train": energy_metrics(train_data, "Ea_pred_physics"),
        "Val": energy_metrics(val_data, "Ea_pred_physics"),
        "All": energy_metrics(data, "Ea_pred_physics"),
    }
    has_physics = "Ea_pred_physics" in data[0] if data else False
    geom_metrics = {"Train": geometry_metrics(train_data), "Val": geometry_metrics(val_data), "All": geometry_metrics(data)}

    def _metric_rows(metric_map, fields):
        # fields: list of (label, key, formatter)
        rows = ""
        for label, key, fmt in fields:
            cells = "".join(f"<td>{fmt(metric_map[s][key])}</td>" for s in ("Train", "Val", "All"))
            rows += f"<tr><td style='color:#94a3b8;'>{label}</td>{cells}</tr>"
        return rows

    f2 = lambda v: f"{v:.2f}"
    f3 = lambda v: f"{v:.3f}"
    f4 = lambda v: f"{v:.4f}"
    fpct = lambda v: f"{v:.1f}%"

    energy_metric_rows = _metric_rows(ea_metrics, [
        ("R² (neural)", "R2", f3),
        ("Pearson R (neural)", "Pearson", f3),
        ("MAE (kcal/mol)", "MAE", f2),
        ("RMSE (kcal/mol)", "RMSE", f2),
        ("MAPE", "MAPE", fpct),
        ("Count", "n", lambda v: str(int(v))),
    ])
    # Append the physics baseline (R²/MAE) so neural vs physics is visible inline.
    if has_physics:
        energy_metric_rows += _metric_rows(ea_phys_metrics, [
            ("R² (physics baseline)", "R2", f3),
            ("MAE physics (kcal/mol)", "MAE", f2),
        ])
    geometry_metric_rows = _metric_rows(geom_metrics, [
        ("R²", "R2", f3),
        ("MAE (Å)", "MAE", f4),
        ("RMSE (Å)", "RMSE", f4),
        ("Guess MAE (Å)", "guess_MAE", f4),
        ("Improvement vs Guess", "improve_pct", fpct),
        ("Pairs", "n", lambda v: str(int(v))),
    ])

    sorted_by_geom = sorted(data, key=lambda x: x["dist_MAE"])
    n_rxns = len(sorted_by_geom)

    best_5 = sorted_by_geom[:5]
    median_5 = sorted_by_geom[n_rxns//2 - 2 : n_rxns//2 + 3]
    worst_5 = sorted_by_geom[-5:]

    representative_list = best_5 + median_5 + worst_5
    representative_data = {}

    for r in representative_list:
        rid = r["rxn_id"]
        dt = np.array(r["D_true"])
        dp = np.array(r["D_pred"])
        di = np.array(r["D_I"])
        atoms = r["atom_types"]
        fragments = fragments_from_mask(r["geom_mask"]) if "geom_mask" in r else find_fragments_from_distances(dt, atoms)

        # The true TS distance matrix is globally metric, so a single MDS reproduces
        # it faithfully. Using per-fragment MDS here would scatter forming/breaking-bond
        # fragments along an arbitrary cursor axis and "explode" the real structure.
        X_true = mds(dt)
        # The model is only supervised on within-fragment (geom_mask) pairs, so the
        # predicted/interpolated inter-fragment distances are untrained. Rebuild each
        # fragment rigidly from its own block, then Kabsch-align it onto the true frame.
        X_pred = mds_by_fragments(dp, atoms, fragments=fragments, reference_coords=X_true)
        X_guess = mds_by_fragments(di, atoms, fragments=fragments, reference_coords=X_true)

        representative_data[rid] = {
            "rxn_id": rid,
            "atom_types": atoms,
            "coords_true": X_true.tolist(),
            "coords_pred": X_pred.tolist(),
            "coords_guess": X_guess.tolist(),
            "bonds_true": get_bonds_from_distances(dt, atoms, fragments=fragments),
            "bonds_pred": get_bonds_from_distances(dp, atoms, fragments=fragments),
            "bonds_guess": get_bonds_from_distances(di, atoms, fragments=fragments),
            "dist_MAE": r["dist_MAE"],
            "Ea_true": r["Ea_true"],
            "Ea_pred": r["Ea_pred"],
            "Ea_error": r["Ea_error"],
            "tier": "Best" if r in best_5 else "Median" if r in median_5 else "Worst"
        }

    summary_list = []
    for r in data:
        summary_list.append({
            "rxn_id": r["rxn_id"],
            "split": r["split"],
            "Ea_true": round(r["Ea_true"], 2),
            "Ea_pred": round(r["Ea_pred"], 2),
            "Ea_error": round(r["Ea_error"], 2),
            "dist_MAE": round(r["dist_MAE"], 4),
            "guess_MAE": round(masked_mae(r["D_I"], r["D_true"], r.get("geom_mask")), 4),
            "n_atoms": r["n_atoms"]
        })

    worst_10_geom = sorted(summary_list, key=lambda x: x["dist_MAE"], reverse=True)[:10]

    html_template = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>PSI Transition State Prediction Dashboard</title>
  <script src="https://cdn.plot.ly/plotly-2.24.1.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/3dmol@2.4.2/build/3Dmol-min.js"></script>
  <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap" rel="stylesheet">
  <style>
    body {{
      background-color: #080c14;
      color: #cbd5e1;
      font-family: 'Outfit', sans-serif;
      margin: 0;
      padding: 0;
    }}
    .container {{
      max-width: 1400px;
      margin: 0 auto;
      padding: 2rem;
    }}
    header {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 2rem;
      border-bottom: 1px solid rgba(255, 255, 255, 0.05);
      padding-bottom: 1.5rem;
    }}
    .header-left h1 {{
      font-size: 2.2rem;
      font-weight: 700;
      margin: 0;
      background: linear-gradient(135deg, #3b82f6, #8b5cf6);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
    }}
    .header-left p {{
      margin: 0.25rem 0 0 0;
      color: #64748b;
      font-size: 0.95rem;
    }}
    .badge-top {{
      background: rgba(59, 130, 246, 0.1);
      border: 1px solid rgba(59, 130, 246, 0.2);
      color: #60a5fa;
      padding: 0.35rem 0.75rem;
      border-radius: 9999px;
      font-weight: 600;
      font-size: 0.85rem;
    }}
    .stats-grid {{
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 1.5rem;
      margin-bottom: 2rem;
    }}
    .card {{
      background: rgba(17, 24, 39, 0.7);
      backdrop-filter: blur(12px);
      border: 1px solid rgba(255, 255, 255, 0.06);
      border-radius: 12px;
      padding: 1.5rem;
      box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
    }}
    .stat-card {{
      position: relative;
      overflow: hidden;
    }}
    .stat-card::before {{
      content: '';
      position: absolute;
      top: 0; left: 0; width: 4px; height: 100%;
      background: #3b82f6;
    }}
    .stat-card.energy::before {{ background: #8b5cf6; }}
    .stat-card.geom::before {{ background: #10b981; }}
    .stat-card.corr::before {{ background: #f59e0b; }}

    .stat-val {{
      font-size: 2.2rem;
      font-weight: 700;
      color: #ffffff;
      margin-top: 0.5rem;
    }}
    .stat-label {{
      font-size: 0.85rem;
      color: #94a3b8;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      font-weight: 600;
    }}
    .charts-grid {{
      display: grid;
      grid-template-columns: repeat(2, 1fr);
      gap: 1.5rem;
      margin-bottom: 2rem;
    }}
    .chart-title {{
      font-size: 1.15rem;
      font-weight: 600;
      color: #ffffff;
      margin-top: 0;
      margin-bottom: 1rem;
    }}
    .viewer-header {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 1rem;
    }}
    .dropdown-container {{
      display: flex;
      align-items: center;
      gap: 0.75rem;
    }}
    label {{
      font-size: 0.9rem;
      color: #94a3b8;
    }}
    select {{
      background: #1e293b;
      border: 1px solid rgba(255, 255, 255, 0.1);
      color: #e2e8f0;
      padding: 0.5rem 1rem;
      border-radius: 6px;
      outline: none;
      cursor: pointer;
      font-family: inherit;
      font-weight: 600;
    }}
    .viewer-layout {{
      display: grid;
      grid-template-columns: 2fr 1fr;
      gap: 1.5rem;
    }}
    .viewer-info-card {{
      background: rgba(255, 255, 255, 0.02);
      border-radius: 8px;
      padding: 1.25rem;
      border: 1px solid rgba(255, 255, 255, 0.04);
    }}
    .info-row {{
      display: flex;
      justify-content: space-between;
      margin-bottom: 0.75rem;
      border-bottom: 1px solid rgba(255, 255, 255, 0.03);
      padding-bottom: 0.5rem;
    }}
    .info-row:last-child {{
      border-bottom: none;
      margin-bottom: 0;
      padding-bottom: 0;
    }}
    .info-label {{
      color: #94a3b8;
      font-size: 0.9rem;
    }}
    .info-val {{
      font-weight: 600;
      color: #ffffff;
    }}
    .legend-container {{
      margin-top: 1.5rem;
      display: flex;
      flex-direction: column;
      gap: 0.75rem;
    }}
    .legend-item {{
      display: flex;
      align-items: center;
      gap: 0.75rem;
    }}
    .legend-dot {{
      width: 12px;
      height: 12px;
      border-radius: 50%;
    }}
    .legend-line {{
      flex-grow: 1;
      border-bottom: 1px dashed rgba(255, 255, 255, 0.1);
    }}
    .table-container {{
      overflow-x: auto;
      margin-top: 1rem;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      text-align: left;
    }}
    th, td {{
      padding: 0.75rem 1rem;
      border-bottom: 1px solid rgba(255, 255, 255, 0.05);
      font-size: 0.9rem;
    }}
    th {{
      color: #94a3b8;
      font-weight: 600;
      text-transform: uppercase;
      font-size: 0.8rem;
      letter-spacing: 0.05em;
    }}
    tr:hover {{
      background: rgba(255, 255, 255, 0.02);
    }}
    .badge {{
      padding: 0.2rem 0.5rem;
      border-radius: 4px;
      font-size: 0.75rem;
      font-weight: 600;
    }}
    .badge-train {{ background: rgba(59, 130, 246, 0.15); color: #60a5fa; }}
    .badge-val {{ background: rgba(139, 92, 246, 0.15); color: #a78bfa; }}
    .badge-tier {{
      padding: 0.15rem 0.4rem;
      font-size: 0.75rem;
      font-weight: 700;
      border-radius: 4px;
    }}
    .badge-tier.Best {{ background: rgba(16, 185, 129, 0.15); color: #34d399; }}
    .badge-tier.Median {{ background: rgba(245, 158, 11, 0.15); color: #fbbf24; }}
    .badge-tier.Worst {{ background: rgba(239, 68, 68, 0.15); color: #f87171; }}
    .metrics-grid {{
      display: grid;
      grid-template-columns: repeat(2, 1fr);
      gap: 1.5rem;
      margin-bottom: 2rem;
    }}
    table.metrics-table {{ width: 100%; border-collapse: collapse; }}
    table.metrics-table th, table.metrics-table td {{
      padding: 0.55rem 0.85rem;
      text-align: right;
      font-size: 0.9rem;
      border-bottom: 1px solid rgba(255, 255, 255, 0.05);
    }}
    table.metrics-table th:first-child, table.metrics-table td:first-child {{ text-align: left; }}
    table.metrics-table thead th {{
      color: #94a3b8; text-transform: uppercase; font-size: 0.78rem; letter-spacing: 0.05em;
    }}
    table.metrics-table td {{ color: #ffffff; font-weight: 600; }}
    table.metrics-table th.col-val {{ color: #a78bfa; }}
  </style>
</head>
<body>
  <div class="container">
    <header>
      <div class="header-left">
        <h1>PSI Results Dashboard</h1>
        <p>Transition State prediction & activation energy regression analysis</p>
      </div>
      <div>
        <span class="badge-top">Dataset Size: {len(data)} Reactions</span>
      </div>
    </header>

    <div class="stats-grid">
      <div class="card stat-card">
        <div class="stat-label">Total Triplets</div>
        <div class="stat-val">{len(data)}</div>
      </div>
      <div class="card stat-card energy">
        <div class="stat-label">Ea MAE</div>
        <div class="stat-val">{np.mean(ea_errors):.2f} <span style="font-size: 1rem; font-weight: normal; color: #94a3b8;">kcal/mol</span></div>
      </div>
      <div class="card stat-card corr">
        <div class="stat-label">Ea Correlation (R)</div>
        <div class="stat-val">{ea_corr:.4f}</div>
      </div>
      <div class="card stat-card geom">
        <div class="stat-label">Avg Distance MAE</div>
        <div class="stat-val">{np.mean(dist_maes):.4f} <span style="font-size: 1rem; font-weight: normal; color: #94a3b8;">Å</span></div>
      </div>
    </div>

    <div class="metrics-grid">
      <div class="card">
        <div class="chart-title">Energy (Ea) Regression Metrics</div>
        <table class="metrics-table">
          <thead>
            <tr><th>Metric</th><th>Train</th><th class="col-val">Val</th><th>All</th></tr>
          </thead>
          <tbody>
            {energy_metric_rows}
          </tbody>
        </table>
      </div>
      <div class="card">
        <div class="chart-title">Geometry (Distance) Metrics</div>
        <table class="metrics-table">
          <thead>
            <tr><th>Metric</th><th>Train</th><th class="col-val">Val</th><th>All</th></tr>
          </thead>
          <tbody>
            {geometry_metric_rows}
          </tbody>
        </table>
      </div>
    </div>

    <div class="charts-grid">
      <div class="card">
        <div class="chart-title">Ea: Actual vs. Predicted</div>
        <div id="ea-scatter" style="height: 400px;"></div>
      </div>
      <div class="card">
        <div class="chart-title">Geometry: Distance MAE Improvement</div>
        <div id="geom-histogram" style="height: 400px;"></div>
      </div>
    </div>

    <div class="card molecular-viewer-card">
      <div class="viewer-header">
        <div class="chart-title" style="margin-bottom: 0;">Interactive 3D Transition State Alignment</div>
        <div class="dropdown-container">
          <label for="rxn-select">Select Reaction Case Study:</label>
          <select id="rxn-select" onchange="updateViewer()">
          </select>
        </div>
      </div>
      <div class="viewer-layout">
        <div id="mol-viewer" style="height: 500px; background: #0c101b; border-radius: 8px; border: 1px solid rgba(255, 255, 255, 0.03);"></div>
        <div class="viewer-info-card">
          <div class="chart-title" style="font-size: 1rem; border-bottom: 1px solid rgba(255, 255, 255, 0.06); padding-bottom: 0.5rem; margin-bottom: 0.75rem;">Case Study Details</div>

          <div class="info-row">
            <span class="info-label">Reaction ID</span>
            <span class="info-val" id="case-id">-</span>
          </div>
          <div class="info-row">
            <span class="info-label">Performance Tier</span>
            <span id="case-tier">-</span>
          </div>
          <div class="info-row">
            <span class="info-label">Atom Count</span>
            <span class="info-val" id="case-atoms">-</span>
          </div>
          <div class="info-row">
            <span class="info-label">Distance MAE (AI)</span>
            <span class="info-val" id="case-dist-mae">-</span>
          </div>
          <div class="info-row">
            <span class="info-label">Ea True</span>
            <span class="info-val" id="case-ea-true">-</span>
          </div>
          <div class="info-row">
            <span class="info-label">Ea Predicted</span>
            <span class="info-val" id="case-ea-pred" style="color: #60a5fa;">-</span>
          </div>
          <div class="info-row">
            <span class="info-label">Ea Error</span>
            <span class="info-val" id="case-ea-error">-</span>
          </div>

          <div class="legend-container">
            <div class="legend-item">
              <div class="legend-dot" style="background: #10b981;"></div>
              <span class="info-label">Ground Truth TS</span>
            </div>
            <div class="legend-item">
              <div class="legend-dot" style="background: #3b82f6;"></div>
              <span class="info-label">AI Predicted TS</span>
            </div>
            <div class="legend-item">
              <div class="legend-dot" style="background: rgba(239, 68, 68, 0.4); border: 1px dashed #ef4444;"></div>
              <span class="info-label">Interpolated Guess</span>
            </div>
          </div>
        </div>
      </div>
    </div>

    <div class="card">
      <div class="chart-title">Worst 10 Geometry Predictions (Diagnostics)</div>
      <div class="table-container">
        <table>
          <thead>
            <tr>
              <th>Reaction ID</th>
              <th>Split</th>
              <th>Atoms</th>
              <th>Guess MAE (Å)</th>
              <th>AI Predicted MAE (Å)</th>
              <th>Ea True (kcal)</th>
              <th>Ea Pred (kcal)</th>
              <th>Ea Error (kcal)</th>
            </tr>
          </thead>
          <tbody id="worst-table-body">
          </tbody>
        </table>
      </div>
    </div>
  </div>

  <script>
    const reactions = {json.dumps(summary_list)};
    const repData = {json.dumps(representative_data)};
    const worst10 = {json.dumps(worst_10_geom)};

    const trainScatter = {{
      x: [], y: [], text: [], mode: 'markers',
      name: 'Train Set',
      marker: {{ color: '#3b82f6', opacity: 0.6, size: 7 }}
    }};
    const valScatter = {{
      x: [], y: [], text: [], mode: 'markers',
      name: 'Val Set',
      marker: {{ color: '#8b5cf6', opacity: 0.8, size: 8 }}
    }};

    for (let r of reactions) {{
      let t = `ID: ${{r.rxn_id}}<br>True: ${{r.Ea_true}} kcal<br>Pred: ${{r.Ea_pred}} kcal<br>Error: ${{r.Ea_error}} kcal`;
      if (r.split === 'train') {{
        trainScatter.x.push(r.Ea_true);
        trainScatter.y.push(r.Ea_pred);
        trainScatter.text.push(t);
      }} else {{
        valScatter.x.push(r.Ea_true);
        valScatter.y.push(r.Ea_pred);
        valScatter.text.push(t);
      }}
    }}

    const minEa = Math.min(...reactions.map(r => r.Ea_true));
    const maxEa = Math.max(...reactions.map(r => r.Ea_true));
    const refLine = {{
      x: [minEa, maxEa], y: [minEa, maxEa],
      mode: 'lines', name: 'y=x Ref',
      line: {{ color: 'rgba(255,255,255,0.2)', dash: 'dash', width: 1.5 }}
    }};

    const scatterLayout = {{
      plot_bgcolor: 'transparent',
      paper_bgcolor: 'transparent',
      margin: {{ l: 50, r: 20, t: 20, b: 50 }},
      xaxis: {{ title: 'True Ea (kcal/mol)', gridcolor: 'rgba(255,255,255,0.05)', tickcolor: '#94a3b8' }},
      yaxis: {{ title: 'Predicted Ea (kcal/mol)', gridcolor: 'rgba(255,255,255,0.05)', tickcolor: '#94a3b8' }},
      legend: {{ font: {{ color: '#cbd5e1' }} }},
      hovermode: 'closest'
    }};
    Plotly.newPlot('ea-scatter', [trainScatter, valScatter, refLine], scatterLayout);

    const guessHist = {{
      x: reactions.map(r => r.guess_MAE),
      type: 'histogram', name: 'Initial Guess MAE',
      opacity: 0.5, marker: {{ color: '#ef4444' }},
      xbins: {{ size: 0.02 }}
    }};
    const aiHist = {{
      x: reactions.map(r => r.dist_MAE),
      type: 'histogram', name: 'AI Predicted MAE',
      opacity: 0.6, marker: {{ color: '#10b981' }},
      xbins: {{ size: 0.02 }}
    }};

    const histLayout = {{
      plot_bgcolor: 'transparent',
      paper_bgcolor: 'transparent',
      margin: {{ l: 50, r: 20, t: 20, b: 50 }},
      xaxis: {{ title: 'Distance MAE (Å)', gridcolor: 'rgba(255,255,255,0.05)', tickcolor: '#94a3b8' }},
      yaxis: {{ title: 'Count', gridcolor: 'rgba(255,255,255,0.05)', tickcolor: '#94a3b8' }},
      barmode: 'overlay',
      legend: {{ font: {{ color: '#cbd5e1' }} }}
    }};
    Plotly.newPlot('geom-histogram', [guessHist, aiHist], histLayout);

    const select = document.getElementById('rxn-select');
    for (let rid in repData) {{
      let opt = document.createElement('option');
      opt.value = rid;
      opt.text = `${{rid}} [${{repData[rid].tier}} tier: MAE=${{repData[rid].dist_MAE.toFixed(4)}}Å]`;
      select.appendChild(opt);
    }}

    // Initialize 3Dmol viewer (guard against the CDN failing to load)
    let viewer = null;
    if (typeof $3Dmol === 'undefined') {{
      document.getElementById('mol-viewer').innerHTML =
        '<div style="display:flex;height:100%;align-items:center;justify-content:center;color:#f87171;text-align:center;padding:1rem;">' +
        '3Dmol.js failed to load (check network / CDN). 3D structures cannot be displayed.</div>';
    }} else {{
      viewer = $3Dmol.createViewer("mol-viewer", {{ backgroundColor: "#0c101b" }});
    }}

    function updateViewer() {{
      const rid = select.value;
      const r = repData[rid];

      document.getElementById('case-id').innerText = r.rxn_id;
      document.getElementById('case-atoms').innerText = r.atom_types.length;
      document.getElementById('case-dist-mae').innerText = r.dist_MAE.toFixed(4) + ' Å';
      document.getElementById('case-ea-true').innerText = r.Ea_true.toFixed(2) + ' kcal/mol';
      document.getElementById('case-ea-pred').innerText = r.Ea_pred.toFixed(2) + ' kcal/mol';
      document.getElementById('case-ea-error').innerText = r.Ea_error.toFixed(2) + ' kcal/mol';

      const tierBadge = document.getElementById('case-tier');
      tierBadge.className = `badge badge-tier ${{r.tier}}`;
      tierBadge.innerText = r.tier;

      if (!viewer) return;
      viewer.clear();

      function makeXYZString(atomTypes, coords) {{
        let lines = [atomTypes.length, "PSI TS Prediction"];
        for (let i = 0; i < atomTypes.length; i++) {{
          lines.push(`${{atomTypes[i]}} ${{coords[i][0]}} ${{coords[i][1]}} ${{coords[i][2]}}`);
        }}
        return lines.join("\\n");
      }}

      const mTrue = viewer.addModel(makeXYZString(r.atom_types, r.coords_true), "xyz");
      viewer.setStyle({{model: mTrue.getID()}}, {{
        stick: {{color: '#10b981', radius: 0.12}},
        sphere: {{color: '#10b981', radius: 0.3}}
      }});

      const mPred = viewer.addModel(makeXYZString(r.atom_types, r.coords_pred), "xyz");
      viewer.setStyle({{model: mPred.getID()}}, {{
        stick: {{color: '#3b82f6', radius: 0.08}},
        sphere: {{color: '#3b82f6', radius: 0.22}}
      }});

      const mGuess = viewer.addModel(makeXYZString(r.atom_types, r.coords_guess), "xyz");
      viewer.setStyle({{model: mGuess.getID()}}, {{
        stick: {{color: '#f87171', radius: 0.05, opacity: 0.4}},
        sphere: {{color: '#f87171', radius: 0.15, opacity: 0.4}}
      }});

      r.atom_types.forEach((type, idx) => {{
        viewer.addLabel(`${{type}}${{idx}}`, {{
          position: {{x: r.coords_true[idx][0], y: r.coords_true[idx][1], z: r.coords_true[idx][2]}},
          backgroundColor: 'rgba(12,16,27,0.8)',
          fontColor: '#cbd5e1',
          fontSize: 10,
          backgroundOpacity: 0.8,
          borderThickness: 0,
          alignment: 'center'
        }});
      }});

      viewer.zoomTo();
      viewer.render();
    }}


    if (select.options.length > 0) {{
      updateViewer();
    }}

    const tableBody = document.getElementById('worst-table-body');
    for (let r of worst10) {{
      let tr = document.createElement('tr');
      tr.innerHTML = `
        <td style="font-weight: 600; color: #ffffff;">${{r.rxn_id}}</td>
        <td><span class="badge badge-${{r.split}}">${{r.split.toUpperCase()}}</span></td>
        <td>${{r.n_atoms}}</td>
        <td>${{r.guess_MAE.toFixed(4)}}</td>
        <td style="color: #f87171; font-weight: 600;">${{r.dist_MAE.toFixed(4)}}</td>
        <td>${{r.Ea_true.toFixed(2)}}</td>
        <td>${{r.Ea_pred.toFixed(2)}}</td>
        <td style="font-weight: 600;">${{r.Ea_error.toFixed(2)}}</td>
      `;
      tableBody.appendChild(tr);
    }}
  </script>
</body>
</html>
"""

    output_html = os.path.join(save_dir, 'psi_results_dashboard.html')
    with open(output_html, 'w', encoding='utf-8') as f:
        f.write(html_template)

    print(f"Interactive dashboard generated successfully: {output_html}")

# =============================================================================
# Command-line interface
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PSI transition-state training, prediction, and visualization (v2)")
    subparsers = parser.add_subparsers(dest="command")
    train_parser = subparsers.add_parser("train", help="Train the PSI model and evaluate known triplets")
    train_parser.add_argument("--epochs", type=int, default=CONFIG["epochs"], help="Training epochs")
    train_parser.add_argument("--batch-size", type=int, default=CONFIG["batch_size"], help="Training batch size")
    train_parser.add_argument("--num-workers", type=int, default=CONFIG["num_workers"], help="DataLoader worker processes")
    train_parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default=CONFIG["device"], help="Training device")
    train_parser.add_argument("--require-cuda", action="store_true", help="Fail instead of falling back to CPU")
    train_parser.add_argument("--no-amp", action="store_true", help="Disable CUDA mixed precision")
    train_parser.add_argument("--patience", type=int, default=CONFIG["patience"], help="Early stopping patience")
    train_parser.add_argument("--lr", type=float, default=CONFIG["lr"], help="Learning rate")
    predict_parser = subparsers.add_parser("predict", help="Predict a transition state from reactant/product logs")
    predict_parser.add_argument("--reactant", "-r", required=True, help="Path to reactant .log file")
    predict_parser.add_argument("--product", "-p", required=True, help="Path to product .log file")
    predict_parser.add_argument("--model", default=os.path.join(CONFIG["save_dir"], "psi_final.pt"), help="Path to psi_final.pt")
    predict_parser.add_argument("--output", "-o", default=os.path.join(CONFIG["save_dir"], "psi_prediction.json"), help="Output JSON path")
    predict_parser.add_argument("--xyz", default=os.path.join(CONFIG["save_dir"], "psi_predicted_ts.xyz"), help="Output XYZ path")
    predict_parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default=CONFIG["device"], help="Prediction device")
    predict_parser.add_argument("--require-cuda", action="store_true", help="Fail instead of falling back to CPU")
    dash_parser = subparsers.add_parser("dashboard", help="Generate the results dashboard from a detailed_analysis.json")
    dash_parser.add_argument("--data", default="detailed_analysis.json", help="Path to detailed_analysis.json")
    dash_parser.add_argument("--save-dir", default=".", help="Directory to save the HTML dashboard")
    import sys
    if "ipykernel" in sys.modules:
        args = parser.parse_args(["train"])
    else:
        args = parser.parse_args()
    if args.command == "predict":
        CONFIG["device"] = args.device
        CONFIG["require_cuda"] = args.require_cuda
        predict_transition_state(CONFIG, args.reactant, args.product, args.model, args.output, args.xyz)
    elif args.command == "dashboard":
        create_dashboard(args.data, args.save_dir)
    else:
        if args.command == "train":
            CONFIG["epochs"] = args.epochs
            CONFIG["batch_size"] = args.batch_size
            CONFIG["num_workers"] = args.num_workers
            CONFIG["device"] = args.device
            CONFIG["require_cuda"] = args.require_cuda
            CONFIG["amp"] = not args.no_amp
            CONFIG["patience"] = args.patience
            CONFIG["lr"] = args.lr
        train_pipeline(CONFIG)
