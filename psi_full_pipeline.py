"""
PSI Full Pipeline: From Tarball to Transition State Prediction
==============================================================
Self-contained script implementing the improved PSI architecture.

Improvements over v1:
- Atom-type embeddings (chemistry-aware)
- 2-layer Bi-GRU with dropout
- 4-layer pre-norm Transformer
- Attention-pooled energy prediction
- Train/validation split with early stopping
- AdamW + cosine annealing + linear warmup
- Gradient clipping
- Huber loss for geometry
- Energy target normalization
- Data augmentation (coordinate noise)
"""

import os
import sys
import json
import math
import tarfile
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset

# ============================================================================
# Configuration
# ============================================================================

# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

CONFIG = {
    "tar_path": r"d:\Transition state\b97d3.tar.gz",
    "dataset_json": r"d:\Transition state\extracted_dataset.json",
    "save_dir": r"d:\Transition state",
    "extraction_limit": 1500,  # Number of log files to extract
    "force_extract": False,    # Rebuild dataset_json instead of reusing stale data
    "max_atoms": 30,           # Standard molecule size padding

    # Gaussian embedding
    "n_gaussians": 32,         # K basis functions
    "gauss_start": 0.4,
    "gauss_stop": 6.0,

    # Atom embeddings
    "atom_embed_dim": 32,      # Per-atom learnable embedding size

    # GRU
    "gru_hidden": 128,         # Per-direction hidden → d_model = 256
    "gru_layers": 2,           # 2-layer Bi-GRU
    "gru_dropout": 0.1,        # Dropout between GRU layers

    # Transformer
    "attn_heads": 8,
    "attn_layers": 3,          # 3 layers (balanced for 499 samples)
    "ff_dim": 512,             # FFN width
    "dropout": 0.1,            # Lower dropout — augmentation provides regularization

    # Loss
    "energy_weight_start": 5.0,  # Start higher — geom loss is ~0.05 while energy loss ~1.0
    "energy_weight_end": 15.0,
    "energy_ramp_epochs": 150,

    # Optimizer
    "lr": 5e-4,                # Slightly higher LR for faster convergence
    "weight_decay": 5e-4,      # Less regularization — small model already
    "warmup_epochs": 15,       # Shorter warmup
    "grad_clip": 1.0,

    # Training
    "batch_size": 32,          # Larger batch for more stable gradients
    "num_workers": 0,
    "pin_memory": True,
    "device": "auto",
    "require_cuda": False,
    "amp": True,
    "epochs": 1500,
    "print_every": 25,
    "val_split": 0.2,          # 20% validation
    "split_seed": 42,          # Reproducible split
    "patience": 150,           # More patience for cosine schedule

    # Data augmentation
    "coord_noise_std": 0.005,  # Smaller noise (was 0.01 — too much for small molecules)

    # TS physical rules
    "spectator_threshold": 0.15,  # Å change below which a bond is considered a spectator
    "spectator_tol": 0.05,         # ±fraction allowed deviation for spectator bonds

    "hartree_to_kcal": 627.509,
}

def resolve_device(config):
    requested = config.get("device", "auto").lower()
    if requested not in {"auto", "cuda", "cpu"}:
        raise ValueError("device must be one of: auto, cuda, cpu")

    if requested == "cpu":
        return torch.device("cpu")

    if requested == "cuda" or config.get("require_cuda", False):
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested, but this Python has CPU-only PyTorch installed. "
                f"Current torch build: {torch.__version__}, torch.version.cuda={torch.version.cuda}. "
                "Install a CUDA-enabled PyTorch build, then rerun with `--device cuda`."
            )
        return torch.device("cuda")

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def configure_torch_runtime(device):
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA runtime from PyTorch: {torch.version.cuda}")
    else:
        print("Using CPU. For GPU, install CUDA-enabled PyTorch and run with `--device cuda`.")

def move_batch_to_device(batch, device):
    return (
        batch["D_R"].to(device, non_blocking=True),
        batch["D_I"].to(device, non_blocking=True),
        batch["D_P"].to(device, non_blocking=True),
        batch["D_TS"].to(device, non_blocking=True),
        batch["mask"].to(device, non_blocking=True),
        batch["Ea"].to(device, non_blocking=True),
        batch["atom_ids"].to(device, non_blocking=True),
        batch["energy_feats"].to(device, non_blocking=True),
    )

# ============================================================================
# 1. & 2. Data Extraction, Loading & Triplet Grouping
# ============================================================================

def parse_log_content(file_content):
    """Extracts energy and nuclear coordinates from a Q-Chem .log file."""
    atoms = []
    energy = None
    lines = file_content.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if "Standard Nuclear Orientation" in line:
            current_atoms = []
            i += 3 # skip header
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

def extract_raw_data(config):
    """Parses tarball and saves results to a JSON file."""
    if os.path.exists(config["dataset_json"]) and not config.get("force_extract", False):
        print(f"Dataset found at {config['dataset_json']}, skipping extraction.")
        return
    
    print(f"Extracting {config['extraction_limit']} logs from {config['tar_path']}...")
    dataset = []
    with tarfile.open(config["tar_path"], "r:gz") as tar:
        for member in tar:
            if member.isfile() and member.name.endswith(".log"):
                file_obj = tar.extractfile(member)
                if file_obj:
                    try:
                        content = file_obj.read().decode('utf-8', errors='ignore')
                        parsed = parse_log_content(content)
                        if parsed["atoms"] and parsed["energy"] is not None:
                            dataset.append({
                                "filename": member.name,
                                "energy": parsed["energy"],
                                "atoms": parsed["atoms"]
                            })
                            if len(dataset) % 10 == 0:
                                print(f"  Extracted {len(dataset)}/{config['extraction_limit']}...")
                    except Exception as e:
                        continue
                if len(dataset) >= config["extraction_limit"]:
                    break
                    
    with open(config["dataset_json"], 'w') as f:
        json.dump(dataset, f, indent=2)
    print(f"Saved {len(dataset)} entries to {config['dataset_json']}\n")

def compute_distance_matrix(coords):
    """Compute NxN Euclidean distance matrix."""
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=-1) + 1e-8)
    return dist.astype(np.float32)

def mds(D, dim=3):
    """Reconstruct approximate coordinates from a distance matrix."""
    n = D.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * H @ (D ** 2) @ H
    evals, evecs = np.linalg.eigh(B)
    idx = np.argsort(evals)[::-1]
    evals = evals[idx]
    evecs = evecs[:, idx]
    return evecs[:, :dim] @ np.diag(np.sqrt(np.maximum(evals[:dim], 0)))

def kabsch(P, Q):
    """Align coordinates P onto Q."""
    P_centered = P - P.mean(axis=0)
    Q_centered = Q - Q.mean(axis=0)
    C = P_centered.T @ Q_centered
    V, _, W = np.linalg.svd(C)
    d = np.linalg.det(V @ W)
    E = np.eye(3)
    if d < 0:
        E[2, 2] = -1
    R = V @ E @ W
    return P_centered @ R + Q.mean(axis=0)


# ============================================================================
# Transition State Physical Rules
# ============================================================================

# Standard covalent radii (Å) used for steric-collision detection.
# Source: Alvarez 2008 / Pyykkö & Atsumi 2009 consensus values.
COVALENT_RADII = {
    'H': 0.31, 'C': 0.76, 'N': 0.71, 'O': 0.66, 'F': 0.57,
    'S': 1.05, 'Cl': 1.02, 'Br': 1.20, 'I': 1.39, 'P': 1.07,
    'Si': 1.11, 'B': 0.84,
}


def kabsch_align_reactant(c_R, c_P, n):
    """Return a copy of c_R whose first n rows are aligned onto c_P[:n].

    This ensures that the midpoint interpolation c_I = (c_R_aligned + c_P) / 2
    is in a consistent reference frame, avoiding collapsed or inverted geometries
    caused by mixing coordinates from different rotational frames.
    """
    c_R_aligned = c_R.copy()
    c_R_aligned[:n] = kabsch(c_R[:n], c_P[:n])
    return c_R_aligned


def clamp_steric_collisions(pred_dist, atom_types):
    """Raise any predicted distance that falls below the physical steric floor.

    The minimum allowed pairwise distance is 85 % of the sum of covalent radii.
    Distances shorter than this represent nuclear overlap and are physically
    impossible — the model must not produce them.

    Args:
        pred_dist: (n, n) symmetric distance matrix (modified in-place).
        atom_types: list of element symbols, length n.

    Returns:
        Modified pred_dist.
    """
    n = len(atom_types)
    for i in range(n):
        for j in range(i + 1, n):
            r_i = COVALENT_RADII.get(atom_types[i], 0.70)
            r_j = COVALENT_RADII.get(atom_types[j], 0.70)
            min_d = 0.85 * (r_i + r_j)
            if pred_dist[i, j] < min_d:
                pred_dist[i, j] = min_d
                pred_dist[j, i] = min_d
    return pred_dist


def classify_bonds(D_R, D_P, n, threshold=0.15):
    """Classify atom pairs into active bonds and spectator bonds.

    Active bonds change length by more than `threshold` Å between R and P
    (breaking or forming bonds on the reaction path).  Spectator bonds are
    structural bonds that should be preserved at their equilibrium length.

    Args:
        D_R: (n, n) reactant distance matrix.
        D_P: (n, n) product  distance matrix.
        n:   number of real atoms (ignoring padding).
        threshold: Å change considered chemically significant.

    Returns:
        (active, spectator) — each a list of (i, j) index tuples, i < j.
    """
    active, spectator = [], []
    for i in range(n):
        for j in range(i + 1, n):
            if abs(D_R[i, j] - D_P[i, j]) > threshold:
                active.append((i, j))
            else:
                spectator.append((i, j))
    return active, spectator


def apply_spectator_constraints(pred_dist, D_R, D_P, n,
                                threshold=0.15, tol=0.05):
    """Constrain spectator-bond distances within a tolerance band.

    Pairs that do not participate in the reaction should have the same bond
    length in the TS as in R/P (within ±tol).  This prevents the model from
    distorting stable ring systems, spectator C–H bonds, etc.

    Args:
        pred_dist: (n, n) predicted distance matrix (modified in-place).
        D_R: (n, n) reactant distance matrix.
        D_P: (n, n) product  distance matrix.
        n:   number of real atoms.
        threshold: Å change threshold for bond classification.
        tol: fractional tolerance around the R/P reference distance.

    Returns:
        Modified pred_dist.
    """
    _, spectator = classify_bonds(D_R, D_P, n, threshold)
    for (i, j) in spectator:
        d_ref = (D_R[i, j] + D_P[i, j]) / 2.0
        lo = d_ref * (1.0 - tol)
        hi = d_ref * (1.0 + tol)
        clamped = float(np.clip(pred_dist[i, j], lo, hi))
        pred_dist[i, j] = clamped
        pred_dist[j, i] = clamped
    return pred_dist


def enforce_triangle_inequality(D):
    """Ensure the distance matrix satisfies the triangle inequality.

    MDS reconstruction requires the matrix to be a valid semi-metric.
    Pairs that violate  D[i,j] <= D[i,k] + D[k,j]  produce imaginary
    eigenvalues that collapse or distort the 3-D reconstruction.

    Uses the Floyd-Warshall relaxation, which is O(n^3) but n <= 30 here.

    Args:
        D: (n, n) distance matrix.

    Returns:
        New, triangle-inequality-consistent distance matrix.
    """
    D = D.copy()
    n = D.shape[0]
    for k in range(n):
        for i in range(n):
            for j in range(n):
                if D[i, j] > D[i, k] + D[k, j]:
                    D[i, j] = D[j, i] = D[i, k] + D[k, j]
    return D


def validate_ts_geometry(pred_dist, D_R, D_P, atom_types, n,
                         spectator_threshold=0.15):
    """Run all physical checks on the predicted TS distance matrix and report.

    Checks performed:
      1. Steric collisions (distances < 85 % of summed covalent radii).
      2. Active-bond out-of-bounds (predicted length outside the R-to-P range
         with a 0.3 Å buffer in each direction).

    Args:
        pred_dist: (n, n) predicted distance matrix.
        D_R: (n, n) reactant distance matrix.
        D_P: (n, n) product  distance matrix.
        atom_types: list of element symbols, length n.
        n:   number of real atoms.
        spectator_threshold: threshold forwarded to classify_bonds.

    Returns:
        True if all checks pass, False if any violation is found.
    """
    issues = []

    # Check 1: steric collisions
    for i in range(n):
        for j in range(i + 1, n):
            r_i = COVALENT_RADII.get(atom_types[i], 0.70)
            r_j = COVALENT_RADII.get(atom_types[j], 0.70)
            min_d = 0.85 * (r_i + r_j)
            if pred_dist[i, j] < min_d:
                issues.append(
                    f"  STERIC   {atom_types[i]}{i}-{atom_types[j]}{j}: "
                    f"{pred_dist[i,j]:.3f} Å < floor {min_d:.3f} Å"
                )

    # Check 2: active bonds out of R→P range
    active, _ = classify_bonds(D_R, D_P, n, spectator_threshold)
    for (i, j) in active:
        d_lo = min(D_R[i, j], D_P[i, j]) - 0.30
        d_hi = max(D_R[i, j], D_P[i, j]) + 0.30
        if not (d_lo <= pred_dist[i, j] <= d_hi):
            issues.append(
                f"  ACT_OOB  {atom_types[i]}{i}-{atom_types[j]}{j}: "
                f"pred={pred_dist[i,j]:.3f} Å  "
                f"(R={D_R[i,j]:.3f}, P={D_P[i,j]:.3f})"
            )

    if issues:
        print(f"[TS Validation] {len(issues)} issue(s) detected:")
        for iss in issues:
            print(iss)
    else:
        print("[TS Validation] Geometry passed all physical checks.")

    return len(issues) == 0


# ============================================================================
# End of Transition State Physical Rules
# ============================================================================

def padded_coords(atoms, max_atoms):
    coords = np.zeros((max_atoms, 3), dtype=np.float32)
    for i, atom in enumerate(atoms):
        coords[i] = [atom["x"], atom["y"], atom["z"]]
    return coords

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


def build_atom_vocab(raw_data):
    """Build atom type vocabulary from dataset. Index 0 is reserved for padding."""
    atom_set = set()
    for entry in raw_data:
        for a in entry["atoms"]:
            atom_set.add(a["atom"])
    # Sort for reproducibility
    sorted_atoms = sorted(atom_set)
    # 0 = padding, 1..N = real atoms
    vocab = {atom: i + 1 for i, atom in enumerate(sorted_atoms)}
    print(f"Atom vocabulary ({len(vocab)} types): {vocab}")
    return vocab


class ReactionDataset(Dataset):
    def __init__(self, config, augment=False):
        self.config = config
        self.augment = augment
        with open(config["dataset_json"], "r") as f:
            raw_data = json.load(f)

        # Build atom vocabulary
        self.atom_vocab = build_atom_vocab(raw_data)
        
        # Triplets grouping
        reactions = {}
        for entry in raw_data:
            parts = entry["filename"].split("/")
            if len(parts) < 3: continue
            rxn_id = parts[1]
            prefix = parts[2].lower()
            role = "r" if prefix.startswith("r") else "p" if prefix.startswith("p") else "ts" if prefix.startswith("ts") else None
            if not role: continue
            if rxn_id not in reactions: reactions[rxn_id] = {}
            reactions[rxn_id][role] = entry

        self.samples = []
        self.atom_types_map = {}  # Lookup map for evaluation
        all_ea = []  # Collect for normalization stats
        
        for rxn_id, roles in sorted(reactions.items()):
            if "r" in roles and "p" in roles and "ts" in roles:
                r_e = roles["r"]; p_e = roles["p"]; ts_e = roles["ts"]
                n = len(ts_e["atoms"])
                if n > config["max_atoms"]: continue

                # Raw coordinates
                c_R = padded_coords(r_e["atoms"], config["max_atoms"])
                c_P = padded_coords(p_e["atoms"], config["max_atoms"])
                c_TS = padded_coords(ts_e["atoms"], config["max_atoms"])
                
                # Atom type IDs
                atom_ids = np.zeros(config["max_atoms"], dtype=np.int64)
                for i, a in enumerate(ts_e["atoms"]):
                    atom_ids[i] = self.atom_vocab.get(a["atom"], 0)

                # Mask
                mask = np.zeros(config["max_atoms"], dtype=np.float32)
                mask[:n] = 1.0

                # Energy: Ea = E_TS - max(E_R, E_P)
                ea = (ts_e["energy"] - max(r_e["energy"], p_e["energy"])) * config["hartree_to_kcal"]
                all_ea.append(ea)

                # Global energy features for the energy head
                # These are the most predictive features (r=0.63 for dE_rxn alone)
                e_r = r_e["energy"] * config["hartree_to_kcal"]
                e_p = p_e["energy"] * config["hartree_to_kcal"]
                de_rxn = abs(e_r - e_p)
                
                # Geometric displacement features (R-P difference)
                diff = c_R[:n] - c_P[:n]
                diff_norms = np.linalg.norm(diff, axis=1)
                
                energy_feats = np.array([
                    de_rxn,                    # Reaction energy difference
                    diff_norms.mean(),         # Mean displacement
                    diff_norms.std(),          # Displacement spread
                    diff_norms.max(),          # Max displacement
                    float(n),                  # Number of atoms
                ], dtype=np.float32)

                self.atom_types_map[rxn_id] = [a["atom"] for a in ts_e["atoms"]]
                self.samples.append({
                    "rxn_id": rxn_id, "n_atoms": n,
                    "c_R": c_R, "c_P": c_P, "c_TS": c_TS,
                    "atom_ids": torch.from_numpy(atom_ids),
                    "mask": torch.from_numpy(mask),
                    "Ea_raw": ea,
                    "energy_feats_raw": energy_feats,
                    "E_R": r_e["energy"],  # Save raw hartree for prediction output
                    "E_P": p_e["energy"],
                })

        # Compute normalization stats for activation energies
        all_ea = np.array(all_ea)
        self.ea_mean = float(all_ea.mean())
        self.ea_std = float(all_ea.std())
        if self.ea_std < 1e-6:
            self.ea_std = 1.0  # Prevent division by zero
        
        # Compute normalization stats for energy features
        all_efeats = np.stack([s["energy_feats_raw"] for s in self.samples])
        self.efeat_mean = all_efeats.mean(axis=0).astype(np.float32)
        self.efeat_std = all_efeats.std(axis=0).astype(np.float32)
        self.efeat_std[self.efeat_std < 1e-6] = 1.0
        self.n_energy_feats = all_efeats.shape[1]
        
        print(f"Loaded {len(self.samples)} complete reaction triplets.")
        print(f"Ea stats: mean={self.ea_mean:.2f}, std={self.ea_std:.2f} kcal/mol")
        print(f"Ea range: [{all_ea.min():.2f}, {all_ea.max():.2f}] kcal/mol")
        print(f"Energy features: {self.n_energy_feats} dims")
        
        # Pre-compute normalized Ea and energy features
        for sample in self.samples:
            sample["Ea"] = torch.tensor(
                (sample["Ea_raw"] - self.ea_mean) / self.ea_std, 
                dtype=torch.float32
            )
            sample["energy_feats"] = torch.from_numpy(
                (sample["energy_feats_raw"] - self.efeat_mean) / self.efeat_std
            )

    def __len__(self): return len(self.samples)
    
    def __getitem__(self, idx):
        s = self.samples[idx]
        c_R = s["c_R"].copy()
        c_P = s["c_P"].copy()
        c_TS = s["c_TS"].copy()
        
        # Data augmentation: add small coordinate noise during training
        if self.augment:
            noise_std = self.config.get("coord_noise_std", 0.01)
            n = s["n_atoms"]
            noise_R = np.random.randn(n, 3).astype(np.float32) * noise_std
            noise_P = np.random.randn(n, 3).astype(np.float32) * noise_std
            c_R[:n] += noise_R
            c_P[:n] += noise_P
            # Note: c_TS is NOT augmented — it's the target

        # Kabsch-align reactant onto product frame before interpolation.
        # This ensures c_R and c_P share the same rotational frame so that
        # the midpoint c_I is a chemically sensible TS guess rather than an
        # average of coordinates from two different orientations.
        n = s["n_atoms"]
        c_R_aligned = kabsch_align_reactant(c_R, c_P, n)
        c_I = np.zeros_like(c_R)
        c_I[:n] = (c_R_aligned[:n] + c_P[:n]) / 2.0

        # Distance matrices
        D_R = compute_distance_matrix(c_R)
        D_P = compute_distance_matrix(c_P)
        D_I = compute_distance_matrix(c_I)
        D_TS = compute_distance_matrix(c_TS)

        return {
            "rxn_id": s["rxn_id"],
            "n_atoms": s["n_atoms"],
            "D_R": torch.from_numpy(D_R),
            "D_I": torch.from_numpy(D_I),
            "D_P": torch.from_numpy(D_P),
            "D_TS": torch.from_numpy(D_TS),
            "mask": s["mask"],
            "Ea": s["Ea"],
            "atom_ids": s["atom_ids"],
            "energy_feats": s["energy_feats"],
        }


# ============================================================================
# 3. Gaussian Embedding (Step 2)
# ============================================================================

class GaussianEmbedding(nn.Module):
    def __init__(self, n_gaussians=50, start=0.4, stop=6.0):
        super().__init__()
        centers = torch.linspace(start, stop, n_gaussians)
        self.register_buffer("centers", centers)
        self.sigma = (stop - start) / (n_gaussians - 1) * 0.5

    def forward(self, D):
        # D: (B, N, N) -> (B, N, N, K)
        d = D.unsqueeze(-1)
        return torch.exp(-0.5 * ((d - self.centers) / self.sigma) ** 2)


# ============================================================================
# 4. & 5. Bi-GRU and Transformer (Steps 3 & 4)
# ============================================================================

class PreNormTransformerLayer(nn.Module):
    """Pre-norm Transformer layer: LayerNorm before attention and FFN for stable training."""
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
        # Pre-norm: norm before attention
        x2 = self.norm1(x)
        x2, _ = self.attn(x2, x2, x2, key_padding_mask=src_key_padding_mask)
        x = x + x2
        # Pre-norm: norm before FFN
        x2 = self.norm2(x)
        x = x + self.ff(x2)
        return x


class PSICore(nn.Module):
    def __init__(self, config, num_atom_types):
        super().__init__()
        N = config["max_atoms"]; K = config["n_gaussians"]
        atom_dim = config["atom_embed_dim"]
        gru_hidden = config["gru_hidden"]
        d_model = gru_hidden * 2  # Bidirectional
        
        # Atom embedding
        self.atom_embed = nn.Embedding(num_atom_types + 1, atom_dim, padding_idx=0)
        
        # Gaussian distance embedding
        self.gaussian = GaussianEmbedding(K, config["gauss_start"], config["gauss_stop"])
        
        # CRITICAL FIX: Project N*K (= 30*32 = 960) down to d_model before GRU
        # Raw N*K is too high-dimensional for a 3-step GRU to learn from
        raw_dim = N * K + atom_dim  # 960 + 32 = 992
        self.input_proj = nn.Sequential(
            nn.Linear(raw_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(config["dropout"]),
        )
        
        # Step 3: 2-layer Bi-GRU with dropout
        self.gru = nn.GRU(
            input_size=d_model,  # Now d_model instead of raw 992
            hidden_size=gru_hidden, 
            num_layers=config["gru_layers"], 
            batch_first=True, 
            bidirectional=True,
            dropout=config["gru_dropout"] if config["gru_layers"] > 1 else 0.0,
        )
        
        # Post-GRU projection with residual-friendly init
        self.gru_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
        )
        
        # Step 4: Pre-norm Transformer
        self.transformer_layers = nn.ModuleList([
            PreNormTransformerLayer(d_model, config["attn_heads"], config["ff_dim"], config["dropout"])
            for _ in range(config["attn_layers"])
        ])
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, D_R, D_I, D_P, mask, atom_ids):
        B, N, _ = D_R.shape
        
        # Atom embeddings: (B, N, atom_dim)
        atom_emb = self.atom_embed(atom_ids)
        
        # Step 2: Gaussian expansion -> (B, N, N*K)
        emb_R = self.gaussian(D_R).view(B, N, -1)
        emb_I = self.gaussian(D_I).view(B, N, -1)
        emb_P = self.gaussian(D_P).view(B, N, -1)

        # Concatenate atom embeddings to each frame
        emb_R = torch.cat([emb_R, atom_emb], dim=-1)
        emb_I = torch.cat([emb_I, atom_emb], dim=-1)
        emb_P = torch.cat([emb_P, atom_emb], dim=-1)

        # CRITICAL: Project down to d_model before GRU
        emb_R = self.input_proj(emb_R)  # (B, N, d_model)
        emb_I = self.input_proj(emb_I)
        emb_P = self.input_proj(emb_P)

        # Step 3: Sequence [R, I, P] through GRU per atom
        seq = torch.stack([emb_R, emb_I, emb_P], dim=2)  # (B, N, 3, d_model)
        seq = seq.view(B*N, 3, -1)
        out, _ = self.gru(seq)
        context = out[:, 1, :]  # Frame I context: (B*N, d_model)
        context = context.view(B, N, -1)
        context = self.gru_proj(context)

        # Step 4: Pre-norm Transformer stack
        pad_mask = (mask == 0)
        x = context
        for layer in self.transformer_layers:
            x = layer(x, src_key_padding_mask=pad_mask)
        return self.final_norm(x)


# ============================================================================
# 6. Output Heads (Step 5)
# ============================================================================

class GeometryHead(nn.Module):
    """Predicts TS distance matrix as D_I + learned delta."""
    def __init__(self, d_model, atom_embed_dim):
        super().__init__()
        # pair features: fi || fj || atom_i_emb || atom_j_emb || D_R || D_I || D_P
        pair_dim = d_model * 2 + atom_embed_dim * 2 + 3
        self.net = nn.Sequential(
            nn.Linear(pair_dim, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 1),
        )
        # Initialize final layer near zero so initial prediction ≈ D_I
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, features, atom_emb, D_R, D_I, D_P, mask):
        B, N, D = features.shape
        atom_dim = atom_emb.shape[-1]
        
        fi = features.unsqueeze(2).expand(B, N, N, D)
        fj = features.unsqueeze(1).expand(B, N, N, D)
        ai = atom_emb.unsqueeze(2).expand(B, N, N, atom_dim)
        aj = atom_emb.unsqueeze(1).expand(B, N, N, atom_dim)
        pair_dist = torch.stack([D_R, D_I, D_P], dim=-1)
        pair = torch.cat([fi, fj, ai, aj, pair_dist], dim=-1)
        delta = self.net(pair).squeeze(-1)
        
        # Residual: predict as correction to interpolated distances
        D_TS_pred = torch.clamp(D_I + delta, min=0.0)
        # Symmetrize
        D_TS_pred = (D_TS_pred + D_TS_pred.transpose(1, 2)) / 2.0
        # Zero diagonal and mask
        eye = torch.eye(N, device=D_TS_pred.device, dtype=D_TS_pred.dtype).unsqueeze(0)
        valid = mask.unsqueeze(-1) * mask.unsqueeze(-2)
        return D_TS_pred * (1.0 - eye) * valid


class EnergyHead(nn.Module):
    """Predicts activation energy using attention pooling + global energy features.
    
    Key insight from diagnostic: geometry alone gives r=0.20, but reaction energy
    (dE_rxn = |E_R - E_P|) gives r=0.63. The model MUST see energy features.
    """
    def __init__(self, d_model, n_energy_feats=5):
        super().__init__()
        self.ln = nn.LayerNorm(d_model)
        
        # Attention pooling: learnable query
        self.attn_query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.attn_proj_k = nn.Linear(d_model, d_model)
        self.attn_proj_v = nn.Linear(d_model, d_model)
        self.attn_scale = d_model ** 0.5
        
        # Energy feature projection
        self.efeat_proj = nn.Sequential(
            nn.Linear(n_energy_feats, 64),
            nn.GELU(),
            nn.Linear(64, 64),
        )
        
        # Combined MLP: pooled geometric features + energy features
        self.net = nn.Sequential(
            nn.Linear(d_model + 64, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

    def forward(self, features, mask, energy_feats):
        features = self.ln(features)
        B, N, D = features.shape
        
        # Attention pooling
        Q = self.attn_query.expand(B, -1, -1)  # (B, 1, D)
        K = self.attn_proj_k(features)           # (B, N, D)
        V = self.attn_proj_v(features)           # (B, N, D)
        
        scores = torch.bmm(Q, K.transpose(1, 2)) / self.attn_scale  # (B, 1, N)
        
        # Mask padding
        pad_mask = (mask == 0).unsqueeze(1)  # (B, 1, N)
        scores = scores.masked_fill(pad_mask, float('-inf'))
        
        attn_weights = F.softmax(scores, dim=-1)  # (B, 1, N)
        pooled = torch.bmm(attn_weights, V).squeeze(1)  # (B, D)
        
        # Combine with energy features
        efeat = self.efeat_proj(energy_feats)  # (B, 64)
        combined = torch.cat([pooled, efeat], dim=-1)  # (B, D + 64)
        
        return self.net(combined).squeeze(-1)


# ============================================================================
# 7. Full PSI Pipeline Integration
# ============================================================================

class PSI(nn.Module):
    def __init__(self, config, num_atom_types, n_energy_feats=5):
        super().__init__()
        d_model = config["gru_hidden"] * 2
        atom_dim = config["atom_embed_dim"]
        self.core = PSICore(config, num_atom_types)
        self.geom_head = GeometryHead(d_model, atom_dim)
        self.ener_head = EnergyHead(d_model, n_energy_feats)

    def forward(self, D_R, D_I, D_P, mask, atom_ids, energy_feats):
        f = self.core(D_R, D_I, D_P, mask, atom_ids)
        atom_emb = self.core.atom_embed(atom_ids)
        return (
            self.geom_head(f, atom_emb, D_R, D_I, D_P, mask), 
            self.ener_head(f, mask, energy_feats)
        )


# ============================================================================
# 8. Learning Rate Schedule with Warmup
# ============================================================================

class CosineAnnealingWarmup(torch.optim.lr_scheduler._LRScheduler):
    """Cosine annealing with linear warmup."""
    def __init__(self, optimizer, warmup_epochs, total_epochs, min_lr=1e-6, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # Linear warmup
            scale = (self.last_epoch + 1) / max(1, self.warmup_epochs)
            return [base_lr * scale for base_lr in self.base_lrs]
        else:
            # Cosine decay
            progress = (self.last_epoch - self.warmup_epochs) / max(1, self.total_epochs - self.warmup_epochs)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return [self.min_lr + (base_lr - self.min_lr) * cosine for base_lr in self.base_lrs]


# ============================================================================
# 9. Training & Evaluation
# ============================================================================

def get_energy_weight(epoch, config):
    """Ramp energy weight from start to end over ramp_epochs."""
    start = config["energy_weight_start"]
    end = config["energy_weight_end"]
    ramp = config["energy_ramp_epochs"]
    if epoch >= ramp:
        return end
    t = epoch / ramp
    return start + (end - start) * t


def run_epoch(model, loader, optimizer, scaler, device, config, use_amp, epoch, is_train=True):
    """Run one training or evaluation epoch. Returns dict of averaged metrics."""
    if is_train:
        model.train()
    else:
        model.eval()
    
    total_loss = 0.0
    total_geom = 0.0
    total_ener = 0.0
    n_batches = 0
    
    energy_w = get_energy_weight(epoch, config)
    
    context = torch.no_grad() if not is_train else torch.enable_grad()
    with context:
        for batch in loader:
            DR, DI, DP, DTS, mask, true_ea, atom_ids, energy_feats = move_batch_to_device(batch, device)
            
            if is_train:
                optimizer.zero_grad(set_to_none=True)
            
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                p_DTS, p_ea = model(DR, DI, DP, mask, atom_ids, energy_feats)
                
                # Geometry: Huber loss (more robust to outliers)
                m2d = mask.unsqueeze(-1) * mask.unsqueeze(-2)
                l_geom = (F.huber_loss(p_DTS * m2d, DTS * m2d, reduction='sum', delta=0.5) 
                          / m2d.sum().clamp(min=1))
                
                # Energy: MSE on normalized targets
                l_ener = F.mse_loss(p_ea, true_ea)
                
                loss = l_geom + energy_w * l_ener
            
            if is_train:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])
                scaler.step(optimizer)
                scaler.update()
            
            total_loss += loss.item()
            total_geom += l_geom.item()
            total_ener += l_ener.item()
            n_batches += 1
    
    return {
        "loss": total_loss / max(n_batches, 1),
        "geom": total_geom / max(n_batches, 1),
        "ener": total_ener / max(n_batches, 1),
    }


def train_pipeline(config):
    device = resolve_device(config)
    configure_torch_runtime(device)

    print("="*70); print(" PSI FULL PIPELINE (v2 - Improved) "); print("="*70)
    
    # 1. & 2. Data
    extract_raw_data(config)
    dataset = ReactionDataset(config, augment=True)
    if len(dataset) == 0:
        print("Error: No complete reaction triplets found."); return
    
    # Train/validation split
    n_total = len(dataset)
    n_val = max(1, int(n_total * config["val_split"]))
    n_train = n_total - n_val
    
    rng = torch.Generator().manual_seed(config["split_seed"])
    indices = torch.randperm(n_total, generator=rng).tolist()
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]
    
    # For validation, create a non-augmenting version
    val_dataset = ReactionDataset(config, augment=False)
    
    train_subset = Subset(dataset, train_indices)
    val_subset = Subset(val_dataset, val_indices)
    
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
    eval_loader = DataLoader(
        Subset(val_dataset, list(range(n_total))),  # Full dataset for final eval
        shuffle=False, **loader_kwargs
    )
    
    # 3-6. Model
    num_atom_types = len(dataset.atom_vocab)
    n_energy_feats = dataset.n_energy_feats
    model = PSI(config, num_atom_types, n_energy_feats).to(device)
    
    # Count parameters
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")
    
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=config["lr"], 
        weight_decay=config["weight_decay"]
    )
    scheduler = CosineAnnealingWarmup(
        optimizer, 
        warmup_epochs=config["warmup_epochs"], 
        total_epochs=config["epochs"]
    )
    use_amp = config["amp"] and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    
    # Save metadata for prediction
    metadata = {
        "atom_vocab": dataset.atom_vocab,
        "ea_mean": dataset.ea_mean,
        "ea_std": dataset.ea_std,
        "efeat_mean": dataset.efeat_mean.tolist(),
        "efeat_std": dataset.efeat_std.tolist(),
        "n_energy_feats": n_energy_feats,
        "config_snapshot": {k: v for k, v in config.items() 
                          if isinstance(v, (int, float, str, bool))},
    }
    
    # Training loop
    print(f"\nTraining for up to {config['epochs']} epochs (patience={config['patience']})...")
    print(f"{'Epoch':>6} | {'Train Loss':>11} | {'Val Loss':>11} | {'T.Geom':>8} | {'T.Ener':>8} | {'V.Geom':>8} | {'V.Ener':>8} | {'LR':>10}")
    print("-" * 95)
    
    best_val_loss = float('inf')
    patience_counter = 0
    history = []
    best_model_path = os.path.join(config["save_dir"], "psi_best.pt")
    
    for epoch in range(1, config["epochs"] + 1):
        # Train
        train_metrics = run_epoch(model, train_loader, optimizer, scaler, device, config, use_amp, epoch, is_train=True)
        
        # Validate
        val_metrics = run_epoch(model, val_loader, None, scaler, device, config, use_amp, epoch, is_train=False)
        
        # Step scheduler
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # Record history
        history.append({
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "val_loss": val_metrics["loss"],
            "train_geom": train_metrics["geom"],
            "val_geom": val_metrics["geom"],
            "train_ener": train_metrics["ener"],
            "val_ener": val_metrics["ener"],
            "lr": current_lr,
        })
        
        # Early stopping check
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            patience_counter = 0
            # Save best model
            torch.save({
                "model_state_dict": model.state_dict(),
                "metadata": metadata,
            }, best_model_path)
        else:
            patience_counter += 1
        
        # Logging
        if epoch % config["print_every"] == 0 or epoch == 1 or patience_counter == 0:
            marker = " *" if patience_counter == 0 else ""
            print(f"{epoch:6d} | {train_metrics['loss']:11.4f} | {val_metrics['loss']:11.4f} | "
                  f"{train_metrics['geom']:8.5f} | {train_metrics['ener']:8.5f} | "
                  f"{val_metrics['geom']:8.5f} | {val_metrics['ener']:8.5f} | "
                  f"{current_lr:10.2e}{marker}")
        
        # Early stopping
        if patience_counter >= config["patience"]:
            print(f"\nEarly stopping at epoch {epoch} (no improvement for {config['patience']} epochs)")
            break
    
    # Save training history
    history_path = os.path.join(config["save_dir"], "training_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"\nTraining history saved to {history_path}")
    
    # Load best model for evaluation
    print(f"\nLoading best model (val_loss={best_val_loss:.4f})...")
    checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    
    # 9. Final Evaluation on FULL dataset
    print("\n" + "="*70); print(" EVALUATION RESULTS "); print("="*70)
    model.eval()
    results = []
    ea_mean = dataset.ea_mean
    ea_std = dataset.ea_std
    
    with torch.no_grad():
        for batch in eval_loader:
            DR, DI, DP, DTS, mask, true_ea_norm, atom_ids, energy_feats = move_batch_to_device(batch, device)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                p_DTS, p_ea_norm = model(DR, DI, DP, mask, atom_ids, energy_feats)
            
            # Denormalize energy predictions
            true_ea_real = true_ea_norm * ea_std + ea_mean
            p_ea_real = p_ea_norm * ea_std + ea_mean
            
            for i in range(len(batch["rxn_id"])):
                rxn_id = batch["rxn_id"][i]
                n = int(mask[i].sum().item())
                di = DI[i, :n, :n].cpu().numpy()
                dp = p_DTS[i, :n, :n].cpu().numpy()
                dt = DTS[i, :n, :n].cpu().numpy()
                
                d_mae = np.abs(dp - dt).mean().item()
                ea_true = true_ea_real[i].item()
                ea_pred = p_ea_real[i].item()
                e_err = abs(ea_pred - ea_true)
                
                split = "val" if batch["rxn_id"][i] in [dataset.samples[vi]["rxn_id"] for vi in val_indices] else "train"
                atom_types = dataset.atom_types_map.get(rxn_id, [])
                
                results.append({
                    "rxn_id": rxn_id,
                    "split": split,
                    "Ea_true": ea_true, "Ea_pred": ea_pred,
                    "Ea_error": e_err, "dist_MAE": d_mae,
                    "n_atoms": n,
                    "atom_types": atom_types,
                    "D_I": di.tolist(), "D_pred": dp.tolist(), "D_true": dt.tolist()
                })

    # Print summary
    train_results = [r for r in results if r["split"] == "train"]
    val_results = [r for r in results if r["split"] == "val"]
    
    def print_stats(name, res_list):
        if not res_list:
            return
        ea_errs = [r["Ea_error"] for r in res_list]
        d_maes = [r["dist_MAE"] for r in res_list]
        ea_trues = [r["Ea_true"] for r in res_list]
        ea_preds = [r["Ea_pred"] for r in res_list]
        corr = np.corrcoef(ea_trues, ea_preds)[0, 1] if len(ea_trues) > 1 else 0.0
        print(f"\n{name} ({len(res_list)} reactions):")
        print(f"  Ea MAE:        {np.mean(ea_errs):8.2f} kcal/mol")
        print(f"  Ea Correlation: {corr:8.4f}")
        print(f"  Dist MAE:      {np.mean(d_maes):8.4f} Å")
        print(f"  Dist MAE std:  {np.std(d_maes):8.4f} Å")
    
    print_stats("TRAIN SET", train_results)
    print_stats("VALIDATION SET", val_results)
    print_stats("ALL DATA", results)
    
    # Detailed table
    print(f"\n{'Reaction':<15} {'Split':<6} {'Ea True':>10} {'Ea Pred':>10} {'Ea Err':>10} {'Dist MAE':>10}")
    for r in sorted(results, key=lambda x: x["rxn_id"]):
        print(f"{r['rxn_id']:<15} {r['split']:<6} {r['Ea_true']:10.2f} {r['Ea_pred']:10.2f} {r['Ea_error']:10.2f} {r['dist_MAE']:10.4f}")
    
    # Save predictions
    output_path = os.path.join(config["save_dir"], "detailed_analysis.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    # Also save as psi_final.pt for backward compat
    final_path = os.path.join(config["save_dir"], "psi_final.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "metadata": metadata,
    }, final_path)
    
    print(f"\nModel saved to {final_path}")
    print(f"Predictions saved to {output_path}")


def predict_transition_state(config, reactant_path, product_path, model_path, output_path, xyz_path=None):
    """Predict a transition-state distance matrix and approximate coordinates."""
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

    # Load checkpoint with metadata
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # Handle both old and new checkpoint formats
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        meta = checkpoint.get("metadata", {})
    else:
        state_dict = checkpoint
        meta = {}
    
    atom_vocab = meta.get("atom_vocab", {})
    ea_mean = meta.get("ea_mean", 0.0)
    ea_std = meta.get("ea_std", 1.0)
    
    # If no saved vocab, build a minimal one from the input
    if not atom_vocab:
        print("WARNING: No atom vocabulary in checkpoint. Building from input atoms.")
        unique_atoms = sorted(set(r_types))
        atom_vocab = {atom: i + 1 for i, atom in enumerate(unique_atoms)}
    
    num_atom_types = max(atom_vocab.values()) if atom_vocab else 0
    n_energy_feats = meta.get("n_energy_feats", 5)
    efeat_mean = np.array(meta.get("efeat_mean", [0.0] * n_energy_feats), dtype=np.float32)
    efeat_std = np.array(meta.get("efeat_std", [1.0] * n_energy_feats), dtype=np.float32)
    
    n = len(r_atoms)
    c_R = padded_coords(r_atoms, config["max_atoms"])
    c_P = padded_coords(p_atoms, config["max_atoms"])
    c_I = (c_R + c_P) / 2.0

    D_R = compute_distance_matrix(c_R)
    D_P = compute_distance_matrix(c_P)
    D_I = compute_distance_matrix(c_I)
    mask = np.zeros(config["max_atoms"], dtype=np.float32)
    mask[:n] = 1.0
    
    # Atom IDs
    atom_ids = np.zeros(config["max_atoms"], dtype=np.int64)
    for i, atom_type in enumerate(r_types):
        atom_ids[i] = atom_vocab.get(atom_type, 0)
    
    # Energy features (same as training)
    e_r = reactant["energy"] * config["hartree_to_kcal"]
    e_p = product["energy"] * config["hartree_to_kcal"]
    de_rxn = abs(e_r - e_p)
    diff = c_R[:n] - c_P[:n]
    diff_norms = np.linalg.norm(diff, axis=1)
    energy_feats = np.array([
        de_rxn, diff_norms.mean(), diff_norms.std(), diff_norms.max(), float(n)
    ], dtype=np.float32)
    energy_feats_norm = (energy_feats - efeat_mean) / efeat_std

    model = PSI(config, num_atom_types, n_energy_feats).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    with torch.no_grad():
        t_DR = torch.from_numpy(D_R).unsqueeze(0).to(device)
        t_DI = torch.from_numpy(D_I).unsqueeze(0).to(device)
        t_DP = torch.from_numpy(D_P).unsqueeze(0).to(device)
        t_mask = torch.from_numpy(mask).unsqueeze(0).to(device)
        t_atom_ids = torch.from_numpy(atom_ids).unsqueeze(0).to(device)
        t_efeats = torch.from_numpy(energy_feats_norm).unsqueeze(0).to(device)
        p_DTS, p_ea_norm = model(t_DR, t_DI, t_DP, t_mask, t_atom_ids, t_efeats)

    pred_dist = p_DTS[0, :n, :n].cpu().numpy()
    pred_dist = np.maximum((pred_dist + pred_dist.T) / 2.0, 0.0)
    np.fill_diagonal(pred_dist, 0.0)

    # -----------------------------------------------------------------------
    # Apply TS physical rules to remove impossible geometries
    # -----------------------------------------------------------------------

    # Rule 0: Kabsch-align reactant → correct reference frame for interpolation
    c_R_aligned = kabsch_align_reactant(c_R, c_P, n)
    c_I_real = (c_R_aligned[:n] + c_P[:n]) / 2.0  # chemically valid midpoint

    # Rule 1: Clamp steric collisions (nuclear overlap prevention)
    pred_dist = clamp_steric_collisions(pred_dist, r_types[:n])

    # Rule 2: Lock spectator bonds within ±tol of their R/P reference value
    pred_dist = apply_spectator_constraints(
        pred_dist,
        D_R[:n, :n], D_P[:n, :n], n,
        threshold=config.get("spectator_threshold", 0.15),
        tol=config.get("spectator_tol", 0.05),
    )

    # Rule 3: Enforce triangle inequality so MDS produces a valid 3-D geometry
    pred_dist = enforce_triangle_inequality(pred_dist)

    # Rule 4: Validate and report any remaining physical violations
    validate_ts_geometry(
        pred_dist, D_R[:n, :n], D_P[:n, :n], r_types[:n], n,
        spectator_threshold=config.get("spectator_threshold", 0.15),
    )

    # -----------------------------------------------------------------------
    # Reconstruct 3-D coordinates from the corrected distance matrix
    # Use the Kabsch-corrected midpoint as the alignment reference
    # -----------------------------------------------------------------------
    pred_coords = kabsch(mds(pred_dist), c_I_real)

    # Denormalize energy prediction
    energy_pred = float(p_ea_norm.item()) * ea_std + ea_mean

    result = {
        "reactant_path": reactant_path,
        "product_path": product_path,
        "model_path": model_path,
        "n_atoms": n,
        "atom_types": r_types,
        "Ea_pred": energy_pred,
        "D_I": D_I[:n, :n].tolist(),
        "D_pred": pred_dist.tolist(),
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
    print(f"Predicted activation energy: {energy_pred:.4f} kcal/mol")
    print(f"Prediction JSON saved to: {output_path}")
    if xyz_path:
        print(f"Predicted TS XYZ saved to: {xyz_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PSI transition-state training and prediction (v2)")
    subparsers = parser.add_subparsers(dest="command")

    train_parser = subparsers.add_parser("train", help="Train the PSI model and evaluate known triplets")
    train_parser.add_argument("--extract-limit", type=int, default=CONFIG["extraction_limit"], help="Number of log files to parse from the tarball")
    train_parser.add_argument("--force-extract", action="store_true", help="Rebuild extracted_dataset.json instead of reusing it")
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

    args = parser.parse_args()
    if args.command == "predict":
        CONFIG["device"] = args.device
        CONFIG["require_cuda"] = args.require_cuda
        predict_transition_state(CONFIG, args.reactant, args.product, args.model, args.output, args.xyz)
    else:
        if args.command == "train":
            CONFIG["extraction_limit"] = args.extract_limit
            CONFIG["force_extract"] = args.force_extract
            CONFIG["epochs"] = args.epochs
            CONFIG["batch_size"] = args.batch_size
            CONFIG["num_workers"] = args.num_workers
            CONFIG["device"] = args.device
            CONFIG["require_cuda"] = args.require_cuda
            CONFIG["amp"] = not args.no_amp
            CONFIG["patience"] = args.patience
            CONFIG["lr"] = args.lr
        train_pipeline(CONFIG)
