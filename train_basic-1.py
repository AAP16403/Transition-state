"""
PSI Model: Predicting Specific Interactions
============================================
AI-Driven Transition State Prediction -- Training Script (Basic-1)

Architecture:
  1. Distance Matrix Construction  (R, I=(R+P)/2, P)
  2. Gaussian Embedding            (RBF expansion of distances)
  3. Bidirectional GRU              (temporal reaction dynamics)
  4. Transformer Self-Attention     (inter-atom geometric reasoning)
  5. Dual Output Heads              (geometry α-scaling + energy scalar)

Trained on 16 reaction triplets from extracted_dataset.json.
"""

import json
import math
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ============================================================================
# Configuration
# ============================================================================

# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

CONFIG = {
    "data_path": r"d:\Transition state\extracted_dataset.json",
    "save_dir": r"d:\Transition state",
    "max_atoms": 17,           # pad all molecules to this size
    "n_gaussians": 32,         # number of Gaussian basis functions
    "gauss_start": 0.5,        # A -- start of Gaussian centers
    "gauss_stop": 5.0,         # A -- end of Gaussian centers
    "gru_hidden": 128,         # hidden dim per direction in Bi-GRU
    "attn_heads": 4,           # number of attention heads
    "attn_layers": 2,          # number of transformer encoder layers
    "ff_dim": 512,             # feedforward dim in transformer  
    "dropout": 0.1,
    "energy_weight": 10.0,     # lambda for energy loss balancing
    "lr": 1e-3,
    "epochs": 500,
    "print_every": 25,
    "hartree_to_kcal": 627.509474,
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


# ============================================================================
# Step 0: Data Loading & Triplet Grouping
# ============================================================================

def load_and_group_reactions(json_path):
    """
    Load the JSON dataset and group entries into (R, P, TS) triplets
    by reaction ID. Skip incomplete reactions.
    """
    with open(json_path, "r") as f:
        raw_data = json.load(f)

    # Group by reaction ID
    reactions = {}
    for entry in raw_data:
        # filename format: "b97d3/rxnXXXXXX/[r|p|ts]XXXXXX.log"
        parts = entry["filename"].split("/")
        rxn_id = parts[1]  # e.g., "rxn009959"
        prefix = parts[2].split(".")[0]  # e.g., "r009959", "p009959", "ts009959"

        if prefix.startswith("ts"):
            role = "ts"
        elif prefix.startswith("r"):
            role = "r"
        elif prefix.startswith("p"):
            role = "p"
        else:
            continue

        if rxn_id not in reactions:
            reactions[rxn_id] = {}
        reactions[rxn_id][role] = entry

    # Filter to complete triplets only
    complete = {}
    for rxn_id, roles in reactions.items():
        if "r" in roles and "p" in roles and "ts" in roles:
            complete[rxn_id] = roles
        else:
            missing = {"r", "p", "ts"} - set(roles.keys())
            print(f"  [SKIP] {rxn_id} — missing: {missing}")

    print(f"  Loaded {len(complete)} complete reaction triplets\n")
    return complete


def extract_coords(entry, max_atoms):
    """
    Extract Cartesian coordinates from a data entry.
    Returns padded coords (max_atoms, 3) and atom count.
    """
    atoms = entry["atoms"]
    n = len(atoms)
    coords = np.zeros((max_atoms, 3), dtype=np.float32)
    for i, a in enumerate(atoms):
        coords[i] = [a["x"], a["y"], a["z"]]
    return coords, n


def extract_atom_types(entry, max_atoms):
    """Extract atom type strings, padded with empty strings."""
    atoms = entry["atoms"]
    types = [""] * max_atoms
    for i, a in enumerate(atoms):
        types[i] = a["atom"]
    return types


# ============================================================================
# Step 1: Distance Matrix Construction
# ============================================================================

def compute_distance_matrix(coords):
    """
    Compute pairwise Euclidean distance matrix.
    coords: (N, 3) numpy array
    Returns: (N, N) distance matrix
    """
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]  # (N, N, 3)
    dist = np.sqrt(np.sum(diff ** 2, axis=-1) + 1e-8)  # (N, N)
    return dist.astype(np.float32)


# ============================================================================
# Step 2: Gaussian Embedding
# ============================================================================

class GaussianEmbedding(nn.Module):
    """
    Expands scalar distances into Gaussian (RBF) feature vectors.

    Each distance d is mapped to a K-dimensional vector:
        phi_k(d) = exp(-(d - mu_k)^2 / (2*sigma^2))

    where mu_k are K centers evenly spaced from start to stop.
    """

    def __init__(self, n_gaussians=32, start=0.5, stop=5.0):
        super().__init__()
        self.n_gaussians = n_gaussians
        centers = torch.linspace(start, stop, n_gaussians)
        self.register_buffer("centers", centers)
        # sigma chosen so adjacent Gaussians overlap well
        self.sigma = (stop - start) / (n_gaussians - 1) * 0.5

    def forward(self, distances):
        """
        distances: (B, N, N)
        returns:   (B, N, N, K) Gaussian embeddings
        """
        d = distances.unsqueeze(-1)  # (B, N, N, 1)
        return torch.exp(-0.5 * ((d - self.centers) / self.sigma) ** 2)


# ============================================================================
# Step 3: Bidirectional GRU ("The Time Machine")
# ============================================================================

class TemporalEncoder(nn.Module):
    """
    Processes the 3-frame sequence [R, I, P] per atom through a Bi-GRU.

    For each atom i, the input is a sequence of 3 vectors (one per frame),
    each capturing atom i's local environment via Gaussian-embedded distances.

    The Bi-GRU learns:
      - Forward pass:  R → I  (structural velocity of bond formation)
      - Backward pass:  P → I  (confirms reverse trajectory)

    Output: context vector per atom with shape (2 * hidden_dim).
    """

    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.norm = nn.LayerNorm(hidden_dim * 2)

    def forward(self, x):
        """
        x: (B, N, 3, D) — 3 frames, D features per atom per frame
        returns: (B, N, 2*hidden_dim) — context vector per atom
        """
        B, N, T, D = x.shape
        # Reshape to process each atom's sequence independently
        x_flat = x.view(B * N, T, D)  # (B*N, 3, D)
        output, h_n = self.gru(x_flat)  # output: (B*N, 3, 2*H)
        # Take output at the middle frame (I) — the transition point
        context = output[:, 1, :]  # (B*N, 2*H), frame index 1 = Interpolated
        context = context.view(B, N, -1)  # (B, N, 2*H)
        return self.norm(context)


# ============================================================================
# Step 4: Transformer Self-Attention ("Shape Logic")
# ============================================================================

class GeometricAttention(nn.Module):
    """
    Multi-head self-attention for inter-atom communication.

    Uses the QKV mechanism:
      - Query: "What do I need to know about my geometric environment?"
      - Key: "Where am I and what is my current bonding state?"
      - Value: "Here is my structural information to share."

    The attention scores determine how heavily each atom's corrected
    geometry is influenced by its neighbors. This is where the model
    distinguishes "Good Weird" (valid TS geometry) from "Bad Weird"
    (interpolation artifacts).
    """

    def __init__(self, d_model, n_heads=4, n_layers=2, ff_dim=512, dropout=0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, padding_mask=None):
        """
        x: (B, N, d_model) — atom context vectors
        padding_mask: (B, N) — True where atoms are padding
        returns: (B, N, d_model) — corrected feature vectors
        """
        out = self.encoder(x, src_key_padding_mask=padding_mask)
        return self.norm(out)


# ============================================================================
# Step 5: Dual Output Heads
# ============================================================================

class GeometryHead(nn.Module):
    """
    Head A — The Geometry Mechanic.

    Reads corrected atom vectors and predicts a scaling matrix alpha (NxN).
    The predicted TS distance matrix = alpha * D_I (element-wise product).

    This preserves the interpolated structure while learning to stretch
    or compress specific bonds to match the true transition state.
    """

    def __init__(self, d_model, max_atoms):
        super().__init__()
        self.max_atoms = max_atoms
        # Per-atom pair prediction: combine atom_i and atom_j features
        self.pair_net = nn.Sequential(
            nn.Linear(d_model * 2, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 1),
        )
        # Bias alpha toward 1.0 (i.e., start close to interpolated geometry)
        nn.init.zeros_(self.pair_net[-1].weight)
        nn.init.ones_(self.pair_net[-1].bias)

    def forward(self, atom_features, D_I):
        """
        atom_features: (B, N, d_model)
        D_I: (B, N, N) — interpolated distance matrix
        returns: (B, N, N) — predicted TS distance matrix
        """
        B, N, D = atom_features.shape
        # Build pairwise features
        fi = atom_features.unsqueeze(2).expand(B, N, N, D)  # (B,N,N,D)
        fj = atom_features.unsqueeze(1).expand(B, N, N, D)  # (B,N,N,D)
        pair = torch.cat([fi, fj], dim=-1)  # (B, N, N, 2D)
        alpha = self.pair_net(pair).squeeze(-1)  # (B, N, N)
        # Ensure alpha > 0 using softplus
        alpha = F.softplus(alpha)
        # Predicted TS distances = α * D_I
        D_TS_pred = alpha * D_I
        # Symmetrize
        D_TS_pred = (D_TS_pred + D_TS_pred.transpose(1, 2)) / 2.0
        return D_TS_pred


class EnergyHead(nn.Module):
    """
    Head B — The Energy Appraiser.

    Aggregates all atomic context vectors into a single global vector
    representing the structural stress of the entire molecule, then
    predicts the activation energy through dense layers.
    """

    def __init__(self, d_model):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

    def forward(self, atom_features, atom_mask):
        """
        atom_features: (B, N, d_model)
        atom_mask: (B, N) — 1.0 for real atoms, 0.0 for padding
        returns: (B,) — predicted activation energy in kcal/mol
        """
        # Masked mean pooling
        mask = atom_mask.unsqueeze(-1)  # (B, N, 1)
        pooled = (atom_features * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        return self.net(pooled).squeeze(-1)


# ============================================================================
# Full PSI Model
# ============================================================================

class PSIModel(nn.Module):
    """
    The complete PSI (Predicting Specific Interactions) model.

    Pipeline:
      Coords(R,P) → DistMatrix → GaussianEmbed → BiGRU → Attention → Geometry + Energy
    """

    def __init__(self, config):
        super().__init__()
        N = config["max_atoms"]
        K = config["n_gaussians"]
        atom_input_dim = N * K  # per-atom feature: distances to all atoms × K Gaussians

        self.max_atoms = N
        self.gaussian_embed = GaussianEmbedding(
            n_gaussians=K,
            start=config["gauss_start"],
            stop=config["gauss_stop"],
        )

        # d_model = 2 * gru_hidden (bidirectional)
        d_model = config["gru_hidden"] * 2

        self.temporal = TemporalEncoder(
            input_dim=atom_input_dim,
            hidden_dim=config["gru_hidden"],
        )

        self.attention = GeometricAttention(
            d_model=d_model,
            n_heads=config["attn_heads"],
            n_layers=config["attn_layers"],
            ff_dim=config["ff_dim"],
            dropout=config["dropout"],
        )

        self.geometry_head = GeometryHead(d_model, N)
        self.energy_head = EnergyHead(d_model)

    def forward(self, D_R, D_I, D_P, atom_mask):
        """
        D_R, D_I, D_P: (B, N, N) — distance matrices for Reactant, Interpolated, Product
        atom_mask: (B, N) — 1.0 for real atoms, 0.0 for padding

        Returns:
            D_TS_pred: (B, N, N) — predicted TS distance matrix
            Ea_pred:   (B,) — predicted activation energy (kcal/mol)
        """
        B, N, _ = D_R.shape

        # Step 2: Gaussian Embedding
        emb_R = self.gaussian_embed(D_R)  # (B, N, N, K)
        emb_I = self.gaussian_embed(D_I)
        emb_P = self.gaussian_embed(D_P)

        # Reshape: per atom, flatten distances-to-all-atoms × K
        # (B, N, N, K) → (B, N, N*K)
        emb_R = emb_R.view(B, N, -1)
        emb_I = emb_I.view(B, N, -1)
        emb_P = emb_P.view(B, N, -1)

        # Step 3: Bi-GRU — stack 3 frames as a sequence
        # (B, N, 3, D)
        temporal_input = torch.stack([emb_R, emb_I, emb_P], dim=2)
        context = self.temporal(temporal_input)  # (B, N, 2*H)

        # Step 4: Transformer Self-Attention
        padding_mask = (atom_mask == 0)  # True = padding
        attended = self.attention(context, padding_mask=padding_mask)  # (B, N, 2*H)

        # Step 5: Dual Output
        D_TS_pred = self.geometry_head(attended, D_I)  # (B, N, N)
        Ea_pred = self.energy_head(attended, atom_mask)  # (B,)

        return D_TS_pred, Ea_pred


# ============================================================================
# Dataset
# ============================================================================

class ReactionDataset(Dataset):
    """
    Dataset of chemical reaction triplets (R, P, TS).

    Each sample contains:
      - Distance matrices for R, I=(R+P)/2, P, and TS (ground truth)
      - Atom mask (which positions are real atoms vs padding)
      - Activation energy Ea = (E_TS - max(E_R, E_P)) in kcal/mol
      - Reaction ID for identification
    """

    def __init__(self, reactions_dict, config):
        self.max_atoms = config["max_atoms"]
        self.h2kcal = config["hartree_to_kcal"]
        self.samples = []

        for rxn_id, roles in sorted(reactions_dict.items()):
            r_entry = roles["r"]
            p_entry = roles["p"]
            ts_entry = roles["ts"]

            # Extract coordinates
            coords_R, n_atoms = extract_coords(r_entry, self.max_atoms)
            coords_P, _ = extract_coords(p_entry, self.max_atoms)
            coords_TS, _ = extract_coords(ts_entry, self.max_atoms)

            # Step 1: Interpolated frame
            coords_I = (coords_R + coords_P) / 2.0

            # Distance matrices
            D_R = compute_distance_matrix(coords_R)
            D_I = compute_distance_matrix(coords_I)
            D_P = compute_distance_matrix(coords_P)
            D_TS = compute_distance_matrix(coords_TS)

            # Atom mask
            atom_mask = np.zeros(self.max_atoms, dtype=np.float32)
            atom_mask[:n_atoms] = 1.0

            # Activation energy: Ea = E_TS - max(E_R, E_P)
            E_R = r_entry["energy"]
            E_P = p_entry["energy"]
            E_TS = ts_entry["energy"]
            Ea = (E_TS - max(E_R, E_P)) * self.h2kcal  # convert to kcal/mol

            self.samples.append({
                "rxn_id": rxn_id,
                "D_R": torch.from_numpy(D_R),
                "D_I": torch.from_numpy(D_I),
                "D_P": torch.from_numpy(D_P),
                "D_TS": torch.from_numpy(D_TS),
                "atom_mask": torch.from_numpy(atom_mask),
                "Ea": torch.tensor(Ea, dtype=torch.float32),
                "n_atoms": n_atoms,
            })

        print(f"  Dataset initialized: {len(self.samples)} reactions")
        print(f"  Activation energies (kcal/mol):")
        for s in self.samples:
            print(f"    {s['rxn_id']}: Ea = {s['Ea'].item():+.2f} kcal/mol  "
                  f"({s['n_atoms']} atoms)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_fn(batch):
    """Custom collate: stack tensors, collect metadata."""
    return {
        "rxn_id": [s["rxn_id"] for s in batch],
        "D_R": torch.stack([s["D_R"] for s in batch]),
        "D_I": torch.stack([s["D_I"] for s in batch]),
        "D_P": torch.stack([s["D_P"] for s in batch]),
        "D_TS": torch.stack([s["D_TS"] for s in batch]),
        "atom_mask": torch.stack([s["atom_mask"] for s in batch]),
        "Ea": torch.stack([s["Ea"] for s in batch]),
        "n_atoms": [s["n_atoms"] for s in batch],
    }


# ============================================================================
# Loss Functions
# ============================================================================

def masked_mse_loss(pred, target, mask_2d):
    """
    MSE loss on distance matrices, ignoring padding.
    mask_2d: (B, N, N) — 1.0 for valid pairs, 0.0 for padding.
    """
    diff = (pred - target) ** 2
    masked_diff = diff * mask_2d
    return masked_diff.sum() / mask_2d.sum().clamp(min=1)


def make_pair_mask(atom_mask):
    """
    Create a 2D mask from atom mask.
    atom_mask: (B, N) → pair_mask: (B, N, N)
    Only pairs where BOTH atoms are real are 1.0.
    """
    return atom_mask.unsqueeze(-1) * atom_mask.unsqueeze(-2)


# ============================================================================
# Training
# ============================================================================

def train(config):
    print("=" * 70)
    print("  PSI Model — Training Script (Basic-1)")
    print("=" * 70)
    print()

    # Load data
    print("[1/5] Loading reaction triplets...")
    reactions = load_and_group_reactions(config["data_path"])

    # Build dataset
    print("[2/5] Building dataset...")
    dataset = ReactionDataset(reactions, config)
    loader = DataLoader(dataset, batch_size=len(dataset), shuffle=True, collate_fn=collate_fn)

    # Build model
    print(f"\n[3/5] Building PSI model...")
    model = PSIModel(config).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model parameters: {n_params:,}")
    print(f"  Architecture:")
    print(f"    Gaussian Embedding: {config['n_gaussians']} centers "
          f"({config['gauss_start']}-{config['gauss_stop']} A)")
    print(f"    Bi-GRU: hidden={config['gru_hidden']}, d_model={config['gru_hidden']*2}")
    print(f"    Transformer: {config['attn_layers']} layers, {config['attn_heads']} heads")
    print(f"    Geometry Head: predicts alpha (NxN scaling matrix)")
    print(f"    Energy Head: global pooling -> MLP -> scalar Ea")

    # Optimizer & scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=50
    )

    lam = config["energy_weight"]

    # Training loop
    print(f"\n[4/5] Training for {config['epochs']} epochs...")
    print(f"  Loss = MSE_geometry + {lam} x MSE_energy")
    print("-" * 70)
    print(f"  {'Epoch':>6}  {'Total':>10}  {'Geom':>10}  {'Energy':>10}  "
          f"{'MAE_dist':>10}  {'MAE_Ea':>10}  {'LR':>12}")
    print("-" * 70)

    best_loss = float("inf")
    history = []

    for epoch in range(1, config["epochs"] + 1):
        model.train()
        epoch_loss = 0.0

        for batch in loader:
            D_R = batch["D_R"].to(DEVICE)
            D_I = batch["D_I"].to(DEVICE)
            D_P = batch["D_P"].to(DEVICE)
            D_TS = batch["D_TS"].to(DEVICE)
            atom_mask = batch["atom_mask"].to(DEVICE)
            Ea_true = batch["Ea"].to(DEVICE)

            pair_mask = make_pair_mask(atom_mask)

            # Forward
            D_TS_pred, Ea_pred = model(D_R, D_I, D_P, atom_mask)

            # Losses
            loss_geom = masked_mse_loss(D_TS_pred, D_TS, pair_mask)
            loss_energy = F.mse_loss(Ea_pred, Ea_true)
            loss = loss_geom + lam * loss_energy

            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            epoch_loss = loss.item()

        scheduler.step(epoch_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        # Compute metrics
        with torch.no_grad():
            model.eval()
            for batch in loader:
                D_R = batch["D_R"].to(DEVICE)
                D_I = batch["D_I"].to(DEVICE)
                D_P = batch["D_P"].to(DEVICE)
                D_TS = batch["D_TS"].to(DEVICE)
                atom_mask = batch["atom_mask"].to(DEVICE)
                Ea_true = batch["Ea"].to(DEVICE)
                pair_mask = make_pair_mask(atom_mask)

                D_TS_pred, Ea_pred = model(D_R, D_I, D_P, atom_mask)

                mae_dist_val = (torch.abs(D_TS_pred - D_TS) * pair_mask).sum() / pair_mask.sum()
                mae_ea_val = torch.abs(Ea_pred - Ea_true).mean()

        record = {
            "epoch": epoch,
            "loss": epoch_loss,
            "loss_geom": loss_geom.item(),
            "loss_energy": loss_energy.item(),
            "mae_dist": mae_dist_val.item(),
            "mae_ea": mae_ea_val.item(),
            "lr": current_lr,
        }
        history.append(record)

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), os.path.join(config["save_dir"], "psi_basic1_best.pt"))

        if epoch % config["print_every"] == 0 or epoch == 1:
            print(f"  {epoch:6d}  {epoch_loss:10.6f}  {loss_geom.item():10.6f}  "
                  f"{loss_energy.item():10.6f}  {mae_dist_val.item():10.4f} A  "
                  f"{mae_ea_val.item():10.4f}  {current_lr:.1e}")

    print("-" * 70)
    print(f"  Training complete. Best loss: {best_loss:.6f}")

    # ========================================================================
    # Step 5: Evaluation & Predictions
    # ========================================================================
    print(f"\n[5/5] Generating predictions...")

    model.eval()
    predictions = []

    with torch.no_grad():
        for batch in loader:
            D_R = batch["D_R"].to(DEVICE)
            D_I = batch["D_I"].to(DEVICE)
            D_P = batch["D_P"].to(DEVICE)
            D_TS = batch["D_TS"].to(DEVICE)
            atom_mask = batch["atom_mask"].to(DEVICE)
            Ea_true = batch["Ea"].to(DEVICE)
            pair_mask = make_pair_mask(atom_mask)

            D_TS_pred, Ea_pred = model(D_R, D_I, D_P, atom_mask)

            for i in range(len(batch["rxn_id"])):
                rxn = batch["rxn_id"][i]
                n = batch["n_atoms"][i]
                ea_t = Ea_true[i].item()
                ea_p = Ea_pred[i].item()

                # Per-reaction distance MAE (valid atoms only)
                mask_i = pair_mask[i, :n, :n]
                d_mae = torch.abs(D_TS_pred[i, :n, :n] - D_TS[i, :n, :n]).mean().item()

                predictions.append({
                    "rxn_id": rxn,
                    "n_atoms": n,
                    "Ea_true_kcal": round(ea_t, 4),
                    "Ea_pred_kcal": round(ea_p, 4),
                    "Ea_error_kcal": round(abs(ea_t - ea_p), 4),
                    "dist_MAE_angstrom": round(d_mae, 4),
                    "D_TS_pred": D_TS_pred[i, :n, :n].cpu().numpy().tolist(),
                    "D_TS_true": D_TS[i, :n, :n].cpu().numpy().tolist(),
                })

    # Print summary table
    print()
    print("=" * 70)
    print("  Per-Reaction Prediction Results")
    print("=" * 70)
    print(f"  {'Reaction':<14} {'Atoms':>5} {'Ea True':>10} {'Ea Pred':>10} "
          f"{'Ea Err':>8} {'Dist MAE':>10}")
    print(f"  {'':14} {'':>5} {'(kcal/mol)':>10} {'(kcal/mol)':>10} "
          f"{'(kcal)':>8} {'(A)':>10}")
    print("-" * 70)

    total_ea_err = 0
    total_d_mae = 0
    for p in sorted(predictions, key=lambda x: x["rxn_id"]):
        print(f"  {p['rxn_id']:<14} {p['n_atoms']:>5} {p['Ea_true_kcal']:>10.2f} "
              f"{p['Ea_pred_kcal']:>10.2f} {p['Ea_error_kcal']:>8.2f} "
              f"{p['dist_MAE_angstrom']:>10.4f}")
        total_ea_err += p["Ea_error_kcal"]
        total_d_mae += p["dist_MAE_angstrom"]

    n_rxn = len(predictions)
    print("-" * 70)
    print(f"  {'AVERAGE':<14} {'':>5} {'':>10} {'':>10} "
          f"{total_ea_err/n_rxn:>8.2f} {total_d_mae/n_rxn:>10.4f}")
    print("=" * 70)

    # Save predictions (without large matrices for readability)
    pred_summary = [{k: v for k, v in p.items()
                     if k not in ("D_TS_pred", "D_TS_true")}
                    for p in predictions]
    pred_path = os.path.join(config["save_dir"], "predictions_basic-1.json")
    with open(pred_path, "w") as f:
        json.dump(pred_summary, f, indent=2)
    print(f"\n  Predictions saved to: {pred_path}")

    # Save full predictions with distance matrices
    full_path = os.path.join(config["save_dir"], "predictions_basic-1_full.json")
    with open(full_path, "w") as f:
        json.dump(predictions, f, indent=2)
    print(f"  Full predictions (with distance matrices) saved to: {full_path}")

    # Save training history
    hist_path = os.path.join(config["save_dir"], "training_history_basic-1.json")
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"  Training history saved to: {hist_path}")

    model_path = os.path.join(config["save_dir"], "psi_basic1_final.pt")
    torch.save(model.state_dict(), model_path)
    print(f"  Final model saved to: {model_path}")
    print(f"\n  Done!")


# ============================================================================
# Entry Point
# ============================================================================

if __name__ == "__main__":
    train(CONFIG)
