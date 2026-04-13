"""
PSI Full Pipeline: From Tarball to Transition State Prediction
==============================================================
Self-contained script implementing the 5-step PSI architecture with 
Data-Driven Smart Initialization.
"""

import os
import sys
import json
import tarfile
import math
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
    "tar_path": r"d:\Transition state\b97d3.tar.gz",
    "dataset_json": r"d:\Transition state\extracted_dataset.json",
    "save_dir": r"d:\Transition state",
    "extraction_limit": 50,    # Number of log files to extract
    "max_atoms": 17,           # Standard molecule size padding
    "n_gaussians": 32,         # K basis functions
    "gauss_start": 0.5,
    "gauss_stop": 5.0,
    "gru_hidden": 128,         # 256 context vector
    "attn_heads": 4,
    "attn_layers": 2,
    "ff_dim": 512,
    "dropout": 0.1,
    "energy_weight": 10.0,     # λ loss scale
    "lr": 1e-3,
    "weight_decay": 1e-4,      # Regularization
    "epochs": 500,
    "print_every": 50,
    "hartree_to_kcal": 627.509,
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    if os.path.exists(config["dataset_json"]):
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

class ReactionDataset(Dataset):
    def __init__(self, config):
        self.config = config
        with open(config["dataset_json"], "r") as f:
            raw_data = json.load(f)

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
        self.atom_types_map = {} # New lookup map
        for rxn_id, roles in sorted(reactions.items()):
            if "r" in roles and "p" in roles and "ts" in roles:
                # Basic info
                r_e = roles["r"]; p_e = roles["p"]; ts_e = roles["ts"]
                n = len(ts_e["atoms"])
                if n > config["max_atoms"]: continue # Skip if too many atoms

                # Coords & Padding
                def get_coords(e):
                    c = np.zeros((config["max_atoms"], 3), dtype=np.float32)
                    for i, a in enumerate(e["atoms"]):
                        c[i] = [a["x"], a["y"], a["z"]]
                    return c
                
                c_R = get_coords(r_e); c_P = get_coords(p_e); c_TS = get_coords(ts_e)
                c_I = (c_R + c_P) / 2.0  # Step 1: Interpolation

                # Distances
                D_R = compute_distance_matrix(c_R)
                D_P = compute_distance_matrix(c_P)
                D_I = compute_distance_matrix(c_I)
                D_TS = compute_distance_matrix(c_TS)

                # Mask
                mask = np.zeros(config["max_atoms"], dtype=np.float32)
                mask[:n] = 1.0

                # Energy: Ea = E_TS - max(E_R, E_P)
                ea = (ts_e["energy"] - max(r_e["energy"], p_e["energy"])) * config["hartree_to_kcal"]

                self.atom_types_map[rxn_id] = [a["atom"] for a in ts_e["atoms"]]
                self.samples.append({
                    "rxn_id": rxn_id, "n_atoms": n,
                    "D_R": torch.from_numpy(D_R), "D_I": torch.from_numpy(D_I), "D_P": torch.from_numpy(D_P),
                    "D_TS": torch.from_numpy(D_TS), "mask": torch.from_numpy(mask),
                    "Ea": torch.tensor(ea, dtype=torch.float32)
                })
        print(f"Loaded {len(self.samples)} complete reaction triplets.")

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx): return self.samples[idx]

def calculate_dataset_stats(samples):
    """Compute mean Ea and average geometry scaling factor (alpha)."""
    eas = [s["Ea"].item() for s in samples]
    mean_ea = sum(eas) / len(eas)
    
    alphas = []
    for s in samples:
        n = s["n_atoms"]
        di = s["D_I"][:n, :n].numpy()
        dts = s["D_TS"][:n, :n].numpy()
        mask = di > 0.1 # Ignore diagonal
        alphas.extend((dts[mask] / di[mask]).flatten())
    mean_alpha = sum(alphas) / len(alphas)
    
    print(f"Dataset Statistics:")
    print(f"  Mean Activation Energy: {mean_ea:.4f} kcal/mol")
    print(f"  Mean Geometry Scaling:  {mean_alpha:.4f}")
    return {"mean_ea": mean_ea, "mean_alpha": mean_alpha}

# ============================================================================
# 3. Gaussian Embedding (Step 2)
# ============================================================================

class GaussianEmbedding(nn.Module):
    def __init__(self, n_gaussians=32, start=0.5, stop=5.0):
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

class PSICore(nn.Module):
    def __init__(self, config):
        super().__init__()
        N = config["max_atoms"]; K = config["n_gaussians"]
        d_model = config["gru_hidden"] * 2
        
        self.gaussian = GaussianEmbedding(K, config["gauss_start"], config["gauss_stop"])
        
        # Step 3: Bi-GRU Temporal Encoder
        self.gru = nn.GRU(input_size=N*K, hidden_size=config["gru_hidden"], 
                          num_layers=1, batch_first=True, bidirectional=True)
        
        # Step 4: Transformer Self-Attention
        t_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=config["attn_heads"], 
                                             dim_feedforward=config["ff_dim"], dropout=config["dropout"], 
                                             batch_first=True, activation="gelu")
        self.transformer = nn.TransformerEncoder(t_layer, num_layers=config["attn_layers"])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, D_R, D_I, D_P, mask):
        B, N, _ = D_R.shape
        # Step 2: Gaussian Expansion & Reshape
        emb_R = self.gaussian(D_R).view(B, N, -1)
        emb_I = self.gaussian(D_I).view(B, N, -1)
        emb_P = self.gaussian(D_P).view(B, N, -1)

        # Step 3: Sequence [R, I, P] through GRU
        seq = torch.stack([emb_R, emb_I, emb_P], dim=2).view(B*N, 3, -1)
        out, _ = self.gru(seq)
        context = out[:, 1, :].view(B, N, -1) # Extract Frame I context

        # Step 4: Transfomer Shape Logic
        pad_mask = (mask == 0)
        final_features = self.transformer(context, src_key_padding_mask=pad_mask)
        return self.norm(final_features)

# ============================================================================
# 6. Output Heads (Step 5)
# ============================================================================

class GeometryHead(nn.Module):
    def __init__(self, d_model, mean_alpha):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_model*2, 256), nn.GELU(), nn.Linear(256, 1))
        
        # Smart Init: set bias so softplus(bias) = mean_alpha
        target_b = math.log(math.exp(mean_alpha) - 1.0)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.constant_(self.net[-1].bias, target_b)

    def forward(self, features, D_I):
        B, N, D = features.shape
        fi = features.unsqueeze(2).expand(B, N, N, D)
        fj = features.unsqueeze(1).expand(B, N, N, D)
        pair = torch.cat([fi, fj], dim=-1)
        # Use softplus(x) to ensure alpha > 0
        alpha = F.softplus(self.net(pair).squeeze(-1))
        D_TS_pred = (alpha * D_I)
        return (D_TS_pred + D_TS_pred.transpose(1, 2)) / 2.0

class EnergyHead(nn.Module):
    def __init__(self, d_model, mean_ea):
        super().__init__()
        # Stability: Added LayerNorm at the interface
        self.ln = nn.LayerNorm(d_model)
        self.net = nn.Sequential(
            nn.Linear(d_model, 128), nn.GELU(), 
            nn.Linear(128, 64), nn.GELU(), 
            nn.Linear(64, 1)
        )
        
        # Smart Init: set final bias to the global Ea average structure
        nn.init.zeros_(self.net[-1].weight)
        nn.init.constant_(self.net[-1].bias, mean_ea)

    def forward(self, features, mask):
        # Stability: Prep features
        features = self.ln(features)
        
        # Masked mean pool
        m = mask.unsqueeze(-1)
        pooled = (features * m).sum(dim=1) / m.sum(dim=1).clamp(min=1)
        return self.net(pooled).squeeze(-1)

# ============================================================================
# 7. Full PSI Pipeline Integration
# ============================================================================

class PSI(nn.Module):
    def __init__(self, config, stats):
        super().__init__()
        d_model = config["gru_hidden"] * 2
        self.core = PSICore(config)
        self.geom_head = GeometryHead(d_model, stats["mean_alpha"])
        self.ener_head = EnergyHead(d_model, stats["mean_ea"])

    def forward(self, D_R, D_I, D_P, mask):
        f = self.core(D_R, D_I, D_P, mask)
        return self.geom_head(f, D_I), self.ener_head(f, mask)

def train_pipeline(config):
    print("="*70); print(" PSI FULL PIPELINE (DATA-DRIVEN INIT) "); print("="*70)
    
    # 1. & 2. Data
    extract_raw_data(config)
    dataset = ReactionDataset(config)
    if len(dataset) == 0:
        print("Error: No complete reaction triplets found."); return
        
    # GLOBAL STRUCTURE ANALYSIS
    stats = calculate_dataset_stats(dataset.samples)
    
    loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
    
    # 3-6. Model with Smart Initialization
    model = PSI(config, stats).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=50, factor=0.5)
    
    # 8. Training Loop
    print(f"\nTraining for {config['epochs']} epochs (N={len(dataset)})...")
    model.train()
    for epoch in range(1, config["epochs"] + 1):
        for batch in loader:
            DR, DI, DP = batch["D_R"].to(DEVICE), batch["D_I"].to(DEVICE), batch["D_P"].to(DEVICE)
            DTS, mask, true_ea = batch["D_TS"].to(DEVICE), batch["mask"].to(DEVICE), batch["Ea"].to(DEVICE)
            
            p_DTS, p_ea = model(DR, DI, DP, mask)
            
            # Loss with masking
            m2d = mask.unsqueeze(-1) * mask.unsqueeze(-2)
            l_geom = ((p_DTS - DTS)**2 * m2d).sum() / m2d.sum().clamp(min=1)
            l_ener = F.mse_loss(p_ea, true_ea)
            loss = l_geom + config["energy_weight"] * l_ener
            
            optimizer.zero_grad(); loss.backward(); optimizer.step()
        
        scheduler.step(loss.item())
        if epoch % config["print_every"] == 0 or epoch == 1:
            print(f"Epoch {epoch:3d} | Loss: {loss.item():10.4f} | Geom MSE: {l_geom.item():8.5f} | Energy MSE: {l_ener.item():8.5f}")

    # 9. Evaluation
    print("\n" + "="*70); print(" EVALUATION RESULTS "); print("="*70)
    model.eval()
    results = []
    with torch.no_grad():
        for batch in loader:
            DR, DI, DP = batch["D_R"].to(DEVICE), batch["D_I"].to(DEVICE), batch["D_P"].to(DEVICE)
            DTS, mask, true_ea = batch["D_TS"].to(DEVICE), batch["mask"].to(DEVICE), batch["Ea"].to(DEVICE)
            p_DTS, p_ea = model(DR, DI, DP, mask)
            
            for i in range(len(batch["rxn_id"])):
                rxn_id = batch["rxn_id"][i]
                n = int(mask[i].sum().item())
                di = DI[i, :n, :n].cpu().numpy()
                dp = p_DTS[i, :n, :n].cpu().numpy()
                dt = DTS[i, :n, :n].cpu().numpy()
                
                d_mae = np.abs(dp - dt).mean().item()
                e_err = abs(p_ea[i].item() - true_ea[i].item())
                
                # Fetch atom types from dataset map
                atom_types = loader.dataset.atom_types_map.get(rxn_id, [])
                
                results.append({
                    "rxn_id": rxn_id, 
                    "Ea_true": true_ea[i].item(), "Ea_pred": p_ea[i].item(),
                    "Ea_error": e_err, "dist_MAE": d_mae,
                    "n_atoms": n,
                    "atom_types": atom_types, 
                    "D_I": di.tolist(), "D_pred": dp.tolist(), "D_true": dt.tolist()
                })

    print(f"{'Reaction':<15} {'Ea True':>10} {'Ea Pred':>10} {'Ea Err':>10} {'Dist MAE':>10}")
    for r in sorted(results, key=lambda x: x["rxn_id"]):
        print(f"{r['rxn_id']:<15} {r['Ea_true']:10.2f} {r['Ea_pred']:10.2f} {r['Ea_error']:10.2f} {r['dist_MAE']:10.4f}")
    
    # Save predictions
    output_path = os.path.join(config["save_dir"], "detailed_analysis.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    torch.save(model.state_dict(), os.path.join(config["save_dir"], "psi_final.pt"))
    print(f"\nModel and predictions saved to {config['save_dir']}")

if __name__ == "__main__":
    train_pipeline(CONFIG)
