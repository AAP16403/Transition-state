import os
import json
import numpy as np

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

    Feature groups:
      [0:10]  reaction energetics + composition (original features)
      [10:18] atomic descriptors (electronegativity, atomic number, mass)
      [18:28] bond-angle statistics for reactant, product, and their change
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

    # --- atomic descriptors ----------------------------------------------
    en = np.array([electronegativity(t) for t in types], dtype=np.float64)
    z = np.array([atomic_number(t) for t in types], dtype=np.float64)
    mass = np.array([atomic_mass(t) for t in types], dtype=np.float64)
    en_mean, en_std, en_min, en_max = _stats4(en)
    z_mean = float(z.mean()) if z.size else 0.0
    z_max = float(z.max()) if z.size else 0.0
    mass_total = float(mass.sum())
    mass_mean = float(mass.mean()) if mass.size else 0.0

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
        # reaction energetics + composition
        de_rxn, de_rxn_signed, float(diff_norms.mean()), float(diff_norms.std()),
        float(diff_norms.max()), float(n),
        float(c_count), float(h_count), float(n_count), float(o_count),
        # atomic descriptors
        en_mean, en_std, en_min, en_max, z_mean, z_max, mass_total, mass_mean,
        # bond-angle statistics
        aR_mean, aR_std, aR_min, aR_max,
        aP_mean, aP_std, aP_min, aP_max,
        ang_change_mean, ang_change_max,
    ], dtype=np.float32)
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

def kabsch_align_reactant(c_R, c_P, n):
    c_R_aligned = c_R.copy()
    c_R_aligned[:n] = kabsch(c_R[:n], c_P[:n])
    return c_R_aligned

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
