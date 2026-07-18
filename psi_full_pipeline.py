import os
import sys
import json
import math
import time
import pickle
import argparse
import traceback
from datetime import datetime
import numpy as np
import torch
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

# =============================================================================
# Reaction-center angle features (targeted at forming/breaking bonds)
# =============================================================================

def _reaction_center_atoms(D_R, D_P, atom_types, n, bond_scale=1.45):
    """Identify atoms involved in forming/breaking bonds.

    Returns the set of reaction-center atom indices, plus the sets of
    formed and broken bond pairs (i, j) with i < j.
    """
    bonds_R = bond_set_from_distance_matrix(D_R, atom_types, n, bond_scale)
    bonds_P = bond_set_from_distance_matrix(D_P, atom_types, n, bond_scale)
    formed = set(bonds_P) - set(bonds_R)
    broken = set(bonds_R) - set(bonds_P)
    rc_atoms = set()
    for (i, j) in formed | broken:
        rc_atoms.add(i)
        rc_atoms.add(j)
    return rc_atoms, formed, broken


def _rc_bond_angles(coords, atom_types, n, rc_atoms, bond_scale=1.45):
    """Bond angles only at reaction-center atoms (central atom in rc_atoms)."""
    all_angles = bond_angles_from_coords(coords, atom_types, n, bond_scale)
    rc_angles = {k: v for k, v in all_angles.items() if k[1] in rc_atoms}
    return rc_angles


def _best_dihedral_across_bond(coords, atom_types, n, bond_pair, adjacency):
    """Compute a single representative dihedral i-a-b-j across bond (a,b).

    Picks the neighbor pair (i of a, j of b) with the heaviest atoms
    to get the most chemically meaningful dihedral.

    Returns (cos_phi, sin_phi) as a periodicity-safe encoding.
    """
    a, b = bond_pair
    nbrs_a = [x for x in adjacency[a] if x != b and x < n]
    nbrs_b = [x for x in adjacency[b] if x != a and x < n]
    if not nbrs_a or not nbrs_b:
        return 0.0, 0.0  # degenerate — no neighbors to define a dihedral

    # Pick heaviest neighbor on each side for chemical relevance
    i = max(nbrs_a, key=lambda x: atomic_mass(atom_types[x]))
    j = max(nbrs_b, key=lambda x: atomic_mass(atom_types[x]))

    # Dihedral i-a-b-j
    b1 = coords[a] - coords[i]
    b2 = coords[b] - coords[a]
    b3 = coords[j] - coords[b]
    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)
    n1_norm = np.linalg.norm(n1)
    n2_norm = np.linalg.norm(n2)
    if n1_norm < 1e-9 or n2_norm < 1e-9:
        return 1.0, 0.0  # degenerate → 0 degrees
    n1 /= n1_norm
    n2 /= n2_norm
    b2_hat = b2 / max(np.linalg.norm(b2), 1e-9)
    cos_phi = float(np.clip(np.dot(n1, n2), -1.0, 1.0))
    sin_phi = float(np.clip(np.dot(np.cross(n1, n2), b2_hat), -1.0, 1.0))
    return cos_phi, sin_phi


def _pyramidalization_angle(coords, center, neighbors):
    """Out-of-plane angle for a trigonal center (degrees).

    For a center atom with ≥3 neighbors, measures average deviation from
    planarity. Captures sp2↔sp3 distortion at the reactive atom.
    """
    if len(neighbors) < 3:
        return 0.0
    # Take first 3 neighbors (sorted by index for reproducibility)
    nbrs = sorted(neighbors)[:3]
    v1 = coords[nbrs[0]] - coords[center]
    v2 = coords[nbrs[1]] - coords[center]
    v3 = coords[nbrs[2]] - coords[center]
    # Normal to the plane of the 3 neighbor vectors
    plane_normal = np.cross(v1 - v2, v1 - v3)
    pn_norm = np.linalg.norm(plane_normal)
    if pn_norm < 1e-9:
        return 0.0
    plane_normal /= pn_norm
    # Average deviation from planarity
    angles = []
    for v in [v1, v2, v3]:
        v_norm = np.linalg.norm(v)
        if v_norm < 1e-9:
            continue
        sin_angle = abs(np.dot(v / v_norm, plane_normal))
        angles.append(float(np.degrees(np.arcsin(np.clip(sin_angle, 0.0, 1.0)))))
    return float(np.mean(angles)) if angles else 0.0


def _rc_angle_features(cR, cP, atom_types, n, bond_scale=1.45):
    """Compute 8 reaction-center angle features from R and P geometries.

    These target the exact gap in the existing feature set: no per-atom or
    per-triplet angle information at the reactive atoms. All angles use
    cos/sin encoding to handle periodicity.

    Features (8D):
      [0] rc_angle_R_mean    — mean cos(bond angle) at reacting atoms in R
      [1] rc_angle_P_mean    — mean cos(bond angle) at reacting atoms in P
      [2] rc_angle_change_max — max |Δangle| at any reacting-atom center, R→P
      [3] rc_dihedral_forming_cos — cos(dihedral) across forming bond region
      [4] rc_dihedral_forming_sin — sin(dihedral) across forming bond region
      [5] rc_dihedral_breaking_cos — cos(dihedral) across breaking bond region
      [6] rc_dihedral_breaking_sin — sin(dihedral) across breaking bond region
      [7] rc_pyramidalization  — avg out-of-plane angle at reacting atoms (deg)
    """
    # Need distance matrices to identify forming/breaking bonds
    D_R = np.zeros((n, n), dtype=np.float64)
    D_P = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            dr = np.linalg.norm(cR[i] - cR[j])
            dp = np.linalg.norm(cP[i] - cP[j])
            D_R[i, j] = D_R[j, i] = dr
            D_P[i, j] = D_P[j, i] = dp

    rc_atoms, formed, broken = _reaction_center_atoms(
        D_R, D_P, atom_types, n, bond_scale
    )

    # --- Feature 0-1: Mean cos(bond angle) at reacting atoms in R and P ---
    rc_ang_R = _rc_bond_angles(cR, atom_types, n, rc_atoms, bond_scale)
    rc_ang_P = _rc_bond_angles(cP, atom_types, n, rc_atoms, bond_scale)
    if rc_ang_R:
        rc_angle_R_mean = float(np.mean([np.cos(np.radians(a)) for a in rc_ang_R.values()]))
    else:
        rc_angle_R_mean = 0.0
    if rc_ang_P:
        rc_angle_P_mean = float(np.mean([np.cos(np.radians(a)) for a in rc_ang_P.values()]))
    else:
        rc_angle_P_mean = 0.0

    # --- Feature 2: Max angle change at any reacting-atom center ----------
    common_rc = set(rc_ang_R) & set(rc_ang_P)
    if common_rc:
        rc_changes = np.array([abs(rc_ang_R[t] - rc_ang_P[t]) for t in common_rc],
                              dtype=np.float64)
        rc_angle_change_max = float(rc_changes.max())
    else:
        rc_angle_change_max = 0.0

    # --- Features 3-6: Dihedrals across forming/breaking bonds ------------
    # Use the *product* adjacency for formed bonds, *reactant* for broken
    adj_R = bond_adjacency_from_coords(cR, atom_types, n, bond_scale)
    adj_P = bond_adjacency_from_coords(cP, atom_types, n, bond_scale)

    # Forming bonds: average dihedral in P geometry (where bond exists)
    form_cos_list, form_sin_list = [], []
    for (a, b) in formed:
        c, s = _best_dihedral_across_bond(cP, atom_types, n, (a, b), adj_P)
        form_cos_list.append(c)
        form_sin_list.append(s)
    rc_dih_form_cos = float(np.mean(form_cos_list)) if form_cos_list else 0.0
    rc_dih_form_sin = float(np.mean(form_sin_list)) if form_sin_list else 0.0

    # Breaking bonds: average dihedral in R geometry (where bond exists)
    break_cos_list, break_sin_list = [], []
    for (a, b) in broken:
        c, s = _best_dihedral_across_bond(cR, atom_types, n, (a, b), adj_R)
        break_cos_list.append(c)
        break_sin_list.append(s)
    rc_dih_break_cos = float(np.mean(break_cos_list)) if break_cos_list else 0.0
    rc_dih_break_sin = float(np.mean(break_sin_list)) if break_sin_list else 0.0

    # --- Feature 7: Pyramidalization at reacting atoms --------------------
    # Average across R and P to capture the sp2↔sp3 distortion trend
    pyram_vals = []
    for atom_idx in rc_atoms:
        nbrs_R = [x for x in adj_R[atom_idx] if x < n]
        nbrs_P = [x for x in adj_P[atom_idx] if x < n]
        pR = _pyramidalization_angle(cR, atom_idx, nbrs_R)
        pP = _pyramidalization_angle(cP, atom_idx, nbrs_P)
        pyram_vals.append((pR + pP) / 2.0)
    rc_pyramidalization = float(np.mean(pyram_vals)) if pyram_vals else 0.0

    return np.array([
        rc_angle_R_mean, rc_angle_P_mean, rc_angle_change_max,
        rc_dih_form_cos, rc_dih_form_sin,
        rc_dih_break_cos, rc_dih_break_sin,
        rc_pyramidalization,
    ], dtype=np.float32)


def build_energy_features(atom_types, n, c_R_aligned, c_P, e_r, e_p, bond_scale=1.45):
    """Construct the energy-head input feature vector from reactant + product only.

    Shared by training (build_reaction_samples) and inference
    (predict_transition_state) so the two can never drift out of sync. All
    inputs are available before the TS is known. Returns float32 of fixed length.

    Feature groups (28D total):
      [0:10]  reaction energetics + composition
      [10:20] bond-angle statistics for reactant, product, and their change
      [20:28] reaction-center angle features (targeted at forming/breaking bonds)

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

    # --- reaction-center angle features (8D) -----------------------------
    rc_feats = _rc_angle_features(cR, cP, types, n, bond_scale)

    feats = np.array([
        # reaction energetics + composition (10 features)
        de_rxn, de_rxn_signed, float(diff_norms.mean()), float(diff_norms.std()),
        float(diff_norms.max()), float(n),
        float(c_count), float(h_count), float(n_count), float(o_count),
        # bond-angle statistics (10 features)
        aR_mean, aR_std, aR_min, aR_max,
        aP_mean, aP_std, aP_min, aP_max,
        ang_change_mean, ang_change_max,
        # reaction-center angle features (8 features)
        rc_feats[0], rc_feats[1], rc_feats[2],
        rc_feats[3], rc_feats[4],
        rc_feats[5], rc_feats[6],
        rc_feats[7],
    ], dtype=np.float32)
    return feats

ATOM_PHYS_DIM = 3  # electronegativity, atomic number, mass
ENERGY_FEAT_DIM = 28  # reaction energetics + composition + bond-angle statistics + RC angles

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

def mds_aligned(D, reference_coords=None, dim=3):
    """Recover one global coordinate set from a full distance matrix.

    A single MDS embedding preserves inter-fragment distances. When a reference
    frame is available, only one global rigid alignment is applied.
    """
    X = mds(D, dim=dim)
    if reference_coords is not None:
        ref = np.asarray(reference_coords, dtype=np.float64)
        if len(X) >= 2:
            X = kabsch(X, ref)
        elif len(X) == 1:
            X[0] = ref[0]
    elif len(X) > 0:
        X = X - X.mean(axis=0)
    return X.astype(np.float32)

STERIC_FLOOR_FRAC = 0.75

_COVALENT_RADII_CACHE = {}


def covalent_radius_lookup(device=None):
    """Covalent radius per atom-vocab id, as a tensor for the training loss.

    Index matches build_atom_vocab()'s id assignment (sorted ATOMIC_NUMBER keys,
    ids 1..V; id 0 = padding). Lets the loss compute a per-pair steric floor
    from batched atom_ids without carrying atom-type strings onto the GPU.
    """
    key = str(device)
    cached = _COVALENT_RADII_CACHE.get(key)
    if cached is not None:
        return cached
    sorted_atoms = sorted(ATOMIC_NUMBER.keys())
    radii = [0.0] + [COVALENT_RADII.get(a, 0.76) for a in sorted_atoms]
    tensor = torch.tensor(radii, dtype=torch.float32, device=device)
    _COVALENT_RADII_CACHE[key] = tensor
    return tensor


def reaction_center_atom_mask(D_R, D_P, atom_ids, mask, bond_scale=1.45):
    """Per-atom [B, N] bool mask of atoms whose bonding changes reactant->product.

    A pair (i, j) is bonded when its distance is below bond_scale * (r_i + r_j)
    (covalent radii). Atoms belonging to any bond that forms or breaks (bonded in
    exactly one of R/P) are reaction-center atoms. Derived from D_R/D_P/atom_ids
    only, so training, inference, and the geometry loss share one definition with
    no extra data plumbing.
    """
    with torch.no_grad():
        N = mask.shape[1]
        radii = covalent_radius_lookup(atom_ids.device)          # [V+1]
        r = radii[atom_ids]                                      # [B, N]
        thr = bond_scale * (r.unsqueeze(2) + r.unsqueeze(1))     # [B, N, N]
        eye = torch.eye(N, device=mask.device).unsqueeze(0)
        pair_valid = (mask.unsqueeze(1) * mask.unsqueeze(2)) * (1.0 - eye) > 0
        bond_R = (D_R < thr) & pair_valid
        bond_P = (D_P < thr) & pair_valid
        changed = bond_R ^ bond_P                                # formed or broken
        return changed.any(dim=2) & (mask > 0)                   # [B, N]

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

def spectator_pairs(D_R, D_P, n, threshold=0.15):
    spectator = []
    for i in range(n):
        for j in range(i + 1, n):
            if abs(D_R[i, j] - D_P[i, j]) <= threshold:
                spectator.append((i, j))
    return spectator

RISK_BOND_TYPES = {
    tuple(sorted(pair))
    for pair in [("N", "N"), ("N", "O"), ("O", "O"), ("C", "N"), ("H", "N")]
}

def bond_set_from_distance_matrix(D, atom_types, n, bond_scale=1.45):
    bonds = {}
    for i in range(n):
        for j in range(i + 1, n):
            cutoff = bond_scale * (covalent_radius(atom_types[i]) + covalent_radius(atom_types[j]))
            if D[i, j] <= cutoff:
                bonds[(i, j)] = tuple(sorted((atom_types[i], atom_types[j])))
    return bonds

def reaction_risk_features(D_R, D_P, atom_types, n, max_atoms, bond_scale=1.45, active_threshold=0.15):
    """Risk flags for reaction classes that showed higher validation Ea error."""
    bonds_R = bond_set_from_distance_matrix(D_R, atom_types, n, bond_scale)
    bonds_P = bond_set_from_distance_matrix(D_P, atom_types, n, bond_scale)
    formed = set(bonds_P) - set(bonds_R)
    broken = set(bonds_R) - set(bonds_P)

    risk_pair_mask = np.zeros((max_atoms, max_atoms), dtype=np.float32)
    risky_bond_types = set()
    for i in range(n):
        for j in range(i + 1, n):
            bond_type = tuple(sorted((atom_types[i], atom_types[j])))
            is_active = abs(D_R[i, j] - D_P[i, j]) > active_threshold
            is_formed_or_broken = (i, j) in formed or (i, j) in broken
            is_risky_type = bond_type in RISK_BOND_TYPES
            if is_active:
                if is_risky_type:
                    risky_bond_types.add(bond_type)
            if is_formed_or_broken and is_risky_type:
                risky_bond_types.add(bond_type)
            if is_active or is_formed_or_broken:
                risk_pair_mask[i, j] = risk_pair_mask[j, i] = 1.0

    formed_n = len(formed)
    broken_n = len(broken)
    changed_n = formed_n + broken_n
    complexity_flag = float((formed_n + broken_n) >= 4 or broken_n >= 3)
    risky_chem_flag = float(len(risky_bond_types) > 0)
    risk_score = complexity_flag + risky_chem_flag
    complexity_margin_penalty = max(0.0, changed_n - 3.0) ** 2 + max(0.0, 1.0 - changed_n) ** 2
    complexity_sigmoid_penalty = 1.0 / (1.0 + math.exp(-1.25 * (changed_n - 4.0)))
    risk_penalty = complexity_margin_penalty + 0.5 * risky_chem_flag
    return {
        "formed_bonds": formed_n,
        "broken_bonds": broken_n,
        "changed_bonds": changed_n,
        "complexity_flag": complexity_flag,
        "risky_chem_flag": risky_chem_flag,
        "risk_score": risk_score,
        "complexity_margin_penalty": complexity_margin_penalty,
        "complexity_sigmoid_penalty": complexity_sigmoid_penalty,
        "risk_penalty": risk_penalty,
        "risk_pair_mask": risk_pair_mask,
        "risky_bond_types": sorted("-".join(t) for t in risky_bond_types),
    }

def continuous_risk_penalty(formed_n, broken_n, risky_chem_flag, mode="margin", safe_min=1.0,
                            safe_max=3.0, sigmoid_center=4.0, sigmoid_k=1.25):
    """Smooth reaction-complexity penalty used for sample-level loss weighting."""
    changed = float(formed_n + broken_n)
    if mode == "binary":
        complexity = float(changed >= safe_max + 1.0 or broken_n >= safe_max)
    elif mode == "sigmoid":
        complexity = 1.0 / (1.0 + math.exp(-sigmoid_k * (changed - sigmoid_center)))
    elif mode == "margin":
        complexity = max(0.0, changed - safe_max) ** 2 + max(0.0, safe_min - changed) ** 2
    else:
        raise ValueError(f"Unknown risk_penalty_mode '{mode}'. Use 'binary', 'margin', or 'sigmoid'.")
    return float(complexity + 0.5 * float(risky_chem_flag > 0.0))

def ensure_sample_risk_features(sample, config, atom_types=None):
    """Populate risk masks/flags for samples created before these fields existed."""
    max_atoms = config["max_atoms"]
    expected_shape = (max_atoms, max_atoms)
    risk_mask = sample.get("risk_pair_mask")
    masks_ready = (
        risk_mask is not None
        and tuple(risk_mask.shape) == expected_shape
    )
    flags_ready = all(
        key in sample
        for key in ("risk_score", "complexity_flag", "risky_chem_flag", "risk_penalty")
    )
    if masks_ready and flags_ready:
        return

    atom_types = atom_types or sample.get("atom_types")
    if atom_types is None:
        rxn_id = sample.get("rxn_id", "<unknown>")
        raise KeyError(
            f"Sample {rxn_id} is missing risk fields and atom types, "
            "so risk_pair_mask cannot be rebuilt."
        )

    n = sample["n_atoms"]
    D_R = compute_distance_matrix(sample["c_R"])
    D_P = compute_distance_matrix(sample["c_P"])
    risk = reaction_risk_features(
        D_R,
        D_P,
        atom_types,
        n,
        max_atoms,
        config["fragment_bond_scale"],
        config["spectator_threshold"],
    )

    sample["risk_pair_mask"] = torch.from_numpy(risk["risk_pair_mask"])
    sample["risk_score"] = risk["risk_score"]
    sample["complexity_flag"] = risk["complexity_flag"]
    sample["risky_chem_flag"] = risk["risky_chem_flag"]
    sample.setdefault("formed_bonds", risk["formed_bonds"])
    sample.setdefault("broken_bonds", risk["broken_bonds"])
    sample.setdefault("changed_bonds", risk["changed_bonds"])
    sample["complexity_margin_penalty"] = risk["complexity_margin_penalty"]
    sample["complexity_sigmoid_penalty"] = risk["complexity_sigmoid_penalty"]
    sample["risk_penalty"] = risk["risk_penalty"]
    sample.setdefault("risky_bond_types", risk["risky_bond_types"])

def apply_spectator_constraints(pred_dist, D_R, D_P, n, threshold=0.15, tol=0.05, pair_mask=None):
    for (i, j) in spectator_pairs(D_R, D_P, n, threshold):
        if pair_mask is not None and pair_mask[i, j] <= 0:
            continue
        d_ref = (D_R[i, j] + D_P[i, j]) / 2.0
        lo = d_ref * (1.0 - tol)
        hi = d_ref * (1.0 + tol)
        clamped = float(np.clip(pred_dist[i, j], lo, hi))
        pred_dist[i, j] = clamped
        pred_dist[j, i] = clamped
    return pred_dist

def enforce_triangle_inequality(D, tol=0.05):
    D = D.copy()
    n = D.shape[0]
    for k in range(n):
        for i in range(n):
            for j in range(n):
                shortcut = D[i, k] + D[k, j]
                if D[i, j] - shortcut > tol:
                    D[i, j] = D[j, i] = shortcut
    return D

def validate_ts_geometry(pred_dist, atom_types, n):
    issues = []
    for i in range(n):
        for j in range(i + 1, n):
            r_i = covalent_radius(atom_types[i])
            r_j = covalent_radius(atom_types[j])
            min_d = STERIC_FLOOR_FRAC * (r_i + r_j)
            if pred_dist[i, j] < min_d:
                issues.append(f"  STERIC   {atom_types[i]}{i}-{atom_types[j]}{j}: {pred_dist[i,j]:.3f} Å < floor {min_d:.3f} Å")
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
    "dataset": "b97d3",             # b97d3 or wb97xd3
    "tar_path": "b97d3.tar.gz",
    "dataset_json": "extracted_b97d3.json",
    "save_dir": ".",
    # ~3 logs (r/p/ts) per reaction, minus those dropped by the max_atoms and
    # negative-Ea filters. Stage-A scale-up targets roughly 20k usable triplets.
    # extraction limit. Set to 1,000,000 to process the entire dataset.
    "extraction_limit": 1000000,
    "target_reactions": 40000,      # RGD1 subset: ~2.5x the old b97d3 set, fast to build/train
    "force_extract": False,
    "max_atoms": 30,
    "n_gaussians": 32,
    "gauss_start": 0.4,
    "gauss_stop": 6.0,
    "atom_embed_dim": 32,
    "gru_hidden": 128,
    "gru_layers": 2,
    "gru_dropout": 0.1,
    "attn_heads": 8,
    "attn_layers": 3,
    "ff_dim": 512,
    "dropout": 0.1,
    "delta_clamp": 3.0,
    # --- EGNN coordinate refiner -----------------------------------------
    # After the geometry head predicts a TS distance matrix, we embed it to 3D
    # (differentiable MDS) and refine the coordinates with an E(n)-equivariant
    # GNN that consumes a per-atom chemical-property vector + the TS coords.
    "egnn_layers": 8,
    "egnn_hidden": 256,
    "egnn_coord_clamp": 3.5,  # max per-step coordinate displacement (Angstrom)
    "geom_coarse_weight": 0.5,  # weight on the pre-EGNN (coarse) distance aux loss
    # --- Learned activation-energy (Ea) head -----------------------------
    # A small head consumes the EGNN's refined per-atom features (h_ts) + the
    # signed reaction energy and regresses Ea. The features are ALWAYS detached
    # before the head, so the Ea gradient never reshapes the EGNN: geometry is
    # trained purely by the geometry loss, and the Ea head reads the settled TS
    # as a fixed input. PhysicsEa (Marcus/Hammond/OLS) is kept as a side-by-side
    # baseline.
    "ea_loss_weight": 0.5,          # de-emphasized: Ea head already converged (~5 kcal); free gradient budget for TS geometry
    "ea_loss_start_epoch": 1,       # train Ea head from the first epoch
    "ea_select_weight": 0.5,        # Ea contribution to checkpoint selection
    "ea_head_dropout": 0.15,        # dropout inside the Ea head MLP
    "ea_head_lr": 3e-4,             # LR for the Ea head
    "ea_head_weight_decay": 1e-3,
    "lr": 1.5e-4,
    "weight_decay": 1e-3,
    "warmup_epochs": 40,
    "grad_clip": 1.0,
    "batch_size": 32,
    "num_workers": 2,
    "pin_memory": True,
    "device": "cuda",
    "require_cuda": True,
    "amp": True,
    "epochs": 800,
    # --- Two-stage decoupled training -----------------------------------
    # Stage 1: train the geometry backbone alone (--geom-only). Stage 2: load
    # that frozen backbone and train only the Ea head on its deterministic
    # predicted TS (--ea-only --backbone-ckpt PATH). Default (both False) is the
    # original joint training.
    "geom_only": False,
    "ea_only": False,
    "backbone_ckpt": None,
    # --- Geometry loss: reaction-role reweighting (hinge fix, SWARM_FAILURE
    # sections 5-6) layered on the inverse-distance weighting. Boost the
    # active<->spectator cross-distances (global orientation) and reactive
    # pairs; damp the static spectator backbone.
    "geom_hinge_cross_weight": 3.0,          # active<->spectator cross pairs
    "geom_active_pair_weight": 2.0,          # active<->active pairs
    "geom_spectator_spectator_weight": 0.25, # static backbone pairs
    # --- Throughput: MDS seed eigh location/precision --------------------
    # False = original CPU-float64 path. True keeps the embedding on the GPU,
    # removing the per-forward device sync + host<->device transfers (the real
    # bottleneck) at the cost of a tiny-matrix GPU eigh. "float32" is safe here
    # because the seed is detached and refined by the EGNN.
    # Set for RTX 4050 (Ada consumer): on-GPU to kill the sync, float32 because
    # consumer cards have ~1/64 FP64 throughput (float64 GPU eigh would be slow).
    "mds_on_gpu": True,
    "mds_dtype": "float32",
    "swa_enabled": True,
    "swa_start": 450,
    "print_every": 25,
    "val_split": 0.15,
    "split_seed": 42,
    "split_strategy": "stratified",
    "split_bins": 5,
    "patience": 120,
    "spectator_threshold": 0.15,
    "spectator_tol": 0.05,
    "risk_penalty_mode": "margin",
    "risk_safe_min": 1.0,
    "risk_safe_max": 3.0,
    "risk_sigmoid_center": 4.0,
    "risk_sigmoid_k": 1.25,
    "risk_weight_alpha": 0.5,
    "risk_weight_max": 3.0,
    "risk_ea_loss_weight": 0.5,      # extra Ea objective weight on high-risk reaction classes
    "risk_geom_loss_weight": 0.2,    # extra geometry loss on active/formed/broken risky pairs
    "steric_loss_weight": 1.0,       # weight on the steric-floor soft penalty in the loss
    "data_dir": None,                # RGD1 data dir; None -> default local path (see loader)
    "sample_cache_path": None,       # explicit sample-cache path; None -> save_dir/samples_cache_rgd1.pkl
    "triangle_loss_weight": 0.05,
    "triangle_coarse_weight": 0.25,
    "triangle_refined_weight": 1.0,
    "triangle_tolerance": 0.02,
    "triangle_triplet_samples": 1024,
    "fragment_bond_scale": 1.45,
    "hartree_to_kcal": 627.509,
    "skip_negative_ea": True,
    # 0.0 = augmentation off. This re-enables the per-sample memoization and the
    # cached noise-free D_R/D_P (built once, persisted in the sample cache), so
    # distance matrices are never recomputed per epoch/run. Set > 0 to trade that
    # speed for coordinate-noise augmentation (matrices then rebuilt each epoch).
    "coord_noise": 0.0,
}

def resolve_device(config):
    print("WARNING: Forcing CUDA device, CPU fallback removed.")
    return torch.device("cuda")

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
        batch["risk_pair_mask"].to(device, non_blocking=True),
        batch["risk_score"].to(device, non_blocking=True),
        batch["risk_penalty"].to(device, non_blocking=True),
        batch["complexity_flag"].to(device, non_blocking=True),
        batch["risky_chem_flag"].to(device, non_blocking=True),
    )

def extract_raw_data(config):
    pass # Skipped for RGD1 dataset

def build_atom_vocab():
    sorted_atoms = sorted(ATOMIC_NUMBER.keys())
    vocab = {atom: i + 1 for i, atom in enumerate(sorted_atoms)}
    return vocab

def _sample_cache_meta(config):
    """Feature-affecting params. A change to any of these invalidates the cache."""
    return {
        "cache_version": 3,  # v3: drop reactions with empty reaction center (no changed bonds)
        "dataset": "rgd1",
        "max_atoms": config["max_atoms"],
        "fragment_bond_scale": config["fragment_bond_scale"],
        "spectator_threshold": config["spectator_threshold"],
        "skip_negative_ea": config["skip_negative_ea"],
    }


def _samples_cache_path(config):
    p = config["sample_cache_path"]
    if p:
        return p
    return os.path.join(config["save_dir"], "samples_cache_rgd1.pkl")


def build_reaction_samples(config):
    """Return built reaction samples, using an on-disk cache when possible.

    The cache stores the full built pool. If a later run requests fewer
    reactions than the cached pool holds (and feature params are unchanged),
    the first N deterministic samples are sliced out without rebuilding.
    """
    target_reactions = config["target_reactions"]
    cache_path = _samples_cache_path(config)
    meta = _sample_cache_meta(config)

    if not config["force_extract"] and os.path.exists(cache_path):
        try:
            with open(cache_path, "rb") as fh:
                cached = pickle.load(fh)
        except (pickle.UnpicklingError, EOFError, OSError, AttributeError, ValueError) as e:
            # Loud, not silent: a corrupt/unreadable cache is surfaced with a clear
            # recovery path instead of quietly triggering a slow full rebuild that
            # could also mask a real bug.
            raise RuntimeError(
                f"Sample cache {cache_path} is unreadable or corrupt ({e}). Delete it "
                "or re-run with --force-extract to rebuild it."
            ) from e
        pool = cached.get("samples", [])
        # Valid when the cached pool covers the request, OR the cache already
        # exhausted the whole dataset (can't produce more no matter the target).
        covers = len(pool) >= target_reactions or cached.get("exhausted", False)
        if cached.get("meta") == meta and covers:
            samples = pool[:target_reactions]
            atom_types_map = {s["rxn_id"]: s["atom_types"] for s in samples}
            print(f"Loaded {len(samples)} reaction samples from cache "
                  f"{cache_path} (built pool={len(pool)}"
                  f"{', dataset-exhausted' if cached.get('exhausted') else ''}).")
            return samples, cached["atom_vocab"], atom_types_map
        if cached.get("meta") != meta:
            print("Sample cache present but feature params changed; rebuilding.")
        else:
            print(f"Sample cache holds {len(pool)} < requested "
                  f"{target_reactions}; rebuilding.")

    samples, atom_vocab, atom_types_map = _build_reaction_samples_from_h5(config)
    # If we produced fewer than requested, the dataset is exhausted: record it so
    # future runs reuse this cache instead of rebuilding to chase an unreachable target.
    exhausted = len(samples) < target_reactions

    with open(cache_path, "wb") as fh:
        pickle.dump(
            {"meta": meta, "samples": samples,
             "atom_vocab": atom_vocab, "exhausted": exhausted},
            fh, protocol=pickle.HIGHEST_PROTOCOL,
        )
    print(f"Cached {len(samples)} reaction samples to {cache_path}.")

    return samples, atom_vocab, atom_types_map


def _build_reaction_samples_from_h5(config):
    import h5py
    import pandas as pd

    DATA_DIR = config["data_dir"] or "d:/Transition state/RGD1_Dataset"
    h5_path = os.path.join(DATA_DIR, "RGD1_CHNO.h5")
    csv_path = os.path.join(DATA_DIR, "DFT_reaction_info.csv")
    
    print(f"Loading metadata from {csv_path}...")
    df = pd.read_csv(csv_path)
    df = df[df['DE_F'].notna()]
    
    atom_vocab = build_atom_vocab()
    INV_ATOMIC_NUMBER = {v: k for k, v in ATOMIC_NUMBER.items()}
    
    samples = []
    atom_types_map = {}
    skipped_no_rc = 0
    target_reactions = config["target_reactions"]

    print(f"Opening HDF5 dataset at {h5_path} to extract {target_reactions} samples...")
    with h5py.File(h5_path, 'r') as f:
        for idx, row in df.iterrows():
            if len(samples) >= target_reactions:
                break
                
            rxn_id = str(row['channel']) if 'channel' in df else str(row.name)
            if rxn_id not in f:
                continue
                
            group = f[rxn_id]
            
            c_R_raw = group['RG'][:]
            c_P_raw = group['PG'][:]
            c_TS_raw = group['TSG'][:]
            elements = group['elements'][:]
            n = len(elements)
            
            if n > config["max_atoms"]:
                continue
                
            atom_types = [INV_ATOMIC_NUMBER.get(int(z), 'C') for z in elements]
            atom_ids = np.zeros(config["max_atoms"], dtype=np.int64)
            for i, a in enumerate(atom_types):
                atom_ids[i] = atom_vocab.get(a, 0)
                
            mask = np.zeros(config["max_atoms"], dtype=np.float32)
            mask[:n] = 1.0
            
            ea = float(row['DE_F'])
            dh = float(row['DH'])
            if config["skip_negative_ea"] and ea < 0:
                continue
                
            c_R = np.zeros((config["max_atoms"], 3), dtype=np.float32)
            c_P = np.zeros((config["max_atoms"], 3), dtype=np.float32)
            c_TS = np.zeros((config["max_atoms"], 3), dtype=np.float32)
            c_R[:n] = c_R_raw
            c_P[:n] = c_P_raw
            c_TS[:n] = c_TS_raw
            
            e_r = 0.0
            e_p = dh
            
            c_R_aligned_init = kabsch_align_reactant_fragments(
                c_R, c_P, atom_types, n, config["fragment_bond_scale"]
            )
            
            energy_feats = build_energy_features(
                atom_types, n, c_R_aligned_init, c_P, e_r, e_p, config["fragment_bond_scale"]
            )
            atom_phys = build_atom_physical_features(
                atom_types, n, config["max_atoms"]
            )
            
            D_R_raw = compute_distance_matrix(c_R)
            D_P_raw = compute_distance_matrix(c_P)
            D_TS = compute_distance_matrix(c_TS)
            
            risk = reaction_risk_features(
                D_R_raw, D_P_raw, atom_types, n, config["max_atoms"],
                config["fragment_bond_scale"], config["spectator_threshold"],
            )
            # Reaction-center pooling requires at least one forming/breaking bond.
            # A reaction whose covalent bond sets are identical in R and P yields an
            # empty RC mask, which the Ea head refuses to pool over (it raises). Drop
            # these at load time rather than degrade the pooling downstream.
            if risk["changed_bonds"] == 0:
                skipped_no_rc += 1
                continue
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
                "de_rxn_raw": dh,
                "energy_feats_raw": energy_feats,
                "atom_phys_raw": atom_phys,
                "D_TS": torch.from_numpy(D_TS),
                # Noise-free reactant/product distance matrices, precomputed here
                # (already built above for risk features) and persisted in the
                # sample cache so __getitem__ never recomputes them per epoch/run.
                "D_R": torch.from_numpy(D_R_raw),
                "D_P": torch.from_numpy(D_P_raw),
                "geom_mask": torch.from_numpy(geom_mask),
                "risk_pair_mask": torch.from_numpy(risk["risk_pair_mask"]),
                "risk_score": risk["risk_score"],
                "risk_penalty": risk["risk_penalty"],
                "complexity_margin_penalty": risk["complexity_margin_penalty"],
                "complexity_sigmoid_penalty": risk["complexity_sigmoid_penalty"],
                "complexity_flag": risk["complexity_flag"],
                "risky_chem_flag": risk["risky_chem_flag"],
                "formed_bonds": risk["formed_bonds"],
                "broken_bonds": risk["broken_bonds"],
                "changed_bonds": risk["changed_bonds"],
                "risky_bond_types": risk["risky_bond_types"],
            })
            
            if len(samples) % 1000 == 0:
                print(f"  Processed {len(samples)}/{target_reactions} reactions...")
                
    if skipped_no_rc:
        print(f"Skipped {skipped_no_rc} reactions with no forming/breaking bonds (empty reaction center).")
    print(f"Loaded {len(samples)} complete reaction triplets.")
    return samples, atom_vocab, atom_types_map


def _split_profile(samples, indices):
    """Compact distribution summary for one dataset split."""
    if not indices:
        return {
            "count": 0,
            "ea_mean": None,
            "ea_std": None,
            "ea_min": None,
            "ea_max": None,
            "atom_mean": None,
            "atom_min": None,
            "atom_max": None,
            "changed_mean": None,
            "changed_max": None,
            "formed_mean": None,
            "broken_mean": None,
            "risk_fraction": None,
            "complex_fraction": None,
            "risky_chem_fraction": None,
        }

    ea = np.array([samples[i]["Ea_raw"] for i in indices], dtype=np.float64)
    atoms = np.array([samples[i]["n_atoms"] for i in indices], dtype=np.float64)
    formed = np.array([samples[i].get("formed_bonds", 0) for i in indices], dtype=np.float64)
    broken = np.array([samples[i].get("broken_bonds", 0) for i in indices], dtype=np.float64)
    changed = formed + broken
    risk = np.array([samples[i].get("risk_score", 0.0) for i in indices], dtype=np.float64)
    complex_flag = np.array([samples[i].get("complexity_flag", 0.0) for i in indices], dtype=np.float64)
    risky_chem = np.array([samples[i].get("risky_chem_flag", 0.0) for i in indices], dtype=np.float64)
    return {
        "count": int(len(indices)),
        "ea_mean": float(ea.mean()),
        "ea_std": float(ea.std()),
        "ea_min": float(ea.min()),
        "ea_max": float(ea.max()),
        "atom_mean": float(atoms.mean()),
        "atom_min": int(atoms.min()),
        "atom_max": int(atoms.max()),
        "changed_mean": float(changed.mean()),
        "changed_max": int(changed.max()),
        "formed_mean": float(formed.mean()),
        "broken_mean": float(broken.mean()),
        "risk_fraction": float(np.mean(risk > 0.0)),
        "complex_fraction": float(np.mean(complex_flag > 0.0)),
        "risky_chem_fraction": float(np.mean(risky_chem > 0.0)),
    }

def _print_split_profile(label, profile):
    if profile["count"] == 0:
        print(f"  {label:<10} N=0")
        return
    print(
        f"  {label:<10} N={profile['count']:>6} | "
        f"Ea {profile['ea_mean']:7.2f}+/-{profile['ea_std']:<6.2f} "
        f"[{profile['ea_min']:.2f}, {profile['ea_max']:.2f}] | "
        f"atoms {profile['atom_mean']:5.1f} [{profile['atom_min']}-{profile['atom_max']}] | "
        f"changed {profile['changed_mean']:4.2f} max {profile['changed_max']:>2} | "
        f"risk {profile['risk_fraction'] * 100:5.1f}%"
    )

def _validate_split(samples, train_indices, val_indices):
    train_set = set(train_indices)
    val_set = set(val_indices)
    overlap = sorted(train_set & val_set)
    if overlap:
        raise ValueError(f"Train/validation split overlap detected at indices: {overlap[:10]}")
    if len(train_indices) != len(train_set) or len(val_indices) != len(val_set):
        raise ValueError("Train/validation split contains duplicate indices.")
    covered = train_set | val_set
    if len(covered) != len(samples):
        missing = sorted(set(range(len(samples))) - covered)
        raise ValueError(f"Train/validation split does not cover all samples. Missing: {missing[:10]}")

    seen_rxns, duplicate_rxns = set(), set()
    for s in samples:
        rxn_id = s["rxn_id"]
        if rxn_id in seen_rxns:
            duplicate_rxns.add(rxn_id)
        seen_rxns.add(rxn_id)
    train_rxns = {samples[i]["rxn_id"] for i in train_indices}
    val_rxns = {samples[i]["rxn_id"] for i in val_indices}
    leaked_rxns = sorted(train_rxns & val_rxns)
    return {
        "duplicate_rxn_ids": sorted(duplicate_rxns)[:20],
        "leaked_rxn_ids": leaked_rxns[:20],
        "has_leakage": bool(leaked_rxns),
    }

def make_train_val_split(samples, config):
    """Create a deterministic train/validation split and report its balance."""
    n_total = len(samples)
    if n_total < 2:
        raise ValueError("Need at least two complete reaction triplets to create train/validation splits.")

    val_split = float(config["val_split"])
    if not 0.0 < val_split < 1.0:
        raise ValueError(f"val_split must be between 0 and 1, got {val_split}.")
    n_val = min(max(1, int(round(n_total * val_split))), n_total - 1)
    seed = int(config["split_seed"])
    strategy = config["split_strategy"].lower()
    rng = np.random.default_rng(seed)

    if strategy == "random":
        indices = np.arange(n_total, dtype=np.int64)
        rng.shuffle(indices)
        val_indices = indices[:n_val].tolist()
        train_indices = indices[n_val:].tolist()
    elif strategy == "stratified":
        ea_values = np.array([s["Ea_raw"] for s in samples], dtype=np.float64)
        n_bins = max(1, int(config["split_bins"]))
        quantiles = np.linspace(0.0, 1.0, min(n_bins, n_total) + 1)[1:-1]
        ea_edges = np.unique(np.quantile(ea_values, quantiles)) if len(quantiles) else np.array([])
        strata = {}
        for i, s in enumerate(samples):
            ea_bin = int(np.searchsorted(ea_edges, s["Ea_raw"], side="right"))
            atom_bin = min(int(s["n_atoms"] // 5), 6)
            changed_bin = min(int(s.get("formed_bonds", 0) + s.get("broken_bonds", 0)), 4)
            risk_bin = int(float(s.get("risk_score", 0.0)) > 0.0)
            key = (ea_bin, atom_bin, changed_bin, risk_bin)
            strata.setdefault(key, []).append(i)

        train_indices, val_indices = [], []
        for key in sorted(strata):
            group = np.array(strata[key], dtype=np.int64)
            rng.shuffle(group)
            group_val = int(round(len(group) * val_split))
            if len(group) <= 1:
                group_val = 0
            else:
                group_val = min(group_val, len(group) - 1)
            val_indices.extend(group[:group_val].tolist())
            train_indices.extend(group[group_val:].tolist())

        rng.shuffle(train_indices)
        rng.shuffle(val_indices)
        if len(val_indices) < n_val:
            move_n = min(n_val - len(val_indices), len(train_indices) - 1)
            val_indices.extend(train_indices[:move_n])
            train_indices = train_indices[move_n:]
        elif len(val_indices) > n_val:
            move_n = len(val_indices) - n_val
            train_indices.extend(val_indices[:move_n])
            val_indices = val_indices[move_n:]
    else:
        raise ValueError(f"Unknown split_strategy '{strategy}'. Use 'random' or 'stratified'.")

    integrity = _validate_split(samples, train_indices, val_indices)
    if integrity["has_leakage"]:
        raise ValueError(f"Reaction IDs leaked across train/validation: {integrity['leaked_rxn_ids']}")

    report = {
        "strategy": strategy,
        "seed": seed,
        "val_split": val_split,
        "n_total": n_total,
        "n_train": len(train_indices),
        "n_val": len(val_indices),
        "integrity": integrity,
        "profiles": {
            "all": _split_profile(samples, list(range(n_total))),
            "train": _split_profile(samples, train_indices),
            "validation": _split_profile(samples, val_indices),
        },
    }

    print(f"\nData split ({strategy}, seed={seed}, requested val_split={val_split:.3f}):")
    _print_split_profile("All", report["profiles"]["all"])
    _print_split_profile("Train", report["profiles"]["train"])
    _print_split_profile("Val", report["profiles"]["validation"])
    if integrity["duplicate_rxn_ids"]:
        print(f"  Warning: duplicate rxn_id values found: {integrity['duplicate_rxn_ids']}")

    split_path = os.path.join(config["save_dir"], "split_diagnostics.json")
    with open(split_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Split diagnostics saved to {split_path}")
    return train_indices, val_indices, report

def compute_normalization(samples, indices):
    """Compute atom-phys + Ea + de_rxn normalization stats over the given indices.

    Restricting to the training indices keeps validation reactions out of the
    normalization statistics.  Ea and de_rxn are normalized (z-scored) so the
    learned Ea head regresses a well-scaled target/input; the physics Ea
    baseline is unaffected (it reads raw kcal/mol values directly).
    """
    if not indices:
        raise ValueError("Cannot compute normalization without at least one training sample.")

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
    # Energy-feature normalization: z-score the 28D reaction descriptor vector.
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

    Returns the raw Ea target (kcal/mol) and the z-scored de_rxn feature for the
    learned Ea head; Ea is normalized inside the training loop using ea_mean/std.

    Speedup: per-sample tensors (distance matrices, normalized features, risk
    penalty) are memoized on first access, so repeat epochs are a cheap dict
    lookup. This is done lazily -- there is no up-front pass over the whole
    dataset -- so only the indices actually drawn by a loader are cached, and
    the cache persists across epochs when the loader uses persistent workers.
    When coord_noise augmentation is active the item is recomputed every epoch
    (never cached) so fresh noise is sampled each time.
    """
    def __init__(self, config, samples, atom_vocab, atom_types_map, stats, is_train=False):
        self.config = config
        self.samples = samples
        self.atom_vocab = atom_vocab
        self.atom_types_map = atom_types_map
        self.aphys_mean = stats["aphys_mean"]
        self.aphys_std = stats["aphys_std"]
        self.de_rxn_mean = stats["de_rxn_mean"]
        self.de_rxn_std = stats["de_rxn_std"]
        self.efeat_mean = stats["efeat_mean"]
        self.efeat_std = stats["efeat_std"]
        self.is_train = is_train
        self._use_noise = is_train and config["coord_noise"] > 0.0
        # Memoized items, keyed by index. Only populated when there is no
        # per-epoch coord-noise augmentation (otherwise every epoch differs).
        self._cache = {}

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        if not self._use_noise and idx in self._cache:
            return self._cache[idx]

        s = self.samples[idx]
        ensure_sample_risk_features(
            s,
            self.config,
            self.atom_types_map.get(s.get("rxn_id"), s.get("atom_types")),
        )
        n = s["n_atoms"]
        if self._use_noise:
            # Coordinate-noise augmentation: perturb coords and recompute the
            # reactant/product distance matrices fresh each epoch. Distance
            # matrices are rotation/translation invariant, so no alignment needed.
            noise_std = self.config["coord_noise"]
            c_R = s["c_R"].copy()
            c_P = s["c_P"].copy()
            c_R[:n] += np.random.normal(scale=noise_std, size=(n, 3)).astype(np.float32)
            c_P[:n] += np.random.normal(scale=noise_std, size=(n, 3)).astype(np.float32)
            D_R_t = torch.from_numpy(compute_distance_matrix(c_R))
            D_P_t = torch.from_numpy(compute_distance_matrix(c_P))
        else:
            # No augmentation: reuse the noise-free distance matrices precomputed
            # at build time and persisted in the sample cache -- no per-epoch or
            # per-run recompute. Present for any cache_version >= 2.
            D_R_t = s["D_R"]
            D_P_t = s["D_P"]
        D_I_t = (D_R_t + D_P_t) / 2.0
        aphys_norm = (s["atom_phys_raw"] - self.aphys_mean) / self.aphys_std
        de_rxn_norm = (s["de_rxn_raw"] - self.de_rxn_mean) / self.de_rxn_std
        efeat_norm = (s["energy_feats_raw"] - self.efeat_mean) / self.efeat_std
        risk_penalty = continuous_risk_penalty(
            s.get("formed_bonds", 0),
            s.get("broken_bonds", 0),
            s.get("risky_chem_flag", 0.0),
            mode=self.config["risk_penalty_mode"],
            safe_min=self.config["risk_safe_min"],
            safe_max=self.config["risk_safe_max"],
            sigmoid_center=self.config["risk_sigmoid_center"],
            sigmoid_k=self.config["risk_sigmoid_k"],
        )
        item = {
            "rxn_id": s["rxn_id"],
            "n_atoms": n,
            "D_R": D_R_t,
            "D_I": D_I_t,
            "D_P": D_P_t,
            "D_TS": s["D_TS"],
            "mask": s["mask"],
            "geom_mask": s["geom_mask"],
            "atom_ids": s["atom_ids"],
            "atom_phys": torch.from_numpy(aphys_norm.astype(np.float32)),
            "Ea": torch.tensor(s["Ea_raw"], dtype=torch.float32),
            "de_rxn": torch.tensor(de_rxn_norm, dtype=torch.float32),
            "energy_feats": torch.from_numpy(efeat_norm.astype(np.float32)),
            "risk_pair_mask": s["risk_pair_mask"],
            "risk_score": torch.tensor(s["risk_score"], dtype=torch.float32),
            "risk_penalty": torch.tensor(risk_penalty, dtype=torch.float32),
            "complexity_flag": torch.tensor(s["complexity_flag"], dtype=torch.float32),
            "risky_chem_flag": torch.tensor(s["risky_chem_flag"], dtype=torch.float32),
        }
        if not self._use_noise:
            self._cache[idx] = item
        return item

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

    """
    key = (a1, a2) if (a1, a2) in BOND_FORCE_CONSTANTS else (a2, a1)
    if key in BOND_FORCE_CONSTANTS:
        return BOND_FORCE_CONSTANTS[key]
    raise KeyError(f"Bond force constant not found for pair: {a1}-{a2}")


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
        if not np.isfinite(lam) or lam < 0.0:
            lam = 0.0
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


def torch_mds_coords(D, mask, dim=3, on_gpu=False, compute_dtype=torch.float64):
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
    out_device, out_dtype = D.device, D.dtype
    # eigh location/precision is a throughput knob. Default (on_gpu=False,
    # float64) reproduces the original CPU-LAPACK path exactly. on_gpu=True keeps
    # the whole embedding on the GPU, which removes the per-forward device sync
    # and the two host<->device transfers -- the actual bottleneck in an async
    # CUDA pipeline -- at the cost of a (tiny-matrix) GPU eigh. This path is
    # detached, so lower precision only affects a seed the EGNN then refines.
    work_device = D.device if on_gpu else torch.device("cpu")
    Dw = D.detach().to(device=work_device, dtype=compute_dtype)
    m = mask.detach().to(device=work_device, dtype=compute_dtype)  # [B, N]
    pair = m.unsqueeze(-1) * m.unsqueeze(-2)         # [B, N, N] valid atom pairs
    cnt = m.sum(dim=1).clamp(min=1.0)                # [B] atoms per molecule
    S = (Dw ** 2) * pair                                 # squared, masked
    # Masked double centering: subtract row/col/grand means over valid atoms.
    row_mean = S.sum(dim=2) / cnt.unsqueeze(-1)                      # [B, N]
    grand = S.sum(dim=(1, 2)) / (cnt ** 2)                          # [B]
    Bmat = -0.5 * (S - row_mean.unsqueeze(-1) - row_mean.unsqueeze(-2)
                   + grand.view(B, 1, 1))
    Bmat = Bmat * pair                               # keep padded rows/cols at 0
    # Symmetrize defensively before eigh.
    Bmat = 0.5 * (Bmat + Bmat.transpose(1, 2))
    # Keep padded atoms out of the top eigenspace. A global positive jitter makes
    # padded dummy modes tie with true zero modes from planar fragments; shifting
    # only dummy diagonals negative prevents that mixing while preserving the
    # valid-block jitter.
    eyeN = torch.eye(N, dtype=Bmat.dtype, device=Bmat.device).unsqueeze(0)
    dummy_shift = (1.0 - m).unsqueeze(-1) * eyeN
    Bmat = Bmat - dummy_shift + 1e-6 * eyeN
    evals, evecs = torch.linalg.eigh(Bmat)       # ascending eigenvalues
    top_vals = evals[:, -dim:].flip(-1).clamp(min=0.0)              # [B, dim]
    top_vecs = evecs[:, :, -dim:].flip(-1)                          # [B, N, dim]
    coords = top_vecs * top_vals.clamp(min=0.0).sqrt().unsqueeze(1)
    coords = coords * m.unsqueeze(-1)                # zero padded atoms
    # eigh can silently return NaN eigenvectors on a degenerate Bmat (no
    # LinAlgError raised), which would poison the EGNN every forward pass. This
    # seed is detached, so replacing a bad embedding with zeros is harmless.
    coords = torch.nan_to_num(coords, nan=0.0, posinf=0.0, neginf=0.0)
    # Return on the original device/dtype for the EGNN (a no-op when on_gpu).
    return coords.to(dtype=out_dtype, device=out_device)


def triangle_inequality_loss(
    D, mask, geom_mask=None, tol=0.02, triplet_samples=1024, stochastic=True
):
    """Differentiable penalty for predicted distances that violate triangle inequality."""
    B, N, _ = D.shape
    if N < 3:
        return D.new_tensor(0.0)

    device = D.device
    triplets = torch.cartesian_prod(
        torch.arange(N, device=device),
        torch.arange(N, device=device),
        torch.arange(N, device=device),
    )
    distinct = (
        (triplets[:, 0] != triplets[:, 1])
        & (triplets[:, 0] != triplets[:, 2])
        & (triplets[:, 1] != triplets[:, 2])
    )
    triplets = triplets[distinct]
    if triplet_samples and 0 < triplet_samples < triplets.shape[0]:
        n_pick = int(triplet_samples)
        if stochastic:
            pick = torch.randperm(triplets.shape[0], device=device)[:n_pick]
        else:
            pick = torch.linspace(
                0,
                triplets.shape[0] - 1,
                steps=n_pick,
                device=device,
            ).long()
        triplets = triplets[pick]

    i, j, k = triplets[:, 0], triplets[:, 1], triplets[:, 2]
    Dij = D[:, i, j].float()
    Dik = D[:, i, k].float()
    Dkj = D[:, k, j].float()
    valid = (mask[:, i] * mask[:, j] * mask[:, k]).float()
    if geom_mask is not None:
        valid = valid * geom_mask[:, i, j].float() * geom_mask[:, i, k].float() * geom_mask[:, k, j].float()
    violation = F.relu(Dij - (Dik + Dkj) - float(tol))
    return ((violation ** 2) * valid).sum() / valid.sum().clamp(min=1.0)


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
        # Equivariant coordinate update (normalized by neighbor count). Clamp
        # the actual displacement vector norm, not just the scalar edge weight.
        raw_trans = rel * self.coord_mlp(m_ij)
        trans_norm = torch.norm(raw_trans, dim=-1, keepdim=True).clamp(min=1e-8)
        trans = raw_trans * (torch.clamp(trans_norm, max=self.coord_clamp) / trans_norm)
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
    attention-pool those per-atom features into a reactive-region descriptor,
    concatenate a masked mean descriptor for global context and a
    reaction-center-focused descriptor, append the separately-encoded physics
    stream, and regress a *normalized* Ea mean. A
    direct linear Bell-Evans-Polanyi term adds the signed reaction energy
    straight onto the mean so the dominant near-linear driver is not diluted
    inside the MLP. The input `h_ts` is detached by the caller, so the Ea
    gradient trains only this head and never reshapes the EGNN geometry.

    Output is a scalar normalized Ea.
    """
    def __init__(
        self, node_dim, hidden, energy_feat_dim=0, dropout=0.25,
    ):
        super().__init__()
        self.energy_feat_dim = int(energy_feat_dim)
        self.attn = nn.Sequential(
            nn.Linear(node_dim, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, 1),
        )
        # Concentration of the reaction-center pool. Used as a softmax temperature
        # on the reaction-center mask: a large value makes the pool a near-uniform
        # mean over the forming/breaking atoms (reproducing the original
        # reaction-center mean). An empty RC mask is NOT silently spread over all
        # atoms -- the forward pass raises loudly, since an empty mask means RC
        # detection failed for that sample (such reactions are filtered at load
        # time in _build_reaction_samples_from_h5).
        self.rc_attn_bias = nn.Parameter(torch.tensor(4.0))
        # --- Balanced three-stream fusion ------------------------------------
        # TS geometry stream: attention-, mean-, and reaction-center-pooled EGNN
        # node features projected to `hidden`. The reaction-center pool reads the
        # forming/breaking atoms directly instead of letting spectators dilute
        # the reactive signal in a whole-molecule mean.
        self.ts_proj = nn.Sequential(
            nn.Linear(3 * node_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        # Physics stream: signed reaction energy + z-scored energy/angle
        # descriptors get their own encoder so their ~29 dims are not drowned by
        # the high-dimensional TS stream in a flat concatenation.
        self.phys_enc = nn.Sequential(
            nn.Linear(1 + self.energy_feat_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        # FiLM: the energetics modulate the TS representation (Hammond/BEP -- the
        # TS position along the reaction coordinate depends on the reaction
        # energy). Zero-init keeps it an identity modulation at the start.
        self.film = nn.Linear(hidden, 2 * hidden)
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)
        final = nn.Linear(hidden // 2, 1)
        nn.init.xavier_uniform_(final.weight, gain=0.1)
        nn.init.zeros_(final.bias)
        self.net = nn.Sequential(
            nn.Linear(2 * hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            final,
        )
        # Direct Bell-Evans-Polanyi path: a linear signed-de_rxn -> normalized Ea
        # mean term added to the MLP output, so the dominant near-linear driver
        # reaches the prediction without being rederived inside the deep stack.
        self.bep = nn.Linear(1, 1)
        nn.init.constant_(self.bep.weight, 0.5)
        nn.init.zeros_(self.bep.bias)

    def forward(self, h_ts, mask, de_rxn, energy_feats, rc_mask):
        if self.energy_feat_dim > 0:
            if energy_feats is None:
                raise ValueError("energy_feats is required when energy_feat_dim > 0.")
            if (
                energy_feats.dim() != 2
                or energy_feats.size(0) != h_ts.size(0)
                or energy_feats.size(-1) != self.energy_feat_dim
            ):
                raise ValueError(
                    f"energy_feats must have shape [{h_ts.size(0)}, {self.energy_feat_dim}], "
                    f"got {tuple(energy_feats.shape)}"
                )
        valid = mask <= 0                                        # [B, N] padding
        m = mask.unsqueeze(-1).to(h_ts.dtype)                    # [B, N, 1]
        mean_pooled = (h_ts * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)
        attn_logits = self.attn(h_ts).squeeze(-1)                 # [B, N]
        attn_logits = attn_logits.masked_fill(valid, -1e4)
        attn_weights = torch.softmax(attn_logits, dim=1).unsqueeze(-1)
        attn_pooled = (h_ts * attn_weights * m).sum(dim=1)
        # Reaction-center pool as a softmax over the forming/breaking mask. With a
        # large temperature the weights are ~uniform over the reaction-center
        # atoms (i.e. their mean). An empty RC mask is NOT silently spread over all
        # atoms: it means RC detection failed for that sample, so we raise loudly
        # instead of degrading to a whole-molecule mean.
        rc_atom_count = (rc_mask.to(h_ts.dtype) * (~valid).to(h_ts.dtype)).sum(dim=1)
        if not bool(torch.all(rc_atom_count > 0)):
            raise ValueError(
                "Reaction-center mask is empty for at least one sample: RC detection "
                "produced no forming/breaking atoms. Refusing to silently pool over all "
                "atoms; fix RC detection or filter the offending reaction."
            )
        rc_logits = self.rc_attn_bias * rc_mask.to(h_ts.dtype)   # [B, N]
        rc_logits = rc_logits.masked_fill(valid, -1e4)
        rc_weights = torch.softmax(rc_logits, dim=1).unsqueeze(-1)
        rc_pooled = (h_ts * rc_weights).sum(dim=1)
        ts = self.ts_proj(torch.cat([attn_pooled, mean_pooled, rc_pooled], dim=-1))
        de_rxn_col = de_rxn.to(dtype=h_ts.dtype).unsqueeze(-1)    # [B, 1]
        phys_parts = [de_rxn_col]
        if self.energy_feat_dim > 0:
            phys_parts.append(energy_feats.to(dtype=h_ts.dtype))
        phys = self.phys_enc(torch.cat(phys_parts, dim=-1))
        gamma, beta = self.film(phys).chunk(2, dim=-1)
        ts_mod = ts * (1.0 + gamma) + beta
        out = self.net(torch.cat([ts_mod, phys], dim=-1))
        bep = self.bep(de_rxn_col)                               # [B, 1] direct BEP term
        return (out + bep).squeeze(-1)


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
        self.coord_noise = config["coord_noise"]
        self.rc_bond_scale = config["fragment_bond_scale"]
        self.mds_on_gpu = config["mds_on_gpu"]
        self.mds_dtype = torch.float32 if config["mds_dtype"] == "float32" else torch.float64
        d_model = config["gru_hidden"] * 2
        atom_dim = config["atom_embed_dim"]
        drop = config["dropout"]
        delta_clamp = config["delta_clamp"]
        self.core = PSICore(config, num_atom_types)
        self.geom_head = GeometryHead(d_model, atom_dim, ATOM_PHYS_DIM, drop, delta_clamp)
        # EGNN coordinate refiner. Node features = chemical-property vector
        # (learned atom embedding + physical descriptors); coordinates come from
        # the MDS embedding of the geometry head's predicted TS distance matrix.
        node_in_dim = atom_dim + ATOM_PHYS_DIM
        self.egnn = EGNN(
            node_in_dim,
            hidden=config["egnn_hidden"],
            n_layers=config["egnn_layers"],
            coord_clamp=config["egnn_coord_clamp"],
            dropout=drop,
        )
        # Learned Ea head on the EGNN's refined per-atom features (h_ts)
        # plus 28D energy descriptor (composition, bond-angle stats, RC angles).
        self.ea_head = EaHead(
            node_dim=config["egnn_hidden"],
            hidden=config["egnn_hidden"],
            energy_feat_dim=ENERGY_FEAT_DIM,
            dropout=config["ea_head_dropout"],
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

    def _reaction_center_mask(self, D_R, D_P, atom_ids, mask):
        return reaction_center_atom_mask(D_R, D_P, atom_ids, mask, self.rc_bond_scale)

    def forward(
        self, D_R, D_I, D_P, mask, atom_ids, atom_phys,
        de_rxn=None, energy_feats=None,
    ):
        """Predict TS distance matrix and (optionally) the learned Ea.

        Args:
            de_rxn: [B] z-scored signed reaction energy, fed to the Ea head.
                    May be None at geometry-only call sites; Ea is then None.
            energy_feats: [B, 28] z-scored molecular descriptor vector
                    (composition, bond-angle stats) for the Ea head.
        Returns:
            D_TS_pred:    [B, N, N] EGNN-refined TS distances
            D_TS_coarse:  [B, N, N] pre-EGNN coarse distances (for aux loss)
            ea_pred_norm: [B] normalized Ea from the EaHead, or None if de_rxn
                          was not supplied.
        """
        f = self.core(D_R, D_I, D_P, mask, atom_ids, atom_phys)
        atom_emb = self.core.atom_embed(atom_ids)
        # Coarse TS distance matrix from the geometry head.
        D_TS_coarse = self.geom_head(f, atom_emb, atom_phys, D_R, D_I, D_P, mask)
        ea_pred_norm = None
        # Chemical properties in one vector; predicted TS coordinates in the
        # other -- both fed to the EGNN. The MDS seed is detached so the
        # geometry head is trained directly by the coarse-distance aux loss
        # (keeping eigh's unstable backward out of the graph) while the EGNN
        # learns the coordinate refinement under the main geometry loss.
        node_feats = torch.cat([atom_emb, atom_phys], dim=-1)
        # Coordinate-space MDS (eigh) seed for the EGNN. Device/precision are
        # config-controlled (mds_on_gpu / mds_dtype): CPU-float64 by default,
        # or on-GPU to remove the per-forward sync + host<->device transfers.
        with torch.amp.autocast(device_type=D_R.device.type, enabled=False):
            x_init = torch_mds_coords(
                D_TS_coarse.detach().float(), mask,
                on_gpu=self.mds_on_gpu, compute_dtype=self.mds_dtype,
            )

        # --- Coordinate Noise Data Augmentation ---
        # Prevents late-stage EGNN geometry memorization (train-val gap)
        if self.training and self.coord_noise > 0.0:
            noise = torch.randn_like(x_init) * self.coord_noise
            x_init = x_init + (noise * mask.unsqueeze(-1))

        h_ts, x_ts = self.egnn(node_feats, x_init, mask)
        D_TS_pred = self._coords_to_distance(x_ts, mask)
        if de_rxn is not None:
            # h_ts is ALWAYS detached before the Ea head: the Ea gradient never
            # reaches the EGNN, so geometry is shaped purely by the geometry loss
            # and the head reads the settled TS as a fixed input.
            if energy_feats is None:
                raise ValueError(
                    "energy_feats is required when de_rxn is supplied to the Ea head; "
                    "refusing to run the Ea head on a missing energy descriptor."
                )
            rc_mask = self._reaction_center_mask(D_R, D_P, atom_ids, mask)
            ea_pred_norm = self.ea_head(
                h_ts.detach(), mask, de_rxn.float(), energy_feats.float(), rc_mask
            )
        return D_TS_pred, D_TS_coarse, ea_pred_norm

class PlateauWarmupScheduler:
    def __init__(self, optimizer, warmup_epochs, factor=0.5, patience=20, min_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.min_lr = min_lr
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        self.last_epoch = 0
        self.plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=factor, patience=patience, min_lr=min_lr
        )
        
    def step(self, metric):
        self.last_epoch += 1
        if self.last_epoch <= self.warmup_epochs:
            scale = self.last_epoch / max(1, self.warmup_epochs)
            for i, group in enumerate(self.optimizer.param_groups):
                group['lr'] = self.base_lrs[i] * scale
        else:
            self.plateau.step(metric)
            
    def state_dict(self):
        return {'last_epoch': self.last_epoch, 'plateau': self.plateau.state_dict(), 'base_lrs': self.base_lrs}
        
    def load_state_dict(self, state_dict):
        self.last_epoch = state_dict['last_epoch']
        self.base_lrs = state_dict['base_lrs']
        self.plateau.load_state_dict(state_dict['plateau'])


def freeze_backbone(model):
    """Freeze the geometry backbone (core + geometry head + EGNN) for Stage-2
    Ea-only training.

    Sets requires_grad=False and puts the submodules in eval() so dropout and
    coordinate-noise augmentation are off and the predicted TS is deterministic.
    Raises if the model has no Ea head / EGNN (Stage 2 is only meaningful then).
    """
    if not hasattr(model, "ea_head"):
        raise AttributeError("freeze_backbone requires a model with an ea_head.")
    for module in (model.core, model.geom_head, model.egnn):
        for p in module.parameters():
            p.requires_grad_(False)
        module.eval()


def build_ea_only_optimizer(model, config):
    """AdamW over the Ea-head parameters only (Stage-2 Ea-only training).

    Raises if the head is absent or has no trainable params -- no silent
    fallback to training the whole model.
    """
    if not hasattr(model, "ea_head"):
        raise AttributeError("ea_only stage requires a model with an ea_head.")
    ea_params = [p for p in model.ea_head.parameters() if p.requires_grad]
    if not ea_params:
        raise ValueError("ea_only stage: ea_head has no trainable parameters to optimize.")
    return torch.optim.AdamW(
        ea_params, lr=config["ea_head_lr"], weight_decay=config["ea_head_weight_decay"]
    )


def build_optimizer(model, config):
    """Use a faster, lighter-decayed optimizer group for the Ea head."""
    if not hasattr(model, "ea_head"):
        raise AttributeError("build_optimizer requires a model with an ea_head.")
    ea_params = list(model.ea_head.parameters())
    if not ea_params:
        raise ValueError("build_optimizer: ea_head has no parameters to optimize.")

    ea_param_ids = {id(p) for p in ea_params}
    base_params = [
        p for p in model.parameters()
        if p.requires_grad and id(p) not in ea_param_ids
    ]
    return torch.optim.AdamW(
        [
            {"params": base_params, "lr": config["lr"], "weight_decay": config["weight_decay"]},
            {
                "params": ea_params,
                "lr": config["ea_head_lr"],
                "weight_decay": config["ea_head_weight_decay"],
            },
        ], foreach=True
    )


def run_epoch(model, loader, optimizer, scaler, device, config, use_amp, epoch, stats, is_train=True):
    """Joint geometry + Ea training loop.

    Geometry is the backbone objective. The learned Ea head trains from
    `ea_loss_start_epoch` on ALWAYS-detached EGNN features, so its gradient
    never reshapes the geometry backbone.
    """
    geom_only = config["geom_only"]
    ea_only = config["ea_only"]
    if ea_only:
        # Backbone stays frozen/deterministic; only the Ea head trains.
        model.eval()
        if is_train:
            model.ea_head.train()
    elif is_train:
        model.train()
    else:
        model.eval()
    total_loss, total_geom, total_triangle = 0.0, 0.0, 0.0
    total_geom_mae_A = 0.0
    total_ea_mae, total_ea_norm, n_batches = 0.0, 0.0, 0
    ea_mean, ea_std = stats["ea_mean"], stats["ea_std"]
    if ea_only:
        # Stage 2: Ea head is the sole objective; the loss is unweighted.
        ea_started, ea_w = True, 1.0
    elif geom_only:
        # Stage 1: geometry backbone only; the Ea head never runs (de_rxn=None).
        ea_started, ea_w = False, 0.0
    else:
        ea_started = epoch >= config["ea_loss_start_epoch"]
        ea_w = config["ea_loss_weight"] if ea_started else 0.0
    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for batch in loader:
            (
                DR, DI, DP, DTS, mask, geom_mask, atom_ids, atom_phys, Ea,
                de_rxn, energy_feats, risk_pair_mask,
                risk_score, risk_penalty, complexity_flag, risky_chem_flag,
            ) = move_batch_to_device(batch, device)
            if is_train:
                optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                p_DTS, p_DTS_coarse, ea_pred_norm = model(
                    DR, DI, DP, mask, atom_ids, atom_phys,
                    None if geom_only else de_rxn, energy_feats,
                )
                if ea_only and ea_pred_norm is None:
                    raise RuntimeError("ea_only stage produced no Ea prediction (de_rxn missing).")
                N = DR.shape[1]
                valid_mask = mask.unsqueeze(-1) * mask.unsqueeze(-2)
                eye = torch.eye(N, device=mask.device, dtype=mask.dtype).unsqueeze(0)
                m2d = valid_mask * (1.0 - eye)
                
                # Inverse Distance Weighting: 1.0 / (DTS + 1.0)
                # This ensures small chemical bonds receive gradients equal in magnitude
                # to large inter-fragment distances, preventing fragment "melting".
                dist_weights = 1.0 / (DTS * m2d + 1.0)
                # --- Reaction-role reweighting (hinge fix; SWARM_FAILURE_ANALYSIS
                # sections 5-6). Pure inverse-distance weighting is near-blind to
                # the long-range active<->spectator cross-distances that encode
                # global fragment orientation. Boost those cross pairs and the
                # reactive pairs, and damp the static spectator backbone.
                active = reaction_center_atom_mask(
                    DR, DP, atom_ids, mask, config["fragment_bond_scale"]
                ).to(dist_weights.dtype)                     # [B, N]
                a_i = active.unsqueeze(2)                    # [B, N, 1]
                a_j = active.unsqueeze(1)                    # [B, 1, N]
                role_weight = (
                    config["geom_hinge_cross_weight"] * (a_i + a_j - 2.0 * a_i * a_j)
                    + config["geom_active_pair_weight"] * (a_i * a_j)
                    + config["geom_spectator_spectator_weight"] * ((1.0 - a_i) * (1.0 - a_j))
                )
                dist_weights = dist_weights * role_weight
                m2d_weighted = m2d * dist_weights
                denom_weighted = m2d_weighted.sum().clamp(min=1)
                
                # Main geometry loss on the EGNN-refined distances.
                l_geom = F.huber_loss(p_DTS * m2d_weighted, DTS * m2d_weighted, reduction='sum', delta=0.5) / denom_weighted
                # Auxiliary loss on the coarse (pre-EGNN) distances trains the
                # geometry head directly, giving the EGNN a stable MDS seed.
                l_geom_coarse = F.huber_loss(p_DTS_coarse * m2d_weighted, DTS * m2d_weighted, reduction='sum', delta=0.5) / denom_weighted
                # Physical, unweighted interatomic-distance MAE (Angstrom): an
                # interpretable readout of geometry quality, independent of the
                # role/inverse-distance weighting used for the gradient.
                geom_mae_A = (torch.abs(p_DTS - DTS) * m2d).sum() / m2d.sum().clamp(min=1.0)

                # --- PINN Matrix-wise Cross Check ---
                # Physics constraint 1: Spectator bonds (abs(DR - DP) < threshold) shouldn't change.
                # Target them towards the reactant/product midpoint (DI).
                spectator_mask = (torch.abs(DR - DP) < config["spectator_threshold"]).float() * m2d
                spectator_denom = spectator_mask.sum().clamp(min=1.0)
                l_pinn_spectator = (
                    F.mse_loss(p_DTS * spectator_mask, DI * spectator_mask, reduction='sum')
                    / spectator_denom
                )

                # Physics constraint 2: steric floor. Penalize predicted TS
                # distances below floor_frac*(r_i + r_j) so the model learns to
                # avoid atomic clashes during training, instead of relying only
                # on the post-hoc clamp_steric_collisions() at inference.
                radii = covalent_radius_lookup(atom_ids.device)          # [V+1]
                r_atom = radii[atom_ids]                                 # [B, N]
                min_dist = STERIC_FLOOR_FRAC * (r_atom.unsqueeze(2) + r_atom.unsqueeze(1))
                steric_violation = F.relu(min_dist.to(p_DTS.dtype) - p_DTS) * m2d
                l_pinn_steric = steric_violation.sum() / m2d.sum().clamp(min=1.0)

                l_pinn = l_pinn_spectator + config["steric_loss_weight"] * l_pinn_steric

                l_triangle_refined = triangle_inequality_loss(
                    p_DTS,
                    mask,
                    geom_mask,
                    tol=config["triangle_tolerance"],
                    triplet_samples=config["triangle_triplet_samples"],
                    stochastic=is_train,
                )
                l_triangle_coarse = triangle_inequality_loss(
                    p_DTS_coarse,
                    mask,
                    geom_mask,
                    tol=config["triangle_tolerance"],
                    triplet_samples=config["triangle_triplet_samples"],
                    stochastic=is_train,
                )
                l_triangle = (
                    config["triangle_refined_weight"] * l_triangle_refined
                    + config["triangle_coarse_weight"] * l_triangle_coarse
                )

                if ea_only:
                    # Geometry backbone is frozen; l_geom / l_triangle above are
                    # computed for metrics only, never optimized.
                    loss = p_DTS.new_zeros(())
                else:
                    loss = (
                        l_geom
                        + config["geom_coarse_weight"] * l_geom_coarse
                        + 0.2 * l_pinn
                        + config["triangle_loss_weight"] * l_triangle
                    )

                risk_scale = (
                    1.0 + config["risk_weight_alpha"] * risk_penalty.float()
                ).clamp(max=config["risk_weight_max"])
                risk_sample = (risk_penalty > 0.0).float()
                risk_pair = risk_pair_mask * m2d
                risk_pair_weight = risk_pair * risk_scale.view(-1, 1, 1)
                risk_pair_denom = risk_pair_weight.sum().clamp(min=1.0)
                if not ea_only and risk_pair.sum() > 0:
                    risk_geom_abs = F.huber_loss(p_DTS, DTS, reduction='none', delta=0.5)
                    l_risk_geom = (risk_geom_abs * risk_pair_weight).sum() / risk_pair_denom
                    loss = loss + config["risk_geom_loss_weight"] * l_risk_geom

                # Learned Ea loss on the normalized target (SmoothL1 regression).
                # The head reads detached TS features, so this gradient trains
                # only the Ea head, never the geometry backbone.
                if ea_pred_norm is not None:
                    ea_mean_norm = ea_pred_norm
                    ea_target_norm = (Ea - ea_mean) / ea_std
                    ea_per_sample = F.smooth_l1_loss(ea_mean_norm, ea_target_norm, reduction='none')
                    l_ea = ea_per_sample.mean()
                    if ea_w > 0.0:
                        loss = loss + ea_w * l_ea
                        if ea_started and risk_sample.sum() > 0:
                            risk_weight = risk_scale * risk_sample
                            l_risk_ea = (ea_per_sample * risk_weight).sum() / risk_weight.sum().clamp(min=1.0)
                            loss = loss + config["risk_ea_loss_weight"] * l_risk_ea
                else:
                    l_ea = None
                    ea_mean_norm = ea_target_norm = None
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
            total_geom_mae_A += geom_mae_A.item()
            total_triangle += l_triangle.item()
            if l_ea is not None:
                total_ea_norm += l_ea.item()
                # Denormalized Ea MAE (kcal/mol) for human-readable tracking.
                total_ea_mae += (ea_mean_norm.detach() - ea_target_norm).abs().mean().item() * ea_std
            n_batches += 1
    nb = max(n_batches, 1)
    return {
        "loss": total_loss / nb,
        "geom": total_geom / nb,
        "geom_mae_A": total_geom_mae_A / nb,
        "triangle": total_triangle / nb,
        "ea_mae": total_ea_mae / nb,
        "ea_norm": total_ea_norm / nb,
    }

class _TeeLogger:
    """Mirror a stream (stdout/stderr) to a log file so the full run transcript is
    saved verbatim, including when the run is interrupted mid-way."""
    def __init__(self, stream, file_handle):
        self._stream = stream
        self._file = file_handle

    def write(self, data):
        self._stream.write(data)
        # Narrow guard: the only expected failure is writing after the file was
        # closed during interpreter teardown; real I/O errors (disk full) still
        # surface via the console stream above.
        try:
            self._file.write(data)
            self._file.flush()
        except ValueError:
            pass
        return len(data)

    def flush(self):
        self._stream.flush()
        try:
            self._file.flush()
        except ValueError:
            pass

    def __getattr__(self, name):
        return getattr(self._stream, name)


def _write_run_report(run_ctx):
    """Atomically write the structured run report so a mid-run stop still leaves a
    complete, uncorrupted snapshot for analysis."""
    report = {
        "status": run_ctx["status"],
        "stop_reason": run_ctx.get("stop_reason"),
        "started_at": run_ctx["started_at"],
        "elapsed_sec": round(time.time() - run_ctx["start_time"], 1),
        "epochs_target": run_ctx["epochs_target"],
        "epochs_completed": run_ctx["epochs_completed"],
        "best_val_select": run_ctx["best_val_select"],
        "best_epoch": run_ctx["best_epoch"],
        "patience_counter": run_ctx["patience_counter"],
        "last_metrics": run_ctx["last_metrics"],
        "error": run_ctx.get("error"),
        "config": run_ctx["config"],
    }
    path = run_ctx["report_path"]
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    os.replace(tmp, path)  # atomic: a crash mid-write never corrupts the report


def _begin_run_logging(config):
    """Install the tee logger and initialise the run report. Returns a run_ctx
    dict the training loop updates and the finaliser flushes."""
    save_dir = config["save_dir"]
    os.makedirs(save_dir, exist_ok=True)
    log_path = os.path.join(save_dir, "run_log.txt")
    report_path = os.path.join(save_dir, "run_report.json")
    log_file = open(log_path, "a", encoding="utf-8", buffering=1)
    stamp = datetime.now().isoformat(timespec="seconds")
    log_file.write(f"\n{'='*70}\n[run start] {stamp}\n{'='*70}\n")
    log_file.flush()
    run_ctx = {
        "log_file": log_file,
        "report_path": report_path,
        "history_path": os.path.join(save_dir, "training_history.json"),
        "orig_stdout": sys.stdout,
        "orig_stderr": sys.stderr,
        "start_time": time.time(),
        "started_at": stamp,
        "status": "running",
        "stop_reason": None,
        "config": {k: v for k, v in config.items()
                   if isinstance(v, (int, float, str, bool, list, tuple, type(None)))},
        "epochs_target": config.get("epochs"),
        "epochs_completed": 0,
        "best_val_select": None,
        "best_epoch": None,
        "patience_counter": 0,
        "last_metrics": None,
    }
    sys.stdout = _TeeLogger(run_ctx["orig_stdout"], log_file)
    sys.stderr = _TeeLogger(run_ctx["orig_stderr"], log_file)
    print(f"[run logging] transcript -> {log_path}")
    print(f"[run logging] live report -> {report_path}")
    _write_run_report(run_ctx)
    return run_ctx


def _finalize_run_logging(run_ctx):
    """Flush the final report and restore the original streams. Always runs, so a
    completed, early-stopped, interrupted, or errored run all end with a report."""
    _write_run_report(run_ctx)
    elapsed = round(time.time() - run_ctx["start_time"], 1)
    print(f"\n[run {run_ctx['status']}] elapsed={elapsed}s "
          f"epochs_completed={run_ctx['epochs_completed']} "
          f"report -> {run_ctx['report_path']}")
    sys.stdout = run_ctx["orig_stdout"]
    sys.stderr = run_ctx["orig_stderr"]
    try:
        run_ctx["log_file"].close()
    except (ValueError, OSError):
        pass


def train_pipeline(config):
    """Thin wrapper that guarantees a verbose saved transcript and a structured
    run report even if training is stopped mid-way (Ctrl-C) or errors out."""
    run_ctx = _begin_run_logging(config)
    try:
        _train_run(config, run_ctx)
        if run_ctx["status"] == "running":
            run_ctx["status"] = "completed"
    except KeyboardInterrupt:
        run_ctx["status"] = "interrupted"
        run_ctx["stop_reason"] = "keyboard_interrupt"
        print("\n[interrupted] KeyboardInterrupt received; run stopped mid-way. "
              "History, latest checkpoint, and report have been saved.")
    except Exception:
        run_ctx["status"] = "error"
        run_ctx["stop_reason"] = "exception"
        run_ctx["error"] = traceback.format_exc()
        print("\n[error] Unhandled exception during run:\n" + run_ctx["error"])
        raise
    finally:
        _finalize_run_logging(run_ctx)


def _train_run(config, run_ctx):
    device = resolve_device(config)
    configure_torch_runtime(device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    os.makedirs(config["save_dir"], exist_ok=True)
    print("="*70); print(" PSI FULL PIPELINE (v2) "); print("="*70)
    extract_raw_data(config)
    samples, atom_vocab, atom_types_map = build_reaction_samples(config)
    if len(samples) == 0:
        print("Error: No complete reaction triplets found.")
        return
    n_total = len(samples)
    train_indices, val_indices, split_report = make_train_val_split(samples, config)
    stats = compute_normalization(samples, train_indices)
    train_dataset = ReactionDataset(config, samples, atom_vocab, atom_types_map, stats, is_train=True)
    eval_dataset = ReactionDataset(config, samples, atom_vocab, atom_types_map, stats, is_train=False)
    train_subset = Subset(train_dataset, train_indices)
    val_subset = Subset(eval_dataset, val_indices)
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
    geom_only = config["geom_only"]
    ea_only = config["ea_only"]
    if geom_only and ea_only:
        raise ValueError("geom_only and ea_only are mutually exclusive training stages.")
    if ea_only:
        backbone_ckpt = config["backbone_ckpt"]
        if not backbone_ckpt or not os.path.exists(backbone_ckpt):
            raise FileNotFoundError(
                f"ea_only stage requires an existing --backbone-ckpt; got {backbone_ckpt!r}"
            )
        bb = torch.load(backbone_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(bb["model_state_dict"])
        freeze_backbone(model)
        model.coord_noise = 0.0
        print(f"Stage 2 (Ea-only): loaded and froze backbone from {backbone_ckpt}")
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params:,}")
    optimizer = build_ea_only_optimizer(model, config) if ea_only else build_optimizer(model, config)
    if hasattr(model, "ea_head"):
        print(f"Learning rates: base={config['lr']:.2e}, ea_head={config['ea_head_lr']:.2e}")
    scheduler = PlateauWarmupScheduler(
        optimizer, warmup_epochs=config["warmup_epochs"], factor=0.5, patience=15, min_lr=1e-6
    )
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
        "split_summary": split_report,
        "config_snapshot": {
            k: v for k, v in config.items()
            if isinstance(v, (int, float, str, bool, list, tuple))
        },
    }
    print(f"\nTraining for up to {config['epochs']} epochs (patience={config['patience']})...")
    print(
        f"  Ea head starts at epoch {config['ea_loss_start_epoch']} on always-detached "
        f"TS features (Ea gradient never reaches the geometry backbone)."
    )
    print(f"{'Epoch':>6} | {'Train Loss':>11} | {'Val Loss':>11} | {'T.Geom':>8} | {'V.Geom':>8} | {'V.dMAE_A':>9} | {'T.EaMAE':>8} | {'V.EaMAE':>8} | {'LR':>10}")
    print("-" * 106)
    best_val_loss = float('inf')
    patience_counter = 0
    history = []
    if ea_only:
        best_model_path = os.path.join(config["save_dir"], "psi_ea_best.pt")
    elif geom_only:
        best_model_path = os.path.join(config["save_dir"], "psi_geom_best.pt")
    else:
        best_model_path = os.path.join(config["save_dir"], "psi_best.pt")
    latest_model_path = os.path.join(config["save_dir"], "psi_latest.pt")
    start_epoch = 1

    from torch.optim.swa_utils import AveragedModel, SWALR
    swa_enabled = config["swa_enabled"]
    swa_start = config["swa_start"]
    swa_model = AveragedModel(model) if swa_enabled else None
    swa_scheduler = SWALR(optimizer, swa_lr=config["lr"] * 0.5) if swa_enabled else None

    for epoch in range(start_epoch, config["epochs"] + 1):
        epoch_t0 = time.time()
        train_metrics = run_epoch(model, train_loader, optimizer, scaler, device, config, use_amp, epoch, stats, is_train=True)
        if swa_enabled and epoch >= swa_start:
            swa_model.update_parameters(model)
            val_metrics = run_epoch(swa_model.module, val_loader, None, scaler, device, config, use_amp, epoch, stats, is_train=False)
        else:
            val_metrics = run_epoch(model, val_loader, None, scaler, device, config, use_amp, epoch, stats, is_train=False)
        
        if ea_only:
            # Stage 2: select and early-stop purely on validation Ea MAE (kcal/mol).
            val_select = val_metrics["ea_mae"]
        elif geom_only:
            # Stage 1: geometry only.
            val_select = val_metrics["geom"]
        else:
            val_select = val_metrics["geom"] + config["ea_select_weight"] * val_metrics["ea_norm"]

        if swa_enabled and epoch >= swa_start:
            swa_scheduler.step()
        else:
            scheduler.step(val_select)
        current_lr = optimizer.param_groups[0]['lr']
        current_ea_lr = optimizer.param_groups[-1]['lr']

        history.append({
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "val_loss": val_metrics["loss"],
            "val_select": val_select,
            "train_geom": train_metrics["geom"],
            "val_geom": val_metrics["geom"],
            "train_geom_mae_A": train_metrics["geom_mae_A"],
            "val_geom_mae_A": val_metrics["geom_mae_A"],
            "train_triangle": train_metrics["triangle"],
            "val_triangle": val_metrics["triangle"],
            "train_ea_mae": train_metrics["ea_mae"],
            "train_ea_norm": train_metrics["ea_norm"],
            "val_ea_mae": val_metrics["ea_mae"],
            "val_ea_norm": val_metrics["ea_norm"],
            "lr": current_lr,
            "ea_lr": current_ea_lr,
        })
        improved = val_select < best_val_loss
        if improved:
            best_val_loss = val_select
            best_state_dict = swa_model.module.state_dict() if (swa_enabled and epoch >= swa_start) else model.state_dict()
            torch.save({"model_state_dict": best_state_dict, "metadata": metadata}, best_model_path)
        patience_counter = 0 if improved else patience_counter + 1

        save_dict = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "epoch": epoch,
            "best_val_loss": best_val_loss,
            "patience_counter": patience_counter,
            "metadata": metadata,
            "history": history
        }
        if swa_enabled and swa_model is not None:
            save_dict["swa_model_state_dict"] = swa_model.state_dict()
            if swa_scheduler is not None:
                save_dict["swa_scheduler_state_dict"] = swa_scheduler.state_dict()
        torch.save(save_dict, latest_model_path)

        # Verbose per-epoch line (every epoch) with wall-time and ETA, plus
        # incremental persistence so an interrupted run is still fully analysable.
        epoch_time = time.time() - epoch_t0
        eta_min = (config["epochs"] - epoch) * epoch_time / 60.0
        marker = " *" if improved else ""
        print(f"{epoch:6d} | {train_metrics['loss']:11.4f} | {val_metrics['loss']:11.4f} | "
              f"{train_metrics['geom']:8.5f} | {val_metrics['geom']:8.5f} | "
              f"{val_metrics['geom_mae_A']:9.4f} | "
              f"{train_metrics['ea_mae']:8.3f} | {val_metrics['ea_mae']:8.3f} | "
              f"{current_lr:10.2e} | {epoch_time:6.1f}s | ETA {eta_min:6.1f}m{marker}")
        with open(run_ctx["history_path"], "w") as f:
            json.dump(history, f, indent=2)
        if improved:
            run_ctx["best_epoch"] = epoch
        run_ctx["epochs_completed"] = epoch
        run_ctx["best_val_select"] = best_val_loss
        run_ctx["patience_counter"] = patience_counter
        run_ctx["last_metrics"] = history[-1]
        _write_run_report(run_ctx)
        if patience_counter >= config["patience"]:
            print(f"\nEarly stopping at epoch {epoch} (no improvement for {config['patience']} epochs)")
            run_ctx["stop_reason"] = "early_stopping_patience"
            break
    else:
        run_ctx["stop_reason"] = "max_epochs_reached"
    history_path = run_ctx["history_path"]
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"\nTraining history saved to {history_path}")
    print(f"\nLoading best model (best val_select={best_val_loss:.4f})...")
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
            (
                DR, DI, DP, DTS, mask, geom_mask, atom_ids, atom_phys, _Ea,
                de_rxn, energy_feats, _risk_pair_mask,
                _risk_score, _risk_penalty, _complexity_flag, _risky_chem_flag,
            ) = move_batch_to_device(batch, device)
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
        # Recover 3D coordinates from the full distance matrix so inter-fragment
        # predictions are preserved.
        c_R = np.asarray(s["c_R"][:n], dtype=np.float64)
        c_P = np.asarray(s["c_P"][:n], dtype=np.float64)
        c_I = (kabsch_align_reactant_fragments(c_R, c_P, atom_types, n, config["fragment_bond_scale"])[:n] + c_P[:n]) / 2.0
        pred_coords = mds_aligned(pred_dist, reference_coords=c_I)
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
        # Learned head is the primary Ea; no silent substitution of the physics
        # baseline. The head must have produced a value for every reaction.
        if rxn_id not in ea_neural_map:
            raise RuntimeError(
                f"Learned Ea missing for reaction {rxn_id}; the Ea head must produce a "
                "value for every evaluated reaction (no silent physics fallback)."
            )
        ea_pred = ea_neural_map[rxn_id]
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
    # Ea / de_rxn normalization stats for the learned head. These MUST be present:
    # predicting with identity/zero normalization would return a silently wrong Ea.
    required_stats = ("ea_mean", "ea_std", "de_rxn_mean", "de_rxn_std", "efeat_mean", "efeat_std")
    missing_stats = [k for k in required_stats if meta.get(k) is None]
    if missing_stats:
        raise ValueError(
            f"Checkpoint metadata is missing normalization stats {missing_stats}; refusing "
            "to predict with identity/zero normalization. Re-run training/evaluation to "
            "produce a complete checkpoint."
        )
    ea_mean = meta["ea_mean"]
    ea_std = meta["ea_std"]
    de_rxn_mean = meta["de_rxn_mean"]
    de_rxn_std = meta["de_rxn_std"]
    efeat_mean = np.array(meta["efeat_mean"], dtype=np.float32)
    efeat_std = np.array(meta["efeat_std"], dtype=np.float32)
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
    # 28D energy descriptor (composition, bond-angle stats, RC angles) for the Ea head.
    energy_feats = build_energy_features(
        r_types, n, c_R_aligned, c_P, e_r, e_p, config["fragment_bond_scale"]
    )
    energy_feats_norm = (energy_feats - efeat_mean) / efeat_std
    model_config = dict(config)
    model = PSI(model_config, num_atom_types).to(device)
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
    # Learned Ea (denormalized). Normalization stats are validated above, so the
    # only way this is None is if the head produced nothing, which we reject below.
    ea_neural = None
    if ea_pred_norm is not None:
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
    pred_dist = enforce_triangle_inequality(pred_dist)
    validate_ts_geometry(pred_dist, r_types[:n], n)
    pred_coords = mds_aligned(pred_dist, reference_coords=c_I_real)
    # Physics-based Ea baseline from the predicted 3D TS coordinates.
    ea_physics = None
    if "physics_ea_coeffs" in meta:
        ea_calculator = PhysicsEaCalculator(
            bond_scale=config["fragment_bond_scale"],
            spectator_threshold=config["spectator_threshold"],
        )
        ea_calculator.coeffs = np.array(meta["physics_ea_coeffs"], dtype=np.float64)
        ea_calculator.fitted = True
        ea_physics = ea_calculator.predict_single(
            c_R[:n], pred_coords, c_P[:n], r_types[:n], n, de_rxn
        )
    # Primary Ea is the learned head. No silent physics substitution: with the
    # normalization stats validated above, the head must have produced a value.
    if ea_neural is None:
        raise RuntimeError(
            "Learned Ea head produced no prediction despite valid normalization stats; "
            "refusing to silently substitute the physics baseline."
        )
    energy_pred = ea_neural
    ea_source = "neural"
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
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    if xyz_path:
        os.makedirs(os.path.dirname(xyz_path) or ".", exist_ok=True)
        write_xyz(xyz_path, r_types, pred_coords, f"PSI predicted TS, Ea={energy_pred:.4f} kcal/mol")
    print("\n" + "="*70)
    print(" PREDICTION RESULT ")
    print("="*70)
    print(f"Atoms: {n}")
    print(f"Predicted activation energy ({ea_source}): {energy_pred:.4f} kcal/mol")
    if ea_physics is not None:
        print(f"  Physics baseline: {ea_physics:.4f} kcal/mol")
    else:
        print("  Physics baseline: unavailable in this checkpoint")
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
    if not data:
        raise ValueError(f"{data_path} does not contain any reaction records.")
    os.makedirs(save_dir, exist_ok=True)

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

        X_true = mds(dt)
        X_pred = mds_aligned(dp, reference_coords=X_true)
        X_guess = mds_aligned(di, reference_coords=X_true)

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
    train_parser.add_argument("--dataset", choices=["b97d3", "wb97xd3"], default=CONFIG.get("dataset", "b97d3"), help="Dataset version to train on")
    train_parser.add_argument("--extract-limit", type=int, default=CONFIG["extraction_limit"], help="Number of log files to parse from the tarball")
    train_parser.add_argument("--target-reactions", type=int, default=CONFIG["target_reactions"], help="Maximum complete reaction triplets to train/evaluate")
    train_parser.add_argument("--force-extract", action="store_true", help="Rebuild extracted_dataset.json instead of reusing it")
    train_parser.add_argument("--epochs", type=int, default=CONFIG["epochs"], help="Training epochs")
    train_parser.add_argument("--batch-size", type=int, default=CONFIG["batch_size"], help="Training batch size")
    train_parser.add_argument("--num-workers", type=int, default=CONFIG["num_workers"], help="DataLoader worker processes")
    train_parser.add_argument("--val-split", type=float, default=CONFIG["val_split"], help="Validation fraction; 0.1 keeps 90%% of data for training")
    train_parser.add_argument("--split-seed", type=int, default=CONFIG["split_seed"], help="Random seed for train/validation splitting")
    train_parser.add_argument("--split-strategy", choices=["random", "stratified"], default=CONFIG["split_strategy"], help="Train/validation split strategy")
    train_parser.add_argument("--split-bins", type=int, default=CONFIG["split_bins"], help="Ea quantile bins for stratified splitting")
    train_parser.add_argument("--risk-penalty-mode", choices=["binary", "margin", "sigmoid"], default=CONFIG["risk_penalty_mode"], help="Sample-level risk weighting function")
    train_parser.add_argument("--risk-weight-alpha", type=float, default=CONFIG["risk_weight_alpha"], help="Scale applied to continuous risk penalty weights")
    train_parser.add_argument("--risk-weight-max", type=float, default=CONFIG["risk_weight_max"], help="Maximum continuous risk loss multiplier")
    train_parser.add_argument("--triangle-loss-weight", type=float, default=CONFIG["triangle_loss_weight"], help="Weight for triangle-inequality PINN loss")
    train_parser.add_argument("--triangle-triplet-samples", type=int, default=CONFIG["triangle_triplet_samples"], help="Triplets sampled per batch for triangle loss; 0 uses all")
    train_parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default=CONFIG["device"], help="Training device")
    train_parser.add_argument("--require-cuda", action="store_true", help="Fail instead of falling back to CPU")
    train_parser.add_argument("--no-amp", action="store_true", help="Disable CUDA mixed precision")
    train_parser.add_argument("--patience", type=int, default=CONFIG["patience"], help="Early stopping patience")
    train_parser.add_argument("--lr", type=float, default=CONFIG["lr"], help="Learning rate")
    train_parser.add_argument("--ea-head-lr", type=float, default=CONFIG["ea_head_lr"], help="Learning rate for the Ea head")
    train_parser.add_argument("--ea-loss-weight", type=float, default=CONFIG["ea_loss_weight"], help="Full joint Ea objective weight")
    train_parser.add_argument("--ea-loss-start-epoch", type=int, default=CONFIG["ea_loss_start_epoch"], help="First epoch that trains the Ea head")
    train_parser.add_argument("--ea-select-weight", type=float, default=CONFIG["ea_select_weight"], help="Ea contribution to validation checkpoint selection")
    train_parser.add_argument("--ea-head-dropout", type=float, default=CONFIG["ea_head_dropout"], help="Dropout inside the Ea head MLP")
    train_parser.add_argument("--swa-start", type=int, default=CONFIG["swa_start"], help="Epoch to start SWA")
    train_parser.add_argument("--no-swa", action="store_true", help="Disable SWA")
    train_parser.add_argument("--save-dir", default=CONFIG["save_dir"], help="Directory to save checkpoints (e.g. runs/phase1_warm_start)")
    train_parser.add_argument("--geom-only", action="store_true", help="Stage 1: train the geometry backbone only (no Ea head/loss); select on validation geometry error")
    train_parser.add_argument("--ea-only", action="store_true", help="Stage 2: freeze a loaded backbone (--backbone-ckpt) and train only the Ea head on its frozen predicted TS")
    train_parser.add_argument("--backbone-ckpt", default=None, help="Stage 2: path to the frozen Stage-1 backbone checkpoint (required with --ea-only)")
    train_parser.add_argument("--data-dir", default=CONFIG.get("data_dir"), help="Directory holding RGD1_CHNO.h5 + DFT_reaction_info.csv. Default: local d:/Transition state/RGD1_Dataset. Set this to run off the hardcoded Windows path (e.g. a Kaggle input mount).")
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
    if len(sys.argv) == 1:
        args = parser.parse_args(["train"])
    else:
        args = parser.parse_args()
    if args.command == "predict":
        CONFIG["device"] = args.device
        if args.require_cuda:
            CONFIG["require_cuda"] = True
        predict_transition_state(CONFIG, args.reactant, args.product, args.model, args.output, args.xyz)
    elif args.command == "dashboard":
        create_dashboard(args.data, args.save_dir)
    else:
        if args.command == "train":
            CONFIG["dataset"] = args.dataset
            CONFIG["tar_path"] = f"{args.dataset}.tar.gz"
            CONFIG["dataset_json"] = f"extracted_{args.dataset}.json"
            CONFIG["extraction_limit"] = args.extract_limit
            CONFIG["target_reactions"] = args.target_reactions
            if args.force_extract:
                CONFIG["force_extract"] = True
            CONFIG["epochs"] = args.epochs
            CONFIG["batch_size"] = args.batch_size
            CONFIG["num_workers"] = args.num_workers
            CONFIG["val_split"] = args.val_split
            CONFIG["split_seed"] = args.split_seed
            CONFIG["split_strategy"] = args.split_strategy
            CONFIG["split_bins"] = args.split_bins
            CONFIG["risk_penalty_mode"] = args.risk_penalty_mode
            CONFIG["risk_weight_alpha"] = args.risk_weight_alpha
            CONFIG["risk_weight_max"] = args.risk_weight_max
            CONFIG["triangle_loss_weight"] = args.triangle_loss_weight
            CONFIG["triangle_triplet_samples"] = args.triangle_triplet_samples
            CONFIG["device"] = args.device
            if args.require_cuda:
                CONFIG["require_cuda"] = True
            if args.no_amp:
                CONFIG["amp"] = False
            CONFIG["patience"] = args.patience
            CONFIG["lr"] = args.lr
            CONFIG["ea_head_lr"] = args.ea_head_lr
            CONFIG["ea_loss_weight"] = args.ea_loss_weight
            CONFIG["ea_loss_start_epoch"] = args.ea_loss_start_epoch
            CONFIG["ea_select_weight"] = args.ea_select_weight
            CONFIG["ea_head_dropout"] = args.ea_head_dropout
            CONFIG["swa_start"] = args.swa_start
            if args.no_swa:
                CONFIG["swa_enabled"] = False
            CONFIG["save_dir"] = args.save_dir
            if args.geom_only:
                CONFIG["geom_only"] = True
            if args.ea_only:
                CONFIG["ea_only"] = True
            if args.backbone_ckpt is not None:
                CONFIG["backbone_ckpt"] = args.backbone_ckpt
            if getattr(args, "data_dir", None):
                CONFIG["data_dir"] = args.data_dir
        train_pipeline(CONFIG)
