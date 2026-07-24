"""PSI transition-state pipeline — Kaggle edition (P100 or T4 x2).

A single self-contained rewrite carrying ONLY the configuration the full pipeline
converged on. Every superseded branch is deleted rather than disabled, so there is
exactly one code path and nothing to configure:

  geometry            coordinate-native (EGNN seeded with the real Kabsch-aligned
                      R/P midpoint, supervised against the DFT TS coordinates).
                      The interpolation-prior GeometryHead and the MDS/eigh seed
                      are GONE — the geometry failure atlas attributed 45.9% of
                      failing validation reactions to `geom_head_interpolation_bound`
                      and only 0.2% to `mds_lossy`, and MDS cost 58% of the forward.
  reaction centre     GFN2-xTB Wiberg bond orders. The covalent-radius cutoff it
                      replaces misses 8,754 reactive atoms (22.0% of reactions get
                      a different reactive-atom set).
  uncertainty         heteroscedastic per-atom log-variance + Kendall-Gal loss
                      attenuation, feeding detached geometry-trust features to the
                      Ea head.
  Ea head             physics prior (linear BEP on dE_rxn + FiLM over the 28D R/P
                      descriptor and the continuous GFN2-xTB reaction-centre
                      descriptor) fused with 13 detached descriptors of the
                      PREDICTED TS -- where it sits along the reaction coordinate,
                      how asynchronous it is, how far the non-reacting scaffold is
                      strained off the R/P midpoint. Held out of the loss for the
                      first EA_START_EPOCH epochs, because none of those channels
                      means anything until the geometry is worth reading.
  batching            batches are length-bucketed and every atom axis is trimmed to
                      the batch's largest molecule. The EGNN is the forward cost and
                      it is quadratic in atom count; RGD1 averages 17.7 atoms
                      against MAX_ATOMS = 30, so ~64% of that work was padding.
  precision           fp16 AMP + GradScaler. Neither P100 (Pascal) nor T4 (Turing)
                      supports bf16, so fp16 is not a choice, it is the only
                      option. torch.compile is enabled only on sm_70+: Triton
                      cannot lower for P100, and attempting it fails at the first
                      kernel.
  parallelism         One GPU (P100) trains in-process with NO distributed group
                      at all -- initialising an NCCL group of size 1 is a real
                      hang/failure point in a Kaggle notebook. Multiple GPUs (T4
                      x2) use DDP via mp.spawn. Every collective is guarded
                      (all_reduce_sum), so the training loop is the same code
                      either way -- the only difference is whether a process group
                      exists to reduce across.

Deleted because coordinate-native makes them dead weight, not because they were
turned off: the coarse-distance aux loss and both triangle-inequality losses (a
distance matrix read off real coordinates satisfies the triangle inequality by
construction), the delta-clamp, MDS, coord-noise augmentation, the ea-only staging,
and the distance-rule spectator mask.

No config dict, no mode flags, no fallback branches, no try/except. Hyperparameters
are module constants below. Anything that cannot proceed correctly raises.

--------------------------------------------------------------------------------
RUNNING ON KAGGLE
--------------------------------------------------------------------------------
There are two ways to run this, and the stage commands differ between them.

  AS A FILE -- upload it, or attach it as a Dataset, then from a cell:

      !python psi_cloud_pipeline.py <stage>

    `%run` works too. IPython injects its own argv ("-f /root/.../kernel-abc.json"),
    which resolve_stage ignores rather than mistaking for a stage name.

  PASTED INTO A CELL -- no upload, no Dataset, no path. Run the cell; with no stage
    name to read, resolve_stage picks DEFAULT_STAGE and it trains. There is no file
    on disk, so `!python ...` has nothing to point at: run any OTHER stage by
    calling its function from a new cell, which the paste has already defined.

      stage_eval()        stage_bond_orders()

    Error messages work this out for themselves -- see how_to_run -- so whichever
    way you are running, what they print is what you can actually type.

Turn Internet ON: the first run streams RGD1_CHNO.h5 (~1.34 GB) and
DFT_reaction_info.csv from Figshare into /kaggle/working/RGD1_Dataset, then parses
them into samples_cache.pkl. Both are reused by every later session, so attach the
saved output as a Dataset and the download happens exactly once.

The simplest path is one command:

      !pip install tblite
      !python psi_cloud_pipeline.py train

On the first run this parses RGD1, builds the GFN2-xTB bond-order cache (a one-time
~1.5-2 h CPU precompute), then trains. Save the notebook output as a Kaggle Dataset
and attach it to later sessions -- every resume then finds both caches and skips
straight to training. Two facts make the auto-build worth understanding:

  * On a GPU session the bond-order build spends its ~1.5-2 h on CPU before the GPU
    does anything. To avoid burning GPU quota on it, run the `bond-orders` stage
    ONCE in a cheaper CPU-only session first, save that as a Dataset, and attach it:

        !pip install tblite
        !python psi_cloud_pipeline.py bond-orders     # GPU OFF, Internet ON

  * `train` auto-builds only when no cache is found (neither an attached dataset
    nor /kaggle/working/bond_orders_cache.pkl).

  Evaluation (GPU, minutes):

      !python psi_cloud_pipeline.py eval

    Scores psi_best.pt per reaction into results.json: distance MAE and coordinate
    RMSD as mean AND median, Ea error, and the chirality-flip rate. Separate from
    training on purpose -- a Kaggle session is cut long before the epoch budget
    runs out, so an evaluation that only fired after the last epoch would never run.

There is NO resume. `train` always starts from initialisation, and the only weights
written are /kaggle/working/psi_best.pt, re-saved whenever val_select improves.
Kaggle caps a session at ~9-12 h, so a run cut short keeps whatever psi_best.pt held
at that point and nothing else -- the epochs themselves are not recoverable and a
later session repeats them. `eval` scores psi_best.pt, so a cut run is still
measurable; it is just not continuable. (The data caches are unaffected: those are
separate files and every session reuses them.)

Per-epoch telemetry covers the two failure modes the legacy runs actually hit:
`lvFloor` is the fraction of atoms pinned at the log-variance floor (it reached
18% and was still climbing at epoch 143, meaning the NLL was partly being reduced
by shrinking sigma rather than by predicting better), and `flip` is the
chirality-flip rate, which is the only metric that can see an enantiomer error
because distance matrices are chirality-blind.

Data: RGD1_CHNO.h5 + DFT_reaction_info.csv are pulled from Figshare on first use
and cached under /kaggle/working/RGD1_Dataset, or picked up from any attached
Kaggle dataset.
"""

import os

# MUST precede numpy/tblite in the parent AND in every spawned worker (spawn
# re-imports this module, so module scope is the only place that works). Without
# it each worker's BLAS grabs every core: measured 2.0 s per xTB single point
# with 6 workers x 12 threads on 12 cores, against 0.276 s single-process.
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import sys
import json
import math
import time
import glob
import pickle
import urllib.request
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, Subset, Sampler

# =============================================================================
# Hyperparameters. Module constants, deliberately not a config dict: every value
# here is the one the full pipeline converged on, and a knob that is never turned
# is a branch that is never tested.
# =============================================================================
MAX_ATOMS = 30
TARGET_REACTIONS = 40_000
SKIP_NEGATIVE_EA = True
FRAGMENT_BOND_SCALE = 1.45
SPECTATOR_THRESHOLD = 0.15          # |D_R - D_P| above this = an "active" pair

# Encoder
N_GAUSSIANS = 32
GAUSS_START, GAUSS_STOP = 0.4, 6.0
ATOM_EMBED_DIM = 32
GRU_HIDDEN, GRU_LAYERS, GRU_DROPOUT = 128, 2, 0.1
ATTN_HEADS, ATTN_LAYERS, FF_DIM = 8, 3, 512
DROPOUT = 0.1

# EGNN coordinate refiner
EGNN_LAYERS, EGNN_HIDDEN = 8, 256
EGNN_COORD_CLAMP = 15.0

# Ea head
EA_LOSS_WEIGHT = 0.5
EA_SELECT_WEIGHT = 4.0              # Ea contribution to checkpoint selection
EA_HEAD_DROPOUT = 0.15
EA_HEAD_LR = 6e-4                   # sqrt-scaled with BATCH_SIZE, same as LR
EA_HEAD_WEIGHT_DECAY = 1e-3
# Epoch the Ea head enters the loss. Before it the run is geometry-only: the head
# reads where the PREDICTED TS sits along the reaction coordinate, and until the
# EGNN has moved off its R/P-midpoint seed that descriptor says "midpoint" for
# every reaction in the set. Fitting Ea to it first teaches a mapping that has to
# be unlearned once the geometry becomes real.
EA_START_EPOCH = 150

# Optimisation
# LR and BATCH_SIZE move together. 48 -> 192 was chosen for the P100, where
# torch.compile is unavailable (sm_60) and an eager step at 48 is dominated by
# kernel-launch and Python overhead rather than by the GPU: measured on the sweep
# below, an N=2 step costs the same as an N=30 one, so the batch is the only lever
# that amortises it. 334 -> 1031 samples/s, at 2.5 GB of the P100's 16 GB.
# LR is sqrt-scaled, not linearly: sqrt(192/48) * 1.5e-4.
LR = 3.0e-4
WEIGHT_DECAY = 1e-3
# In EPOCHS, and an epoch is now 177 steps rather than 708 -- so this is 1,770
# warmup steps against the 3,540 the old 5 epochs bought. Raise to 20 if the first
# epochs show grad_skips or a clip_rate pinned at 1.00.
WARMUP_EPOCHS = 10
GRAD_CLIP = 1.0
BATCH_SIZE = 192                    # PER RANK; global batch is 2x this on T4 x2
EPOCHS = 800
SWA_START = 450                     # also the cosine horizon
PATIENCE = 120
# 4 fits Kaggle's 4 vCPU on a single P100, where no second rank competes for them.
# On T4 x2 this is 4 per rank on the same 4 cores; drop it to 2 there.
NUM_WORKERS = 4
PREFETCH_FACTOR = 4
# Length bucketing: how many batches are drawn from one shuffled pool before it is
# sorted by atom count and cut up. Counted in BATCHES, not reactions, so the pool
# is always a whole number of batches -- a pool that did not divide evenly would
# leave a short batch at the end of EVERY pool for drop_last to discard, silently
# losing thousands of reactions per epoch. 8 x 192 keeps the pool at ~1,536
# reactions: wide enough that a batch's membership is redrawn every epoch, narrow
# enough that its members are still close in size.
BUCKET_POOL_BATCHES = 8

# Geometry loss: reaction-role reweighting
GEOM_HINGE_CROSS_WEIGHT = 3.0       # active <-> spectator cross pairs
GEOM_ACTIVE_PAIR_WEIGHT = 2.0       # active <-> active
GEOM_SPECTATOR_SPECTATOR_WEIGHT = 0.25
GEOM_MOVE_WEIGHT = 2.0              # movement-aware weight floor (under-shoot fix)
GEOM_HUBER_DELTA = 1.0
COORD_LOSS_WEIGHT = 1.0
PINN_WEIGHT = 0.2
STERIC_LOSS_WEIGHT = 1.0
STERIC_FLOOR_FRAC = 0.75
LOGVAR_CLAMP = 7.0                  # exp(7) ~ 1096 stays finite in fp16

# Risk-weighted loss
RISK_WEIGHT_ALPHA = 0.5
RISK_WEIGHT_MAX = 3.0
RISK_EA_LOSS_WEIGHT = 0.5
RISK_GEOM_LOSS_WEIGHT = 0.2

# xTB reaction centre
BO_CHANGE_THRESHOLD = 0.5           # |bo_R - bo_P| above this = reactive pair
BO_BONDED_MIN = 0.5

# Split
VAL_SPLIT = 0.15
SPLIT_SEED = 42
SPLIT_BINS = 5

# Log-variance histogram. The range is EXACTLY the clamp, so bin 0 is
# [-7.0, -6.9) and bin -1 is [6.9, 7.0] -- the two saturation boundaries. Widening
# it (say to +/-8) would leave bin 0 permanently empty and report pinned_lo = 0
# however hard the head is pressed against its floor, which is the single
# diagnostic this histogram exists to provide.
LOGVAR_HIST_BINS = 140
LOGVAR_HIST_MIN, LOGVAR_HIST_MAX = -LOGVAR_CLAMP, LOGVAR_CLAMP
# A reflected superposition fitting this much better than a proper rotation, on a
# structure that is not already near-perfect, means the predicted TS is the
# ENANTIOMER of the truth. Thresholds match the geometry failure atlas.
CHIRALITY_FLIP_RATIO = 0.7
CHIRALITY_MIN_RMSD = 0.2

WORK_DIR = "/kaggle/working" if os.path.isdir("/kaggle/working") else os.getcwd()
SAMPLE_CACHE = os.path.join(WORK_DIR, "samples_cache.pkl")
BOND_ORDER_CACHE = os.path.join(WORK_DIR, "bond_orders_cache.pkl")
BEST_CKPT = os.path.join(WORK_DIR, "psi_best.pt")
HISTORY_PATH = os.path.join(WORK_DIR, "training_history.json")
RESULTS_PATH = os.path.join(WORK_DIR, "results.json")
SPLIT_PATH = os.path.join(WORK_DIR, "split.json")
# `__file__` is undefined when this code runs from a notebook CELL rather than a
# file -- which is the common Kaggle case, because pasting the script into a cell
# needs no dataset, no upload and no path. Help text has to know which it is: with
# no file on disk there is nothing for `!python` to point at, and telling the user
# to run one is telling them to run something that cannot work. Every def has
# already executed into the notebook's globals by then, so the stage function IS
# the instruction, and `how_to_run` emits whichever form actually applies.
SCRIPT_NAME = os.path.basename(globals().get("__file__", "psi_cloud_pipeline.py"))
IS_SCRIPT = "__file__" in globals()


def how_to_run(stage):
    """The literal text to type in order to run `stage`, however this file loaded."""
    if IS_SCRIPT:
        return f"!python {SCRIPT_NAME} {stage}"
    return f"{STAGES[stage].__name__}()   # in a new cell; already defined by the paste"

# =============================================================================
# Chemistry tables
# =============================================================================
COVALENT_RADII = {'H': 0.31, 'C': 0.76, 'N': 0.71, 'O': 0.66, 'F': 0.57,
                  'S': 1.05, 'Cl': 1.02, 'Br': 1.20, 'I': 1.39, 'P': 1.07,
                  'Si': 1.11, 'B': 0.84}
PAULING_EN = {'H': 2.20, 'C': 2.55, 'N': 3.04, 'O': 3.44, 'F': 3.98,
              'S': 2.58, 'Cl': 3.16, 'Br': 2.96, 'I': 2.66, 'P': 2.19,
              'Si': 1.90, 'B': 2.04}
ATOMIC_NUMBER = {'H': 1, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'S': 16, 'Cl': 17,
                 'Br': 35, 'I': 53, 'P': 15, 'Si': 14, 'B': 5}
ATOMIC_MASS = {'H': 1.008, 'C': 12.011, 'N': 14.007, 'O': 15.999, 'F': 18.998,
               'S': 32.06, 'Cl': 35.45, 'Br': 79.904, 'I': 126.904, 'P': 30.974,
               'Si': 28.085, 'B': 10.811}
# Bond types whose reactions showed elevated validation Ea error.
RISK_BOND_TYPES = {tuple(sorted(p)) for p in
                   [("N", "N"), ("N", "O"), ("O", "O"), ("C", "N"), ("H", "N")]}

ATOM_PHYS_DIM = 3                   # electronegativity, Z, mass
ENERGY_FEAT_DIM = 28                # energetics + composition + angles + RC angles
BO_FEAT_DIM = 12                    # continuous Wiberg reaction-centre descriptors


def covalent_radius(a):
    return COVALENT_RADII.get(a, 0.76)


def electronegativity(a):
    return PAULING_EN.get(a, 2.55)


def atomic_number(a):
    return ATOMIC_NUMBER.get(a, 6)


def atomic_mass(a):
    return ATOMIC_MASS.get(a, 12.011)


def build_atom_vocab():
    return {a: i + 1 for i, a in enumerate(sorted(ATOMIC_NUMBER))}


# =============================================================================
# Geometry / graph utilities (numpy, dataset-build time only)
# =============================================================================
def compute_distance_matrix(coords):
    diff = coords[:, None, :] - coords[None, :, :]
    return np.sqrt((diff ** 2).sum(-1) + 1e-8).astype(np.float32)


def bond_adjacency(coords, atom_types, n, scale=FRAGMENT_BOND_SCALE):
    adj = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            cut = scale * (covalent_radius(atom_types[i]) + covalent_radius(atom_types[j]))
            if np.linalg.norm(coords[i] - coords[j]) <= cut:
                adj[i].append(j)
                adj[j].append(i)
    return adj


def bond_set_from_distances(D, atom_types, n, scale=FRAGMENT_BOND_SCALE):
    bonds = {}
    for i in range(n):
        for j in range(i + 1, n):
            cut = scale * (covalent_radius(atom_types[i]) + covalent_radius(atom_types[j]))
            if D[i, j] <= cut:
                bonds[(i, j)] = tuple(sorted((atom_types[i], atom_types[j])))
    return bonds


def connected_components(adj):
    seen, frags = set(), []
    for start in range(len(adj)):
        if start in seen:
            continue
        stack, frag = [start], []
        seen.add(start)
        while stack:
            node = stack.pop()
            frag.append(node)
            for nbr in adj[node]:
                if nbr not in seen:
                    seen.add(nbr)
                    stack.append(nbr)
        frags.append(sorted(frag))
    return sorted(frags, key=lambda f: (f[0], len(f)))


def find_fragments(coords, atom_types, n, scale=FRAGMENT_BOND_SCALE):
    return connected_components(bond_adjacency(coords, atom_types, n, scale))



def kabsch(P, Q):
    """Rotate+translate P onto Q. Reflection-corrected without a second SVD."""
    Pc = P - P.mean(axis=0)
    Qc = Q - Q.mean(axis=0)
    V, _, W = np.linalg.svd(Pc.T @ Qc)
    if np.linalg.det(V @ W) < 0.0:
        V[:, -1] *= -1.0
    return Pc @ (V @ W) + Q.mean(axis=0)


def global_kabsch_align(c_R, c_P, n):
    """ONE rigid transform for the whole reactant, never per fragment.

    Per-fragment alignment translates every reactant fragment onto its PRODUCT
    fragment's centroid, overwriting the reactant's inter-fragment arrangement
    (separation, relative orientation, approach direction) with the product's.
    For coordinate-native geometry that is fatal: the seed would carry zero
    reactant information about how the fragments approach, and
    `cross_fragment_orientation` is 31.9% of failing validation reactions. One
    global transform fits each fragment worse but PRESERVES the degree of freedom
    the model is supposed to predict. Refining a coarse seed is the EGNN's job;
    recovering deleted information is not.
    """
    out = c_R.copy()
    out[:n] = kabsch(c_R[:n], c_P[:n]) if n >= 2 else c_P[:n]
    return out


def kabsch_align_fragments(c_R, c_P, atom_types, n, scale=FRAGMENT_BOND_SCALE):
    """Per-fragment alignment. ONLY for build_energy_features, never for the seed."""
    out = c_R.copy()
    frags_R = find_fragments(c_R, atom_types, n, scale)
    frags_P = find_fragments(c_P, atom_types, n, scale)
    frags = frags_P if len(frags_P) > len(frags_R) else frags_R
    for frag in frags:
        idx = np.array(frag, dtype=np.int64)
        out[idx] = kabsch(c_R[idx], c_P[idx]) if len(idx) >= 2 else c_P[idx]
    return out


def coordinate_seed(c_R, c_P, n):
    """The R/P midpoint the EGNN starts from, in the product frame."""
    c_R_aligned = global_kabsch_align(c_R, c_P, n)
    c_I = np.zeros_like(c_P)
    c_I[:n] = 0.5 * (c_R_aligned[:n] + c_P[:n])
    return c_I


# =============================================================================
# 28D energy descriptor for the Ea head
# =============================================================================
def _stats4(values):
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return 0.0, 0.0, 0.0, 0.0
    return float(arr.mean()), float(arr.std()), float(arr.min()), float(arr.max())


def bond_angles(coords, atom_types, n, scale=FRAGMENT_BOND_SCALE):
    """Every bonded triplet (i, j, k) with central atom j and i<k -> angle in degrees."""
    adj = bond_adjacency(coords, atom_types, n, scale)
    angles = {}
    for j in range(n):
        nbrs = sorted(adj[j])
        for a in range(len(nbrs)):
            for b in range(a + 1, len(nbrs)):
                i, k = nbrs[a], nbrs[b]
                v1, v2 = coords[i] - coords[j], coords[k] - coords[j]
                n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
                if n1 < 1e-9 or n2 < 1e-9:
                    continue
                cos = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
                angles[(i, j, k)] = float(np.degrees(np.arccos(cos)))
    return angles


def _best_dihedral(coords, atom_types, n, bond_pair, adj):
    """Representative dihedral i-a-b-j across bond (a,b), heaviest neighbour each side.

    Returned as (cos, sin) so the encoding is periodicity-safe.
    """
    a, b = bond_pair
    nbrs_a = [x for x in adj[a] if x != b and x < n]
    nbrs_b = [x for x in adj[b] if x != a and x < n]
    if not nbrs_a or not nbrs_b:
        return 0.0, 0.0
    i = max(nbrs_a, key=lambda x: atomic_mass(atom_types[x]))
    j = max(nbrs_b, key=lambda x: atomic_mass(atom_types[x]))
    b1, b2, b3 = coords[a] - coords[i], coords[b] - coords[a], coords[j] - coords[b]
    n1, n2 = np.cross(b1, b2), np.cross(b2, b3)
    n1n, n2n = np.linalg.norm(n1), np.linalg.norm(n2)
    if n1n < 1e-9 or n2n < 1e-9:
        return 1.0, 0.0
    n1, n2 = n1 / n1n, n2 / n2n
    b2h = b2 / max(np.linalg.norm(b2), 1e-9)
    return (float(np.clip(np.dot(n1, n2), -1.0, 1.0)),
            float(np.clip(np.dot(np.cross(n1, n2), b2h), -1.0, 1.0)))


def _pyramidalization(coords, center, neighbors):
    """Mean out-of-plane angle (deg) at a trigonal centre — the sp2<->sp3 distortion."""
    if len(neighbors) < 3:
        return 0.0
    nbrs = sorted(neighbors)[:3]
    v1 = coords[nbrs[0]] - coords[center]
    v2 = coords[nbrs[1]] - coords[center]
    v3 = coords[nbrs[2]] - coords[center]
    normal = np.cross(v1 - v2, v1 - v3)
    nn = np.linalg.norm(normal)
    if nn < 1e-9:
        return 0.0
    normal /= nn
    angles = []
    for v in (v1, v2, v3):
        vn = np.linalg.norm(v)
        if vn < 1e-9:
            continue
        angles.append(float(np.degrees(np.arcsin(
            np.clip(abs(np.dot(v / vn, normal)), 0.0, 1.0)))))
    return float(np.mean(angles)) if angles else 0.0


def _rc_angle_features(cR, cP, atom_types, n, scale=FRAGMENT_BOND_SCALE):
    """8D angular description of the reacting atoms specifically.

    The rest of the descriptor is whole-molecule statistics, which say nothing
    about the geometry AT the forming/breaking bonds.
    """
    D_R = compute_distance_matrix(cR.astype(np.float32))[:n, :n].astype(np.float64)
    D_P = compute_distance_matrix(cP.astype(np.float32))[:n, :n].astype(np.float64)
    bonds_R = bond_set_from_distances(D_R, atom_types, n, scale)
    bonds_P = bond_set_from_distances(D_P, atom_types, n, scale)
    formed = set(bonds_P) - set(bonds_R)
    broken = set(bonds_R) - set(bonds_P)
    rc_atoms = {i for pair in (formed | broken) for i in pair}

    all_R = bond_angles(cR, atom_types, n, scale)
    all_P = bond_angles(cP, atom_types, n, scale)
    rc_R = {k: v for k, v in all_R.items() if k[1] in rc_atoms}
    rc_P = {k: v for k, v in all_P.items() if k[1] in rc_atoms}
    mean_R = float(np.mean([np.cos(np.radians(a)) for a in rc_R.values()])) if rc_R else 0.0
    mean_P = float(np.mean([np.cos(np.radians(a)) for a in rc_P.values()])) if rc_P else 0.0
    common = set(rc_R) & set(rc_P)
    change_max = float(max(abs(rc_R[t] - rc_P[t]) for t in common)) if common else 0.0

    adj_R = bond_adjacency(cR, atom_types, n, scale)
    adj_P = bond_adjacency(cP, atom_types, n, scale)
    # Forming bonds are measured in P (where the bond exists), broken bonds in R.
    fc = [_best_dihedral(cP, atom_types, n, p, adj_P) for p in formed]
    bc = [_best_dihedral(cR, atom_types, n, p, adj_R) for p in broken]
    f_cos = float(np.mean([x[0] for x in fc])) if fc else 0.0
    f_sin = float(np.mean([x[1] for x in fc])) if fc else 0.0
    b_cos = float(np.mean([x[0] for x in bc])) if bc else 0.0
    b_sin = float(np.mean([x[1] for x in bc])) if bc else 0.0

    pyr = [0.5 * (_pyramidalization(cR, i, [x for x in adj_R[i] if x < n])
                  + _pyramidalization(cP, i, [x for x in adj_P[i] if x < n]))
           for i in rc_atoms]
    return np.array([mean_R, mean_P, change_max, f_cos, f_sin, b_cos, b_sin,
                     float(np.mean(pyr)) if pyr else 0.0], dtype=np.float32)


def build_energy_features(atom_types, n, c_R_aligned, c_P, e_r, e_p,
                          scale=FRAGMENT_BOND_SCALE):
    """28D reaction descriptor, computable from R and P alone (no TS leakage).

      [0:10]  energetics + composition
      [10:20] bond-angle statistics for R, P, and their change
      [20:28] reaction-centre angle features
    """
    cR = np.asarray(c_R_aligned, dtype=np.float64)[:n]
    cP = np.asarray(c_P, dtype=np.float64)[:n]
    types = list(atom_types[:n])
    diff_norms = np.linalg.norm(cR - cP, axis=1)
    ang_R, ang_P = bond_angles(cR, types, n, scale), bond_angles(cP, types, n, scale)
    aR = _stats4(ang_R.values())
    aP = _stats4(ang_P.values())
    common = set(ang_R) & set(ang_P)
    changes = np.array([abs(ang_R[t] - ang_P[t]) for t in common]) if common else np.zeros(1)
    rc = _rc_angle_features(cR, cP, types, n, scale)
    return np.array([
        abs(e_r - e_p), e_p - e_r,
        float(diff_norms.mean()), float(diff_norms.std()), float(diff_norms.max()),
        float(n),
        float(sum(1 for t in types if t == 'C')), float(sum(1 for t in types if t == 'H')),
        float(sum(1 for t in types if t == 'N')), float(sum(1 for t in types if t == 'O')),
        aR[0], aR[1], aR[2], aR[3], aP[0], aP[1], aP[2], aP[3],
        float(changes.mean()) if common else 0.0,
        float(changes.max()) if common else 0.0,
        rc[0], rc[1], rc[2], rc[3], rc[4], rc[5], rc[6], rc[7],
    ], dtype=np.float32)


def build_atom_physical_features(atom_types, n, max_atoms):
    """Per-atom [EN, Z, mass], zero-padded. Attached to nodes so the encoder knows
    WHICH atom carries which property, not just global statistics."""
    feats = np.zeros((max_atoms, ATOM_PHYS_DIM), dtype=np.float32)
    for i in range(n):
        t = atom_types[i]
        feats[i] = [electronegativity(t), float(atomic_number(t)), atomic_mass(t)]
    return feats


def reaction_risk_features(D_R, D_P, atom_types, n, max_atoms):
    """Complexity/chemistry flags for the reaction classes with elevated Ea error."""
    bonds_R = bond_set_from_distances(D_R, atom_types, n)
    bonds_P = bond_set_from_distances(D_P, atom_types, n)
    formed = set(bonds_P) - set(bonds_R)
    broken = set(bonds_R) - set(bonds_P)
    risk_pair_mask = np.zeros((max_atoms, max_atoms), dtype=np.float32)
    risky_types = set()
    for i in range(n):
        for j in range(i + 1, n):
            btype = tuple(sorted((atom_types[i], atom_types[j])))
            active = abs(D_R[i, j] - D_P[i, j]) > SPECTATOR_THRESHOLD
            fb = (i, j) in formed or (i, j) in broken
            if (active or fb) and btype in RISK_BOND_TYPES:
                risky_types.add(btype)
            if active or fb:
                risk_pair_mask[i, j] = risk_pair_mask[j, i] = 1.0
    formed_n, broken_n = len(formed), len(broken)
    changed = formed_n + broken_n
    risky_chem = float(len(risky_types) > 0)
    return {
        "formed_bonds": formed_n, "broken_bonds": broken_n, "changed_bonds": changed,
        "complexity_flag": float(changed >= 4 or broken_n >= 3),
        "risky_chem_flag": risky_chem,
        "risk_score": float(changed >= 4 or broken_n >= 3) + risky_chem,
        # Margin penalty: quadratic outside the 1..3 changed-bond "safe" band.
        "risk_penalty": max(0.0, changed - 3.0) ** 2 + max(0.0, 1.0 - changed) ** 2
                        + 0.5 * risky_chem,
        "risk_pair_mask": risk_pair_mask,
    }


# =============================================================================
# RGD1 data resolution — Kaggle input, then Figshare
# =============================================================================
RGD1_FILES = {
    "RGD1_CHNO.h5": "https://ndownloader.figshare.com/files/38170323",
    "DFT_reaction_info.csv": "https://ndownloader.figshare.com/files/40273231",
}
RGD1_MIN_SIZES = {"RGD1_CHNO.h5": 1_200_000_000, "DFT_reaction_info.csv": 20_000_000}


def _download(url, dest):
    """Stream to `dest`, atomic via .part so an interrupted session leaves no
    half-file that a later run would mistake for a complete download."""
    tmp = dest + ".part"
    print(f"[data] downloading {os.path.basename(dest)}", flush=True)
    req = urllib.request.Request(url, headers={"User-Agent": "psi-cloud"})
    with urllib.request.urlopen(req) as resp, open(tmp, "wb") as out:
        total = int(resp.headers.get("Content-Length", 0))
        done, last = 0, -5
        while True:
            buf = resp.read(1 << 20)
            if not buf:
                break
            out.write(buf)
            done += len(buf)
            pct = int(done * 100 / total) if total else 0
            if pct >= last + 5:
                print(f"  {pct}% ({done/1e6:.0f}/{total/1e6:.0f} MB)", flush=True)
                last = pct
    os.replace(tmp, dest)


def resolve_rgd1():
    """(h5_path, csv_path) from an attached Kaggle dataset, else Figshare."""
    found = {}
    for fname in RGD1_FILES:
        hits = glob.glob(f"/kaggle/input/**/{fname}", recursive=True)
        found[fname] = hits[0] if hits else None
    if all(found.values()):
        print(f"[data] using attached Kaggle dataset:\n       " +
              "\n       ".join(found.values()))
        return found["RGD1_CHNO.h5"], found["DFT_reaction_info.csv"]

    cache = os.path.join(WORK_DIR, "RGD1_Dataset")
    os.makedirs(cache, exist_ok=True)
    paths = {}
    for fname, url in RGD1_FILES.items():
        dest = os.path.join(cache, fname)
        big_enough = os.path.exists(dest) and os.path.getsize(dest) >= RGD1_MIN_SIZES[fname]
        if not big_enough:
            _download(url, dest)
        print(f"[data] {fname}: {os.path.getsize(dest)/1e6:.0f} MB")
        paths[fname] = dest
    return paths["RGD1_CHNO.h5"], paths["DFT_reaction_info.csv"]


# =============================================================================
# Sample construction
# =============================================================================
def _find_cache(filename, working_path):
    """An existing on-disk cache, or None. Looks at an ATTACHED Kaggle dataset
    (/kaggle/input) first, then the working dir. This is what makes a saved and
    re-attached cache actually get reused across sessions instead of rebuilt:
    /kaggle/working does not persist, so a cache from a prior session only comes
    back mounted read-only under /kaggle/input."""
    hits = glob.glob(f"/kaggle/input/**/{filename}", recursive=True)
    if hits:
        return hits[0]
    return working_path if os.path.exists(working_path) else None


def find_sample_cache():
    return _find_cache("samples_cache.pkl", SAMPLE_CACHE)


def build_samples():
    """Parse RGD1 into training samples, cached to disk after the first pass.

    Reactions with an empty reaction centre under the covalent-radius rule are
    dropped here: the Ea head pools over the forming/breaking atoms and refuses to
    spread an empty mask over the whole molecule.
    """
    cached = find_sample_cache()
    if cached is not None:
        # Reused as-is: no re-parse, and resolve_rgd1() is never called, so the
        # 1.34 GB RGD1 download is skipped entirely when the samples already exist.
        with open(cached, "rb") as fh:
            blob = pickle.load(fh)
        print(f"[data] loaded {len(blob['samples'])} samples from {cached}")
        return blob["samples"], blob["atom_vocab"]

    import h5py
    import pandas as pd
    h5_path, csv_path = resolve_rgd1()
    print(f"[data] parsing {h5_path}", flush=True)
    df = pd.read_csv(csv_path)
    df = df[df["DE_F"].notna()]
    atom_vocab = build_atom_vocab()
    inv_z = {v: k for k, v in ATOMIC_NUMBER.items()}
    samples, skipped_no_rc, t0 = [], 0, time.time()

    with h5py.File(h5_path, "r") as f:
        for _, row in df.iterrows():
            if len(samples) >= TARGET_REACTIONS:
                break
            rxn_id = str(row["channel"])
            if rxn_id not in f:
                continue
            grp = f[rxn_id]
            elements = grp["elements"][:]
            n = len(elements)
            if n > MAX_ATOMS:
                continue
            ea = float(row["DE_F"])
            if SKIP_NEGATIVE_EA and ea < 0:
                continue
            dh = float(row["DH"])

            atom_types = [inv_z.get(int(z), "C") for z in elements]
            atom_ids = np.zeros(MAX_ATOMS, dtype=np.int64)
            for i, a in enumerate(atom_types):
                atom_ids[i] = atom_vocab.get(a, 0)
            mask = np.zeros(MAX_ATOMS, dtype=np.float32)
            mask[:n] = 1.0

            c_R = np.zeros((MAX_ATOMS, 3), dtype=np.float32)
            c_P = np.zeros((MAX_ATOMS, 3), dtype=np.float32)
            c_TS = np.zeros((MAX_ATOMS, 3), dtype=np.float32)
            c_R[:n], c_P[:n], c_TS[:n] = grp["RG"][:], grp["PG"][:], grp["TSG"][:]

            D_R = compute_distance_matrix(c_R)
            D_P = compute_distance_matrix(c_P)
            risk = reaction_risk_features(D_R, D_P, atom_types, n, MAX_ATOMS)
            if risk["changed_bonds"] == 0:
                skipped_no_rc += 1
                continue

            # Per-fragment alignment ONLY here: the descriptor wants each fragment's
            # internal geometry compared like-for-like. The EGNN seed uses the
            # global alignment instead (see global_kabsch_align).
            c_R_frag = kabsch_align_fragments(c_R, c_P, atom_types, n)
            samples.append({
                "rxn_id": rxn_id, "n_atoms": n, "atom_types": atom_types,
                "c_R": c_R, "c_P": c_P, "c_TS": c_TS,
                "c_I": coordinate_seed(c_R, c_P, n),
                "atom_ids": atom_ids, "mask": mask,
                "D_R": D_R, "D_P": D_P, "D_TS": compute_distance_matrix(c_TS),
                "Ea_raw": ea, "de_rxn_raw": dh,
                "energy_feats_raw": build_energy_features(
                    atom_types, n, c_R_frag, c_P, 0.0, dh),
                "atom_phys_raw": build_atom_physical_features(atom_types, n, MAX_ATOMS),
                "risk_pair_mask": risk["risk_pair_mask"],
                "risk_score": risk["risk_score"],
                "risk_penalty": risk["risk_penalty"],
                "complexity_flag": risk["complexity_flag"],
                "risky_chem_flag": risk["risky_chem_flag"],
                "formed_bonds": risk["formed_bonds"],
                "broken_bonds": risk["broken_bonds"],
            })
            if len(samples) % 2000 == 0:
                print(f"  {len(samples)}/{TARGET_REACTIONS} "
                      f"({time.time()-t0:.0f}s)", flush=True)

    print(f"[data] built {len(samples)} samples; "
          f"skipped {skipped_no_rc} with an empty reaction centre")
    with open(SAMPLE_CACHE, "wb") as fh:
        pickle.dump({"samples": samples, "atom_vocab": atom_vocab}, fh,
                    protocol=pickle.HIGHEST_PROTOCOL)
    return samples, atom_vocab


# =============================================================================
# Stage 1: GFN2-xTB Wiberg bond orders
# =============================================================================
BOHR = 1.8897259886


def _bond_orders_one(args):
    """One reaction -> (rxn_id, {n, bo_R, bo_P}) or (rxn_id, {"error": ...}).

    Only R and P are computed. TS bond orders are deliberately NOT cached: the TS
    is the prediction target, so feeding TS-derived quantities to the model is
    label leakage.

    This is the ONE deliberate try/except in the file. GFN2-xTB fails to converge
    on a handful of geometries (measured: 6 of 40,000) and there is no way to ask
    in advance, so without it a single unavoidable failure destroys a ~2 h
    precompute at an arbitrary point. Recording the failure is not a fallback:
    the reaction is EXCLUDED downstream by filter_to_bond_orders, never patched up
    with a substitute. Returning a zero matrix here would read as "every bond
    broke" and silently poison the reaction centre for that reaction.
    """
    from tblite.interface import Calculator
    rxn_id, numbers, c_R, c_P, n = args
    out = []
    try:
        for coords in (c_R, c_P):
            # RGD1 CHNO is neutral and closed-shell throughout; stated explicitly
            # rather than left to a library default.
            calc = Calculator("GFN2-xTB", numbers, coords[:n] * BOHR, charge=0, uhf=0)
            calc.set("verbosity", 0)
            bo = np.asarray(calc.singlepoint().get("bond-orders"))
            # tblite can return [n, n, nspin]. Summing over spin is required, not
            # cosmetic: leaving it 3-D makes every downstream [n, n] index wrong,
            # and this is a ~2 h precompute to discover that at the end of.
            bo = bo.sum(axis=2) if bo.ndim == 3 else bo
            out.append(bo.astype(np.float16))
    except Exception as exc:
        return rxn_id, {"error": f"{type(exc).__name__}: {exc}"}
    return rxn_id, {"n": n, "bo_R": out[0], "bo_P": out[1]}


def _require_tblite():
    """Import tblite, or raise the one-line install command. Checked ONCE in the
    parent so a missing package fails immediately and legibly, instead of every
    worker throwing ImportError, being recorded as an SCF 'failure', and the run
    ending with the misleading 'left no usable reactions'."""
    try:
        import tblite.interface  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "GFN2-xTB bond orders need the 'tblite' package, which is not "
            "installed. In a Kaggle cell run:\n    !pip install tblite\nthen "
            "re-run. (tblite is CPU-only; it is not needed once the cache exists.)"
        ) from exc


def find_bond_order_cache():
    """Path to an existing bond-order cache (attached Kaggle dataset first, then
    the working dir), or None if it has not been built yet."""
    return _find_cache("bond_orders_cache.pkl", BOND_ORDER_CACHE)


def stage_bond_orders():
    """Precompute the Wiberg bond orders for every reaction and cache them.

    A fitted bond-order(length) curve was tried first and rejected: calibrated
    from mode pairing alone it flagged 76.5% of pairs whose length changed by a
    mere 0.05-0.10 A -- ordinary conformational relaxation -- as order changes.
    GFN2-xTB reads the order off the wavefunction, so there is nothing to calibrate.

    ~0.28 s per single point x 2 geometries x 40k reactions, hence the process
    pool and the on-disk cache: pay it once per dataset, never per run.

    Idempotent: if a cache already exists (an attached Kaggle dataset, or a build
    from an earlier session in this working dir) it is left untouched and the
    ~1.5-2 h recompute is skipped. Delete the file to force a rebuild -- e.g. after
    changing BO_CHANGE_THRESHOLD, the only parameter that alters what it contains.
    """
    existing = find_bond_order_cache()
    if existing is not None:
        print(f"[xtb] cache already present at {existing}; skipping rebuild. "
              f"Delete it to recompute.")
        return
    _require_tblite()
    samples, _ = build_samples()
    n_workers = max(1, os.cpu_count() or 1)
    print(f"[xtb] {len(samples)} reactions on {n_workers} workers", flush=True)

    jobs = [(s["rxn_id"],
             np.array([atomic_number(a) for a in s["atom_types"]], dtype=np.int64),
             s["c_R"].astype(np.float64), s["c_P"].astype(np.float64), s["n_atoms"])
            for s in samples]

    orders, failures, t0 = {}, {}, time.time()
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        for k, (rxn_id, rec) in enumerate(pool.map(_bond_orders_one, jobs, chunksize=16), 1):
            # Failures are collected, never written into `orders`. A reaction
            # absent from `orders` is dropped by filter_to_bond_orders, which is
            # the correct outcome; an entry with garbage in it would not be.
            target = failures if "error" in rec else orders
            target[rxn_id] = rec.get("error", rec)
            if k % 500 == 0:
                rate = k / (time.time() - t0)
                print(f"  {k}/{len(jobs)}  {rate:.1f} rxn/s  "
                      f"eta {(len(jobs)-k)/rate/60:.0f} min  "
                      f"{len(failures)} SCF failure(s)", flush=True)

    with open(BOND_ORDER_CACHE, "wb") as fh:
        pickle.dump({"meta": {"method": "GFN2-xTB", "n_reactions": len(orders),
                              "n_failures": len(failures)},
                     "orders": orders, "failures": failures},
                    fh, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[xtb] wrote {BOND_ORDER_CACHE} ({len(orders)} reactions, "
          f"{len(failures)} SCF failures, {(time.time()-t0)/60:.0f} min)")
    print("[xtb] Save this notebook's output as a Kaggle Dataset and attach it to "
          "your training sessions.")


def load_bond_orders():
    """Load the cached bond orders. `train` builds the cache before calling this
    (see stage_train), so an absence here means a direct `eval` on a session that
    never trained -- point at how to produce it rather than the covalent-radius
    cutoff, which is a different reaction-centre definition."""
    path = find_bond_order_cache()
    if path is None:
        raise FileNotFoundError(
            f"No bond-order cache found (looked under /kaggle/input and at "
            f"{BOND_ORDER_CACHE}). Build it with:\n"
            f"    !pip install tblite\n"
            f"    {how_to_run('bond-orders')}\n"
            "or just run `train`, which builds it automatically on first use. "
            "Refusing to substitute the covalent-radius cutoff, which is a "
            "different reaction-centre definition."
        )
    with open(path, "rb") as fh:
        orders = pickle.load(fh)["orders"]
    print(f"[xtb] loaded bond orders for {len(orders)} reactions from {path}")
    return orders


def bond_order_masks(bo_R, bo_P, n):
    """(reactive_atom [MAX_ATOMS], spectator_pair [N, N], reactive_pair [N, N]), bool.

    reactive pair : |bo_R - bo_P| > BO_CHANGE_THRESHOLD
    reactive atom : incident on any reactive pair
    spectator pair: bonded in BOTH R and P and not reactive — the only pairs whose
                    true TS distance actually sits near the R/P midpoint (measured
                    |D_TS - D_I| = 0.0143 A, against 0.1218 A for the distance rule).

    The reactive pair mask used to be a local of this function. It is returned now
    because the Ea head measures reaction-coordinate progress ON those pairs (see
    PSI.geom_trust): the atom-level mask cannot say WHICH bond a reactive atom is
    forming, and a reactive atom in a three-centre TS is on two of them.

    Bool, not float32: these are memoised per sample in every persistent worker,
    where the [30, 30] spectator mask costs 3,600 B/sample as float32 (~123 MB per
    worker over 34k samples) against 900 B as bool. Every consumer casts anyway.
    """
    d = np.abs(bo_R.astype(np.float32) - bo_P.astype(np.float32))
    off = ~np.eye(n, dtype=bool)
    reactive = (d > BO_CHANGE_THRESHOLD) & off
    rc_atom = np.zeros(MAX_ATOMS, dtype=bool)
    rc_atom[:n] = reactive.any(axis=1)
    bonded_both = ((bo_R.astype(np.float32) > BO_BONDED_MIN)
                   & (bo_P.astype(np.float32) > BO_BONDED_MIN) & off)
    spec = np.zeros((MAX_ATOMS, MAX_ATOMS), dtype=bool)
    spec[:n, :n] = bonded_both & ~reactive
    react = np.zeros((MAX_ATOMS, MAX_ATOMS), dtype=bool)
    react[:n, :n] = reactive
    return rc_atom, spec, react


def bond_order_features(bo_R, bo_P, n):
    """Continuous GFN2-xTB reaction-centre descriptors -> [BO_FEAT_DIM] float32.

    bond_order_masks throws the MAGNITUDES away: every pair over the 0.5 threshold
    becomes one identical bit. A half-shifted pi bond and a fully severed sigma
    bond are the same reaction centre to that mask and are not the same barrier, so
    the Ea head gets the numbers themselves — total bond reorganisation, how it
    splits between formation and cleavage, how evenly it is spread over the
    reacting bonds, and how strong the bonds being broken were to begin with.

    R and P only. TS bond orders are never computed at all (see _bond_orders_one),
    so there is nothing here that could leak the target.

    Upper triangle, not the full matrix: a Wiberg matrix is symmetric, and counting
    both halves would double every sum and report twice the chemical pair count.
    """
    r = bo_R.astype(np.float32)
    p = bo_P.astype(np.float32)
    off = ~np.eye(n, dtype=bool)
    delta = (p - r) * off
    # filter_to_bond_orders drops every reaction with an empty reaction centre, so
    # `tri` always has at least one entry and the reductions below are defined.
    tri = np.triu((np.abs(delta) > BO_CHANGE_THRESHOLD) & off, 1)
    mag = np.abs(delta)[tri]
    signed = delta[tri]
    valence = np.abs(r.sum(axis=1) - p.sum(axis=1))[:n]
    return np.array([
        float(tri.sum()),
        float(mag.sum()), float(mag.mean()), float(mag.std()),
        float(mag.max()), float(mag.min()),
        float(signed[signed > 0.0].sum()), float(-signed[signed < 0.0].sum()),
        float(r[tri].mean()), float(p[tri].mean()),
        float(valence.max()), float(valence.mean()),
    ], dtype=np.float32)


def filter_to_bond_orders(samples, orders):
    """Drop every reaction the xTB reaction centre cannot describe.

    Two disjoint reasons, both fatal later if left in:
      * no cached bond orders — GFN2-xTB fails to converge on a handful of
        geometries (6 of 40,000 measured).
      * an EMPTY xTB reaction centre — 76 of 39,994 at threshold 0.5. The sample
        builder's empty-RC check uses the DISTANCE rule, and that guarantee does
        NOT carry over to a different definition. These reach the Ea head's
        empty-RC guard, which raises: a crash partway through epoch 1, not a
        silent degradation. Measured by threshold: 0.3 -> 0, 0.5 -> 76, 0.7 -> 337.
    """
    def empty_rc(rec):
        d = np.abs(rec["bo_R"].astype(np.float32) - rec["bo_P"].astype(np.float32))
        return not ((d > BO_CHANGE_THRESHOLD) & ~np.eye(rec["n"], dtype=bool)).any()

    missing = [s["rxn_id"] for s in samples if s["rxn_id"] not in orders]
    empty = [s["rxn_id"] for s in samples
             if s["rxn_id"] in orders and empty_rc(orders[s["rxn_id"]])]
    drop = set(missing) | set(empty)
    keep = [s for s in samples if s["rxn_id"] not in drop]
    print(f"[xtb] dropped {len(drop)} reaction(s): {len(missing)} without bond "
          f"orders, {len(empty)} with an empty xTB reaction centre; {len(keep)} remain")
    if not keep:
        raise RuntimeError("The bond-order cache does not match this sample set.")
    return keep, sorted(drop)


# =============================================================================
# Split and normalisation
# =============================================================================
def make_split(samples):
    """Stratified train/val split over (Ea bin, size bin, changed-bond bin, risk)."""
    n_total = len(samples)
    n_val = min(max(1, int(round(n_total * VAL_SPLIT))), n_total - 1)
    rng = np.random.default_rng(SPLIT_SEED)
    ea = np.array([s["Ea_raw"] for s in samples], dtype=np.float64)
    edges = np.unique(np.quantile(ea, np.linspace(0, 1, SPLIT_BINS + 1)[1:-1]))
    strata = {}
    for i, s in enumerate(samples):
        key = (int(np.searchsorted(edges, s["Ea_raw"], side="right")),
               min(int(s["n_atoms"] // 5), 6),
               min(int(s["formed_bonds"] + s["broken_bonds"]), 4),
               int(s["risk_score"] > 0.0))
        strata.setdefault(key, []).append(i)
    train_idx, val_idx = [], []
    for key in sorted(strata):
        group = np.array(strata[key], dtype=np.int64)
        rng.shuffle(group)
        k = 0 if len(group) <= 1 else min(int(round(len(group) * VAL_SPLIT)), len(group) - 1)
        val_idx.extend(group[:k].tolist())
        train_idx.extend(group[k:].tolist())
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    assert not (set(train_idx) & set(val_idx)), "train/val leakage"
    print(f"[split] {len(train_idx)} train / {len(val_idx)} val "
          f"(requested {n_val} val)")
    return train_idx, val_idx


def compute_normalization(samples, indices, orders):
    """z-score statistics from the TRAIN split only."""
    aphys = np.stack([samples[i]["atom_phys_raw"] for i in indices])
    real = aphys.reshape(-1, ATOM_PHYS_DIM)
    real = real[real.any(axis=1)]                       # padding rows would skew it
    aphys_mean, aphys_std = real.mean(0), real.std(0)
    aphys_std[aphys_std < 1e-6] = 1.0
    ea = np.array([samples[i]["Ea_raw"] for i in indices], dtype=np.float64)
    de = np.array([samples[i]["de_rxn_raw"] for i in indices], dtype=np.float64)
    ef = np.stack([samples[i]["energy_feats_raw"] for i in indices]).astype(np.float32)
    ef_std = ef.std(0)
    ef_std[ef_std < 1e-6] = 1.0
    bo = np.stack([bond_order_features(orders[samples[i]["rxn_id"]]["bo_R"],
                                       orders[samples[i]["rxn_id"]]["bo_P"],
                                       samples[i]["n_atoms"]) for i in indices])
    bo_std = bo.std(0)
    bo_std[bo_std < 1e-6] = 1.0
    stats = {
        "aphys_mean": aphys_mean.astype(np.float32), "aphys_std": aphys_std.astype(np.float32),
        "ea_mean": float(ea.mean()), "ea_std": max(float(ea.std()), 1e-6),
        "de_rxn_mean": float(de.mean()), "de_rxn_std": max(float(de.std()), 1e-6),
        "efeat_mean": ef.mean(0), "efeat_std": ef_std,
        "bo_mean": bo.mean(0), "bo_std": bo_std,
    }
    print(f"[norm] Ea {stats['ea_mean']:.2f} +/- {stats['ea_std']:.2f} kcal/mol")
    return stats


# =============================================================================
# Dataset
# =============================================================================
class ReactionDataset(Dataset):
    """Prebuilt samples -> tensors, memoised after first touch.

    Everything per-sample is deterministic (there is no augmentation), so an item
    is built once per worker and reused for every later epoch.
    """

    def __init__(self, samples, stats, orders):
        self.samples = samples
        self.stats = stats
        self.orders = orders
        self.cache = {}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        hit = self.cache.get(idx)
        if hit is not None:
            return hit
        s = self.samples[idx]
        n = s["n_atoms"]
        st = self.stats
        rec = self.orders[s["rxn_id"]]
        D_R = torch.from_numpy(s["D_R"])
        D_P = torch.from_numpy(s["D_P"])
        rc_atom, spec, react = bond_order_masks(rec["bo_R"], rec["bo_P"], n)
        bo = (bond_order_features(rec["bo_R"], rec["bo_P"], n)
              - st["bo_mean"]) / st["bo_std"]
        n_frags = len(find_fragments(s["c_R"], s["atom_types"], n))
        item = {
            "rxn_id": s["rxn_id"],
            "n_fragments": torch.tensor(n_frags, dtype=torch.long),
            # Plain int, never a tensor: collate_dynamic_padding reads it to size
            # the batch and then drops it, so it must not reach the GPU.
            "n_atoms": n,
            "D_R": D_R, "D_P": D_P, "D_I": (D_R + D_P) / 2.0,
            "D_TS": torch.from_numpy(s["D_TS"]),
            "c_I": torch.from_numpy(s["c_I"].astype(np.float32)),
            "c_TS": torch.from_numpy(s["c_TS"].astype(np.float32)),
            "mask": torch.from_numpy(s["mask"]),
            "atom_ids": torch.from_numpy(s["atom_ids"]),
            "atom_phys": torch.from_numpy(
                ((s["atom_phys_raw"] - st["aphys_mean"]) / st["aphys_std"]).astype(np.float32)),
            "Ea": torch.tensor(s["Ea_raw"], dtype=torch.float32),
            "de_rxn": torch.tensor(
                (s["de_rxn_raw"] - st["de_rxn_mean"]) / st["de_rxn_std"], dtype=torch.float32),
            "energy_feats": torch.from_numpy(
                ((s["energy_feats_raw"] - st["efeat_mean"]) / st["efeat_std"]).astype(np.float32)),
            "bo_feats": torch.from_numpy(bo.astype(np.float32)),
            "risk_pair_mask": torch.from_numpy(s["risk_pair_mask"]),
            "risk_penalty": torch.tensor(s["risk_penalty"], dtype=torch.float32),
            "rc_atom": torch.from_numpy(rc_atom),
            "spectator_pair": torch.from_numpy(spec),
            "reactive_pair": torch.from_numpy(react),
        }
        self.cache[idx] = item
        return item


# Which axes of each field are indexed by atom. An explicit table rather than
# inspecting shapes: a [30, 30] pair matrix and a [30, 3] coordinate array are both
# "two-dimensional with 30 in front", and trimming the wrong one of those produces
# a batch that is silently wrong rather than one that raises.
COLLATE_ATOM_AXES = {
    "D_R": (0, 1), "D_P": (0, 1), "D_I": (0, 1), "D_TS": (0, 1),
    "risk_pair_mask": (0, 1), "spectator_pair": (0, 1), "reactive_pair": (0, 1),
    "c_I": (0,), "c_TS": (0,), "atom_phys": (0,),
    "mask": (0,), "atom_ids": (0,), "rc_atom": (0,),
    "Ea": (), "de_rxn": (), "risk_penalty": (), "energy_feats": (), "bo_feats": (), "n_fragments": (),
}


def collate_dynamic_padding(items):
    """Stack a batch, trimming every atom axis to the batch's largest molecule.

    The EGNN is essentially the entire forward cost and it is quadratic in atom
    count: each EGCL materialises a [B, N, N, 2*EGNN_HIDDEN+1] edge tensor. Padding
    every batch to MAX_ATOMS = 30 when the mean RGD1 molecule has 17.7 atoms means
    most of that tensor is masked-out zeros. Trimming to max(n_atoms) removes
    nothing but padding, so it is exact, not an approximation.

    It only pays alongside LengthBucketedBatchSampler. Measured over the 39,964
    reactions: batches of 48 drawn uniformly have E[max n_atoms] = 26.7 and cut the
    quadratic work by 20%; length-bucketed batches cut it by 64%.

    PSICore's input projection still expects a fixed MAX_ATOMS-wide neighbour row
    per atom and pads its own argument back (see PSICore.forward), which is why the
    encoder weights are unaffected by this.
    """
    n = max(it["n_atoms"] for it in items)
    out = {"rxn_id": [it["rxn_id"] for it in items]}
    for key, axes in COLLATE_ATOM_AXES.items():
        column = [it[key] for it in items]
        for axis in axes:
            column = [t.narrow(axis, 0, n) for t in column]
        out[key] = torch.stack(column)
    return out


class LengthBucketedBatchSampler(Sampler):
    """Batches of similarly-sized molecules, sharded across ranks.

    Replaces DistributedSampler outright, on one GPU as well (num_replicas=1), so
    there is a single sampler path rather than a distributed one and a plain one.

    A global sort by atom count would make every batch a fixed set of molecules of
    one size — a permanent correlation between batch composition and molecule size,
    and the same 48 reactions in the same batch for 800 epochs. Instead each epoch
    reshuffles, cuts the stream into pools of BUCKET_POOL_BATCHES batches, sorts
    within a pool, and shuffles the resulting batch order: batches stay size
    homogeneous while their membership is redrawn every epoch.
    """

    def __init__(self, n_atoms, batch_size, num_replicas, rank, drop_last):
        self.n_atoms = np.asarray(n_atoms)
        self.batch_size = batch_size
        self.num_replicas = num_replicas
        self.rank = rank
        self.drop_last = drop_last
        self.epoch = 0
        full, remainder = divmod(len(self.n_atoms), batch_size)
        total = full if drop_last else full + int(remainder > 0)
        # Every rank must run the same number of steps or DDP's gradient all-reduce
        # blocks forever on whichever rank still has batches left, so the shared
        # batch list is truncated to a whole multiple of the world size.
        self.num_batches = total // num_replicas

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        # Seeded by epoch, so every rank builds the IDENTICAL batch list and then
        # takes its own stride through it. Validation never calls set_epoch, which
        # makes its batching deterministic across the whole run.
        rng = np.random.default_rng(SPLIT_SEED + self.epoch)
        order = rng.permutation(len(self.n_atoms))
        pool = self.batch_size * BUCKET_POOL_BATCHES
        batches = []
        for start in range(0, len(order), pool):
            chunk = order[start:start + pool]
            chunk = chunk[np.argsort(self.n_atoms[chunk], kind="stable")]
            batches.extend(chunk[i:i + self.batch_size].tolist()
                           for i in range(0, len(chunk), self.batch_size))
        if self.drop_last:
            batches = [b for b in batches if len(b) == self.batch_size]
        batches = [batches[i] for i in rng.permutation(len(batches))]
        return iter(batches[self.rank::self.num_replicas][:self.num_batches])


def to_device(batch, device):
    """Every tensor field to GPU. rxn_id is a list of str and stays on the host."""
    return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
            for k, v in batch.items()}


# =============================================================================
# Model
# =============================================================================
class GaussianEmbedding(nn.Module):
    """Radial basis expansion of a distance matrix."""

    def __init__(self):
        super().__init__()
        self.register_buffer("centers", torch.linspace(GAUSS_START, GAUSS_STOP, N_GAUSSIANS))
        self.sigma = (GAUSS_STOP - GAUSS_START) / (N_GAUSSIANS - 1) * 0.5

    def forward(self, D):
        return torch.exp(-0.5 * ((D.unsqueeze(-1) - self.centers) / self.sigma) ** 2)


class PreNormTransformerLayer(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, ATTN_HEADS, dropout=DROPOUT,
                                          batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, FF_DIM), nn.GELU(), nn.Dropout(DROPOUT),
            nn.Linear(FF_DIM, d_model), nn.Dropout(DROPOUT))

    def forward(self, x, pad_mask):
        h = self.norm1(x)
        x = x + self.attn(h, h, h, key_padding_mask=pad_mask, need_weights=False)[0]
        return x + self.ff(self.norm2(x))


class PSICore(nn.Module):
    """Reaction encoder: (D_R, D_I, D_P) + atom identity -> per-atom context.

    Each atom's row of each distance matrix is expanded into radial bases, so the
    three endpoint geometries become a length-3 sequence per atom; a bidirectional
    GRU reads that sequence (R -> midpoint -> P is a reaction coordinate), and the
    transformer then mixes information between atoms.
    """

    def __init__(self, num_atom_types):
        super().__init__()
        d_model = GRU_HIDDEN * 2
        self.atom_embed = nn.Embedding(num_atom_types + 1, ATOM_EMBED_DIM, padding_idx=0)
        self.gaussian = GaussianEmbedding()
        self.input_proj = nn.Sequential(
            nn.Linear(MAX_ATOMS * N_GAUSSIANS + ATOM_EMBED_DIM + ATOM_PHYS_DIM, d_model),
            nn.LayerNorm(d_model), nn.GELU(), nn.Dropout(DROPOUT))
        self.gru = nn.GRU(d_model, GRU_HIDDEN, num_layers=GRU_LAYERS, batch_first=True,
                          bidirectional=True, dropout=GRU_DROPOUT)
        self.gru_proj = nn.Sequential(nn.Linear(d_model, d_model), nn.LayerNorm(d_model))
        self.layers = nn.ModuleList([PreNormTransformerLayer(d_model)
                                     for _ in range(ATTN_LAYERS)])
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, D_R, D_I, D_P, mask, atom_ids, atom_phys):
        B, N, _ = D_R.shape
        # input_proj reads a FIXED MAX_ATOMS-wide neighbour row per atom, so the
        # dynamic collate's trim is undone here and only here. The zeros go onto the
        # DISTANCES, before the radial expansion, not onto the expansion afterwards:
        # a padded distance of 0 expands to a small non-zero radial basis, and
        # zero-padding the basis instead would be a different input for the same
        # molecule. This way the activations are identical to static padding and an
        # existing checkpoint stays valid.
        pad = MAX_ATOMS - N
        atom_feat = torch.cat([self.atom_embed(atom_ids), atom_phys], dim=-1)
        embs = [self.input_proj(torch.cat(
                    [self.gaussian(F.pad(D, (0, pad))).view(B, N, -1), atom_feat], -1))
                for D in (D_R, D_I, D_P)]
        seq = torch.stack(embs, dim=2).view(B * N, 3, -1)
        # The midpoint step carries both directions of the bidirectional pass.
        context = self.gru_proj(self.gru(seq)[0][:, 1, :].view(B, N, -1))
        pad_mask = mask == 0
        for layer in self.layers:
            context = layer(context, pad_mask)
        return self.final_norm(context)


class EGCL(nn.Module):
    """One E(n)-equivariant graph convolution (Satorras et al., 2021).

    Messages depend only on squared interatomic distances, so node features stay
    E(3)-invariant while the coordinate update is E(3)-equivariant.
    """

    def __init__(self):
        super().__init__()
        h = EGNN_HIDDEN
        self.edge_mlp = nn.Sequential(
            nn.Linear(h * 2 + 1, h), nn.GELU(), nn.Dropout(DROPOUT),
            nn.Linear(h, h), nn.GELU())
        self.node_mlp = nn.Sequential(
            nn.Linear(h * 2, h), nn.GELU(), nn.Dropout(DROPOUT), nn.Linear(h, h))
        self.coord_mlp = nn.Sequential(nn.Linear(h, h), nn.GELU(), nn.Linear(h, 1))
        # Zero-init: the layer starts as an identity map on coordinates, so the
        # EGNN begins exactly at the R/P midpoint seed and learns to move away.
        nn.init.zeros_(self.coord_mlp[-1].weight)
        nn.init.zeros_(self.coord_mlp[-1].bias)

    def forward(self, h, x, mask):
        B, N, H = h.shape
        pair = mask.unsqueeze(-1) * mask.unsqueeze(-2)
        eye = torch.eye(N, device=h.device, dtype=h.dtype).unsqueeze(0)
        emask = (pair * (1.0 - eye)).unsqueeze(-1)
        rel = x.unsqueeze(2) - x.unsqueeze(1)
        dist2 = (rel ** 2).sum(-1, keepdim=True)
        hi = h.unsqueeze(2).expand(B, N, N, H)
        hj = h.unsqueeze(1).expand(B, N, N, H)
        m = self.edge_mlp(torch.cat([hi, hj, dist2], dim=-1)) * emask

        # Equivariant coordinate update. The per-edge displacement VECTOR is what
        # gets length-limited, not the scalar edge weight, so a single layer cannot
        # translate an atom further than EGNN_COORD_CLAMP no matter how many edges
        # agree. The epsilon lives INSIDE the sqrt on purpose: d/dr sqrt(r) is
        # infinite at r = 0, and raw is exactly zero on every self-pair, on every
        # padded pair, and on EVERY edge at initialisation (coord_mlp is zero-init).
        # Adding it after the sqrt instead would emit NaN into `rel`, and from there
        # into the encoder, on the very first backward pass.
        raw = rel * self.coord_mlp(m)
        norm = raw.pow(2).sum(-1, keepdim=True).add(1e-12).sqrt()
        trans = raw * (norm.clamp(max=EGNN_COORD_CLAMP) / norm) * emask
        deg = emask.sum(dim=2).clamp(min=1.0)
        x = (x + trans.sum(dim=2) / deg) * mask.unsqueeze(-1)

        # Zero centre-of-mass. The output IS coordinates, so a free global
        # translation would otherwise be an unconstrained direction for the loss to
        # wander along: `m` is built from the ORDERED pair (h_i, h_j, d_ij), so
        # w_ij != w_ji and the displacements do not cancel. Masked mean, never
        # x.mean(1) -- padded rows are zero, so dividing by N would drag real atoms
        # toward the origin in proportion to how much padding a molecule carries.
        com = (x * mask.unsqueeze(-1)).sum(1, keepdim=True) / mask.sum(1).view(B, 1, 1).clamp(min=1.0)
        x = (x - com) * mask.unsqueeze(-1)

        h = (h + self.node_mlp(torch.cat([h, m.sum(dim=2)], dim=-1))) * mask.unsqueeze(-1)
        return h, x


class EGNN(nn.Module):
    def __init__(self, node_in_dim):
        super().__init__()
        self.embed_in = nn.Sequential(
            nn.Linear(node_in_dim, EGNN_HIDDEN), nn.GELU(), nn.LayerNorm(EGNN_HIDDEN))
        self.layers = nn.ModuleList([EGCL() for _ in range(EGNN_LAYERS)])

    def forward(self, node_feats, x, mask):
        h = self.embed_in(node_feats) * mask.unsqueeze(-1)
        for layer in self.layers:
            h, x = layer(h, x, mask)
        return h, x


class EaHead(nn.Module):
    """Activation energy from the EGNN's refined node features.

    Three pooled views of the TS (attention, mean, reaction-centre) are fused with
    a separately-encoded physics stream, FiLM-modulated by the energetics
    (Hammond/BEP: where the TS sits along the reaction coordinate depends on the
    reaction energy), plus a direct linear BEP term so the dominant near-linear
    driver is not rederived inside the MLP.

    The physics stream is dE_rxn, the 28D R/P descriptor, and the continuous
    GFN2-xTB reaction-centre descriptor (bond_order_features) — the last of which
    is how much bond order actually moves and how it splits between formation and
    cleavage, rather than the thresholded bit the reaction-centre mask keeps.

    `h_ts` is detached by the caller, so this gradient never reshapes geometry.
    """

    def __init__(self, geom_trust_dim):
        super().__init__()
        h = EGNN_HIDDEN
        self.geom_trust_dim = geom_trust_dim
        self.attn = nn.Sequential(nn.Linear(h, h // 2), nn.GELU(), nn.Linear(h // 2, 1))
        # Softmax temperature on the reaction-centre mask: large -> a near-uniform
        # mean over the forming/breaking atoms.
        self.rc_attn_bias = nn.Parameter(torch.tensor(4.0))
        self.ts_proj = nn.Sequential(nn.Linear(3 * h, h), nn.GELU(), nn.Dropout(EA_HEAD_DROPOUT))
        self.phys_enc = nn.Sequential(
            nn.Linear(1 + ENERGY_FEAT_DIM + BO_FEAT_DIM, h), nn.GELU(),
            nn.Dropout(EA_HEAD_DROPOUT))
        # Trust stream gets its own encoder and running normalisation: its channels
        # are raw model quantities on wildly different scales (Angstrom displacement
        # vs clamped log-variance) and in a flat concat they swamped the z-scored
        # descriptors. Zero-init means it contributes EXACTLY zero at step 0, so the
        # head starts identical to the no-trust baseline and self-gates upward only
        # once displacement becomes meaningful.
        self.register_buffer("trust_mean", torch.zeros(geom_trust_dim))
        self.register_buffer("trust_var", torch.ones(geom_trust_dim))
        self.trust_enc = nn.Sequential(
            nn.Linear(geom_trust_dim, h), nn.GELU(), nn.Dropout(EA_HEAD_DROPOUT),
            nn.Linear(h, h))
        nn.init.zeros_(self.trust_enc[-1].weight)
        nn.init.zeros_(self.trust_enc[-1].bias)
        self.film = nn.Linear(h, 2 * h)
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)
        final = nn.Linear(h // 2, 1)
        nn.init.xavier_uniform_(final.weight, gain=0.1)
        nn.init.zeros_(final.bias)
        self.net = nn.Sequential(
            nn.Linear(2 * h, h), nn.GELU(), nn.Dropout(EA_HEAD_DROPOUT),
            nn.Linear(h, h // 2), nn.GELU(), nn.Dropout(EA_HEAD_DROPOUT), final)
        self.bep = nn.Linear(1, 1)
        nn.init.constant_(self.bep.weight, 0.5)
        nn.init.zeros_(self.bep.bias)

    def forward(self, h_ts, mask, de_rxn, energy_feats, bo_feats, rc_mask, geom_trust):
        pad = mask <= 0
        m = mask.unsqueeze(-1).to(h_ts.dtype)
        mean_pooled = (h_ts * m).sum(1) / m.sum(1).clamp(min=1.0)
        logits = self.attn(h_ts).squeeze(-1).masked_fill(pad, -1e4)
        attn_pooled = (h_ts * torch.softmax(logits, 1).unsqueeze(-1) * m).sum(1)
        # An all-false reaction-centre mask must NOT reach the softmax: every
        # logit would be equal and the "reaction-centre pool" would quietly become
        # a whole-molecule mean, training on a silently different feature. That is
        # how the 76 empty-xTB-centre reactions stay findable -- filter_to_bond_orders
        # removes them at load time, and this is the assertion that the filter
        # actually worked. Costs one sync per forward; worth it.
        if not bool(torch.all((rc_mask & (mask > 0)).sum(1) > 0)):
            raise ValueError(
                "Reaction-centre mask is empty for at least one sample. Refusing to "
                "pool over every atom instead. filter_to_bond_orders should have "
                "dropped this reaction -- the bond-order cache and the sample set "
                "are out of step."
            )
        rc_logits = (self.rc_attn_bias * rc_mask.to(h_ts.dtype)).masked_fill(pad, -1e4)
        rc_pooled = (h_ts * torch.softmax(rc_logits, 1).unsqueeze(-1)).sum(1)
        ts = self.ts_proj(torch.cat([attn_pooled, mean_pooled, rc_pooled], -1))

        de_col = de_rxn.to(h_ts.dtype).unsqueeze(-1)
        phys = self.phys_enc(torch.cat(
            [de_col, energy_feats.to(h_ts.dtype), bo_feats.to(h_ts.dtype)], -1))
        gt = geom_trust.float()
        # The finiteness test is what makes a geometry blow-up SURVIVABLE. These are
        # buffers, not parameters: nothing downstream can ever repair them, so a
        # single non-finite gt would write NaN into the running statistics and every
        # forward from then on would return NaN Ea -- with the weights themselves
        # untouched, because clip_grad_norm_ correctly skipped the step. The run
        # then burns its remaining epochs skipping every batch of every epoch.
        # Observed exactly that on the first P100 run. Holding the last good
        # statistics for one step is the same trade the optimiser already makes when
        # it skips a non-finite gradient, and it is visible in the same place: a
        # non-finite gt implies a non-finite loss, so the step is counted in
        # grad_skips.
        if self.training and gt.size(0) > 1 and bool(torch.isfinite(gt).all()):
            with torch.no_grad():
                self.trust_mean.mul_(0.9).add_(0.1 * gt.mean(0))
                self.trust_var.mul_(0.9).add_(0.1 * gt.var(0, unbiased=False))
        phys = phys + self.trust_enc(
            (gt - self.trust_mean) / (self.trust_var + 1e-5).sqrt()).to(phys.dtype)

        gamma, beta = self.film(phys).chunk(2, dim=-1)
        out = self.net(torch.cat([ts * (1.0 + gamma) + beta, phys], -1))
        return (out + self.bep(de_col)).squeeze(-1)


class PSI(nn.Module):
    """Coordinate-native transition-state predictor.

    encoder -> EGNN (seeded at the R/P midpoint) -> TS coordinates
                                                 -> distances, log-variance, Ea
    """

    def __init__(self, num_atom_types):
        super().__init__()
        d_model = GRU_HIDDEN * 2
        self.core = PSICore(num_atom_types)
        # The encoder output feeds the EGNN's node features directly. This is the
        # ONLY consumer of `core` now that the geometry head is gone -- omit it and
        # PSICore would run every forward, receive zero gradient and be discarded,
        # while the EGNN saw no reaction context at all: nothing would tell it
        # which bonds are forming or breaking.
        # We now add + 1 to the input dimension to explicitly pass the reaction center mask.
        self.egnn_general = EGNN(ATOM_EMBED_DIM + ATOM_PHYS_DIM + d_model + 1)
        self.egnn_specialist = EGNN(ATOM_EMBED_DIM + ATOM_PHYS_DIM + d_model + 1)
        # 3 displacement scalars (mean/max/reaction-centre) + 2 log-variance scalars
        # + 8 reaction-coordinate scalars read off the predicted TS. See geom_trust.
        self.geom_trust_dim = 13
        self.ea_head = EaHead(self.geom_trust_dim)
        self.geom_logvar_head = nn.Sequential(
            nn.Linear(EGNN_HIDDEN, EGNN_HIDDEN // 2), nn.GELU(),
            nn.Linear(EGNN_HIDDEN // 2, 1))
        # Zero-init -> log-variance starts at 0 (variance 1) -> the Kendall-Gal
        # attenuation is exactly neutral at step 0.
        nn.init.zeros_(self.geom_logvar_head[-1].weight)
        nn.init.zeros_(self.geom_logvar_head[-1].bias)

    @staticmethod
    def coords_to_distance(x, mask):
        N = x.shape[1]
        d = torch.sqrt(((x.unsqueeze(2) - x.unsqueeze(1)) ** 2).sum(-1) + 1e-8)
        eye = torch.eye(N, device=x.device, dtype=x.dtype).unsqueeze(0)
        return d * (1.0 - eye) * (mask.unsqueeze(-1) * mask.unsqueeze(-2))

    def geom_trust(self, x_init, x_ts, D_pred, logvar, mask, rc_mask,
                   D_R, D_I, D_P, reactive_pair):
        """Detached descriptors of the PREDICTED TS for the Ea head — [B, 13].

          [0:5]  confidence and displacement off the seed. Large or
                 reaction-centre-concentrated movement away from the R/P midpoint
                 marks a non-interpolative TS: the hard structural class.
          [5:13] where the predicted TS actually SITS along the reaction
                 coordinate. Progress on a reactive pair is

                     f = |d_TS - d_R| / (|d_TS - d_R| + |d_TS - d_P|)

                 which is 0 at the reactant's bond length and 1 at the product's.
                 Written as a ratio of absolute deviations rather than the obvious
                 (d_TS - d_R) / (d_P - d_R) because that denominator vanishes on
                 any pair whose bond ORDER changes while its LENGTH barely does —
                 a pi bond — and f is then unbounded on exactly the pairs that
                 matter most. This form lands in [0, 1] for every input with no
                 clamp anywhere.

                 The MEAN of f over the reactive pairs is the Hammond position
                 measured instead of inferred from dE_rxn, its SPREAD is
                 asynchronicity, and the deviation of the non-reacting scaffold
                 from the midpoint is the ring/steric strain no linear BEP term
                 can see. These are the three things the Marcus/BEP relations get
                 wrong on RGD1, and they only become readable once the geometry is
                 good — which is what EA_START_EPOCH is for.

        Everything is detached and computed in fp32. Detached because this informs
        the Ea head and must never reshape geometry; fp32 because the variance of f
        is O(1e-3) squared and would flush to zero as fp16.
        """
        m = mask.float()
        cnt = m.sum(1).clamp(min=1.0)
        rc = rc_mask.float() * m
        rc_cnt = rc.sum(1).clamp(min=1.0)
        disp = ((x_ts - x_init).detach().float() ** 2).sum(-1).add(1e-12).sqrt() * m
        lv = logvar.detach().float() * m

        d = D_pred.detach().float()
        eye = torch.eye(d.shape[1], device=d.device, dtype=d.dtype).unsqueeze(0)
        pair = m.unsqueeze(-1) * m.unsqueeze(-2) * (1.0 - eye)
        react = reactive_pair.float() * pair
        react_bool = react > 0.0
        # The clamps on this and on `scaffold` below are unreachable, not defensive:
        # measured over the whole bond-order cache, 0 of 39,994 reactions have an
        # empty scaffold (the smallest has 4 pairs; the smallest RGD1 molecule has 6
        # atoms) and the 76 with an empty reaction centre are removed by
        # filter_to_bond_orders before they reach a loader. They are kept because
        # the alternative to a division guard here is not a raise -- it is a silent
        # NaN entering the Ea head. The loud check for the same invariant already
        # exists one frame up, in EaHead.forward.
        n_react = react.sum((1, 2)).clamp(min=1.0)
        to_R = (d - D_R.float()).abs()
        to_P = (d - D_P.float()).abs()
        f = to_R / (to_R + to_P + 1e-6)
        f_mean = (f * react).sum((1, 2)) / n_react
        f_var = ((f - f_mean.view(-1, 1, 1)) ** 2 * react).sum((1, 2)) / n_react
        # f is in [0, 1] by construction, so 1.0 and 0.0 are exact identities for a
        # masked min and a masked max -- no sentinel large enough to matter is
        # needed, and none can leak into the result.
        f_min = f.masked_fill(~react_bool, 1.0).amin((1, 2))
        f_max = f.masked_fill(~react_bool, 0.0).amax((1, 2))

        off_mid = (d - D_I.float()).abs()
        scaffold = pair - react
        return torch.stack([
            disp.sum(1) / cnt, disp.max(1).values, (disp * rc).sum(1) / rc_cnt,
            lv.sum(1) / cnt, (lv * rc).sum(1) / rc_cnt,
            f_mean, f_var.add(1e-12).sqrt(), f_min, f_max,
            (d * react).sum((1, 2)) / n_react,
            to_R.masked_fill(~react_bool, 0.0).amax((1, 2)),
            (off_mid * scaffold).sum((1, 2)) / scaffold.sum((1, 2)).clamp(min=1.0),
            (off_mid * react).sum((1, 2)) / n_react], dim=-1)

    def forward(self, D_R, D_I, D_P, mask, atom_ids, atom_phys, de_rxn,
                energy_feats, bo_feats, c_seed, rc_atom, reactive_pair, specialist_mask=None):
        f = self.core(D_R, D_I, D_P, mask, atom_ids, atom_phys)
        atom_emb = self.core.atom_embed(atom_ids)
        # Inject reaction center explicit mask into node_feats so EGNN knows which bonds are breaking
        node_feats = torch.cat([atom_emb, atom_phys, f, rc_atom.unsqueeze(-1).to(torch.float32)], dim=-1)
        x_init = c_seed.to(torch.float32) * mask.unsqueeze(-1)
        # Sparse MoE Routing
        h_ts = torch.empty(mask.shape[0], mask.shape[1], EGNN_HIDDEN, device=mask.device, dtype=node_feats.dtype)
        x_ts = torch.empty_like(x_init)
        
        if specialist_mask is None:
            raise RuntimeError(
                "specialist_mask must be provided to PSI.forward(). "
                "Compute it as: (batch['risk_penalty'] > 0) | (batch['n_fragments'] == 1)"
            )
            
        gen_mask = ~specialist_mask
        if gen_mask.any():
            h_gen, x_gen = self.egnn_general(node_feats[gen_mask], x_init[gen_mask], mask[gen_mask])
            h_ts[gen_mask], x_ts[gen_mask] = h_gen, x_gen
            
        if specialist_mask.any():
            h_spec, x_spec = self.egnn_specialist(node_feats[specialist_mask], x_init[specialist_mask], mask[specialist_mask])
            h_ts[specialist_mask], x_ts[specialist_mask] = h_spec, x_spec
        D_pred = self.coords_to_distance(x_ts, mask)
        # Clamped before it reaches exp(-logvar): unbounded negative log-variance
        # overflows fp16 (exp(7) ~ 1096 is safe, exp(20) is not).
        logvar = self.geom_logvar_head(h_ts).squeeze(-1)
        logvar = logvar.clamp(-LOGVAR_CLAMP, LOGVAR_CLAMP).masked_fill(mask <= 0, 0.0)
        rc_mask = rc_atom.to(torch.bool) & (mask > 0)
        ea = self.ea_head(
            h_ts.detach(), mask, de_rxn, energy_feats, bo_feats, rc_mask,
            self.geom_trust(x_init, x_ts, D_pred, logvar, mask, rc_mask,
                            D_R, D_I, D_P, reactive_pair))
        return D_pred, x_ts, logvar, ea


# =============================================================================
# Losses
# =============================================================================
def kabsch_align_torch(X, Y, mask):
    """Batched rigid superposition of X onto Y over real atoms only.

    Without this the coordinate loss would punish a perfect structure for being
    rotated: the prediction lives in the seed's frame, c_TS in the DFT frame.

    The rotation is DETACHED. It is itself a function of X, and backpropagating
    through SVD reintroduces exactly the degenerate-spectrum instability that
    coordinate-native geometry exists to avoid (the 3x3 cross-covariance goes
    degenerate for linear or near-planar fragments). Treating R as constant is the
    standard registration-loss choice: the gradient still points at the shape error.

    Forced to fp32 — cuSOLVER has no fp16 batched SVD, and orthogonalising a 3x3
    in fp16 would be badly conditioned even where it is supported.
    """
    with torch.amp.autocast(device_type=X.device.type, enabled=False):
        X, Y = X.float(), Y.float()
        m = mask.unsqueeze(-1).float()
        cnt = m.sum(1, keepdim=True).clamp(min=1.0)
        Xc = (X - (X * m).sum(1, keepdim=True) / cnt) * m
        Yc = (Y - (Y * m).sum(1, keepdim=True) / cnt) * m
        U, _, Vt = torch.linalg.svd(Xc.detach().transpose(1, 2) @ Yc)
        d = torch.sign(torch.det(Vt.transpose(1, 2) @ U.transpose(1, 2)))
        # sign() is 0 for an exactly singular cross-covariance (collinear atoms),
        # which would zero the third row of R; send it to the non-reflecting branch.
        d = torch.where(d == 0, torch.ones_like(d), d)
        diag = torch.diag_embed(torch.stack([torch.ones_like(d), torch.ones_like(d), d], -1))
        # Proper rotations only: a reflection would let an enantiomer score as a
        # perfect match, and chirality is the whole point of predicting coordinates.
        R = (Vt.transpose(1, 2) @ diag @ U.transpose(1, 2)).detach()
        # The unrestricted best-orthogonal fit, from the SAME decomposition, so
        # measuring chirality costs no extra SVD. It is allowed to reflect; where
        # it beats the proper rotation, the prediction is the mirror image.
        R_reflect = (Vt.transpose(1, 2) @ U.transpose(1, 2)).detach()
        return (Xc @ R.transpose(1, 2)) * m, Yc, (Xc @ R_reflect.transpose(1, 2)) * m


def per_atom_rmsd(x, y, mask):
    sq = ((x - y) ** 2).sum(-1) * mask.to(x.dtype)
    return torch.sqrt((sq.sum(1) / mask.sum(1).clamp(min=1.0)).clamp(min=1e-12))


def coordinate_loss(x_pred, c_true, mask):
    """(huber, rmsd_A, flip_rate).

    The Huber term trains. rmsd_A is the interpretable number comparable to
    React-OT / OA-ReactDiff, which report coordinate RMSD rather than a
    distance-matrix MAE.

    flip_rate is the fraction of structures whose MIRROR IMAGE fits the truth
    materially better than any proper rotation does. It is the only signal in the
    whole pipeline that can see a chirality error: distance matrices are
    chirality-blind, so a model that learned every enantiomer would score a
    perfect distance MAE. Getting handedness right is a large part of why this
    pipeline predicts coordinates at all, so the rate is measured every epoch
    rather than assumed to be zero.
    """
    x, y, x_reflect = kabsch_align_torch(x_pred, c_true, mask)
    m = mask.unsqueeze(-1).to(x.dtype)
    l = (F.huber_loss(x, y, reduction="none", delta=GEOM_HUBER_DELTA) * m).sum() \
        / (m.sum().clamp(min=1.0) * 3.0)
    proper = per_atom_rmsd(x, y, mask)
    reflected = per_atom_rmsd(x_reflect, y, mask)
    # The min-RMSD guard keeps near-perfect structures out: for those the two fits
    # are numerically indistinguishable and the ratio test is meaningless noise.
    flipped = (reflected < CHIRALITY_FLIP_RATIO * proper) & (proper > CHIRALITY_MIN_RMSD)
    return l, proper.mean(), flipped.float().mean()


def geometry_pair_weights(D_R, D_P, D_TS, mask, rc_atom):
    """Per-pair weight for the distance loss.

    Inverse-distance weighting alone is near-blind to the long-range
    active<->spectator cross distances that encode global fragment orientation, so
    those are boosted and the static spectator backbone is damped. On top of that,
    pairs that MOVE a lot from R to P are lifted regardless of role: in a
    unimolecular rearrangement most moving pairs sit between two non-reaction-centre
    atoms and would otherwise be damped to 0.25x and under-trained, which is what
    made the model hedge toward the R/P midpoint.
    """
    B, N, _ = D_R.shape
    valid = mask.unsqueeze(-1) * mask.unsqueeze(-2)
    eye = torch.eye(N, device=mask.device, dtype=mask.dtype).unsqueeze(0)
    m2d = valid * (1.0 - eye)
    w = 1.0 / (D_TS * m2d + 1.0)
    a = rc_atom.to(w.dtype)
    ai, aj = a.unsqueeze(2), a.unsqueeze(1)
    role = (GEOM_HINGE_CROSS_WEIGHT * (ai + aj - 2.0 * ai * aj)
            + GEOM_ACTIVE_PAIR_WEIGHT * (ai * aj)
            + GEOM_SPECTATOR_SPECTATOR_WEIGHT * ((1.0 - ai) * (1.0 - aj)))
    move = (torch.abs(D_R - D_P) * m2d).to(w.dtype)
    move = move / move.amax(dim=(1, 2), keepdim=True).clamp(min=1e-6)
    return m2d, w * torch.maximum(role, GEOM_MOVE_WEIGHT * move)


def compute_loss(batch, out, ea_mean, ea_std, ea_scale):
    """Total loss plus the metrics worth logging.

    `ea_scale` is 0.0 before EA_START_EPOCH and 1.0 from it onward. A multiplier
    rather than an `if`, because dropping the Ea terms from the graph outright
    would leave every Ea-head parameter without a gradient, and DDP raises on that
    unless find_unused_parameters is set — which costs a full autograd-graph
    traversal on every step of the run, to buy nothing after epoch 150. Multiplying
    by zero gives them a zero gradient instead: the head does not move, and the
    reported ea_mae stays honest and unweighted throughout.

    It isolates GRADIENTS, not values: 0.0 * nan is nan, so a non-finite Ea still
    reaches the total loss during the geometry-only stage. That is deliberate. The
    only way the Ea branch goes non-finite is a non-finite predicted geometry, and
    then the geometry loss is non-finite in the same step anyway — so there is
    nothing to salvage and the step gets skipped either way. What must NOT happen is
    the state surviving the step; see the finiteness guard in EaHead.forward.
    """
    D_TS, mask = batch["D_TS"], batch["mask"]
    D_pred, x_ts, logvar, ea_pred = out
    m2d, weights = geometry_pair_weights(
        batch["D_R"], batch["D_P"], D_TS, mask, batch["rc_atom"])
    wm = m2d * weights
    denom = wm.sum().clamp(min=1)

    # Kendall-Gal heteroscedastic attenuation: precision-weight each pair's error
    # and pay a log-variance penalty, so the model can declare a pair hard instead
    # of hedging every prediction toward the midpoint.
    per_pair = F.huber_loss(D_pred, D_TS, reduction="none", delta=GEOM_HUBER_DELTA)
    lv_pair = 0.5 * (logvar.unsqueeze(2) + logvar.unsqueeze(1))
    
    # Unimolecular reactions suffer full-scaffold reorganization, so we double their geometry weight
    unimol_weight = torch.where(batch["n_fragments"] == 1, 2.0, 1.0).view(-1, 1, 1).to(D_pred.dtype).to(D_pred.device)
    l_geom = ((torch.exp(-lv_pair) * per_pair + 0.5 * lv_pair) * wm * unimol_weight).sum() / denom

    l_coord, rmsd, flip = coordinate_loss(x_ts, batch["c_TS"], mask)

    # Spectator pairs pulled toward the R/P midpoint. The xTB bonded-in-both rule
    # is what makes this legitimate: measured |D_TS - D_I| is 0.0143 A on the pairs
    # it selects, against 0.1218 A for the distance rule, whose 0.041 A systematic
    # bias sat on the same axis as the under-shoot failure.
    spec = batch["spectator_pair"].to(D_pred.dtype) * m2d
    l_spec = F.mse_loss(D_pred * spec, batch["D_I"] * spec, reduction="sum") \
        / spec.sum().clamp(min=1.0)
    # Steric floor, so clashes are trained against rather than clamped post hoc.
    radii = COVALENT_RADII_T.to(batch["atom_ids"].device)[batch["atom_ids"]]
    floor = STERIC_FLOOR_FRAC * (radii.unsqueeze(2) + radii.unsqueeze(1))
    l_steric = (F.relu(floor.to(D_pred.dtype) - D_pred) * m2d).sum() / m2d.sum().clamp(min=1.0)

    loss = (l_geom + COORD_LOSS_WEIGHT * l_coord
            + PINN_WEIGHT * (l_spec + STERIC_LOSS_WEIGHT * l_steric))

    # Up-weight the reactions the error analysis flagged as hard.
    risk_scale = (1.0 + RISK_WEIGHT_ALPHA * batch["risk_penalty"].float()).clamp(max=RISK_WEIGHT_MAX)
    risk_pair = batch["risk_pair_mask"] * m2d * risk_scale.view(-1, 1, 1)
    loss = loss + RISK_GEOM_LOSS_WEIGHT * (
        F.huber_loss(D_pred, D_TS, reduction="none", delta=0.5) * risk_pair
    ).sum() / risk_pair.sum().clamp(min=1.0)

    ea_target = (batch["Ea"] - ea_mean) / ea_std
    
    # 1. Asymmetric Loss Formula (SmoothL1 for normal, strict L1 for low barriers)
    loss_smooth = F.smooth_l1_loss(ea_pred, ea_target, reduction="none")
    loss_l1 = F.l1_loss(ea_pred, ea_target, reduction="none")
    ea_per = torch.where(batch["Ea"] < 20.0, loss_l1, loss_smooth)
    
    # 2. Continuous Dynamic Weighting (Smooth Ramp)
    # Ramps up smoothly from 1.0x (normal reactions) to 2.5x (very low barriers)
    dynamic_weight = 1.0 + 1.5 * torch.sigmoid((30.0 - batch["Ea"]) / 5.0).to(ea_pred.device)
    ea_per = ea_per * dynamic_weight

    risk_w = risk_scale * (batch["risk_penalty"] > 0.0).float()
    loss = loss + ea_scale * (EA_LOSS_WEIGHT * ea_per.mean() + RISK_EA_LOSS_WEIGHT * (
        (ea_per * risk_w).sum() / risk_w.sum().clamp(min=1.0)))

    # Unweighted physical readouts, independent of the training weighting.
    geom_mae = (torch.abs(D_pred - D_TS) * m2d).sum() / m2d.sum().clamp(min=1.0)
    ea_mae = (ea_pred - ea_target).abs().mean() * ea_std
    return loss, {"loss": loss.detach(), "geom_mae_A": geom_mae.detach(),
                  "rmsd_A": rmsd.detach(), "ea_mae": ea_mae.detach(),
                  "coord": l_coord.detach(), "chirality_flip": flip.detach()}


COVALENT_RADII_T = torch.tensor(
    [0.0] + [COVALENT_RADII.get(a, 0.76) for a in sorted(ATOMIC_NUMBER)],
    dtype=torch.float32)


# =============================================================================
# Schedule
# =============================================================================
class WarmupCosine:
    """Linear warmup then cosine to min_lr over SWA_START epochs.

    The horizon is the SWA start, not the run length: past that point SWALR takes
    over at a constant rate and weight averaging supplies the remaining gain.
    """

    def __init__(self, optimizer, min_lr=1e-6):
        self.opt = optimizer
        self.min_lr = min_lr
        self.base = [g["lr"] for g in optimizer.param_groups]
        self.epoch = 0

    def step(self):
        self.epoch += 1
        if self.epoch <= WARMUP_EPOCHS:
            scale = self.epoch / WARMUP_EPOCHS
            for g, b in zip(self.opt.param_groups, self.base):
                g["lr"] = b * scale
            return
        p = min((self.epoch - WARMUP_EPOCHS) / max(1, SWA_START - WARMUP_EPOCHS), 1.0)
        cos = 0.5 * (1.0 + math.cos(math.pi * p))
        for g, b in zip(self.opt.param_groups, self.base):
            g["lr"] = self.min_lr + (b - self.min_lr) * cos

    def state_dict(self):
        return {"epoch": self.epoch}

    def load_state_dict(self, sd):
        self.epoch = sd["epoch"]


# =============================================================================
# Distributed helpers
# =============================================================================
def all_reduce_sum(t):
    """In-place SUM across ranks, or a no-op on a single GPU. There is no process
    group when world_size == 1 (P100), so calling dist.all_reduce would raise;
    with one rank the local tensor already IS the sum."""
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t


def reduce_metrics(sums, count, device):
    """Sum per-rank metric totals and counts, then divide. Exact means across ranks."""
    keys = sorted(sums)
    t = torch.tensor([sums[k] for k in keys] + [count], dtype=torch.float64, device=device)
    all_reduce_sum(t)
    n = t[-1].clamp(min=1.0)
    return {k: (t[i] / n).item() for i, k in enumerate(keys)}


def logvar_summary(hist):
    """Percentiles and clamp-saturation fractions from the accumulated histogram.

    `pinned_lo` is the diagnostic that matters: the fraction of atoms sitting in
    the lowest bin, i.e. pressed against the log-variance floor. The Kendall-Gal
    objective can be reduced either by predicting better OR by shrinking sigma,
    and a pinned_lo that climbs without plateauing means it is doing the second.
    Every such atom also carries the full exp(LOGVAR_CLAMP) ~ 1097x gradient
    amplification, so this is a stability signal as well as an honesty one.
    """
    total = hist.sum().clamp(min=1.0)
    cdf = (hist.cumsum(0) / total)
    width = (LOGVAR_HIST_MAX - LOGVAR_HIST_MIN) / LOGVAR_HIST_BINS
    centers = LOGVAR_HIST_MIN + (torch.arange(
        LOGVAR_HIST_BINS, device=hist.device, dtype=hist.dtype) + 0.5) * width
    pick = lambda p: centers[int((cdf >= p).nonzero()[0, 0])].item()
    return {"p1": pick(0.01), "p50": pick(0.50), "p99": pick(0.99),
            "pinned_lo": (hist[0] / total).item(),
            "pinned_hi": (hist[-1] / total).item()}


def percentiles(values):
    """p50/p99/max of the per-step gradient norms."""
    t = torch.tensor(values, dtype=torch.float64)
    return {"p50": t.quantile(0.50).item(), "p99": t.quantile(0.99).item(),
            "max": t.max().item()}


def run_epoch(model, loader, optimizer, scaler, device, ea_mean, ea_std, ea_scale, train):
    model.train(train)
    sums, count = {}, 0
    grad_skips, clipped, grad_norms = 0, 0, []
    logvar_hist = torch.zeros(LOGVAR_HIST_BINS, device=device, dtype=torch.float64)
    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for batch in loader:
            batch = to_device(batch, device)
            if train:
                optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                specialist = (batch["risk_penalty"] > 0) | (batch["n_fragments"] == 1)
                out = model(batch["D_R"], batch["D_I"], batch["D_P"], batch["mask"],
                            batch["atom_ids"], batch["atom_phys"], batch["de_rxn"],
                            batch["energy_feats"], batch["bo_feats"], batch["c_I"],
                            batch["rc_atom"], batch["reactive_pair"], specialist_mask=specialist)
                loss, metrics = compute_loss(batch, out, ea_mean, ea_std, ea_scale)
            # Real atoms only: logvar is masked_fill(0) on padding, which would
            # otherwise pile a spurious spike in the middle of the distribution.
            lv = out[2].detach()[batch["mask"] > 0].float()
            logvar_hist += torch.histc(lv, bins=LOGVAR_HIST_BINS,
                                       min=LOGVAR_HIST_MIN, max=LOGVAR_HIST_MAX).double()
            if train:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                # DDP has already all-reduced the gradients, so every rank computes
                # the same norm here and no further reduction is needed.
                gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                # Skip the step outright rather than trusting GradScaler alone to
                # notice: a non-finite gradient that lands in the weights is
                # unrecoverable, and the count makes instability visible.
                if torch.isfinite(gnorm):
                    scaler.step(optimizer)
                    grad_norms.append(gnorm.item())
                    clipped += int(gnorm.item() > GRAD_CLIP)
                else:
                    grad_skips += 1
                scaler.update()
            bs = batch["mask"].shape[0]
            count += bs
            for k, v in metrics.items():
                sums[k] = sums.get(k, 0.0) + v.item() * bs
    all_reduce_sum(logvar_hist)
    reduced = reduce_metrics(sums, count, device)
    reduced["grad_skips"] = grad_skips
    # Checked BEFORE logvar_summary, which cannot survive this case either: a model
    # emitting NaN emits a NaN log-variance too, torch.histc drops every NaN outside
    # its bins, and summarising the resulting all-zero histogram indexes an empty
    # tensor. Ordering the fatal check first is the difference between this message
    # and an IndexError three frames down in a percentile helper.
    # An epoch in which EVERY step was skipped is not instability, it is a dead run:
    # the weights are byte-identical to where they started and no further epoch can
    # differ, so the remaining budget would be spent recomputing the same NaN. Raise
    # here rather than let it reach the log line, which used to subscript the None
    # below and die with a bare TypeError several frames away from the cause.
    if train and not grad_norms:
        raise RuntimeError(
            f"Every one of {grad_skips} training steps this epoch produced a "
            f"non-finite gradient norm, so no weight was updated. The usual cause "
            f"is a NaN that has become STICKY rather than a transient one -- check "
            f"ea_head.trust_mean / trust_var in the checkpoint, and the loss "
            f"metrics for this epoch: {dict(reduced)}"
        )
    reduced["logvar"] = logvar_summary(logvar_hist)
    # clip_rate pinned at 1.00 means the clip, not the schedule, is setting every
    # step size. Measured and dismissed as a lever on the legacy pipeline (a
    # 1/5/15 sweep moved val MAE by 0.0024 A, under the 0.0141 A noise floor),
    # but it is cheap to keep watching.
    reduced["grad_norm"] = ({"clip_rate": clipped / max(len(grad_norms), 1),
                             **percentiles(grad_norms)} if grad_norms else None)
    return reduced


# =============================================================================
# Training
# =============================================================================
def prepare_data(verbose):
    """Samples, split and normalisation. Deterministic, so every rank agrees.

    Called INSIDE each worker rather than passed through mp.spawn: spawn pickles
    its arguments to every child, and 40k samples carrying several [30, 30]
    float32 arrays each is well over a gigabyte per rank, paid twice (once to
    serialise in the parent, once to materialise in the child) before training
    starts. Loading from the on-disk cache in each worker skips both copies.
    """
    samples, atom_vocab = build_samples()
    orders = load_bond_orders()
    samples, dropped = filter_to_bond_orders(samples, orders)
    train_idx, val_idx = make_split(samples)
    stats = compute_normalization(samples, train_idx, orders)
    if verbose:
        # The dropped ids are recorded because make_split is POSITIONAL: it bins by
        # list index and shuffles each bin, so removing any reaction re-deals every
        # assignment rather than just its own. A run that dropped these does not
        # share a validation set with one that did not, and without the ids there
        # is no way to compare the two on their intersection afterwards.
        with open(SPLIT_PATH, "w") as fh:
            json.dump({"seed": SPLIT_SEED, "val_split": VAL_SPLIT,
                       "n_total": len(samples), "n_train": len(train_idx),
                       "n_val": len(val_idx), "dropped_rxn_ids": dropped,
                       "ea_mean": stats["ea_mean"], "ea_std": stats["ea_std"],
                       "val_rxn_ids": [samples[i]["rxn_id"] for i in val_idx]}, fh)
        print(f"[split] recorded to {SPLIT_PATH} ({len(dropped)} dropped)")
    return samples, atom_vocab, train_idx, val_idx, stats, orders


def train_worker(rank, world_size):
    # Distributed ONLY when there is more than one GPU. A single P100 session has
    # no peer to reduce with, and initialising an NCCL group of size 1 is a real
    # failure/hang point in a Kaggle notebook -- so it is skipped entirely rather
    # than made a no-op. Every collective downstream is already guarded
    # (all_reduce_sum), so the training code below is byte-identical either way.
    distributed = world_size > 1
    if distributed:
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29500")
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    is_main = rank == 0

    # Silence the duplicate diagnostics from every non-zero rank; the work itself
    # is identical and deterministic on all of them.
    stdout = sys.stdout
    sys.stdout = stdout if is_main else open(os.devnull, "w")
    samples, atom_vocab, train_idx, val_idx, stats, orders = prepare_data(is_main)
    sys.stdout = stdout

    # Kaggle GPU setup. TF32 and cudnn autotuning cost nothing on Turing and pay
    # off immediately if the session lands on a newer accelerator.
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    dataset = ReactionDataset(samples, stats, orders)
    # One sampler on one and two GPUs alike: LengthBucketedBatchSampler shards
    # across ranks itself, so there is no separate single-GPU path to diverge.
    train_sampler = LengthBucketedBatchSampler(
        [samples[i]["n_atoms"] for i in train_idx], BATCH_SIZE,
        num_replicas=world_size, rank=rank, drop_last=True)
    train_loader = DataLoader(
        Subset(dataset, train_idx), batch_sampler=train_sampler,
        collate_fn=collate_dynamic_padding,
        num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True,
        prefetch_factor=PREFETCH_FACTOR)
    # Validation is sharded by slicing, never by a distributed sampler: those pad
    # the last shard with duplicates to equalise rank sizes, which would silently
    # double-count a few reactions in the metric this run is selected on. Its
    # sampler therefore sees a single replica and keeps the short final batch --
    # dropping it would throw up to 47 reactions out of the validation set.
    val_shard = val_idx[rank::world_size]
    val_loader = DataLoader(
        Subset(dataset, val_shard),
        batch_sampler=LengthBucketedBatchSampler(
            [samples[i]["n_atoms"] for i in val_shard], BATCH_SIZE,
            num_replicas=1, rank=0, drop_last=False),
        collate_fn=collate_dynamic_padding,
        num_workers=NUM_WORKERS, pin_memory=True,
        persistent_workers=True, prefetch_factor=PREFETCH_FACTOR)

    model = PSI(len(atom_vocab)).to(device)
    if is_main:
        print(f"[model] {sum(p.numel() for p in model.parameters()):,} parameters")
    # DDP only wraps a real multi-GPU group -- with one GPU it would just add a
    # gradient-sync hook that reduces against nobody. The train loop calls the
    # model through `net` either way.
    net = DDP(model, device_ids=[rank]) if distributed else model
    # torch.compile lowers through Triton, which supports compute capability 7.0
    # and up. Kaggle's P100 is Pascal (6.0), where inductor fails outright at the
    # first kernel; T4 is Turing (7.5) and compiles fine. Checked rather than
    # attempted-and-caught: this is a capability fact, not an error condition.
    capability = torch.cuda.get_device_capability(rank)
    # dynamic=True is REQUIRED now, not a tuning choice: collate_dynamic_padding
    # hands the model a different atom count almost every batch. On the default
    # (dynamic=None) inductor specialises on the first shape, recompiles on the
    # second, and once it has seen cache_size_limit (8) distinct shapes it stops
    # compiling and silently runs eager for the rest of the run -- about 15
    # distinct N occur here, so that fallback is certain rather than possible.
    compiled = torch.compile(net, dynamic=True) if capability >= (7, 0) else net
    if is_main:
        print(f"[gpu] {torch.cuda.get_device_name(rank)} sm_{capability[0]}{capability[1]} "
              f"| torch.compile {'on' if capability >= (7, 0) else 'OFF (needs sm_70+)'} "
              f"| fp16 AMP | TF32 {'on' if capability >= (8, 0) else 'n/a'}")
        print(f"[batch] dynamic padding, length-bucketed | "
              f"{len(train_sampler)} train steps/epoch/rank | "
              f"Ea head enters the loss at epoch {EA_START_EPOCH}")

    # The Ea head gets its own group: it reads detached features, so it is a
    # separate regression problem that tolerates a faster rate.
    ea_params = list(model.ea_head.parameters())
    ea_ids = {id(p) for p in ea_params}
    base_params = [p for p in model.parameters() if id(p) not in ea_ids]
    optimizer = torch.optim.AdamW([
        {"params": base_params, "lr": LR, "weight_decay": WEIGHT_DECAY},
        {"params": ea_params, "lr": EA_HEAD_LR, "weight_decay": EA_HEAD_WEIGHT_DECAY},
    ], fused=True)
    scheduler = WarmupCosine(optimizer)
    scaler = torch.amp.GradScaler()
    swa_model = torch.optim.swa_utils.AveragedModel(model)
    swa_scheduler = None

    # No resume. Every `train` starts from initialisation, so there is no optimiser,
    # scheduler, scaler or SWA state to reload and no epoch to continue from -- and
    # therefore no way for a checkpoint written by a diverged run to be read back
    # into a fresh one. What a session does not finish, it does not keep.
    best, patience, history = float("inf"), 0, []

    ea_mean, ea_std = stats["ea_mean"], stats["ea_std"]
    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        # Geometry-only until EA_START_EPOCH; see the constant for why.
        ea_scale = float(epoch >= EA_START_EPOCH)
        train_sampler.set_epoch(epoch)
        tr = run_epoch(compiled, train_loader, optimizer, scaler, device,
                       ea_mean, ea_std, ea_scale, True)
        # The averaged weights MUST be updated before they are validated. Reading
        # swa_model.module first would, on the very first SWA epoch, evaluate the
        # copy AveragedModel took at construction time -- i.e. the untrained
        # initialisation -- and record it as this epoch's score.
        if epoch == SWA_START:
            # Auto-tune SWA learning rate based on the optimizer's active converged state
            current_lr = optimizer.param_groups[0]['lr']
            swa_scheduler = torch.optim.swa_utils.SWALR(optimizer, swa_lr=current_lr * 2.0)

        if epoch >= SWA_START:
            swa_model.update_parameters(model)
            swa_scheduler.step()
        else:
            scheduler.step()
        # Past SWA_START the averaged weights are what gets selected and saved, so
        # they are what must be measured.
        eval_model = swa_model.module if epoch >= SWA_START else compiled
        va = run_epoch(eval_model, val_loader, None, scaler, device,
                       ea_mean, ea_std, ea_scale, False)

        # The selection metric CHANGES DEFINITION at EA_START_EPOCH, so nothing from
        # the geometry-only stage is comparable with anything after it. Left alone,
        # `best` would hold a geometry-only score that no later epoch can beat: the
        # Ea term only ever adds to it. psi_best.pt would freeze at a checkpoint
        # whose Ea head is untrained and the run would burn its whole patience
        # budget in the 120 epochs after the switch. Both are reset instead.
        if epoch == EA_START_EPOCH:
            best, patience = float("inf"), 0
            if is_main:
                print(f"[stage] epoch {epoch}: Ea head enters the loss; selection "
                      f"metric and patience reset", flush=True)
        # Selected on raw distance MAE, never on the loss: with the uncertainty head
        # the loss is an NLL whose precision term grows without bound as the model
        # gets confident, so it rises on validation even while accuracy improves.
        val_select = va["geom_mae_A"] + ea_scale * EA_SELECT_WEIGHT * va["ea_mae"] / ea_std
        improved = val_select < best
        best = min(best, val_select)
        patience = 0 if improved else patience + 1

        if is_main:
            history.append({"epoch": epoch, "lr": optimizer.param_groups[0]["lr"],
                            "val_select": val_select,
                            **{f"train_{k}": v for k, v in tr.items()},
                            **{f"val_{k}": v for k, v in va.items()}})
            print(f"{epoch:5d} | trGeom {tr['geom_mae_A']:.4f} | vaGeom {va['geom_mae_A']:.4f} "
                  f"| gap {va['geom_mae_A']-tr['geom_mae_A']:.4f} "
                  f"| vaRMSD {va['rmsd_A']:.3f}A | flip {va['chirality_flip']*100:4.1f}% "
                  f"| trEa {tr['ea_mae']:6.3f} | vaEa {va['ea_mae']:6.3f} "
                  f"| lvFloor {tr['logvar']['pinned_lo']*100:4.1f}% "
                  f"| clip {tr['grad_norm']['clip_rate']*100:3.0f}% "
                  f"| lr {optimizer.param_groups[0]['lr']:.2e} "
                  f"| {time.time()-t0:5.1f}s{' *' if improved else ''}", flush=True)
            # Only the best model is written. The full training state -- optimiser,
            # scheduler, scaler, SWA weights and averager -- existed to make a run
            # resumable, and with no resume it is several hundred MB serialised every
            # epoch that nothing will ever read.
            if improved:
                torch.save({"model": (swa_model.module if epoch >= SWA_START else model).state_dict(),
                            "stats": stats, "atom_vocab": atom_vocab, "epoch": epoch,
                            "val_select": val_select}, BEST_CKPT)
            with open(HISTORY_PATH, "w") as fh:
                json.dump(history, fh, indent=2)
        if patience >= PATIENCE:
            break

    if is_main:
        print(f"[done] best val_select {best:.4f}; {BEST_CKPT} holds the selected "
              f"model, {HISTORY_PATH} the per-epoch log")
        print("[done] There is no resume: a later session re-runs this from scratch.")
    if distributed:
        dist.destroy_process_group()


def stage_train():
    world_size = torch.cuda.device_count()
    if world_size == 0:
        raise RuntimeError(
            "No CUDA device. Set the Kaggle accelerator to GPU P100 or GPU T4 x2.")
    # Build the sample cache and prepare the bond orders HERE, in the parent, so
    # anything missing surfaces once with a readable message instead of as a spawn
    # traceback from two ranks at once. The parent then drops its copy; each worker
    # reloads from the on-disk cache.
    samples, _ = build_samples()
    n_samples = len(samples)
    del samples
    # Build the xTB reaction-centre cache on first use rather than making the user
    # run a separate stage, save a dataset, and re-attach it. This is a one-time
    # ~1.5-2 h CPU precompute; it lands in WORK_DIR and every resume reuses it, so
    # subsequent sessions skip straight to training. NOT a fallback -- it produces
    # the same GFN2-xTB cache the `bond-orders` stage would, never a substitute.
    if find_bond_order_cache() is None:
        print("[train] no bond-order cache found -> building it now (one-time, "
              "~1.5-2 h on CPU; save this notebook's output as a Dataset so later "
              "sessions skip it).", flush=True)
        stage_bond_orders()
    load_bond_orders()
    name = torch.cuda.get_device_name(0)
    print(f"[train] {n_samples} samples | {world_size}x {name} | "
          f"global batch {BATCH_SIZE * world_size}")
    # Single GPU runs IN-PROCESS. mp.spawn would have to pickle train_worker out
    # of __main__, which does not exist as an importable module inside a notebook
    # cell -- so on a one-GPU Kaggle session (P100) spawning fails outright, and
    # even where it works it buys nothing. The worker body is identical either
    # way: a world_size=1 process group makes every all_reduce a no-op, so there
    # is one training code path, not two.
    launch = (lambda: train_worker(0, 1)) if world_size == 1 else (
        lambda: mp.spawn(train_worker, args=(world_size,), nprocs=world_size, join=True))
    launch()


def stage_eval():
    """Score the best checkpoint per reaction and write RESULTS_PATH.

    Kept as its own stage rather than tacked onto the end of training: on Kaggle a
    session is cut long before the epoch budget runs out, so an evaluation that
    only fires after the final epoch would essentially never run.

    Reports the MEDIAN alongside the mean for every geometry metric. On the last
    converged legacy run those were 0.0983 A and 0.1241 A -- a 26% spread that
    says a structurally hard minority carries the error. Reporting only the mean
    hides that; reporting only the median (as the literature comparison invites)
    flatters it.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(BEST_CKPT, map_location=device, weights_only=False)
    samples, atom_vocab, train_idx, val_idx, stats, orders = prepare_data(True)
    model = PSI(len(atom_vocab)).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    print(f"[eval] checkpoint from epoch {ck['epoch']} "
          f"(val_select {ck['val_select']:.4f})")

    # Keyed by reaction id, not by position: the bucketed sampler does not walk the
    # dataset in order, so a running counter would label the wrong split.
    val_ids = {samples[i]["rxn_id"] for i in val_idx}
    loader = DataLoader(
        ReactionDataset(samples, stats, orders),
        batch_sampler=LengthBucketedBatchSampler(
            [s["n_atoms"] for s in samples], BATCH_SIZE,
            num_replicas=1, rank=0, drop_last=False),
        collate_fn=collate_dynamic_padding, num_workers=NUM_WORKERS, pin_memory=True)
    ea_mean, ea_std = stats["ea_mean"], stats["ea_std"]
    records = []
    with torch.no_grad():
        for batch in loader:
            batch = to_device(batch, device)
            with torch.amp.autocast(device_type=device.type, dtype=torch.float16):
                D_pred, x_ts, logvar, ea = model(
                    batch["D_R"], batch["D_I"], batch["D_P"], batch["mask"],
                    batch["atom_ids"], batch["atom_phys"], batch["de_rxn"],
                    batch["energy_feats"], batch["bo_feats"], batch["c_I"],
                    batch["rc_atom"], batch["reactive_pair"])
            mask = batch["mask"]
            N = mask.shape[1]
            eye = torch.eye(N, device=device).unsqueeze(0)
            m2d = mask.unsqueeze(-1) * mask.unsqueeze(-2) * (1.0 - eye)
            # Per-REACTION, not batch-averaged: the distribution is the point.
            mae = ((D_pred.float() - batch["D_TS"]).abs() * m2d).sum((1, 2)) \
                / m2d.sum((1, 2)).clamp(min=1.0)
            x, y, x_ref = kabsch_align_torch(x_ts.float(), batch["c_TS"], mask)
            proper = per_atom_rmsd(x, y, mask)
            reflected = per_atom_rmsd(x_ref, y, mask)
            flip = (reflected < CHIRALITY_FLIP_RATIO * proper) & (proper > CHIRALITY_MIN_RMSD)
            ea_kcal = ea.float() * ea_std + ea_mean
            lv = (logvar.float() * mask).sum(1) / mask.sum(1).clamp(min=1.0)
            for i in range(mask.shape[0]):
                records.append({
                    "rxn_id": batch["rxn_id"][i],
                    "split": "val" if batch["rxn_id"][i] in val_ids else "train",
                    "n_atoms": int(mask[i].sum().item()),
                    "dist_MAE": mae[i].item(), "rmsd": proper[i].item(),
                    "rmsd_reflected": reflected[i].item(),
                    "chirality_flip": bool(flip[i].item()),
                    "mean_logvar": lv[i].item(),
                    "Ea_true": batch["Ea"][i].item(), "Ea_pred": ea_kcal[i].item(),
                    "Ea_error": abs(ea_kcal[i].item() - batch["Ea"][i].item()),
                })

    def summarise(rows):
        d = np.array([r["dist_MAE"] for r in rows])
        rm = np.array([r["rmsd"] for r in rows])
        e = np.array([r["Ea_error"] for r in rows])
        return {"n": len(rows),
                "dist_MAE_mean": float(d.mean()), "dist_MAE_median": float(np.median(d)),
                "dist_MAE_p90": float(np.percentile(d, 90)),
                "rmsd_mean": float(rm.mean()), "rmsd_median": float(np.median(rm)),
                "Ea_MAE": float(e.mean()), "Ea_median": float(np.median(e)),
                "Ea_RMSE": float(np.sqrt((e ** 2).mean())),
                "chirality_flip_rate": float(np.mean([r["chirality_flip"] for r in rows]))}

    summary = {"epoch": ck["epoch"],
               "train": summarise([r for r in records if r["split"] == "train"]),
               "val": summarise([r for r in records if r["split"] == "val"])}
    with open(RESULTS_PATH, "w") as fh:
        json.dump({"summary": summary, "records": records}, fh)

    for split in ("train", "val"):
        s = summary[split]
        print(f"[eval] {split:>5}  n={s['n']:<6} "
              f"D-MAE mean {s['dist_MAE_mean']:.4f} median {s['dist_MAE_median']:.4f} "
              f"p90 {s['dist_MAE_p90']:.4f} A | RMSD median {s['rmsd_median']:.3f} A "
              f"| Ea MAE {s['Ea_MAE']:.3f} RMSE {s['Ea_RMSE']:.3f} kcal/mol "
              f"| chirality flips {s['chirality_flip_rate']*100:.2f}%")
    gap = summary["val"]["dist_MAE_mean"] / max(summary["train"]["dist_MAE_mean"], 1e-9)
    print(f"[eval] overfitting gap {gap:.2f}x    -> {RESULTS_PATH}")


STAGES = {"bond-orders": stage_bond_orders, "train": stage_train,
          "eval": stage_eval}
DEFAULT_STAGE = "train"


def resolve_stage(argv):
    """Pick the stage, tolerating the argv a notebook kernel injects.

    Run from a Jupyter/Kaggle cell -- `%run psi_cloud_pipeline.py`, or the file
    pasted straight in -- sys.argv belongs to IPython, not to us: it looks like
    ['.../ipykernel_launcher.py', '-f', '/root/.../kernel-abc.json']. Reading
    argv[1] positionally then reports `Unknown stage '-f'` and exits.

    So: honour a recognised stage name wherever it appears, ignore IPython's flag
    and its kernel .json, and fall back to DEFAULT_STAGE when nothing was asked
    for. A genuine typo still raises -- silently training for nine hours because
    'trian' was misspelt is exactly the kind of quiet wrong turn this pipeline
    refuses everywhere else.
    """
    named = [a for a in argv if a in STAGES]
    if named:
        return named[0]
    typo = [a for a in argv
            if not a.startswith("-") and not a.endswith((".json", ".py"))]
    if typo:
        raise SystemExit(
            f"Unknown stage {typo[0]!r}. Choose one of: {', '.join(STAGES)}")
    return DEFAULT_STAGE


if __name__ == "__main__":
    stage = resolve_stage(sys.argv[1:])
    print(f"[stage] {stage}")
    STAGES[stage]()
