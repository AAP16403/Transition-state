"""
PSI Interactive Result Visualizer: Analysis of AI Improvements
============================================================
1. Metric Comparison (3D Interactive Bars)
2. Coordinate Reconstruction (MDS)
3. Structural Alignment (Kabsch)
4. Interactive Dashboard Export (HTML)
"""

import os
import json
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ============================================================================
# 1. Geometry Algorithms
# ============================================================================

def mds(D, dim=3):
    """Classical Multidimensional Scaling (MDS)"""
    N = D.shape[0]
    H = np.eye(N) - np.ones((N, N))/N
    B = -0.5 * H @ (D**2) @ H
    evals, evecs = np.linalg.eigh(B)
    idx = np.argsort(evals)[::-1]
    evals = evals[idx]
    evecs = evecs[:, idx]
    X = evecs[:, :dim] @ np.diag(np.sqrt(np.maximum(evals[:dim], 0)))
    return X

def kabsch(P, Q):
    """Aligns P to Q using Kabsch algorithm (Procrustes Analysis)"""
    P_centered = P - P.mean(axis=0); Q_centered = Q - Q.mean(axis=0)
    C = P_centered.T @ Q_centered
    V, S, W = np.linalg.svd(C)
    d = np.linalg.det(V @ W)
    E = np.eye(3)
    if d < 0: E[2, 2] = -1
    R = V @ E @ W
    return P_centered @ R + Q.mean(axis=0)

# ============================================================================
# 2. Interactive Dashboard Components
# ============================================================================

def create_dashboard(data_path, save_dir):
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    rxn_ids = []; guess_maes = []; ai_maes = []
    for r in data:
        rxn_ids.append(r["rxn_id"])
        di, dp, dt = np.array(r["D_I"]), np.array(r["D_pred"]), np.array(r["D_true"])
        guess_maes.append(np.mean(np.abs(di - dt)))
        ai_maes.append(np.mean(np.abs(dp - dt)))

    # Create Subplots: 3D Scene 1 (Metrics) | 3D Scene 2 (Molecular Overlay)
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{'type': 'scene'}, {'type': 'scene'}]],
        subplot_titles=("3D Performance Comparison", "Interactive Structural Overlay")
    )

    # --- 1. 3D METRIC BARS ---
    # We use vertical lines to emulate 3D bars with interactive markers
    for i, (rid, g, a) in enumerate(zip(rxn_ids, guess_maes, ai_maes)):
        # Guess Bar
        fig.add_trace(go.Scatter3d(
            x=[i, i], y=[0, 0], z=[0, g],
            mode='lines+markers', line=dict(color='salmon', width=12),
            marker=dict(size=4, color='salmon'),
            name=f"Guess: {rid}", hovertext=f"Guess MAE: {g:.4f}", showlegend=False
        ), row=1, col=1)
        # AI Bar
        fig.add_trace(go.Scatter3d(
            x=[i, i], y=[1, 1], z=[0, a],
            mode='lines+markers', line=dict(color='deepskyblue', width=12),
            marker=dict(size=4, color='deepskyblue'),
            name=f"AI: {rid}", hovertext=f"AI MAE: {a:.4f}", showlegend=False
        ), row=1, col=1)

    # --- 2. STRUCTURAL OVERLAY (Best Reaction) ---
    best_idx = np.argmin(ai_maes); best = data[best_idx]
    dt, dp, di = np.array(best["D_true"]), np.array(best["D_pred"]), np.array(best["D_I"])
    atoms = best["atom_types"]
    
    X_true = mds(dt); X_pred = kabsch(mds(dp), X_true); X_guess = kabsch(mds(di), X_true)

    # Helper: Add Molecule Trace
    def add_mol(X, color, name, mode='markers', size=8, opacity=1.0):
        # Atoms
        fig.add_trace(go.Scatter3d(
            x=X[:, 0], y=X[:, 1], z=X[:, 2],
            mode=mode, marker=dict(size=size, color=color, opacity=opacity),
            text=atoms, hoverinfo='text+name', name=name
        ), row=1, col=2)
        # Bonds (Simple distance heuristic)
        for i in range(len(X)):
            for j in range(i+1, len(X)):
                if np.linalg.norm(X[i] - X[j]) < 1.6:
                    fig.add_trace(go.Scatter3d(
                        x=[X[i, 0], X[j, 0]], y=[X[i, 1], X[j, 1]], z=[X[i, 2], X[j, 2]],
                        mode='lines', line=dict(color=color, width=2),
                        opacity=opacity*0.5, showlegend=False, hoverinfo='skip'
                    ), row=1, col=2)

    add_mol(X_true, 'forestgreen', 'Actual TS (Ground Truth)', size=10)
    add_mol(X_pred, 'dodgerblue', 'AI Predicted TS', size=8, opacity=0.7)
    add_mol(X_guess, 'lightcoral', 'Interpolated Guess', size=4, opacity=0.4)

    # Final Layout
    fig.update_layout(
        title_text=f"PSI Result Dashboard - Featuring Reaction {best['rxn_id']}",
        scene1=dict(xaxis_title="Simulation ID", yaxis_title="Guess/AI", zaxis_title="Distance MAE (Å)"),
        scene2=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z"),
        height=800, width=1500,
        showlegend=True
    )
    
    output_html = os.path.join(save_dir, 'psi_results_dashboard.html')
    fig.write_html(output_html)
    
    # Cleanup static PNGs
    for f in ['comparative_metrics_3d.png', 'structural_overlay.png', 'comparative_metrics.png']:
        p = os.path.join(save_dir, f)
        if os.path.exists(p): os.remove(p)
    
    print(f"Interactive dashboard saved to: {output_html}")
    print("Static PNG drafts deleted.")

if __name__ == "__main__":
    DATA_PATH = r"d:\Transition state\detailed_analysis.json"
    SAVE_DIR = r"d:\Transition state"
    if os.path.exists(DATA_PATH): create_dashboard(DATA_PATH, SAVE_DIR)
