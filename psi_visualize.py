import os
import json
import numpy as np

def mds(D, dim=3):
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
    P_centered = P - P.mean(axis=0); Q_centered = Q - Q.mean(axis=0)
    C = P_centered.T @ Q_centered
    V, _, W = np.linalg.svd(C)
    if np.linalg.det(V @ W) < 0.0:
        P_centered = P_centered.copy()
        P_centered[:, 2] *= -1.0
        C = P_centered.T @ Q_centered
        V, _, W = np.linalg.svd(C)
    R = V @ W
    return P_centered @ R + Q.mean(axis=0)

def get_bonds(coords, atom_types):
    bonds = []
    n = len(coords)
    radii = {'H': 0.31, 'C': 0.76, 'N': 0.71, 'O': 0.66, 'F': 0.57, 'S': 1.05, 'Cl': 1.02, 'P': 1.07}
    for i in range(n):
        for j in range(i+1, n):
            r_i = radii[atom_types[i]]
            r_j = radii[atom_types[j]]
            dist = np.linalg.norm(coords[i] - coords[j])
            if dist < 1.45 * (r_i + r_j):
                bonds.append((i, j))
    return bonds

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
        guess_maes.append(np.mean(np.abs(di - dt)))
    guess_maes = np.array(guess_maes)

    ea_corr = np.corrcoef(ea_trues, ea_preds)[0, 1] if len(ea_trues) > 1 else 0.0

    train_data = [r for r in data if r["split"] == "train"]
    val_data = [r for r in data if r["split"] == "val"]

    train_ea_mae = np.mean([r["Ea_error"] for r in train_data]) if train_data else 0.0
    val_ea_mae = np.mean([r["Ea_error"] for r in val_data]) if val_data else 0.0
    train_dist_mae = np.mean([r["dist_MAE"] for r in train_data]) if train_data else 0.0
    val_dist_mae = np.mean([r["dist_MAE"] for r in val_data]) if val_data else 0.0

    train_corr = np.corrcoef([r["Ea_true"] for r in train_data], [r["Ea_pred"] for r in train_data])[0, 1] if len(train_data) > 1 else 0.0
    val_corr = np.corrcoef([r["Ea_true"] for r in val_data], [r["Ea_pred"] for r in val_data])[0, 1] if len(val_data) > 1 else 0.0

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

        X_true = mds(dt)
        X_pred = kabsch(mds(dp), X_true)
        X_guess = kabsch(mds(di), X_true)

        representative_data[rid] = {
            "rxn_id": rid,
            "atom_types": atoms,
            "coords_true": X_true.tolist(),
            "coords_pred": X_pred.tolist(),
            "coords_guess": X_guess.tolist(),
            "bonds_true": get_bonds(X_true, atoms),
            "bonds_pred": get_bonds(X_pred, atoms),
            "bonds_guess": get_bonds(X_guess, atoms),
            "dist_MAE": r["dist_MAE"],
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
            "guess_MAE": round(np.mean(np.abs(np.array(r["D_I"]) - np.array(r["D_true"]))), 4),
            "n_atoms": r["n_atoms"]
        })

    worst_10_geom = sorted(summary_list, key=lambda x: x["dist_MAE"], reverse=True)[:10]

    html_template = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>PSI Transition State Prediction Dashboard</title>
  <script src="https://cdn.plot.ly/plotly-2.24.1.min.js"></script>
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

    function updateViewer() {{
      const rid = select.value;
      const r = repData[rid];

      document.getElementById('case-id').innerText = r.rxn_id;
      document.getElementById('case-atoms').innerText = r.atom_types.length;
      document.getElementById('case-dist-mae').innerText = r.dist_MAE.toFixed(4) + ' Å';
      document.getElementById('case-ea-error').innerText = r.Ea_error.toFixed(2) + ' kcal/mol';
      
      const tierBadge = document.getElementById('case-tier');
      tierBadge.className = `badge badge-tier ${{r.tier}}`;
      tierBadge.innerText = r.tier;

      const traces = [];

      function addStructure(coords, bonds, color, name, size, opacity) {{
        traces.push({{
          x: coords.map(c => c[0]),
          y: coords.map(c => c[1]),
          z: coords.map(c => c[2]),
          mode: 'markers+text',
          type: 'scatter3d',
          name: name,
          text: r.atom_types,
          textposition: 'top center',
          marker: {{ size: size, color: color, opacity: opacity }},
          hoverinfo: 'name+text'
        }});

        let bx = [], by = [], bz = [];
        for (let b of bonds) {{
          bx.push(coords[b[0]][0], coords[b[1]][0], null);
          by.push(coords[b[0]][1], coords[b[1]][1], null);
          bz.push(coords[b[0]][2], coords[b[1]][2], null);
        }}
        traces.push({{
          x: bx, y: by, z: bz,
          mode: 'lines',
          type: 'scatter3d',
          line: {{ color: color, width: 2.5, opacity: opacity * 0.7 }},
          showlegend: false,
          hoverinfo: 'skip'
        }});
      }}

      addStructure(r.coords_true, r.bonds_true, '#10b981', 'Ground Truth TS', 7, 0.95);
      addStructure(r.coords_pred, r.bonds_pred, '#3b82f6', 'AI Predicted TS', 5.5, 0.85);
      addStructure(r.coords_guess, r.bonds_guess, '#ef4444', 'Interpolated Guess', 3.5, 0.35);

      const viewerLayout = {{
        paper_bgcolor: 'transparent',
        margin: {{ l: 0, r: 0, t: 0, b: 0 }},
        scene: {{
          xaxis: {{ visible: false }},
          yaxis: {{ visible: false }},
          zaxis: {{ visible: false }},
          camera: {{ eye: {{ x: 1.25, y: 1.25, z: 1.25 }} }}
        }},
        legend: {{
          x: 0, y: 1,
          font: {{ color: '#cbd5e1' }},
          bgcolor: 'rgba(0,0,0,0.5)'
        }}
      }};

      Plotly.newPlot('mol-viewer', traces, viewerLayout);
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

if __name__ == "__main__":
    DATA_PATH = r"d:\Transition state\detailed_analysis.json"
    SAVE_DIR = r"d:\Transition state"
    create_dashboard(DATA_PATH, SAVE_DIR)
