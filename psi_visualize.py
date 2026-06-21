import os
import json
import numpy as np
import argparse

from psi_utils import (
    mds,
    covalent_radius,
    kabsch,
    connected_components,
    find_fragments_from_distances as fragments_from_distances,
    fragments_from_mask,
    mds_by_fragments,
    masked_mae,
    get_bonds_from_distances,
)


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


def energy_metrics(records):
    """Regression metrics for the activation-energy prediction over `records`."""
    if not records:
        return {"n": 0, "MAE": 0.0, "RMSE": 0.0, "R2": 0.0, "Pearson": 0.0, "MAPE": 0.0}
    true = np.array([r["Ea_true"] for r in records], dtype=np.float64)
    pred = np.array([r["Ea_pred"] for r in records], dtype=np.float64)
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

    train_ea_mae = np.mean([r["Ea_error"] for r in train_data]) if train_data else 0.0
    val_ea_mae = np.mean([r["Ea_error"] for r in val_data]) if val_data else 0.0
    train_dist_mae = np.mean([r["dist_MAE"] for r in train_data]) if train_data else 0.0
    val_dist_mae = np.mean([r["dist_MAE"] for r in val_data]) if val_data else 0.0

    train_corr = np.corrcoef([r["Ea_true"] for r in train_data], [r["Ea_pred"] for r in train_data])[0, 1] if len(train_data) > 1 else 0.0
    val_corr = np.corrcoef([r["Ea_true"] for r in val_data], [r["Ea_pred"] for r in val_data])[0, 1] if len(val_data) > 1 else 0.0

    # Full regression metric breakdown (Train / Val / All) for both heads.
    ea_metrics = {"Train": energy_metrics(train_data), "Val": energy_metrics(val_data), "All": energy_metrics(data)}
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
        ("R²", "R2", f3),
        ("Pearson R", "Pearson", f3),
        ("MAE (kcal/mol)", "MAE", f2),
        ("RMSE (kcal/mol)", "RMSE", f2),
        ("MAPE", "MAPE", fpct),
        ("Count", "n", lambda v: str(int(v))),
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
        fragments = fragments_from_mask(r["geom_mask"]) if "geom_mask" in r else fragments_from_distances(dt, atoms)

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate PSI Results Dashboard")
    parser.add_argument("--data", default="detailed_analysis.json", help="Path to detailed_analysis.json")
    parser.add_argument("--save-dir", default=".", help="Directory to save the HTML dashboard")
    args = parser.parse_args()
    
    create_dashboard(args.data, args.save_dir)

