"""
Experiment 3: Visualizations
Generates all HTML plots from cached experiment 3 results.

Outputs:
  steering_summary.html  — before/after probability bar charts per steering pair
  graded_sweep.html      — P(day) vs rotation angle for the sweep
  ring_overlay.html      — 3D ring with original + steered activation positions

Run with: python3.11 visualize.py
(Must run run_experiment3.py first to generate the .npy / .json caches.)
"""

import json
import numpy as np
from pathlib import Path
from sklearn.decomposition import PCA
from scipy.interpolate import CubicSpline
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import colorsys

# ── Config ─────────────────────────────────────────────────────────────────────
DAYS    = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
DAY_IDX = {d: i for i, d in enumerate(DAYS)}
EXP1_DIR = Path("../experiment1")

day_colors = [
    '#{:02x}{:02x}{:02x}'.format(*[int(c * 255) for c in colorsys.hsv_to_rgb(i / 7, 0.85, 0.88)])
    for i in range(7)
]
DAY_COLOR = dict(zip(DAYS, day_colors))

# ── Load cached data ───────────────────────────────────────────────────────────
with open("steering_results.json") as f:
    steering_results = json.load(f)

with open("graded_sweep.json") as f:
    sweep_results = json.load(f)

ring_basis       = np.load("ring_basis.npy")         # (2, 4096)
centroid_angles  = np.load("centroid_angles.npy")   # (7,)
centroids_4096   = np.load("centroids_4096.npy")    # (7, 4096)
ring_center_ring = np.load("ring_center_ring.npy")  # (2,)
src_acts        = np.load("source_acts.npy")        # (N, 4096)
src_labels      = np.load("source_labels.npy", allow_pickle=True)

with open("source_base_logits.json") as f:
    src_base_logits = json.load(f)


# ── Steering helpers (reproduced for visualize.py) ────────────────────────────

def steer(h, ring_basis, ring_center_ring, src_angle, tgt_angle):
    h_ring   = ring_basis @ h
    h_perp   = h - ring_basis.T @ h_ring
    h_ring_c = h_ring - ring_center_ring
    delta    = tgt_angle - src_angle
    delta    = (delta + np.pi) % (2 * np.pi) - np.pi  # shortest path
    cos_d, sin_d = np.cos(delta), np.sin(delta)
    R = np.array([[cos_d, -sin_d], [sin_d, cos_d]])
    h_ring_new = R @ h_ring_c + ring_center_ring
    return ring_basis.T @ h_ring_new + h_perp


# ── Rebuild 3D projection from experiment 1 pipeline ─────────────────────────
# We need pca1 and pca2 to project activations into the same 3D space as exp 1.

all_acts_exp1 = np.load(EXP1_DIR / "all_acts.npy")  # (84, 4096)
labels_exp1   = np.load(EXP1_DIR / "labels.npy", allow_pickle=True)

N_INTER    = min(64, len(all_acts_exp1))
pca1       = PCA(n_components=N_INTER)
acts_inter = pca1.fit_transform(all_acts_exp1)
cents_64   = np.array([acts_inter[labels_exp1 == d].mean(axis=0) for d in DAYS])

cents_periodic = np.vstack([cents_64, cents_64[0:1]])
spline         = CubicSpline(np.arange(8, dtype=float), cents_periodic, bc_type='periodic')
t_dense        = np.linspace(0, 7, 300)
curve_inter    = spline(t_dense)

pca2          = PCA(n_components=3)
curve_3d      = pca2.fit_transform(curve_inter)
centroids_3d  = pca2.transform(cents_64)
pca2_var      = pca2.explained_variance_ratio_

# Project source activations into 3D
src_inter = pca1.transform(src_acts)
src_3d    = pca2.transform(src_inter)

colorscale_ring = [[i / 6, day_colors[i]] for i in range(7)]


# ── Plot 1: Before/after probability bar chart ────────────────────────────────
print("Building steering_summary.html ...")

n_pairs = len(steering_results)
pair_titles = [f"{r['source_day']} → {r['target_day']}" for r in steering_results]

fig_bar = make_subplots(
    rows=1, cols=n_pairs,
    subplot_titles=pair_titles,
    shared_yaxes=True,
)

for col, pair_result in enumerate(steering_results, start=1):
    src     = pair_result["source_day"]
    tgt     = pair_result["target_day"]
    records = pair_result["records"]
    if not records:
        continue

    avg = lambda key: {d: np.mean([r[key][d] for r in records]) for d in DAYS}
    avg_base    = avg("base_probs")
    avg_steered = avg("steered_probs")
    avg_control = avg("control_probs")

    show_legend = (col == 1)

    # Highlight target day bars with a border
    marker_base    = dict(color='#bbbbbb', line=dict(color='black', width=[2 if d == tgt else 0 for d in DAYS]))
    marker_steered = dict(color=[DAY_COLOR[d] for d in DAYS], line=dict(color='black', width=[2 if d == tgt else 0 for d in DAYS]))
    marker_control = dict(color='#f4a261', line=dict(color='black', width=[2 if d == tgt else 0 for d in DAYS]))

    fig_bar.add_trace(go.Bar(
        name="Base", x=DAYS,
        y=[avg_base[d] for d in DAYS],
        marker=marker_base,
        showlegend=show_legend,
        legendgroup="Base",
    ), row=1, col=col)

    fig_bar.add_trace(go.Bar(
        name="Steered (ring rotation)", x=DAYS,
        y=[avg_steered[d] for d in DAYS],
        marker=marker_steered,
        showlegend=show_legend,
        legendgroup="Steered",
    ), row=1, col=col)

    fig_bar.add_trace(go.Bar(
        name="Control (⊥ plane)", x=DAYS,
        y=[avg_control[d] for d in DAYS],
        marker=marker_control,
        showlegend=show_legend,
        legendgroup="Control",
    ), row=1, col=col)

fig_bar.update_layout(
    barmode='group',
    title=dict(
        text="Manifold Steering — P(day) Before vs After Ring Rotation<br>"
             "<sup>Bars show average probability over day tokens. Bold border = target day.</sup>",
        x=0.5, font=dict(size=14),
    ),
    yaxis_title="P(day)  [softmax over day tokens]",
    height=520,
    width=max(900, 280 * n_pairs),
    legend=dict(orientation="h", y=-0.18, x=0.5, xanchor="center"),
    margin=dict(t=110, b=120, l=60, r=20),
)
fig_bar.write_html("steering_summary.html")
print("  Saved → steering_summary.html")


# ── Plot 2: Graded sweep ──────────────────────────────────────────────────────
print("Building graded_sweep.html ...")

sweep_src = sweep_results["source_day"]
sweep_tgt = sweep_results["target_day"]
steps     = sweep_results["steps"]
prompt    = sweep_results["prompt"]

alphas    = [s["alpha"]    for s in steps]
delta_deg = [s["delta_deg"] for s in steps]

fig_sweep = go.Figure()

for day in DAYS:
    probs     = [s["probs"][day] for s in steps]
    is_key    = day in (sweep_src, sweep_tgt)
    fig_sweep.add_trace(go.Scatter(
        x=delta_deg, y=probs,
        name=day,
        mode='lines+markers',
        line=dict(
            color=DAY_COLOR[day],
            width=3 if is_key else 1.5,
            dash='solid' if is_key else 'dot',
        ),
        marker=dict(size=7 if is_key else 3, color=DAY_COLOR[day]),
        opacity=1.0 if is_key else 0.45,
    ))

# Vertical line at full rotation
if delta_deg:
    fig_sweep.add_vline(
        x=delta_deg[-1],
        line_dash="dash", line_color="#888",
        annotation_text=f"Full Δθ → {sweep_tgt}",
        annotation_position="top left",
        annotation_font_size=11,
    )

fig_sweep.update_layout(
    title=dict(
        text=(f"Graded Steering Sweep: {sweep_src} → {sweep_tgt}<br>"
              f"<sup>P(day) as ring-plane rotation increases from 0 to full Δθ | "
              f"Prompt: \"{prompt}\"</sup>"),
        x=0.5, font=dict(size=13),
    ),
    xaxis_title="Ring-plane rotation applied (degrees)",
    yaxis_title="P(day)  [softmax over day tokens]",
    height=520, width=820,
    legend=dict(x=1.02, y=1, xanchor="left"),
    margin=dict(t=120, r=160),
)
fig_sweep.write_html("graded_sweep.html")
print("  Saved → graded_sweep.html")


# ── Plot 3: Ring overlay — original + steered activation positions ─────────────
print("Building ring_overlay.html ...")

# Use the first steering pair for the overlay
first_pair = steering_results[0]
steer_src  = first_pair["source_day"]
steer_tgt  = first_pair["target_day"]

src_angle = centroid_angles[DAY_IDX[steer_src]]
tgt_angle = centroid_angles[DAY_IDX[steer_tgt]]

# Recompute steered activations (not saved separately — recompute from ring_basis)
src_mask    = src_labels == steer_src
src_subset  = src_acts[src_mask]
src_3d_sub  = src_3d[src_mask]

steered_acts  = np.array([steer(h, ring_basis, ring_center_ring, src_angle, tgt_angle) for h in src_subset])
steered_inter = pca1.transform(steered_acts)
steered_3d    = pca2.transform(steered_inter)

fig_ring = go.Figure()

# Background ring manifold (faded)
fig_ring.add_trace(go.Scatter3d(
    x=curve_3d[:, 0], y=curve_3d[:, 1], z=curve_3d[:, 2],
    mode='lines', name='Ring manifold',
    line=dict(color=t_dense, colorscale=colorscale_ring, width=5, showscale=False),
    opacity=0.35,
    hovertemplate='t=%.2f<extra>Ring</extra>',
))

# Day centroids
fig_ring.add_trace(go.Scatter3d(
    x=centroids_3d[:, 0], y=centroids_3d[:, 1], z=centroids_3d[:, 2],
    mode='markers+text', name='Day centroids',
    text=DAYS, textposition='top center',
    textfont=dict(size=12, color='black', family='Arial Black'),
    marker=dict(size=10, color=day_colors, line=dict(color='black', width=1.5)),
    hovertemplate='%{text}<extra>Centroid</extra>',
))

# Original source activations (circles)
fig_ring.add_trace(go.Scatter3d(
    x=src_3d_sub[:, 0], y=src_3d_sub[:, 1], z=src_3d_sub[:, 2],
    mode='markers',
    name=f'Original ({steer_src} prompts)',
    marker=dict(
        size=8, color=DAY_COLOR[steer_src], symbol='circle',
        line=dict(color='black', width=1.5),
    ),
    hovertemplate=f'{steer_src} (original)<extra></extra>',
))

# Steered activations (diamonds)
fig_ring.add_trace(go.Scatter3d(
    x=steered_3d[:, 0], y=steered_3d[:, 1], z=steered_3d[:, 2],
    mode='markers',
    name=f'Steered (→ {steer_tgt})',
    marker=dict(
        size=8, color=DAY_COLOR[steer_tgt], symbol='diamond',
        line=dict(color='black', width=1.5),
    ),
    hovertemplate=f'Steered → {steer_tgt}<extra></extra>',
))

# Dotted lines connecting each original to its steered counterpart
for orig, strd in zip(src_3d_sub, steered_3d):
    fig_ring.add_trace(go.Scatter3d(
        x=[orig[0], strd[0]],
        y=[orig[1], strd[1]],
        z=[orig[2], strd[2]],
        mode='lines',
        line=dict(color='gray', width=1.5, dash='dot'),
        showlegend=False,
        hoverinfo='skip',
    ))

fig_ring.update_layout(
    title=dict(
        text=(f"Ring Overlay: {steer_src} → {steer_tgt} Steering — Llama 3.1 8B Layer 28<br>"
              f"<sup>Circles = original activations  |  Diamonds = steered activations  |  "
              f"Dotted lines = displacement vector</sup>"),
        x=0.5, font=dict(size=13),
    ),
    scene=dict(
        xaxis_title=f"PC1 ({pca2_var[0]*100:.1f}%)",
        yaxis_title=f"PC2 ({pca2_var[1]*100:.1f}%)",
        zaxis_title=f"PC3 ({pca2_var[2]*100:.1f}%)",
        aspectmode='cube',
        camera=dict(eye=dict(x=1.4, y=1.4, z=0.9)),
    ),
    legend=dict(x=0.02, y=0.98),
    width=920, height=740,
    margin=dict(t=110, b=20, l=20, r=20),
)
fig_ring.write_html("ring_overlay.html")
print("  Saved → ring_overlay.html")

print("\nAll visualizations complete.")
print("Open the HTML files in your browser:")
print("  steering_summary.html  — before/after bar charts")
print("  graded_sweep.html      — rotation sweep line chart")
print("  ring_overlay.html      — 3D ring with steered positions")
