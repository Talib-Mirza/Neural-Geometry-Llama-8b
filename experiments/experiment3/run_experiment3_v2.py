"""
Experiment 3 v2: Manifold Steering -- Correct Experimental Design
==================================================================

Engels et al. correct setup:
  - Source prompts: those where base model RELIABLY predicts day X
  - Steering target: day Y != X
  - Metric: does model output Y after patching? (steering success rate)

Why this design matters:
  - Previous v2 used "What day was it k days before X?" prompts where
    base model accuracy is only 3.2%. Can't measure steering if model
    doesn't know the answer to begin with.
  - Engels uses prompts where base is reliable, then measures whether
    the ring-plane intervention can redirect the output.

Method:
  - Source: exp1-style prompts "Today is {day}. What day is it?"
    (model reliably outputs the correct day, ~85-100% base accuracy)
  - For each prompt with true day X, steer toward next day Y = (X+1)%7
  - Patch at layers 8, 15, 20, 28
  - Measure: P(output == Y) after patching  [steering success]
  - Baseline: P(output == Y) without patching  [should be ~0%]

Steering formula (ring-plane average ablation):
  h_steered = source_h + ring_basis^T @ (r_Y*c_Y - r_X*c_X)
  [Rotate in ring plane from source day X to target day Y, keep complement]

  OR full average ablation:
  h_steered = (source_h - ring_proj(source_h)) + ring_basis^T @ r_Y*c_Y
  [Replace ring component with target, average-ablate complement from x_mean]
  -- This is what Engels does but requires large ring signal.

  We use a HYBRID: keep source complement, rotate ring component to target.
  This is the cleanest test of whether the ring IS the day representation.

Run with:
  cd experiments/experiment3
  export $(grep -v '^#' ../../app/backend/.env | xargs)
  python3.12 run_experiment3_v2.py
"""

import json
import time
import numpy as np
from pathlib import Path
from sklearn.decomposition import PCA
from scipy.interpolate import CubicSpline
import torch

from nnsight import LanguageModel

# ---- Config ------------------------------------------------------------------
MODEL_ID = "meta-llama/Meta-Llama-3.1-8B"
LAYERS   = [8, 15, 20, 28]
DAYS     = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
DAY_IDX  = {d: i for i, d in enumerate(DAYS)}
EXP1_DIR = Path("../experiment1")
EXP3_DIR = Path(".")
N_INTER  = 64

# ---- Prompts -----------------------------------------------------------------
def make_exp1_prompts():
    """84 prompts (12 per day) where base model is reliable."""
    templates = [
        "Today is {day}. What day is it?",
        "The day is {day}. Which day of the week is today?",
        "It is {day}. What is the current day?",
        "We are on {day}. What day is today?",
        "If today is {day}, what day of the week is it?",
        "On {day}, what day is it?",
        "It's {day}. What day is it?",
        "The current day is {day}. What day is today?",
        "Today being {day}, what day are we on?",
        "On this {day}, what day of the week is it?",
        "Since today is {day}, what day is it?",
        "Given that it is {day}, what is today?",
    ]
    prompts, labels = [], []
    for day in DAYS:
        for tmpl in templates:
            prompts.append(tmpl.format(day=day))
            labels.append(day)
    return prompts, labels

# ---- Ring geometry -----------------------------------------------------------
def build_ring_geometry(acts, labels):
    """Recover ring plane from activations. Returns rb(2,D), rc(2,), ca(7,), c4(7,D), cr(7,)"""
    n_inter = min(N_INTER, len(acts))
    pca1 = PCA(n_components=n_inter)
    acts_inter = pca1.fit_transform(acts)
    centroids_inter = np.array([acts_inter[np.array(labels) == d].mean(0) for d in DAYS])
    centroids_D     = np.array([acts[np.array(labels) == d].mean(0)       for d in DAYS])
    cents_periodic  = np.vstack([centroids_inter, centroids_inter[0:1]])
    spline = CubicSpline(np.arange(8, dtype=float), cents_periodic, bc_type='periodic')
    curve_inter = spline(np.linspace(0, 7, 300))
    pca2 = PCA(n_components=3)
    pca2.fit(curve_inter)
    ring_basis = pca2.components_[:2] @ pca1.components_
    ring_basis[1] -= np.dot(ring_basis[1], ring_basis[0]) * ring_basis[0]
    ring_basis    /= np.linalg.norm(ring_basis, axis=1, keepdims=True)
    cents_ring     = ring_basis @ centroids_D.T
    ring_center    = cents_ring.mean(axis=1)
    cents_centered = cents_ring - ring_center[:, np.newaxis]
    centroid_angles = np.arctan2(cents_centered[1], cents_centered[0])
    centroid_radii  = np.linalg.norm(cents_centered, axis=0)
    return ring_basis, ring_center, centroid_angles, centroids_D, centroid_radii

# ---- Steering ----------------------------------------------------------------
def steer_ring_rotate(source_h, src_day_idx, tgt_day_idx,
                      ring_basis, ring_center, centroid_angles, centroid_radii):
    """
    Rotate source activation from src_day position to tgt_day position in ring plane.
    Keeps the complement (out-of-plane) from the source activation.

    This is the cleanest test: only ring-plane component changes.
    delta = (r_tgt * c_tgt) - (r_src * c_src)
    h_steered = source_h + ring_basis^T @ delta
    """
    r_src = centroid_radii[src_day_idx]
    a_src = centroid_angles[src_day_idx]
    r_tgt = centroid_radii[tgt_day_idx]
    a_tgt = centroid_angles[tgt_day_idx]

    src_ring = r_src * np.array([np.cos(a_src), np.sin(a_src)])
    tgt_ring = r_tgt * np.array([np.cos(a_tgt), np.sin(a_tgt)])
    delta_ring = tgt_ring - src_ring                    # (2,)

    return (source_h + ring_basis.T @ delta_ring).astype(np.float32)

# ---- Day token IDs ----------------------------------------------------------
with open(EXP3_DIR / "day_token_ids.json") as f:
    day_token_ids_dict = json.load(f)
DAY_TOKEN_IDS = [day_token_ids_dict[d] for d in DAYS]

def day_probs_from_logits(logits_tensor):
    if isinstance(logits_tensor, torch.Tensor):
        logits_np = logits_tensor.detach().float().cpu().numpy()
    else:
        logits_np = np.array(logits_tensor, dtype=np.float32)
    logits_np  = logits_np.squeeze()
    day_logits = logits_np[DAY_TOKEN_IDS]
    day_logits = day_logits - day_logits.max()
    exp_l = np.exp(day_logits)
    return dict(zip(DAYS, (exp_l / exp_l.sum()).tolist()))

# =============================================================================
# Phase 1: Load exp1 cached data + ring geometry (no API)
# =============================================================================
print("=" * 70)
print("Phase 1: Load ring geometry (layer 28, no API)")
print("=" * 70)

exp1_acts_28   = np.load(EXP1_DIR / "all_acts.npy")
exp1_labels_28 = np.load(EXP1_DIR / "labels.npy", allow_pickle=True)
exp1_prompts, exp1_labels_list = make_exp1_prompts()

# Cached ring geometry from v1
ring_basis      = np.load(EXP3_DIR / "ring_basis.npy")
ring_center     = np.load(EXP3_DIR / "ring_center_ring.npy")
centroid_angles = np.load(EXP3_DIR / "centroid_angles.npy")
centroids_4096  = np.load(EXP3_DIR / "centroids_4096.npy")
cents_ring      = ring_basis @ centroids_4096.T
cents_centered  = cents_ring - ring_center[:, np.newaxis]
centroid_radii  = np.linalg.norm(cents_centered, axis=0)

print(f"  Centroid angles (°): {np.degrees(centroid_angles).round(1).tolist()}")
print(f"  Centroid radii:      {centroid_radii.round(3).tolist()}")

# Verify ring geometry gives correct day ordering (should be roughly circular)
test_delta = steer_ring_rotate(
    exp1_acts_28[0], 0, 1,   # Monday → Tuesday
    ring_basis, ring_center, centroid_angles, centroid_radii
)
print(f"  Ring delta norm (Mon→Tue): {np.linalg.norm(test_delta - exp1_acts_28[0]):.4f}")
print()

# =============================================================================
# Phase 2: Extract layer acts at 8, 15, 20 for exp1 prompts
# =============================================================================
print("=" * 70)
print("Phase 2: Extract / load exp1 acts at layers 8, 15, 20")
print("=" * 70)

LAYERS_NEW = [8, 15, 20]
need_exp1  = any(not (EXP3_DIR / f"exp1_acts_layer{l}.npy").exists() for l in LAYERS_NEW)
exp1_acts_new = {}

if not need_exp1:
    print("  Layer 8/15/20 exp1 act caches found -- loading.")
    for l in LAYERS_NEW:
        exp1_acts_new[l] = np.load(EXP3_DIR / f"exp1_acts_layer{l}.npy")
else:
    model = LanguageModel(MODEL_ID, device_map="auto")

    def extract_multi_layer(prompts, layers_wanted, desc=""):
        n = len(prompts)
        acts_by_layer = {l: np.zeros((n, 4096), dtype=np.float32) for l in layers_wanted}
        for i, prompt in enumerate(prompts):
            if (i + 1) % 10 == 0 or i == 0:
                print(f"    [{i+1}/{n}] {desc}: {prompt[:55]}...")
            for attempt in range(3):
                try:
                    s8 = s15 = s20 = None
                    with model.trace(prompt, remote=True):
                        if 8  in layers_wanted: s8  = model.model.layers[8].output[:, -1, :].save()
                        if 15 in layers_wanted: s15 = model.model.layers[15].output[:, -1, :].save()
                        if 20 in layers_wanted: s20 = model.model.layers[20].output[:, -1, :].save()
                    if 8  in layers_wanted: acts_by_layer[8][i]  = s8.squeeze(0).float().cpu().numpy()
                    if 15 in layers_wanted: acts_by_layer[15][i] = s15.squeeze(0).float().cpu().numpy()
                    if 20 in layers_wanted: acts_by_layer[20][i] = s20.squeeze(0).float().cpu().numpy()
                    break
                except Exception as e:
                    print(f"    Prompt {i} attempt {attempt+1} failed: {e}")
                    if attempt < 2: time.sleep(5)
                    else: raise
            time.sleep(0.2)
        return acts_by_layer

    print(f"  Extracting layers {LAYERS_NEW} for {len(exp1_prompts)} exp1 prompts...")
    exp1_acts_new = extract_multi_layer(exp1_prompts, LAYERS_NEW, desc="exp1")
    for l in LAYERS_NEW:
        np.save(EXP3_DIR / f"exp1_acts_layer{l}.npy", exp1_acts_new[l])

print()

# =============================================================================
# Phase 2b: Run base forward passes on exp1 prompts (get base accuracy)
# =============================================================================
print("=" * 70)
print("Phase 2b: Base forward passes to establish ground truth")
print("=" * 70)

BASE_CACHE = EXP3_DIR / "exp1_base_logits.json"
if BASE_CACHE.exists():
    print("  Loading cached base logits...")
    with open(BASE_CACHE) as f:
        exp1_base_logits = json.load(f)
else:
    if 'model' not in dir():
        model = LanguageModel(MODEL_ID, device_map="auto")
    exp1_base_logits = []
    print(f"  Running {len(exp1_prompts)} base forward passes...")
    for i, prompt in enumerate(exp1_prompts):
        if (i + 1) % 15 == 0 or i == 0:
            print(f"    [{i+1}/{len(exp1_prompts)}] {prompt[:55]}...")
        for attempt in range(3):
            try:
                with model.trace(prompt, remote=True):
                    logits = model.lm_head.output[:, -1, :].save()
                exp1_base_logits.append(day_probs_from_logits(logits))
                break
            except Exception as e:
                print(f"    Prompt {i} attempt {attempt+1} failed: {e}")
                if attempt < 2: time.sleep(5)
                else: raise
        time.sleep(0.2)
    with open(BASE_CACHE, "w") as f:
        json.dump(exp1_base_logits, f)

# Base accuracy
base_correct = sum(1 for probs, lbl in zip(exp1_base_logits, exp1_labels_list)
                   if max(probs, key=probs.get) == lbl)
print(f"  Base model accuracy on 84 exp1 prompts: {base_correct}/84 = {base_correct/84*100:.1f}%")

# Filter to prompts where base model is CORRECT (steering from known state)
correct_idx = [i for i, (probs, lbl) in enumerate(zip(exp1_base_logits, exp1_labels_list))
               if max(probs, key=probs.get) == lbl]
print(f"  Using {len(correct_idx)} prompts where base model is correct")
print()

# =============================================================================
# Phase 2c: Build ring geometry at each layer
# =============================================================================
print("=" * 70)
print("Phase 2c: Build ring geometry at each layer")
print("=" * 70)

all_acts_by_layer = {
    8:  exp1_acts_new[8],
    15: exp1_acts_new[15],
    20: exp1_acts_new[20],
    28: exp1_acts_28,
}

geom_by_layer = {}
for layer_idx in LAYERS:
    acts = all_acts_by_layer[layer_idx]
    labels = np.array(exp1_labels_list) if layer_idx != 28 else exp1_labels_28
    rb, rc, ca, c4, cr = build_ring_geometry(acts, labels)
    geom_by_layer[layer_idx] = (rb, rc, ca, c4, cr)
    print(f"  Layer {layer_idx:2d}: angle spread={np.degrees(ca.max()-ca.min()):.1f}°, "
          f"mean radius={cr.mean():.4f}")

print()

# =============================================================================
# Phase 3: Patched forward passes -- steer exp1 prompts to NEXT day
# =============================================================================
print("=" * 70)
print("Phase 3: Patched steering -- exp1 prompts, steer to next day in week")
print("=" * 70)
print("  Steering: day X -> (X+1)%7   [next day]")
print()

if 'model' not in dir():
    model = LanguageModel(MODEL_ID, device_map="auto")

results_by_layer = {}

for layer_idx in LAYERS:
    rb, rc, ca, c4, cr = geom_by_layer[layer_idx]
    src_acts_layer = all_acts_by_layer[layer_idx]
    records = []

    print(f"  Layer {layer_idx}: patching {len(correct_idx)} prompts (base-correct subset)...")
    for i in correct_idx:
        prompt    = exp1_prompts[i]
        src_label = exp1_labels_list[i]
        src_idx   = DAY_IDX[src_label]
        tgt_idx   = (src_idx + 1) % 7
        tgt_label = DAYS[tgt_idx]

        source_h  = src_acts_layer[i].astype(np.float64)
        h_steered = steer_ring_rotate(source_h, src_idx, tgt_idx, rb, rc, ca, cr)
        steered_t = torch.from_numpy(h_steered)

        for attempt in range(3):
            try:
                with model.trace(prompt, remote=True):
                    model.model.layers[layer_idx].output[:, -1, :] = steered_t
                    logits_patched = model.lm_head.output[:, -1, :].save()
                break
            except Exception as e:
                print(f"    Prompt {i} attempt {attempt+1} failed: {e}")
                if attempt < 2: time.sleep(5)
                else: raise
        time.sleep(0.2)

        probs   = day_probs_from_logits(logits_patched)
        top1    = max(probs, key=probs.get)
        steered = top1 == tgt_label      # did it steer to target?
        stayed  = top1 == src_label      # did it stay at source?

        records.append({
            "prompt":     prompt,
            "src_label":  src_label,
            "tgt_label":  tgt_label,
            "top1":       top1,
            "steered":    steered,
            "stayed":     stayed,
            "probs":      probs,
        })

    n = len(records)
    n_steered = sum(r["steered"] for r in records)
    n_stayed  = sum(r["stayed"]  for r in records)
    steer_rate = n_steered / n
    results_by_layer[layer_idx] = {
        "steer_rate": steer_rate,
        "n_steered":  n_steered,
        "n_stayed":   n_stayed,
        "n_total":    n,
        "records":    records,
    }
    print(f"  --> Layer {layer_idx:2d}: steered={n_steered}/{n}={steer_rate*100:.1f}%,  "
          f"stayed={n_stayed}/{n}={n_stayed/n*100:.1f}%")

print()

# =============================================================================
# Phase 4: Results table + save
# =============================================================================
print("=" * 70)
print("RESULTS -- Steering success rate (output == target day) by layer")
print("=" * 70)
print(f"  Baseline (no patch):  0/N = 0%   [base model was correct → source day]")
print(f"  Random chance:        1/7 = 14.3%")
print(f"  Engels Llama 3 8B:    ~59% (next-day steering)")
print()
print(f"{'Layer':>6}  {'Steered':>8}  {'Stayed':>8}  {'Total':>6}  {'Steer%':>8}  {'Stay%':>7}")
print("-" * 52)
for layer_idx in LAYERS:
    r = results_by_layer[layer_idx]
    print(f"{layer_idx:>6}  {r['n_steered']:>8}  {r['n_stayed']:>8}  "
          f"{r['n_total']:>6}  {r['steer_rate']*100:>7.1f}%  "
          f"{r['n_stayed']/r['n_total']*100:>6.1f}%")
print("-" * 52)

best_layer = max(LAYERS, key=lambda l: results_by_layer[l]["steer_rate"])
print(f"\nBest layer: {best_layer} ({results_by_layer[best_layer]['steer_rate']*100:.1f}% steering success)")

# Per-source-day breakdown at best layer
print(f"\nPer-source-day breakdown at layer {best_layer}:")
day_steered = {d: 0 for d in DAYS}
day_total   = {d: 0 for d in DAYS}
for r in results_by_layer[best_layer]["records"]:
    day_total[r["src_label"]]   += 1
    day_steered[r["src_label"]] += int(r["steered"])
print(f"  {'Source':12} -> {'Target':12}  {'Steer':>6} {'Total':>6} {'%':>6}")
print(f"  {'-'*48}")
for d in DAYS:
    tgt = DAYS[(DAY_IDX[d]+1)%7]
    if day_total[d] > 0:
        pct = day_steered[d] / day_total[d] * 100
        print(f"  {d:12} -> {tgt:12}  {day_steered[d]:>6} {day_total[d]:>6} {pct:>5.1f}%")

output = {
    "method": "ring_rotate_steer",
    "steering_direction": "next_day",
    "layers": LAYERS,
    "best_layer": best_layer,
    "results_by_layer": {
        str(l): {k: v for k, v in results_by_layer[l].items() if k != "records"}
        for l in LAYERS
    }
}
with open(EXP3_DIR / "v2_results.json", "w") as f:
    json.dump(output, f, indent=2)

print(f"\nSaved -> v2_results.json")
print("Done.")
