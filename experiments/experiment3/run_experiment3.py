"""
Experiment 3: Manifold Steering
Tests whether the days-of-week ring in Llama 3.1 8B is causally functional:
rotate activations along the ring plane, measure whether model output shifts
toward the target day.

Phases:
  1 — Recover ring geometry from experiment 1 saved data (no API calls)
  2 — Extract source activations + base logits (new unseen prompts)
  3a — Main steering experiment (patched forward passes)
  3b — Graded sweep (sweep rotation angle 0 → full Δθ)
  4  — Print results summary (visualizations in visualize.py)

Run with: python3.11 run_experiment3.py
"""

import os
import json
import time
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
from sklearn.decomposition import PCA
from scipy.interpolate import CubicSpline
from huggingface_hub import login as hf_login
import nnsight
from nnsight import LanguageModel

# ── Config ─────────────────────────────────────────────────────────────────────
DAYS      = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
DAY_IDX   = {d: i for i, d in enumerate(DAYS)}
LAYER_IDX = 28
MODEL_ID  = "meta-llama/Meta-Llama-3.1-8B"
EXP1_DIR  = Path("../experiment1")

HF_TOKEN = os.environ.get("HF_TOKEN", "")
API_KEY  = os.environ.get("NDIF_API_KEY", "")

# Steering pairs to test: (source_day, target_day)
STEERING_PAIRS = [
    ("Tuesday",   "Friday"),     # +3 forward
    ("Monday",    "Thursday"),   # +3 forward
    ("Wednesday", "Sunday"),     # +4 forward
    ("Friday",    "Monday"),     # +3 forward, wraps around
    ("Sunday",    "Wednesday"),  # +3 forward, wraps around
]

SWEEP_PAIR  = ("Tuesday", "Friday")   # which pair to use for graded sweep
SWEEP_STEPS = 20                      # number of rotation steps from 0 → full Δθ

RNG = np.random.default_rng(42)

# ── Steering math ──────────────────────────────────────────────────────────────

def steer(h, ring_basis, ring_center_ring, src_angle, tgt_angle):
    """
    Rotate activation h from src_angle to tgt_angle in the ring plane.
    Rotation is performed around the ring's center (mean of centroid projections),
    not the origin. Components orthogonal to the ring plane are left unchanged.

    Args:
        h                : (4096,) activation vector
        ring_basis       : (2, 4096) orthonormal basis for the ring plane
        ring_center_ring : (2,) center of the ring in the 2D ring plane
        src_angle        : float, angle of source day centroid (relative to ring center)
        tgt_angle        : float, angle of target day centroid (relative to ring center)

    Returns:
        h_steered : (4096,) steered activation vector
    """
    h_ring = ring_basis @ h                         # (2,) — position in ring plane
    h_perp = h - ring_basis.T @ h_ring             # (4096,) — out-of-plane (unchanged)

    # Center relative to ring center before rotating
    h_ring_c = h_ring - ring_center_ring            # (2,)

    # Wrap to shortest path [-π, π] so we always rotate the short way around
    delta    = tgt_angle - src_angle
    delta    = (delta + np.pi) % (2 * np.pi) - np.pi
    cos_d, sin_d = np.cos(delta), np.sin(delta)
    R = np.array([[cos_d, -sin_d],
                  [sin_d,  cos_d]])

    h_ring_c_rot = R @ h_ring_c                     # rotate in centered plane
    h_ring_new   = h_ring_c_rot + ring_center_ring  # translate back

    return ring_basis.T @ h_ring_new + h_perp


def steer_partial(h, ring_basis, ring_center_ring, src_angle, tgt_angle, alpha):
    """Steer by fraction alpha of the full rotation (0 = no change, 1 = full steer)."""
    return steer(h, ring_basis, ring_center_ring,
                 src_angle, src_angle + alpha * (tgt_angle - src_angle))


def out_of_plane_control(h, ring_basis, steer_magnitude, rng):
    """
    Random perturbation orthogonal to the ring plane, with the same L2 magnitude
    as the corresponding steering vector. Used to check whether effects are
    ring-specific or just caused by any perturbation of that size.
    """
    v = rng.standard_normal(h.shape[0]).astype(np.float64)
    # Gram-Schmidt: remove ring-plane components
    for basis_vec in ring_basis:
        v -= np.dot(v, basis_vec) * basis_vec
    v = v / np.linalg.norm(v) * steer_magnitude
    return h + v


def day_softmax(logits_np, day_token_ids):
    """Softmax restricted to the 7 day tokens."""
    vals  = np.array([logits_np[day_token_ids[d]] for d in DAYS])
    probs = np.exp(vals - vals.max())
    probs /= probs.sum()
    return {d: float(p) for d, p in zip(DAYS, probs)}


# ── Phase 1: Recover ring geometry ─────────────────────────────────────────────
print("=" * 64)
print("PHASE 1 — Recover ring geometry from experiment 1 (no API)")
print("=" * 64)

RING_BASIS_CACHE     = Path("ring_basis.npy")
CENT_ANGLES_CACHE    = Path("centroid_angles.npy")
CENTS_4096_CACHE     = Path("centroids_4096.npy")
RING_CENTER_CACHE    = Path("ring_center_ring.npy")

if RING_BASIS_CACHE.exists() and CENT_ANGLES_CACHE.exists() and RING_CENTER_CACHE.exists():
    print("Ring geometry cache found — loading from disk.")
    ring_basis       = np.load(RING_BASIS_CACHE)     # (2, 4096)
    centroid_angles  = np.load(CENT_ANGLES_CACHE)    # (7,)
    centroids_4096   = np.load(CENTS_4096_CACHE)     # (7, 4096)
    ring_center_ring = np.load(RING_CENTER_CACHE)    # (2,)
else:
    print("Re-running experiment 1 PCA pipeline on saved activations...")
    all_acts = np.load(EXP1_DIR / "all_acts.npy")   # (84, 4096)
    labels   = np.load(EXP1_DIR / "labels.npy", allow_pickle=True)

    # PCA step 1: 4096D → 64D
    N_INTER    = min(64, len(all_acts))
    pca1       = PCA(n_components=N_INTER)
    acts_inter = pca1.fit_transform(all_acts)        # (84, 64)

    # Per-day centroids in 64D and 4096D
    centroids_64  = np.array([acts_inter[labels == d].mean(axis=0) for d in DAYS])
    centroids_4096 = np.array([all_acts[labels == d].mean(axis=0)  for d in DAYS])

    # Periodic spline in 64D (same as experiment 1)
    cents_periodic = np.vstack([centroids_64, centroids_64[0:1]])
    spline         = CubicSpline(np.arange(8, dtype=float), cents_periodic, bc_type='periodic')
    curve_inter    = spline(np.linspace(0, 7, 300))  # (300, 64)

    # PCA step 2: fit on spline samples → 3D
    pca2 = PCA(n_components=3)
    pca2.fit(curve_inter)

    # Ring basis in 4096D by composing the two linear PCA maps:
    #   pca2.components_[:2] : (2, 64)  — ring plane directions in 64D
    #   pca1.components_     : (64, 4096) — basis of 64D subspace in 4096D
    ring_basis = pca2.components_[:2] @ pca1.components_   # (2, 4096)

    # Re-orthonormalise to guard against floating-point drift
    ring_basis[1] -= np.dot(ring_basis[1], ring_basis[0]) * ring_basis[0]
    ring_basis    /= np.linalg.norm(ring_basis, axis=1, keepdims=True)

    # Project centroids into the ring plane
    cents_ring       = ring_basis @ centroids_4096.T          # (2, 7)

    # Ring center = mean of centroid projections in the ring plane.
    # We rotate around this center, not the origin, so angles must be
    # computed relative to it — otherwise all centroids cluster near the
    # same angle (they all point roughly the same direction in raw 4096D).
    ring_center_ring = cents_ring.mean(axis=1)                # (2,)
    cents_ring_c     = cents_ring - ring_center_ring[:, np.newaxis]  # (2, 7)
    centroid_angles  = np.arctan2(cents_ring_c[1], cents_ring_c[0])  # (7,)

    np.save(RING_BASIS_CACHE,  ring_basis)
    np.save(CENT_ANGLES_CACHE, centroid_angles)
    np.save(CENTS_4096_CACHE,  centroids_4096)
    np.save(RING_CENTER_CACHE, ring_center_ring)
    print("Ring geometry saved to disk.")

orthogonality = abs(ring_basis[0] @ ring_basis[1])
print(f"\nring_basis shape : {ring_basis.shape}")
print(f"Orthogonality    : {orthogonality:.2e}  (should be ~0)")
print("\nCentroid angles:")
for day, ang in zip(DAYS, centroid_angles):
    print(f"  {day:10s}: {ang:+.4f} rad  ({np.degrees(ang):+7.2f}°)")


# ── Phase 2: Source activation extraction ─────────────────────────────────────
print("\n" + "=" * 64)
print("PHASE 2 — Source activation extraction")
print("=" * 64)

# New templates — all use backward-stepping ("days before") so they don't
# overlap with experiment 1's forward-stepping ("days after") prompts.
SOURCE_TEMPLATES = [
    lambda day, k: f"What day was it {k} days before {day}?",
    lambda day, k: f"If today is {day}, what day was it {k} days ago?",
    lambda day, k: f"Going back {k} days from {day}, which day do you land on?",
]
K_SOURCE = range(1, 4)   # k ∈ {1, 2, 3}

src_prompts, src_labels = [], []
for day_idx_i, day in enumerate(DAYS):
    for k in K_SOURCE:
        for tmpl in SOURCE_TEMPLATES:
            src_prompts.append(tmpl(day, k))
            src_labels.append(DAYS[(day_idx_i - k) % 7])
src_labels = np.array(src_labels)

print(f"Source prompts : {len(src_prompts)}  ({len(src_prompts) // len(DAYS)} per answer-day)")
print(f"First 3 examples:")
for i in range(3):
    print(f"  '{src_prompts[i]}' → {src_labels[i]}")

CACHE_SRC      = Path("source_acts.npy")
CACHE_SRC_LBL  = Path("source_labels.npy")
CACHE_BASE_LGT = Path("source_base_logits.json")
CACHE_TOKEN_IDS = Path("day_token_ids.json")

STEER_RESULTS_CACHE = Path("steering_results.json")
SWEEP_CACHE         = Path("graded_sweep.json")

# Determine if we need the model at all this run
NEED_MODEL = not (
    CACHE_SRC.exists()
    and CACHE_BASE_LGT.exists()
    and STEER_RESULTS_CACHE.exists()
    and SWEEP_CACHE.exists()
)

# Initialise model + tokenizer once if needed
model          = None
day_token_ids  = None

if NEED_MODEL:
    from transformers import AutoTokenizer
    hf_login(token=HF_TOKEN, add_to_git_credential=False)
    nnsight.CONFIG.set_default_api_key(API_KEY)
    model = LanguageModel(MODEL_ID)
    print(f"\nModel ready: {MODEL_ID}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN)
    day_token_ids = {}
    print("Day token IDs (space-prefixed, mid-sentence form):")
    for day in DAYS:
        toks = tokenizer.encode(f" {day}", add_special_tokens=False)
        day_token_ids[day] = int(toks[0])
        print(f"  ' {day}' → token {toks[0]}  (decoded: '{tokenizer.decode([toks[0]])}')")

    with open(CACHE_TOKEN_IDS, "w") as f:
        json.dump(day_token_ids, f, indent=2)

# Load token IDs from cache if model wasn't initialised this run
if day_token_ids is None:
    if CACHE_TOKEN_IDS.exists():
        with open(CACHE_TOKEN_IDS) as f:
            day_token_ids = {k: int(v) for k, v in json.load(f).items()}
    else:
        # Must load tokenizer to get IDs
        from transformers import AutoTokenizer
        hf_login(token=HF_TOKEN, add_to_git_credential=False)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN)
        day_token_ids = {d: int(tokenizer.encode(f" {d}", add_special_tokens=False)[0]) for d in DAYS}
        with open(CACHE_TOKEN_IDS, "w") as f:
            json.dump(day_token_ids, f, indent=2)

# Extract source activations + base logits
if CACHE_SRC.exists() and CACHE_BASE_LGT.exists():
    print("\nSource activation cache found — loading from disk.")
    src_acts       = np.load(CACHE_SRC)
    src_labels     = np.load(CACHE_SRC_LBL, allow_pickle=True)
    with open(CACHE_BASE_LGT) as f:
        src_base_logits = json.load(f)
else:
    print(f"\nExtracting {len(src_prompts)} source activations from layer {LAYER_IDX}...")
    src_acts        = []
    src_base_logits = []

    for i, prompt in enumerate(tqdm(src_prompts, desc="Source extraction")):
        for attempt in range(3):
            try:
                with model.trace(prompt, remote=True):
                    hidden = model.model.layers[LAYER_IDX].output[:, -1, :].save()
                    logits = model.lm_head.output[:, -1, :].save()
                act = hidden.squeeze(0).cpu().float().numpy()
                lgt = logits.squeeze(0).cpu().float().numpy()
                src_acts.append(act)
                src_base_logits.append(day_softmax(lgt, day_token_ids))
                break
            except Exception as e:
                print(f"\n  Prompt {i} attempt {attempt+1} failed: {e}")
                if attempt < 2:
                    time.sleep(5)
                else:
                    raise
        time.sleep(0.25)

    src_acts = np.stack(src_acts)
    np.save(CACHE_SRC,     src_acts)
    np.save(CACHE_SRC_LBL, src_labels)
    with open(CACHE_BASE_LGT, "w") as f:
        json.dump(src_base_logits, f, indent=2)
    print(f"\nSaved: src_acts={src_acts.shape}, {len(src_base_logits)} base logit records")

print(f"\nSource activations: {src_acts.shape}")


# ── Phase 3a: Main steering experiment ────────────────────────────────────────
print("\n" + "=" * 64)
print("PHASE 3a — Main steering experiment")
print("=" * 64)

if STEER_RESULTS_CACHE.exists():
    print("Steering results cache found — loading from disk.")
    with open(STEER_RESULTS_CACHE) as f:
        steering_results = json.load(f)
else:
    steering_results = []

    for src_day, tgt_day in STEERING_PAIRS:
        src_angle = centroid_angles[DAY_IDX[src_day]]
        tgt_angle = centroid_angles[DAY_IDX[tgt_day]]
        delta_deg = np.degrees(tgt_angle - src_angle)

        print(f"\n  Steering {src_day} → {tgt_day}  (Δθ = {delta_deg:.1f}°)")

        mask    = src_labels == src_day
        indices = np.where(mask)[0]
        pair_records = []

        for i in tqdm(indices, desc=f"  {src_day}→{tgt_day}"):
            h          = src_acts[i]
            prompt     = src_prompts[i]
            base_probs = src_base_logits[i]

            # Compute steered and control activations
            h_steered  = steer(h, ring_basis, ring_center_ring, src_angle, tgt_angle)
            steer_mag  = float(np.linalg.norm(h_steered - h))
            h_control  = out_of_plane_control(h, ring_basis, steer_mag, RNG)

            steered_probs = None
            control_probs = None

            for attempt in range(3):
                try:
                    # ── Steered forward pass ──
                    steered_t = torch.from_numpy(h_steered.astype(np.float32))
                    with model.trace(prompt, remote=True):
                        model.model.layers[LAYER_IDX].output[:, -1, :] = steered_t
                        logits_s = model.lm_head.output[:, -1, :].save()
                    steered_probs = day_softmax(
                        logits_s.squeeze(0).cpu().float().numpy(), day_token_ids
                    )
                    time.sleep(0.25)

                    # ── Control forward pass ──
                    control_t = torch.from_numpy(h_control.astype(np.float32))
                    with model.trace(prompt, remote=True):
                        model.model.layers[LAYER_IDX].output[:, -1, :] = control_t
                        logits_c = model.lm_head.output[:, -1, :].save()
                    control_probs = day_softmax(
                        logits_c.squeeze(0).cpu().float().numpy(), day_token_ids
                    )
                    time.sleep(0.25)
                    break

                except Exception as e:
                    print(f"\n    Prompt {i} attempt {attempt+1} failed: {e}")
                    if attempt < 2:
                        time.sleep(5)
                    else:
                        print(f"    Skipping prompt {i} after 3 failures.")

            if steered_probs is not None:
                pair_records.append({
                    "prompt":        prompt,
                    "source_day":    src_day,
                    "target_day":    tgt_day,
                    "base_probs":    base_probs,
                    "steered_probs": steered_probs,
                    "control_probs": control_probs,
                    "steer_mag":     steer_mag,
                })
                p_base    = base_probs[tgt_day]
                p_steered = steered_probs[tgt_day]
                p_control = control_probs[tgt_day]
                print(f"    P({tgt_day}): {p_base:.3f} → {p_steered:.3f}  "
                      f"[control: {p_control:.3f}]")

        steering_results.append({
            "source_day": src_day,
            "target_day": tgt_day,
            "records":    pair_records,
        })

    with open(STEER_RESULTS_CACHE, "w") as f:
        json.dump(steering_results, f, indent=2)
    print(f"\nSaved → {STEER_RESULTS_CACHE}")


# ── Phase 3b: Graded sweep ────────────────────────────────────────────────────
print("\n" + "=" * 64)
print("PHASE 3b — Graded sweep")
print("=" * 64)

sweep_src, sweep_tgt = SWEEP_PAIR

if SWEEP_CACHE.exists():
    print("Sweep cache found — loading from disk.")
    with open(SWEEP_CACHE) as f:
        sweep_results = json.load(f)
else:
    src_angle = centroid_angles[DAY_IDX[sweep_src]]
    tgt_angle = centroid_angles[DAY_IDX[sweep_tgt]]
    alphas    = np.linspace(0, 1, SWEEP_STEPS)

    # Use the first source prompt whose answer is sweep_src
    idx    = int(np.where(src_labels == sweep_src)[0][0])
    h      = src_acts[idx]
    prompt = src_prompts[idx]

    print(f"Sweeping {sweep_src} → {sweep_tgt} in {SWEEP_STEPS} steps")
    print(f"Prompt: '{prompt}'")
    print(f"Full rotation: {np.degrees(tgt_angle - src_angle):.1f}°\n")

    sweep_steps_data = []
    for alpha in tqdm(alphas, desc="Graded sweep"):
        h_partial = steer_partial(h, ring_basis, ring_center_ring, src_angle, tgt_angle, float(alpha))
        partial_t = torch.from_numpy(h_partial.astype(np.float32))
        delta_deg = float(np.degrees(alpha * (tgt_angle - src_angle)))

        probs = None
        for attempt in range(3):
            try:
                with model.trace(prompt, remote=True):
                    model.model.layers[LAYER_IDX].output[:, -1, :] = partial_t
                    logits_p = model.lm_head.output[:, -1, :].save()
                probs = day_softmax(logits_p.squeeze(0).cpu().float().numpy(), day_token_ids)
                break
            except Exception as e:
                if attempt < 2:
                    time.sleep(5)

        if probs is not None:
            sweep_steps_data.append({
                "alpha":     float(alpha),
                "delta_deg": delta_deg,
                "probs":     probs,
            })
        time.sleep(0.25)

    sweep_results = {
        "source_day": sweep_src,
        "target_day": sweep_tgt,
        "prompt":     prompt,
        "steps":      sweep_steps_data,
    }
    with open(SWEEP_CACHE, "w") as f:
        json.dump(sweep_results, f, indent=2)
    print(f"\nSaved → {SWEEP_CACHE}")


# ── Phase 4: Results summary ───────────────────────────────────────────────────
print("\n" + "=" * 64)
print("RESULTS SUMMARY")
print("=" * 64)

all_pass = True
for pair_result in steering_results:
    src     = pair_result["source_day"]
    tgt     = pair_result["target_day"]
    records = pair_result["records"]
    if not records:
        print(f"\n  ⚠  {src} → {tgt}: no records")
        continue

    avg_base    = np.mean([r["base_probs"][tgt]    for r in records])
    avg_steered = np.mean([r["steered_probs"][tgt] for r in records])
    avg_control = np.mean([r["control_probs"][tgt] for r in records])
    lift        = avg_steered / max(avg_base, 1e-8)
    ring_specific = avg_steered > avg_control * 1.5   # steered >> control

    if avg_steered > avg_base * 2 and ring_specific:
        flag = "✓"
    elif avg_steered > avg_base:
        flag = "~"
        all_pass = False
    else:
        flag = "✗"
        all_pass = False

    print(f"\n  {flag} {src} → {tgt}  ({len(records)} prompts)")
    print(f"    P({tgt}) base    : {avg_base:.4f}")
    print(f"    P({tgt}) steered : {avg_steered:.4f}  (×{lift:.1f}×)")
    print(f"    P({tgt}) control : {avg_control:.4f}  "
          f"{'[ring-specific ✓]' if ring_specific else '[not ring-specific ✗]'}")

# Graded sweep summary
steps     = sweep_results["steps"]
p_start   = steps[0]["probs"][sweep_tgt]
p_end     = steps[-1]["probs"][sweep_tgt]
monotone  = all(
    steps[k]["probs"][sweep_tgt] <= steps[k+1]["probs"][sweep_tgt]
    for k in range(len(steps) - 1)
)

print(f"\n  Graded sweep ({sweep_src} → {sweep_tgt}):")
print(f"    P({sweep_tgt}) at α=0  : {p_start:.4f}")
print(f"    P({sweep_tgt}) at α=1  : {p_end:.4f}")
print(f"    Monotone increase: {'✓ Yes' if monotone else '✗ No (check graded_sweep.html)'}")

print("\n" + "=" * 64)
overall = "PASS — ring is causally functional" if all_pass else "PARTIAL — see individual results above"
print(f"Overall: {overall}")
print("=" * 64)
print("\nRun visualize.py to generate HTML plots.")
