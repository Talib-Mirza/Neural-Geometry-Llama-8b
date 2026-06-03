# Experiment 3: Manifold Steering

Using the ring geometry discovered in experiments 1 & 2 to *intervene* on Llama 3.1 8B's
internal representations — rotating activations along the days-of-week ring and verifying
that the model's output changes accordingly.

---

## Goal

Experiments 1 and 2 showed that the model *has* a ring. This experiment asks whether that
ring is *functional*: if you surgically rotate a "Tuesday" activation toward "Friday" inside
the model's residual stream, does the model then behave as if it's thinking about Friday?

The hypothesis is yes — and if it is, it means the ring isn't just a geometric artifact of
averaging. It's the actual representational substrate the model uses for day-of-week reasoning.

---

## High-Level Idea

1. **Recover the ring subspace** from experiment 1's saved activations — specifically, the
   2D plane in 4096D space that the ring lives in.
2. **Pick a source prompt** — a new, unseen prompt whose correct answer is some source day
   (e.g. "Tuesday").
3. **Extract its layer-28 activation** and project it onto the ring subspace.
4. **Rotate** the ring-subspace component by the angular difference between the source day
   centroid and a target day centroid (e.g. Tuesday → Friday).
5. **Reconstruct** the full activation: rotated ring component + unchanged out-of-plane component.
6. **Patch** the modified activation back into the model at layer 28 using nnsight's
   intervention API.
7. **Measure** whether the model's output (next-token probabilities or generated text) shifts
   toward the target day.

---

## The Ring Subspace: How to Recover It in 4096D

The pipeline in experiments 1 & 2 produces geometry in a *compressed* space (64D then 3D),
not the original 4096D space. To steer, we need the ring's basis vectors back in 4096D so
we can manipulate actual model activations.

The two PCA stages create a composed linear map:

```
4096D --[pca1]--> 64D --[pca2]--> 3D
```

Both PCA stages are linear, so their composition is also linear. The ring lives in the top
2 dimensions of pca2 (PC1 and PC2 span the ring plane). We can express those 2 directions
directly in 4096D:

```python
# pca1.components_ is shape (64, 4096)  — 64 basis vectors in 4096D
# pca2.components_ is shape (3, 64)    — 3 basis vectors in 64D

# The 2D ring basis in 4096D:
ring_basis = pca2.components_[:2] @ pca1.components_  # shape (2, 4096)
```

`ring_basis[0]` and `ring_basis[1]` are the two orthogonal directions in 4096D that span
the ring plane. These are the only directions we touch during steering. Everything orthogonal
to this plane is left exactly as-is.

---

## The Steering Operation: Step by Step

Given a source activation `h` (shape `(4096,)`) from a layer-28 hook, and a target day:

### Step 1 — Project onto ring plane
```python
# Coordinates of h in the ring plane (a 2D vector)
h_ring = ring_basis @ h          # shape (2,)

# Component of h outside the ring plane (unchanged during steering)
h_perp = h - ring_basis.T @ h_ring   # shape (4096,)
```

### Step 2 — Find source and target angles
Each day centroid has a known location in the ring plane. Compute the angle of each:

```python
# Project centroids into the ring plane
cents_ring = ring_basis @ centroids.T   # shape (2, 7)

source_angle = np.arctan2(cents_ring[1, source_idx], cents_ring[0, source_idx])
target_angle = np.arctan2(cents_ring[1, target_idx], cents_ring[0, target_idx])

delta_theta = target_angle - source_angle   # rotation to apply
```

### Step 3 — Rotate in the ring plane
```python
cos_t, sin_t = np.cos(delta_theta), np.sin(delta_theta)
R = np.array([[cos_t, -sin_t],
              [sin_t,  cos_t]])   # 2x2 rotation matrix

h_ring_rotated = R @ h_ring      # shape (2,)
```

### Step 4 — Reconstruct the full activation
```python
h_steered = ring_basis.T @ h_ring_rotated + h_perp   # shape (4096,)
```

`h_steered` is the intervention vector: same as `h` everywhere except it has been rotated
inside the ring plane.

### Step 5 — Patch into the model
Using nnsight's intervention API at layer 28:

```python
with model.trace(source_prompt, remote=True):
    # Read the current output at layer 28
    hidden = model.model.layers[28].output[:, -1, :]

    # Overwrite with the steered activation
    model.model.layers[28].output[:, -1, :] = torch.tensor(h_steered)

    # Collect the model's next-token logits
    logits = model.lm_head.output[:, -1, :].save()
```

---

## Validation Protocol

Three levels of evidence, in increasing strength:

### Level 1 — Next-token probability shift (primary metric)
For a prompt like *"What day is 2 days after Sunday?"* (correct answer: Tuesday), measure
the softmax probability of each day token before and after steering toward Friday:

```
Before steering:  P(Monday)=0.02, P(Tuesday)=0.71, ..., P(Friday)=0.04
After steering:   P(Monday)=0.03, P(Tuesday)=0.09, ..., P(Friday)=0.61
```

A successful intervention shows the target day's probability rising substantially while the
source day's probability falls.

### Level 2 — Geometric verification
After patching, re-extract the actual layer-28 activation and check that it moved in the
expected direction:

- Distance from `h_steered` to target centroid < distance from `h_original` to target centroid
- Distance from `h_steered` to source centroid > distance from `h_original` to source centroid
- The angular position of `h_steered` in the ring plane is close to `target_angle`

### Level 3 — Graded steering (sweep across rotation angles)
Instead of jumping all the way to the target centroid's angle, sweep delta_theta from 0 to
the full rotation in small steps. Plot P(target day) vs delta_theta. A functional ring should
show a monotonic increase. This is the cleanest test that the geometric structure is causally
connected to model behavior.

### Control — Out-of-plane perturbation
Apply a perturbation of the same magnitude as the steering vector, but in a random direction
*orthogonal* to the ring plane. If the ring is causally special, this control perturbation
should *not* shift day-token probabilities. If it does, the effect isn't ring-specific.

---

## Configuration

| Parameter | Value | Notes |
|-----------|-------|-------|
| **Model** | `meta-llama/Meta-Llama-3.1-8B` | Same as experiments 1 & 2 |
| **Layer** | 28 | Same layer where the ring is cleanest |
| **Source activations** | Loaded from `experiment1/all_acts.npy` | Reuse saved data, no re-extraction needed |
| **Ring basis** | Recomputed from `experiment1/all_acts.npy` | Run the same PCA pipeline, save `ring_basis (2, 4096)` |
| **Source prompts** | ~5 new unseen prompts per source day | Must not overlap with experiment 1 prompts |
| **Steering pairs** | (Tuesday→Friday), (Monday→Thursday), (Wednesday→Sunday) | Tests both short and long rotations |
| **Token vocabulary** | `["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]` | Measure logits over exactly these 7 tokens |
| **Graded sweep** | 20 steps from 0 to delta_theta | For Level 3 validation |

---

## Implementation Plan

### Phase 1 — Recover geometry (no new API calls needed)
- Load `experiment1/all_acts.npy` and `experiment1/labels.npy`
- Re-run the experiment 1 PCA pipeline to recover `pca1` and `pca2` objects
- Compute `ring_basis = pca2.components_[:2] @ pca1.components_`
- Project all 7 centroids into the ring plane, compute their angles
- Save: `ring_basis.npy`, `centroid_angles.npy`

### Phase 2 — Extract source activations
- Write 5 new unseen prompts per day (35 total) — different templates from experiment 1
- Extract layer-28 activations for each via nnsight
- Save: `source_acts.npy`, `source_labels.npy`

### Phase 3 — Run steering + measure
- For each (source prompt, target day) pair:
  - Compute `h_steered` via the rotation math above
  - Run patched forward pass with nnsight
  - Collect logits over the 7 day tokens
  - Record before/after probabilities
- Run the out-of-plane control for the same prompts
- Run the graded sweep for a representative subset (one prompt per source day)

### Phase 4 — Visualize
- Bar chart: before/after day probabilities for each steering pair
- Line chart: P(target day) vs delta_theta for graded sweep
- 3D scatter: original vs steered activation position on the ring (overlay on experiment 1 plot)

---

## Output Files

| File | Description |
|------|-------------|
| `README.md` | This file |
| `run_experiment3.py` | Main script — phases 1–3 |
| `ring_basis.npy` | Ring subspace basis vectors `(2, 4096)` |
| `centroid_angles.npy` | Angular position of each day centroid on the ring `(7,)` |
| `source_acts.npy` | New source prompt activations `(35, 4096)` |
| `source_labels.npy` | Source day labels `(35,)` |
| `steering_results.json` | Before/after logits for all steering pairs |
| `visualize.py` | Plotting script for all figures |
| `steering_summary.html` | Interactive before/after probability chart |
| `graded_sweep.html` | P(target) vs rotation angle plot |
| `ring_overlay.html` | 3D ring with original + steered activation positions |

---

## What Success Looks Like

The experiment is a success if:
1. Steering consistently raises the target day's probability by a large margin (>3× baseline)
2. The graded sweep shows a monotonic P(target) increase as delta_theta increases
3. The out-of-plane control shows no significant day-probability shift

If (1) holds but (3) fails, the effect is real but not ring-specific — the model is sensitive
to any perturbation at that layer, not specifically to ring-plane rotations. That would still
be interesting but would be a weaker result.

If (1) and (3) hold but (2) fails (non-monotonic sweep), the ring angle doesn't linearly
correspond to the model's day-belief — the geometry exists but isn't used the way we think.

---

## Open Questions Going In

- **Scale of intervention**: How large does the rotation need to be to have a detectable
  effect? The ring radius (in 4096D) determines this — a very small-radius ring would require
  tiny perturbations, a large-radius ring larger ones.
- **Layer specificity**: Does steering at layer 28 work better than layer 20 or 31? Could
  run the same intervention at other layers as a secondary test.
- **Prompt dependence**: Does steering work equally well regardless of the source prompt's
  surface form, or only for prompts that already activate a strong day representation?
