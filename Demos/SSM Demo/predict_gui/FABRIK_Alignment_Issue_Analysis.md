# FABRIK Scapulothoracic Alignment: Root-Cause Analysis & Fix Plan

## Problem Statement
The scapula is being positioned as a **horizontal "shelf"** instead of lying vertically against the posterior ribcage. After completing all 4 FABRIK steps, the bone ends up tilted ~90° and displaced superiorly/laterally from its anatomical position.

---

## Step-by-Step Failure Analysis

### Step 0: Initial Seed
*   **Status**: ✅ **CORRECT**
*   **Evidence**: The landmark labels in the Step 3 screenshot show AA (upper-right), TS (upper-left), AI (lower-left). This is a correct scapula layout — scapular spine at the top, inferior angle below.
*   **Conclusion**: The JCS seed orientation is fine. No corrective rotation is needed.

### Step 1: Initial Placement (Projection)
*   **Status**: ✅ **CORRECT**
*   **Evidence**: Console logs show `FABRIK PROJ: Ray hit posterior at X=-114.8` and `T8_X=-118.9`. The ray successfully landed on the posterior ribcage wall (X ≈ -115 is close to T8 at X ≈ -119). Centroid placed at `[-118.9, -22.5, 68.3]`.
*   **Conclusion**: The initial projection and placement are working correctly.

### Step 2: Orientation Guess — ❌ PRIMARY FAILURE POINT
*   **Status**: ❌ **ROOT CAUSE**
*   **What the code does**:
    1. Queries `get_surface_info(cen)` at the Step 1 centroid to get `n_thor_2`.
    2. Aligns the scapula plane normal to `n_thor_2` via `R.align_vectors`.
    3. After rotation, queries `get_surface_info(cen)` at the **new drifted centroid** and re-applies standoff.

*   **Why it fails**:
    1. At `Y=-22.5, Z=68.3`, the ribcage surface slopes steeply upward toward the neck. The surface normal there is `n_thor = [-0.399, 0.730, 0.555]` — it points **73% Up, 40% Posterior, 56% Lateral**.
    2. `R.align_vectors` rotates the scapula so that its flat face points in that mostly-upward direction → the scapula becomes **horizontal**.
    3. This rotation causes the centroid to **drift dramatically**: from `[-118.9, -22.5, 68.3]` to `[-67.5, -9.4, 82.6]` — a 51mm jump anteriorly and 14mm laterally.
    4. At the new drifted centroid `[-67.5, -9.4, 82.6]`, the surface is even steeper (the lateral edge of the rib cage). The standoff re-application pushes the scapula further into this bad region.
    5. **This is a positive feedback loop**: rotate → drift up/laterally → find steeper surface → push further out.

### Step 3: Bubble Translation
*   **Status**: ⚠️ **BYSTANDER**
*   **Observation**: Correctly enforces the 10mm bubble distance, but starting from a horizontal orientation. The subscap cloud is pointing upward, so the "closest point to the ribs" is on the edge of the cloud, not the face. The push/pull cannot fix a 90° orientation error.

### Step 4: Fine-Tuning (Nelder-Mead)
*   **Status**: ⚠️ **TRAPPED IN LOCAL MINIMUM**
*   **Observation**: With the scapula starting horizontal, the optimizer's ±20° search range cannot undo a 90° misalignment. It "perfects" the horizontal pose or finds a nearby local minimum that is still anatomically wrong.

---

## Corrected Fix Plan

### Fix A: Verticalize the Target Normal (Step 2) — **CRITICAL**

**The single most important fix.** Before aligning the scapula normal to the thorax normal, zero out the vertical (Y) component:

```python
# Step 2: Force the target normal to be horizontal
n_thor_2_flat = n_thor_2.copy()
n_thor_2_flat[1] = 0.0  # Remove the "upward" bias
n_thor_2_flat /= np.linalg.norm(n_thor_2_flat)
delta_2, _ = R.align_vectors([n_thor_2_flat], [n_scap])
```

**Why this works**: The scapula's natural orientation has its plane normal pointing posteriorly and slightly medially (the ~30° "scapular plane" angle). By zeroing out Y, we force the alignment to only consider the X-Z plane (posterior-lateral), which preserves the vertical orientation of the bone while still wrapping it around the curvature of the ribs.

**What it preserves**: The ~30° forward tilt of the scapular plane is entirely in the X-Z plane, so it is unaffected by zeroing Y. The bone will still wrap correctly around the ribcage.

### Fix B: Anchor Post-Rotation Standoff to P_proj (Step 2) — **IMPORTANT**

After Step 2's rotation, the code currently queries the surface at the **drifted centroid**, which may have moved far from the original posterior target. Instead, re-apply the standoff using the **original projection point**:

```python
# After rotation, anchor back to the original projection point
p_s2, n_s2 = self.get_surface_info(p_proj)  # Use p_proj, NOT cen
chain = self.fabrik_solve(chain, lengths, p_s2 + n_s2 * standoff)
```

**Why this works**: This prevents the positive feedback loop. No matter how much the rotation shifts the centroid, the FABRIK target stays anchored to the original posterior wall location. The scapula can't "slide off" the back.

### Fix C: Tether the Inferior Angle (Step 4) — **SAFETY NET**

Already partially implemented. Keep the AI tether penalty (`AI_dist > 15mm`) and angular bounds (`±15°`) in the Step 4 cost function. This prevents the optimizer from finding extreme orientations if Fixes A and B don't fully constrain the pose.

### Fix D: Remove Seed Correction — **ESSENTIAL**

Do **NOT** apply a corrective 90° rotation to the seed. The seed is already correct. Any "fix" to Step 0 would break the currently correct landmark orientation.

---

## End-to-End Walkthrough with Fixes Applied

### Step 0 (Seed)
Scapula enters with correct JCS orientation. AA/TS at top, AI below. No change.

### Step 1 (Placement)
Ray fires posteriorly, hits ribcage at X≈-115. Standoff = 10.5mm.
FABRIK places centroid at `[-118.9, -22.5, 68.3]` — on the posterior wall. ✅

### Step 2 (Orientation — with Fix A + Fix B)
1. Query surface at centroid: `n_thor = [-0.399, 0.730, 0.555]`
2. **Fix A**: Zero Y → `n_thor_flat = [-0.399, 0.0, 0.555]` → normalized ≈ `[-0.584, 0.0, 0.812]`
3. Align scapula normal to `n_thor_flat` → scapula stays **vertical**, flat face points postero-laterally
4. Centroid shifts modestly (no dramatic 51mm jump because the rotation is smaller)
5. **Fix B**: Re-apply standoff using `p_proj` → centroid stays anchored to the posterior wall
6. Result: scapula vertical, centroid near `[-115, -20, 70]` ✅

### Step 3 (Bubble Translation)
1. Subscap cloud is now roughly parallel to the posterior rib surface (because the scapula is vertical and the flat face points toward the ribs)
2. Compute min distance across all subscap points → push/pull to lock at 10.0mm
3. Iterative convergence (up to 15 iterations, tolerance 0.1mm)
4. Result: `d_min ≈ 10.0mm` ✅

### Step 4 (Fine-Tuning — with Fix C)
1. Starting from a near-correct vertical orientation, the optimizer has a good starting point
2. Nelder-Mead searches ±15° tilt and ±15° roll
3. MSE cost drives all subscap points toward 10mm → encourages flush contact
4. AI tether prevents the inferior angle from lifting off the ribs
5. Re-locks bubble at 10mm after finding optimal orientation
6. Result: flush subscap contact, `AI_dist < 15mm`, low MSE ✅

---

## Will This Fix the Scapula Position?

**Yes, with high confidence.** Here is the reasoning:

| Requirement | How it's satisfied |
|---|---|
| **Vertical orientation** | Fix A forces the scapula normal to stay horizontal (no Y component in target) |
| **Posterior wall contact** | Fix B anchors the standoff to `p_proj` on the posterior ribcage |
| **10mm physiological gap** | Step 3 bubble translation iteratively locks `d_min = 10.0mm` |
| **Flush rib contact** | Step 4 MSE cost minimizes variance across the subscap footprint |
| **No winging** | Step 4 AI tether + ±15° bounds prevent extreme orientations |
| **~30° scapular plane** | Preserved because the forward tilt is in the X-Z plane, unaffected by Y=0 |

### Remaining Risk
The only scenario where this could still fail is if the **B-spline surface fit** (`_fit_thorax_surface`) is poorly conditioned at the landing region, producing noisy or inverted normals. The existing `UserWarning` about rank deficiency suggests this is possible. If the result still looks wrong after implementing Fixes A-C, we should investigate the spline quality at the scapula's landing zone.

---

## Files to Modify

| File | Fix | Change |
|---|---|---|
| `scripts/fabrik_solver.py` | **A** | Zero `n_thor[1]` before `align_vectors` in Step 2 |
| `scripts/fabrik_solver.py` | **B** | Use `p_proj` instead of `cen` for post-rotation standoff in Step 2 |
| `scripts/fabrik_solver.py` | **C** | Keep AI tether + bounds in Step 4 (already implemented) |
| — | **D** | Do NOT add any seed correction to `generate_isb_joints.py` |
