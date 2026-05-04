# FABRIK Scapulothoracic Alignment Algorithm

This document outlines the multi-step kinematic procedure for aligning the scapula to the thorax using the FABRIK (Forward And Backward Reaching Inverse Kinematics) solver.
This algorithm should start from the initial assembled position of the bones.

## 1. Geometric Definitions

*   **SC Joint ($J_{SC}$)**: Sternoclavicular joint, acting as the fixed root of the kinematic chain.
*   **AC Joint ($J_{AC}$)**: Acromioclavicular joint, connecting the clavicle and scapula.
*   **Scapula Centroid ($C_{scap}$)**: The geometric center of the scapular plane, defined as the average of landmarks **AA** (Angulus Acromialis), **TS** (Trigonum Spinae), and **AI** (Angulus Inferior).
*   **Scapula Plane Normal ($\mathbf{n}_{scap}$)**: The unit vector perpendicular to the plane defined by AA, TS, and AI, oriented so it points **away** from the subscapularis fossa (i.e., toward the dorsal/posterior surface of the scapula). This matches the convention of $\mathbf{n}_{thor}$, so that $\mathbf{n}_{scap} \parallel \mathbf{n}_{thor}$ means the subscapularis face is correctly oriented toward the ribs.
*   **Subscapularis Point Cloud**: In the `MAS_103` dataset, the subscapularis attachment site corresponds to point cloud ID **69** (`69_NodeNo_2.csv` / `69_NodeNo_2.ply`).
*   **Thorax Normal ($\mathbf{n}_{thor}$)**: The outward surface normal of the thoracic ribcage at a specific point, pointing posteriorly (away from the ribs, toward the scapula).
*   **Projected Point ($P_{proj}$)**: The target intersection point on the thorax surface calculated by projecting $C_{scap}$ along $\mathbf{n}_{scap}$ toward the ribcage.

## 2. Kinematic Chain
The assembly is treated as a two-link chain with fixed segment lengths:
1.  **Clavicle Link**: Length $L_{clav} = ||J_{AC} - J_{SC}||$
2.  **Scapula Link**: Length $L_{scap} = ||C_{scap} - J_{AC}||$

The FABRIK solver iteratively updates the positions of $J_{AC}$ and $C_{scap}$ to satisfy target constraints while preserving these lengths.

## 3. Gap Convention
All gap distances are measured from the **bone surface** (subscapularis fossa), not from the centroid. In practice, the centroid hovers farther from the ribcage than the bone surface. The algorithm estimates the bone-surface offset as half the scapular plane thickness (~5 mm) and adds it to the target gap when positioning the centroid.

---

## 4. The Four-Step Alignment Procedure

### Step 1: Position with Built-In Gap
*   **Objective**: Place the scapula at its target location on the posterior-lateral thorax with a physiological standoff gap.
*   **Projection**: From the seed pose, cast a ray from $C_{scap}$ along $\mathbf{n}_{scap}$ to find $P_{proj}$ on the thorax surface. Extract $\mathbf{n}_{thor}(P_{proj})$.
*   **Target Position**: Compute $T = P_{proj} + (d_{gap} + d_{bone}) \cdot \mathbf{n}_{thor}$, where $d_{gap} = 5\text{ mm}$ (subscapularis-to-rib clearance) and $d_{bone} = 5\text{ mm}$ (half scapular thickness estimate). This places the centroid ~10 mm from the thorax surface.
*   **IK Solver**: Run FABRIK to find the $J_{AC}$ position that satisfies the reach from $J_{SC}$ to $T$ while preserving $L_{clav}$ and $L_{scap}$.

### Step 2: Orient Scapula (Tangency Alignment)
*   **Objective**: Align the scapula plane tangentially to the local thorax curvature so the subscapularis fossa faces the ribs.
*   **Target Orientation**: Identify the thorax surface point $P_{close}$ closest to the current $C_{scap}$. Extract $\mathbf{n}_{thor}(P_{close})$.
*   **Mechanism**: Rotate the scapula around the $J_{AC}$ pivot so that $\mathbf{n}_{scap}$ is parallel to $\mathbf{n}_{thor}(P_{close})$.

### Step 3: Penetration Check & Correction
*   **Objective**: Verify that no part of the scapula intersects the thoracic volume.
*   **Detection**: Evaluate the signed distance of key landmarks (AA, TS, AI, mid-medial-border) from the thorax surface. A negative distance indicates penetration.
*   **Correction**: If any landmark penetrates, push $C_{scap}$ outward along $\mathbf{n}_{thor}$ by the penetration depth plus a 2 mm margin, then re-run FABRIK and re-apply the orientation from Step 2.
*   **Constraint**: The chain $J_{SC} \to J_{AC} \to C_{scap}$ must remain connected throughout.

### Step 4: Medial Border Clearance (Fine-Tune)
*   **Objective**: Ensure the medial border hovers at a physiologically appropriate distance from the ribcage.
*   **Target Clearance**: **TS** and **AI** should be approximately **5 mm** above the thoracic sliding surface (matching typical serratus anterior thickness).
*   **Mechanism**: Roll/tilt the scapula around the $J_{AC} \to C_{scap}$ axis. This adjusts only the bone's roll without changing the centroid's standoff distance.
*   **Constraint**: The centroid gap established in Steps 1-3 is preserved; only the orientation around the principal axis changes.
