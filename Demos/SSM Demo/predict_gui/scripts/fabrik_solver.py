import numpy as np
from scipy.interpolate import LSQBivariateSpline
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as R


class FabrikScapulaSolver:
    """
    4-step FABRIK Scapulothoracic Alignment.
    Chain: SC (fixed) -> AC -> Scapula Centroid

    The Subscapularis point cloud (ID 69) acts as a physiological 'bubble':
    the closest point of the cloud to the thorax must equal SUBSCAP_BUBBLE_MM.
    """

    SUBSCAP_BUBBLE_MM = 10.0   # target min clearance: subscap cloud → thorax
    MEDIAL_CLEARANCE  = 5.0    # legacy fallback (no subscap cloud)

    def __init__(self, thorax_mesh, sc_joint):
        self.sc_joint    = np.array(sc_joint,    dtype=float)
        self.thorax_mesh = np.array(thorax_mesh, dtype=float)
        self.spline      = None
        self._fit_thorax_surface()

    # ── Surface fitting ──────────────────────────────────────────────────────

    def _fit_thorax_surface(self):
        """Fits B-Spline X=f(Y,Z) to the posterior thorax glide area."""
        pts   = self.thorax_mesh
        x_mid = (pts[:, 0].min() + pts[:, 0].max()) / 2.0
        y_min = pts[:, 1].min() + (pts[:, 1].max() - pts[:, 1].min()) * 0.2
        mask  = (pts[:, 0] < x_mid + 20) & (pts[:, 1] > y_min)
        glide = pts[mask]
        if len(glide) < 100:
            print("FABRIK WARNING: glide filter too strict – using full mesh.")
            glide = pts
        y_pts, z_pts, x_pts = glide[:, 1], glide[:, 2], glide[:, 0]
        ky = np.linspace(y_pts.min(), y_pts.max(), 10)[1:-1]
        kz = np.linspace(z_pts.min(), z_pts.max(), 10)[1:-1]
        self.spline = LSQBivariateSpline(y_pts, z_pts, x_pts, ky, kz, kx=3, ky=3)
        print(f"FABRIK: Thorax glide surface fit with {len(glide)} points.")

    def get_surface_info(self, pt):
        """Return (surface_point, outward_normal) at the Y,Z of pt.
        Normal points posteriorly (negative X in this JCS)."""
        kn = self.spline.get_knots()
        y_c = np.clip(pt[1], kn[0][0], kn[0][-1])
        z_c = np.clip(pt[2], kn[1][0], kn[1][-1])
        x   = float(self.spline.ev(y_c, z_c))
        dfdy = float(self.spline.ev(y_c, z_c, dx=1, dy=0))
        dfdz = float(self.spline.ev(y_c, z_c, dx=0, dy=1))
        normal = np.array([1.0, -dfdy, -dfdz])
        nlen   = np.linalg.norm(normal)
        if nlen > 1e-6:
            normal /= nlen
        if normal[0] > 0:
            normal = -normal
        return np.array([x, y_c, z_c]), normal

    # ── FABRIK core ──────────────────────────────────────────────────────────

    def fabrik_solve(self, p_start, lengths, target, iterations=50, tolerance=0.01):
        """2-segment FABRIK: p[0]=SC(fixed), p[1]=AC, p[2]=Centroid."""
        p = [np.array(x, dtype=float) for x in p_start]
        t = np.array(target, dtype=float)
        for _ in range(iterations):
            p[2] = t.copy()
            for i in range(1, -1, -1):
                r = np.linalg.norm(p[i+1] - p[i])
                if r < 1e-9: continue
                p[i] = (1 - lengths[i]/r)*p[i+1] + (lengths[i]/r)*p[i]
            p[0] = self.sc_joint.copy()
            for i in range(2):
                r = np.linalg.norm(p[i+1] - p[i])
                if r < 1e-9: continue
                p[i+1] = (1 - lengths[i]/r)*p[i] + (lengths[i]/r)*p[i+1]
            if np.linalg.norm(p[2] - t) < tolerance:
                break
        return p

    # ── Main solver ──────────────────────────────────────────────────────────

    def solve_alignment(self, initial_ac, initial_centroid, lms_local,
                        scap_mesh_local, p_proj, subscap_seed=None,
                        initial_rot=None, max_step=4):
        """
        FABRIK alignment pipeline (Steps 0-4).

        Parameters
        ----------
        initial_ac, initial_centroid : seed positions from JCS assembly.
        lms_local   : dict {'aa','ts','ai'} – vectors relative to initial_ac.
        p_proj      : anchor point on the posterior thorax surface.
        subscap_seed: (N,3) subscapularis point cloud in world frame (seed pose).
        initial_rot : scipy Rotation from JCS assembly (prevents flipping).
        max_step    : 0=seed only, 1-4=run up to that step.
        """
        # ── Segment lengths (preserved throughout) ────────────────────────────
        l1 = float(np.linalg.norm(np.asarray(initial_ac) - self.sc_joint))
        l2 = float(np.linalg.norm(np.asarray(initial_centroid) - np.asarray(initial_ac)))
        lengths = [l1, l2]

        # ── Local landmark vectors (relative to AC) ───────────────────────────
        aa_loc  = np.array(lms_local['aa'], dtype=float)
        ts_loc  = np.array(lms_local['ts'], dtype=float)
        ai_loc  = np.array(lms_local['ai'], dtype=float)
        centroid_loc = (aa_loc + ts_loc + ai_loc) / 3.0

        # ── Subscap local frame ───────────────────────────────────────────────
        has_subscap = subscap_seed is not None and len(subscap_seed) > 0
        if has_subscap:
            subscap_local_pts = np.array(subscap_seed, dtype=float) - np.array(initial_ac, dtype=float)
        else:
            subscap_local_pts = None
            print("  FABRIK WARNING: No subscap cloud – using landmark fallback.")

        # ── Scapular normal (dorsal, away from ribs) ─────────────────────────
        n_scap_loc = np.cross(ai_loc - ts_loc, aa_loc - ts_loc)
        n_scap_loc /= np.linalg.norm(n_scap_loc)
        seed_rot = initial_rot if initial_rot is not None else R.identity()
        _, n_ref = self.get_surface_info(initial_centroid)
        if np.dot(seed_rot.apply(n_scap_loc), n_ref) < 0:
            n_scap_loc = -n_scap_loc
            print("  FABRIK: Flipped n_scap to point dorsally.")

        # ── Mutable state ─────────────────────────────────────────────────────
        ac    = np.array(initial_ac,       dtype=float)
        cen   = np.array(initial_centroid, dtype=float)
        rot   = seed_rot
        chain = [self.sc_joint.copy(), ac.copy(), cen.copy()]

        # ── Inner helpers ─────────────────────────────────────────────────────
        def n_scap_world():
            return rot.apply(n_scap_loc)

        def world_lm(loc):
            return rot.apply(loc) + ac

        def rotate_ac(delta):
            nonlocal rot, cen
            rot = delta * rot
            cen = rot.apply(centroid_loc) + ac

        def sync_rot_after_fabrik():
            nonlocal rot
            v_new = cen - ac
            v_old = rot.apply(centroid_loc)
            if np.linalg.norm(v_new) > 1e-6 and np.linalg.norm(v_old) > 1e-6:
                d, _ = R.align_vectors([v_new], [v_old])
                rot = d * rot

        def subscap_world():
            if subscap_local_pts is None: return None
            return rot.apply(subscap_local_pts) + ac

        def subscap_clearances(sw=None):
            """Signed distances from every subscap point to thorax surface."""
            if sw is None: sw = subscap_world()
            if sw is None: return None
            return np.array([np.dot(pt - self.get_surface_info(pt)[0],
                                    self.get_surface_info(pt)[1]) for pt in sw])

        def apply_bubble_translation():
            """Push/pull scapula so min subscap clearance = SUBSCAP_BUBBLE_MM.
            Translates ac+cen rigidly along the scapular dorsal normal — no
            FABRIK solve, no sync_rot.  Rigid translation preserves the
            centroid-AC direction, so rot stays fully consistent throughout."""
            nonlocal ac, cen, chain
            if has_subscap:
                d_min_first = None
                for _ in range(15):
                    dists = subscap_clearances()
                    d_min = float(np.min(dists))
                    if d_min_first is None:
                        d_min_first = d_min
                    delta = self.SUBSCAP_BUBBLE_MM - d_min
                    if abs(delta) < 0.1:
                        break
                    push = n_scap_world() * delta  # dorsal = away from thorax
                    ac  = ac  + push
                    cen = cen + push
                chain = [self.sc_joint.copy(), ac.copy(), cen.copy()]
                return d_min_first
            else:
                lm_pts = [world_lm(aa_loc), world_lm(ts_loc),
                          world_lm(ai_loc), 0.5*(world_lm(ts_loc)+world_lm(ai_loc))]
                dists_lm = [np.dot(pt - self.get_surface_info(pt)[0],
                                   self.get_surface_info(pt)[1]) for pt in lm_pts]
                d_min = min(dists_lm)
                if d_min < 0.0:
                    _, n_cen = self.get_surface_info(cen)
                    push = abs(d_min) + 2.0
                    chain = [self.sc_joint.copy(), ac.copy(), cen.copy()]
                    chain = self.fabrik_solve(chain, lengths, cen + n_cen * push)
                    ac, cen = chain[1].copy(), chain[2].copy()
                    sync_rot_after_fabrik()
                return d_min

        # ── Step 0: Seed ──────────────────────────────────────────────────────
        if max_step <= 0:
            return ac, cen, rot

        # ══════════════════════════════════════════════════════════════════════
        # STEP 1 – Initial placement
        # standoff = SUBSCAP_BUBBLE_MM + d_close_local
        # d_close_local = how far the deepest subscap point protrudes toward
        #                 the ribs relative to the centroid (uses min, not mean).
        # ══════════════════════════════════════════════════════════════════════
        p_surf, n_surf = self.get_surface_info(p_proj)

        if has_subscap:
            protrusions  = np.dot(subscap_local_pts - centroid_loc, -n_scap_loc)
            d_close_local = float(max(np.max(protrusions), 0.0))
            standoff = self.SUBSCAP_BUBBLE_MM + d_close_local
        else:
            d_close_local = 0.0
            standoff = 10.0   # legacy

        chain = self.fabrik_solve(chain, lengths, p_surf + n_surf * standoff)
        ac, cen = chain[1].copy(), chain[2].copy()
        sync_rot_after_fabrik()
        print(f"  FABRIK Step 1: Centroid [{cen[0]:.1f},{cen[1]:.1f},{cen[2]:.1f}] "
              f"standoff={standoff:.1f}mm (d_close={d_close_local:.1f}mm)")

        if max_step <= 1:
            return ac, cen, rot

        # ══════════════════════════════════════════════════════════════════════
        # STEP 2 – Initial orientation: AA-TS-AI tangency (anatomical plane guess)
        # ══════════════════════════════════════════════════════════════════════
        _, n_thor_2 = self.get_surface_info(cen)
        
        # FIX A: Verticalize the target normal (zero out Y) to prevent "shelf" pose
        n_thor_2_flat = n_thor_2.copy()
        n_thor_2_flat[1] = 0.0
        nlen = np.linalg.norm(n_thor_2_flat)
        if nlen > 1e-6:
            n_thor_2_flat /= nlen
        else:
            n_thor_2_flat = n_thor_2 # fallback
            
        n_scap = n_scap_world()
        print(f"  DIAG Step 2: n_scap={[f'{x:.3f}' for x in n_scap]}, "
              f"n_thor_flat={[f'{x:.3f}' for x in n_thor_2_flat]}")
              
        delta_2, _ = R.align_vectors([n_thor_2_flat], [n_scap])
        rotate_ac(delta_2)

        # Re-apply standoff after rotation
        # FIX B: Anchor back to the original projection point (p_proj) 
        # to prevent the centroid from sliding off the posterior wall.
        p_s2, n_s2 = self.get_surface_info(p_proj)
        chain = [self.sc_joint.copy(), ac.copy(), cen.copy()]
        chain = self.fabrik_solve(chain, lengths, p_s2 + n_s2 * standoff)
        ac, cen = chain[1].copy(), chain[2].copy()
        sync_rot_after_fabrik()
        print(f"  FABRIK Step 2: AA-TS-AI tangency. Centroid [{cen[0]:.1f},{cen[1]:.1f},{cen[2]:.1f}]")

        if max_step <= 2:
            return ac, cen, rot

        # ══════════════════════════════════════════════════════════════════════
        # STEP 3 – Bubble translation
        # Bidirectional push/pull: lock min(subscap → thorax) = SUBSCAP_BUBBLE_MM
        # ══════════════════════════════════════════════════════════════════════
        d_before = apply_bubble_translation()
        d_after  = float(np.min(subscap_clearances())) if has_subscap else None
        if d_after is not None:
            print(f"  FABRIK Step 3: d_min {d_before:.1f}->{d_after:.1f}mm "
                  f"(target={self.SUBSCAP_BUBBLE_MM:.0f}mm)")
        else:
            print(f"  FABRIK Step 3: landmark clearance={d_before:.1f}mm")

        if max_step <= 3:
            return ac, cen, rot

        # ══════════════════════════════════════════════════════════════════════
        # STEP 4 – Bubble fine-tuning
        # Nelder-Mead over (tilt_deg, roll_deg) to minimise variance of
        # subscap→thorax clearances (flush fit). Then re-lock Step 3.
        # ══════════════════════════════════════════════════════════════════════
        if has_subscap:
            roll_axis = cen - ac
            rlen = np.linalg.norm(roll_axis)
            if rlen < 1e-6:
                return ac, cen, rot
            roll_axis /= rlen

            tilt_axis = np.cross(roll_axis, n_scap_world())
            tlen = np.linalg.norm(tilt_axis)
            if tlen < 1e-6:
                tilt_axis = np.cross(roll_axis, np.array([0., 1., 0.]))
                tlen = np.linalg.norm(tilt_axis)
            tilt_axis /= tlen

            # Snapshot for cost function (avoid mutating mutable state)
            rot_snap = rot
            ac_snap  = ac.copy()

            def bubble_cost(params):
                td, rd = params
                
                # 1. HARD ANGULAR LIMITS: Force search to stay within ±40 degrees
                if abs(td) > 40.0 or abs(rd) > 40.0:
                    return 1e9 

                trial = (R.from_rotvec(np.radians(td)*tilt_axis) *
                         R.from_rotvec(np.radians(rd)*roll_axis) * rot_snap)
                
                # Calculate transformed Subscapularis points and clearances
                sw = trial.apply(subscap_local_pts) + ac_snap
                dists = np.array([np.dot(pt - self.get_surface_info(pt)[0],
                                         self.get_surface_info(pt)[1]) for pt in sw])
                
                # 2. CONTACT COST: Target exactly 10mm.
                # Use Weighted Absolute Error for better physical intuition.
                errors = dists - self.SUBSCAP_BUBBLE_MM
                
                # Base cost is the absolute error from 10mm
                cost = float(np.mean(np.abs(errors)))
                
                # 3. PENETRATION PENALTY (CRITICAL): 
                # Massive penalty for any point actually inside the ribs (dist < 0)
                penetration_mask = dists < 0.0
                if np.any(penetration_mask):
                    cost += float(np.sum(np.abs(dists[penetration_mask])) * 500.0)
                
                # 4. AI TETHER: Prevent Inferior Angle lift-off (>15mm)
                # Use absolute value to prevent penetration here as well
                ai_w = trial.apply(ai_loc) + ac_snap
                p_ai_surf, n_ai_surf = self.get_surface_info(ai_w)
                ai_dist = np.dot(ai_w - p_ai_surf, n_ai_surf)
                
                if ai_dist < 0: # Inside ribs
                    cost += abs(ai_dist) * 10000.0
                elif ai_dist > 15.0: # Too far away
                    cost += (ai_dist - 15.0) * 1000.0

                # 5. CLAVICLE COLLISION GUARD: Check shaft midpoint
                # The clavicle rod goes from SC to AC.
                c_mid = 0.5 * (self.sc_joint + ac_snap)
                p_c_mid, n_c_mid = self.get_surface_info(c_mid)
                c_dist = np.dot(c_mid - p_c_mid, n_c_mid)
                if c_dist < 5.0: # Must stay at least 5mm clear of the ribs
                    cost += abs(c_dist - 5.0) * 50000.0
                
                return cost

            res = minimize(bubble_cost, x0=[0., 0.], method='Nelder-Mead',
                           options={'xatol': 0.05, 'fatol': 0.01, 'maxiter': 800})
            bt, br = res.x
            rotate_ac(R.from_rotvec(np.radians(bt)*tilt_axis) *
                      R.from_rotvec(np.radians(br)*roll_axis))

            # Final re-lock of the bubble gap
            apply_bubble_translation()
            df = subscap_clearances()
            
            # Diagnostic for AI distance
            ai_fin = world_lm(ai_loc)
            p_ai_fin, n_ai_fin = self.get_surface_info(ai_fin)
            ai_dist_fin = np.dot(ai_fin - p_ai_fin, n_ai_fin)
            
            print(f"  FABRIK Step 4: Optimized tilt={bt:.1f}°, roll={br:.1f}°, Cost={res.fun:.2f}, d_min={np.min(df):.1f}mm, AI_dist={ai_dist_fin:.1f}mm")
        else:
            # Legacy: single-axis roll sweep for medial clearance
            p_c4, n_t4 = self.get_surface_info(cen)
            roll_axis = cen - ac
            rlen = np.linalg.norm(roll_axis)
            if rlen < 1e-6: return ac, cen, rot
            roll_axis /= rlen
            best_deg, best_err = 0, float('inf')
            for deg in range(-45, 46, 1):
                tr = R.from_rotvec(np.radians(deg)*roll_axis) * rot
                ai_d = np.dot(tr.apply(ai_loc) + ac - p_c4, n_t4)
                ts_d = np.dot(tr.apply(ts_loc) + ac - p_c4, n_t4)
                err = (max(0., self.MEDIAL_CLEARANCE - ai_d)**2 +
                       max(0., self.MEDIAL_CLEARANCE - ts_d)**2)
                if err < best_err:
                    best_err, best_deg = err, deg
            if best_deg != 0:
                rotate_ac(R.from_rotvec(np.radians(best_deg)*roll_axis))
            ai_w = world_lm(ai_loc); ts_w = world_lm(ts_loc)
            p_c4, n_t4 = self.get_surface_info(cen)
            print(f"  FABRIK Step 4 (legacy): AI={np.dot(ai_w-p_c4,n_t4):.1f}mm, "
                  f"TS={np.dot(ts_w-p_c4,n_t4):.1f}mm (roll={best_deg}°)")

        return ac, cen, rot


# ── Public API ────────────────────────────────────────────────────────────────

def apply_fabrik_alignment(side, thorax_mesh, sc_joint, ac_joint,
                           aa_pt, ts_pt, ai_pt, scap_mesh,
                           p_proj=None, subscap_seed=None,
                           initial_rot=None, max_step=4):
    """Run solver and transform the scapula mesh."""
    solver   = FabrikScapulaSolver(thorax_mesh, sc_joint)
    centroid = (np.array(aa_pt) + np.array(ts_pt) + np.array(ai_pt)) / 3.0
    lms_local = {
        'aa': np.array(aa_pt) - np.array(ac_joint),
        'ts': np.array(ts_pt) - np.array(ac_joint),
        'ai': np.array(ai_pt) - np.array(ac_joint),
    }
    if p_proj is None:
        p_proj, _ = solver.get_surface_info(centroid)

    ac_sol, cen_sol, rot_sol = solver.solve_alignment(
        ac_joint, centroid, lms_local, None, p_proj,
        subscap_seed=subscap_seed,
        initial_rot=initial_rot,
        max_step=max_step,
    )

    mesh_centered = np.array(scap_mesh) - np.array(ac_joint)
    final_mesh = rot_sol.apply(mesh_centered) + ac_sol
    return final_mesh, ac_sol, rot_sol
