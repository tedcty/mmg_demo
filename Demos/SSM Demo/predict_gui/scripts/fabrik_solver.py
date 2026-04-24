import numpy as np
from scipy.interpolate import LSQBivariateSpline
from scipy.spatial.transform import Rotation as R


class FabrikScapulaSolver:
    """
    Implements the 4-step FABRIK Scapulothoracic Alignment Algorithm.
    Chain: SC (Fixed) -> AC -> Scapula Centroid

    Gap is measured from the bone surface (subscapularis fossa).
    State (ac_pos, centroid_pos, rot) threads incrementally through each step.
    """

    # Gap parameters (mm)
    BONE_HALF_THICKNESS = 5.0   # Estimated half-thickness of scapular body
    SUBSCAP_GAP = 5.0           # Subscapularis fossa to rib surface
    MEDIAL_CLEARANCE = 5.0      # TS / AI clearance above sliding surface

    def __init__(self, thorax_mesh, sc_joint):
        self.sc_joint = np.array(sc_joint, dtype=float)
        self.thorax_mesh = np.array(thorax_mesh, dtype=float)
        self.spline = None
        self._fit_thorax_surface()

    # ── Surface fitting ──────────────────────────────────────────────────────

    def _fit_thorax_surface(self):
        """Fits a B-Spline X=f(Y,Z) to the posterior thorax glide area."""
        pts = self.thorax_mesh
        x_mid = (pts[:, 0].min() + pts[:, 0].max()) / 2.0
        y_min = pts[:, 1].min() + (pts[:, 1].max() - pts[:, 1].min()) * 0.2

        # X negative = Posterior (T8_X ≈ -119, PX_X ≈ +56)
        mask = (pts[:, 0] < x_mid + 20) & (pts[:, 1] > y_min)
        glide_pts = pts[mask]

        if len(glide_pts) < 100:
            print("FABRIK WARNING: Glide area filter too strict. Using full mesh.")
            glide_pts = pts

        y_pts, z_pts, x_pts = glide_pts[:, 1], glide_pts[:, 2], glide_pts[:, 0]
        ky = np.linspace(y_pts.min(), y_pts.max(), 10)[1:-1]
        kz = np.linspace(z_pts.min(), z_pts.max(), 10)[1:-1]
        self.spline = LSQBivariateSpline(y_pts, z_pts, x_pts, ky, kz, kx=3, ky=3)
        print(f"FABRIK: Thorax glide surface fit with {len(glide_pts)} points.")

    def get_surface_info(self, pt):
        """Returns (surface_point, outward_normal) at the Y,Z of pt.

        The normal points posteriorly (away from ribcage interior, toward the
        scapula). In this coordinate system that is negative X.
        """
        y_c = np.clip(pt[1], self.spline.get_knots()[0][0], self.spline.get_knots()[0][-1])
        z_c = np.clip(pt[2], self.spline.get_knots()[1][0], self.spline.get_knots()[1][-1])
        x = float(self.spline.ev(y_c, z_c))
        dfdy = float(self.spline.ev(y_c, z_c, dx=1, dy=0))
        dfdz = float(self.spline.ev(y_c, z_c, dx=0, dy=1))

        normal = np.array([1.0, -dfdy, -dfdz])
        norm_val = np.linalg.norm(normal)
        if norm_val > 1e-6:
            normal /= norm_val

        # Ensure normal points posteriorly (negative X)
        if normal[0] > 0:
            normal = -normal

        return np.array([x, y_c, z_c]), normal

    # ── FABRIK core ──────────────────────────────────────────────────────────

    def fabrik_solve(self, p_start, lengths, target, iterations=50, tolerance=0.01):
        """Standard 2-segment FABRIK: p[0]=SC (fixed), p[1]=AC, p[2]=Centroid."""
        p = [np.array(x, dtype=float) for x in p_start]
        target = np.array(target, dtype=float)

        for _ in range(iterations):
            # Backward pass
            p[2] = target.copy()
            for i in range(1, -1, -1):
                r = np.linalg.norm(p[i + 1] - p[i])
                if r < 1e-9:
                    continue
                lam = lengths[i] / r
                p[i] = (1 - lam) * p[i + 1] + lam * p[i]

            # Forward pass
            p[0] = self.sc_joint.copy()
            for i in range(2):
                r = np.linalg.norm(p[i + 1] - p[i])
                if r < 1e-9:
                    continue
                lam = lengths[i] / r
                p[i + 1] = (1 - lam) * p[i] + lam * p[i + 1]

            if np.linalg.norm(p[2] - target) < tolerance:
                break

        return p

    # ── Main solver ──────────────────────────────────────────────────────────

    def solve_alignment(self, initial_ac, initial_centroid, lms_local,
                        scap_mesh_local, p_proj, initial_rot=None, max_step=4):
        """
        Executes the FABRIK alignment pipeline up to max_step (0-4).

        Parameters
        ----------
        initial_ac, initial_centroid : positions from the JCS assembly seed.
        lms_local : dict  {'aa', 'ts', 'ai'} relative to initial_ac.
        p_proj : projected anchor point on the posterior thorax surface.
        initial_rot : scipy Rotation from the JCS assembly (prevents flipping).
        max_step : 0=seed, 1-4=FABRIK steps.
        """
        # ── Constants ─────────────────────────────────────────────────────────
        l1 = float(np.linalg.norm(np.asarray(initial_ac) - self.sc_joint))
        l2 = float(np.linalg.norm(np.asarray(initial_centroid) - np.asarray(initial_ac)))
        lengths = [l1, l2]

        centroid_gap = self.SUBSCAP_GAP + self.BONE_HALF_THICKNESS  # mm from surface to centroid

        # ── Local landmark vectors (relative to AC) ───────────────────────────
        aa_loc = np.array(lms_local['aa'], dtype=float)
        ts_loc = np.array(lms_local['ts'], dtype=float)
        ai_loc = np.array(lms_local['ai'], dtype=float)
        centroid_loc = (aa_loc + ts_loc + ai_loc) / 3.0

        # ── Local scapular normal (sign-locked AWAY from ribcage = dorsal side) ──
        # Convention: n_scap points outward from the dorsal surface, same
        # direction as n_thor (which points away from the rib interior).
        # This way R.align_vectors([n_thor], [n_scap]) keeps subscapularis
        # facing the ribs.
        n_scap_loc = np.cross(ai_loc - ts_loc, aa_loc - ts_loc)
        n_scap_loc /= np.linalg.norm(n_scap_loc)

        seed_rot = initial_rot if initial_rot is not None else R.identity()
        n_scap_seed = seed_rot.apply(n_scap_loc)
        _, n_ref = self.get_surface_info(initial_centroid)
        # n_ref points away from ribcage; n_scap should too
        if np.dot(n_scap_seed, n_ref) < 0:
            n_scap_loc = -n_scap_loc
            print("  FABRIK: Flipped local scapular normal to point dorsally (away from ribs).")

        # ── Mutable state ─────────────────────────────────────────────────────
        ac = np.array(initial_ac, dtype=float)
        cen = np.array(initial_centroid, dtype=float)
        rot = seed_rot
        chain = [self.sc_joint.copy(), ac.copy(), cen.copy()]

        # ── Helpers ───────────────────────────────────────────────────────────
        def n_scap_world():
            return rot.apply(n_scap_loc)

        def world_lm(loc):
            return rot.apply(loc) + ac

        def rotate_ac(delta):
            nonlocal rot, cen
            rot = delta * rot
            cen = rot.apply(centroid_loc) + ac

        def sync_rot_after_fabrik():
            """Incrementally update rot after FABRIK moves ac/cen."""
            nonlocal rot
            v_new = cen - ac
            v_old = rot.apply(centroid_loc)
            if np.linalg.norm(v_new) > 1e-6 and np.linalg.norm(v_old) > 1e-6:
                d, _ = R.align_vectors([v_new], [v_old])
                rot = d * rot

        # ── Step 0: Seed ──────────────────────────────────────────────────────
        if max_step <= 0:
            return ac, cen, rot

        # ══════════════════════════════════════════════════════════════════════
        # STEP 1  –  Position with built-in gap
        # Target = P_proj + centroid_gap * n_thor(P_proj)
        # ══════════════════════════════════════════════════════════════════════
        p_surf, n_surf = self.get_surface_info(p_proj)
        target_1 = p_surf + n_surf * centroid_gap

        chain = self.fabrik_solve(chain, lengths, target_1)
        ac, cen = chain[1].copy(), chain[2].copy()
        sync_rot_after_fabrik()

        print(f"  FABRIK Step 1: Centroid at [{cen[0]:.1f}, {cen[1]:.1f}, {cen[2]:.1f}] "
              f"(gap={centroid_gap:.0f}mm from surface)")

        if max_step <= 1:
            return ac, cen, rot

        # ══════════════════════════════════════════════════════════════════════
        # STEP 2  –  Orient scapula (tangency alignment)
        # Rotate around J_AC so n_scap ∥ n_thor(P_close)
        # ══════════════════════════════════════════════════════════════════════
        _, n_thor_2 = self.get_surface_info(cen)

        n_scap = n_scap_world()
        print(f"  DIAG Step 2: n_scap={[f'{x:.3f}' for x in n_scap]}, "
              f"n_thor={[f'{x:.3f}' for x in n_thor_2]}")

        delta_2, _ = R.align_vectors([n_thor_2], [n_scap])
        rotate_ac(delta_2)

        # Re-apply gap: orientation moved centroid, so FABRIK it back
        p_surf_2, n_surf_2 = self.get_surface_info(cen)
        target_2 = p_surf_2 + n_surf_2 * centroid_gap
        chain = [self.sc_joint.copy(), ac.copy(), cen.copy()]
        chain = self.fabrik_solve(chain, lengths, target_2)
        ac, cen = chain[1].copy(), chain[2].copy()
        sync_rot_after_fabrik()

        print(f"  FABRIK Step 2: Oriented + gap restored. Centroid at [{cen[0]:.1f}, {cen[1]:.1f}, {cen[2]:.1f}]")

        if max_step <= 2:
            return ac, cen, rot

        # ══════════════════════════════════════════════════════════════════════
        # STEP 3  –  Penetration check & correction
        # Push outward if any landmark is inside the thorax volume
        # ══════════════════════════════════════════════════════════════════════
        p_close_3, n_push_3 = self.get_surface_info(cen)

        aa_w = world_lm(aa_loc)
        ts_w = world_lm(ts_loc)
        ai_w = world_lm(ai_loc)
        mb_w = 0.5 * (ts_w + ai_w)

        lm_names = ['AA', 'TS', 'AI', 'MB']
        lm_pts = [aa_w, ts_w, ai_w, mb_w]
        dists = []
        for name, pt in zip(lm_names, lm_pts):
            # Distance from the surface point closest to EACH landmark
            p_lm_surf, n_lm_surf = self.get_surface_info(pt)
            d = np.dot(pt - p_lm_surf, n_lm_surf)
            dists.append(d)
            print(f"    {name}: {d:.1f}mm from surface")

        min_d = min(dists)
        print(f"  FABRIK Step 3: Min landmark clearance = {min_d:.2f}mm")

        if min_d < 0.0:
            # Iterative push: only push enough to clear the worst penetration
            for iteration in range(5):
                # Recompute world landmark positions with current state
                cur_lm_pts = [world_lm(aa_loc), world_lm(ts_loc),
                              world_lm(ai_loc), 0.5 * (world_lm(ts_loc) + world_lm(ai_loc))]
                
                # Find the worst-penetrating landmark
                worst_d = 0
                for pt in cur_lm_pts:
                    p_lm_surf, n_lm_surf = self.get_surface_info(pt)
                    d = np.dot(pt - p_lm_surf, n_lm_surf)
                    if d < worst_d:
                        worst_d = d

                if worst_d >= 0.0:
                    print(f"  FABRIK Step 3: All clear after {iteration} iterations")
                    break

                # Push centroid along its local surface normal by just enough
                p_cen_surf, n_cen = self.get_surface_info(cen)
                push = abs(worst_d) + 2.0
                push_target = cen + n_cen * push
                chain = [self.sc_joint.copy(), ac.copy(), cen.copy()]
                chain = self.fabrik_solve(chain, lengths, push_target)
                ac, cen = chain[1].copy(), chain[2].copy()
                sync_rot_after_fabrik()

                print(f"  FABRIK Step 3: Iter {iteration+1} pushed {push:.1f}mm, Centroid X={cen[0]:.1f}")

        if max_step <= 3:
            return ac, cen, rot

        # ══════════════════════════════════════════════════════════════════════
        # STEP 4  –  Medial border clearance (roll fine-tune)
        # Roll around AC→Centroid axis to achieve ~5mm TS/AI clearance
        # ══════════════════════════════════════════════════════════════════════
        p_close_4, n_thor_4 = self.get_surface_info(cen)

        roll_axis = cen - ac
        roll_len = np.linalg.norm(roll_axis)
        if roll_len < 1e-6:
            return ac, cen, rot
        roll_axis /= roll_len

        best_deg = 0
        best_err = float('inf')

        for deg in range(-45, 46, 1):
            test_rot = R.from_rotvec(np.radians(deg) * roll_axis) * rot

            ai_d = np.dot(test_rot.apply(ai_loc) + ac - p_close_4, n_thor_4)
            ts_d = np.dot(test_rot.apply(ts_loc) + ac - p_close_4, n_thor_4)

            err = (max(0.0, self.MEDIAL_CLEARANCE - ai_d) ** 2 +
                   max(0.0, self.MEDIAL_CLEARANCE - ts_d) ** 2)
            if err < best_err:
                best_err = err
                best_deg = deg

        if best_deg != 0:
            rotate_ac(R.from_rotvec(np.radians(best_deg) * roll_axis))

        ai_w = world_lm(ai_loc)
        ts_w = world_lm(ts_loc)
        ai_d = np.dot(ai_w - p_close_4, n_thor_4)
        ts_d = np.dot(ts_w - p_close_4, n_thor_4)
        print(f"  FABRIK Step 4: AI={ai_d:.1f}mm, TS={ts_d:.1f}mm (roll={best_deg}°)")

        return ac, cen, rot


# ── Public API ────────────────────────────────────────────────────────────────

def apply_fabrik_alignment(side, thorax_mesh, sc_joint, ac_joint,
                           aa_pt, ts_pt, ai_pt, scap_mesh,
                           p_proj=None, initial_rot=None, max_step=4):
    """Run the solver and transform the scapula mesh."""
    solver = FabrikScapulaSolver(thorax_mesh, sc_joint)

    centroid = (aa_pt + ts_pt + ai_pt) / 3.0
    lms_local = {
        'aa': aa_pt - ac_joint,
        'ts': ts_pt - ac_joint,
        'ai': ai_pt - ac_joint,
    }

    if p_proj is None:
        p_proj, _ = solver.get_surface_info(centroid)

    ac_sol, cen_sol, rot_sol = solver.solve_alignment(
        ac_joint, centroid, lms_local, None, p_proj,
        initial_rot=initial_rot, max_step=max_step,
    )

    # Transform mesh: zero at original AC → rotate → translate to solved AC
    mesh_centered = np.array(scap_mesh) - np.array(ac_joint)
    final_mesh = rot_sol.apply(mesh_centered) + ac_sol

    return final_mesh, ac_sol, rot_sol
