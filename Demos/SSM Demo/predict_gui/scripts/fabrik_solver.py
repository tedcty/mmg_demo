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

    SUBSCAP_BUBBLE_MM = 3.0    # target min clearance: subscap cloud → thorax
                               # (in-vivo bursa is 2-5mm; subscap muscle compresses)
    MEDIAL_CLEARANCE  = 5.0    # legacy fallback (no subscap cloud)

    def __init__(self, thorax_mesh, sc_joint):
        self.sc_joint    = np.array(sc_joint,    dtype=float)
        self.thorax_mesh = np.array(thorax_mesh, dtype=float)
        self.spline      = None
        # Right SC is at positive Z in ISB frame; left at negative Z.
        self.side_sign   = 1.0 if self.sc_joint[2] >= 0 else -1.0
        self._fit_thorax_surface()

    # ── Surface fitting ──────────────────────────────────────────────────────

    def _fit_thorax_surface(self):
        """Fits B-Spline X=f(Y,Z) to the posterior thorax glide area."""
        pts   = self.thorax_mesh
        x_mid = (pts[:, 0].min() + pts[:, 0].max()) / 2.0
        # Use 5% cutoff (was 20%) so the B-spline surface extends down to T8
        # level and can correctly anchor the scapula inferior angle.
        y_min = pts[:, 1].min() + (pts[:, 1].max() - pts[:, 1].min()) * 0.05
        mask  = (pts[:, 0] < x_mid + 20) & (pts[:, 1] > y_min)
        glide = pts[mask]
        if len(glide) < 100:
            print("FABRIK WARNING: glide filter too strict – using full mesh.")
            glide = pts
        # LSQBivariateSpline needs C-contiguous float64 arrays -- glide[:, i]
        # column slices are strided views, which FITPACK's Fortran layer
        # silently mishandles (returns a "no approximation"/rank-deficient
        # warning and a garbage spline, not an exception -- this was failing
        # on every call, including FABRIK's own real usage, not just here).
        y_pts = np.ascontiguousarray(glide[:, 1], dtype=np.float64)
        z_pts = np.ascontiguousarray(glide[:, 2], dtype=np.float64)
        x_pts = np.ascontiguousarray(glide[:, 0], dtype=np.float64)
        # The glide region is a curved, non-rectangular patch of the ribcage,
        # so its bounding box has structurally empty corners (e.g. extreme
        # height + extreme lateral offset together don't occur on a tapering
        # thorax) -- a fixed evenly-spaced knot grid reaching the exact
        # min/max always includes some empty tensor-product cell, which
        # FITPACK rejects outright (fits fine in isolation, not as a grid).
        # A few percent inset + fewer knots keeps every cell populated.
        ky = np.ascontiguousarray(
            np.linspace(np.percentile(y_pts, 3), np.percentile(y_pts, 97), 7)[1:-1]
        )
        kz = np.ascontiguousarray(
            np.linspace(np.percentile(z_pts, 3), np.percentile(z_pts, 97), 7)[1:-1]
        )
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
                        initial_rot=None, c7=None, t8=None, max_step=4,
                        correction=None):
        """
        FABRIK alignment pipeline (Steps 0-4).

        Parameters
        ----------
        initial_ac, initial_centroid : seed positions from JCS assembly.
        lms_local   : dict {'aa','ts','ai'[,'cp']} – vectors relative to initial_ac.
                      'cp' (coracoid process) is optional; when present, Step 4
                      penalizes it penetrating the thorax the same way AI does.
        p_proj      : anchor point on the posterior thorax surface.
        subscap_seed: (N,3) subscapularis point cloud in world frame (seed pose).
        initial_rot : scipy Rotation from JCS assembly (prevents flipping).
        c7, t8      : world-space thorax spine landmarks (midline). Optional;
                      when present, Step 4/4b add a trapezius-informed spring
                      toward the C7-AC / T8-TS lengths measured at the seed pose
                      (the SSM's uncorrected prediction) — an approximation of
                      the muscle resisting the scapula drifting from its
                      anatomical position while the geometric ST-contact terms
                      are free to nudge it for thorax-collision avoidance.
        max_step    : 0=seed only, 1-4=run up to that step.
        correction  : optional (tilt_deg, roll_deg, t_push, t_slide) from a
                      previous full Step-4 solve (e.g. on the mean shape).
                      When given, Step 4 skips the Nelder-Mead search entirely
                      and just reapplies these numbers against THIS call's own
                      freshly-computed push_dir/slide_dir/tilt_axis/roll_axis —
                      i.e. "same correction, new geometry" rather than a stale
                      cached pose. Steps 1-3 (seed, tangency, bubble
                      translation) always run fresh regardless, since they're
                      cheap and shape-dependent. This trades Step 4's ~90s
                      multi-start search for a single direct evaluation
                      (<10ms), at the cost of not re-verifying that the
                      correction is still collision-free on this specific
                      geometry — fine for previewing nearby shapes, not a
                      substitute for a real solve on far-off geometry.

        Returns
        -------
        (ac, cen, rot, correction_used) — correction_used is the
        (tilt_deg, roll_deg, t_push, t_slide) actually applied at Step 4
        (freshly solved or reused), or None if Step 4 didn't run at all
        (max_step < 4, or no subscap cloud).
        """
        # ── Segment lengths (preserved throughout) ────────────────────────────
        l1 = float(np.linalg.norm(np.asarray(initial_ac) - self.sc_joint))
        l2 = float(np.linalg.norm(np.asarray(initial_centroid) - np.asarray(initial_ac)))
        lengths = [l1, l2]

        # ── Local landmark vectors (relative to AC) ───────────────────────────
        aa_loc  = np.array(lms_local['aa'], dtype=float)
        ts_loc  = np.array(lms_local['ts'], dtype=float)
        ai_loc  = np.array(lms_local['ai'], dtype=float)
        cp_loc  = np.array(lms_local['cp'], dtype=float) if 'cp' in lms_local else None
        centroid_loc = (aa_loc + ts_loc + ai_loc) / 3.0

        # ── Trapezius seed (rest) lengths — measured now, before any FABRIK
        # step moves anything, so they reflect the SSM's own predicted anatomy.
        c7 = np.array(c7, dtype=float) if c7 is not None else None
        t8 = np.array(t8, dtype=float) if t8 is not None else None
        seed_rot_for_trap = initial_rot if initial_rot is not None else R.identity()
        trap_upper_L0 = float(np.linalg.norm(np.asarray(initial_ac) - c7)) if c7 is not None else None
        trap_lower_L0 = (float(np.linalg.norm(seed_rot_for_trap.apply(ts_loc) + np.asarray(initial_ac) - t8))
                          if t8 is not None else None)

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
                best_err = None
                best_ac, best_cen = ac.copy(), cen.copy()
                for _ in range(30):
                    dists = subscap_clearances()
                    d_min = float(np.min(dists))
                    if d_min_first is None:
                        d_min_first = d_min
                    err = abs(d_min - self.SUBSCAP_BUBBLE_MM)
                    if best_err is None or err < best_err:
                        best_err = err
                        best_ac, best_cen = ac.copy(), cen.copy()
                    delta = self.SUBSCAP_BUBBLE_MM - d_min
                    if abs(delta) < 0.1:
                        break
                    # Damped (under-relaxed) proportional push: a full-step push
                    # assumes the local surface is planar along the fixed dorsal
                    # normal, which doesn't hold everywhere. On some geometry
                    # this overshoots the best-fit point -- clearance improves
                    # for a few iterations, then a different (now-closer)
                    # subscap point becomes the new minimum and clearance
                    # degrades again, a genuine limit cycle rather than noisy
                    # convergence, since the push direction stays fixed
                    # throughout the loop. Keeping the best (closest-to-target)
                    # state seen across all iterations, rather than trusting
                    # wherever the loop happens to stop, sidesteps that without
                    # needing the push itself to be perfectly stable.
                    push = n_scap_world() * delta * 0.5
                    ac  = ac  + push
                    cen = cen + push
                ac, cen = best_ac, best_cen
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
            return ac, cen, rot, None

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
            return ac, cen, rot, None

        # ══════════════════════════════════════════════════════════════════════
        # STEP 2 – Initial orientation: AA-TS-AI tangency (anatomical plane guess)
        # ══════════════════════════════════════════════════════════════════════
        _, n_thor_2 = self.get_surface_info(cen)
        
        # Zero out the Y (superior) component of the thorax normal to prevent
        # the scapula from aligning to the upward-facing shoulder shelf surface.
        # Without this, the spline normal at high Y values points superiorly,
        # which rotates the scapula flat/horizontal.
        n_thor_2_flat = n_thor_2.copy()
        n_thor_2_flat[1] = 0.0
        nlen = np.linalg.norm(n_thor_2_flat)
        if nlen > 1e-6:
            n_thor_2_flat /= nlen
        else:
            n_thor_2_flat = n_thor_2

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
            return ac, cen, rot, None

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
            return ac, cen, rot, None

        # ══════════════════════════════════════════════════════════════════════
        # STEP 4 – Joint rotation + translation fine-tuning
        # Nelder-Mead over 4 params: (tilt_deg, roll_deg, t_push, t_slide).
        # Earlier versions ran rotation-only, then a separate sequential
        # translation patch (Step 4b) if the coracoid was still penetrating
        # after Step 4 converged. That sequential design couldn't resolve
        # cases where subscap contact and coracoid clearance genuinely
        # conflict — the patch would fix one by re-breaking the other, since
        # each was solved blind to what the other needed, and got reverted.
        # Folding translation into the same search lets the optimizer trade
        # off subscap contact, AI/coracoid clearance, and the trapezius spring
        # together, so it can land on a genuine compromise instead of an
        # all-or-nothing patch.
        #
        # Translation is 2 DOF, not a free 3: t_push (dorsal, into/out of the
        # ribs — the same direction Step 3's subscap bubble already pushes
        # along) and t_slide (lateral, along the ribcage's tangent plane —
        # real scapular protraction/retraction). A free (dx,dy,dz) was tried
        # first and let the optimizer slide the scapula 20-30mm straight up
        # (any direction was equally cheap, and "up" happened to reduce
        # coracoid cost, which isn't how a scapula actually clears a rib
        # collision). Collapsing to push-only fixed that but overcorrected —
        # the coracoid and subscap cloud sit at different positions on a
        # curved surface, so one fixed direction usually can't satisfy both,
        # and the optimizer mostly just stopped moving. slide_dir is
        # constructed perpendicular to both push_dir and the vertical axis,
        # so it structurally cannot reproduce the vertical-drift failure
        # either way the search goes — not relying on a penalty to catch it.
        # Segment lengths — real bone lengths — are preserved via fabrik_solve,
        # not a free shift of AC; tilt/roll then apply an additional local
        # reorientation on top, same as before.
        # ══════════════════════════════════════════════════════════════════════
        correction_used = None
        if has_subscap:
            roll_axis = cen - ac
            rlen = np.linalg.norm(roll_axis)
            if rlen < 1e-6:
                return ac, cen, rot, None
            roll_axis /= rlen

            tilt_axis = np.cross(roll_axis, n_scap_world())
            tlen = np.linalg.norm(tilt_axis)
            if tlen < 1e-6:
                tilt_axis = np.cross(roll_axis, np.array([0., 1., 0.]))
                tlen = np.linalg.norm(tilt_axis)
            tilt_axis /= tlen

            # Seed snapshot for the cost function (avoid mutating mutable state
            # mid-search). cen_snap is needed alongside ac_snap/rot_snap now
            # that translation is part of the joint search below.
            rot_snap = rot
            ac_snap  = ac.copy()
            cen_snap = cen.copy()

            # Translation basis — 2 DOF, not 3, matching real scapulothoracic
            # motion instead of either extreme already tried:
            #   push_dir  — thorax's own local surface normal at the scapula's
            #               position (get_surface_info, the same posterior-
            #               pointing normal every clearance check here already
            #               trusts, grounded in C7/T8/IJ/PX via
            #               Thorax.build_jcs — not the scapula's own possibly-
            #               wrong AA-TS-AI normal). "Push into/out of the ribs."
            #   slide_dir — perpendicular to push_dir AND to the vertical (Y)
            #               axis by construction (cross product), so it has
            #               zero vertical component no matter what push_dir
            #               is. "Slide sideways along the ribcage" — real
            #               scapular protraction/retraction.
            # A free (dx,dy,dz) let the optimizer slide the scapula 20-30mm
            # straight UP (equally cheap in any direction, and "up" happened
            # to reduce coracoid cost). Collapsing to push_dir alone fixed
            # that but overcorrected: the coracoid and the subscapularis cloud
            # sit at different positions on a curved surface, so one fixed
            # direction usually isn't the direction that helps the coracoid
            # without also risking the tuned subscap fit — the optimizer
            # mostly stopped moving at all (observed: push settled at
            # 0.4-0.7mm, CP clearance back to -27/-29mm). Vertical drift is
            # excluded structurally here, not by hoping a penalty catches it,
            # so this can't reproduce the "sitting too high" failure either
            # way the search goes.
            # Flatten push_dir's vertical (Y) component before using it — the
            # raw spline normal from get_surface_info isn't purely horizontal,
            # it tilts with local rib curvature (the same issue Step 2 already
            # works around: "the spline normal at high Y values points
            # superiorly... rotates the scapula flat/horizontal" — see
            # n_thor_2_flat above). slide_dir was already built with zero
            # vertical component by construction, but push_dir wasn't, so a
            # large push (left needed -28.5mm) could still carry a real
            # vertical shift straight through unconstrained — confirmed
            # visually (scapula sitting well below where it should relative
            # to C7/T8, i.e. below the T2-T7 range) even though the dedicated
            # anti-drift term (slide_dir) was doing its job correctly.
            _, push_dir_raw = self.get_surface_info(cen_snap)
            push_dir = push_dir_raw.copy()
            push_dir[1] = 0.0
            push_len = np.linalg.norm(push_dir)
            if push_len < 1e-6:
                # Degenerate case: the surface normal points (near-)vertically
                # here, so there's no meaningful horizontal push direction —
                # disable the 2-DOF translation search for this joint instead
                # of dividing by ~0 (mirrors the tilt_axis/roll_axis guards
                # above).
                push_dir = np.zeros(3)
                slide_dir = np.zeros(3)
            else:
                push_dir /= push_len
                slide_dir = np.cross(push_dir, np.array([0., 1., 0.]))
                slide_dir /= np.linalg.norm(slide_dir)

            def _translate_and_rotate(t_push, t_slide, td, rd):
                """One joint trial: move the centroid target by t_push*push_dir
                + t_slide*slide_dir, re-solve the SC-AC-Centroid chain
                (preserves segment lengths — this is not a free shift of AC,
                the clavicle/AC-to-centroid reach are real bone lengths),
                derive the rotation translation alone implies (same technique
                as sync_rot_after_fabrik), then layer the (td, rd) tilt/roll on
                top as an additional local reorientation around the new AC.
                Returns (ac_eval, trial_rotation)."""
                offset = push_dir * t_push + slide_dir * t_slide
                if np.linalg.norm(offset) > 1e-9:
                    tchain = self.fabrik_solve(
                        [self.sc_joint.copy(), ac_snap.copy(), cen_snap.copy()],
                        lengths, cen_snap + offset)
                    ac_t, cen_t = tchain[1], tchain[2]
                    v_new, v_old = cen_t - ac_t, rot_snap.apply(centroid_loc)
                    if np.linalg.norm(v_new) > 1e-6 and np.linalg.norm(v_old) > 1e-6:
                        d_align, _ = R.align_vectors([v_new], [v_old])
                        rot_t = d_align * rot_snap
                    else:
                        rot_t = rot_snap
                else:
                    ac_t, rot_t = ac_snap, rot_snap
                trial = (R.from_rotvec(np.radians(td)*tilt_axis) *
                         R.from_rotvec(np.radians(rd)*roll_axis) * rot_t)
                return ac_t, trial

            def joint_cost(params):
                td, rd, t_push, t_slide = params

                # 1. HARD LIMITS: rotation within ±25° (tightened from ±40 —
                # larger rotations made the old bubble-translation runaway).
                # Push/slide each capped at 40mm, combined magnitude at 50mm —
                # corrective nudges for thorax-collision avoidance, not a free
                # reposition.
                if abs(td) > 25.0 or abs(rd) > 25.0:
                    return 1e9
                if abs(t_push) > 40.0 or abs(t_slide) > 40.0:
                    return 1e9
                t_mag = float(np.hypot(t_push, t_slide))
                if t_mag > 50.0:
                    return 1e9

                ac_eval, trial = _translate_and_rotate(t_push, t_slide, td, rd)

                # 1B. HARD VERTICAL-DRIFT LIMIT: push_dir/slide_dir are
                # horizontal by construction, but fabrik_solve's 2-segment
                # reach (SC sits at a different height than the centroid
                # target) can still shift AC's actual height as a side effect
                # of the IK chain reconfiguring for a large horizontal reach —
                # confirmed visually (left, which needed a much larger
                # combined push+slide than right, sat noticeably lower even
                # though neither direction vector had a vertical component).
                # At t_push=t_slide=0 this is always exactly 0 (ac_eval is
                # ac_snap unchanged), so the search always has a feasible
                # starting point — same reasoning as the trapezius limit that
                # had to be reverted earlier, applied correctly this time.
                if abs(ac_eval[1] - ac_snap[1]) > 15.0:
                    return 1e9

                # NOTE: a hard trapezius-deviation limit was tried here and
                # removed. Gated against the true rest length, it made the
                # (0,0,0,0,0) starting point itself infeasible whenever Steps
                # 1-3 (which have no muscle awareness at all — apply_bubble_
                # translation only cares about subscap contact) already used
                # most of the budget just satisfying subscap contact, and
                # Nelder-Mead collapsed to the reject wall with nowhere
                # feasible to go. Gated against wherever Step 4 itself starts
                # from instead, it stayed feasible but only bounded Step 4's
                # own contribution — Steps 1-3's own drift wasn't bounded by
                # anything, so the cumulative total could still end up
                # arbitrarily far from rest. Since the demo has to handle
                # arbitrary user-entered measurements, no fixed cutoff on
                # cumulative deviation can be assumed safe for every input
                # either. See section 9 below for the real constraint now.

                # Calculate transformed Subscapularis points and clearances
                sw = trial.apply(subscap_local_pts) + ac_eval
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

                # 4. AI TETHER: keep the Inferior Angle from penetrating the
                # ribs, and softly cap excessive lift-off. The threshold must be
                # reachable for this geometry — the AI naturally sits ~27mm off
                # the glide surface once Step 3's bubble translation stands the
                # scapula off. An 8mm target was unreachable, so its penalty
                # dominated the cost (~16000) and left the roll DOF undriven,
                # which broke L/R symmetry (each side settled in a different
                # noise-driven local min). A 30mm ceiling lets the symmetric
                # anatomical terms below shape the pose consistently on both sides.
                ai_w = trial.apply(ai_loc) + ac_eval
                p_ai_surf, n_ai_surf = self.get_surface_info(ai_w)
                ai_dist = np.dot(ai_w - p_ai_surf, n_ai_surf)

                if ai_dist < 0: # Inside ribs
                    cost += abs(ai_dist) * 10000.0
                elif ai_dist > 15.0: # Excessive lift-off (EXPERIMENTAL: was 30.0)
                    cost += (ai_dist - 15.0) * 1000.0

                # 4B. CORACOID CLEARANCE: DISABLED.
                #
                # Two reweighting attempts on this term (flat 8000/mm, then a
                # tolerance-then-stiffen shape like trapezius) both failed to
                # produce a sane pose on the harder (left) side — full history
                # in git blame. Root cause turned out to be upstream of any
                # weight: _fit_thorax_surface fits the B-spline to the
                # *posterior* glide region only (points within ~20mm of the
                # mid-X plane). The coracoid process sits well lateral/
                # anterior of that — verified empirically (SSM_103 mean
                # shape) that CP's Z coordinate falls ~30-34mm outside the
                # spline's fitted knot range on BOTH sides. get_surface_info
                # clips out-of-range queries to the nearest knot before
                # evaluating, so the "clearance" this term was penalizing was
                # the CP's distance from a clamped, physically meaningless
                # surface value — not real bone-to-rib clearance. That's what
                # was dragging the whole scapula (up to 45mm) to satisfy a
                # signal that didn't correspond to actual anatomy: the
                # optimizer really was reaching CP_dist=0.0mm, just against
                # the wrong target. Subscap contact and the AI tether are
                # unaffected — both sit inside the fitted posterior region,
                # where this surface is meaningful. Re-enable only after the
                # underlying surface is extended (or a second surface fit) to
                # actually cover where the coracoid sits.
                #
                # if cp_loc is not None:
                #     cp_w = trial.apply(cp_loc) + ac_eval
                #     p_cp_surf, n_cp_surf = self.get_surface_info(cp_w)
                #     cp_dist = np.dot(cp_w - p_cp_surf, n_cp_surf)
                #     if cp_dist < 0:  # Inside ribs
                #         cost += abs(cp_dist) * 8000.0

                # 5. CLAVICLE COLLISION GUARD: DISABLED.
                # The B-spline X=f(Y,Z) is fit to both anterior and posterior
                # thorax points and is single-valued, so it falsely reports the
                # clavicle midpoint (anterior, near IJ level) as "inside" the
                # posterior wall.  At c_mid≈(0, 0, 90) the spline returns the
                # posterior X≈-60, so c_dist≈-54mm and the 50k coefficient
                # produces a ~3 million constant cost that drowns out every
                # real signal in the Nelder-Mead search.

                # 6. SCAPULAR PLANE ANGLE: The dorsal normal projected onto the
                # horizontal XZ plane should be 25–50° from pure posterior (-X),
                # angled medially. This enforces the anatomical ~30–45° scapular
                # plane orientation visible in a top-down view.
                n_scap_trial = trial.apply(n_scap_loc)
                n_xz = np.array([n_scap_trial[0], 0.0, n_scap_trial[2]])
                nxz_len = np.linalg.norm(n_xz)
                if nxz_len > 1e-6:
                    n_xz /= nxz_len
                    angle_deg = np.degrees(np.arccos(np.clip(-n_xz[0], -1.0, 1.0)))
                    if angle_deg < 25.0:
                        cost += (25.0 - angle_deg) ** 2 * 2.0
                    elif angle_deg > 50.0:
                        cost += (angle_deg - 50.0) ** 2 * 2.0
                    # Lateral component must point toward the spine (medially).
                    # medial_component > 0 means pointing toward the body midline.
                    medial_component = -n_xz[2] * self.side_sign
                    if medial_component < 0.1:
                        cost += (0.1 - medial_component) * 50.0

                # 7. TS MEDIAL BORDER: Trigonum Spinae should stay within 120mm
                # of the spinal midline (Z = 0). Prevents excessive lateral drift.
                ts_w = trial.apply(ts_loc) + ac_eval
                ts_lat = abs(ts_w[2])
                if ts_lat > 120.0:
                    cost += (ts_lat - 120.0) ** 2 * 0.3

                # 8. UPWARD ROTATION: The Zs axis (TS→AA) should be near-horizontal
                # in neutral standing (~0–5°). Excessive upward rotation swings the
                # inferior angle (AI) medially toward the spine, as seen in the
                # top-down and front views. Penalise if the Y component of Zs exceeds
                # sin(10°) ≈ 0.17 upward, or sin(5°) ≈ 0.09 downward.
                zs_local_vec = aa_loc - ts_loc
                zs_len = np.linalg.norm(zs_local_vec)
                if zs_len > 1e-6:
                    zs_local_vec = zs_local_vec / zs_len
                    zs_world_vec = trial.apply(zs_local_vec)
                    upward_rot_sin = zs_world_vec[1]
                    if upward_rot_sin > 0.17:   # > ~10° upward rotation
                        cost += (upward_rot_sin - 0.17) ** 2 * 800.0
                    elif upward_rot_sin < -0.09:  # > ~5° downward rotation
                        cost += (upward_rot_sin + 0.09) ** 2 * 400.0

                # 9. TRAPEZIUS-INFORMED PULL: toward the C7-AC (upper fibres)
                # and T8-TS (lower fibres) lengths measured at the seed pose
                # (the SSM's uncorrected prediction). Now genuinely meaningful
                # for both terms since AC moves with dx/dy/dz.
                #
                # Not a plain spring from zero deviation — a real muscle has
                # slack before it starts resisting, then stiffens, so this
                # only penalizes the deviation BEYOND a tolerance. A flat
                # quadratic from zero couldn't be weighted well either way:
                # at 1.5, a 65mm compression cost ~6300 — a rounding error
                # against the 8000x-per-mm coracoid term, so the optimizer
                # spent it freely (observed). At 40, it correctly blocked
                # that, but also suppressed the right side's legitimate
                # ~15mm-deviation correction, making its coracoid clearance
                # *worse* (observed: -19.5mm -> -29.3mm) — a flat weight can't
                # simultaneously tolerate a normal correction and reject an
                # extreme one. A 20mm free tolerance plus a steep penalty only
                # on the excess does both: a 25mm deviation (5mm excess) costs
                # ~375 — cheap, barely discouraged; a 65mm deviation (45mm
                # excess) costs ~30375 — comparable to several mm of
                # coracoid penetration, a real deterrent.
                TRAP_TOLERANCE_MM = 20.0
                TRAP_K = 15.0
                if trap_upper_L0 is not None:
                    ac_dist = np.linalg.norm(ac_eval - c7)
                    excess = max(0.0, abs(ac_dist - trap_upper_L0) - TRAP_TOLERANCE_MM)
                    cost += (excess ** 2) * TRAP_K
                if trap_lower_L0 is not None:
                    ts_w_trial = trial.apply(ts_loc) + ac_eval
                    ts_dist = np.linalg.norm(ts_w_trial - t8)
                    excess = max(0.0, abs(ts_dist - trap_lower_L0) - TRAP_TOLERANCE_MM)
                    cost += (excess ** 2) * TRAP_K

                # 10. TRANSLATION REGULARIZATION: mild preference for not
                # translating more than necessary, so the optimizer only
                # spends the push/slide budget when rotation alone genuinely
                # can't resolve things — preserves the "no correction needed"
                # case (e.g. this model's right side previously converged at
                # tilt=roll=0 with no translation) rather than wandering for
                # no benefit.
                #
                # TRIED raising this a lot (2.0) to fight the flat-8000/mm
                # coracoid penalty's willingness to spend unlimited
                # translation. Didn't work — even doubled, its marginal
                # cost at 45mm (~180/mm) was nowhere near 8000/mm, so the
                # solution didn't move at all; reverted to the original
                # mild value. See 4B for why softening coracoid itself
                # (the other attempted fix) also didn't hold up.
                cost += (t_mag ** 2) * 0.05

                return cost

            if correction is not None:
                # Reuse a previously-solved correction (e.g. from the mean
                # shape) instead of re-running the search below — see the
                # `correction` parameter docstring above for the trade-off.
                bt, br, t_push_f, t_slide_f = correction
                cost_label = "n/a (reused, not re-searched)"
            else:
                # Multi-start: Nelder-Mead is a local search, sensitive to where it
                # begins, and this cost landscape is rugged (several steep,
                # competing penalty terms). A single run from the all-zero seed
                # left one side (right, on the SSM_103 mean-anthropometry case)
                # essentially stuck at its starting point while the other made
                # real progress from the same seed on a structurally identical
                # problem — evidence of a poor local minimum, not a harder
                # geometry.
                #
                # TRIED: cutting this to 3 slide-only seeds (origin + t_slide
                # ±20), on the theory that splitting push into its own DOF
                # (t_push vs t_slide, above) already fixed the underlying
                # capacity problem that motivated multi-start in the first
                # place. Verified empirically that it does NOT — left regressed
                # from cost=9545 (CP_dist=0.0mm, clean) to cost=65325
                # (CP_dist=-6.8mm, actively penetrating) because none of the
                # slide-only seeds reach the basin its real solution lives in
                # (push=-32.8mm, slide=-31.0mm — needs a push-biased seed to find
                # it). Multi-start across BOTH axes is load-bearing, not padding.
                # Reverted to the full seed set. The one real win kept from that
                # attempt: early-exit once a seed already lands a good-enough
                # cost, so the common/easy side doesn't pay for seeds it doesn't
                # need — this is safe because it only stops after finding a
                # result at least as good as the threshold, never trades quality
                # for speed.
                starts = [
                    [0., 0., 0., 0.],
                    [0., 0., 20., 0.], [0., 0., -20., 0.],
                    [0., 0., 0., 20.], [0., 0., 0., -20.],
                    [0., 0., 15., 15.], [0., 0., 15., -15.],
                ]
                GOOD_ENOUGH_COST = 2000.0
                best_res = None
                best_idx = -1
                for i, x0 in enumerate(starts):
                    r = minimize(joint_cost, x0=x0, method='Nelder-Mead',
                                 options={'xatol': 0.05, 'fatol': 0.01, 'maxiter': 3000})
                    if best_res is None or r.fun < best_res.fun:
                        best_res = r
                        best_idx = i
                    if best_res.fun < GOOD_ENOUGH_COST:
                        break
                res = best_res
                print(f"  FABRIK Step 4: {best_idx + 1}/{len(starts)} seed(s) tried, "
                      f"winning seed={starts[best_idx]}, cost={res.fun:.2f}")
                bt, br, t_push_f, t_slide_f = res.x
                cost_label = f"{res.fun:.2f}"

            ac, trial_final = _translate_and_rotate(t_push_f, t_slide_f, bt, br)
            rot = trial_final
            cen = rot.apply(centroid_loc) + ac
            chain = [self.sc_joint.copy(), ac.copy(), cen.copy()]

            df = subscap_clearances()

            ai_fin = world_lm(ai_loc)
            p_ai_fin, n_ai_fin = self.get_surface_info(ai_fin)
            ai_dist_fin = np.dot(ai_fin - p_ai_fin, n_ai_fin)

            cp_msg = ""
            if cp_loc is not None:
                cp_fin = world_lm(cp_loc)
                p_cp_fin, n_cp_fin = self.get_surface_info(cp_fin)
                cp_dist_fin = np.dot(cp_fin - p_cp_fin, n_cp_fin)
                cp_msg = f", CP_dist={cp_dist_fin:.1f}mm"

            trap_msg_s4 = ""
            if trap_upper_L0 is not None:
                trap_msg_s4 += f", trap_upper={np.linalg.norm(ac - c7):.1f}mm(rest={trap_upper_L0:.1f})"
            if trap_lower_L0 is not None:
                ts_trap_fin = world_lm(ts_loc)
                trap_msg_s4 += f", trap_lower={np.linalg.norm(ts_trap_fin - t8):.1f}mm(rest={trap_lower_L0:.1f})"

            step4_mode = "cached" if correction is not None else "joint"
            print(f"  FABRIK Step 4 ({step4_mode}): tilt={bt:.1f}°, roll={br:.1f}°, "
                  f"push={t_push_f:.1f}mm (dorsal), slide={t_slide_f:.1f}mm (lateral), "
                  f"Cost={cost_label}, d_min={np.min(df):.1f}mm, AI_dist={ai_dist_fin:.1f}mm{cp_msg}{trap_msg_s4}")
            correction_used = (bt, br, t_push_f, t_slide_f)
        else:
            # Legacy: single-axis roll sweep for medial clearance
            p_c4, n_t4 = self.get_surface_info(cen)
            roll_axis = cen - ac
            rlen = np.linalg.norm(roll_axis)
            if rlen < 1e-6: return ac, cen, rot, None
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

        return ac, cen, rot, correction_used


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

    ac_sol, cen_sol, rot_sol, _correction = solver.solve_alignment(
        ac_joint, centroid, lms_local, None, p_proj,
        subscap_seed=subscap_seed,
        initial_rot=initial_rot,
        max_step=max_step,
    )

    mesh_centered = np.array(scap_mesh) - np.array(ac_joint)
    final_mesh = rot_sol.apply(mesh_centered) + ac_sol
    return final_mesh, ac_sol, rot_sol
