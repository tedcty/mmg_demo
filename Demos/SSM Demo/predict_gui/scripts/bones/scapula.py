"""Scapula — the most complex bone in the assembly.

Runs the full 4-step FABRIK pipeline on both sides
(FabrikScapulaSolver is imported directly from fabrik_solver).

`assemble_simple` is retained as a JCS-only fallback.
"""
from __future__ import annotations
import os
import numpy as np
from scipy.spatial.transform import Rotation as R
from ptb.util.math.transformation import Cloud
from .base_bone import BoneBase
from .thorax import Thorax
from .clavicle import Clavicle


class Scapula(BoneBase):

    def __init__(self, side: str):
        super().__init__(side=side)
        self.label = f"{'R' if side == 'right' else 'L'} Scapula"
        self.color = "#FFA040" if side == "right" else "#FFE060"

        # World-space anatomical landmarks (set after assemble)
        self.aa: np.ndarray = np.zeros(3)
        self.ts: np.ndarray = np.zeros(3)
        self.ai: np.ndarray = np.zeros(3)
        self.cp: np.ndarray = np.zeros(3)  # Coracoid Process (CAP)
        self.ac_joint: np.ndarray = np.zeros(3)
        self.gh_joint_seed: np.ndarray = np.zeros(3)  # GH on scapula side
        self.plane_normal: np.ndarray = np.array([-1., 0., 0.])

        # Solved rotation (used by Humerus to follow GH)
        self.solved_rot: R = R.identity()
        self._ac_seed: np.ndarray = np.zeros(3)   # AC before FABRIK
        # Step-4 (tilt_deg, roll_deg, t_push, t_slide) actually applied —
        # set by assemble_fabrik, reusable via its `correction` param later.
        self.fabrik_correction: tuple | None = None

        # Subscapularis cloud
        self.subscapularis: np.ndarray | None = None

    # ── Loading ───────────────────────────────────────────────────────────────

    def load(
        self,
        case_arr: np.ndarray,
        all_faces: list,
        maps_dir: str,
        res_dir: str,
    ) -> "Scapula":
        s = self.side
        prefix = "r" if s == "right" else "l"
        bone_csv = "R_scap.csv" if s == "right" else "L_scap.csv"

        verts, inds, valid_ids = self.filter_bone_indices(case_arr, all_faces, maps_dir, bone_csv)
        self._raw_verts = verts
        self.indices    = inds
        self._valid_ids = valid_ids

        self._aa_raw = self.get_landmark(case_arr, maps_dir, f"sca_{prefix}_aa.csv")
        self._ai_raw = self.get_landmark(case_arr, maps_dir, f"sca_{prefix}_ai.csv")
        self._ts_raw = self.get_landmark(case_arr, maps_dir, f"sca_{prefix}_ts.csv")
        self._cp_raw = self.get_landmark(case_arr, maps_dir, f"sca_{prefix}_cap.csv")
        self._gh_raw = self.get_sphere_center(case_arr, maps_dir, f"scap_ghj_{prefix}.csv")

        # Subscapularis cloud (ID 69)
        subscap_dir  = "Scapula_right" if s == "right" else "Scapula_left"
        subscap_path = os.path.join(res_dir, "MAS_103", subscap_dir, "69_NodeNo_2.csv")
        self._subscap_path = subscap_path
        self._subscap_raw = self.load_muscle_cloud(case_arr, subscap_path)
        return self

    # ── JCS ───────────────────────────────────────────────────────────────────

    def _build_jcs_matrix(self, thorax: Thorax) -> np.ndarray:
        """Compute scapula JCS (xs, ys, zs) and return the 4×4 transform.

        The AA-TS-AI triangle has opposite chirality between sides, so the
        cross-product axis flips sign. We flip zs and the cross-product
        operands for the LEFT side so its source frame ends up aligned with
        world identity (same convention as Clavicle.assemble and the
        reference articulatingTest6th implementation).
        """
        aa, ai, ts = self._aa_raw, self._ai_raw, self._ts_raw
        if self.side == "right":
            zs = (aa - ts) / np.linalg.norm(aa - ts)
            xs = np.cross(ai - aa, ts - aa)
        else:
            zs = (ts - aa) / np.linalg.norm(ts - aa)
            xs = np.cross(ts - aa, ai - aa)
        xs /= np.linalg.norm(xs)
        ys = np.cross(zs, xs)
        s_source = np.array([xs, ys, zs]).T
        s_mat = Cloud.transform_between_3x3_points_sets(s_source, np.eye(3))

        return s_mat

    # ── Assembly (no FABRIK — left side) ─────────────────────────────────────

    def assemble_simple(self, thorax: Thorax, clavicle: Clavicle) -> "Scapula":
        """Simple JCS seed assembly (used for the left scapula)."""
        ij    = thorax.ij_pt
        s_mat = self._build_jcs_matrix(thorax)

        ac_offset = clavicle.ac_joint - (s_mat[:3, :3] @ (self._aa_raw - ij))

        self.vertices = self.transform_mesh(self._raw_verts, ij, s_mat) + ac_offset
        self.aa       = self.transform_mesh([self._aa_raw], ij, s_mat)[0] + ac_offset
        self.ts       = self.transform_mesh([self._ts_raw], ij, s_mat)[0] + ac_offset
        self.ai       = self.transform_mesh([self._ai_raw], ij, s_mat)[0] + ac_offset
        self.cp       = self.transform_mesh([self._cp_raw], ij, s_mat)[0] + ac_offset
        self.ac_joint = clavicle.ac_joint.copy()
        self.origin   = self.ac_joint.copy()

        # GH on scapula side
        self.gh_joint_seed = (s_mat[:3, :3] @ (self._gh_raw - ij)) + ac_offset

        # Subscapularis
        if self._subscap_raw is not None:
            self.subscapularis = self.transform_mesh(self._subscap_raw, ij, s_mat) + ac_offset

        self._compute_plane_normal()
        return self

    # ── Assembly with FABRIK (right side) ─────────────────────────────────────

    def assemble_fabrik(
        self,
        thorax: Thorax,
        clavicle: Clavicle,
        fabrik_step: int = 4,
        correction: tuple | None = None,
    ) -> "Scapula":
        """Full FABRIK pipeline assembly (used for the right scapula).

        `correction`, when given, is a previously-solved Step-4
        (tilt_deg, roll_deg, t_push, t_slide) reused instead of re-running
        the ~90s Nelder-Mead search — see solve_alignment's docstring. The
        actually-applied correction (freshly solved, reused, or None if
        Step 4 didn't run) ends up on self.fabrik_correction either way, so
        a caller can capture it from a full solve and pass it back in later.
        """
        # Import here to keep the module dependency explicit
        from fabrik_solver import FabrikScapulaSolver

        ij    = thorax.ij_pt
        s_mat = self._build_jcs_matrix(thorax)
        ac_seed = clavicle.ac_joint.copy()

        ac_offset    = ac_seed - (s_mat[:3, :3] @ (self._aa_raw - ij))
        aa_seed      = (s_mat[:3, :3] @ (self._aa_raw - ij)) + ac_offset
        ts_seed      = (s_mat[:3, :3] @ (self._ts_raw - ij)) + ac_offset
        ai_seed      = (s_mat[:3, :3] @ (self._ai_raw - ij)) + ac_offset
        centroid_seed = (aa_seed + ts_seed + ai_seed) / 3.0
        mesh_seed    = self.transform_mesh(self._raw_verts, ij, s_mat) + ac_offset
        gh_seed      = (s_mat[:3, :3] @ (self._gh_raw - ij)) + ac_offset
        cp_seed      = (s_mat[:3, :3] @ (self._cp_raw - ij)) + ac_offset

        subscap_seed = None
        if self._subscap_raw is not None:
            subscap_seed = self.transform_mesh(self._subscap_raw, ij, s_mat) + ac_offset

        # Projection target on posterior ribcage
        p_proj = thorax.project_scapula(aa_seed, ts_seed, ai_seed, self.side)

        # World-space C7/T8 (midline spine landmarks, already loaded by Thorax) —
        # trapezius origin proxies for Step 4's joint rotation+translation
        # search.
        tho_rot3 = thorax.jcs_matrix[:3, :3]
        c7_g = tho_rot3 @ (thorax.c7_pt - thorax.ij_pt)
        t8_g = tho_rot3 @ (thorax.t8_pt - thorax.ij_pt)

        # Run FABRIK
        solver = FabrikScapulaSolver(thorax.vertices, clavicle.sc_joint)

        # DIAGNOSTIC: raw JCS-seed clearance, before ANY FABRIK correction
        # (Steps 1-4 haven't run yet) — isolates whether left/right asymmetry
        # downstream traces back to the seed itself (possible chirality issue
        # in _build_jcs_matrix, which already uses a different formula per
        # side) versus the correction search. If left's seed is dramatically
        # worse than right's here, that's the JCS construction, not Step 4.
        p_cp_seed, n_cp_seed = solver.get_surface_info(cp_seed)
        cp_seed_clearance = np.dot(cp_seed - p_cp_seed, n_cp_seed)
        subscap_seed_min = None
        if subscap_seed is not None:
            seed_dists = np.array([
                np.dot(pt - solver.get_surface_info(pt)[0], solver.get_surface_info(pt)[1])
                for pt in np.array(subscap_seed)
            ])
            subscap_seed_min = float(np.min(seed_dists))
        subscap_msg = f"{subscap_seed_min:.1f}mm" if subscap_seed_min is not None else "n/a"
        print(f"  FABRIK SEED DIAG ({self.side}): raw JCS seed (pre-correction) — "
              f"CP_clearance={cp_seed_clearance:.1f}mm, subscap_d_min={subscap_msg}")

        # Validate anchor: must be on the correct side and within 120 mm of
        # the seed centroid in Y.  A bad anchor (wrong side or wildly too low)
        # causes FABRIK to chase the wrong wall and dislocate the chain.
        expected_z_sign = 1.0 if self.side == "right" else -1.0
        anchor_z_ok = (p_proj[2] * expected_z_sign) > 0
        anchor_y_ok = abs(p_proj[1] - centroid_seed[1]) < 120.0
        if not anchor_z_ok or not anchor_y_ok:
            print(
                f"  FABRIK WARNING ({self.side}): anchor failed validation "
                f"(z_ok={anchor_z_ok}, y_ok={anchor_y_ok}, "
                f"anchor_Y={p_proj[1]:.1f}, seed_Y={centroid_seed[1]:.1f}). "
                f"Falling back to centroid projection."
            )
            p_proj, _ = solver.get_surface_info(centroid_seed)
        lms_local = {
            'aa': aa_seed - ac_seed,
            'ts': ts_seed - ac_seed,
            'ai': ai_seed - ac_seed,
            'cp': cp_seed - ac_seed,   # coracoid — Step 4 now penalizes it penetrating the thorax
        }
        # lms_local vectors are in world-offset space (world minus ac_seed), so the
        # solver's initial rotation must be identity — the local frame IS world.
        # Using R.from_matrix(s_mat) caused double-rotation for the left side
        # (seed_rot_L ≠ identity) while accidentally working for the right side
        # (seed_rot_R ≈ identity because the right JCS aligns with world axes).
        rot_seed = R.identity()
        ac_sol, cen_sol, rot_sol, self.fabrik_correction = solver.solve_alignment(
            ac_seed, centroid_seed, lms_local, None, p_proj,
            subscap_seed=subscap_seed,
            initial_rot=rot_seed,
            c7=c7_g, t8=t8_g,
            max_step=fabrik_step,
            correction=correction,
        )

        # SANITY CLAMP: FABRIK should never move AC more than 100 mm in Y or
        # 130 mm total from the seed. If it does, the chain has gone unstable
        # (bad anchor / runaway bubble translation) and we fall back to the
        # JCS seed pose to avoid visible dislocation.
        # Was 60/80mm — calibrated when Step 4 was rotation-only, so all AC
        # movement came from Steps 1-3. Step 4 now includes a deliberate
        # translation search (up to 40mm) as part of resolving genuine
        # subscap-vs-coracoid conflicts, so the same cumulative movement that
        # used to only mean "something went wrong" can now also mean "the
        # optimizer found a real fix" — the old threshold was discarding
        # correct, intentional results (observed: 81.6mm total for a result
        # that brought coracoid clearance from -15.8mm to -3.8mm) along with
        # actually-broken ones. Widened rather than removed — a genuinely
        # unstable chain (bad anchor, runaway search) should still get caught.
        d_total = float(np.linalg.norm(ac_sol - ac_seed))
        d_y     = float(abs(ac_sol[1] - ac_seed[1]))
        if d_y > 100.0 or d_total > 130.0:
            print(
                f"  FABRIK SANITY ({self.side}): ac_sol drifted too far "
                f"(|Δy|={d_y:.1f}mm, |Δ|={d_total:.1f}mm). Falling back to seed pose."
            )
            ac_sol  = ac_seed.copy()
            cen_sol = centroid_seed.copy()
            rot_sol = R.identity()

        # Apply solved rotation to mesh and landmarks
        self._ac_seed  = ac_seed
        self.solved_rot = rot_sol

        # Store for replay_vertices — the seed-stage transform (ij/s_mat/
        # ac_offset) plus the solved correction stage (_ac_seed/solved_rot/
        # ac_sol, the last set below as self.ac_joint) together fully define
        # how raw mesh vertices become world-space ones. Freezing all of
        # these lets a later call re-derive .vertices from a *different*
        # mesh's raw vertices while reusing this exact joint pose.
        self._ij        = ij
        self._s_mat      = s_mat
        self._ac_offset  = ac_offset

        def _apply(pt: np.ndarray) -> np.ndarray:
            return rot_sol.apply(pt - ac_seed) + ac_sol

        self.vertices  = rot_sol.apply(mesh_seed - ac_seed) + ac_sol
        self.aa        = _apply(aa_seed)
        self.ts        = _apply(ts_seed)
        self.ai        = _apply(ai_seed)
        self.cp        = _apply(cp_seed)
        self.ac_joint  = ac_sol.copy()
        self.origin    = ac_sol.copy()
        self.gh_joint_seed = _apply(gh_seed)

        # --- CORACOID CLEARANCE CHECK (diagnostic only — NOT reliable) ---
        # Step 4's coracoid penalty is disabled (see fabrik_solver.joint_cost,
        # term 4B) because this check itself is unsound: _fit_thorax_surface
        # only fits the posterior glide region, and the coracoid process sits
        # ~30-34mm outside that fitted range on both sides (verified on the
        # SSM_103 mean shape) — get_surface_info clips out-of-range queries
        # to the nearest knot, so the number below is CP's distance from a
        # clamped, not-actually-fitted surface value, not real bone-to-rib
        # clearance. Left in only as a rough heads-up, not a pass/fail check.
        p_cp_surf, n_cp_surf = solver.get_surface_info(self.cp)
        cp_clearance = np.dot(self.cp - p_cp_surf, n_cp_surf)
        print(f"  FABRIK: Coracoid (CAP) clearance from thorax = {cp_clearance:.1f}mm "
              f"(diagnostic only, surface not fitted this far out — see comment above)")

        if subscap_seed is not None:
            self.subscapularis = rot_sol.apply(subscap_seed - ac_seed) + ac_sol

        self._compute_plane_normal()
        return self

    def replay(self, case_arr: np.ndarray, maps_dir: str, clavicle: Clavicle) -> "Scapula":
        """Recompute the visible mesh AND all scapula landmarks (aa/ts/ai/
        cp/ac_joint/gh_joint_seed/subscapularis) from a new case_arr,
        reusing this instance's already-solved _ij/_s_mat (seed JCS
        orientation) and solved_rot (the FABRIK Step-4 orientation) — the
        scapula's ORIENTATION stays frozen — but re-deriving every position
        from fresh landmarks on the new mesh via `clavicle`'s (already
        replayed) ac_joint as the new seed anchor.

        The FABRIK correction is reused as a *relative* displacement
        (self.ac_joint - self._ac_seed, i.e. how far Step 4 moved AC from
        its seed) rather than an absolute world point, so it still makes
        sense once the seed itself has moved — e.g. because this PC weight
        made the scapula a different size. No search re-runs; this is a
        direct evaluation. See Thorax's version of this method for more
        context on the general freeze-rotation/recompute-position approach.
        """
        prefix = "r" if self.side == "right" else "l"
        raw_verts = case_arr[self._valid_ids]

        aa_raw = self.get_landmark(case_arr, maps_dir, f"sca_{prefix}_aa.csv")
        ai_raw = self.get_landmark(case_arr, maps_dir, f"sca_{prefix}_ai.csv")
        ts_raw = self.get_landmark(case_arr, maps_dir, f"sca_{prefix}_ts.csv")
        cp_raw = self.get_landmark(case_arr, maps_dir, f"sca_{prefix}_cap.csv")
        gh_raw = self.get_sphere_center(case_arr, maps_dir, f"scap_ghj_{prefix}.csv")

        ij, s_mat = self._ij, self._s_mat
        ac_seed = clavicle.ac_joint.copy()
        ac_offset = ac_seed - (s_mat[:3, :3] @ (aa_raw - ij))

        def _seed(pt: np.ndarray) -> np.ndarray:
            return (s_mat[:3, :3] @ (pt - ij)) + ac_offset

        mesh_seed = self.transform_mesh(raw_verts, ij, s_mat) + ac_offset
        aa_seed, ts_seed, ai_seed, cp_seed, gh_seed = (
            _seed(aa_raw), _seed(ts_raw), _seed(ai_raw), _seed(cp_raw), _seed(gh_raw)
        )

        rot_sol = self.solved_rot
        ac_sol = ac_seed + (self.ac_joint - self._ac_seed)  # frozen delta, fresh seed

        def _apply(pt: np.ndarray) -> np.ndarray:
            return rot_sol.apply(pt - ac_seed) + ac_sol

        self.vertices      = rot_sol.apply(mesh_seed - ac_seed) + ac_sol
        self.aa            = _apply(aa_seed)
        self.ts            = _apply(ts_seed)
        self.ai            = _apply(ai_seed)
        self.cp            = _apply(cp_seed)
        self.ac_joint      = ac_sol.copy()
        self.origin        = ac_sol.copy()
        self.gh_joint_seed = _apply(gh_seed)

        subscap_raw = self.load_muscle_cloud(case_arr, self._subscap_path)
        if subscap_raw is not None:
            subscap_seed = self.transform_mesh(subscap_raw, ij, s_mat) + ac_offset
            self.subscapularis = rot_sol.apply(np.array(subscap_seed) - ac_seed) + ac_sol

        self._compute_plane_normal()
        return self

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _compute_plane_normal(self) -> None:
        n = np.cross(self.ai - self.ts, self.aa - self.ts)
        n /= np.linalg.norm(n)
        if n[0] > 0:
            n = -n
        self.plane_normal = n

    # ── Diagnostics ───────────────────────────────────────────────────────────

    def get_diagnostic_markers(self) -> list[dict]:
        p = "R" if self.side == "right" else "L"
        return [
            {"pos": self.aa.tolist(), "label": f"{p}_AA", "color": "#FF4444"},
            {"pos": self.ts.tolist(), "label": f"{p}_TS", "color": "#44FF44"},
            {"pos": self.ai.tolist(), "label": f"{p}_AI", "color": "#4444FF"},
            {"pos": ((self.aa + self.ts + self.ai) / 3).tolist(), "label": f"{p}_Centroid", "color": "#FFFFFF"},
        ]

    def scapular_plane_dict(self) -> dict:
        return {
            "aa":       self.aa.tolist(),
            "ts":       self.ts.tolist(),
            "ai":       self.ai.tolist(),
            "centroid": ((self.aa + self.ts + self.ai) / 3).tolist(),
            "normal":   self.plane_normal.tolist(),
        }

    def landmark_globals(self) -> dict:
        return {
            "scapula_ac": self.ac_joint.tolist(),
            "scapula_aa": self.aa.tolist(),
            "scapula_ts": self.ts.tolist(),
            "scapula_ai": self.ai.tolist(),
        }
