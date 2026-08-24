"""Thorax — root bone of the shoulder kinematic chain.

Owns:
- The ISB thoracic coordinate frame (IJ origin).
- Both Sternoclavicular (SC) joint centres.
- The posterior-wall projection method used to anchor the Scapula.
"""
from __future__ import annotations
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.interpolate import LSQBivariateSpline
from ptb.util.math.transformation import Cloud
from .base_bone import BoneBase


class Thorax(BoneBase):
    label = "Thorax"
    color = "#90CFF0"

    def __init__(self):
        super().__init__(side=None)
        # Anatomical landmarks (raw model space, set by load())
        self.ij_pt: np.ndarray = np.zeros(3)
        self.px_pt: np.ndarray = np.zeros(3)
        self.c7_pt: np.ndarray = np.zeros(3)
        self.t8_pt: np.ndarray = np.zeros(3)

        # World-space SC joint centres (set by assemble())
        self.sc_r: np.ndarray = np.zeros(3)
        self.sc_l: np.ndarray = np.zeros(3)

        # 4×4 JCS transform (raw → global)
        self.jcs_matrix: np.ndarray = np.eye(4)

    # ── Loading ───────────────────────────────────────────────────────────────

    def load(
        self,
        case_arr: np.ndarray,
        all_faces: list,
        maps_dir: str,
    ) -> "Thorax":
        """Load mesh and extract all thorax landmarks."""
        self.ij_pt = self.get_landmark(case_arr, maps_dir, "tho_ij.csv")
        self.px_pt = self.get_landmark(case_arr, maps_dir, "tho_px.csv")
        self.c7_pt = 0.5 * (
            self.get_landmark(case_arr, maps_dir, "tho_c7_r.csv")
            + self.get_landmark(case_arr, maps_dir, "tho_c7_l.csv")
        )
        self.t8_pt = 0.5 * (
            self.get_landmark(case_arr, maps_dir, "tho_t8_r.csv")
            + self.get_landmark(case_arr, maps_dir, "tho_t8_l.csv")
        )
        verts, inds, valid_ids = self.filter_bone_indices(case_arr, all_faces, maps_dir, "Tho.csv")
        self._raw_verts = verts
        self.indices = inds
        self._valid_ids = valid_ids

        # SC joint centres (raw)
        self._sc_r_raw = self.get_sphere_center(case_arr, maps_dir, "tho_scj_r.csv")
        self._sc_l_raw = self.get_sphere_center(case_arr, maps_dir, "tho_scj_l.csv")
        return self

    # ── JCS ───────────────────────────────────────────────────────────────────

    def build_jcs(self) -> "Thorax":
        """Compute the ISB thoracic coordinate frame and transform the mesh."""
        ij, px, c7, t8 = self.ij_pt, self.px_pt, self.c7_pt, self.t8_pt
        mid_px_t8 = 0.5 * (px + t8)
        mid_ij_c7 = 0.5 * (ij + c7)

        yt = (mid_ij_c7 - mid_px_t8)
        yt /= np.linalg.norm(yt)
        zt = np.cross(c7 - ij, mid_px_t8 - ij)
        zt /= np.linalg.norm(zt)
        xt = np.cross(yt, zt)

        t_source = np.array([xt, yt, zt]).T
        self.jcs_matrix = Cloud.transform_between_3x3_points_sets(t_source, np.eye(3))

        self.vertices = self.transform_mesh(self._raw_verts, ij, self.jcs_matrix)

        rot3 = self.jcs_matrix[:3, :3]
        self.sc_r = rot3 @ (self._sc_r_raw - ij)
        self.sc_l = rot3 @ (self._sc_l_raw - ij)

        # Store axes for children
        self.yt = yt
        self.origin = np.zeros(3)
        return self

    def replay(self, case_arr: np.ndarray, maps_dir: str) -> "Thorax":
        """Recompute the visible mesh AND joint centres (sc_r/sc_l) from a
        new case_arr, reusing this instance's already-solved ij_pt/
        jcs_matrix/_valid_ids — i.e. the thorax's ORIENTATION stays frozen
        (that's the part a JCS derivation could get subtly wrong on extreme
        shapes), but sc_r/sc_l are re-derived from fresh landmarks on the
        new mesh so they track the new anatomy instead of leaving a gap
        where the clavicle no longer quite reaches. `self.indices` (the
        bone-local face list) is untouched: it's purely a function of mesh
        topology, which never changes across PC weights. Used by
        generate_isb_joints.replay_shape for the PC-adjustment tab."""
        raw_verts = case_arr[self._valid_ids]
        self.vertices = self.transform_mesh(raw_verts, self.ij_pt, self.jcs_matrix)

        sc_r_raw = self.get_sphere_center(case_arr, maps_dir, "tho_scj_r.csv")
        sc_l_raw = self.get_sphere_center(case_arr, maps_dir, "tho_scj_l.csv")
        rot3 = self.jcs_matrix[:3, :3]
        self.sc_r = rot3 @ (sc_r_raw - self.ij_pt)
        self.sc_l = rot3 @ (sc_l_raw - self.ij_pt)
        return self

    # ── Scapula projection ────────────────────────────────────────────────────

    def project_scapula(
        self,
        aa: np.ndarray,
        ts: np.ndarray,
        ai: np.ndarray,
        side: str,
    ) -> np.ndarray:
        """Return a stable posterior-wall anchor point for the given scapula triangle.

        Uses a height-consistent (same Y as centroid), laterally-bounded landing
        zone to prevent drift onto the shoulder slope.
        """
        mesh = self.vertices
        centroid = (aa + ts + ai) / 3.0

        rot3 = self.jcs_matrix[:3, :3]
        px_g = rot3 @ (self.px_pt - self.ij_pt)
        t8_g = rot3 @ (self.t8_pt - self.ij_pt)
        print(f"  DIAG: PX_X={px_g[0]:.1f}, T8_X={t8_g[0]:.1f}")

        x_mid = (mesh[:, 0].min() + mesh[:, 0].max()) / 2.0
        z_mid = (mesh[:, 2].min() + mesh[:, 2].max()) / 2.0
        y_t8  = t8_g[1]

        # Laterally-bounded posterior zone. Try progressively wider bands
        # but always keep the same-side sign so we never anchor to the wrong wall.
        side_sign = 1.0 if side == "right" else -1.0
        post_mask = mesh[:, 0] < x_mid - 20

        def _side_mask(inner_mm, outer_mm):
            lo = z_mid + side_sign * inner_mm
            hi = z_mid + side_sign * outer_mm
            if side_sign > 0:
                return (mesh[:, 2] > lo) & (mesh[:, 2] < hi)
            else:
                return (mesh[:, 2] < lo) & (mesh[:, 2] > hi)

        glide_pts = None
        for inner, outer in [(80, 220), (50, 250), (20, 300)]:
            mask = _side_mask(inner, outer) & post_mask & (mesh[:, 1] > y_t8 - 150)
            pts  = mesh[mask]
            if len(pts) >= 50:
                glide_pts = pts
                break

        if glide_pts is None:
            # Last resort: broad same-side posterior zone, no height lower-bound
            mask = _side_mask(10, 300) & (mesh[:, 0] < x_mid)
            glide_pts = mesh[mask]
            if len(glide_pts) < 10:
                print("  FABRIK WARNING: Cannot find anchor – using centroid.")
                return centroid

        # Anatomy-driven anchor height: place the anchor so the scapula's
        # inferior angle (AI) ends up at approximately T7–T8 level in neutral.
        centroid_above_ai = centroid[1] - ai[1]
        target_centroid_y = y_t8 + 20.0 + centroid_above_ai

        # Try height bands of increasing width; if all fail pick closest-in-Y
        # (not most-posterior) to avoid anchoring at the bottom of the ribcage.
        candidates = None
        for window in [30.0, 60.0, 120.0]:
            hm = np.abs(glide_pts[:, 1] - target_centroid_y) < window
            if np.any(hm):
                candidates = glide_pts[hm]
                break

        if candidates is not None:
            best_idx = np.argmin(candidates[:, 0])   # most posterior within band
        else:
            best_idx = np.argmin(np.abs(glide_pts[:, 1] - target_centroid_y))

        projected_pt = glide_pts[best_idx] if candidates is None else candidates[best_idx]

        print(
            f"  FABRIK PROJ ({side}): target_Y={target_centroid_y:.1f} (T8={y_t8:.1f}), "
            f"anchored Y={projected_pt[1]:.1f} X={projected_pt[0]:.1f} Z={projected_pt[2]:.1f}"
        )
        return projected_pt

    # ── Diagnostics ───────────────────────────────────────────────────────────

    def landmark_globals(self) -> dict:
        """World-space thorax landmarks for JSON export."""
        rot3 = self.jcs_matrix[:3, :3]
        ij   = self.ij_pt
        return {
            "thorax_ij": [0, 0, 0],
            "thorax_px": (rot3 @ (self.px_pt - ij)).tolist(),
            "thorax_c7": (rot3 @ (self.c7_pt - ij)).tolist(),
            "thorax_t8": (rot3 @ (self.t8_pt - ij)).tolist(),
        }
