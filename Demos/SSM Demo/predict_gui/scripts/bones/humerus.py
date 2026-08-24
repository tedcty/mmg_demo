"""Humerus — distal bone in the shoulder kinematic chain.

Aligns its Glenohumeral (GH) joint to the scapula's solved GH centre.
"""
from __future__ import annotations
import numpy as np
from ptb.util.math.transformation import Cloud
from .base_bone import BoneBase
from .scapula import Scapula
from .thorax import Thorax


class Humerus(BoneBase):

    def __init__(self, side: str):
        super().__init__(side=side)
        self.label = f"{'R' if side == 'right' else 'L'} Humerus"
        self.color = "#FF6060"
        self.gh_joint: np.ndarray = np.zeros(3)   # World GH centre

    # ── Loading ───────────────────────────────────────────────────────────────

    def load(
        self,
        case_arr: np.ndarray,
        all_faces: list,
        maps_dir: str,
    ) -> "Humerus":
        s = self.side
        prefix = "r" if s == "right" else "l"
        bone_csv = "R_hum.csv" if s == "right" else "L_hum.csv"

        verts, inds, valid_ids = self.filter_bone_indices(case_arr, all_faces, maps_dir, bone_csv)
        self._raw_verts = verts
        self.indices    = inds
        self._valid_ids = valid_ids

        self._gh_raw = self.get_sphere_center(case_arr, maps_dir, f"hum_ghj_{prefix}.csv")
        self._el_raw = self.get_landmark(case_arr, maps_dir, f"hum_{prefix}_el.csv")
        self._em_raw = self.get_landmark(case_arr, maps_dir, f"hum_{prefix}_em.csv")
        return self

    # ── Assembly ──────────────────────────────────────────────────────────────

    def assemble(self, thorax: Thorax, scapula: Scapula) -> "Humerus":
        """Align humerus GH joint to the scapula's solved GH centre."""
        ij  = thorax.ij_pt
        gh  = self._gh_raw
        el, em = self._el_raw, self._em_raw
        mid_ep = 0.5 * (el + em)

        yh = (gh - mid_ep) / np.linalg.norm(gh - mid_ep)

        if self.side == "right":
            xh_raw = np.cross(el - gh, em - gh)
        else:
            xh_raw = np.cross(em - gh, el - gh)
        xh = xh_raw / np.linalg.norm(xh_raw)
        zh = np.cross(xh, yh)

        h_mat = Cloud.transform_between_3x3_points_sets(
            np.array([xh, yh, zh]).T, np.eye(3)
        )

        # Align world GH to the scapula's solved GH
        sca_gh_world = scapula.gh_joint_seed
        gh_offset = sca_gh_world - (h_mat[:3, :3] @ (gh - ij))

        self.vertices  = self.transform_mesh(self._raw_verts, ij, h_mat) + gh_offset
        self.gh_joint  = sca_gh_world.copy()
        self.origin    = sca_gh_world.copy()

        # Store for replay
        self._ij    = ij
        self._h_mat = h_mat
        return self

    def replay(self, case_arr: np.ndarray, maps_dir: str, scapula: Scapula) -> "Humerus":
        """Recompute the visible mesh AND gh_joint from a new case_arr,
        reusing this instance's already-solved _ij/_h_mat (the humerus's
        ORIENTATION stays frozen) but re-deriving gh_joint from a fresh GH
        landmark and `scapula`'s (already replayed) gh_joint_seed, so it
        tracks the new anatomy instead of leaving a gap at the GH joint.
        See Thorax's version of this method for context."""
        prefix = "r" if self.side == "right" else "l"
        raw_verts = case_arr[self._valid_ids]

        gh_raw = self.get_sphere_center(case_arr, maps_dir, f"hum_ghj_{prefix}.csv")
        sca_gh_world = scapula.gh_joint_seed
        gh_offset = sca_gh_world - (self._h_mat[:3, :3] @ (gh_raw - self._ij))

        self.vertices = self.transform_mesh(raw_verts, self._ij, self._h_mat) + gh_offset
        self.gh_joint = sca_gh_world.copy()
        self.origin   = sca_gh_world.copy()
        return self
