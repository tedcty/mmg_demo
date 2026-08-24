"""Clavicle — second link in the shoulder kinematic chain.

Connects the SC joint (on Thorax) to the AC joint (on Scapula).
Owns its JCS orientation and the sync logic that rotates the mesh
when the Scapula's FABRIK solver moves the AC joint.
"""
from __future__ import annotations
import numpy as np
from scipy.spatial.transform import Rotation as R
from ptb.util.math.transformation import Cloud
from .base_bone import BoneBase
from .thorax import Thorax


class Clavicle(BoneBase):
    color = "#C080FF"   # right default; overridden for left in __init__

    def __init__(self, side: str):
        super().__init__(side=side)
        self.label = f"{'R' if side == 'right' else 'L'} Clavicle"
        self.color = "#C080FF" if side == "right" else "#FFB0D0"

        self.sc_joint: np.ndarray = np.zeros(3)   # World SC joint (= thorax SC)
        self.ac_joint: np.ndarray = np.zeros(3)   # World AC joint

    # ── Loading ───────────────────────────────────────────────────────────────

    def load(
        self,
        case_arr: np.ndarray,
        all_faces: list,
        maps_dir: str,
    ) -> "Clavicle":
        s = self.side
        prefix = "r" if s == "right" else "l"
        bone_csv = "R_clav.csv" if s == "right" else "L_clav.csv"

        verts, inds, valid_ids = self.filter_bone_indices(case_arr, all_faces, maps_dir, bone_csv)
        self._raw_verts = verts
        self.indices = inds
        self._valid_ids = valid_ids

        self._sc_pt  = self.get_landmark(case_arr, maps_dir, f"cla_{prefix}_sc.csv")
        self._ac_pt  = self.get_landmark(case_arr, maps_dir, f"cla_{prefix}_ac.csv")
        self._sc_raw = self.get_sphere_center(case_arr, maps_dir, f"cla_scj_{prefix}.csv")
        return self

    # ── JCS + assembly ────────────────────────────────────────────────────────

    def assemble(self, thorax: Thorax) -> "Clavicle":
        """Build clavicle JCS and align SC joint centres."""
        yt = thorax.yt
        ij = thorax.ij_pt
        jcs = thorax.jcs_matrix

        sc_pt, ac_pt = self._sc_pt, self._ac_pt

        if self.side == "right":
            zc = (ac_pt - sc_pt) / np.linalg.norm(ac_pt - sc_pt)
        else:
            zc = (sc_pt - ac_pt) / np.linalg.norm(sc_pt - ac_pt)

        xc_raw = np.cross(yt, zc)
        xc     = xc_raw / np.linalg.norm(xc_raw)
        yc     = np.cross(zc, xc)
        c_mat  = Cloud.transform_between_3x3_points_sets(
            np.array([xc, yc, zc]).T, np.eye(3)
        )

        # SC joint alignment
        sc_world = thorax.sc_r if self.side == "right" else thorax.sc_l
        sc_offset = sc_world - (c_mat[:3, :3] @ (self._sc_raw - ij))

        self.vertices = self.transform_mesh(self._raw_verts, ij, c_mat) + sc_offset
        self.sc_joint = sc_world.copy()
        self.ac_joint = self.transform_mesh([ac_pt], ij, c_mat)[0] + sc_offset
        self.origin   = self.sc_joint.copy()

        # Store for sync_to_scapula and replay
        self._c_mat     = c_mat
        self._ij        = ij
        return self

    def replay(self, case_arr: np.ndarray, maps_dir: str, thorax: Thorax) -> "Clavicle":
        """Recompute the visible mesh AND sc_joint/ac_joint from a new
        case_arr, reusing this instance's already-solved _ij/_c_mat (the
        clavicle's ORIENTATION stays frozen) but re-deriving sc_joint/
        ac_joint from fresh landmarks — via `thorax`'s (already replayed)
        sc_r/sc_l — so they track the new anatomy instead of leaving a gap
        at either end. See Thorax's version of this method for context."""
        prefix = "r" if self.side == "right" else "l"
        raw_verts = case_arr[self._valid_ids]

        ac_pt  = self.get_landmark(case_arr, maps_dir, f"cla_{prefix}_ac.csv")
        sc_raw = self.get_sphere_center(case_arr, maps_dir, f"cla_scj_{prefix}.csv")
        sc_world = thorax.sc_r if self.side == "right" else thorax.sc_l
        sc_offset = sc_world - (self._c_mat[:3, :3] @ (sc_raw - self._ij))

        self.vertices = self.transform_mesh(raw_verts, self._ij, self._c_mat) + sc_offset
        self.sc_joint = sc_world.copy()
        self.ac_joint = self.transform_mesh([ac_pt], self._ij, self._c_mat)[0] + sc_offset
        self.origin   = self.sc_joint.copy()
        return self

    def sync_to_scapula(self, new_ac: np.ndarray) -> None:
        """Rotate clavicle mesh around SC joint to follow new AC position."""
        v_old = self.ac_joint - self.sc_joint
        v_new = new_ac         - self.sc_joint
        if np.linalg.norm(v_new) < 1e-6 or np.linalg.norm(v_old) < 1e-6:
            return
        rot_clav, _ = R.align_vectors([v_new], [v_old])
        self.vertices = rot_clav.apply(self.vertices - self.sc_joint) + self.sc_joint
        self.ac_joint = new_ac.copy()
