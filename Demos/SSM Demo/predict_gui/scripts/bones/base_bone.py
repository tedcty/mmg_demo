"""BoneBase — abstract base for all shoulder bones.

Provides shared mesh loading, landmark extraction, and mesh transformation
utilities used by all concrete bone subclasses.
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
import vtk
from ptb.util.data import VTKMeshUtl
from ptb.util.math.transformation import Cloud


def sphere_fit(points: np.ndarray) -> np.ndarray:
    """Fit a sphere to ``points`` and return its centre."""
    p_mean = np.nanmean(points, axis=0)
    n = points.shape[0]
    a = np.eye(3)
    for i in range(3):
        a[i, 0] = np.nansum([(points[x, i] * (points[x, 0] - p_mean[0])) / n for x in range(n)])
        a[i, 1] = np.nansum([(points[x, i] * (points[x, 1] - p_mean[1])) / n for x in range(n)])
        a[i, 2] = np.nansum([(points[x, i] * (points[x, 2] - p_mean[2])) / n for x in range(n)])
    a = 2 * a
    b = np.zeros((3, 1))
    s = np.sum(points ** 2, axis=1)
    b[0, 0] = np.sum(s * (points[:, 0] - p_mean[0]) / n)
    b[1, 0] = np.sum(s * (points[:, 1] - p_mean[1]) / n)
    b[2, 0] = np.sum(s * (points[:, 2] - p_mean[2]) / n)
    c = np.linalg.solve(np.dot(a.T, a), np.dot(a.T, b))
    return np.squeeze(c)


class BoneBase:
    """Abstract base class for a single bone in the shoulder assembly.

    Subclasses implement ``load()``, ``build_jcs()``, and ``assemble()``.
    All shared data-loading and mesh transformation utilities live here.
    """

    label: str = "BoneBase"
    color: str = "#FFFFFF"

    def __init__(self, side: str | None = None):
        self.side = side                   # 'right', 'left', or None (Thorax)
        self.vertices: np.ndarray | None = None   # Final world-space vertices (N,3)
        self.indices: list[int] = []               # Flattened triangle indices
        self.origin: np.ndarray = np.zeros(3)      # Joint centre (rotation pivot)

    # ── Static data-loading utilities ─────────────────────────────────────────

    @staticmethod
    def get_landmark(case_verts: np.ndarray, maps_dir: str, filename: str) -> np.ndarray:
        """Return the mean position of vertices listed in *filename* CSV."""
        fpath = os.path.join(maps_dir, filename)
        if not os.path.exists(fpath):
            print(f"  Warning: Landmark file {filename} not found.")
            return np.zeros(3)
        idm = pd.read_csv(fpath)['idm'].to_list()
        return np.mean(case_verts[idm], axis=0)

    @staticmethod
    def get_sphere_center(case_verts: np.ndarray, maps_dir: str, filename: str) -> np.ndarray:
        """Fit a sphere to the vertices listed in *filename* CSV and return its centre."""
        fpath = os.path.join(maps_dir, filename)
        if not os.path.exists(fpath):
            print(f"  Warning: Sphere map {filename} not found.")
            return np.zeros(3)
        idm = pd.read_csv(fpath)['idm'].to_list()
        return sphere_fit(case_verts[idm])

    @staticmethod
    def filter_bone_indices(
        all_verts: np.ndarray,
        all_faces: list[tuple[int, int, int]],
        maps_dir: str,
        filename: str,
    ) -> tuple[list, list, list]:
        """Extract per-bone vertices, remapped triangle indices, and the
        original full-mesh vertex ids used (`valid_ids`) — the latter is
        purely a function of the landmark-mapping CSV, never of `all_verts`/
        `all_faces`, so callers can cache it and later slice a *different*
        mesh's vertices by the same ids without re-scanning faces at all
        (see replay_vertices on each bone subclass)."""
        fpath = os.path.join(maps_dir, filename)
        if not os.path.exists(fpath):
            return [], [], []
        idm_set = set(pd.read_csv(fpath)['idm'].to_list())
        valid_ids = sorted(idm_set)
        old_to_new = {v: i for i, v in enumerate(valid_ids)}
        bone_verts = [all_verts[i].tolist() for i in valid_ids]
        bone_faces: list[int] = []
        for f in all_faces:
            if f[0] in idm_set and f[1] in idm_set and f[2] in idm_set:
                bone_faces.extend([old_to_new[f[0]], old_to_new[f[1]], old_to_new[f[2]]])
        return bone_verts, bone_faces, valid_ids

    @staticmethod
    def load_muscle_cloud(case_verts: np.ndarray, fpath: str) -> list | None:
        """Load a muscle attachment point cloud from a CSV of vertex indices."""
        if not os.path.exists(fpath):
            return None
        idm = pd.read_csv(fpath)['idm'].to_list()
        return [case_verts[idx].tolist() for idx in idm]

    # ── Static mesh transformation ─────────────────────────────────────────────

    @staticmethod
    def transform_mesh(
        verts,
        trans_vec: np.ndarray,
        rot_mat: np.ndarray,
    ) -> np.ndarray:
        """Translate then rotate *verts* by *trans_vec* and 4×4 *rot_mat*.

        ``verts`` may be a list of lists or an (N,3) ndarray.
        Returns an (N,3) ndarray.
        """
        v = np.array(verts, dtype=float) - trans_vec
        homo = np.hstack((v, np.ones((v.shape[0], 1))))
        return (rot_mat @ homo.T).T[:, :3]

    # ── JSON serialisation ─────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        """Return the bone as a dict suitable for ``bones.json``."""
        return {
            "label": self.label,
            "color": self.color,
            "vertices": self.vertices.tolist() if self.vertices is not None else [],
            "indices": self.indices,
            "origin": self.origin.tolist(),
        }
