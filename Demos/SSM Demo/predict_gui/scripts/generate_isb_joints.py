"""generate_isb_joints.py — Shoulder Assembly Orchestrator.

Loads the PLY model, assembles all bones in kinematic order, and writes
the bones.json file consumed by the Tauri GUI.

Bone-specific logic lives in scripts/bones/:
  Thorax   → thorax.py
  Clavicle → clavicle.py
  Scapula  → scapula.py  (FABRIK on both sides)
  Humerus  → humerus.py
"""
import os
import json
import vtk
import numpy as np
from ptb.util.data import VTKMeshUtl
from bones import Thorax, Clavicle, Scapula, Humerus
from bones.base_bone import BoneBase


def _extract_faces(polydata) -> list:
    faces = []
    polys = polydata.GetPolys()
    polys.InitTraversal()
    idList = vtk.vtkIdList()
    while polys.GetNextCell(idList):
        if idList.GetNumberOfIds() == 3:
            faces.append((int(idList.GetId(0)), int(idList.GetId(1)), int(idList.GetId(2))))
    return faces


def process_and_export(target_ply: str | None = None, fabrik_step: int = 1, export_path: str | None = None):
    print("Starting Global ISB Assembly Pipeline (Recursive JCS)...")

    script_dir  = os.path.dirname(os.path.abspath(__file__))
    res_dir     = os.path.join(script_dir, '..', 'Resources')
    maps_dir    = os.path.join(res_dir, "landmarks", "maps to mean")
    if export_path is None:
        export_path = os.path.join(script_dir, '..', 'TauriGUI', 'public', 'bones.json')

    if target_ply is None:
        target_ply = os.path.join(
            res_dir, "SSM_shape_model_103", "CombinedSSM_103_PCA_mean.ply"
        )

    if not os.path.exists(target_ply):
        print(f"Error: Target PLY not found at {target_ply}")
        return

    # ── Load mesh ─────────────────────────────────────────────────────────────
    reader = vtk.vtkPLYReader()
    reader.SetFileName(target_ply)
    reader.Update()
    polydata     = reader.GetOutput()
    case_arr     = np.array(VTKMeshUtl.extract_points(polydata))
    all_faces    = _extract_faces(polydata)

    # ── 1. Thorax (root) ──────────────────────────────────────────────────────
    thorax = Thorax().load(case_arr, all_faces, maps_dir).build_jcs()

    # ── 2. Right side ─────────────────────────────────────────────────────────
    print("  Assembling RIGHT side...")
    clav_r = Clavicle("right").load(case_arr, all_faces, maps_dir).assemble(thorax)

    print("  FABRIK: Optimizing Right Scapula Alignment...")
    scap_r = (
        Scapula("right")
        .load(case_arr, all_faces, maps_dir, res_dir)
        .assemble_fabrik(thorax, clav_r, fabrik_step)
    )
    clav_r.sync_to_scapula(scap_r.ac_joint)
    hum_r  = Humerus("right").load(case_arr, all_faces, maps_dir).assemble(thorax, scap_r)

    # ── 3. Left side ──────────────────────────────────────────────────────────
    print("  Assembling LEFT side...")
    clav_l = Clavicle("left").load(case_arr, all_faces, maps_dir).assemble(thorax)

    print("  FABRIK: Optimizing Left Scapula Alignment...")
    scap_l = (
        Scapula("left")
        .load(case_arr, all_faces, maps_dir, res_dir)
        .assemble_fabrik(thorax, clav_l, fabrik_step)
    )
    clav_l.sync_to_scapula(scap_l.ac_joint)
    hum_l  = Humerus("left").load(case_arr, all_faces, maps_dir).assemble(thorax, scap_l)

    # ── 4. Diagnostics markers ────────────────────────────────────────────────
    print("Calculating Scapulothoracic Projection Markers...")
    markers = [{"pos": [0, 0, 0], "label": "IJ", "color": "yellow"}]
    
    # Right Scapula landmarks
    markers.append({"pos": scap_r.aa.tolist(), "label": "R_AA", "color": "#FF4444"})
    markers.append({"pos": scap_r.ts.tolist(), "label": "R_TS", "color": "#44FF44"})
    markers.append({"pos": scap_r.ai.tolist(), "label": "R_AI", "color": "#4444FF"})
    markers.append({"pos": scap_r.cp.tolist(), "label": "R_CP", "color": "#FF88FF"})

    proj_r = thorax.project_scapula(scap_r.aa, scap_r.ts, scap_r.ai, "right")
    proj_l = thorax.project_scapula(scap_l.aa, scap_l.ts, scap_l.ai, "left")
    if proj_r is not None:
        markers.append({"pos": proj_r.tolist(), "label": "R_Proj", "color": "cyan"})
    if proj_l is not None:
        markers.append({"pos": proj_l.tolist(), "label": "L_Proj", "color": "cyan"})

    # ── 5. Subscapularis meshes ───────────────────────────────────────────────
    def _subscap_dict(scap: Scapula) -> dict:
        side_label = "R" if scap.side == "right" else "L"
        verts = scap.subscapularis.tolist() if scap.subscapularis is not None else []
        return {
            "label": f"{side_label} Subscapularis",
            "color": "#FF4444",
            "vertices": verts,
            "indices": [],
            "origin": scap.ac_joint.tolist(),
        }

    # ── 6. Build payload ──────────────────────────────────────────────────────
    tho_lm = thorax.landmark_globals()
    payload = {
        "center": [0, 0, 0],
        "spread": 400,
        "bones": [
            thorax.to_dict(),
            clav_r.to_dict(),
            clav_l.to_dict(),
            scap_r.to_dict(),
            scap_l.to_dict(),
            _subscap_dict(scap_r),
            _subscap_dict(scap_l),
            hum_r.to_dict(),
            hum_l.to_dict(),
        ],
        "markers": markers,
        "scapular_planes": {
            "right": scap_r.scapular_plane_dict(),
            "left":  scap_l.scapular_plane_dict(),
        },
        "anatomical_landmarks": {
            "right": {
                **tho_lm,
                "thorax_sc":  thorax.sc_r.tolist(),
                "clavicle_sc": thorax.sc_r.tolist(),
                "clavicle_ac": clav_r.ac_joint.tolist(),
                **scap_r.landmark_globals(),
            },
            "left": {
                **tho_lm,
                "thorax_sc":  thorax.sc_l.tolist(),
                "clavicle_sc": thorax.sc_l.tolist(),
                "clavicle_ac": clav_l.ac_joint.tolist(),
                **scap_l.landmark_globals(),
            },
        },
        "isb_joints": {
            "right": {
                "sc": thorax.sc_r.tolist(),
                "ac": clav_r.ac_joint.tolist(),
                "gh": hum_r.gh_joint.tolist(),
                "angles": [0, 0, 0, 0, 0, 0],
            },
            "left": {
                "sc": thorax.sc_l.tolist(),
                "ac": clav_l.ac_joint.tolist(),
                "gh": hum_l.gh_joint.tolist(),
                "angles": [0, 0, 0, 0, 0, 0],
            },
        },
    }

    with open(export_path, 'w', encoding='utf-8') as f:
        json.dump(payload, f)
    print(f"Hierarchical Assembly Complete! File: {export_path}")


if __name__ == "__main__":
    process_and_export()
