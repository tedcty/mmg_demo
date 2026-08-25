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
import copy
import orjson
import vtk
import numpy as np
from scipy.spatial.transform import Rotation as R
from ptb.util.data import VTKMeshUtl
from bones import Thorax, Clavicle, Scapula, Humerus
from bones.base_bone import BoneBase


def compute_joint_angles(sync_rot: R, solved_rot: R, side: str) -> dict:
    """Decomposes a side's clavicle sync rotation (= its SC joint angle
    directly) and scapula FABRIK-correction rotation into SC/AC/GH joint
    angles, in App.vue's Kinematics-tab convention (its qSC/qAC/qGH
    construction: 'YXZ' Euler order, angles named per axis, sign flipped by
    a per-side jointSide so a positive slider means the same anatomical
    motion on both shoulders). AC is not an independent quantity here — it's
    derived as clavicle_world^-1 @ scapula_world, i.e. SC^-1 @ GH^-1's
    inverse — so it's a *consequence* of SC and GH, not separately solvable.
    Used both to report the current angles (server.py's
    /api/pc_reference_angles) and by _mirror_left_joint_rotation below."""
    js = 1 if side == 'right' else -1

    e = sync_rot.as_euler('YXZ', degrees=True)
    sc = {'abduction': e[0] / js, 'elevation': -e[1] / js, 'upward_rot': -e[2] / js}

    gh_rot = solved_rot.inv()
    e = gh_rot.as_euler('YXZ', degrees=True)
    gh = {'internal': -e[0] / js, 'abduction': e[1] / js, 'flexion': e[2]}

    ac_rot = sync_rot.inv() * solved_rot
    e = ac_rot.as_euler('YXZ', degrees=True)
    ac = {'internal': -e[0] / js, 'upward': -e[1] / js, 'posterior': -e[2] / js}

    return {'sc': sc, 'ac': ac, 'gh': gh}


def _mirror_left_joint_rotation(scap_r_solved_rot: R, clav_r_sync_rot: R) -> tuple[R, R]:
    """Given the RIGHT side's already-solved scapula FABRIK-correction
    rotation and clavicle sync-to-scapula rotation, returns the LEFT-side
    equivalents that produce numerically identical SC and GH joint *angles*
    (see compute_joint_angles) once decomposed for the opposite side's sign
    convention. AC then improves (since it's a consequence of SC+GH, not an
    independent parameter — see compute_joint_angles' docstring) but isn't
    forced to match exactly.

    Joint *positions* are untouched by this — only orientation. Right's
    rotations are independently solved from real (slightly asymmetric)
    anatomy each side; this makes left's orientation match right's rather
    than its own independent solve, which was found to differ by a few
    degrees (up to ~4° on some axes) even at the mean shape.
    """
    angles = compute_joint_angles(clav_r_sync_rot, scap_r_solved_rot, 'right')
    sc, gh = angles['sc'], angles['gh']

    # Recompose for LEFT (jointSide = -1) using RIGHT's SAME angle values.
    js = -1
    sync_rot_l = R.from_euler(
        'YXZ', [js * sc['abduction'], -js * sc['elevation'], -js * sc['upward_rot']], degrees=True)
    qgh_l = R.from_euler(
        'YXZ', [-js * gh['internal'], js * gh['abduction'], gh['flexion']], degrees=True)
    solved_rot_l = qgh_l.inv()
    return solved_rot_l, sync_rot_l


def _extract_faces(polydata) -> list:
    faces = []
    polys = polydata.GetPolys()
    polys.InitTraversal()
    idList = vtk.vtkIdList()
    while polys.GetNextCell(idList):
        if idList.GetNumberOfIds() == 3:
            faces.append((int(idList.GetId(0)), int(idList.GetId(1)), int(idList.GetId(2))))
    return faces


def process_and_export(target_ply: str | None = None, fabrik_step: int = 1,
                        export_path: str | None = None, corrections: dict | None = None):
    """
    `corrections`, when given, is {'right': (tilt,roll,push,slide), 'left': ...}
    from a previous full solve (see Scapula.assemble_fabrik / solve_alignment's
    `correction` param) — reused instead of re-running each side's ~90s
    Step-4 search.

    Returns a dict — {'corrections', 'bones', 'payload', 'maps_dir'} — that
    a caller can both read the corrections from AND pass straight into
    replay_shape() later for a much cheaper shape-only re-render of a
    different mesh sharing this same joint pose.
    """
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
        .assemble_fabrik(thorax, clav_r, fabrik_step, correction=(corrections or {}).get('right'))
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
        .assemble_fabrik(thorax, clav_l, fabrik_step, correction=(corrections or {}).get('left'))
    )

    # Force left's joint *angles* (GH/SC, and consequently AC) to match
    # right's rather than left's own independent solve — see
    # _mirror_left_joint_rotation's docstring. Joint *positions* (ac_seed/
    # ac_sol/sc_joint, all derived from this side's own real landmarks)
    # are untouched.
    solved_rot_l, sync_rot_l = _mirror_left_joint_rotation(scap_r.solved_rot, clav_r._sync_rot)
    scap_l.apply_solved_rotation(solved_rot_l)
    clav_l.sync_to_scapula(scap_l.ac_joint, rotation_override=sync_rot_l)
    hum_l  = Humerus("left").load(case_arr, all_faces, maps_dir).assemble(thorax, scap_l)

    payload = _build_payload(thorax, clav_r, clav_l, scap_r, scap_l, hum_r, hum_l)

    with open(export_path, 'wb') as f:
        f.write(orjson.dumps(payload))
    print(f"Hierarchical Assembly Complete! File: {export_path}")

    return {
        'corrections': {'right': scap_r.fabrik_correction, 'left': scap_l.fabrik_correction},
        'bones': {
            'thorax': thorax, 'clav_r': clav_r, 'clav_l': clav_l,
            'scap_r': scap_r, 'scap_l': scap_l, 'hum_r': hum_r, 'hum_l': hum_l,
        },
        'payload': payload,
        'maps_dir': maps_dir,
    }


def _build_payload(thorax: Thorax, clav_r: Clavicle, clav_l: Clavicle,
                    scap_r: Scapula, scap_l: Scapula, hum_r: Humerus, hum_l: Humerus) -> dict:
    """Builds the bones.json payload from fully-assembled bones — shared by
    process_and_export (fresh assembly) and replay_shape (frozen-pose
    replay), so both stay byte-for-byte consistent in how they turn bone
    objects into JSON."""
    print("Calculating Scapulothoracic Projection Markers...")
    markers = [{"pos": [0, 0, 0], "label": "IJ", "color": "yellow"}]

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

    tho_lm = thorax.landmark_globals()
    return {
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


def replay_shape(reference: dict, target_ply: str, export_path: str) -> dict:
    """Rebuild bones.json from target_ply's mesh shape while reusing every
    bone's already-solved ORIENTATION from a prior process_and_export()
    call — no landmark-derived rotation matrix or FABRIK search re-runs.
    Joint *positions* (sc/ac/gh) and landmarks (aa/ts/ai/cp/subscapularis)
    ARE recomputed fresh each call, from this new mesh's own landmarks
    pushed through those frozen rotations — so a PC weight that changes a
    bone's length correctly moves that bone's joints with it, instead of
    leaving a gap where a resized bone no longer quite reaches a joint
    frozen at the mean shape's position. This is still dramatically cheaper
    than a full solve, since deriving a rotation from 3 landmark points and
    re-running FABRIK's Step-4 search are the only two things skipped — but
    it is a shape-only preview, not a substitute for a real solve.

    `reference` is the dict returned by process_and_export(). Returns the
    built payload dict directly (as well as writing it to export_path) so
    an in-process caller (unlike predict_headless.py's subprocess callers,
    which have no choice but to go through the file) doesn't have to pay a
    second full JSON parse just to read back what it already has.
    """
    # Only the raw vertex positions are needed — face topology and the
    # per-bone vertex-id mapping are frozen (cached on each bone as
    # _valid_ids/indices at reference time), so this skips both the face
    # traversal (_extract_faces) and the per-bone face-filtering that a
    # fresh .load() call would otherwise redo on every single PC update.
    reader = vtk.vtkPLYReader()
    reader.SetFileName(target_ply)
    reader.Update()
    case_arr = np.array(VTKMeshUtl.extract_points(reader.GetOutput()))

    maps_dir = reference['maps_dir']
    # Shallow-copy so concurrent requests each replay onto their own bone
    # objects rather than racing to mutate the shared cached reference ones.
    # A shallow copy is enough — replay() only ever *reassigns* attributes
    # to brand-new values, never mutates an old array in place, so the
    # original reference bone's data is untouched either way.
    bones = {name: copy.copy(bone) for name, bone in reference['bones'].items()}

    thorax = bones['thorax']
    thorax.replay(case_arr, maps_dir)

    clav_r, scap_r, hum_r = bones['clav_r'], bones['scap_r'], bones['hum_r']
    clav_r.replay(case_arr, maps_dir, thorax)
    scap_r.replay(case_arr, maps_dir, clav_r)
    clav_r.sync_to_scapula(scap_r.ac_joint)
    hum_r.replay(case_arr, maps_dir, scap_r)

    clav_l, scap_l, hum_l = bones['clav_l'], bones['scap_l'], bones['hum_l']
    clav_l.replay(case_arr, maps_dir, thorax)
    scap_l.replay(case_arr, maps_dir, clav_l)
    clav_l.sync_to_scapula(scap_l.ac_joint)
    hum_l.replay(case_arr, maps_dir, scap_l)

    payload = _build_payload(thorax, clav_r, clav_l, scap_r, scap_l, hum_r, hum_l)

    with open(export_path, 'wb') as f:
        f.write(orjson.dumps(payload))
    print(f"Shape replay complete! File: {export_path}")

    return payload


if __name__ == "__main__":
    # fabrik_step defaults to 1 (coarse centroid placement only — Steps 2-4,
    # which do the actual thorax-surface tangency fit and tilt/roll/push/
    # slide correction, never run) unless explicitly requested. server.py's
    # _get_pc_model() always passes fabrik_step=4; this CLI entry point
    # must match it, or a standalone run silently produces a scapula that
    # isn't properly seated against the thorax (previously the case here —
    # this is what caused a visibly "floating" scapula after a manual
    # regeneration).
    #
    # Also auto-load the cached FABRIK correction if one exists, same as
    # _get_pc_model() does — otherwise this CLI invocation re-solves Step 4
    # fresh, and since that Nelder-Mead search isn't fully deterministic,
    # running this script standalone can silently produce a *different*
    # (though individually valid/symmetric) pose than whatever the live
    # server is using for predictions. Reusing the cached correction keeps
    # them bit-for-bit consistent instead.
    _corrections_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), '..', 'TauriGUI', 'public', 'pc_corrections.json')
    _corrections = None
    if os.path.exists(_corrections_path):
        with open(_corrections_path) as _f:
            _cached = json.load(_f)
        _corrections = {'right': tuple(_cached['right']), 'left': tuple(_cached['left'])}
    process_and_export(fabrik_step=4, corrections=_corrections)
