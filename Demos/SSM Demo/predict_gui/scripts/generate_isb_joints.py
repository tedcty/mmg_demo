import os
import json
import numpy as np
import pandas as pd
import vtk
from ptb.util.data import VTKMeshUtl
from ptb.util.math.transformation import Cloud
from scapulothoracic_constraint import solve_hierarchical_shoulder

def sphere_fit(points):
    p_mean = np.nanmean(points, axis=0)
    n = points.shape[0]
    a = np.eye(3)
    for i in range(0, 3):
        a[i, 0] = np.nansum([(points[x, i] * (points[x, 0] - p_mean[0])) / n for x in range(0, n)])
        a[i, 1] = np.nansum([(points[x, i] * (points[x, 1] - p_mean[1])) / n for x in range(0, n)])
        a[i, 2] = np.nansum([(points[x, i] * (points[x, 2] - p_mean[2])) / n for x in range(0, n)])
    a = 2 * a
    b = np.zeros((3, 1))
    sum_axis = np.sum(points**2, axis=1)
    b[0, 0] = np.sum(sum_axis * (points[:, 0] - p_mean[0]) / n)
    b[1, 0] = np.sum(sum_axis * (points[:, 1] - p_mean[1]) / n)
    b[2, 0] = np.sum(sum_axis * (points[:, 2] - p_mean[2]) / n)
    c = np.linalg.solve(np.dot(a.T, a), np.dot(a.T, b))
    return np.squeeze(c)

def _get_landmark(case_verts, maps_dir, filename):
    fpath = os.path.join(maps_dir, filename)
    if not os.path.exists(fpath):
        print(f"Warning: Landmark file {filename} not found.")
        return np.array([0., 0., 0.])
    idm = pd.read_csv(fpath)['idm'].to_list()
    return np.mean(case_verts[idm], axis=0)

def _get_sphere_center(case_verts, maps_dir, filename):
    fpath = os.path.join(maps_dir, filename)
    if not os.path.exists(fpath):
        print(f"Warning: Sphere map {filename} not found.")
        return np.array([0., 0., 0.])
    idm = pd.read_csv(fpath)['idm'].to_list()
    return sphere_fit(case_verts[idm])

def extract_faces(polydata):
    faces = []
    polys = polydata.GetPolys()
    polys.InitTraversal()
    idList = vtk.vtkIdList()
    while polys.GetNextCell(idList):
        if idList.GetNumberOfIds() == 3:
            faces.append((int(idList.GetId(0)), int(idList.GetId(1)), int(idList.GetId(2))))
    return faces

def filter_bone_indices(all_verts, all_faces, maps_dir, filename):
    fpath = os.path.join(maps_dir, filename)
    if not os.path.exists(fpath):
        return None, None
    idm_set = set(pd.read_csv(fpath)['idm'].to_list())
    valid_old_ids = sorted(list(idm_set))
    old_to_new = {old_id: new_id for new_id, old_id in enumerate(valid_old_ids)}
    bone_verts = [all_verts[old_id].tolist() for old_id in valid_old_ids]
    bone_faces = []
    for f in all_faces:
        if f[0] in idm_set and f[1] in idm_set and f[2] in idm_set:
            bone_faces.extend([old_to_new[f[0]], old_to_new[f[1]], old_to_new[f[2]]])
    return bone_verts, bone_faces

def process_and_export(target_ply=None):
    print("Starting Global ISB Assembly Pipeline (Recursive JCS)...")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    res_dir = os.path.join(script_dir, '..', 'Resources')
    maps_dir = os.path.join(res_dir, "landmarks", "maps to mean")
    
    if target_ply is None:
        target_ply = os.path.join(res_dir, "SSM_shape_model_103", "CombinedSSM_103_PCA_mean.ply")
        
    export_path = os.path.join(script_dir, '..', 'TauriGUI', 'public', 'bones.json')

    if not os.path.exists(target_ply):
        print(f"Error: Target PLY not found at {target_ply}")
        return

    reader = vtk.vtkPLYReader()
    reader.SetFileName(target_ply)
    reader.Update()
    polydata = reader.GetOutput()
    
    current_case = VTKMeshUtl.extract_points(polydata)
    all_faces = extract_faces(polydata)
    current_case_arr = np.array(current_case)

    markers = []
    axes = []
    
    # --- 1. THORAX (Root) ---
    ij_pt = _get_landmark(current_case_arr, maps_dir, "tho_ij.csv")
    px_pt = _get_landmark(current_case_arr, maps_dir, "tho_px.csv")
    c7_pt = 0.5 * (_get_landmark(current_case_arr, maps_dir, "tho_c7_r.csv") + _get_landmark(current_case_arr, maps_dir, "tho_c7_l.csv"))
    t8_pt = 0.5 * (_get_landmark(current_case_arr, maps_dir, "tho_t8_r.csv") + _get_landmark(current_case_arr, maps_dir, "tho_t8_l.csv"))
    
    mid_px_t8 = 0.5 * (px_pt + t8_pt)
    mid_ij_c7 = 0.5 * (ij_pt + c7_pt)
    
    yt_raw = mid_ij_c7 - mid_px_t8
    yt = yt_raw / np.linalg.norm(yt_raw)
    yz1_raw = mid_px_t8 - ij_pt
    yz2_raw = c7_pt - ij_pt
    zt_raw = np.cross(yz2_raw, yz1_raw)
    zt = zt_raw / np.linalg.norm(zt_raw)
    xt = np.cross(yt, zt)
    
    t_source = np.array([xt, yt, zt]).T
    t_target = np.eye(3)
    t_t_mat = Cloud.transform_between_3x3_points_sets(t_source, t_target)
    
    def transform_mesh(verts, trans_vec, rot_mat):
        v = np.array(verts) - trans_vec
        homo = np.hstack((v, np.ones((v.shape[0], 1))))
        return (rot_mat @ homo.T).T[:, :3]

    tho_verts, tho_inds = filter_bone_indices(current_case_arr, all_faces, maps_dir, "Tho.csv")
    final_thorax = transform_mesh(tho_verts, ij_pt, t_t_mat)
    
    # Global vectors for child bones to align to
    t_transformed_vectors = (t_t_mat[:3, :3] @ t_source).T 
    
    # --- 2. RIGHT SIDE ASSEMBLY ---
    
    # 2a. Clavicle
    cla_r_verts, cla_r_inds = filter_bone_indices(current_case_arr, all_faces, maps_dir, "R_clav.csv")
    sc_r_pt = _get_landmark(current_case_arr, maps_dir, "cla_r_sc.csv")
    ac_r_pt = _get_landmark(current_case_arr, maps_dir, "cla_r_ac.csv")
    
    zc_raw = ac_r_pt - sc_r_pt
    zc = zc_raw / np.linalg.norm(zc_raw)
    xc_raw = np.cross(yt, zc) # Use raw Thorax Y
    xc = xc_raw / np.linalg.norm(xc_raw)
    yc = np.cross(zc, xc)
    c_source = np.array([xc, yc, zc]).T
    
    # Sphere fits for joint center
    rc_sc_cla = _get_sphere_center(current_case_arr, maps_dir, "cla_scj_r.csv")
    rc_sc_tho = _get_sphere_center(current_case_arr, maps_dir, "tho_scj_r.csv")
    
    # Transform Clavicle to Global (Thorax) frame
    # 1. Align orientation to Thorax (global)
    c_t_mat = Cloud.transform_between_3x3_points_sets(c_source, t_target)
    # 2. Align SC joint centers
    sc_offset = (t_t_mat[:3, :3] @ (rc_sc_tho - ij_pt)) - (c_t_mat[:3, :3] @ (rc_sc_cla - ij_pt))
    
    final_clav_r = transform_mesh(cla_r_verts, ij_pt, c_t_mat) + sc_offset
    c_ac_r_glob = transform_mesh([ac_r_pt], ij_pt, c_t_mat)[0] + sc_offset
    
    # 2b. Scapula
    sca_r_verts, sca_r_inds = filter_bone_indices(current_case_arr, all_faces, maps_dir, "R_scap.csv")
    aa_pt = _get_landmark(current_case_arr, maps_dir, "sca_r_aa.csv")
    ai_pt = _get_landmark(current_case_arr, maps_dir, "sca_r_ai.csv")
    ts_pt = _get_landmark(current_case_arr, maps_dir, "sca_r_ts.csv")
    
    zs_raw = aa_pt - ts_pt
    zs = zs_raw / np.linalg.norm(zs_raw)
    xs_raw = np.cross(ai_pt - aa_pt, ts_pt - aa_pt)
    xs = xs_raw / np.linalg.norm(xs_raw)
    ys = np.cross(zs, xs)
    s_source = np.array([xs, ys, zs]).T
    
    # AC joint on scapula side
    sca_ac_r_pt = _get_landmark(current_case_arr, maps_dir, "sca_r_aa.csv") # AA is often AC joint
    
    s_t_mat = Cloud.transform_between_3x3_points_sets(s_source, t_target)
    # Align Scapula AC to Clavicle AC
    ac_offset = c_ac_r_glob - (s_t_mat[:3, :3] @ (sca_ac_r_pt - ij_pt))
    
    final_scap_r = transform_mesh(sca_r_verts, ij_pt, s_t_mat) + ac_offset
    
    # 2c. Humerus
    hum_r_verts, hum_r_inds = filter_bone_indices(current_case_arr, all_faces, maps_dir, "R_hum.csv")
    gh_r_pt = _get_sphere_center(current_case_arr, maps_dir, "hum_ghj_r.csv")
    el_r_pt = _get_landmark(current_case_arr, maps_dir, "hum_r_el.csv")
    em_r_pt = _get_landmark(current_case_arr, maps_dir, "hum_r_em.csv")
    
    mid_ep = 0.5 * (el_r_pt + em_r_pt)
    yh_raw = gh_r_pt - mid_ep
    yh = yh_raw / np.linalg.norm(yh_raw)
    xh_raw = np.cross(el_r_pt - gh_r_pt, em_r_pt - gh_r_pt)
    xh = xh_raw / np.linalg.norm(xh_raw)
    zh = np.cross(xh, yh)
    h_source = np.array([xh, yh, zh]).T
    
    # GH joint on scapula side
    sca_gh_r_pt = _get_sphere_center(current_case_arr, maps_dir, "scap_ghj_r.csv")
    sca_gh_r_glob = (s_t_mat[:3, :3] @ (sca_gh_r_pt - ij_pt)) + ac_offset
    
    h_t_mat = Cloud.transform_between_3x3_points_sets(h_source, t_target)
    # Align Humerus GH to Scapula GH
    gh_offset = sca_gh_r_glob - (h_t_mat[:3, :3] @ (gh_r_pt - ij_pt))
    
    final_hum_r = transform_mesh(hum_r_verts, ij_pt, h_t_mat) + gh_offset
    
    # --- 3. LEFT SIDE ASSEMBLY ---
    
    # 3a. Clavicle
    cla_l_verts, cla_l_inds = filter_bone_indices(current_case_arr, all_faces, maps_dir, "L_clav.csv")
    sc_l_pt = _get_landmark(current_case_arr, maps_dir, "cla_l_sc.csv")
    ac_l_pt = _get_landmark(current_case_arr, maps_dir, "cla_l_ac.csv")
    
    zc_raw_l = sc_l_pt - ac_l_pt
    zc_l = zc_raw_l / np.linalg.norm(zc_raw_l)
    xc_raw_l = np.cross(yt, zc_l)
    xc_l = xc_raw_l / np.linalg.norm(xc_raw_l)
    yc_l = np.cross(zc_l, xc_l)
    c_source_l = np.array([xc_l, yc_l, zc_l]).T
    
    rc_sc_cla_l = _get_sphere_center(current_case_arr, maps_dir, "cla_scj_l.csv")
    rc_sc_tho_l = _get_sphere_center(current_case_arr, maps_dir, "tho_scj_l.csv")
    
    cl_t_mat = Cloud.transform_between_3x3_points_sets(c_source_l, t_target)
    sc_offset_l = (t_t_mat[:3, :3] @ (rc_sc_tho_l - ij_pt)) - (cl_t_mat[:3, :3] @ (rc_sc_cla_l - ij_pt))
    
    final_clav_l = transform_mesh(cla_l_verts, ij_pt, cl_t_mat) + sc_offset_l
    c_ac_l_glob = transform_mesh([ac_l_pt], ij_pt, cl_t_mat)[0] + sc_offset_l
    
    # 3b. Scapula
    sca_l_verts, sca_l_inds = filter_bone_indices(current_case_arr, all_faces, maps_dir, "L_scap.csv")
    aa_l_pt = _get_landmark(current_case_arr, maps_dir, "sca_l_aa.csv")
    ts_l_pt = _get_landmark(current_case_arr, maps_dir, "sca_l_ts.csv")
    ai_l_pt = _get_landmark(current_case_arr, maps_dir, "sca_l_ai.csv")
    
    zs_raw_l = ts_l_pt - aa_l_pt
    zs_l = zs_raw_l / np.linalg.norm(zs_raw_l)
    xs_raw_l = np.cross(ts_l_pt - aa_l_pt, ai_l_pt - aa_l_pt)
    xs_l = xs_raw_l / np.linalg.norm(xs_raw_l)
    ys_l = np.cross(zs_l, xs_l)
    s_source_l = np.array([xs_l, ys_l, zs_l]).T
    
    sca_ac_l_pt = _get_landmark(current_case_arr, maps_dir, "sca_l_aa.csv")
    
    sl_t_mat = Cloud.transform_between_3x3_points_sets(s_source_l, t_target)
    ac_offset_l = c_ac_l_glob - (sl_t_mat[:3, :3] @ (sca_ac_l_pt - ij_pt))
    
    final_scap_l = transform_mesh(sca_l_verts, ij_pt, sl_t_mat) + ac_offset_l
    
    # 3c. Humerus
    hum_l_verts, hum_l_inds = filter_bone_indices(current_case_arr, all_faces, maps_dir, "L_hum.csv")
    gh_l_pt = _get_sphere_center(current_case_arr, maps_dir, "hum_ghj_l.csv")
    el_l_pt = _get_landmark(current_case_arr, maps_dir, "hum_l_el.csv")
    em_l_pt = _get_landmark(current_case_arr, maps_dir, "hum_l_em.csv")
    
    mid_ep_l = 0.5 * (el_l_pt + em_l_pt)
    yh_raw_l = gh_l_pt - mid_ep_l
    yh_l = yh_raw_l / np.linalg.norm(yh_raw_l)
    xh_raw_l = np.cross(em_l_pt - gh_l_pt, el_l_pt - gh_l_pt)
    xh_l = xh_raw_l / np.linalg.norm(xh_raw_l)
    zh_l = np.cross(xh_l, yh_l)
    h_source_l = np.array([xh_l, yh_l, zh_l]).T
    
    sca_gh_l_pt = _get_sphere_center(current_case_arr, maps_dir, "scap_ghj_l.csv")
    sca_gh_l_glob = (sl_t_mat[:3, :3] @ (sca_gh_l_pt - ij_pt)) + ac_offset_l
    
    hl_t_mat = Cloud.transform_between_3x3_points_sets(h_source_l, t_target)
    gh_offset_l = sca_gh_l_glob - (hl_t_mat[:3, :3] @ (gh_l_pt - ij_pt))
    
    final_hum_l = transform_mesh(hum_l_verts, ij_pt, hl_t_mat) + gh_offset_l

    # --- 4. EXPORT ---
    tho_sc_r_glob = (t_t_mat[:3, :3] @ (rc_sc_tho - ij_pt))
    tho_sc_l_glob = (t_t_mat[:3, :3] @ (rc_sc_tho_l - ij_pt))
    
    # Prepare markers for visualization
    markers.append({"pos": [0,0,0], "label": "IJ", "color": "yellow"})
    
    # Export JSON Payload
    payload = {
        "center": [0,0,0],
        "spread": 400,
        "bones": [
            {"label": "Thorax", "color": "#90CFF0", "vertices": final_thorax.tolist(), "indices": tho_inds, "origin": [0,0,0]},
            {"label": "R Clavicle", "color": "#C080FF", "vertices": final_clav_r.tolist(), "indices": cla_r_inds, "origin": tho_sc_r_glob.tolist()},
            {"label": "L Clavicle", "color": "#FFB0D0", "vertices": final_clav_l.tolist(), "indices": cla_l_inds, "origin": tho_sc_l_glob.tolist()},
            {"label": "R Scapula", "color": "#FFA040", "vertices": final_scap_r.tolist(), "indices": sca_r_inds, "origin": c_ac_r_glob.tolist()},
            {"label": "L Scapula", "color": "#FFE060", "vertices": final_scap_l.tolist(), "indices": sca_l_inds, "origin": c_ac_l_glob.tolist()},
            {"label": "R Humerus", "color": "#FF6060", "vertices": final_hum_r.tolist(), "indices": hum_r_inds, "origin": sca_gh_r_glob.tolist()},
            {"label": "L Humerus", "color": "#FF6060", "vertices": final_hum_l.tolist(), "indices": hum_l_inds, "origin": sca_gh_l_glob.tolist()}
        ],
        "markers": markers,
        "anatomical_landmarks": {
            "right": {
                "thorax_sc": tho_sc_r_glob.tolist(),
                "thorax_ij": [0,0,0],
                "thorax_px": (t_t_mat[:3, :3] @ (px_pt - ij_pt)).tolist(),
                "thorax_c7": (t_t_mat[:3, :3] @ (c7_pt - ij_pt)).tolist(),
                "thorax_t8": (t_t_mat[:3, :3] @ (t8_pt - ij_pt)).tolist(),
                "clavicle_sc": tho_sc_r_glob.tolist(),
                "clavicle_ac": c_ac_r_glob.tolist(),
                "scapula_ac": c_ac_r_glob.tolist(),
                "scapula_aa": (transform_mesh([aa_pt], ij_pt, s_t_mat)[0] + ac_offset).tolist(),
                "scapula_ts": (transform_mesh([ts_pt], ij_pt, s_t_mat)[0] + ac_offset).tolist(),
                "scapula_ai": (transform_mesh([ai_pt], ij_pt, s_t_mat)[0] + ac_offset).tolist(),
            },
            "left": {
                "thorax_sc": tho_sc_l_glob.tolist(),
                "thorax_ij": [0,0,0],
                "thorax_px": (t_t_mat[:3, :3] @ (px_pt - ij_pt)).tolist(),
                "thorax_c7": (t_t_mat[:3, :3] @ (c7_pt - ij_pt)).tolist(),
                "thorax_t8": (t_t_mat[:3, :3] @ (t8_pt - ij_pt)).tolist(),
                "clavicle_sc": tho_sc_l_glob.tolist(),
                "clavicle_ac": c_ac_l_glob.tolist(),
                "scapula_ac": c_ac_l_glob.tolist(),
                "scapula_aa": (transform_mesh([aa_l_pt], ij_pt, sl_t_mat)[0] + ac_offset_l).tolist(),
                "scapula_ts": (transform_mesh([ts_l_pt], ij_pt, sl_t_mat)[0] + ac_offset_l).tolist(),
                "scapula_ai": (transform_mesh([ai_l_pt], ij_pt, sl_t_mat)[0] + ac_offset_l).tolist(),
            }
        },
        "isb_joints": {
            "right": {
                "sc": tho_sc_r_glob.tolist(),
                "ac": c_ac_r_glob.tolist(),
                "gh": sca_gh_r_glob.tolist(),
                "angles": [0.0, 0.0, 0.0]
            },
            "left": {
                "sc": tho_sc_l_glob.tolist(),
                "ac": c_ac_l_glob.tolist(),
                "gh": sca_gh_l_glob.tolist(),
                "angles": [0.0, 0.0, 0.0]
            }
        }
    }

    with open(export_path, 'w', encoding='utf-8') as f:
        json.dump(payload, f)
        
    print(f"Hierarchical Assembly Complete! File: {export_path}")

if __name__ == "__main__":
    process_and_export()
