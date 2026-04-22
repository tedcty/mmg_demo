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

def process_and_export():
    print("Starting Global ISB Assembly Pipeline...")
    base_dir = r"W:\R_SSM_UpperLimb_Prediction\workflow2"
    maps_dir = os.path.join(base_dir, "maps to mean")
    mean_ply = os.path.join(base_dir, "shape_model", "combinedSSM_mean.ply")
    export_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'TauriGUI', 'public', 'bones.json')

    reader = vtk.vtkPLYReader()
    reader.SetFileName(mean_ply)
    reader.Update()
    polydata = reader.GetOutput()
    
    current_case = VTKMeshUtl.extract_points(polydata)
    all_faces = extract_faces(polydata)

    markers = []
    axes = []
    t_target = np.array([[1,0,0], [0,1,0], [0,0,1]])

    # 1. THORAX (Root)
    ij_pt = _get_landmark(current_case, maps_dir, "tho_ij.csv")
    px_pt = _get_landmark(current_case, maps_dir, "tho_px.csv")
    c7_pt = 0.5 * (_get_landmark(current_case, maps_dir, "tho_c7_r.csv") + _get_landmark(current_case, maps_dir, "tho_c7_l.csv"))
    t8_pt = 0.5 * (_get_landmark(current_case, maps_dir, "tho_t8_r.csv") + _get_landmark(current_case, maps_dir, "tho_t8_l.csv"))
    mid_px_t8 = 0.5 * (px_pt + t8_pt)
    mid_ij_c7 = 0.5 * (ij_pt + c7_pt)
    yt_raw = mid_ij_c7 - mid_px_t8
    yt = yt_raw / np.linalg.norm(yt_raw)
    yz1 = mid_px_t8 - ij_pt
    yz2 = c7_pt - ij_pt
    zt_raw = np.cross(yz2, yz1)
    zt = zt_raw / np.linalg.norm(zt_raw)
    xt = np.cross(yt, zt)
    
    t_vectors = (np.array([xt, yt, zt])).T
    t_source = t_vectors # Already 3x3 local orientation
    t_t_mat = Cloud.transform_between_3x3_points_sets(t_source, t_target)
    
    def transform_mesh(verts, trans_vec, rot_mat):
        v = np.array(verts) - trans_vec
        homo = np.hstack((v, np.ones((v.shape[0], 1))))
        return (rot_mat @ homo.T).T[:, :3]

    tho_verts, tho_inds = filter_bone_indices(current_case, all_faces, maps_dir, "Tho.csv")
    final_thorax = transform_mesh(tho_verts, ij_pt, t_t_mat)
    
    # Filter Thorax vertices by side for independent constraints
    # ISB X-axis points Right. X > 0 is Right, X < 0 is Left.
    final_thorax_r = final_thorax[final_thorax[:, 0] > -10] # Small overlap for robustness
    final_thorax_l = final_thorax[final_thorax[:, 0] < 10]
    
    t_transformed_vectors = (t_t_mat[:3, :3] @ t_vectors).T 
    markers.append({"pos": [0,0,0], "label": "IJ", "color": "yellow"})
    for i, c in enumerate(["#ff0000", "#00ff00", "#0000ff"]):
        axes.append({"start": [0,0,0], "dir": t_transformed_vectors[i].tolist(), "color": c})

    # 2. HIERARCHICAL ASSEMBLY (Right Side)
    # Get joint centers in global space
    sc_r_glob = _get_sphere_center(current_case, maps_dir, "tho_scj_r.csv")
    ac_r_glob = _get_sphere_center(current_case, maps_dir, "scap_acj_r.csv")
    gh_r_glob = _get_sphere_center(current_case, maps_dir, "scap_ghj_r.csv")
    
    # Map joint centers to Thorax ISB frame (IJ = 0,0,0)
    def map_to_isb(glob_pt):
        return (t_t_mat[:3, :3] @ (glob_pt - ij_pt).T).T
        
    lm_r = {
        'SC': map_to_isb(sc_r_glob),
        'AC': map_to_isb(ac_r_glob),
        'GH': map_to_isb(gh_r_glob)
    }
    
    # Extract and Transform Meshes into Thorax-ISB frame
    cla_r_v, cla_r_i = filter_bone_indices(current_case, all_faces, maps_dir, "R_clav.csv")
    sca_r_v, sca_r_i = filter_bone_indices(current_case, all_faces, maps_dir, "R_scap.csv")
    hum_r_v, hum_r_i = filter_bone_indices(current_case, all_faces, maps_dir, "R_hum.csv")
    
    cla_r_v_isb = transform_mesh(cla_r_v, ij_pt, t_t_mat)
    sca_r_v_isb = transform_mesh(sca_r_v, ij_pt, t_t_mat)
    hum_r_v_isb = transform_mesh(hum_r_v, ij_pt, t_t_mat)

    # Run Hierarchical Optimizer (Right)
    res_r = solve_hierarchical_shoulder(
        final_thorax_r,
        {'Clavicle': cla_r_v_isb, 'Scapula': sca_r_v_isb, 'Humerus': hum_r_v_isb},
        {k: np.squeeze(v) for k,v in lm_r.items()},
        is_left=False
    )
    
    # 3. HIERARCHICAL ASSEMBLY (Left Side)
    sc_l_glob = _get_sphere_center(current_case, maps_dir, "tho_scj_l.csv")
    ac_l_glob = _get_sphere_center(current_case, maps_dir, "scap_acj_l.csv")
    gh_l_glob = _get_sphere_center(current_case, maps_dir, "scap_ghj_l.csv")
    
    lm_l = {
        'SC': map_to_isb(sc_l_glob),
        'AC': map_to_isb(ac_l_glob),
        'GH': map_to_isb(gh_l_glob)
    }
    
    # Extract and Transform Meshes into Thorax-ISB frame
    cla_l_v, cla_l_i = filter_bone_indices(current_case, all_faces, maps_dir, "L_clav.csv")
    sca_l_v, sca_l_i = filter_bone_indices(current_case, all_faces, maps_dir, "L_scap.csv")
    hum_l_v, hum_l_i = filter_bone_indices(current_case, all_faces, maps_dir, "L_hum.csv")

    cla_l_v_isb = transform_mesh(cla_l_v, ij_pt, t_t_mat)
    sca_l_v_isb = transform_mesh(sca_l_v, ij_pt, t_t_mat)
    hum_l_v_isb = transform_mesh(hum_l_v, ij_pt, t_t_mat)

    # Run Hierarchical Optimizer (Left with Mirror-to-Right)
    res_l = solve_hierarchical_shoulder(
        final_thorax_l,
        {'Clavicle': cla_l_v_isb, 'Scapula': sca_l_v_isb, 'Humerus': hum_l_v_isb},
        {k: np.squeeze(v) for k,v in lm_l.items()},
        is_left=True
    )

    # Markers for Visualization
    markers.append({"pos": lm_r['SC'].tolist(), "label": "R_SC", "color": "pink"})
    markers.append({"pos": lm_l['SC'].tolist(), "label": "L_SC", "color": "pink"})
    
    # Export JSON Payload
    payload = {
        "center": [0,0,0],
        "spread": 400,
        "bones": [
            {"label": "Thorax", "color": "#90CFF0", "vertices": final_thorax.tolist(), "indices": tho_inds},
            {"label": "R Clavicle", "color": "#C080FF", "vertices": cla_r_v_isb.tolist(), "indices": cla_r_i},
            {"label": "L Clavicle", "color": "#FFB0D0", "vertices": cla_l_v_isb.tolist(), "indices": cla_l_i},
            {"label": "R Scapula", "color": "#FFA040", "vertices": sca_r_v_isb.tolist(), "indices": sca_r_i},
            {"label": "L Scapula", "color": "#FFE060", "vertices": sca_l_v_isb.tolist(), "indices": sca_l_i},
            {"label": "R Humerus", "color": "#FF6060", "vertices": hum_r_v_isb.tolist(), "indices": hum_r_i},
            {"label": "L Humerus", "color": "#FF6060", "vertices": hum_l_v_isb.tolist(), "indices": hum_l_i}
        ],
        "markers": markers,
        "axes": axes,
        "isb_joints": {
            "right": {
                "sc": lm_r['SC'].tolist(),
                "ac": lm_r['AC'].tolist(),
                "gh": lm_r['GH'].tolist(),
                "angles": res_r['angles'].tolist()
            },
            "left": {
                "sc": lm_l['SC'].tolist(),
                "ac": lm_l['AC'].tolist(),
                "gh": lm_l['GH'].tolist(),
                "angles": res_l['angles'].tolist()
            }
        }
    }

    with open(export_path, 'w', encoding='utf-8') as f:
        json.dump(payload, f)
        
    print(f"Hierarchical Assembly Complete! File: {export_path}")

if __name__ == "__main__":
    process_and_export()
