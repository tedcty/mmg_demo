import os
import json
import numpy as np
import pandas as pd
import vtk
import copy
from ptb.util.data import VTKMeshUtl
from ptb.util.math.transformation import Cloud
from scapulothoracic_constraint import ShoulderKinematicTree
from scipy.spatial.transform import Rotation as R
from scipy.interpolate import LSQBivariateSpline
from scapulothoracic_constraint import solve_hierarchical_shoulder
from fabrik_solver import apply_fabrik_alignment

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

def load_muscle_cloud(case_verts, fpath):
    if not os.path.exists(fpath):
        return None
    idm = pd.read_csv(fpath)['idm'].to_list()
    return [case_verts[idx].tolist() for idx in idm]

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

def process_and_export(target_ply=None, fabrik_step=1):
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
    
    # --- 2. ASSEMBLY SETUP ---
    # Sphere fits for joint centers
    rc_sc_tho = _get_sphere_center(current_case_arr, maps_dir, "tho_scj_r.csv")
    rc_sc_tho_l = _get_sphere_center(current_case_arr, maps_dir, "tho_scj_l.csv")
    
    tho_sc_r_glob = (t_t_mat[:3, :3] @ (rc_sc_tho - ij_pt))
    tho_sc_l_glob = (t_t_mat[:3, :3] @ (rc_sc_tho_l - ij_pt))

    def project_scapula_to_thorax(tho_mesh, aa, ts, ai):
        centroid = (aa + ts + ai) / 3.0
        
        # Diagnostic: Print landmark relative positions
        px_glob = (t_t_mat[:3, :3] @ (px_pt - ij_pt))
        t8_glob = (t_t_mat[:3, :3] @ (t8_pt - ij_pt))
        print(f"  DIAG: PX_X={px_glob[0]:.1f}, T8_X={t8_glob[0]:.1f}")
        
        v1, v2 = aa - ts, ai - ts
        n = np.cross(v1, v2)
        n /= np.linalg.norm(n)
        
        # X negative is Posterior in this JCS (T8_X < PX_X).
        # The scapula is often anterior in the seed, so we project posteriorly (negative X).
        if n[0] > 0: n = -n
        
        x_mid = (tho_mesh[:, 0].min() + tho_mesh[:, 0].max()) / 2.0
        z_mid = (tho_mesh[:, 2].min() + tho_mesh[:, 2].max()) / 2.0
        y_t8 = (t_t_mat[:3, :3] @ (t8_pt - ij_pt))[1]
        
        # Filter for the correct side (Lateral Z) and posterior aspect (X < x_mid)
        side_mask = (tho_mesh[:, 2] > z_mid) if (centroid[2] > z_mid) else (tho_mesh[:, 2] < z_mid)
        # The posterior surface of the thorax is at the LOWER X values (negative X is Posterior).
        post_mask = (tho_mesh[:, 0] < x_mid) 
        glide_mask = side_mask & post_mask & (tho_mesh[:, 1] > y_t8 - 150)
        
        glide_pts = tho_mesh[glide_mask]
        if len(glide_pts) < 50: return centroid # fallback
        
        # Spline fit for the glide area
        # We'll use Y and Z as predictors for X (depth)
        y_pts, z_pts, x_pts = glide_pts[:, 1], glide_pts[:, 2], glide_pts[:, 0]
        ky = np.linspace(y_pts.min(), y_pts.max(), 5)[1:-1]
        kz = np.linspace(z_pts.min(), z_pts.max(), 5)[1:-1]
        spline = LSQBivariateSpline(y_pts, z_pts, x_pts, ky, kz)
        
        # Find intersection
        # Pt = Centroid + t * n
        # find t such that Pt[0] = spline.ev(Pt[1], Pt[2])
        best_idx = np.argmin(np.linalg.norm(np.cross(glide_pts - centroid, n), axis=1))
        t_guess = np.dot(glide_pts[best_idx] - centroid, n)
        
        from scipy.optimize import fsolve
        def intersect_err(t):
            px = centroid[0] + t * n[0]
            py = centroid[1] + t * n[1]
            pz = centroid[2] + t * n[2]
            sx = spline.ev(py, pz)
            return px - sx
            
        t_sol, info, ier, msg = fsolve(intersect_err, t_guess, full_output=True)
        
        projected_pt = centroid + t_sol[0] * n
        
        # Validation: Is the result actually on the posterior?
        # If fsolve failed or the point is too far anterior, use fallback
        if ier != 1 or projected_pt[0] > x_mid:
            best_idx = np.argmin(np.linalg.norm(glide_pts - centroid, axis=1))
            projected_pt = glide_pts[best_idx]
            print(f"  FABRIK PROJ: Ray missed or hit anterior. Using closest posterior point at {projected_pt[0]:.1f}")
        else:
            print(f"  FABRIK PROJ: Ray hit posterior at X={projected_pt[0]:.1f}")
            
        return projected_pt
    
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
    
    # Transform Clavicle to Global (Thorax) frame
    # 1. Align orientation to Thorax (global)
    c_t_mat = Cloud.transform_between_3x3_points_sets(c_source, t_target)
    # 2. Align SC joint centers
    sc_offset = tho_sc_r_glob - (c_t_mat[:3, :3] @ (rc_sc_cla - ij_pt))
    
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
    
    # Seed positions for FABRIK
    aa_glob_seed = (s_t_mat[:3, :3] @ (aa_pt - ij_pt)) + ac_offset
    ts_glob_seed = (s_t_mat[:3, :3] @ (ts_pt - ij_pt)) + ac_offset
    ai_glob_seed = (s_t_mat[:3, :3] @ (ai_pt - ij_pt)) + ac_offset
    scap_mesh_seed = transform_mesh(sca_r_verts, ij_pt, s_t_mat) + ac_offset
    
    # Load and transform Subscapularis point cloud (ID 69)
    subscap_r_path = os.path.join(res_dir, "MAS_103", "Scapula_right", "69_NodeNo_2.csv")
    subscap_r_pts = load_muscle_cloud(current_case_arr, subscap_r_path)
    subscap_r_seed = transform_mesh(subscap_r_pts, ij_pt, s_t_mat) + ac_offset if subscap_r_pts else None

    print("  FABRIK: Optimizing Right Scapula Alignment...")
    proj_r_target = project_scapula_to_thorax(final_thorax, aa_glob_seed, ts_glob_seed, ai_glob_seed)
    
    # Pass the initial JCS orientation to prevent flipping
    rot_r_seed = R.from_matrix(s_t_mat[:3, :3])
    
    final_scap_r, ac_r_opt, rot_r_opt = apply_fabrik_alignment(
        "right", final_thorax, tho_sc_r_glob, c_ac_r_glob,
        aa_glob_seed, ts_glob_seed, ai_glob_seed, scap_mesh_seed, 
        p_proj=proj_r_target, initial_rot=rot_r_seed, max_step=fabrik_step
    )
    
    # Store old AC joint for transformations
    c_ac_r_glob_old = c_ac_r_glob
    
    # Apply same rotation to subscapularis points
    final_subscap_r = None
    if subscap_r_seed is not None:
        final_subscap_r = rot_r_opt.apply(subscap_r_seed - c_ac_r_glob_old) + ac_r_opt
    
    # 2d. Synchronize Clavicle to the new AC joint position
    v_clav_old = c_ac_r_glob - tho_sc_r_glob
    v_clav_new = ac_r_opt - tho_sc_r_glob
    if np.linalg.norm(v_clav_new) > 1e-6 and np.linalg.norm(v_clav_old) > 1e-6:
        # Rotate clavicle mesh around SC joint
        rot_clav, _ = R.align_vectors([v_clav_new], [v_clav_old])
        final_clav_r = rot_clav.apply(final_clav_r - tho_sc_r_glob) + tho_sc_r_glob
    
    # Update global AC and landmarks for child (Humerus) and diagnostics
    c_ac_r_glob_old = c_ac_r_glob
    c_ac_r_glob = ac_r_opt
    
    # If Step 0, rot_r_opt is identity, and ac_r_opt is c_ac_r_glob_old
    aa_r_glob = rot_r_opt.apply(aa_glob_seed - c_ac_r_glob_old) + ac_r_opt
    ts_r_glob = rot_r_opt.apply(ts_glob_seed - c_ac_r_glob_old) + ac_r_opt
    ai_r_glob = rot_r_opt.apply(ai_glob_seed - c_ac_r_glob_old) + ac_r_opt
    
    # Compute corrected scapular plane normal (dorsal direction, away from ribs)
    v1_scap = ai_r_glob - ts_r_glob
    v2_scap = aa_r_glob - ts_r_glob
    n_scap_r = np.cross(v1_scap, v2_scap)
    n_scap_r /= np.linalg.norm(n_scap_r)
    # Dorsal direction = away from ribs = posterior = negative X
    if n_scap_r[0] > 0:
        n_scap_r = -n_scap_r
    
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
    sca_gh_r_glob_seed = (s_t_mat[:3, :3] @ (sca_gh_r_pt - ij_pt)) + ac_offset
    
    # Update GH joint to follow solved Scapula
    sca_gh_r_glob = rot_r_opt.apply(sca_gh_r_glob_seed - c_ac_r_glob_old) + ac_r_opt
    
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
    
    # Left subscapularis
    subscap_l_path = os.path.join(res_dir, "MAS_103", "Scapula_left", "69_NodeNo_2.csv")
    subscap_l_pts = load_muscle_cloud(current_case_arr, subscap_l_path)
    final_subscap_l = transform_mesh(subscap_l_pts, ij_pt, sl_t_mat) + ac_offset_l if subscap_l_pts else None
    
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

    # Calculate Scapular Landmarks for Diagnostic
    # Landmarks for diagnostics (Right side already calculated via FABRIK)
    # Left Side Assembly
    
    aa_l_glob = (transform_mesh([aa_l_pt], ij_pt, sl_t_mat)[0] + ac_offset_l)
    ts_l_glob = (transform_mesh([ts_l_pt], ij_pt, sl_t_mat)[0] + ac_offset_l)
    ai_l_glob = (transform_mesh([ai_l_pt], ij_pt, sl_t_mat)[0] + ac_offset_l)

    # --- 4. SCAPULOTHORACIC PROJECTION DIAGNOSTIC ---
    print("Calculating Scapulothoracic Projection Markers...")
    
    # Right Side Projection
    if proj_r_target is not None:
        markers.append({"pos": proj_r_target.tolist(), "label": "R_Proj", "color": "cyan"})
        
    # Left Side Projection (Diagnostic only)
    proj_l = project_scapula_to_thorax(final_thorax, aa_l_glob, ts_l_glob, ai_l_glob)
    if proj_l is not None:
        markers.append({"pos": proj_l.tolist(), "label": "L_Proj", "color": "cyan"})

    # --- Right Scapular Plane Landmarks (FABRIK-solved) ---
    markers.append({"pos": aa_r_glob.tolist(), "label": "R_AA", "color": "#FF4444"})
    markers.append({"pos": ts_r_glob.tolist(), "label": "R_TS", "color": "#44FF44"})
    markers.append({"pos": ai_r_glob.tolist(), "label": "R_AI", "color": "#4444FF"})
    
    # Scapular plane centroid
    centroid_r_glob = (aa_r_glob + ts_r_glob + ai_r_glob) / 3.0
    markers.append({"pos": centroid_r_glob.tolist(), "label": "R_Centroid", "color": "#FFFFFF"})

    # Initialize angles as zero for export
    q_r = np.zeros(6)
    q_l = np.zeros(6)

    # --- 5. EXPORT ---
    final_scap_r_rel = final_scap_r - ij_pt
    final_hum_r_rel = final_hum_r - ij_pt
    tho_sc_r_rel = tho_sc_r_glob # This was (mat @ (pt - ij)) which IS relative to ij
    c_ac_r_rel = c_ac_r_glob - ij_pt
    sca_gh_r_rel = sca_gh_r_glob - ij_pt
    aa_r_rel = aa_r_glob - ij_pt
    ts_r_rel = ts_r_glob - ij_pt
    ai_r_rel = ai_r_glob - ij_pt
    
    # --- 5. EXPORT ---

    # --- 4. EXPORT ---
    
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
            {"label": "R Subscapularis", "color": "#FF4444", "vertices": final_subscap_r.tolist() if final_subscap_r is not None else [], "indices": [], "origin": c_ac_r_glob.tolist()},
            {"label": "L Subscapularis", "color": "#FF4444", "vertices": final_subscap_l.tolist() if final_subscap_l is not None else [], "indices": [], "origin": c_ac_l_glob.tolist()},
            {"label": "R Humerus", "color": "#FF6060", "vertices": final_hum_r.tolist(), "indices": hum_r_inds, "origin": sca_gh_r_glob.tolist()},
            {"label": "L Humerus", "color": "#FF6060", "vertices": final_hum_l.tolist(), "indices": hum_l_inds, "origin": sca_gh_l_glob.tolist()}
        ],
        "markers": markers,
        "scapular_planes": {
            "right": {
                "aa": aa_r_glob.tolist(),
                "ts": ts_r_glob.tolist(),
                "ai": ai_r_glob.tolist(),
                "centroid": centroid_r_glob.tolist(),
                "normal": n_scap_r.tolist()
            },
            "left": {
                "aa": aa_l_glob.tolist(),
                "ts": ts_l_glob.tolist(),
                "ai": ai_l_glob.tolist(),
                "centroid": ((aa_l_glob + ts_l_glob + ai_l_glob) / 3.0).tolist()
            }
        },
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
                "scapula_aa": aa_r_glob.tolist(),
                "scapula_ts": ts_r_glob.tolist(),
                "scapula_ai": ai_r_glob.tolist(),
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
                "scapula_aa": aa_l_glob.tolist(),
                "scapula_ts": ts_l_glob.tolist(),
                "scapula_ai": ai_l_glob.tolist(),
            }
        },
        "isb_joints": {
            "right": {
                "sc": tho_sc_r_glob.tolist(),
                "ac": c_ac_r_glob.tolist(),
                "gh": sca_gh_r_glob.tolist(),
                "angles": q_r.tolist()
            },
            "left": {
                "sc": tho_sc_l_glob.tolist(),
                "ac": c_ac_l_glob.tolist(),
                "gh": sca_gh_l_glob.tolist(),
                "angles": q_l.tolist()
            }
        }
    }

    with open(export_path, 'w', encoding='utf-8') as f:
        json.dump(payload, f)
        
    print(f"Hierarchical Assembly Complete! File: {export_path}")

if __name__ == "__main__":
    process_and_export()
