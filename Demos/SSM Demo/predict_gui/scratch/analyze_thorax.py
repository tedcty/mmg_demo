import numpy as np
import pandas as pd
import os
import vtk

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

def main():
    res_dir = r"e:\Repo\mmg_demo\Demos\SSM Demo\Resources"
    maps_dir = os.path.join(res_dir, "landmarks", "maps to mean")
    target_ply = os.path.join(res_dir, "SSM_shape_model_103", "CombinedSSM_103_PCA_mean.ply")
    
    reader = vtk.vtkPLYReader()
    reader.SetFileName(target_ply)
    reader.Update()
    polydata = reader.GetOutput()
    
    from ptb.util.data import VTKMeshUtl
    current_case = VTKMeshUtl.extract_points(polydata)
    pts = np.array(current_case)
    
    def get_lm(name):
        f = os.path.join(maps_dir, name)
        idm = pd.read_csv(f)['idm'].to_list()
        return np.mean(pts[idm], axis=0)

    ij = get_lm("tho_ij.csv")
    px = get_lm("tho_px.csv")
    c7_r = get_lm("tho_c7_r.csv")
    c7_l = get_lm("tho_c7_l.csv")
    t8_r = get_lm("tho_t8_r.csv")
    t8_l = get_lm("tho_t8_l.csv")
    
    c7 = 0.5 * (c7_r + c7_l)
    t8 = 0.5 * (t8_r + t8_l)
    
    mid_px_t8 = 0.5 * (px + t8)
    mid_ij_c7 = 0.5 * (ij + c7)
    
    yt = (mid_ij_c7 - mid_px_t8)
    yt /= np.linalg.norm(yt)
    
    yz1 = mid_px_t8 - ij
    yz2 = c7 - ij
    zt = np.cross(yz2, yz1)
    zt /= np.linalg.norm(zt)
    
    xt = np.cross(yt, zt)
    
    print("Axes (Raw Space):")
    print(f"  XT: {xt}")
    print(f"  YT: {yt}")
    print(f"  ZT: {zt}")
    
    t_source = np.array([xt, yt, zt]).T
    t_target = np.eye(3)
    rot = np.linalg.solve(t_source, t_target).T
    
    print("\nTransformed Landmarks (IJ at [0,0,0]):")
    def trans(p): return rot @ (p - ij)
    
    print(f"  IJ: {trans(ij)}")
    print(f"  PX: {trans(px)}")
    print(f"  C7: {trans(c7)}")
    print(f"  T8: {trans(t8)}")
    print(f"  C7_R: {trans(c7_r)}")
    print(f"  C7_L: {trans(c7_l)}")
    
    # Scapula Right Seed
    aa_r = get_lm("sca_r_aa.csv")
    ts_r = get_lm("sca_r_ts.csv")
    ai_r = get_lm("sca_r_ai.csv")
    
    print(f"\nScapula R (Seed approx):")
    print(f"  Centroid: {trans((aa_r + ts_r + ai_r)/3.0)}")
    
    # Mesh Bounds
    fpath = os.path.join(maps_dir, "Tho.csv")
    idm_set = set(pd.read_csv(fpath)['idm'].to_list())
    tho_pts = pts[list(idm_set)]
    tho_trans = np.array([trans(p) for p in tho_pts])
    
    print(f"\nThorax Transformed Bounds:")
    print(f"  X: {tho_trans[:,0].min():.1f} to {tho_trans[:,0].max():.1f}")
    print(f"  Y: {tho_trans[:,1].min():.1f} to {tho_trans[:,1].max():.1f}")
    print(f"  Z: {tho_trans[:,2].min():.1f} to {tho_trans[:,2].max():.1f}")

if __name__ == "__main__":
    main()
