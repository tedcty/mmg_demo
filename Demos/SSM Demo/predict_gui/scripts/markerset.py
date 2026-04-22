import os
import numpy as np
import pandas as pd
import vtk
from ptb.util.data import VTKMeshUtl
import pyvista as pv

# bone ply
thorax = r"C:\Users\rrag962\Documents\shapemodel\model\Shoulder 3\Segmentations\SSM\Thorax\big thorax mesh\a3.merged_fitted_thorax\15_6499_thorax.ply"
clav= r"C:\Users\rrag962\Documents\shapemodel\model\Shoulder 3\Segmentations\SSM\Clavicle\Right\fitted_meshes\15_6499_R_Clavicle_rm_rbfreg.ply"
scap = r"C:\Users\rrag962\Documents\shapemodel\model\Shoulder 3\Segmentations\SSM\Scapula\Right\fitted_meshes\15_6499_R_Scapula_rm_rbfreg.ply"
hum = r"C:\Users\rrag962\Documents\shapemodel\model\Shoulder 3\Segmentations\SSM\Humerus\Right\fitted_meshes\15_6499_R_Humerus_rm_rbfreg.ply"

# maps of landmark .csv
IJ = pd.read_csv(r"C:\Users\rrag962\Documents\shapemodel\model\Shoulder 3\Segmentations\Landmarking\VIFM_Markerset\maps to mean\IJ.csv")
PX = pd.read_csv(r"C:\Users\rrag962\Documents\shapemodel\model\Shoulder 3\Segmentations\Landmarking\VIFM_Markerset\maps to mean\PX.csv")
C1_1 = pd.read_csv(r"C:\Users\rrag962\Documents\shapemodel\model\Shoulder 3\Segmentations\Landmarking\VIFM_Markerset\maps to mean\C1_1.csv")
C1_2 = pd.read_csv(r"C:\Users\rrag962\Documents\shapemodel\model\Shoulder 3\Segmentations\Landmarking\VIFM_Markerset\maps to mean\C1_2.csv")
T8_1 = pd.read_csv(r"C:\Users\rrag962\Documents\shapemodel\model\Shoulder 3\Segmentations\Landmarking\VIFM_Markerset\maps to mean\T8_1.csv")
T8_2 = pd.read_csv(r"C:\Users\rrag962\Documents\shapemodel\model\Shoulder 3\Segmentations\Landmarking\VIFM_Markerset\maps to mean\T8_2.csv")
C_SC = pd.read_csv(r"C:\Users\rrag962\Documents\shapemodel\model\Shoulder 3\Segmentations\Landmarking\VIFM_Markerset\maps to mean\C_SC.csv")
C_AC = pd.read_csv(r"C:\Users\rrag962\Documents\shapemodel\model\Shoulder 3\Segmentations\Landmarking\VIFM_Markerset\maps to mean\C_AC.csv")
CAP = pd.read_csv(r"C:\Users\rrag962\Documents\shapemodel\model\Shoulder 3\Segmentations\Landmarking\VIFM_Markerset\maps to mean\CAP.csv")
SA = pd.read_csv(r"C:\Users\rrag962\Documents\shapemodel\model\Shoulder 3\Segmentations\Landmarking\VIFM_Markerset\maps to mean\SA.csv")
IA = pd.read_csv(r"C:\Users\rrag962\Documents\shapemodel\model\Shoulder 3\Segmentations\Landmarking\VIFM_Markerset\maps to mean\IA.csv")
TS = pd.read_csv(r"C:\Users\rrag962\Documents\shapemodel\model\Shoulder 3\Segmentations\Landmarking\VIFM_Markerset\maps to mean\TS.csv")
DEL = pd.read_csv(r"C:\Users\rrag962\Documents\shapemodel\model\Shoulder 3\Segmentations\Landmarking\VIFM_Markerset\maps to mean\DEL.csv")
EL = pd.read_csv(r"C:\Users\rrag962\Documents\shapemodel\model\Shoulder 3\Segmentations\Landmarking\VIFM_Markerset\maps to mean\EL.csv")
EM = pd.read_csv(r"C:\Users\rrag962\Documents\shapemodel\model\Shoulder 3\Segmentations\Landmarking\VIFM_Markerset\maps to mean\EM.csv")

def read_mesh_points(file_path):
    reader = vtk.vtkPLYReader()
    reader.SetFileName(file_path)
    reader.Update()
    mesh_data = reader.GetOutput()
    return VTKMeshUtl.extract_points(mesh_data)

def write_static_trc(filename, marker_names, marker_coords):
    with open(filename, "w") as f:
        f.write(f"PathFileType\t4\tX/Y/Z\t{filename}\n")
        f.write("DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\tOrigDataRate\tOrigDataStartFrame\tOrigNumFrames\n")
        f.write(f"100\t100\t1\t{len(marker_names)}\tmm\t100\t1\t1\n")
        f.write("Frame#\tTime\t" + "\t\t\t".join(marker_names) + "\n")
        xyz_labels = "\t".join([f"X{i + 1}\tY{i + 1}\tZ{i + 1}" for i in range(len(marker_names))])
        f.write("\t\t" + xyz_labels + "\n")
        coords_flat = "\t".join([f"{x:.5f}\t{y:.5f}\t{z:.5f}" for x, y, z in marker_coords])
        f.write(f"1\t0.00\t{coords_flat}\n")

thorax_pts = read_mesh_points(thorax)
ij_point = np.mean(thorax_pts[IJ['idm'].to_list()], axis=0)
px_point = np.mean(thorax_pts[PX['idm'].to_list()], axis=0)
c1_1_point = np.mean(thorax_pts[C1_1['idm'].to_list()], axis=0)
c1_2_point = np.mean(thorax_pts[C1_2['idm'].to_list()], axis=0)
t8_1_point = np.mean(thorax_pts[T8_1['idm'].to_list()], axis=0)
t8_2_point = np.mean(thorax_pts[T8_2['idm'].to_list()], axis=0)
c7_point = 0.5 * (c1_1_point + c1_2_point)
t8_point = 0.5 * (t8_1_point + t8_2_point)

clav_pts = read_mesh_points(clav)
c_sc_point = np.mean(clav_pts[C_SC['idm'].to_list()], axis=0)
c_ac_point = np.mean(clav_pts[C_AC['idm'].to_list()], axis=0)

scap_pts = read_mesh_points(scap)
cap_point = np.mean(scap_pts[CAP['idm'].to_list()], axis=0) # aa point
sa_point = np.mean(scap_pts[SA['idm'].to_list()], axis=0)
ia_point = np.mean(scap_pts[IA['idm'].to_list()], axis=0)
ts_point = np.mean(scap_pts[TS['idm'].to_list()], axis=0)

hum_pts = read_mesh_points(hum)
del_point = np.mean(hum_pts[DEL['idm'].to_list()], axis=0)
el_point = np.mean(hum_pts[EL['idm'].to_list()], axis=0)
em_point = np.mean(hum_pts[EM['idm'].to_list()], axis=0)

marker_names = ['ij', 'px', 'c7', 't8', 'sc', 'acc', 'aa', 'sa', 'ai', 'ts', 'del', 'EpL', 'EpM']
marker_coords = np.array([ij_point, px_point, c7_point, t8_point, c_sc_point, c_ac_point, cap_point, sa_point, ia_point, ts_point, del_point, el_point, em_point])
# sf = 0.1
# scaled_0_1_markers = marker_coords * sf

write_static_trc(r"C:\Users\rrag962\Documents\shapemodel\model\Shoulder 3\Segmentations\Articulation\markers_ssm.trc", marker_names, marker_coords)

print("a")