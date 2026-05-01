import os
import numpy as np
import pandas as pd
import vtk
from vtkmodules.util.numpy_support import vtk_to_numpy
from ptb.util.data import VTKMeshUtl
import pyvista as pv
from enum import Enum
from ptb.util.math.transformation import Cloud
import copy
import pymeshlab as pm

class Keywords(Enum):
    actor = [0, 'actor']
    polydata = [1, 'polydata']
    vertices = [2, 'vertices']
    idx = [3, 'idx']
    idm = [4, 'idm']


def sphere_fit(points):
    """
    :param points: ndarray that is n x 3
    :return: center of the points
    """
    p_mean = np.nanmean(points, axis=0)
    n = points.shape[0]
    a = np.eye(3)
    for i in range(0, 3):
        a[i, 0] = np.nansum([(points[x, i] * (points[x, 0] - p_mean[0])) / n for x in range(0, n)])
        a[i, 1] = np.nansum([(points[x, i] * (points[x, 1] - p_mean[1])) / n for x in range(0, n)])
        a[i, 2] = np.nansum([(points[x, i] * (points[x, 2] - p_mean[2])) / n for x in range(0, n)])
    a: np.ndarray = 2 * a
    b: np.ndarray = np.atleast_2d([[0.0], [0.0], [0.0]])
    xc = np.array([points[x, 0] ** 2 for x in range(0, n)])
    yc = np.array([points[x, 1] ** 2 for x in range(0, n)])
    zc = np.array([points[x, 2] ** 2 for x in range(0, n)])
    sum_axis = xc + yc + zc
    xb = sum_axis * (np.transpose(points[:, 0]) - p_mean[0]) / n
    yb = sum_axis * (np.transpose(points[:, 1]) - p_mean[1]) / n
    zb = sum_axis * (np.transpose(points[:, 2]) - p_mean[2]) / n

    b[0, 0] = np.sum(xb)
    b[1, 0] = np.sum(yb)
    b[2, 0] = np.sum(zb)
    c = np.matmul(np.linalg.inv(np.matmul(a.transpose(), a)), np.matmul(a.transpose(), b))
    return np.squeeze(c)

def sphere_fit_with_radius(points, desired_radius=None):
    p_mean = np.nanmean(points, axis=0)
    n = points.shape[0]
    a = np.eye(3)
    for i in range(3):
        a[i, 0] = np.nansum([(points[x, i] * (points[x, 0] - p_mean[0])) / n for x in range(n)])
        a[i, 1] = np.nansum([(points[x, i] * (points[x, 1] - p_mean[1])) / n for x in range(n)])
        a[i, 2] = np.nansum([(points[x, i] * (points[x, 2] - p_mean[2])) / n for x in range(n)])
    a *= 2
    b = np.zeros((3, 1))
    sum_axis = np.sum(points**2, axis=1)
    b[0, 0] = np.sum(sum_axis * (points[:, 0] - p_mean[0]) / n)
    b[1, 0] = np.sum(sum_axis * (points[:, 1] - p_mean[1]) / n)
    b[2, 0] = np.sum(sum_axis * (points[:, 2] - p_mean[2]) / n)
    c = np.linalg.solve(np.dot(a.T, a), np.dot(a.T, b))
    center = np.squeeze(c)

    if desired_radius is not None:
        # Adjust the center to fit the desired radius
        current_radius = np.sqrt(np.mean(np.sum((points - center)**2, axis=1)))
        scale_factor = desired_radius / current_radius
        center = p_mean + scale_factor * (center - p_mean)

    return center

def transform_point(point, translation_vector, transform_matrix):
    translated = point - translation_vector
    homogeneous = np.append(translated, 1)
    transformed = transform_matrix @ homogeneous
    return transformed[:3]

def get_closest_point(mesh_df, landmark_ids, reference_point):
    index_map = pd.Series(mesh_df.index.values, index=mesh_df['idm']).to_dict()
    landmark_indices = [index_map[idm] for idm in landmark_ids]
    landmark_df = mesh_df.iloc[landmark_indices]
    landmark_coords = landmark_df[['x', 'y', 'z']].to_numpy()
    reference_point = np.array(reference_point).reshape(1, 3)
    distances = np.linalg.norm(landmark_coords - reference_point, axis=1)
    closest_idx = np.argmin(distances)
    return landmark_df.iloc[closest_idx][['x', 'y', 'z']].to_numpy()

def get_MAS_xyz(mesh_df,MAS):
    MAS_points = MAS['idm'].to_list()
    MAS_map = pd.Series(mesh_df.index.values, index=mesh_df['idm']).to_dict()
    MAS_idm = [MAS_map[idm] for idm in MAS_points]
    MAS_df = mesh_df.iloc[MAS_idm]
    MAS_coords = MAS_df[['x', 'y', 'z']].to_numpy()

    return MAS_coords


# import csv
# import os
#
#
# def save_to_csv(data_list, filename="isb_data.csv"):
#     # 1. Define the order of the joints (the keys in your dictionary)
#     joints = ['IJ', 'r_SC', 'l_SC', 'r_AC', 'l_AC', 'r_GH', 'l_GH']
#
#     # 2. Create the header: JointName_X, JointName_Y, JointName_Z for each joint
#     header = []
#     for j in joints:
#         header.extend([f"{j}_X", f"{j}_Y", f"{j}_Z"])
#
#     file_exists = os.path.isfile(filename)
#
#     with open(filename, 'a', newline='') as f:
#         writer = csv.writer(f)
#
#         # Write header only if the file is new
#         if not file_exists:
#             writer.writerow(header)
#
#         # 3. Process each dictionary in your list
#         for entry in data_list:
#             row = []
#             for j in joints:
#                 coords = entry.get(j, [None, None, None])  # Get [x,y,z] or Nones
#                 row.extend(coords)
#             writer.writerow(row)

# export path
articulation_folder = r"C:\Users\rrag962\Documents\shapemodel\model\Shoulder_3\Segmentations\2502"

# Body segment paths
# combined = r"C:\Users\rrag962\Documents\shapemodel\model\Shoulder 3\Segmentations\SSM\Combined\combinedSSM"
# base directory
base_dir = r"C:\Users\rrag962\Documents\shapemodel\model\Shoulder_3\Segmentations"

# intermediate directories
landmarking_dir = os.path.join(base_dir, "Landmarking", "6TH_combined")
maps_to_mean_dir = os.path.join(landmarking_dir, "maps to mean")
test_maps_to_mean_dir = os.path.join(landmarking_dir, "test_maps to mean")
ply_sep_dir = os.path.join(landmarking_dir, "Ply_separated_predicted_bones")
attachment_sites_dir = os.path.join(base_dir, "Attachment sites", "MAS")

# Body segment paths
# combined = os.path.join(base_dir, "SSM", "Combined", "combinedSSM")
mean_ssm = os.path.join(base_dir, "Fit_combined_model_to_partial_data", "predicted_full_upper_body_SSM_n_103_Nancy.ply")
tho_map = pd.read_csv(os.path.join(maps_to_mean_dir, "Tho.csv"))
l_clav = pd.read_csv(os.path.join(maps_to_mean_dir, "L_clav.csv"))
r_clav = pd.read_csv(os.path.join(maps_to_mean_dir, "R_clav.csv"))
l_scap = pd.read_csv(os.path.join(maps_to_mean_dir, "L_scap.csv"))
r_scap = pd.read_csv(os.path.join(maps_to_mean_dir, "R_scap.csv"))
l_hum = pd.read_csv(os.path.join(maps_to_mean_dir, "L_hum.csv"))
r_hum = pd.read_csv(os.path.join(maps_to_mean_dir, "R_hum.csv"))
l_rad = pd.read_csv(os.path.join(maps_to_mean_dir, "L_rad.csv"))
r_rad = pd.read_csv(os.path.join(maps_to_mean_dir, "R_rad.csv"))
l_ulna = pd.read_csv(os.path.join(maps_to_mean_dir, "L_ulna.csv"))
r_ulna = pd.read_csv(os.path.join(maps_to_mean_dir, "R_ulna.csv"))

# landmarking csv maps
C71 = pd.read_csv(os.path.join(maps_to_mean_dir, "tho_c7_r.csv"))
C72 = pd.read_csv(os.path.join(maps_to_mean_dir, "tho_c7_l.csv"))
T81 = pd.read_csv(os.path.join(maps_to_mean_dir, "tho_t8_r.csv"))
T82 = pd.read_csv(os.path.join(maps_to_mean_dir, "tho_t8_l.csv"))
IJ = pd.read_csv(os.path.join(maps_to_mean_dir, "tho_ij.csv"))
PX = pd.read_csv(os.path.join(maps_to_mean_dir, "tho_px.csv"))
ThoIJsphere_r = pd.read_csv(os.path.join(maps_to_mean_dir, "tho_scj_r.csv"))
ThoIJsphere_L = pd.read_csv(os.path.join(maps_to_mean_dir, "tho_scj_l.csv"))
SC_r = pd.read_csv(os.path.join(test_maps_to_mean_dir, "TEST_cla_sc_r.csv"))
AC_r = pd.read_csv(os.path.join(test_maps_to_mean_dir, "TEST_cla_ac_r.csv"))
SC_l = pd.read_csv(os.path.join(test_maps_to_mean_dir, "TEST_cla_sc_l.csv"))
AC_l = pd.read_csv(os.path.join(test_maps_to_mean_dir, "TEST_cla_ac_l.csv"))
ClaSCsphere_r = pd.read_csv(os.path.join(maps_to_mean_dir, "cla_scj_r.csv"))
ClaACsphere_r = pd.read_csv(os.path.join(maps_to_mean_dir, "cla_acj_r.csv"))
ClaSCsphere_l = pd.read_csv(os.path.join(maps_to_mean_dir, "cla_scj_l.csv"))
ClaACsphere_l = pd.read_csv(os.path.join(maps_to_mean_dir, "cla_acj_l.csv"))
sca_AC_r = pd.read_csv(os.path.join(test_maps_to_mean_dir, "TEST_sca_ac_r.csv"))
sca_AC_l = pd.read_csv(os.path.join(test_maps_to_mean_dir, "TEST_sca_ac_l.csv"))
AA_r = pd.read_csv(os.path.join(maps_to_mean_dir, "sca_r_aa.csv"))
AI_r = pd.read_csv(os.path.join(maps_to_mean_dir, "sca_r_ai.csv"))
TS_r = pd.read_csv(os.path.join(maps_to_mean_dir, "sca_r_ts.csv"))
SA_r = pd.read_csv(os.path.join(maps_to_mean_dir, "sca_r_as.csv"))
AA_l = pd.read_csv(os.path.join(maps_to_mean_dir, "sca_l_aa.csv"))
AI_l = pd.read_csv(os.path.join(maps_to_mean_dir, "sca_l_ai.csv"))
TS_l = pd.read_csv(os.path.join(maps_to_mean_dir, "sca_l_ts.csv"))
SA_l = pd.read_csv(os.path.join(maps_to_mean_dir, "sca_l_as.csv"))
S_CAP = pd.read_csv(os.path.join(maps_to_mean_dir, "sca_r_cap.csv"))
S_SA = pd.read_csv(os.path.join(maps_to_mean_dir, "sca_r_as.csv"))
S_IA = pd.read_csv(os.path.join(maps_to_mean_dir, "sca_r_ai.csv"))
S_TS = pd.read_csv(os.path.join(maps_to_mean_dir, "sca_r_ts.csv"))
ScaACsphere_r = pd.read_csv(os.path.join(maps_to_mean_dir, "scap_acj_r.csv"))
glenoid_r = pd.read_csv(os.path.join(maps_to_mean_dir, "scap_ghj_r.csv"))
ScaACsphere_l = pd.read_csv(os.path.join(maps_to_mean_dir, "scap_acj_l.csv"))
glenoid_l = pd.read_csv(os.path.join(maps_to_mean_dir, "scap_ghj_l.csv"))
L_S_CAP = pd.read_csv(os.path.join(maps_to_mean_dir, "sca_l_cap.csv"))
L_S_SA = pd.read_csv(os.path.join(maps_to_mean_dir, "sca_l_as.csv"))
L_S_IA = pd.read_csv(os.path.join(maps_to_mean_dir, "sca_l_ai.csv"))
L_S_TS = pd.read_csv(os.path.join(maps_to_mean_dir, "sca_l_ts.csv"))
EL_r = pd.read_csv(os.path.join(maps_to_mean_dir, "hum_r_el.csv"))
EM_r = pd.read_csv(os.path.join(maps_to_mean_dir, "hum_r_em.csv"))
EL_l = pd.read_csv(os.path.join(maps_to_mean_dir, "hum_l_el.csv"))
EM_l = pd.read_csv(os.path.join(maps_to_mean_dir, "hum_l_em.csv"))
DEL = pd.read_csv(os.path.join(maps_to_mean_dir, "hum_r_del.csv"))
DEL_l = pd.read_csv(os.path.join(maps_to_mean_dir, "hum_l_del.csv"))
humeral_head_r = pd.read_csv(os.path.join(maps_to_mean_dir, "hum_ghj_r.csv"))
HumUlna_r = pd.read_csv(os.path.join(maps_to_mean_dir, "hum_radj_r.csv"))
humeral_head_l = pd.read_csv(os.path.join(maps_to_mean_dir, "hum_ghj_l.csv"))
HumUlna_l = pd.read_csv(os.path.join(maps_to_mean_dir, "hum_radj_l.csv"))
US_r = pd.read_csv(os.path.join(maps_to_mean_dir, "uln_r_us.csv"))
US_l = pd.read_csv(os.path.join(maps_to_mean_dir, "uln_l_us.csv"))
UlnHum_r = pd.read_csv(os.path.join(maps_to_mean_dir, "uln_uocj_r.csv"))
UlnHum_l = pd.read_csv(os.path.join(maps_to_mean_dir, "uln_uocj_l.csv"))
RS_r = pd.read_csv(os.path.join(maps_to_mean_dir, "rad_r_rsp.csv"))
RS_l = pd.read_csv(os.path.join(maps_to_mean_dir, "rad_l_rsp.csv"))
HumRadius_r = pd.read_csv(os.path.join(maps_to_mean_dir, "hum_radj_r.csv"))
RadHumerus_r = pd.read_csv(os.path.join(maps_to_mean_dir, "rad_humj_r.csv"))
HumRadius_l = pd.read_csv(os.path.join(maps_to_mean_dir, "hum_radj_l.csv"))
RadHumerus_l = pd.read_csv(os.path.join(maps_to_mean_dir, "rad_humj_l.csv"))

# separated_bones
ssm_thorax_mesh = os.path.join(ply_sep_dir, "thorax.ply")
ssm_cla_mesh = os.path.join(ply_sep_dir, "r_clav.ply")
l_ssm_cla_mesh = os.path.join(ply_sep_dir, "l_clav.ply")
ssm_SCA_mesh = os.path.join(ply_sep_dir, "r_sca.ply")
l_ssm_SCA_mesh = os.path.join(ply_sep_dir, "l_sca.ply")
ssm_HUM_mesh = os.path.join(ply_sep_dir, "r_hum.ply")
l_ssm_HUM_mesh = os.path.join(ply_sep_dir, "l_hum.ply")

# muscle_mapping_csv
r_cla_MAS_folder = os.path.join(attachment_sites_dir, "Clavicle_right")
l_cla_MAS_folder = os.path.join(attachment_sites_dir, "Clavicle_left")
r_scap_MAS_folder = os.path.join(attachment_sites_dir, "Scapula_right")
l_scap_MAS_folder = os.path.join(attachment_sites_dir, "Scapula_left")
r_hum_MAS_folder = os.path.join(attachment_sites_dir, "Humerus_right")
l_hum_MAS_folder = os.path.join(attachment_sites_dir, "Humerus_left")


ply_reader = vtk.vtkPLYReader()
ply_reader.SetFileName(mean_ssm)
ply_reader.Update()
mesh_data = ply_reader.GetOutput()
mesh_poly = VTKMeshUtl.load(mean_ssm, True)
current_case = VTKMeshUtl.extract_points(mesh_data)

isb_joint_centers = []
# thorax
thorax_map = tho_map['idm'].to_list()
bone_colour = [144 / 255, 207 / 255, 252 / 255]
thorax_data = {Keywords.vertices: current_case[thorax_map, :]}
c71_point = np.mean(current_case[C71['idm'].to_list()], axis=0)
c72_point = np.mean(current_case[C72['idm'].to_list()], axis=0)
t81_point = np.mean(current_case[T81['idm'].to_list()], axis=0)
t82_point = np.mean(current_case[T82['idm'].to_list()], axis=0)
ij_point = np.mean(current_case[IJ['idm'].to_list()], axis=0)
px_point = np.mean(current_case[PX['idm'].to_list()], axis=0)

c7_point = 0.5 * (c72_point + c71_point)
t8_point = 0.5 * (t82_point + t81_point)
mid_px_t8 = 0.5 * (px_point + t8_point)
mid_ij_c7 = 0.5 * (ij_point + c7_point)
yt_raw = mid_ij_c7 - mid_px_t8
yt = (1 / np.linalg.norm(yt_raw)) * yt_raw
yz1_raw = mid_px_t8 - ij_point
yz2_raw = c7_point - ij_point
zt_raw = np.cross(yz2_raw, yz1_raw)
zt = (1 / np.linalg.norm(zt_raw)) * zt_raw
xt = np.cross(yt, zt)
tho_without_magnitude = np.array([[0, 0, 0], xt, yt, zt])

A = np.tile(ij_point,4)
A_mat = np.reshape(A, (4, 3))
translate_tho_vector = tho_without_magnitude - A_mat
tho_translated = pd.DataFrame(data=translate_tho_vector, columns=["x", "y", "z"])
vectors = np.zeros((3, 3))

colour = ['red', 'green', 'blue']
for i in range(3):
    vectors[i, :] = translate_tho_vector[i + 1, :] - translate_tho_vector[0, :]

translation_vector = ij_point  # translation vector to global origin
translated_thorax = thorax_data[Keywords.vertices] - translation_vector  # translated  thorax to global origin
translated_ij_point = ij_point - translation_vector  # Origin (ij_point) ###IJ marker

source_points = (np.array(vectors)).T
target_points = np.array([[1,0,0], [0,1,0], [0,0,1]])
t_t_mat = Cloud.transform_between_3x3_points_sets(source_points, target_points)
torso_points = np.array(translated_thorax)
t_x = np.hstack((torso_points, np.ones((torso_points.shape[0], 1))))
transformed_torso = (t_t_mat @ t_x.T).T
transformed_torso3d = transformed_torso[:, :3]
t_transformed_vectors = (t_t_mat[:3, :3] @  vectors.T).T

## thorax markers
transformed_px = transform_point(px_point, translation_vector, t_t_mat)  # px marker
transformed_c7 = transform_point(c7_point, translation_vector, t_t_mat)  # c7 marker
transformed_t8 = transform_point(t8_point, translation_vector, t_t_mat)  # t8 marker

plotter = pv.Plotter()
# plotter.add_mesh(current_case, color='red')
plotter.add_mesh(transformed_torso3d, color='cyan')
plotter.add_points(translated_ij_point, color='yellow', point_size=20)
# plotter.add_points(most_negative_x_point, color='black', point_size=20)
for i, vector in enumerate(t_transformed_vectors):
    plotter.add_mesh(pv.Arrow(start=translated_ij_point, direction=vector, scale=20), color=colour[i])
# plotter.add_mesh(pv.Arrow(start=[0,0,0], direction=[1,0,0], scale=40), color='red')
# plotter.add_mesh(pv.Arrow(start=[0,0,0], direction=[0,1,0], scale=40), color='green')
# plotter.add_mesh(pv.Arrow(start=[0,0,0], direction=[0,0,1], scale=40), color='blue')
plotter.show()

# export mesh as ply and stl
ply_reader = vtk.vtkPLYReader()
ply_reader.SetFileName(ssm_thorax_mesh)
ply_reader.Update()
mesh_data = ply_reader.GetOutput()
torso_mesh = VTKMeshUtl.update_poly_w_points(transformed_torso3d, mesh_data)
out_path_mesh = output_path = os.path.join(articulation_folder, "Meshes")
VTKMeshUtl.write(os.path.join(out_path_mesh, "torso.ply"), torso_mesh)
VTKMeshUtl.write(os.path.join(out_path_mesh, "torso.stl"), torso_mesh)
# remesh torso for osim articulation gui
ms = pm.MeshSet()
input_mesh_path = (os.path.join(out_path_mesh, "torso.stl"))
ms.load_new_mesh(input_mesh_path)
b = pm.PureValue(4)
ms.meshing_isotropic_explicit_remeshing(iterations=10, targetlen=b, maxsurfdist=b)
output_mesh_path = os.path.join(out_path_mesh, "torso_RM.ply")
ms.save_current_mesh(output_mesh_path)
# scaling the remeshed torso
rm_torso_path = output_mesh_path
rm_torso_poly = VTKMeshUtl.load(rm_torso_path, True)
mesh_verts = VTKMeshUtl.extract_points(rm_torso_poly)
sf = 0.001
scaled_torso = mesh_verts * sf
torso_mesh1 = VTKMeshUtl.update_poly_w_points(scaled_torso, copy.deepcopy(rm_torso_poly))

VTKMeshUtl.write(os.path.join(out_path_mesh, "torso_rm_scaled.stl"), torso_mesh1)

# Right_Clavicle
cla_r_map = r_clav['idm'].to_list()
cla_r_data = {Keywords.vertices: current_case[cla_r_map, :]}
sc_r_point = np.mean(current_case[SC_r['idm'].to_list()], axis=0)
ac_r_point = np.mean(current_case[AC_r['idm'].to_list()], axis=0)
zc_raw = ac_r_point - sc_r_point
zc = (1 / np.linalg.norm(zc_raw)) * zc_raw
xc_raw = np.cross(yt, zc)
xc = (1 / np.linalg.norm(zc_raw)) * xc_raw
yc = np.cross(zc, xc)
cla_without_magnitude_R = np.array([[0, 0, 0], xc, yc, zc])

rc_SC = sphere_fit(current_case[ClaSCsphere_r['idm'].to_list()])
rc_IJ = sphere_fit(current_case[ThoIJsphere_r['idm'].to_list()])
rc_IJ_2 = rc_IJ - translation_vector
rc_IJ_3 = (t_t_mat[:3, :3] @ rc_IJ_2.T).T
SCjoint1 = np.squeeze(rc_IJ_3 - rc_SC)
translated_clavicle1 = (cla_r_data[Keywords.vertices]) + SCjoint1
A_mat = np.tile(SCjoint1, (4, 1))
translate_cla_vector = cla_without_magnitude_R - A_mat
cla_translated = pd.DataFrame(data=translate_cla_vector, columns=["x", "y", "z"])

# vectors
cvectors = np.zeros((3, 3))
colour = ['red', 'green', 'blue']
for i in range(3):
    cvectors[i, :] = translate_cla_vector[i + 1, :] - translate_cla_vector[0, :]

source_points_0 = (np.array(cvectors)).T
source_points_1 = source_points_0
a=np.linalg.norm(source_points_1, axis=0)
source_points = source_points_1/a

target_points = (np.array(t_transformed_vectors)).T
glo_origin = [0, 0, 0]
translate_clavicle_vector2 = glo_origin - rc_IJ_3
translated_clavicle2 = translated_clavicle1 + translate_clavicle_vector2  # clavicle translated to global
t_mat = Cloud.transform_between_3x3_points_sets(source_points, target_points)
clav_points = np.array(translated_clavicle2)
x = np.hstack((clav_points, np.ones((clav_points.shape[0], 1))))
transformed_clavicle = (t_mat @ x.T).T
transformed_clavicle3d = transformed_clavicle[:, :3]
transformed_vectors = (t_mat[:3, :3] @  cvectors.T).T
translated_clavicle4 = transformed_clavicle3d - translate_clavicle_vector2

transformed_clavicle_df = pd.DataFrame(translated_clavicle4, columns=['x', 'y', 'z'])
transformed_clavicle_df['idm'] = cla_r_map
df = transformed_clavicle_df
mapping_lib_df = pd.Series(df.index.values, index=df['idm']).to_dict()
cla_sc_points = SC_r['idm'].to_list()
cla_sc_idx = [mapping_lib_df[idm] for idm in cla_sc_points]
cla_sc = df.iloc[cla_sc_idx]
c_sc_points = cla_sc[['x', 'y', 'z']].to_numpy()
translated_sc_joint2 = np.array(rc_IJ_3).reshape(1, 3)
distances = np.linalg.norm(c_sc_points - rc_IJ_3, axis=1)
closest_index = np.argmin(distances)
closest_point = cla_sc.iloc[closest_index]
closest_point_xyz = closest_point[['x', 'y', 'z']].to_numpy()
translate_cla_vector2 = translated_sc_joint2 - closest_point_xyz
translated_clavicle5 = translated_clavicle4 + translate_cla_vector2
c_SCjoint = closest_point_xyz + translate_cla_vector2 # C_SC Marker
C_SC = c_SCjoint[0] # C_SC Marker

clav_to_og = glo_origin - c_SCjoint

# mapping muscles on r_cla
c_results = {}
mean_points = []
for file in os.listdir(r_cla_MAS_folder):
    if file.lower().endswith(".csv"):
        csv_path = os.path.join(r_cla_MAS_folder, file)
        basename = os.path.basename(csv_path)
        C_MAS = pd.read_csv(csv_path)
        C_MAS_coords = get_MAS_xyz(df, C_MAS)
        C_Muscle_Attachment_Sites = (C_MAS_coords + translate_cla_vector2)
        mean_C_MAS_xyz = np.mean(C_Muscle_Attachment_Sites, axis=0)
        c_results[basename] = mean_C_MAS_xyz

        # project onto bone (nearest node number)
        mas_id_list = C_MAS['idm'].to_list()
        mas_indices = [mapping_lib_df[idm] for idm in mas_id_list]
        mas_points_df = df.iloc[mas_indices]
        mas_points_xyz = mas_points_df[['x', 'y', 'z']].to_numpy()
        distances = np.linalg.norm(mas_points_xyz - mean_C_MAS_xyz, axis=1)
        closest_idx = np.argmin(distances)
        closest_point = mas_points_df.iloc[closest_idx]
        C_MAS_point = closest_point[['x', 'y', 'z']].to_numpy()
        c_OG_mean_xyz = (C_MAS_point - clav_to_og).flatten()
        mean_points.append(
            {'Muscle name': basename, 'mean_x': c_OG_mean_xyz[0], 'mean_y': c_OG_mean_xyz[1], 'mean_z': c_OG_mean_xyz[2]})

c_MAS_df = pd.DataFrame(mean_points)
out_path = (os.path.join(articulation_folder, 'patient_MAS', 'Cla_MAS.csv'))
c_MAS_df.to_csv(out_path, index=False)
points = np.array(list(c_results.values()))
cloud = pv.PolyData(points)
labels = list(c_results.keys())
cloud["labels"] = labels

plotter = pv.Plotter()
# thorax
# plotter.add_mesh(current_case, color='red')
plotter.add_mesh(transformed_torso3d, color='cyan')
for i, vector in enumerate(t_transformed_vectors):
    plotter.add_mesh(pv.Arrow(start=translated_ij_point, direction=vector, scale=20), color=colour[i])

# right clavicle
plotter.add_mesh(pv.PolyData(c_SCjoint), color='pink',render_points_as_spheres=True,  point_size=20)
plotter.add_points(translated_clavicle5, color='purple', point_size=10)
plotter.add_mesh(pv.PolyData(C_Muscle_Attachment_Sites), color='red', point_size=15)
for name, coord in c_results.items():
    plotter.add_point_labels(np.array([coord]), [name], point_size=0, font_size=12)
for i, cvector in enumerate(transformed_vectors):
    plotter.add_mesh(pv.Arrow(start=c_SCjoint, direction=cvector, scale=20), color=colour[i])

plotter.show()

##### export mesh as ply and stl for articulation
# translate clavicle to origin
clav_to_og = glo_origin - c_SCjoint
cla_mesh = translated_clavicle5 + clav_to_og
cla_mesh2 = translated_clavicle5
Origin_c_SCjoint = c_SCjoint + clav_to_og

ply_reader = vtk.vtkPLYReader()
ply_reader.SetFileName(ssm_cla_mesh)
ply_reader.Update()
mesh_data = ply_reader.GetOutput()
new_cla_mesh = VTKMeshUtl.update_poly_w_points(cla_mesh, mesh_data)
cla_mesh_verts = VTKMeshUtl.extract_points(new_cla_mesh)
scaled_r_cla = cla_mesh_verts * sf
cla_mesh1 = VTKMeshUtl.update_poly_w_points(scaled_r_cla, copy.deepcopy(new_cla_mesh))

VTKMeshUtl.write(os.path.join(out_path_mesh, "r_cla.ply"), new_cla_mesh)
VTKMeshUtl.write(os.path.join(out_path_mesh, "r_cla_scaled.stl"), cla_mesh1)

# save the meshes in isb coords
ply_reader = vtk.vtkPLYReader()
ply_reader.SetFileName(ssm_cla_mesh)
ply_reader.Update()
mesh_data = ply_reader.GetOutput()
new_cla_mesh2 = VTKMeshUtl.update_poly_w_points(translated_clavicle5, mesh_data)
cla_mesh_verts2 = VTKMeshUtl.extract_points(new_cla_mesh2)
scaled_r_cla2 = cla_mesh_verts2 * sf
cla_mesh1_2 = VTKMeshUtl.update_poly_w_points(scaled_r_cla2, copy.deepcopy(new_cla_mesh2))
VTKMeshUtl.write(os.path.join(out_path_mesh, "r_cla_isb.ply"), new_cla_mesh2)
VTKMeshUtl.write(os.path.join(out_path_mesh, "r_cla_scaled_isb.stl"), cla_mesh1_2)

# ISB coords for Left_Clavicle
cla_l_map = l_clav['idm'].to_list()
cla_l_data = {Keywords.vertices: current_case[cla_l_map, :]}
sc_point_L = np.mean(current_case[SC_l['idm'].to_list()], axis=0)
ac_point_L = np.mean(current_case[AC_l['idm'].to_list()], axis=0)
zc_raw_L = sc_point_L - ac_point_L
zc_L = (1 / np.linalg.norm(zc_raw_L)) * zc_raw_L
xc_raw_L = np.cross(yt, zc_L)
xc_L = (1 / np.linalg.norm(zc_raw_L)) * xc_raw_L
yc_L = np.cross(zc_L, xc_L)
cla_without_magnitude_L = np.array([[0, 0, 0], xc_L, yc_L, zc_L])

# sphere fitting
rc_SC_L = sphere_fit(current_case[ClaSCsphere_l['idm'].to_list()])
rc_IJ_L = sphere_fit(current_case[ThoIJsphere_L['idm'].to_list()])
rc_IJ_L2 = rc_IJ_L - translation_vector
rc_IJ_L3 = (t_t_mat[:3, :3] @ rc_IJ_L2.T).T
SCjoint_L1 = np.squeeze(rc_IJ_L3 - rc_SC_L)
translated_l_clavicle1 = (cla_l_data[Keywords.vertices]) + SCjoint_L1
A_mat_L = np.tile(SCjoint_L1, (4, 1))
translate_cla_vector_L = cla_without_magnitude_L - A_mat_L
cla_translated_L = pd.DataFrame(data=translate_cla_vector_L, columns=["x", "y", "z"])

# vectors
clvectors = np.zeros((3,3))
colour = ['red', 'green', 'blue']
for i in range(3):
    clvectors[i, :] = translate_cla_vector_L[i + 1, :] - translate_cla_vector_L[0, :]

cl_source_points_0 = (np.array(clvectors)).T  # left clavicle vectors
cl_source_points_1 = cl_source_points_0
al=np.linalg.norm(cl_source_points_1, axis=0)
cl_source_points = cl_source_points_1/al

target_points = (np.array(t_transformed_vectors)).T  # torso vectors
cl_translate_clavicle_vector2 = glo_origin - rc_IJ_L3
cl_translated_clavicle2 = translated_l_clavicle1 + cl_translate_clavicle_vector2
cl_t_mat = Cloud.transform_between_3x3_points_sets(cl_source_points, target_points)
cl_clav_points = np.array(cl_translated_clavicle2)
cl_x = np.hstack((cl_clav_points, np.ones((cl_clav_points.shape[0], 1))))
cl_transformed_clavicle = (cl_t_mat @ cl_x.T).T
cl_transformed_clavicle3d = cl_transformed_clavicle[:, :3]
cl_transformed_vectors = (cl_t_mat[:3, :3] @  clvectors.T).T
cl_translated_clavicle4 = cl_transformed_clavicle3d - cl_translate_clavicle_vector2

# mapping to translated clavicle left
cl_transformed_clavicle_df = pd.DataFrame(cl_translated_clavicle4, columns=['x', 'y', 'z'])
cl_transformed_clavicle_df['idm'] = cla_l_map
cl_df = cl_transformed_clavicle_df
cl_mapping_lib_df = pd.Series(cl_df.index.values, index=cl_df['idm']).to_dict()
cl_cla_sc_points = SC_l['idm'].to_list()
cl_cla_sc_idx = [cl_mapping_lib_df[idm] for idm in cl_cla_sc_points]
cl_cla_sc = cl_df.iloc[cl_cla_sc_idx]
cl_c_sc_points = cl_cla_sc[['x', 'y', 'z']].to_numpy()
cl_translated_sc_joint2 = np.array(rc_IJ_L3).reshape(1, 3)
dist = np.linalg.norm(cl_c_sc_points - rc_IJ_L3, axis=1)
cl_closest_index = np.argmin(dist)
cl_closest_point = cl_cla_sc.iloc[cl_closest_index]
cl_closest_point_xyz = cl_closest_point[['x', 'y', 'z']].to_numpy()
cl_translate_cla_vector2 = cl_translated_sc_joint2 - cl_closest_point_xyz
cl_translated_clavicle5 = cl_translated_clavicle4 + cl_translate_cla_vector2
cl_SCjoint = cl_closest_point_xyz + cl_translate_cla_vector2 # L_C_SC marker
L_C_SC = cl_SCjoint[0]

l_clav_to_og = glo_origin - cl_SCjoint

# mapping muscles on r_cla
cl_results = {}
mean_points = []
for file in os.listdir(l_cla_MAS_folder):
    if file.lower().endswith(".csv"):
        csv_path = os.path.join(l_cla_MAS_folder, file)
        basename = os.path.basename(csv_path)
        Cl_MAS = pd.read_csv(csv_path)
        Cl_MAS_coords = get_MAS_xyz(cl_df, Cl_MAS)
        Cl_Muscle_Attachment_Sites = (Cl_MAS_coords + cl_translate_cla_vector2)
        mean_Cl_MAS_xyz = np.mean(Cl_Muscle_Attachment_Sites, axis=0)
        cl_results[basename] = mean_Cl_MAS_xyz

        # project onto bone (nearest node number)
        mas_id_list = Cl_MAS['idm'].to_list()
        mas_indices = [cl_mapping_lib_df[idm] for idm in mas_id_list]
        mas_points_df = cl_df.iloc[mas_indices]
        mas_points_xyz = mas_points_df[['x', 'y', 'z']].to_numpy()
        distances = np.linalg.norm(mas_points_xyz - mean_Cl_MAS_xyz, axis=1)
        closest_idx = np.argmin(distances)
        closest_point = mas_points_df.iloc[closest_idx]
        Cl_MAS_point = closest_point[['x', 'y', 'z']].to_numpy()
        cl_OG_mean_xyz = (Cl_MAS_point - l_clav_to_og).flatten()
        mean_points.append(
            {'Muscle name': basename, 'mean_x': cl_OG_mean_xyz[0], 'mean_y': cl_OG_mean_xyz[1], 'mean_z': cl_OG_mean_xyz[2]})

cl_MAS_df = pd.DataFrame(mean_points)
out_path = os.path.join(articulation_folder, 'patient_MAS', 'Cla_L_MAS.csv')
cl_MAS_df.to_csv(out_path, index=False)
points = np.array(list(cl_results.values()))
cloud = pv.PolyData(points)
labels = list(cl_results.keys())
cloud["labels"] = labels
plotter = pv.Plotter()
# thorax
plotter.add_mesh(pv.PolyData(np.array([0, 0, 0])), color='green', render_points_as_spheres=True, point_size=20)  # Origin (ij_point)
plotter.add_mesh(transformed_torso3d, color='cyan')
for i, vector in enumerate(t_transformed_vectors):
    plotter.add_mesh(pv.Arrow(start=translated_ij_point, direction=vector, scale=20), color=colour[i])

# right clavicle
plotter.add_mesh(pv.PolyData(c_SCjoint), color='pink',render_points_as_spheres=True,  point_size=20)
plotter.add_mesh(pv.PolyData(translated_clavicle5), color='purple')
for i, tcvector in enumerate(transformed_vectors):
    plotter.add_mesh(pv.Arrow(start=c_SCjoint, direction=tcvector, scale=20), color=colour[i])

# left clavicle
plotter.add_mesh(pv.PolyData(cl_SCjoint), color='yellow', render_points_as_spheres=True, point_size=20)
plotter.add_points(pv.PolyData(cl_translated_clavicle5), color='pink')
plotter.add_mesh(pv.PolyData(Cl_Muscle_Attachment_Sites), color='red', point_size=15)
for name, coord in cl_results.items():
    plotter.add_point_labels(np.array([coord]), [name], point_size=0, font_size=12)
for i,clvector in enumerate(cl_transformed_vectors):
    plotter.add_mesh(pv.Arrow(start=cl_SCjoint, direction=clvector, scale=20), color=colour[i])
plotter.show()
##### export mesh as ply and stl for articulation
# translate clavicle to origin
l_clav_to_og = glo_origin - cl_SCjoint
l_cla_mesh = cl_translated_clavicle5 + l_clav_to_og
l_Origin_cl_SCjoint = cl_SCjoint + l_clav_to_og

ply_reader = vtk.vtkPLYReader()
ply_reader.SetFileName(l_ssm_cla_mesh)
ply_reader.Update()
mesh_data = ply_reader.GetOutput()
l_new_cla_mesh = VTKMeshUtl.update_poly_w_points(l_cla_mesh, mesh_data)
l_cla_mesh_verts = VTKMeshUtl.extract_points(l_new_cla_mesh)
l_scaled_cla = l_cla_mesh_verts * sf
l_cla_mesh1 = VTKMeshUtl.update_poly_w_points(l_scaled_cla, copy.deepcopy(l_new_cla_mesh))
VTKMeshUtl.write(os.path.join(out_path_mesh, "l_cla.ply"), l_new_cla_mesh)
VTKMeshUtl.write(os.path.join(out_path_mesh, "l_cla_scaled.stl"), l_cla_mesh1)

# save the meshes in isb coords
ply_reader = vtk.vtkPLYReader()
ply_reader.SetFileName(l_ssm_cla_mesh)
ply_reader.Update()
mesh_data = ply_reader.GetOutput()
l_new_cla_mesh_2 = VTKMeshUtl.update_poly_w_points(cl_translated_clavicle5, mesh_data)
l_cla_mesh_verts_2 = VTKMeshUtl.extract_points(l_new_cla_mesh_2)
l_scaled_cla_2 = l_cla_mesh_verts_2 * sf
l_cla_mesh1_2 = VTKMeshUtl.update_poly_w_points(l_scaled_cla_2, copy.deepcopy(l_new_cla_mesh_2))
VTKMeshUtl.write(os.path.join(out_path_mesh, "l_cla_isb.ply"), l_new_cla_mesh_2)
VTKMeshUtl.write(os.path.join(out_path_mesh, "l_cla_scaled_isb.stl"), l_cla_mesh1_2)

# ISB coords for Right_Scapula
sca_r_map = r_scap['idm'].to_list()
sca_r_data = {Keywords.vertices: current_case[sca_r_map, :]}
aa_point = np.mean(current_case[AA_r['idm'].to_list()], axis=0)
ai_point = np.mean(current_case[AI_r['idm'].to_list()], axis=0)
ts_point = np.mean(current_case[TS_r['idm'].to_list()], axis=0)
sa_point = np.mean(current_case[SA_r['idm'].to_list()], axis=0)
zs_raw = aa_point - ts_point
zs = (1 / np.linalg.norm(zs_raw)) * zs_raw
yx1_raw = ts_point - aa_point
yx2_raw = ai_point - aa_point
xs_raw = np.cross(yx2_raw, yx1_raw)
xs = (1 / np.linalg.norm(xs_raw)) * xs_raw
ys = np.cross(zs, xs)
sca_without_magnitude = np.array([[0, 0, 0], xs, ys, zs])

cla_ac_mean_point = np.mean(current_case[AC_r['idm'].to_list()], axis=0)

# Mapping ac landmarks on translated_clavicle5
translated_clavicle5_df = pd.DataFrame(translated_clavicle5, columns=['x', 'y', 'z'])
translated_clavicle5_df['idm'] = cla_r_map
trans_cla5_df = translated_clavicle5_df
cla5_mapping_lib_df = pd.Series(trans_cla5_df.index.values, index=trans_cla5_df['idm']).to_dict()
cla_ac_points = AC_r['idm'].to_list()
cla_ac_idx = [cla5_mapping_lib_df[idm] for idm in cla_ac_points]
cla_ac = trans_cla5_df.iloc[cla_ac_idx]
c_ac_points = cla_ac[['x', 'y', 'z']].to_numpy()
cla_ac_mean_point2 = np.array(cla_ac_mean_point).reshape(1, 3)
cla_ac_dist = np.linalg.norm(c_ac_points - cla_ac_mean_point, axis=1)
cla_ac_closest_idx = np.argmin(cla_ac_dist)
cla_ac_closest_point = cla_ac.iloc[cla_ac_closest_idx]
cla_ac_closest_point_xyz = cla_ac_closest_point[['x', 'y', 'z']].to_numpy()
cla_ac_point = cla_ac_closest_point_xyz
C_AC = cla_ac_point # C_AC Marker

# mapping ac landmark on scapula
sca_ac_mean_point = np.mean(current_case[sca_AC_r['idm'].to_list()], axis=0)
scapula_xyz_df = pd.DataFrame(sca_r_data[Keywords.vertices], columns=['x', 'y', 'z'])
scapula_xyz_df['idm'] = sca_r_map
sca_df = scapula_xyz_df
sca_mapping_lib_df = pd.Series(sca_df.index.values, index=sca_df['idm']).to_dict()
sca_ac_points = sca_AC_r['idm'].to_list()
sca_ac_idx = [sca_mapping_lib_df[idm] for idm in sca_ac_points]
sca_ac = sca_df.iloc[sca_ac_idx]
s_ac_points = sca_ac[['x', 'y', 'z']].to_numpy()
sca_ac_mean_point2 = np.array(sca_ac_mean_point).reshape(1, 3)
sca_ac_dist = np.linalg.norm(s_ac_points - sca_ac_mean_point, axis=1)
sca_ac_closest_idx = np.argmin(sca_ac_dist)
sca_ac_closest_point = sca_ac.iloc[sca_ac_closest_idx]
sca_ac_closest_point_xyz = sca_ac_closest_point[['x', 'y', 'z']].to_numpy()
sca_ac_point = sca_ac_closest_point_xyz

# translating scapula vertices to scapula origin (ac) and creating the vectors for scapula
sca_translation_vector = cla_ac_point - sca_ac_point
ACjoint = sca_ac_point
translated_ACjoint = ACjoint + sca_translation_vector
translated_scapula = (sca_r_data[Keywords.vertices]) + sca_translation_vector
A_mat = np.tile(ACjoint, (4, 1))
translate_sca_vector = sca_without_magnitude - A_mat
sca_translated = pd.DataFrame(data=translate_sca_vector, columns=["x", "y", "z"])

# vectors
svectors = np.zeros((3, 3))
colour = ['red', 'green', 'blue']
for i in range(3):
    svectors[i, :] = translate_sca_vector[i + 1, :] - translate_sca_vector[0, :]

# translate scapula to global origin
glo_origin = [0, 0, 0]
translate_scapula_vector = glo_origin - translated_ACjoint   # vector to translate translated scapula to global origin
translated_scapula2 = translated_scapula + translate_scapula_vector
translated_ACjoint2 = translated_ACjoint + translate_scapula_vector

# transform scapula
sca_source_points = (np.array(svectors)).T
sca_t_mat = Cloud.transform_between_3x3_points_sets(sca_source_points, target_points)
scap_points = np.array(translated_scapula2)
sca_x = np.hstack((scap_points, np.ones((scap_points.shape[0], 1))))
transformed_scapula = (sca_t_mat @ sca_x.T).T
transformed_scapula3d = transformed_scapula[:, :3]
sca_transformed_vectors = (sca_t_mat[:3, :3] @  svectors.T).T
translated_scapula3 = transformed_scapula3d - translate_scapula_vector # translate scapula to transformed clav
translated_ACjoint3 = translated_ACjoint2 - translate_scapula_vector

# mapping markers on transformed r_scapula
s_cap_mean_point = np.mean(current_case[S_CAP['idm'].to_list()], axis=0)
s_sa_mean_point = np.mean(current_case[S_SA['idm'].to_list()], axis=0)
s_ia_mean_point = np.mean(current_case[S_IA['idm'].to_list()], axis=0)
s_ts_mean_point = np.mean(current_case[S_TS['idm'].to_list()], axis=0)
s_cap_point = get_closest_point(sca_df, S_CAP['idm'].to_list(), s_cap_mean_point)
s_sa_point = get_closest_point(sca_df, S_SA['idm'].to_list(), s_sa_mean_point)
s_ia_point = get_closest_point(sca_df, S_IA['idm'].to_list(), s_ia_mean_point)
s_ts_point = get_closest_point(sca_df, S_TS['idm'].to_list(), s_ts_mean_point)
tsl_cap = (s_cap_point + sca_translation_vector) + translate_scapula_vector
tsl_sa = (s_sa_point + sca_translation_vector) + translate_scapula_vector
tsl_ia = (s_ia_point + sca_translation_vector) + translate_scapula_vector
tsl_ts = (s_ts_point + sca_translation_vector) + translate_scapula_vector
cap_point = np.append(tsl_cap, 1)
sa_point = np.append(tsl_sa, 1)
ia_point = np.append(tsl_ia, 1)
ts_point = np.append(tsl_ts, 1)
tsf_cap = (sca_t_mat @ cap_point.T).T
tsf_sa = (sca_t_mat @ sa_point.T).T
tsf_ia = (sca_t_mat @ ia_point.T).T
tsf_ts = (sca_t_mat @ ts_point.T).T
R_CAP = (tsf_cap[:3] - translate_scapula_vector)  # CAP Marker
R_SA = (tsf_sa[:3] - translate_scapula_vector)  # SA Marker
R_IA = (tsf_ia[:3] - translate_scapula_vector)  # IA Marker
R_TS = (tsf_ts[:3] - translate_scapula_vector)  # TS Marker

sca_to_og = glo_origin - translated_ACjoint3

results = {}
mean_points = []
for file in os.listdir(r_scap_MAS_folder):
    if file.lower().endswith(".csv"):
        csv_path = os.path.join(r_scap_MAS_folder, file)
        basename = os.path.basename(csv_path)
        S_MAS = pd.read_csv(csv_path)
        s_MAS_coords = get_MAS_xyz(sca_df, S_MAS)
        tsl_s_MAS = (s_MAS_coords + sca_translation_vector) + translate_scapula_vector
        s_MAS_x = np.hstack((tsl_s_MAS, np.ones((tsl_s_MAS.shape[0], 1))))
        tsf_s_MAS = (sca_t_mat @ s_MAS_x.T).T
        tsf_s_MAS_3d = tsf_s_MAS[:, :3]
        S_Muscle_Attachment_Sites = tsf_s_MAS_3d - translate_scapula_vector
        mean_S_MAS_xyz = np.mean(S_Muscle_Attachment_Sites, axis=0)
        results[basename] = mean_S_MAS_xyz

        # project onto bone (nearest node number)
        mas_id_list = S_MAS['idm'].to_list()
        mas_indices = [sca_mapping_lib_df[idm] for idm in mas_id_list]
        mas_points_df = sca_df.iloc[mas_indices]
        mas_points_xyz = mas_points_df[['x', 'y', 'z']].to_numpy()
        distances = np.linalg.norm(mas_points_xyz - mean_S_MAS_xyz, axis=1)
        closest_idx = np.argmin(distances)
        closest_point = mas_points_df.iloc[closest_idx]
        S_MAS_point = closest_point[['x', 'y', 'z']].to_numpy()
        OG_mean_xyz = S_MAS_point - sca_to_og
        mean_points.append({'Muscle name': basename, 'mean_x': OG_mean_xyz[0], 'mean_y': OG_mean_xyz[1], 'mean_z': OG_mean_xyz[2]})

s_MAS_df = pd.DataFrame(mean_points)
out_path = (os.path.join(articulation_folder, 'patient_MAS', 'Sca_MAS.csv'))
s_MAS_df.to_csv(out_path, index=False)
points = np.array(list(results.values()))
cloud = pv.PolyData(points)
labels = list(results.keys())
cloud["labels"] = labels

plotter = pv.Plotter()
# thorax
plotter.add_mesh(transformed_torso3d, color='cyan')
for i, vector in enumerate(t_transformed_vectors):
    plotter.add_mesh(pv.Arrow(start=translated_ij_point, direction=vector, scale=20), color=colour[i])

# right clavicle
plotter.add_mesh(pv.PolyData(c_SCjoint), color='pink',render_points_as_spheres=True,  point_size=20)
plotter.add_mesh(pv.PolyData(cla_ac_point), color='yellow', point_size=20) # C_AC Marker
plotter.add_mesh(pv.PolyData(translated_clavicle5), color='purple')
for i, tcvector in enumerate(transformed_vectors):
    plotter.add_mesh(pv.Arrow(start=c_SCjoint, direction=tcvector, scale=20), color=colour[i])

# left clavicle
plotter.add_mesh(pv.PolyData(cl_SCjoint), color='yellow', render_points_as_spheres=True, point_size=20)
plotter.add_mesh(pv.PolyData(cl_translated_clavicle5), color='pink')
for i,clvector in enumerate(cl_transformed_vectors):
    plotter.add_mesh(pv.Arrow(start=cl_SCjoint, direction=clvector, scale=20), color=colour[i])

# right scapula
plotter.add_mesh(pv.PolyData(translated_ACjoint3), color='blue', render_points_as_spheres=True, point_size=5)
plotter.add_mesh(pv.PolyData(translated_scapula3), color='orange')
plotter.add_mesh(pv.PolyData(S_Muscle_Attachment_Sites), color='red', point_size=15)
plotter.add_mesh(cloud, point_size=40, color='pink')
for name, coord in results.items():
    plotter.add_point_labels(np.array([coord]), [name], point_size=0, font_size=12)

plotter.add_mesh(pv.PolyData(R_CAP), color='yellow', point_size=20)
plotter.add_mesh(pv.PolyData(R_SA), color='yellow', point_size=20)
plotter.add_mesh(pv.PolyData(R_IA), color='yellow', point_size=20)
plotter.add_mesh(pv.PolyData(R_TS), color='yellow', point_size=20)
for i,svector in enumerate(sca_transformed_vectors):
    plotter.add_mesh(pv.Arrow(start=translated_ACjoint3, direction=svector, scale=20), color=colour[i])
plotter.show()

##### export mesh as ply and stl for articulation
# translate scapula to origin

sca_mesh = translated_scapula3 + sca_to_og
Origin_ACjoint = translated_ACjoint3 + sca_to_og
Origin_MAS = S_Muscle_Attachment_Sites + sca_to_og

ply_reader = vtk.vtkPLYReader()
ply_reader.SetFileName(ssm_SCA_mesh)
ply_reader.Update()
mesh_data = ply_reader.GetOutput()
new_sca_mesh = VTKMeshUtl.update_poly_w_points(sca_mesh, mesh_data)
sca_mesh_verts = VTKMeshUtl.extract_points(new_sca_mesh)
scaled_r_sca = sca_mesh_verts * sf
sca_mesh1 = VTKMeshUtl.update_poly_w_points(scaled_r_sca, copy.deepcopy(new_sca_mesh))
VTKMeshUtl.write(os.path.join(out_path_mesh, "r_sca.ply"), new_sca_mesh)
VTKMeshUtl.write(os.path.join(out_path_mesh, "r_sca_scaled.stl"), sca_mesh1)

# save the meshes in isb coords
ply_reader = vtk.vtkPLYReader()
ply_reader.SetFileName(ssm_SCA_mesh)
ply_reader.Update()
mesh_data = ply_reader.GetOutput()
new_sca_mesh2 = VTKMeshUtl.update_poly_w_points(translated_scapula3, mesh_data)
sca_mesh_verts2 = VTKMeshUtl.extract_points(new_sca_mesh2)
scaled_r_sca2 = sca_mesh_verts2 * sf
sca_mesh1_2 = VTKMeshUtl.update_poly_w_points(scaled_r_sca2, copy.deepcopy(new_sca_mesh2))
VTKMeshUtl.write(os.path.join(out_path_mesh, "r_sca_isb.ply"), new_sca_mesh2)
VTKMeshUtl.write(os.path.join(out_path_mesh, "r_sca_scaled_isb.stl"), sca_mesh1_2)

# ISB coords for Left_Scapula
sca_l_map = l_scap['idm'].to_list()
sca_l_data = {Keywords.vertices: current_case[sca_l_map, :]}
aa_point_L = np.mean(current_case[AA_l['idm'].to_list()], axis=0)
ai_point_L = np.mean(current_case[AI_l['idm'].to_list()], axis=0)
ts_point_L = np.mean(current_case[TS_l['idm'].to_list()], axis=0)
sa_point_L = np.mean(current_case[SA_l['idm'].to_list()], axis=0)
zs_raw_L = ts_point_L - aa_point_L
zs_L = (1 / np.linalg.norm(zs_raw_L)) * zs_raw_L
yx1_raw_L = ts_point_L - aa_point_L
yx2_raw_L = ai_point_L - aa_point_L
xs_raw_L = np.cross(yx1_raw_L, yx2_raw_L)
xs_L = (1 / np.linalg.norm(xs_raw_L)) * xs_raw_L
ys_L = np.cross(zs_L, xs_L)
sca_without_magnitude_L = np.array([[0, 0, 0], xs_L, ys_L, zs_L])

# mapping left ac landmarks on  cl_translated_clavicle5
cl_cla_ac_mean_point = np.mean(current_case[AC_l['idm'].to_list()], axis=0)
cl_translated_clavicle5_df = pd.DataFrame(cl_translated_clavicle5, columns=['x', 'y', 'z'])
cl_translated_clavicle5_df['idm'] = cla_l_map
cl_trans_cla5_df = cl_translated_clavicle5_df
cl_cla5_mapping_lib_df = pd.Series(cl_trans_cla5_df.index.values, index=cl_trans_cla5_df['idm'])
cl_cla_ac_points = AC_l['idm'].to_list()
cl_cla_ac_idx = [cl_cla5_mapping_lib_df[idm] for idm in cl_cla_ac_points]
cl_cla_ac = cl_trans_cla5_df.iloc[cl_cla_ac_idx]
cl_c_ac_points = cl_cla_ac[['x', 'y', 'z']].to_numpy()
cl_cla_ac_mean_point2 = np.array(cl_cla_ac_mean_point).reshape(1, 3)
cl_cla_ac_dist = np.linalg.norm(cl_c_ac_points - cl_cla_ac_mean_point, axis=1)
cl_cla_ac_closest_idx = np.argmin(cl_cla_ac_dist)
cl_cla_ac_closest_point = cl_cla_ac.iloc[cl_cla_ac_closest_idx]
cl_cla_ac_closest_point_xyz = cl_cla_ac_closest_point[['x', 'y', 'z']].to_numpy()
cl_cla_ac_point = cl_cla_ac_closest_point_xyz
L_C_AC = cl_cla_ac_point # L_C_AC marker

# mapping ac landmark on left scapula
sl_sca_ac_mean_point = np.mean(current_case[sca_AC_l['idm'].to_list()], axis=0)
sl_scapula_xyz_df = pd.DataFrame(sca_l_data[Keywords.vertices], columns=['x', 'y', 'z'])
sl_scapula_xyz_df['idm'] = sca_l_map
sl_sca_df = sl_scapula_xyz_df
sl_sca_mapping_lib_df = pd.Series(sl_sca_df.index.values, index=sl_sca_df['idm']).to_dict()
sl_sca_ac_points = sca_AC_l['idm'].to_list()
sl_sca_ac_idx = [sl_sca_mapping_lib_df[idm] for idm in sl_sca_ac_points]
sl_sca_ac = sl_sca_df.iloc[sl_sca_ac_idx]
l_s_ac_points = sl_sca_ac[['x', 'y', 'z']].to_numpy()
sl_sca_ac_mean_point2 = np.array(sl_sca_ac_mean_point).reshape(1, 3)
sl_sca_ac_dist = np.linalg.norm(l_s_ac_points - sl_sca_ac_mean_point, axis=1)
sl_sca_ac_closest_idx = np.argmin(sl_sca_ac_dist)
sl_sca_ac_closest_point = sl_sca_ac.iloc[sl_sca_ac_closest_idx]
sl_sca_ac_closest_point_xyz = sl_sca_ac_closest_point[['x', 'y', 'z']].to_numpy()
sl_sca_ac_point = sl_sca_ac_closest_point_xyz

# joint center and translation
l_sca_translation_vector = cl_cla_ac_point - sl_sca_ac_point
ACjoint_L = sl_sca_ac_point
l_translated_ACjoint = ACjoint_L + l_sca_translation_vector
l_translated_scapula = (sca_l_data[Keywords.vertices]) + l_sca_translation_vector
A_mat_L = np.tile(ACjoint_L, (4, 1))
translate_sca_vector_L = sca_without_magnitude_L - A_mat_L
sca_translated_L = pd.DataFrame(data=translate_sca_vector_L, columns=["x", "y", "z"])

# vectors
slvectors = np.zeros((3, 3))
colour = ['red', 'green', 'blue']
for i in range(3):
    slvectors[i, :] = translate_sca_vector_L[i + 1, :] - translate_sca_vector_L[0, :]

# translate left scapula to global origin
l_translate_scapula_vector = glo_origin - l_translated_ACjoint
l_translated_scapula2 = l_translated_scapula + l_translate_scapula_vector
l_translated_ACjoint2 = l_translated_ACjoint + l_translate_scapula_vector

# transform left scapula
l_sca_source_points = (np.array(slvectors)).T
l_sca_t_mat = Cloud.transform_between_3x3_points_sets(l_sca_source_points, target_points)
l_scap_points = np.array(l_translated_scapula2)
l_sca_x = np.hstack((l_scap_points, np.ones((l_scap_points.shape[0], 1))))
l_transformed_scapula = (l_sca_t_mat @ l_sca_x.T).T
l_transformed_scapula3d = l_transformed_scapula[:, :3]
l_sca_transformed_vectors = (l_sca_t_mat[:3, :3] @  slvectors.T).T
l_translated_scapula3 = l_transformed_scapula3d - l_translate_scapula_vector
l_translated_ACjoint3 = l_translated_ACjoint2 - l_translate_scapula_vector

# mapping markers on transformed l_scapula
sl_cap_mean_point = np.mean(current_case[L_S_CAP['idm'].to_list()], axis=0)
sl_sa_mean_point = np.mean(current_case[L_S_SA['idm'].to_list()], axis=0)
sl_ia_mean_point = np.mean(current_case[L_S_IA['idm'].to_list()], axis=0)
sl_ts_mean_point = np.mean(current_case[L_S_TS['idm'].to_list()], axis=0)
sl_cap_point = get_closest_point(sl_sca_df, L_S_CAP['idm'].to_list(), sl_cap_mean_point)
sl_sa_point = get_closest_point(sl_sca_df, L_S_SA['idm'].to_list(), sl_sa_mean_point)
sl_ia_point = get_closest_point(sl_sca_df, L_S_IA['idm'].to_list(), sl_ia_mean_point)
sl_ts_point = get_closest_point(sl_sca_df, L_S_TS['idm'].to_list(), sl_ts_mean_point)
l_tsl_cap = (sl_cap_point + l_sca_translation_vector) + l_translate_scapula_vector
l_sa_cap = (sl_sa_point + l_sca_translation_vector) + l_translate_scapula_vector
l_ia_cap = (sl_ia_point + l_sca_translation_vector) + l_translate_scapula_vector
l_ts_cap = (sl_ts_point + l_sca_translation_vector) + l_translate_scapula_vector
l_cap_point = np.append(l_tsl_cap, 1)
l_sa_point = np.append(l_sa_cap, 1)
l_ia_point = np.append(l_ia_cap, 1)
l_ts_point = np.append(l_ts_cap, 1)
l_tsf_cap = (l_sca_t_mat @ l_cap_point.T).T
l_tsf_sa = (l_sca_t_mat @ l_sa_point.T).T
l_tsf_ia = (l_sca_t_mat @ l_ia_point.T).T
l_tsf_ts = (l_sca_t_mat @ l_ts_point.T).T
L_CAP = (l_tsf_cap[:3] - l_translate_scapula_vector)  # L_CAP Marker
L_SA = (l_tsf_sa[:3] - l_translate_scapula_vector)  # L_SA Marker
L_IA = (l_tsf_ia[:3] - l_translate_scapula_vector)  # L_IA Marker
L_TS = (l_tsf_ts[:3] - l_translate_scapula_vector)  # L_CAP Marker

l_sca_to_og = glo_origin - l_translated_ACjoint3

sl_results = {}
sl_mean_points = []
for file in os.listdir(l_scap_MAS_folder):
    if file.lower().endswith(".csv"):
        csv_path = os.path.join(l_scap_MAS_folder, file)
        basename = os.path.basename(csv_path)
        L_S_MAS = pd.read_csv(csv_path)
        l_s_MAS_coords = get_MAS_xyz(sl_sca_df, L_S_MAS)
        tsl_sl_MAS = (l_s_MAS_coords + l_sca_translation_vector) + l_translate_scapula_vector
        sl_MAS_x = np.hstack((tsl_sl_MAS, np.ones((tsl_sl_MAS.shape[0], 1))))
        tsf_sl_MAS = (l_sca_t_mat @ sl_MAS_x.T).T
        tsf_sl_MAS_3d = tsf_sl_MAS[:, :3]
        L_S_Muscle_Attachment_Sites = tsf_sl_MAS_3d - l_translate_scapula_vector
        mean_S_L_MAS_xyz = np.mean(L_S_Muscle_Attachment_Sites, axis=0)
        sl_results[basename] = mean_S_L_MAS_xyz

        # project onto bone (nearest node number)
        sl_mas_id_list = L_S_MAS['idm'].to_list()
        sl_mas_indices = [sl_sca_mapping_lib_df[idm] for idm in sl_mas_id_list]
        sl_mas_points_df = sl_sca_df.iloc[sl_mas_indices]
        sl_mas_points_xyz = sl_mas_points_df[['x', 'y', 'z']].to_numpy()
        sl_distances = np.linalg.norm(sl_mas_points_xyz - mean_S_L_MAS_xyz, axis=1)
        sl_closest_idx = np.argmin(sl_distances)
        sl_closest_point = sl_mas_points_df.iloc[sl_closest_idx]
        SL_MAS_point = sl_closest_point[['x', 'y', 'z']].to_numpy()
        sl_OG_mean_xyz = SL_MAS_point - l_sca_to_og
        sl_mean_points.append(
            {'Muscle name': basename, 'mean_x': sl_OG_mean_xyz[0], 'mean_y': sl_OG_mean_xyz[1], 'mean_z': sl_OG_mean_xyz[2]})

sl_MAS_df = pd.DataFrame(sl_mean_points)
out_path = os.path.join(articulation_folder, 'patient_MAS', 'Sca_L_mas.csv')
sl_MAS_df.to_csv(out_path, index=False)
sl_points = np.array(list(sl_results.values()))
sl_cloud = pv.PolyData(sl_points)
sl_labels = list(sl_results.keys())
sl_cloud["labels"] = sl_labels
plotter = pv.Plotter()

# thorax
plotter.add_mesh(transformed_torso3d, color='cyan')
for i, vector in enumerate(t_transformed_vectors):
    plotter.add_mesh(pv.Arrow(start=translated_ij_point, direction=vector, scale=20), color=colour[i])

# right clavicle
plotter.add_mesh(pv.PolyData(c_SCjoint), color='pink',render_points_as_spheres=True,  point_size=20)
plotter.add_mesh(pv.PolyData(translated_clavicle5), color='purple')
for i, tcvector in enumerate(transformed_vectors):
    plotter.add_mesh(pv.Arrow(start=c_SCjoint, direction=tcvector, scale=20), color=colour[i])

# left clavicle
plotter.add_mesh(pv.PolyData(cl_SCjoint), color='yellow', render_points_as_spheres=True, point_size=20)
plotter.add_mesh(pv.PolyData(cl_translated_clavicle5), color='pink')
for i,clvector in enumerate(cl_transformed_vectors):
    plotter.add_mesh(pv.Arrow(start=cl_SCjoint, direction=clvector, scale=20), color=colour[i])

# right scapula
plotter.add_mesh(pv.PolyData(translated_ACjoint3), color='blue', render_points_as_spheres=True, point_size=5)
plotter.add_mesh(pv.PolyData(translated_scapula3), color='orange')
for i,svector in enumerate(sca_transformed_vectors):
    plotter.add_mesh(pv.Arrow(start=translated_ACjoint3, direction=svector, scale=20), color=colour[i])

# left scapula
plotter.add_mesh(pv.PolyData(l_translated_ACjoint3), color='blue', render_points_as_spheres=True, point_size=5)
plotter.add_mesh(pv.PolyData(l_translated_scapula3), color='red')
plotter.add_mesh(pv.PolyData(L_S_Muscle_Attachment_Sites), color='red')
plotter.add_mesh(sl_cloud, point_size=40, color='pink')
for name, coord in sl_results.items():
    plotter.add_point_labels(np.array([coord]), [name], point_size=0, font_size=12)

plotter.add_mesh(pv.PolyData(L_CAP), color='yellow', point_size=20)
plotter.add_mesh(pv.PolyData(L_SA), color='yellow', point_size=20)
plotter.add_mesh(pv.PolyData(L_IA), color='yellow', point_size=20)
plotter.add_mesh(pv.PolyData(L_TS), color='yellow', point_size=20)
for i,slvector in enumerate(l_sca_transformed_vectors):
    plotter.add_mesh(pv.Arrow(start=l_translated_ACjoint3, direction=slvector, scale=20), color=colour[i])

plotter.show()

##### export mesh as ply and stl for articulation
# translate scapula to origin

l_sca_mesh = l_translated_scapula3 + l_sca_to_og
l_Origin_ACjoint = l_translated_ACjoint3 + l_sca_to_og

ply_reader = vtk.vtkPLYReader()
ply_reader.SetFileName(l_ssm_SCA_mesh)
ply_reader.Update()
mesh_data = ply_reader.GetOutput()
l_new_sca_mesh = VTKMeshUtl.update_poly_w_points(l_sca_mesh, mesh_data)
l_sca_mesh_verts = VTKMeshUtl.extract_points(l_new_sca_mesh)
l_scaled_r_sca = l_sca_mesh_verts * sf
l_sca_mesh1 = VTKMeshUtl.update_poly_w_points(l_scaled_r_sca, copy.deepcopy(l_new_sca_mesh))
VTKMeshUtl.write(os.path.join(out_path_mesh, "l_sca.ply"), l_new_sca_mesh)
VTKMeshUtl.write(os.path.join(out_path_mesh, "l_sca_scaled.stl"), l_sca_mesh1)

# save the meshes in isb coords
ply_reader = vtk.vtkPLYReader()
ply_reader.SetFileName(l_ssm_SCA_mesh)
ply_reader.Update()
mesh_data = ply_reader.GetOutput()
l_new_sca_mesh_2 = VTKMeshUtl.update_poly_w_points(l_translated_scapula3, mesh_data)
l_sca_mesh_verts_2 = VTKMeshUtl.extract_points(l_new_sca_mesh_2)
l_scaled_sca_2 = l_sca_mesh_verts_2 * sf
l_sca_mesh1_2 = VTKMeshUtl.update_poly_w_points(l_scaled_sca_2, copy.deepcopy(l_new_sca_mesh_2))
VTKMeshUtl.write(os.path.join(out_path_mesh, "l_sca_isb.ply"), l_new_sca_mesh_2)
VTKMeshUtl.write(os.path.join(out_path_mesh, "l_sca_scaled_isb.stl"), l_sca_mesh1_2)

# Right_Humerus
hum_r_map = r_hum['idm'].to_list()
hum_r_data = {Keywords.vertices: current_case[hum_r_map, :]}
rc = np.squeeze(sphere_fit(current_case[humeral_head_r['idm'].to_list()]))
el_point = np.nanmean(current_case[EL_r['idm'].to_list()], axis=0)
em_point = np.nanmean(current_case[EM_r['idm'].to_list()], axis=0)
mid_ep = 0.5 * (em_point + el_point)
# calculate coord system
yh_raw = rc - mid_ep
yh = (1 / np.linalg.norm(yh_raw)) * yh_raw
X0 = em_point - rc
X1 = el_point - rc
xh_raw = np.cross(X1, X0)
xh = (1 / np.linalg.norm(xh_raw)) * xh_raw
zh = np.cross(xh, yh)
hum_without_magnitude = np.array([[0, 0, 0], xh, yh, zh])

rc_scap = sphere_fit(current_case[glenoid_r['idm'].to_list()])
rc_scap_2 = rc_scap + sca_translation_vector
rc_scap_3 = rc_scap_2 + translate_scapula_vector
rc_scap_4 = (sca_t_mat[:3, :3] @ rc_scap_3.T).T
rc_scap_5 = rc_scap_4 - translate_scapula_vector
rc_hum = sphere_fit(current_case[humeral_head_r['idm'].to_list()])
GHjoint1 = np.squeeze(rc_scap_5 - rc_hum)
translated_humerus = hum_r_data[Keywords.vertices] + GHjoint1
A_mat = np.tile(GHjoint1, (4, 1))
translate_hum_vector = hum_without_magnitude - A_mat

hum_translated = pd.DataFrame(data=translate_hum_vector, columns=["x", "y", "z"])
hvectors = np.zeros((3, 3))
colour = ['red', 'green', 'blue']
for i in range(3):
    hvectors[i, :] = translate_hum_vector[i + 1, :] - translate_hum_vector[0, :]

translate_humerus_vector = glo_origin - rc_scap_5
translated_humerus2 = translated_humerus + translate_humerus_vector # translate humerus to glo origin
hum_source_points = (np.array(hvectors)).T
hum_t_mat = Cloud.transform_between_3x3_points_sets(hum_source_points, target_points)
hum_points = np.array(translated_humerus2)
hum_x = np.hstack((hum_points, np.ones((hum_points.shape[0], 1))))
transformed_humerus = (hum_t_mat @ hum_x.T).T
transformed_humerus3d = transformed_humerus[:, :3]
hum_transformed_vectors = (hum_t_mat[:3, :3] @ hvectors.T).T
translated_humerus3 = transformed_humerus3d - translate_humerus_vector

# mapping markers on transformed r_humerus
hum_xyz_df = pd.DataFrame(hum_r_data[Keywords.vertices], columns=['x', 'y', 'z'])
hum_xyz_df['idm'] = hum_r_map
hum_df = hum_xyz_df
hum_mapping_lib_df = pd.Series(hum_df.index.values, index=hum_df['idm']).to_dict()

del_mean_point = np.mean(current_case[DEL['idm'].to_list()], axis=0)
el_mean_point = np.mean(current_case[EL_r['idm'].to_list()], axis=0)
em_mean_point = np.mean(current_case[EM_r['idm'].to_list()], axis=0)
del_point = get_closest_point(hum_df, DEL['idm'].to_list(), del_mean_point)
el_point = get_closest_point(hum_df, EL_r['idm'].to_list(), el_mean_point)
em_point = get_closest_point(hum_df, EM_r['idm'].to_list(), em_mean_point)
tsl_del = (del_point + GHjoint1) + translate_humerus_vector
tsl_el = (el_point + GHjoint1) + translate_humerus_vector
tsl_em = (em_point + GHjoint1) + translate_humerus_vector
del_point = np.append(tsl_del, 1)
el_point = np.append(tsl_el, 1)
em_point = np.append(tsl_em, 1)
tsf_del = (hum_t_mat @ del_point.T).T
tsf_el = (hum_t_mat @ el_point.T).T
tsf_em = (hum_t_mat @ em_point.T).T
R_DEL = (tsf_del[:3] - translate_humerus_vector)  # DEL Marker
R_EL = (tsf_el[:3] - translate_humerus_vector)  # EL Marker
R_EM = (tsf_em[:3] - translate_humerus_vector)  # EM Marker

# mapping muscles on r_hum
hum_to_og = glo_origin - rc_scap_5
h_results = {}
h_mean_points = []
for file in os.listdir(r_hum_MAS_folder):
    if file.lower().endswith(".csv"):
        csv_path = os.path.join(r_hum_MAS_folder, file)
        basename = os.path.basename(csv_path)
        H_MAS = pd.read_csv(csv_path)
        H_MAS_coords = get_MAS_xyz(hum_df, H_MAS)
        tsl_h_MAS = (H_MAS_coords + GHjoint1) + translate_humerus_vector
        h_MAS_x = np.hstack((tsl_h_MAS, np.ones((tsl_h_MAS.shape[0], 1))))
        tsf_h_MAS = (hum_t_mat @ h_MAS_x.T).T
        tsf_h_MAS_3d = tsf_h_MAS[:, :3]
        H_Muscle_Attachment_Sites = tsf_h_MAS_3d - translate_humerus_vector
        mean_H_MAS_xyz = np.mean(H_Muscle_Attachment_Sites, axis=0)
        h_results[basename] = mean_H_MAS_xyz

        # project onto bone (nearest node number)
        mas_id_list = H_MAS['idm'].to_list()
        mas_indices = [hum_mapping_lib_df[idm] for idm in mas_id_list]
        mas_points_df = hum_df.iloc[mas_indices]
        mas_points_xyz = mas_points_df[['x', 'y', 'z']].to_numpy()
        distances = np.linalg.norm(mas_points_xyz - mean_H_MAS_xyz, axis=1)
        closest_idx = np.argmin(distances)
        closest_point = mas_points_df.iloc[closest_idx]
        H_MAS_point = closest_point[['x', 'y', 'z']].to_numpy()
        OG_mean_xyz = H_MAS_point - hum_to_og
        h_mean_points.append(
            {'Muscle name': basename, 'mean_x': OG_mean_xyz[0], 'mean_y': OG_mean_xyz[1], 'mean_z': OG_mean_xyz[2]})

h_MAS_df = pd.DataFrame(mean_points)
out_path = (os.path.join(articulation_folder, 'patient_MAS', 'Hum_MAS.csv'))
h_MAS_df.to_csv(out_path, index=False)
h_points = np.array(list(h_results.values()))
h_cloud = pv.PolyData(h_points)
h_labels = list(h_results.keys())
h_cloud["labels"] = h_labels

# finding the most distal point in -z direction for gh joint
R = hum_transformed_vectors
local_points = translated_humerus3 @ R.T
tolerance = 100
mask = np.abs(local_points[:, 1]) < tolerance
filtered_points = local_points[mask]
if filtered_points.size == 0:
    raise ValueError("No points found near Y=0. Increase tolerance.")
min_idx = np.argmin(filtered_points[:, 2])
original_idx = np.where(mask)[0][min_idx]
most_distal_point = translated_humerus3[original_idx]
GHjoint = most_distal_point

plotter = pv.Plotter()
# thorax
plotter.add_mesh(transformed_torso3d, color='cyan')
for i, vector in enumerate(t_transformed_vectors):
    plotter.add_mesh(pv.Arrow(start=translated_ij_point, direction=vector, scale=20), color=colour[i])

# right clavicle
plotter.add_mesh(pv.PolyData(c_SCjoint), color='pink',render_points_as_spheres=True,  point_size=20)
plotter.add_mesh(pv.PolyData(translated_clavicle5), color='purple')
for i, tcvector in enumerate(transformed_vectors):
    plotter.add_mesh(pv.Arrow(start=c_SCjoint, direction=tcvector, scale=20), color=colour[i])

# left clavicle
plotter.add_mesh(pv.PolyData(cl_SCjoint), color='yellow', render_points_as_spheres=True, point_size=20)
plotter.add_mesh(pv.PolyData(cl_translated_clavicle5), color='pink')
for i,clvector in enumerate(cl_transformed_vectors):
    plotter.add_mesh(pv.Arrow(start=cl_SCjoint, direction=clvector, scale=20), color=colour[i])

# right scapula
plotter.add_mesh(pv.PolyData(translated_ACjoint3), color='blue', render_points_as_spheres=True, point_size=5)
plotter.add_mesh(pv.PolyData(translated_scapula3), color='orange')
for i,svector in enumerate(sca_transformed_vectors):
    plotter.add_mesh(pv.Arrow(start=translated_ACjoint3, direction=svector, scale=20), color=colour[i])

# left scapula
plotter.add_mesh(pv.PolyData(l_translated_ACjoint3), color='blue', render_points_as_spheres=True, point_size=5)
plotter.add_mesh(pv.PolyData(l_translated_scapula3), color='red')
for i,slvector in enumerate(l_sca_transformed_vectors):
    plotter.add_mesh(pv.Arrow(start=l_translated_ACjoint3, direction=slvector, scale=20), color=colour[i])

# right_humerus
plotter.add_mesh(pv.PolyData(rc_scap_5), color='blue', render_points_as_spheres=True, point_size=20)
plotter.add_mesh(pv.PolyData(most_distal_point), color='green', render_points_as_spheres=True, point_size=20)
plotter.add_mesh(pv.PolyData(translated_humerus3), color = 'green')
plotter.add_mesh(pv.PolyData(H_Muscle_Attachment_Sites), color = 'red')
plotter.add_mesh(h_cloud, point_size=40, color='pink')
for name, coord in h_results.items():
    plotter.add_point_labels(np.array([coord]), [name], point_size=0, font_size=12)

plotter.add_mesh(pv.PolyData(R_DEL), color='yellow', point_size=20)
plotter.add_mesh(pv.PolyData(R_EL), color='yellow', point_size=20)
plotter.add_mesh(pv.PolyData(R_EM), color='yellow', point_size=20)
for i,hvector in enumerate(hum_transformed_vectors):
    plotter.add_mesh(pv.Arrow(start=GHjoint, direction=hvector, scale=20), color=colour[i])

plotter.show()

##### export mesh as ply and stl for articulation
# translate humerus to origin
# rotation point for humerus is floating point inside humeral head
hum_to_og = glo_origin - rc_scap_5
hum_mesh = translated_humerus3 + hum_to_og
Origin_GHjoint = rc_scap_5 + hum_to_og
ply_reader = vtk.vtkPLYReader()
ply_reader.SetFileName(ssm_HUM_mesh)
ply_reader.Update()
mesh_data = ply_reader.GetOutput()
new_hum_mesh = VTKMeshUtl.update_poly_w_points(hum_mesh, mesh_data)
hum_mesh_verts = VTKMeshUtl.extract_points(new_hum_mesh)
scaled_r_hum = hum_mesh_verts * sf
hum_mesh1 = VTKMeshUtl.update_poly_w_points(scaled_r_hum, copy.deepcopy(new_hum_mesh))
VTKMeshUtl.write(os.path.join(out_path_mesh, "r_hum.ply"), new_hum_mesh)
VTKMeshUtl.write(os.path.join(out_path_mesh, "r_hum_scaled.stl"), hum_mesh1)

# save the meshes in isb coords
ply_reader = vtk.vtkPLYReader()
ply_reader.SetFileName(ssm_HUM_mesh)
ply_reader.Update()
mesh_data = ply_reader.GetOutput()
new_hum_mesh2 = VTKMeshUtl.update_poly_w_points(translated_humerus3, mesh_data)
hum_mesh_verts2 = VTKMeshUtl.extract_points(new_hum_mesh2)
scaled_r_hum2 = hum_mesh_verts2 * sf
hum_mesh1_2 = VTKMeshUtl.update_poly_w_points(scaled_r_hum2, copy.deepcopy(new_hum_mesh2))
VTKMeshUtl.write(os.path.join(out_path_mesh, "r_hum_isb.ply"), new_hum_mesh2)
VTKMeshUtl.write(os.path.join(out_path_mesh, "r_hum_scaled_isb.stl"), hum_mesh1_2)

# Left_Humerus
hum_l_map = l_hum['idm'].to_list()
hum_l_data = {Keywords.vertices: current_case[hum_l_map, :]}
rc_L = np.squeeze(sphere_fit(current_case[humeral_head_l['idm'].to_list()]))
el_point_L = np.nanmean(current_case[EL_l['idm'].to_list()], axis=0)
em_point_L = np.nanmean(current_case[EM_l['idm'].to_list()], axis=0)
mid_ep_L = 0.5 * (em_point_L + el_point_L)
yh_raw_L = rc_L - mid_ep_L
yh_L = (1 / np.linalg.norm(yh_raw_L)) * yh_raw_L
X0_L = em_point_L - rc_L
X1_L = el_point_L - rc_L
xh_raw_L = np.cross(X0_L, X1_L)
xh_L = (1 / np.linalg.norm(xh_raw_L)) * xh_raw_L
zh_L = np.cross(xh_L, yh_L)
hum_without_magnitude_L = np.array([[0, 0, 0], xh_L, yh_L, zh_L])

rc_scap_L = sphere_fit(current_case[glenoid_l['idm'].to_list()])
rc_scap_L2 = rc_scap_L + l_sca_translation_vector
rc_scap_L3 = rc_scap_L2 + l_translate_scapula_vector
rc_scap_L4 = (l_sca_t_mat[:3, :3] @ rc_scap_L3.T).T
rc_scap_L5 = rc_scap_L4 - l_translate_scapula_vector
rc_hum_L = sphere_fit(current_case[humeral_head_l['idm'].to_list()])
GHjoint_L = np.squeeze(rc_scap_L5 - rc_hum_L)
translated_humerus_L = hum_l_data[Keywords.vertices] + GHjoint_L
A_mat_L = np.tile(GHjoint_L, (4, 1))
translate_hum_vector_L = hum_without_magnitude_L - A_mat_L

hum_translated_L = pd.DataFrame(data=translate_hum_vector_L, columns=["x", "y", "z"])
hlvectors = np.zeros((3, 3))
for i in range(3):
     hlvectors[i, :] = translate_hum_vector_L[i + 1, :] - translate_hum_vector_L[0, :]

l_translate_humerus_vector = glo_origin - rc_scap_L5
l_translated_humerus2 = translated_humerus_L + l_translate_humerus_vector
l_hum_source_points = (np.array(hlvectors)).T
l_hum_t_mat = Cloud.transform_between_3x3_points_sets(l_hum_source_points, target_points)
l_hum_points = np.array(l_translated_humerus2)
l_hum_x = np.hstack((l_hum_points, np.ones((l_hum_points.shape[0], 1))))
l_transformed_humerus = (l_hum_t_mat @ l_hum_x.T).T
l_transformed_humerus3d = l_transformed_humerus[:, :3]
l_hum_transformed_vectors = (l_hum_t_mat[:3, :3] @ hlvectors.T).T
l_translated_humerus3 = l_transformed_humerus3d - l_translate_humerus_vector

# mapping markers on transformed r_humerus
l_hum_xyz_df = pd.DataFrame(hum_l_data[Keywords.vertices], columns=['x', 'y', 'z'])
l_hum_xyz_df['idm'] = hum_l_map
l_hum_df = l_hum_xyz_df
l_hum_mapping_lib_df = pd.Series(l_hum_df.index.values, index=l_hum_df['idm']).to_dict()

l_del_mean_point = np.mean(current_case[DEL_l['idm'].to_list()], axis=0)
l_el_mean_point = np.mean(current_case[EL_l['idm'].to_list()], axis=0)
l_em_mean_point = np.mean(current_case[EM_l['idm'].to_list()], axis=0)
l_del_point = get_closest_point(l_hum_df, DEL_l['idm'].to_list(), l_del_mean_point)
l_el_point = get_closest_point(l_hum_df, EL_l['idm'].to_list(), l_el_mean_point)
l_em_point = get_closest_point(l_hum_df, EM_l['idm'].to_list(), l_em_mean_point)
l_tsl_del = (l_del_point + GHjoint_L) + l_translate_humerus_vector
l_tsl_el = (l_el_point + GHjoint_L) + l_translate_humerus_vector
l_tsl_em = (l_em_point + GHjoint_L) + l_translate_humerus_vector
l_del_point = np.append(l_tsl_del, 1)
l_el_point = np.append(l_tsl_el, 1)
l_em_point = np.append(l_tsl_em, 1)
l_tsf_del = (l_hum_t_mat @ l_del_point.T).T
l_tsf_el = (l_hum_t_mat @ l_el_point.T).T
l_tsf_em = (l_hum_t_mat @ l_em_point.T).T
L_DEL = (l_tsf_del[:3] - l_translate_humerus_vector)  # l_DEL Marker
L_EL = (l_tsf_el[:3] - l_translate_humerus_vector)  # l_EL Marker
L_EM = (l_tsf_em[:3] - l_translate_humerus_vector)  # l_EM Marker

# mapping muscles on l_hum
l_hum_to_og = glo_origin - rc_scap_L5
hl_results = {}
hl_mean_points = []
for file in os.listdir(l_hum_MAS_folder):
    if file.lower().endswith(".csv"):
        csv_path = os.path.join(l_hum_MAS_folder, file)
        basename = os.path.basename(csv_path)
        HL_MAS = pd.read_csv(csv_path)
        HL_MAS_coords = get_MAS_xyz(l_hum_df, HL_MAS)
        l_tsl_h_MAS = (HL_MAS_coords + GHjoint_L) + l_translate_humerus_vector
        hl_MAS_x = np.hstack((l_tsl_h_MAS, np.ones((l_tsl_h_MAS.shape[0], 1))))
        l_tsf_h_MAS = (l_hum_t_mat @ hl_MAS_x.T).T
        l_tsf_h_MAS_3d = l_tsf_h_MAS[:, :3]
        HL_Muscle_Attachment_Sites = l_tsf_h_MAS_3d - l_translate_humerus_vector
        l_mean_H_MAS_xyz = np.mean(HL_Muscle_Attachment_Sites, axis=0)
        hl_results[basename] = l_mean_H_MAS_xyz

        # project onto bone (nearest node number)
        mas_id_list = HL_MAS['idm'].to_list()
        mas_indices = [l_hum_mapping_lib_df[idm] for idm in mas_id_list]
        mas_points_df = l_hum_df.iloc[mas_indices]
        mas_points_xyz = mas_points_df[['x', 'y', 'z']].to_numpy()
        distances = np.linalg.norm(mas_points_xyz - l_mean_H_MAS_xyz, axis=1)
        closest_idx = np.argmin(distances)
        closest_point = mas_points_df.iloc[closest_idx]
        HL_MAS_point = closest_point[['x', 'y', 'z']].to_numpy()
        HL_OG_mean_xyz = HL_MAS_point - l_hum_to_og
        hl_mean_points.append(
            {'Muscle name': basename, 'mean_x': HL_OG_mean_xyz[0], 'mean_y': HL_OG_mean_xyz[1], 'mean_z': HL_OG_mean_xyz[2]})

hl_MAS_df = pd.DataFrame(hl_mean_points)
out_path = (os.path.join(articulation_folder, 'patient_MAS', 'Hum_L_MAS.csv'))
hl_MAS_df.to_csv(out_path, index=False)
hl_points = np.array(list(hl_results.values()))
hl_cloud = pv.PolyData(hl_points)
hl_labels = list(hl_results.keys())
hl_cloud["labels"] = hl_labels

# finding the most distal point in -z direction for LEFT gh joint
l_R = l_hum_transformed_vectors
l_local_points = l_translated_humerus3 @ l_R.T
tolerance = 100
l_mask = np.abs(l_local_points[:, 1]) < tolerance
l_filtered_points = l_local_points[l_mask]
if l_filtered_points.size == 0:
    raise ValueError("No points found near Y=0. Increase tolerance.")
l_max_idx = np.argmax(l_filtered_points[:, 2])
l_original_idx = np.where(l_mask)[0][l_max_idx]
l_most_proximal_point = l_translated_humerus3[l_original_idx]
l_GHjoint = l_most_proximal_point

plotter = pv.Plotter()
# thorax
plotter.add_mesh(transformed_torso3d, color='cyan')
for i, vector in enumerate(t_transformed_vectors):
    plotter.add_mesh(pv.Arrow(start=translated_ij_point, direction=vector, scale=20), color=colour[i])

# right clavicle
plotter.add_mesh(pv.PolyData(c_SCjoint), color='pink',render_points_as_spheres=True,  point_size=20)
plotter.add_mesh(pv.PolyData(translated_clavicle5), color='purple')
for i, tcvector in enumerate(transformed_vectors):
    plotter.add_mesh(pv.Arrow(start=c_SCjoint, direction=tcvector, scale=20), color=colour[i])

# left clavicle
plotter.add_mesh(pv.PolyData(cl_SCjoint), color='yellow', render_points_as_spheres=True, point_size=20)
plotter.add_mesh(pv.PolyData(cl_translated_clavicle5), color='pink')
for i,clvector in enumerate(cl_transformed_vectors):
    plotter.add_mesh(pv.Arrow(start=cl_SCjoint, direction=clvector, scale=20), color=colour[i])

# right scapula
plotter.add_mesh(pv.PolyData(translated_ACjoint3), color='blue', render_points_as_spheres=True, point_size=5)
plotter.add_mesh(pv.PolyData(translated_scapula3), color='orange')
for i,svector in enumerate(sca_transformed_vectors):
    plotter.add_mesh(pv.Arrow(start=translated_ACjoint3, direction=svector, scale=20), color=colour[i])

# left scapula
plotter.add_mesh(pv.PolyData(l_translated_ACjoint3), color='blue', render_points_as_spheres=True, point_size=5)
plotter.add_mesh(pv.PolyData(l_translated_scapula3), color='red')
for i,slvector in enumerate(l_sca_transformed_vectors):
    plotter.add_mesh(pv.Arrow(start=l_translated_ACjoint3, direction=slvector, scale=20), color=colour[i])

# right_humerus
plotter.add_mesh(pv.PolyData(GHjoint), color='green', render_points_as_spheres=True, point_size=5)
plotter.add_mesh(pv.PolyData(translated_humerus3), color = 'green')
for i,hvector in enumerate(hum_transformed_vectors):
    plotter.add_mesh(pv.Arrow(start=GHjoint, direction=hvector, scale=20), color=colour[i])

# left_humerus
plotter.add_mesh(pv.PolyData(rc_scap_L5), color='green', render_points_as_spheres=True, point_size=5)
plotter.add_mesh(pv.PolyData(l_translated_humerus3), color= 'blue')
# plotter.add_mesh(pv.PolyData(L_H_Muscle_Attachment_Sites), color = 'red')
plotter.add_mesh(hl_cloud, point_size=40, color='pink')
for name, coord in hl_results.items():
    plotter.add_point_labels(np.array([coord]), [name], point_size=0, font_size=12)

plotter.add_mesh(pv.PolyData(L_DEL), color='yellow', point_size=20)
plotter.add_mesh(pv.PolyData(L_EL), color='yellow', point_size=20)
plotter.add_mesh(pv.PolyData(L_EM), color='yellow', point_size=20)
for i,hlvector in enumerate(l_hum_transformed_vectors):
    plotter.add_mesh(pv.Arrow(start=l_GHjoint, direction=hlvector, scale=20), color=colour[i])

plotter.show()

marker_names = ['IJ', 'PX', 'C7', 'T8', 'SC', 'C_AC', 'CAP', 'SA', 'IA', 'TS', 'DEL', 'EL', 'EM']
marker_coords = np.array([translated_ij_point, transformed_px, transformed_c7, transformed_t8, C_SC, C_AC, R_CAP, R_SA, R_IA, R_TS, R_DEL, R_EL, R_EM])
df_markers = pd.DataFrame(marker_coords, index=marker_names, columns=['x', 'y', 'z'])


##### export mesh as ply and stl for articulation
# translate humerus to origin
# rotation point for humerus is floating point inside humeral head
l_hum_to_og = glo_origin - rc_scap_L5
l_hum_mesh = l_translated_humerus3 + l_hum_to_og
l_Origin_GHjoint = rc_scap_L5 + l_hum_to_og
ply_reader = vtk.vtkPLYReader()
ply_reader.SetFileName(l_ssm_HUM_mesh)
ply_reader.Update()
mesh_data = ply_reader.GetOutput()
l_new_hum_mesh = VTKMeshUtl.update_poly_w_points(l_hum_mesh, mesh_data)
l_hum_mesh_verts = VTKMeshUtl.extract_points(l_new_hum_mesh)
l_scaled_r_hum = l_hum_mesh_verts * sf
l_hum_mesh1 = VTKMeshUtl.update_poly_w_points(l_scaled_r_hum, copy.deepcopy(l_new_hum_mesh))
VTKMeshUtl.write(os.path.join(out_path_mesh, "l_hum.ply"), l_new_hum_mesh)
VTKMeshUtl.write(os.path.join(out_path_mesh, "l_hum_scaled.stl"), l_hum_mesh1)

# save the meshes in isb coords
ply_reader = vtk.vtkPLYReader()
ply_reader.SetFileName(l_ssm_HUM_mesh)
ply_reader.Update()
mesh_data = ply_reader.GetOutput()
l_new_hum_mesh_2 = VTKMeshUtl.update_poly_w_points(l_translated_humerus3, mesh_data)
l_hum_mesh_verts_2 = VTKMeshUtl.extract_points(l_new_hum_mesh_2)
l_scaled_hum_2 = l_hum_mesh_verts_2 * sf
l_hum_mesh1_2 = VTKMeshUtl.update_poly_w_points(l_scaled_hum_2, copy.deepcopy(l_new_hum_mesh_2))
VTKMeshUtl.write(os.path.join(out_path_mesh, "l_hum_isb.ply"), l_new_hum_mesh_2)
VTKMeshUtl.write(os.path.join(out_path_mesh, "l_hum_scaled_isb.stl"), l_hum_mesh1_2)

# isb joint centers
#
isb_joint_centers.append({'IJ': translated_ij_point, 'r_SC': c_SCjoint, 'l_SC': cl_SCjoint, 'r_AC': translated_ACjoint3,
                          'l_AC': l_translated_ACjoint3, 'r_GH': rc_scap_5, 'l_GH': rc_scap_L5})
isb_JC_df = pd.DataFrame(isb_joint_centers)
isb_JC_df.to_csv(os.path.join(articulation_folder, 'patient_MAS', 'ISB_JointCenters.csv'))

# .osim coords for alignment
clav_to_og1 = clav_to_og
ac = clav_to_og1 + C_AC
os_r_ac_joint_offset = sca_to_og - glo_origin
cap = os_r_ac_joint_offset + R_CAP
sa = os_r_ac_joint_offset + R_SA
ia = os_r_ac_joint_offset + R_IA
ts = os_r_ac_joint_offset + R_TS
os_r_gh_joint_offset = hum_to_og - sca_to_og
os_r_gh_joint_offset1 = hum_to_og - glo_origin
deltoid = os_r_gh_joint_offset1 + R_DEL
el = os_r_gh_joint_offset1 + R_EL
em = os_r_gh_joint_offset1 + R_EM

l_clav_to_og1 = l_clav_to_og
os_l_ac_joint_offset = l_sca_to_og - l_clav_to_og
os_l_gh_joint_offset = l_hum_to_og - l_sca_to_og









