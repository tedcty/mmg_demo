import os
import numpy as np
import pandas as pd
import vtk
from yatpkg.util.data import VTKMeshUtl
import pyvista as pv
from enum import Enum


class Keywords(Enum):
    actor = [0, 'actor']
    polydata = [1, 'polydata']
    vertices = [2, 'vertices']
    idx = [3, 'idx']
    idm = [4, 'idm']

class asm:
    def __init__(self, pc: str = None, mean_mesh=None, maps=None):
        s = np.load(pc,allow_pickle=True)
        self.mean = s['mean']
        self.weights = s['weights']  # PC weights are variance
        self.modes = s['modes']
        self.SD = s['SD']
        self.projectedWeights = s['projectedWeights']
        self.mean_mesh = mean_mesh

    def extract_parts(self, node_id_list, export_vert=True):
        mesh_data_copy = vtk.vtkPolyData()
        mesh_org = self.mean_mesh['polydata']
        mesh_data_copy.DeepCopy(mesh_org)
        cell_data = mesh_data_copy.GetPolys()
        vertices = None
        if export_vert:
            # Get the point data and cell data of the mesh
            points_l = []
            for i in range(len(node_id_list)):
                p = mesh_data_copy.GetPoint(node_id_list[i])
                points_l.append(p)
            vertices = np.array(points_l)

        for i in range(cell_data.GetNumberOfCells()):
            cell = mesh_data_copy.GetCell(i)
            p1 = cell.GetPointId(0)
            p2 = cell.GetPointId(1)
            p3 = cell.GetPointId(2)
            if not (p1 in node_id_list and p2 in node_id_list and p3 in node_id_list):
                mesh_data_copy.DeleteCell(i)
        mesh_data_copy.RemoveDeletedCells()

        return [mesh_data_copy, vertices]

    def create_part(self, color, force_build, mean_mesh, part_map):
        mesh = np.reshape(self.mean, [int(self.mean.shape[0] / 3), 3])
        part = mesh[part_map, :]
        new_shape = None
        if force_build or not os.path.exists(mean_mesh):
            new_shape = self.extract_parts(part_map)
            # Write the mesh to file "*.ply"
            w = vtk.vtkPLYWriter()
            w.SetInputData(new_shape[0])
            w.SetFileName(mean_mesh)
            w.Write()
        if force_build:
            polydata = new_shape[0]
        else:
            reader = vtk.vtkPLYReader()
            reader.SetFileName(mean_mesh)
            reader.Update()
            polydata = reader.GetOutput()
        mapper = vtk.vtkPolyDataMapper()
        if vtk.VTK_MAJOR_VERSION <= 5:
            # mapper.SetInput(reader.GetOutput())
            mapper.SetInput(polydata)
        else:
            mapper.SetInputData(polydata)
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(color[0], color[1], color[2])
        return {Keywords.actor: actor, Keywords.polydata: polydata, Keywords.vertices: part}

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


# Body segment paths
combined = r"C:\Users\rrag962\Documents\shapemodel\model\Shoulder 3\Segmentations\SSM\Combined\combinedSSM"
mean_pc = r"C:\Users\rrag962\Documents\shapemodel\model\Shoulder 3\Segmentations\SSM\Combined\shape_model\combinedSSM.pc.npz"
tho_map = pd.read_csv(r"C:\Users\rrag962\Documents\shapemodel\model\Shoulder 3\Segmentations\Landmarking\6TH_combined\maps to mean\Tho.csv")
l_clav = pd.read_csv(r"C:\Users\rrag962\Documents\shapemodel\model\Shoulder 3\Segmentations\Landmarking\6TH_combined\maps to mean\L_clav.csv")
r_clav = pd.read_csv(r"C:\Users\rrag962\Documents\shapemodel\model\Shoulder 3\Segmentations\Landmarking\6TH_combined\maps to mean\R_clav.csv")
l_scap = pd.read_csv(r"C:\Users\rrag962\Documents\shapemodel\model\Shoulder 3\Segmentations\Landmarking\6TH_combined\maps to mean\L_scap.csv")
r_scap = pd.read_csv(r"C:\Users\rrag962\Documents\shapemodel\model\Shoulder 3\Segmentations\Landmarking\6TH_combined\maps to mean\R_scap.csv")
l_hum = pd.read_csv(r"C:\Users\rrag962\Documents\shapemodel\model\Shoulder 3\Segmentations\Landmarking\6TH_combined\maps to mean\L_hum.csv")
r_hum = pd.read_csv(r"C:\Users\rrag962\Documents\shapemodel\model\Shoulder 3\Segmentations\Landmarking\6TH_combined\maps to mean\R_hum.csv")
l_rad = pd.read_csv(r"C:\Users\rrag962\Documents\shapemodel\model\Shoulder 3\Segmentations\Landmarking\6TH_combined\maps to mean\L_rad.csv")
r_rad = pd.read_csv(r"C:\Users\rrag962\Documents\shapemodel\model\Shoulder 3\Segmentations\Landmarking\6TH_combined\maps to mean\R_rad.csv")
l_ulna = pd.read_csv(r"C:\Users\rrag962\Documents\shapemodel\model\Shoulder 3\Segmentations\Landmarking\6TH_combined\maps to mean\L_ulna.csv")
r_ulna = pd.read_csv(r"C:\Users\rrag962\Documents\shapemodel\model\Shoulder 3\Segmentations\Landmarking\6TH_combined\maps to mean\R_ulna.csv")

# landmarking csv maps
C71 = pd.read_csv(r"C:\Users\rrag962\Documents\shapemodel\model\Shoulder 3\Segmentations\Landmarking\6TH_combined\maps to mean\tho_c7_r.csv")
C72 = pd.read_csv(r"C:\Users\rrag962\Documents\shapemodel\model\Shoulder 3\Segmentations\Landmarking\6TH_combined\maps to mean\tho_c7_l.csv")
T81 = pd.read_csv(r"C:\Users\rrag962\Documents\shapemodel\model\Shoulder 3\Segmentations\Landmarking\6TH_combined\maps to mean\tho_t8_r.csv")
T82 = pd.read_csv(r"C:\Users\rrag962\Documents\shapemodel\model\Shoulder 3\Segmentations\Landmarking\6TH_combined\maps to mean\tho_t8_l.csv")
IJ = pd.read_csv(r"C:\Users\rrag962\Documents\shapemodel\model\Shoulder 3\Segmentations\Landmarking\6TH_combined\maps to mean\tho_ij.csv")
PX = pd.read_csv(r"C:\Users\rrag962\Documents\shapemodel\model\Shoulder 3\Segmentations\Landmarking\6TH_combined\maps to mean\tho_px.csv")
ThoIJsphere_r = pd.read_csv(r"C:\Users\rrag962\Documents\shapemodel\model\Shoulder 3\Segmentations\Landmarking\6TH_combined\maps to mean\tho_scj_r.csv")
ThoIJsphere_L = pd.read_csv(r"C:\Users\rrag962\Documents\shapemodel\model\Shoulder 3\Segmentations\Landmarking\6TH_combined\maps to mean\tho_scj_l.csv")
SC_r = pd.read_csv(r"C:\Users\rrag962\Documents\shapemodel\model\Shoulder 3\Segmentations\Landmarking\6TH_combined\maps to mean\cla_r_sc.csv")
AC_r = pd.read_csv(r"C:\Users\rrag962\Documents\shapemodel\model\Shoulder 3\Segmentations\Landmarking\6TH_combined\maps to mean\cla_r_ac.csv")
SC_l = pd.read_csv(r"C:\Users\rrag962\Documents\shapemodel\model\Shoulder 3\Segmentations\Landmarking\6TH_combined\maps to mean\cla_l_sc.csv")
AC_l = pd.read_csv(r"C:\Users\rrag962\Documents\shapemodel\model\Shoulder 3\Segmentations\Landmarking\6TH_combined\maps to mean\cla_l_ac.csv")
ClaSCsphere_r = pd.read_csv(r"C:\Users\rrag962\Documents\shapemodel\model\Shoulder 3\Segmentations\Landmarking\6TH_combined\maps to mean\cla_scj_r.csv")
ClaACsphere_r = pd.read_csv(r"C:\Users\rrag962\Documents\shapemodel\model\Shoulder 3\Segmentations\Landmarking\6TH_combined\maps to mean\cla_acj_r.csv")
ClaSCsphere_l = pd.read_csv(r"C:\Users\rrag962\Documents\shapemodel\model\Shoulder 3\Segmentations\Landmarking\6TH_combined\maps to mean\cla_scj_l.csv")
ClaACsphere_l = pd.read_csv(r"C:\Users\rrag962\Documents\shapemodel\model\Shoulder 3\Segmentations\Landmarking\6TH_combined\maps to mean\cla_acj_l.csv")
AA_r = pd.read_csv(r"C:\Users\rrag962\Documents\shapemodel\model\Shoulder 3\Segmentations\Landmarking\6TH_combined\maps to mean\sca_r_aa.csv")
AI_r = pd.read_csv(r"C:\Users\rrag962\Documents\shapemodel\model\Shoulder 3\Segmentations\Landmarking\6TH_combined\maps to mean\sca_r_ai.csv")
TS_r = pd.read_csv(r"C:\Users\rrag962\Documents\shapemodel\model\Shoulder 3\Segmentations\Landmarking\6TH_combined\maps to mean\sca_r_ts.csv")
SA_r = pd.read_csv(r"C:\Users\rrag962\Documents\shapemodel\model\Shoulder 3\Segmentations\Landmarking\6TH_combined\maps to mean\sca_r_as.csv")
AA_l = pd.read_csv(r"C:\Users\rrag962\Documents\shapemodel\model\Shoulder 3\Segmentations\Landmarking\6TH_combined\maps to mean\sca_l_aa.csv")
AI_l = pd.read_csv(r"C:\Users\rrag962\Documents\shapemodel\model\Shoulder 3\Segmentations\Landmarking\6TH_combined\maps to mean\sca_l_ai.csv")
TS_l = pd.read_csv(r"C:\Users\rrag962\Documents\shapemodel\model\Shoulder 3\Segmentations\Landmarking\6TH_combined\maps to mean\sca_l_ts.csv")
SA_l = pd.read_csv(r"C:\Users\rrag962\Documents\shapemodel\model\Shoulder 3\Segmentations\Landmarking\6TH_combined\maps to mean\sca_l_as.csv")
ScaACsphere_r = pd.read_csv(r"C:\Users\rrag962\Documents\shapemodel\model\Shoulder 3\Segmentations\Landmarking\6TH_combined\maps to mean\scap_acj_r.csv")
glenoid_r = pd.read_csv(r"C:\Users\rrag962\Documents\shapemodel\model\Shoulder 3\Segmentations\Landmarking\6TH_combined\maps to mean\scap_ghj_r.csv")
ScaACsphere_l = pd.read_csv(r"C:\Users\rrag962\Documents\shapemodel\model\Shoulder 3\Segmentations\Landmarking\6TH_combined\maps to mean\scap_acj_l.csv")
glenoid_l = pd.read_csv(r"C:\Users\rrag962\Documents\shapemodel\model\Shoulder 3\Segmentations\Landmarking\6TH_combined\maps to mean\scap_ghj_l.csv")
EL_r = pd.read_csv(r"C:\Users\rrag962\Documents\shapemodel\model\Shoulder 3\Segmentations\Landmarking\6TH_combined\maps to mean\hum_r_el.csv")
EM_r = pd.read_csv(r"C:\Users\rrag962\Documents\shapemodel\model\Shoulder 3\Segmentations\Landmarking\6TH_combined\maps to mean\hum_r_em.csv")
EL_l = pd.read_csv(r"C:\Users\rrag962\Documents\shapemodel\model\Shoulder 3\Segmentations\Landmarking\6TH_combined\maps to mean\hum_l_el.csv")
EM_l = pd.read_csv(r"C:\Users\rrag962\Documents\shapemodel\model\Shoulder 3\Segmentations\Landmarking\6TH_combined\maps to mean\hum_l_em.csv")
humeral_head_r = pd.read_csv(r"C:\Users\rrag962\Documents\shapemodel\model\Shoulder 3\Segmentations\Landmarking\6TH_combined\maps to mean\hum_ghj_r.csv")
HumUlna_r = pd.read_csv(r"C:\Users\rrag962\Documents\shapemodel\model\Shoulder 3\Segmentations\Landmarking\6TH_combined\maps to mean\hum_radj_r.csv")
humeral_head_l = pd.read_csv(r"C:\Users\rrag962\Documents\shapemodel\model\Shoulder 3\Segmentations\Landmarking\6TH_combined\maps to mean\hum_ghj_l.csv")
HumUlna_l = pd.read_csv(r"C:\Users\rrag962\Documents\shapemodel\model\Shoulder 3\Segmentations\Landmarking\6TH_combined\maps to mean\hum_radj_l.csv")
US_r = pd.read_csv(r"C:\Users\rrag962\Documents\shapemodel\model\Shoulder 3\Segmentations\Landmarking\6TH_combined\maps to mean\uln_r_us.csv")
US_l = pd.read_csv(r"C:\Users\rrag962\Documents\shapemodel\model\Shoulder 3\Segmentations\Landmarking\6TH_combined\maps to mean\uln_l_us.csv")
UlnHum_r = pd.read_csv(r"C:\Users\rrag962\Documents\shapemodel\model\Shoulder 3\Segmentations\Landmarking\6TH_combined\maps to mean\uln_uocj_r.csv")
UlnHum_l = pd.read_csv(r"C:\Users\rrag962\Documents\shapemodel\model\Shoulder 3\Segmentations\Landmarking\6TH_combined\maps to mean\uln_uocj_l.csv")
RS_r = pd.read_csv(r"C:\Users\rrag962\Documents\shapemodel\model\Shoulder 3\Segmentations\Landmarking\6TH_combined\maps to mean\rad_r_rsp.csv")
RS_l = pd.read_csv(r"C:\Users\rrag962\Documents\shapemodel\model\Shoulder 3\Segmentations\Landmarking\6TH_combined\maps to mean\rad_l_rsp.csv")
HumRadius_r = pd.read_csv(r"C:\Users\rrag962\Documents\shapemodel\model\Shoulder 3\Segmentations\Landmarking\6TH_combined\maps to mean\hum_radj_r.csv")
RadHumerus_r = pd.read_csv(r"C:\Users\rrag962\Documents\shapemodel\model\Shoulder 3\Segmentations\Landmarking\6TH_combined\maps to mean\rad_humj_r.csv")
HumRadius_l = pd.read_csv(r"C:\Users\rrag962\Documents\shapemodel\model\Shoulder 3\Segmentations\Landmarking\6TH_combined\maps to mean\hum_radj_l.csv")
RadHumerus_l = pd.read_csv(r"C:\Users\rrag962\Documents\shapemodel\model\Shoulder 3\Segmentations\Landmarking\6TH_combined\maps to mean\rad_humj_l.csv")

combined_files = [f for f in os.listdir(combined)]
for combined_file in combined_files:
    model = asm(mean_pc)
    case = combined_file[:8]
    ply_reader = vtk.vtkPLYReader()
    ply_reader.SetFileName(os.path.join(combined, combined_file))
    ply_reader.Update()
    mesh_data = ply_reader.GetOutput()
    current_case = VTKMeshUtl.extract_points(mesh_data)
    # thorax
    thorax_map = tho_map['idm'].to_list()
    colour = [144 / 255, 207 / 255, 252 / 255]
    mean_mesh = r"C:\Users\rrag962\Documents\shapemodel\model\Shoulder 3\Segmentations\Landmarking\6TH_combined\mean combined mesh\combinedSSM_mean.ply"
    thorax_data = model.create_part(colour, False, mean_mesh, thorax_map)
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
    plotter = pv.Plotter()
    plotter.add_mesh(thorax_data[Keywords.vertices])
    for i, vector in enumerate(vectors):
        plotter.add_mesh(pv.Arrow(start=ij_point, direction=vector, scale=20), color=colour[i])
    plotter.show()

    # Right_Clavicle
    sc_r_point = np.mean(current_case[SC_r['idm'].to_list()], axis=0)
    ac_r_point = np.mean(current_case[AC_r['idm'].to_list()], axis=0)
    cla_r_map = r_clav['idm'].to_list()
    colour = [144 / 255, 207 / 255, 252 / 255]
    mean_mesh = r"C:\Users\rrag962\Documents\shapemodel\model\Shoulder 3\Segmentations\Landmarking\6TH_combined\mean combined mesh\combinedSSM_mean.ply"
    cla_r_data = model.create_part(colour, False, mean_mesh, cla_r_map)
    zc_raw = ac_r_point - sc_r_point
    zc = (1 / np.linalg.norm(zc_raw)) * zc_raw
    xc_raw = np.cross(yt, zc)
    xc = (1 / np.linalg.norm(zc_raw)) * xc_raw
    yc = np.cross(zc, xc)
    cla_without_magnitude_R = np.array([[0, 0, 0], xc, yc, zc])
    #Left_Clavicle
    sc_point_L = np.mean(current_case[SC_l['idm'].to_list()], axis=0)
    ac_point_L = np.mean(current_case[AC_l['idm'].to_list()], axis=0)
    cla_l_map = l_clav['idm'].to_list()
    colour = [144 / 255, 207 / 255, 252 / 255]
    mean_mesh = r"C:\Users\rrag962\Documents\shapemodel\model\Shoulder 3\Segmentations\Landmarking\6TH_combined\mean combined mesh\combinedSSM_mean.ply"
    cla_l_data = model.create_part(colour, False, mean_mesh, cla_l_map)
    zc_raw_L = sc_point_L - ac_point_L
    zc_L = (1 / np.linalg.norm(zc_raw_L)) * zc_raw_L
    xc_raw_L = np.cross(yt, zc_L)
    xc_L = (1 / np.linalg.norm(zc_raw_L)) * xc_raw_L
    yc_L = np.cross(zc_L, xc_L)
    cla_without_magnitude_L = np.array([[0, 0, 0], xc_L, yc_L, zc_L])

    # sphere fit
    rc_SC = sphere_fit(current_case[ClaSCsphere_r['idm'].to_list()])
    rc_SC_L = sphere_fit(current_case[ClaSCsphere_l['idm'].to_list()])
    rc_IJ = sphere_fit(current_case[ThoIJsphere_r['idm'].to_list()])
    rc_IJ_L = sphere_fit(current_case[ThoIJsphere_L['idm'].to_list()])

    SCjoint = np.squeeze(rc_IJ - rc_SC)
    translated_clavicle = (cla_r_data[Keywords.vertices]) + SCjoint
    A_mat = np.tile(SCjoint, (4, 1))
    translate_cla_vector = cla_without_magnitude_R - A_mat

    SCjoint_L = np.squeeze(rc_IJ_L - rc_SC_L)
    translated_clavicle_L = (cla_l_data[Keywords.vertices]) + SCjoint_L
    A_mat_L = np.tile(SCjoint_L, (4, 1))
    translate_cla_vector_L = cla_without_magnitude_L - A_mat_L

    cla_translated = pd.DataFrame(data=translate_cla_vector, columns=["x", "y", "z"])
    cla_translated_L = pd.DataFrame(data=translate_cla_vector_L, columns=["x", "y", "z"])
    cvectors = np.zeros((3, 3))
    clvectors = np.zeros((3, 3))

    colour = ['red', 'green', 'blue']
    for i in range(3):
        cvectors[i, :] = translate_cla_vector[i + 1, :] - translate_cla_vector[0, :]
        clvectors[i, :] = translate_cla_vector_L[i + 1, :] - translate_cla_vector_L[0, :]
    plotter = pv.Plotter()
    plotter.add_mesh(translated_clavicle, color='blue')
    plotter.add_mesh(translated_clavicle_L, color='pink')
    plotter.add_mesh(thorax_data[Keywords.vertices])
    for i, cvector in enumerate(cvectors):
        plotter.add_mesh(pv.Arrow(start=SCjoint, direction=cvector, scale=20), color=colour[i])
    for i,clvector in enumerate(clvectors):
        plotter.add_mesh(pv.Arrow(start=SCjoint_L, direction=clvector, scale=20), color=colour[i])
    for i, vector in enumerate(vectors):
        plotter.add_mesh(pv.Arrow(start=ij_point, direction=vector, scale=20), color=colour[i])
    plotter.show()

    # Right_Scapula
    aa_point = np.mean(current_case[AA_r['idm'].to_list()], axis=0)
    ai_point = np.mean(current_case[AI_r['idm'].to_list()], axis=0)
    ts_point = np.mean(current_case[TS_r['idm'].to_list()], axis=0)
    sa_point = np.mean(current_case[SA_r['idm'].to_list()], axis=0)
    sca_r_map = r_scap['idm'].to_list()
    colour = [144 / 255, 207 / 255, 252 / 255]
    mean_mesh = r"C:\Users\rrag962\Documents\shapemodel\model\Shoulder 3\Segmentations\Landmarking\6TH_combined\mean combined mesh\combinedSSM_mean.ply"
    sca_r_data = model.create_part(colour, False, mean_mesh, sca_r_map)
    zs_raw = aa_point - ts_point
    zs = (1 / np.linalg.norm(zs_raw)) * zs_raw
    yx1_raw = ts_point - aa_point
    yx2_raw = ai_point - aa_point
    xs_raw = np.cross(yx2_raw, yx1_raw)
    xs = (1 / np.linalg.norm(xs_raw)) * xs_raw
    ys = np.cross(zs, xs)
    sca_without_magnitude = np.array([[0, 0, 0], xs, ys, zs])
    # Left_Scapula
    aa_point_L = np.mean(current_case[AA_l['idm'].to_list()], axis=0)
    ai_point_L = np.mean(current_case[AI_l['idm'].to_list()], axis=0)
    ts_point_L = np.mean(current_case[TS_l['idm'].to_list()], axis=0)
    sa_point_L = np.mean(current_case[SA_l['idm'].to_list()], axis=0)
    sca_l_map = r_scap['idm'].to_list()
    colour = [144 / 255, 207 / 255, 252 / 255]
    mean_mesh = r"C:\Users\rrag962\Documents\shapemodel\model\Shoulder 3\Segmentations\Landmarking\6TH_combined\mean combined mesh\combinedSSM_mean.ply"
    sca_l_data = model.create_part(colour, False, mean_mesh, sca_l_map)
    zs_raw_L = ts_point_L - aa_point_L
    zs_L = (1 / np.linalg.norm(zs_raw_L)) * zs_raw_L
    yx1_raw_L = ts_point_L - aa_point_L
    yx2_raw_L = ai_point_L - aa_point_L
    xs_raw_L = np.cross(yx1_raw_L, yx2_raw_L)
    xs_L = (1 / np.linalg.norm(xs_raw_L)) * xs_raw_L
    ys_L = np.cross(zs_L, xs_L)
    sca_without_magnitude_L = np.array([[0, 0, 0], xs_L, ys_L, zs_L])

    rc_CAC = sphere_fit(translated_clavicle[ClaACsphere_r['idm'].to_list()])
    rc_CAC_L = sphere_fit(translated_clavicle_L[ClaACsphere_l['idm'].to_list()])
    rc_SAC = sphere_fit(current_case[ScaACsphere_r['idm'].to_list()])
    rc_SAC_L = sphere_fit(current_case[ScaACsphere_l['idm'].to_list()])

    ACjoint = np.squeeze(rc_CAC - rc_SAC)
    translated_scapula = (sca_r_data[Keywords.vertices]) + ACjoint
    A_mat = np.tile(ACjoint, (4, 1))
    translate_sca_vector = sca_without_magnitude - A_mat

    ACjoint_L = np.squeeze(rc_CAC_L - rc_SAC_L)
    translated_scapula_L = (sca_l_data[Keywords.vertices]) + ACjoint_L
    A_mat_L = np.tile(ACjoint_L, (4, 1))
    translate_sca_vector_L = sca_without_magnitude_L - A_mat_L

    sca_translated = pd.DataFrame(data=translate_sca_vector, columns=["x", "y", "z"])
    sca_translated_L = pd.DataFrame(data=translate_sca_vector_L, columns=["x", "y", "z"])
    svectors = np.zeros((3, 3))
    slvectors = np.zeros((3, 3))

    colour = ['red', 'green', 'blue']
    for i in range(3):
        svectors[i, :] = translate_sca_vector[i + 1, :] - translate_sca_vector[0, :]
        slvectors[i, :] = translate_sca_vector_L[i + 1, :] - translate_sca_vector_L[0, :]
    plotter = pv.Plotter()
    plotter.add_mesh(translated_scapula, color='red')
    plotter.add_mesh(translated_scapula_L, color='green')
    plotter.add_mesh(translated_clavicle, color='blue')
    plotter.add_mesh(translated_clavicle_L, color='pink')
    plotter.add_mesh(thorax_data[Keywords.vertices])
    for i, svector in enumerate(svectors):
        plotter.add_mesh(pv.Arrow(start=ACjoint, direction=svector, scale=20), color=colour[i])
    for i, slvector in enumerate(slvectors):
        plotter.add_mesh(pv.Arrow(start=ACjoint_L, direction=slvector, scale=20), color=colour[i])
    for i, cvector in enumerate(cvectors):
        plotter.add_mesh(pv.Arrow(start=SCjoint, direction=cvector, scale=20), color=colour[i])
    for i, clvector in enumerate(clvectors):
        plotter.add_mesh(pv.Arrow(start=SCjoint_L, direction=clvector, scale=20), color=colour[i])
    for i, vector in enumerate(vectors):
        plotter.add_mesh(pv.Arrow(start=ij_point, direction=vector, scale=20), color=colour[i])
    plotter.show()

    # Right_Humerus
    Hum_r_map = np.mean(current_case[r_hum['idm'].to_list()], axis=0)
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
    # Left_Humerus
    Hum_l_map = np.mean(current_case[l_hum['idm'].to_list()], axis=0)
    rc_L = np.squeeze(sphere_fit(current_case[humeral_head_l['idm'].to_list()]))
    el_point_L = np.nanmean(current_case[EL_l['idm'].to_list()], axis=0)
    em_point_L = np.nanmean(current_case[EM_l['idm'].to_list()], axis=0)
    mid_ep_L = 0.5 * (em_point_L + el_point_L)
    yh_raw_L = rc_L - mid_ep_L
    yh_L = (1 / np.linalg.norm(yh_raw_L)) * yh_raw_L
    X0_L = em_point_L - rc_L
    X1_L = el_point_L - rc_L
    xh_raw_L = np.cross(X1_L, X0_L)
    xh_L = (1 / np.linalg.norm(xh_raw_L)) * xh_raw_L
    zh_L = np.cross(xh_L, yh_L)
    hum_without_magnitude_L = np.array([[0, 0, 0], xh_L, yh_L, zh_L])
# translate
    rc_scap = sphere_fit(translated_scapula[glenoid_r['idm'].to_list()])
    rc_scap_L = sphere_fit(translated_scapula_L[glenoid_l['idm'].to_list()])
    rc_hum = sphere_fit(current_R_humerus[humeral_head_r['idm'].to_list()])
    rc_hum_L = sphere_fit(current_L_humerus[humeral_head_l['idm'].to_list()])

    GHjoint = np.squeeze(rc_scap - rc_hum)
    translated_humerus = current_R_humerus + GHjoint
    A_mat = np.tile(GHjoint, (4, 1))
    translate_hum_vector = hum_without_magnitude - A_mat

    GHjoint_L = np.squeeze(rc_scap_L - rc_hum_L)
    translated_humerus_L = current_L_humerus + GHjoint_L
    A_mat_L = np.tile(GHjoint_L, (4, 1))
    translate_hum_vector_L = hum_without_magnitude_L - A_mat_L

    hum_translated = pd.DataFrame(data=translate_hum_vector, columns=["x", "y", "z"])
    hum_translated_L = pd.DataFrame(data=translate_hum_vector_L, columns=["x", "y", "z"])
    hvectors = np.zeros((3, 3))
    hlvectors = np.zeros((3, 3))

    # colour = ['red', 'green', 'blue']
    # for i in range(3):
    #     hvectors[i, :] = translate_hum_vector[i + 1, :] - translate_hum_vector[0, :]
    #     hlvectors[i, :] = translate_hum_vector_L[i + 1, :] - translate_hum_vector_L[0, :]
    # plotter = pv.Plotter()
    # plotter.add_mesh(translated_humerus, color='magenta')
    # plotter.add_mesh(translated_humerus_L, color='cyan')
    # plotter.add_mesh(translated_scapula, color='red')
    # plotter.add_mesh(translated_scapula_L, color='green')
    # plotter.add_mesh(translated_clavicle, color='blue')
    # plotter.add_mesh(translated_clavicle_L, color='pink')
    # plotter.add_mesh(current_thorax)
    # for i, hvector in enumerate(hvectors):
    #     plotter.add_mesh(pv.Arrow(start=GHjoint, direction=hvector, scale=20),
    #                      color=colour[i])
    # for i, hlvector in enumerate(hlvectors):
    #     plotter.add_mesh(pv.Arrow(start=GHjoint_L, direction=hlvector, scale=20),
    #                      color=colour[i])
    # for i, svector in enumerate(svectors):
    #     plotter.add_mesh(pv.Arrow(start=ACjoint, direction=svector, scale=20),
    #                      color=colour[i])
    # for i, slvector in enumerate(slvectors):
    #     plotter.add_mesh(pv.Arrow(start=ACjoint_L, direction=slvector, scale=20),
    #                      color=colour[i])
    # for i, cvector in enumerate(cvectors):
    #     plotter.add_mesh(pv.Arrow(start=SCjoint, direction=cvector, scale=20),
    #                      color=colour[i])
    # for i, clvector in enumerate(clvectors):
    #     plotter.add_mesh(pv.Arrow(start=SCjoint_L, direction=clvector, scale=20),
    #                      color=colour[i])
    # for i, vector in enumerate(vectors):
    #     plotter.add_mesh(pv.Arrow(start=ij_point, direction=vector, scale=20),
    #                      color=colour[i])
    # plotter.show()

    # Right_Ulna
    Ulna_r_map = np.mean(current_case[r_ulna['idm'].to_list()], axis=0)
    us_point = np.nanmean(current_case[US_r['idm'].to_list()], axis=0)
    yu_raw = mid_ep - us_point
    yu = (1 / np.linalg.norm(yu_raw)) * yu_raw
    X0u = em_point - us_point
    X1u = el_point - us_point
    xu_raw = np.cross(X1u, X0u)
    xu = (1 / np.linalg.norm(xu_raw)) * xu_raw
    zu = np.cross(xu, yu)
    uln_without_magnitude = np.array([[0, 0, 0], xu, yu, zu])
    # Left_Ulna
    Ulna_l_map = np.mean(current_case[l_ulna['idm'].to_list()], axis=0)
    us_point_L = np.nanmean(current_case[US_l['idm'].to_list()], axis=0)
    yu_raw_L = mid_ep_L - us_point_L
    yu_L = (1 / np.linalg.norm(yu_raw_L)) * yu_raw_L
    X0u_L = em_point_L - us_point_L
    X1u_L = el_point_L - us_point_L
    xu_raw_L = np.cross(X1u_L, X0u_L)
    xu_L = (1 / np.linalg.norm(xu_raw_L)) * xu_raw_L
    zu_L = np.cross(xu_L, yu_L)
    uln_without_magnitude_L = np.array([[0, 0, 0], xu_L, yu_L, zu_L])

    rc_humUlna = sphere_fit(translated_humerus[HumUlna_r['idm'].to_list()])
    rc_ulnHum = sphere_fit(current_R_ulna[UlnHum_r['idm'].to_list()])
    rc_humUlna_L = sphere_fit(translated_humerus_L[HumUlna_l['idm'].to_list()])
    rc_ulnHum_L = sphere_fit(current_L_ulna[UlnHum_l['idm'].to_list()])

    HumUlnJoint = np.squeeze(rc_humUlna - rc_ulnHum)
    translated_ulna = current_R_ulna + HumUlnJoint
    A_mat = np.tile(HumUlnJoint, (4, 1))
    translate_uln_vector = uln_without_magnitude - A_mat

    HumUlnJoint_L = np.squeeze(rc_humUlna_L - rc_ulnHum_L)
    translated_ulna_L = current_L_ulna + HumUlnJoint_L
    A_mat_L = np.tile(HumUlnJoint_L, (4, 1))
    translate_uln_vector_L = uln_without_magnitude_L - A_mat_L

    uln_translated = pd.DataFrame(data=translate_uln_vector,
                                  columns=["x", "y", "z"])
    uln_translated_L = pd.DataFrame(data=translate_uln_vector_L,
                                    columns=["x", "y", "z"])
    uvectors = np.zeros((3, 3))
    ulvectors = np.zeros((3, 3))

    # colour = ['red', 'green', 'blue']
    # for i in range(3):
    #     uvectors[i, :] = translate_uln_vector[i + 1, :] - translate_uln_vector[0, :]
    #     ulvectors[i, :] = translate_uln_vector_L[i + 1, :] - translate_uln_vector_L[
    #                                                          0, :]
    # plotter = pv.Plotter()
    # plotter.add_mesh(translated_ulna, color='purple')
    # plotter.add_mesh(translated_ulna_L, color='yellow')
    # plotter.add_mesh(translated_humerus, color='magenta')
    # plotter.add_mesh(translated_humerus_L, color='cyan')
    # plotter.add_mesh(translated_scapula, color='red')
    # plotter.add_mesh(translated_scapula_L, color='green')
    # plotter.add_mesh(translated_clavicle, color='blue')
    # plotter.add_mesh(translated_clavicle_L, color='pink')
    # plotter.add_mesh(current_thorax)
    # for i, uvector in enumerate(uvectors):
    #     plotter.add_mesh(pv.Arrow(start=HumUlnJoint, direction=uvector, scale=20),
    #                      color=colour[i])
    # for i, ulvector in enumerate(ulvectors):
    #     plotter.add_mesh(pv.Arrow(start=HumUlnJoint_L, direction=ulvector, scale=20),
    #                      color=colour[i])
    # for i, hvector in enumerate(hvectors):
    #     plotter.add_mesh(pv.Arrow(start=GHjoint, direction=hvector, scale=20),
    #                      color=colour[i])
    # for i, hlvector in enumerate(hlvectors):
    #     plotter.add_mesh(pv.Arrow(start=GHjoint_L, direction=hlvector, scale=20),
    #                      color=colour[i])
    # for i, svector in enumerate(svectors):
    #     plotter.add_mesh(pv.Arrow(start=ACjoint, direction=svector, scale=20),
    #                      color=colour[i])
    # for i, slvector in enumerate(slvectors):
    #     plotter.add_mesh(pv.Arrow(start=ACjoint_L, direction=slvector, scale=20),
    #                      color=colour[i])
    # for i, cvector in enumerate(cvectors):
    #     plotter.add_mesh(pv.Arrow(start=SCjoint, direction=cvector, scale=20),
    #                      color=colour[i])
    # for i, clvector in enumerate(clvectors):
    #     plotter.add_mesh(pv.Arrow(start=SCjoint_L, direction=clvector, scale=20),
    #                      color=colour[i])
    # for i, vector in enumerate(vectors):
    #     plotter.add_mesh(pv.Arrow(start=ij_point, direction=vector, scale=20),
    #                      color=colour[i])
    # plotter.show()
    # Right_Radius
    rs_point = np.nanmean(current_case[RS_r['idm'].to_list()], axis=0)
    yr_raw = el_point - rs_point
    yr = (1 / np.linalg.norm(yr_raw)) * yr_raw
    X0 = us_point - el_point
    X1 = rs_point - el_point
    xr_raw = np.cross(X1, X0)
    xr = (1 / np.linalg.norm(xr_raw)) * xr_raw
    zr = np.cross(xr, yr)
    rad_without_magnitude = np.array([[0, 0, 0], xr, yr, zr])
    #Left_Radius
    rs_point_L = np.nanmean(current_case[RS_l['idm'].to_list()], axis=0)
    yr_raw_L = el_point_L - rs_point_L
    yr_L = (1 / np.linalg.norm(yr_raw_L)) * yr_raw_L
    X0_L = us_point_L - el_point_L
    X1_L = rs_point_L - el_point_L
    xr_raw_L = np.cross(X1_L, X0_L)
    xr_L = (1 / np.linalg.norm(xr_raw_L)) * xr_raw_L
    zr_L = np.cross(xr_L, yr_L)
    rad_without_magnitude_L = np.array([[0, 0, 0], xr_L, yr_L, zr_L])

    desired_radius = 10.0  # mm
    hum_points = np.array(translated_humerus[HumRadius_r['idm'].to_list()])
    rad_points = np.array(current_R_rad[RadHumerus_r['idm'].to_list()])
    hum_points_L = np.array(translated_humerus_L[HumRadius_l['idm'].to_list()])
    rad_points_L = np.array(current_L_rad[RadHumerus_l['idm'].to_list()])

    hum_center = sphere_fit_with_radius(hum_points, desired_radius)
    rad_center = sphere_fit_with_radius(rad_points, desired_radius)
    hum_center_L = sphere_fit_with_radius(hum_points_L, desired_radius)
    rad_center_L = sphere_fit_with_radius(rad_points_L, desired_radius)
    # Create spheres with the fitted centers and target radius
    # sphere_humerus = pv.Sphere(radius=desired_radius, center=hum_center)
    # sphere_radius = pv.Sphere(radius=desired_radius, center=rad_center)
    HRJoint = (hum_center - rad_center)
    translated_radius = current_R_rad + HRJoint
    A = np.tile(HRJoint, 4)
    A_mat = np.reshape(A, (4, 3))
    translate_rad_vector = rad_without_magnitude - A_mat

    HRJoint_L = (hum_center_L - rad_center_L)
    translated_radius_L = current_L_rad + HRJoint_L
    A_L = np.tile(HRJoint_L, 4)
    A_mat_L = np.reshape(A_L, (4, 3))
    translate_rad_vector_L = rad_without_magnitude_L - A_mat_L

    rad_translated = pd.DataFrame(data=translate_rad_vector,
                                  columns=["x", "y", "z"])
    rad_translated_L = pd.DataFrame(data=translate_rad_vector_L,
                                  columns=["x", "y", "z"])

    rvectors = np.zeros((3, 3))
    rlvectors = np.zeros((3, 3))

    colour = ['red', 'green', 'blue']
    for i in range(3):
        rvectors[i, :] = translate_rad_vector[i + 1,
                         :] - translate_rad_vector[0, :]
        rlvectors[i, :] = translate_rad_vector_L[i + 1,
                          :] - translate_rad_vector_L[
                               0, :]
    plotter = pv.Plotter()
    plotter.add_mesh(translated_radius, color='yellow')
    plotter.add_mesh(translated_radius_L, color='purple')
    plotter.add_mesh(translated_ulna, color='purple')
    plotter.add_mesh(translated_ulna_L, color='yellow')
    plotter.add_mesh(translated_humerus, color='magenta')
    plotter.add_mesh(translated_humerus_L, color='cyan')
    plotter.add_mesh(translated_scapula, color='red')
    plotter.add_mesh(translated_scapula_L, color='green')
    plotter.add_mesh(translated_clavicle, color='blue')
    plotter.add_mesh(translated_clavicle_L, color='pink')
    plotter.add_mesh(current_thorax)
    #plotter.add_mesh(current_thorax)
    for i, rvector in enumerate(rvectors):
        plotter.add_mesh(
            pv.Arrow(start=HRJoint, direction=rvector, scale=20),
            color=colour[i])
    for i, rlvector in enumerate(rlvectors):
        plotter.add_mesh(
            pv.Arrow(start=HRJoint_L, direction=rlvector, scale=20),
            color=colour[i])
    for i, uvector in enumerate(uvectors):
        plotter.add_mesh(
            pv.Arrow(start=HumUlnJoint, direction=uvector, scale=20),
            color=colour[i])
    for i, ulvector in enumerate(ulvectors):
        plotter.add_mesh(
            pv.Arrow(start=HumUlnJoint_L, direction=ulvector, scale=20),
            color=colour[i])
    for i, hvector in enumerate(hvectors):
        plotter.add_mesh(
            pv.Arrow(start=GHjoint, direction=hvector, scale=20),
            color=colour[i])
    for i, hlvector in enumerate(hlvectors):
        plotter.add_mesh(
            pv.Arrow(start=GHjoint_L, direction=hlvector, scale=20),
            color=colour[i])
    for i, svector in enumerate(svectors):
        plotter.add_mesh(
            pv.Arrow(start=ACjoint, direction=svector, scale=20),
            color=colour[i])
    for i, slvector in enumerate(slvectors):
        plotter.add_mesh(
            pv.Arrow(start=ACjoint_L, direction=slvector, scale=20),
            color=colour[i])
    for i, cvector in enumerate(cvectors):
        plotter.add_mesh(
            pv.Arrow(start=SCjoint, direction=cvector, scale=20),
            color=colour[i])
    for i, clvector in enumerate(clvectors):
        plotter.add_mesh(
            pv.Arrow(start=SCjoint_L, direction=clvector, scale=20),
            color=colour[i])
    for i, vector in enumerate(vectors):
        plotter.add_mesh(
            pv.Arrow(start=ij_point, direction=vector, scale=20),
            color=colour[i])
    plotter.show()







