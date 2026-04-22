import os
import numpy as np
import pandas as pd
import vtk
from yatpkg.util.data import VTKMeshUtl
import pyvista as pv


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

# Body segment paths
thorax = r"C:\Users\rrag962\Documents\shapemodel\model\Shoulder 3\Segmentations\Combined shape model\100_aligned\thorax_aligned_meshes\Right"
clavicle = r"C:\Users\rrag962\Documents\shapemodel\model\Shoulder 3\Segmentations\Combined shape model\100_aligned\clavicle_aligned_meshes"
scapula = r"C:\Users\rrag962\Documents\shapemodel\model\Shoulder 3\Segmentations\Combined shape model\100_aligned\scapula_aligned_meshes"
humerus = r"C:\Users\rrag962\Documents\shapemodel\model\Shoulder 3\Segmentations\Combined shape model\100_aligned\humerus_aligned_meshes"

# landmarking csv maps
C71 = pd.read_csv(r"C:\Users\rrag962\Documents\shapemodel\model\Shoulder 3\Segmentations\Landmarking\4TH\maps to mean\ThoC71.csv")
C72 = pd.read_csv(r"C:\Users\rrag962\Documents\shapemodel\model\Shoulder 3\Segmentations\Landmarking\4TH\maps to mean\ThoC72.csv")
T81 = pd.read_csv(r"C:\Users\rrag962\Documents\shapemodel\model\Shoulder 3\Segmentations\Landmarking\4TH\maps to mean\ThoT81.csv")
T82 = pd.read_csv(r"C:\Users\rrag962\Documents\shapemodel\model\Shoulder 3\Segmentations\Landmarking\4TH\maps to mean\ThoT82.csv")
IJ = pd.read_csv(r"C:\Users\rrag962\Documents\shapemodel\model\Shoulder 3\Segmentations\Landmarking\4TH\maps to mean\ThoIJ2.csv")
PX = pd.read_csv(r"C:\Users\rrag962\Documents\shapemodel\model\Shoulder 3\Segmentations\Landmarking\4TH\maps to mean\ThoPX.csv")
SC = pd.read_csv(r"C:\Users\rrag962\Documents\shapemodel\model\Shoulder 3\Segmentations\Landmarking\4TH\maps to mean\ClavSC.csv")
AC = pd.read_csv(r"C:\Users\rrag962\Documents\shapemodel\model\Shoulder 3\Segmentations\Landmarking\4TH\maps to mean\ClavAC.csv")
AA = pd.read_csv(r"C:\Users\rrag962\Documents\shapemodel\model\Shoulder 3\Segmentations\Landmarking\4TH\maps to mean\ScapAA.csv")
AI = pd.read_csv(r"C:\Users\rrag962\Documents\shapemodel\model\Shoulder 3\Segmentations\Landmarking\4TH\maps to mean\ScapAI.csv")
TS = pd.read_csv(r"C:\Users\rrag962\Documents\shapemodel\model\Shoulder 3\Segmentations\Landmarking\4TH\maps to mean\ScapTS.csv")
SA = pd.read_csv(r"C:\Users\rrag962\Documents\shapemodel\model\Shoulder 3\Segmentations\Landmarking\4TH\maps to mean\ScapSA.csv")
humeral_head = pd.read_csv(r"C:\Users\rrag962\Documents\shapemodel\model\Shoulder 3\Segmentations\Landmarking\4TH\maps to mean\HumH.csv")
EL = pd.read_csv(r"C:\Users\rrag962\Documents\shapemodel\model\Shoulder 3\Segmentations\Landmarking\4TH\maps to mean\HumEL.csv")
EM = pd.read_csv(r"C:\Users\rrag962\Documents\shapemodel\model\Shoulder 3\Segmentations\Landmarking\4TH\maps to mean\HumEM.csv")
glenoid = pd.read_csv(r"C:\Users\rrag962\Documents\shapemodel\model\Shoulder 3\Segmentations\Landmarking\4TH\maps to mean\Glenoid.csv")

thorax_files = [f for f in os.listdir(thorax)]
clavicle_files = [f for f in os.listdir(clavicle)]
scapula_files = [f for f in os.listdir(scapula)]
humerus_files = [f for f in os.listdir(humerus)]

for thorax_file in thorax_files:
    case = thorax_file[:7]
    matching_cases = [clavicle_file for clavicle_file in clavicle_files if clavicle_file.startswith(case)]
    for matching_case in matching_cases:
        # THORAX
        ply_reader = vtk.vtkPLYReader()
        ply_reader.SetFileName(os.path.join(thorax, thorax_file))
        ply_reader.Update()
        mesh_data = ply_reader.GetOutput()
        current_thorax = VTKMeshUtl.extract_points(mesh_data)
        c71_point = np.mean(current_thorax[C71['idm'].to_list()], axis=0)
        c72_point = np.mean(current_thorax[C72['idm'].to_list()], axis=0)
        t81_point = np.mean(current_thorax[T81['idm'].to_list()], axis=0)
        t82_point = np.mean(current_thorax[T82['idm'].to_list()], axis=0)
        ij_point = np.mean(current_thorax[IJ['idm'].to_list()], axis=0)
        px_point = np.mean(current_thorax[PX['idm'].to_list()], axis=0)

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
        plotter.add_mesh(current_thorax)
        for i, vector in enumerate(vectors):
            plotter.add_mesh(pv.Arrow(start=ij_point, direction=vector, scale=20), color=colour[i])
        plotter.show()

        # CLAVICLE
        ply_reader = vtk.vtkPLYReader()
        ply_reader.SetFileName(os.path.join(clavicle, matching_case))
        ply_reader.Update()
        mesh_data = ply_reader.GetOutput()
        current_clavicle = VTKMeshUtl.extract_points(mesh_data)

        sc_point = np.mean(current_clavicle[SC['idm'].to_list()], axis=0)
        ac_point = np.mean(current_clavicle[AC['idm'].to_list()], axis=0)

        zc_raw = ac_point - sc_point
        zc = (1 / np.linalg.norm(zc_raw)) * zc_raw
        xc_raw = np.cross(yt, zc)
        xc = (1 / np.linalg.norm(zc_raw)) * xc_raw
        yc = np.cross(zc, xc)
        cla_without_magnitude = np.array([[0, 0, 0], xc, yc, zc])

        A = np.tile(sc_point, 4)
        A_mat = np.reshape(A, (4, 3))
        translate_cla_vector = cla_without_magnitude - A_mat
        cla_translated = pd.DataFrame(data=translate_cla_vector, columns=["x", "y", "z"])
        cvectors = np.zeros((3, 3))

        colour = ['red', 'green', 'blue']
        for i in range(3):
            cvectors[i, :] = translate_cla_vector[i + 1, :] - translate_cla_vector[0, :]
        plotter = pv.Plotter()
        plotter.add_mesh(current_clavicle)
        for i, cvector in enumerate(cvectors):
            plotter.add_mesh(pv.Arrow(start=sc_point, direction=cvector, scale=20), color=colour[i])
        plotter.show()

        # SCAPULA
        matching_cases2 = [scapula_file for scapula_file in scapula_files if scapula_file.startswith(case)]
        for matching_case2 in matching_cases2:
            ply_reader = vtk.vtkPLYReader()
            ply_reader.SetFileName(os.path.join(scapula, matching_case2))
            ply_reader.Update()
            mesh_data = ply_reader.GetOutput()
            current_scapula = VTKMeshUtl.extract_points(mesh_data)

            aa_point = np.mean(current_scapula[AA['idm'].to_list()], axis=0)
            ai_point = np.mean(current_scapula[AI['idm'].to_list()], axis=0)
            ts_point = np.mean(current_scapula[TS['idm'].to_list()], axis=0)
            sa_point = np.mean(current_scapula[SA['idm'].to_list()], axis=0)

            zs_raw = aa_point - ts_point
            zs = (1 / np.linalg.norm(zs_raw)) * zs_raw
            yx1_raw = ts_point - aa_point
            yx2_raw = ai_point - aa_point
            xs_raw = np.cross(yx2_raw, yx1_raw)
            xs = (1 / np.linalg.norm(xs_raw)) * xs_raw
            ys = np.cross(zs, xs)
            sca_without_magnitude = np.array([[0, 0, 0], xs, ys, zs])
            # scap = pd.DataFrame(data=without_magnitude, columns=["x", "y", "z"])
            A = np.tile(aa_point,4)
            A_mat = np.reshape(A, (4, 3))
            translate_scap_vector = sca_without_magnitude - A_mat
            scap_translated = pd.DataFrame(data=translate_scap_vector, columns=["x", "y", "z"])
            svectors = np.zeros((3, 3))

            colour = ['red', 'green', 'blue']
            for i in range(3):
                svectors[i, :] = translate_scap_vector[i + 1, :] - translate_scap_vector[0, :]
            plotter = pv.Plotter()
            plotter.add_mesh(current_scapula)
            for i, svector in enumerate(svectors):
                plotter.add_mesh(pv.Arrow(start=aa_point, direction=svector, scale=20), color=colour[i])
            plotter.show()

            # HUMERUS
            matching_cases3 = [humerus_file for humerus_file in humerus_files if humerus_file.startswith(case)]
            for matching_case3 in matching_cases3:
                ply_reader = vtk.vtkPLYReader()
                ply_reader.SetFileName(os.path.join(humerus, matching_case3))
                ply_reader.Update()
                mesh_data = ply_reader.GetOutput()
                current_humerus = VTKMeshUtl.extract_points(mesh_data)

                rc = np.squeeze(sphere_fit(current_humerus[humeral_head['idm'].to_list()]))
                current_humerus_rc = rc
                el_point = np.nanmean(current_humerus[EL['idm'].to_list()], axis=0)
                em_point = np.nanmean(current_humerus[EM['idm'].to_list()], axis=0)
                mid_ep = 0.5 * (em_point + el_point)

                # calculate coord system
                yh_raw = rc - mid_ep
                yh = (1 / np.linalg.norm(yh_raw)) * yh_raw
                X0 = em_point - rc
                X1 = el_point - rc
                xh_raw = np.cross(X1, X0)
                xh = (1 / np.linalg.norm(xh_raw)) * xh_raw
                zh = np.cross(xh, yh)
                without_magnitude = np.array([[0, 0, 0], xh, yh, zh])

                # translate
                rc_scap = sphere_fit(current_scapula[glenoid['idm'].to_list()])
                rc_hum = sphere_fit(current_humerus[humeral_head['idm'].to_list()])

                glen_hum = np.squeeze(rc_scap - rc_hum)
                translated_humerus = current_humerus + glen_hum
                A = np.tile(glen_hum, 4)
                A_mat = np.reshape(A, (4, 3))
                translate_hum_vector = without_magnitude - A_mat
                hum_translated = pd.DataFrame(data=translate_hum_vector, columns=["x", "y", "z"])
                hvectors = np.zeros((3, 3))

                colour = ['red', 'green', 'blue']
                for i in range(3):
                    hvectors[i, :] = translate_hum_vector[i + 1, :] - translate_hum_vector[0, :]
                plotter = pv.Plotter()
                # plotter.add_mesh(current_humerus, color='yellow', opacity=0.5)
                plotter.add_mesh(translated_humerus, color='blue')
                # plotter.add_mesh(current_scapula, color='pink')
                for i, hvector in enumerate(hvectors):
                    # plotter.add_mesh(pv.Arrow(start=rc_hum, direction=hvector, scale=20), color=colour[i])
                    plotter.add_mesh(pv.Arrow(start=rc_scap, direction=hvector, scale=20), color=colour[i])
                    # plotter.add_mesh(pv.Arrow(start=glen_hum, direction=hvector, scale=20), color=colour[i])
                plotter.show()