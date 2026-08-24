"""pc_shape — shared PCA shape-model helpers.

Used both by predict_headless.py (run as a one-shot subprocess for the
anthropometric/PLSR prediction) and directly imported in-process by
DemoServer/server.py for the Shape (PCA) tab, which needs to reconstruct a
mesh from manual PC weights on every slider release without re-paying the
cost of importing gias3/vtk each time.
"""
import os
import numpy as np
import pandas as pd
from gias3.learning import PCA
from sklearn.cross_decomposition import PLSRegression
import vtk
from ptb.util.data import VTKMeshUtl


def load_pca_model(ssm_fpath):
    # Flexible search for .pc or .pc.npz
    pc_files = [f for f in os.listdir(ssm_fpath) if (f.endswith('.pc') or f.endswith('.pc.npz')) and not f.startswith('._')]
    if not pc_files:
        raise FileNotFoundError(f"No .pc or .pc.npz file found in {ssm_fpath}")
    print(f"STATUS|Loading PCA model: {pc_files[0]}", flush=True)
    return PCA.loadPrincipalComponents(os.path.join(ssm_fpath, pc_files[0]))


def find_mean_mesh_path(ssm_fpath):
    mean_mesh_files = [f for f in os.listdir(ssm_fpath) if 'mean' in f.lower() and f.endswith('.ply') and not f.startswith('._')]
    if not mean_mesh_files:
        raise FileNotFoundError(f"No mean mesh found in {ssm_fpath}")
    return os.path.join(ssm_fpath, mean_mesh_files[0])


def load_mean_mesh(ssm_fpath):
    mean_mesh_path = find_mean_mesh_path(ssm_fpath)
    print(f"STATUS|Using mean mesh: {os.path.basename(mean_mesh_path)}", flush=True)
    ply_reader = vtk.vtkPLYReader()
    ply_reader.SetFileName(mean_mesh_path)
    ply_reader.Update()
    mesh_data = ply_reader.GetOutput()
    mean_mesh_verts = VTKMeshUtl.extract_points(mesh_data)
    return mesh_data, mean_mesh_verts


def compute_pc_info(coupled_pcs, max_modes=10):
    n_modes_total = coupled_pcs.modes.shape[-1]

    # projectedWeights might be (n_samples, n_modes) or (n_modes, n_samples) —
    # orient it against the known mode count rather than assuming.
    Y = coupled_pcs.projectedWeights
    if Y.shape[1] != n_modes_total:
        Y = Y.T

    n = min(max_modes, n_modes_total)
    variances = np.var(Y, axis=0)
    total_var = variances.sum()

    return {
        "n_modes": int(n),
        "std": [float(np.std(Y[:, i])) for i in range(n)],
        "variance_pct": [float(variances[i] / total_var * 100) for i in range(n)],
    }


def predict_weights_from_anthro(coupled_pcs, anthro_path, case_data):
    """PLSR-predict PC weights from anthropometric measurements. `case_data`
    is [sex, age, height, weight, r_clav_len, r_hum_len, r_hum_epi_width].
    Shared by predict_headless.py (the full-solve subprocess path) and
    server.py's in-process fast-prediction path, so both train the exact
    same regression the exact same way."""
    P = pd.read_csv(anthro_path, header=None)
    # Assuming the CSV structure is fixed as per the project requirements
    predictors_train = P.iloc[:, [0, 1, 2, 3, 4, 5, 8, 9]].copy()
    predictors_train.drop([0], axis=0, inplace=True)
    predictors_train.drop([0], axis=1, inplace=True)

    # projectedWeights might be (n_samples, n_modes). We need (n_samples, n_modes) for fit.
    Y = coupled_pcs.projectedWeights
    if Y.shape[0] != predictors_train.shape[0]:
        Y = Y.T

    # n_components must be <= min(n_samples, n_features)
    n_comp = min(10, Y.shape[1], predictors_train.shape[1], predictors_train.shape[0])
    pls2 = PLSRegression(n_components=n_comp, scale=True)
    pls2.fit(predictors_train, Y)
    return pls2.predict([case_data])[0]


def reconstruct_mesh(coupled_pcs, mesh_data, mean_mesh_verts, weights):
    """Mean + Sum(weight_i * mode_i). `weights` is zero-padded/truncated to
    the model's actual mode count.

    `mesh_data` is expected to be a shared, cached vtkPolyData (see
    server.py's _get_pc_model) reused across every request/session — deep
    copy it before mutating so concurrent requests don't race on the same
    object. mean_mesh_verts/coupled_pcs are only read here, never mutated,
    so they're safe to share without copying.
    """
    n_modes_total = coupled_pcs.modes.shape[-1]
    supplied = np.array([float(w) for w in weights])
    pred_weights = np.zeros(n_modes_total)
    n = min(len(supplied), n_modes_total)
    pred_weights[:n] = supplied[:n]

    modes = coupled_pcs.modes  # Usually (3*N, M)
    # Note: Depending on gias3 version, modes might be (n_points, 3, n_modes) or (3*n_points, n_modes)
    if len(modes.shape) == 2:
        offset = np.dot(modes, pred_weights)
        reconstruction = mean_mesh_verts + offset.reshape(-1, 3)
    else:
        offset = np.sum(modes * pred_weights, axis=2)
        reconstruction = mean_mesh_verts + offset

    mesh_copy = vtk.vtkPolyData()
    mesh_copy.DeepCopy(mesh_data)
    return VTKMeshUtl.update_poly_w_points(reconstruction, mesh_copy)
