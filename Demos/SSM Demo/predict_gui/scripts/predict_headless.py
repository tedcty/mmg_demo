import sys
import json
import numpy as np
import os
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
    ssm_pc_file = os.path.join(ssm_fpath, pc_files[0])
    print(f"STATUS|Loading PCA model: {pc_files[0]}", flush=True)
    return PCA.loadPrincipalComponents(ssm_pc_file)

def run_prediction(json_args_str):
    try:
        args = json.loads(json_args_str)
        # Validate keys
        keys = ['sex', 'age', 'height', 'weight', 'r_clav_len', 'r_hum_len', 'r_hum_epi_width', 
                'anthro_path', 'ssm_path', 'out_path']
        for k in keys:
            if k not in args:
                raise ValueError(f"Missing required argument: {k}")

        case_data = [
            float(args['sex']), float(args['age']), float(args['height']),
            float(args['weight']), float(args['r_clav_len']),
            float(args['r_hum_len']), float(args['r_hum_epi_width'])
        ]

        print("STATUS|Starting PLSR training...", flush=True)

        P = pd.read_csv(args['anthro_path'], header=None)
        # Assuming the CSV structure is fixed as per the project requirements
        predictors_train = P.iloc[:, [0, 1, 2, 3, 4, 5, 8, 9]].copy()
        predictors_train.drop([0], axis=0, inplace=True)
        predictors_train.drop([0], axis=1, inplace=True)

        print("STATUS|Loading PCA shape model...", flush=True)
        coupled_pcs = load_pca_model(args['ssm_path'])
        
        # projectedWeights might be (n_samples, n_modes). We need (n_samples, n_modes) for fit.
        # Check orientation
        Y = coupled_pcs.projectedWeights
        if Y.shape[0] != predictors_train.shape[0]:
            Y = Y.T
            
        print(f"STATUS|Running PLSR with {Y.shape[1]} modes...", flush=True)
        # n_components must be <= min(n_samples, n_features)
        n_comp = min(10, Y.shape[1], predictors_train.shape[1], predictors_train.shape[0])
        pls2 = PLSRegression(n_components=n_comp, scale=True)
        pls2.fit(predictors_train, Y)
        pred_weights = pls2.predict([case_data])[0]

        # Use the PCA object's weights (eigenvalues) for normalization if needed
        # In this workflow, pred_weights are the absolute weights
        
        print("STATUS|Reconstructing 3D Mesh...", flush=True)
        mean_mesh_files = [f for f in os.listdir(args['ssm_path']) if 'mean' in f.lower() and f.endswith('.ply') and not f.startswith('._')]
        if not mean_mesh_files:
            raise FileNotFoundError(f"No mean mesh found in {args['ssm_path']}")
        
        mean_mesh_path = os.path.join(args['ssm_path'], mean_mesh_files[0])
        print(f"STATUS|Using mean mesh: {mean_mesh_files[0]}", flush=True)

        ply_reader = vtk.vtkPLYReader()
        ply_reader.SetFileName(mean_mesh_path)
        ply_reader.Update()
        mesh_data = ply_reader.GetOutput()
        mean_mesh_verts = VTKMeshUtl.extract_points(mesh_data)

        # Reconstruct using the PCA object's modes and pred_weights
        # Reconstruction: Mean + Sum(weight_i * mode_i)
        reconstruction = np.zeros_like(mean_mesh_verts)
        modes = coupled_pcs.modes # Usually (3*N, M)
        
        # Apply weights
        # Note: Depending on gias3 version, modes might be (n_points, 3, n_modes) or (3*n_points, n_modes)
        if len(modes.shape) == 2:
            # Flattened modes
            offset = np.dot(modes, pred_weights)
            reconstruction = mean_mesh_verts + offset.reshape(-1, 3)
        else:
            # Already (N, 3, M)
            offset = np.sum(modes * pred_weights, axis=2)
            reconstruction = mean_mesh_verts + offset

        mesh = VTKMeshUtl.update_poly_w_points(reconstruction, mesh_data)

        print("STATUS|Saving output model...", flush=True)
        VTKMeshUtl.write(args['out_path'], mesh)

        print("STATUS|Running Joint Assembly Pipeline...", flush=True)
        try:
            from generate_isb_joints import process_and_export
            process_and_export(args['out_path'])
        except Exception as assembly_err:
            import traceback
            traceback.print_exc()
            print(f"STATUS|Warning: Joint assembly failed: {assembly_err}", flush=True)

        print(f"SUCCESS|Model saved to: {args['out_path']}", flush=True)

    except Exception as e:
        print(f"ERROR|{str(e)}", file=sys.stderr, flush=True)
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("ERROR|Missing JSON argument", file=sys.stderr, flush=True)
        sys.exit(1)
    
    run_prediction(sys.argv[1])
