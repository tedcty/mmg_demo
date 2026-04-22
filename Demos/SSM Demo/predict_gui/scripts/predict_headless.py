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
    ssm_pc_file = os.path.join(ssm_fpath, "combinedSSM.pc.npz")
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
        predictors_train = P.iloc[:, [0, 1, 2, 3, 4, 5, 8, 9]].copy()
        predictors_train.drop([0], axis=0, inplace=True)
        predictors_train.drop([0], axis=1, inplace=True)

        print("STATUS|Loading PCA shape model...", flush=True)
        coupled_pcs = load_pca_model(args['ssm_path'])
        Y = coupled_pcs.projectedWeights.T

        print("STATUS|Running PLSR...", flush=True)
        pls2 = PLSRegression(scale=True)
        pls2.fit(predictors_train, Y)
        pred_weights = pls2.predict([case_data])[0]

        pred_sd = np.zeros((len(pred_weights)))
        for j in range(len(pred_weights)):
            pred_sd[j] = pred_weights[j] / np.sqrt(coupled_pcs.weights[j])

        print("STATUS|Reconstructing 3D Mesh...", flush=True)
        mean_mesh_file = [f for f in os.listdir(args['ssm_path']) if f.startswith('combinedSSM_mean') and f.endswith('.ply')][0]
        mean_mesh_path = os.path.join(args['ssm_path'], mean_mesh_file)

        ply_reader = vtk.vtkPLYReader()
        ply_reader.SetFileName(mean_mesh_path)
        ply_reader.Update()
        mesh_data = ply_reader.GetOutput()
        mean_mesh_verts = VTKMeshUtl.extract_points(mesh_data)

        pc_file = [f for f in os.listdir(args['ssm_path']) if f.startswith('combinedSSM') and f.endswith('.pc.npz')][0]
        pc_file_path = os.path.join(args['ssm_path'], pc_file)
        pc = np.load(pc_file_path, allow_pickle=True)

        pc_modes = pc['modes']
        pc_weight = pc['weights']
        w_list = np.sqrt(pc_weight)
        scaled_weights = (pred_sd * w_list).reshape(1, -1)

        verts = np.dot(scaled_weights, pc_modes.T)
        reconstructed_verts = np.reshape(verts, [int(verts.shape[1] / 3), 3]) + mean_mesh_verts
        mesh = VTKMeshUtl.update_poly_w_points(reconstructed_verts, mesh_data)

        print("STATUS|Saving output model...", flush=True)
        VTKMeshUtl.write(args['out_path'], mesh)

        print(f"SUCCESS|Model saved to: {args['out_path']}", flush=True)

    except Exception as e:
        print(f"ERROR|{str(e)}", file=sys.stderr, flush=True)
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("ERROR|Missing JSON argument", file=sys.stderr, flush=True)
        sys.exit(1)
    
    run_prediction(sys.argv[1])
