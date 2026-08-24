import sys
# Announce before the heavy scientific imports below — they take ~10-30s to load
# and emit nothing, which otherwise leaves the progress bar frozen at the start.
print("STATUS|Loading libraries...", flush=True)
import json
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from ptb.util.data import VTKMeshUtl
from pc_shape import load_pca_model, load_mean_mesh, reconstruct_mesh


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
        mesh_data, mean_mesh_verts = load_mean_mesh(args['ssm_path'])
        mesh = reconstruct_mesh(coupled_pcs, mesh_data, mean_mesh_verts, pred_weights)

        print("STATUS|Saving output model...", flush=True)
        VTKMeshUtl.write(args['out_path'], mesh)

        print("STATUS|Running Joint Assembly Pipeline...", flush=True)
        try:
            from generate_isb_joints import process_and_export
            fabrik_step = args.get('fabrik_step', 1)
            process_and_export(args['out_path'], fabrik_step=fabrik_step, export_path=args.get('export_path'))
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
