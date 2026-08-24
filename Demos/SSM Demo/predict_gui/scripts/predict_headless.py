import sys
# Announce before the heavy scientific imports below — they take ~10-30s to load
# and emit nothing, which otherwise leaves the progress bar frozen at the start.
print("STATUS|Loading libraries...", flush=True)
import json
from ptb.util.data import VTKMeshUtl
from pc_shape import load_pca_model, load_mean_mesh, reconstruct_mesh, predict_weights_from_anthro


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

        print("STATUS|Loading PCA shape model...", flush=True)
        coupled_pcs = load_pca_model(args['ssm_path'])

        pred_weights = predict_weights_from_anthro(coupled_pcs, args['anthro_path'], case_data)

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
