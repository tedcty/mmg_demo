"""Standalone CLI for inspecting a shape model's PCA modes (debugging aid).

DemoServer/server.py no longer shells out to this — it imports pc_shape
directly and keeps the model warm in-process. This script is kept only for
manual/offline inspection, e.g. `python pc_info.py <ssm_path>`.
"""
import sys
import json
from pc_shape import load_pca_model, compute_pc_info


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("ERROR|Missing ssm_path argument", file=sys.stderr, flush=True)
        sys.exit(1)
    try:
        coupled_pcs = load_pca_model(sys.argv[1])
        result = compute_pc_info(coupled_pcs)
        print("PCINFO|" + json.dumps(result), flush=True)
    except Exception as e:
        print(f"ERROR|{str(e)}", file=sys.stderr, flush=True)
        sys.exit(1)
