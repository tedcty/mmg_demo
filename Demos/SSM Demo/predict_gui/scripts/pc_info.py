import sys
import os
import json
import numpy as np
from gias3.learning import PCA


def load_pca_model(ssm_fpath):
    pc_files = [f for f in os.listdir(ssm_fpath) if (f.endswith('.pc') or f.endswith('.pc.npz')) and not f.startswith('._')]
    if not pc_files:
        raise FileNotFoundError(f"No .pc or .pc.npz file found in {ssm_fpath}")
    return PCA.loadPrincipalComponents(os.path.join(ssm_fpath, pc_files[0]))


def main(ssm_path, max_modes=10):
    coupled_pcs = load_pca_model(ssm_path)
    n_modes_total = coupled_pcs.modes.shape[-1]

    # projectedWeights might be (n_samples, n_modes) or (n_modes, n_samples) —
    # orient it against the known mode count rather than assuming.
    Y = coupled_pcs.projectedWeights
    if Y.shape[1] != n_modes_total:
        Y = Y.T

    n = min(max_modes, n_modes_total)
    variances = np.var(Y, axis=0)
    total_var = variances.sum()

    result = {
        "n_modes": int(n),
        "std": [float(np.std(Y[:, i])) for i in range(n)],
        "variance_pct": [float(variances[i] / total_var * 100) for i in range(n)],
    }
    print("PCINFO|" + json.dumps(result), flush=True)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("ERROR|Missing ssm_path argument", file=sys.stderr, flush=True)
        sys.exit(1)
    try:
        main(sys.argv[1])
    except Exception as e:
        print(f"ERROR|{str(e)}", file=sys.stderr, flush=True)
        sys.exit(1)
